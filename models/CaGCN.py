import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import AUROC, Accuracy, F1Score
from torch_geometric.nn.models import GCN

from models.VanillaGNN import VanillaGNN


class CaGCNModule(L.LightningModule):
    """
    CaGCN (as in the official repo):
      - Frozen base model -> logits
      - A GCN over logits -> scalar per-node t
      - Calibrated logits = logits * log(exp(t) + 1.1)
    Training (final scaling stage): minimize NLL on the *validation nodes* only.
    """

    def __init__(
        self,
        checkpoint_path: str,
        calib_hidden: int = 16,
        calib_layers: int = 2,
        lr: float = 1e-2,
        weight_decay: float = 5e-3,
        softplus_eps: float = 1.1,   # log(exp(t) + 1.1) as in CaGCN repo
        ood_in_val: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---- frozen base model ----
        self.base: VanillaGNN = VanillaGNN.load_from_checkpoint(checkpoint_path)
        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad = False

        self.C = self.base.C
        self.ood_in_val = ood_in_val
        self.softplus_eps = softplus_eps

        # ---- scaling GCN: input = logits (dim C), output = 1 (temperature per node) ----
        self.scaler = GCN(
            in_channels=self.C,
            hidden_channels=calib_hidden,
            num_layers=calib_layers,
            out_channels=1,
            act=F.sigmoid,  # consistent with repo style; does not alter theory
        )

        # metrics
        self.auroc_metric = AUROC(task="binary")
        self.acc_metric   = Accuracy(task="multiclass", num_classes=self.C)
        self.f1_metric    = F1Score(task="multiclass", num_classes=self.C)

        self.lr = lr
        self.weight_decay = weight_decay

    # ---------------- core ops ----------------
    @torch.no_grad()
    def _base_logits(self, data):
        return self.base(data)  # [N, C]

    def _temperature(self, logits, edge_index):
        # t_raw: [N, 1]
        t_raw = self.scaler(logits, edge_index)
        # CaGCN repo mapping: t = log(exp(t) + 1.1)
        t = torch.log(torch.exp(t_raw) + self.softplus_eps)
        return t  # [N, 1] > 0

    def _calibrated_logits(self, data):
        logits = self._base_logits(data)               # [N, C]
        t = self._temperature(logits, data.edge_index) # [N, 1]
        return logits * t                              # broadcast mult

    @staticmethod
    def _onehot_to_indices(y_onehot: torch.Tensor) -> torch.Tensor:
        return torch.argmax(y_onehot, dim=1)

    def _ood_auroc_from_logits_and_mask(self, logits, y_onehot, mask):
        # MSP based OOD = -max softmax
        logits_m = logits[mask]
        y_m = y_onehot[mask]
        probs = F.softmax(logits_m, dim=1)
        msp, _ = torch.max(probs, dim=1)
        ood_scores = -msp
        ood_targets = 1 - y_m.sum(dim=1)  # 1 = OOD, 0 = ID
        return self.auroc_metric(ood_scores, ood_targets)

    # ---------------- Lightning ----------------
    def forward(self, data):
        return self._calibrated_logits(data)

    def training_step(self, batch, batch_idx):
        # final scaling stage trains on validation nodes (as in the repo when sign=False)
        if not hasattr(batch, "val_mask"):
            raise ValueError("CaGCN requires 'val_mask' in the batch for calibration training.")
        if not torch.any(batch.val_mask):
            raise ValueError("'val_mask' is empty or all False.")

        logits_cal = self(batch)                # calibrated logits
        val_mask = batch.val_mask
        y_val = batch.y[val_mask]
        logits_val = logits_cal[val_mask]
        labels_val = self._onehot_to_indices(y_val)

        loss = F.cross_entropy(logits_val, labels_val)

        # optional logging: OOD AUROC on val
        if self.ood_in_val:
            val_auroc = self._ood_auroc_from_logits_and_mask(logits_cal, batch.y, val_mask)
            self.log("val_auroc", val_auroc, prog_bar=True)

        self.log("train_loss", loss, prog_bar=True)
        self.log("val_nll", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if not hasattr(batch, "val_mask") or not torch.any(batch.val_mask):
            raise ValueError("Validation requires a non-empty 'val_mask'.")
        logits_cal = self(batch)
        val_auroc = self._ood_auroc_from_logits_and_mask(logits_cal, batch.y, batch.val_mask)
        self.log("val_auroc_loop", val_auroc, prog_bar=True)
        return val_auroc

    def test_step(self, batch, batch_idx):
        if not hasattr(batch, "test_mask") or not torch.any(batch.test_mask):
            raise ValueError("Test requires a non-empty 'test_mask'.")
        logits_cal = self(batch)

        # OOD AUROC (MSP)
        test_auroc = self._ood_auroc_from_logits_and_mask(logits_cal, batch.y, batch.test_mask)
        self.log("test_auroc", test_auroc, prog_bar=True)

        # ID metrics
        logits_test = logits_cal[batch.test_mask]
        y_test = batch.y[batch.test_mask]
        id_mask = (y_test.sum(dim=1) == 1)
        if id_mask.any():
            id_logits = logits_test[id_mask]
            id_labels = torch.argmax(y_test[id_mask], dim=1)
            id_preds = torch.argmax(id_logits, dim=1)
            self.log("test_acc", self.acc_metric(id_preds, id_labels))
            self.log("test_f1",  self.f1_metric(id_preds, id_labels))

        return test_auroc

    def configure_optimizers(self):
        params = [p for p in self.scaler.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
