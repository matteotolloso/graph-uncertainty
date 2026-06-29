import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from torchmetrics import AUROC
from torchmetrics.classification import F1Score

from models.VanillaGNN import VanillaGNN
from models.credal_layer import CredalLayer
from models.credal_loss import CreNetLoss
from utils.math import compute_uncertainties


class CredalFrozenJoint(L.LightningModule):
    """
    Ablation: load a pretrained VanillaGNN, freeze it, extract the JOINT latent
    representation (x^0 + each layer's output), and train ONLY a Credal layer.

    Joint dim is computed analytically from backbone hyperparams, so the
    credal head is created eagerly (no empty optimizer issue).
    """

    def __init__(
        self,
        checkpoint_path: str,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        delta: float = 0.5,
        ood_in_val: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---- Frozen backbone ----
        self.backbone: VanillaGNN = VanillaGNN.load_from_checkpoint(checkpoint_path)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Pull hyperparams saved by VanillaGNN.save_hyperparameters()
        # Fallbacks in case of attribute access differences across Lightning versions
        hp = getattr(self.backbone, "hparams", {})
        in_channels   = int(hp.get("in_channels", None))
        hidden_ch     = int(hp.get("hidden_channels", None))
        num_layers    = int(hp.get("num_layers", None))
        out_channels  = int(hp.get("out_channels", None))

        if None in (in_channels, hidden_ch, num_layers, out_channels):
            # As a robust fallback, try grabbing from backbone.gnn_model if present
            gm = self.backbone.gnn_model
            in_channels  = in_channels  if in_channels  is not None else getattr(gm, "in_channels", None)
            hidden_ch    = hidden_ch    if hidden_ch    is not None else getattr(gm, "hidden_channels", None)
            num_layers   = num_layers   if num_layers   is not None else getattr(gm, "num_layers", None)
            out_channels = out_channels if out_channels is not None else getattr(gm, "out_channels", None)
            if None in (in_channels, hidden_ch, num_layers, out_channels):
                raise RuntimeError("Unable to infer backbone hyperparameters to build joint representation.")

        # Joint = x^0 (in_channels) + (num_layers-1)*hidden + last_layer(out_channels)
        # because PyG high-level models typically produce hidden for all but last layer, then out for final.
        joint_dim = in_channels + max(num_layers - 1, 0) * hidden_ch + out_channels

        self.C = out_channels
        self.ood_in_val = ood_in_val

        # ---- Credal head (trainable) ----
        self.credal_layer_model = CredalLayer(input_dim=joint_dim, C=self.C)
        self._init_weights(self.credal_layer_model)

        # ---- Loss & metrics ----
        self.criterion = CreNetLoss(delta=delta)
        self.lr = lr
        self.weight_decay = weight_decay

        self.f1_metric = F1Score(task="multiclass", num_classes=self.C)
        self.auroc_metric = AUROC(task="binary")

    def _split_mask(self, batch, split):
        if hasattr(batch, "batch_size") and hasattr(batch, "n_id"):
            mask = torch.zeros(batch.num_nodes, dtype=torch.bool, device=batch.x.device)
            mask[:batch.batch_size] = True
            return mask
        return getattr(batch, f"{split}_mask")

    def _f1_score(self, preds, labels):
        return F1Score(task="multiclass", num_classes=self.C).to(preds.device)(preds, labels)

    def _auroc_score(self, scores, targets):
        return AUROC(task="binary").to(scores.device)(scores, targets)

    # ---------------------------
    # Backbone helpers
    # ---------------------------
    @torch.no_grad()
    def _get_joint_embeddings(self, data) -> torch.Tensor:
        """
        Concatenate: input features (z^0) + outputs of each backbone GNN layer.
        Backbone is frozen/eval.
        """
        all_embeddings = [data.x]
        gnn = self.backbone.gnn_model

        if not hasattr(gnn, "convs") or not hasattr(gnn, "act"):
            raise NotImplementedError("Backbone must expose 'convs' and 'act' attributes.")

        x = data.x
        for i in range(gnn.num_layers):
            x = gnn.convs[i](x, data.edge_index)
            if i < gnn.num_layers - 1:
                x = gnn.act(x)
            all_embeddings.append(x)

        joint = torch.cat(all_embeddings, dim=1)  # [N, D_joint]
        # Optional sanity check (first batch hit): assert joint.size(1) == credal input dim
        if self.credal_layer_model is not None and joint.size(1) != self.credal_layer_model.input_dim:
            raise RuntimeError(
                f"Computed joint dim {joint.size(1)} != credal input_dim {self.credal_layer_model.input_dim}. "
                "Check backbone hyperparams vs runtime shapes."
            )
        return joint

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        for child in m.children():
            CredalFrozenJoint._init_weights(child)

    # ---------------------------
    # Lightning: forward & steps
    # ---------------------------
    def forward(self, data):
        with torch.no_grad():
            joint = self._get_joint_embeddings(data)
        q_L, q_U = self.credal_layer_model(joint)
        return q_L, q_U

    def training_step(self, batch, batch_idx):
        if not hasattr(batch, "train_mask") and not (hasattr(batch, "batch_size") and hasattr(batch, "n_id")):
            raise ValueError("Training batch must include 'train_mask'.")
        train_mask = self._split_mask(batch, "train")
        if not torch.any(train_mask):
            raise ValueError("'train_mask' is empty or all False in training batch.")

        q_L, q_U = self(batch)
        y_train = batch.y[train_mask]
        q_U_train = q_U[train_mask]
        q_L_train = q_L[train_mask]

        loss = self.criterion(q_L_train, q_U_train, y_train)

        preds_U = torch.argmax(q_U_train, dim=1)
        preds_L = torch.argmax(q_L_train, dim=1)
        labels  = torch.argmax(y_train, dim=1)
        f1_U = self._f1_score(preds_U, labels)
        f1_L = self._f1_score(preds_L, labels)
        acc_U = (preds_U == labels).float().mean()
        acc_L = (preds_L == labels).float().mean()

        n = train_mask.sum()
        self.log("train_loss", loss, batch_size=n)
        self.log("train_acc_U", acc_U, batch_size=n)
        self.log("train_acc_L", acc_L, batch_size=n)
        self.log("train_f1_U",  f1_U,  batch_size=n)
        self.log("train_f1_L",  f1_L,  batch_size=n)
        return loss

    def on_validation_epoch_start(self):
        self._validation_outputs = []

    def validation_step(self, batch, batch_idx):
        if not hasattr(batch, "val_mask") and not (hasattr(batch, "batch_size") and hasattr(batch, "n_id")):
            raise ValueError("Validation batch must include 'val_mask'.")
        val_mask = self._split_mask(batch, "val")
        if not torch.any(val_mask):
            raise ValueError("'val_mask' is empty or all False in validation batch.")

        q_L, q_U = self(batch)
        q_L_val = q_L[val_mask].detach()
        q_U_val = q_U[val_mask].detach()
        y_val   = batch.y[val_mask].detach()

        id_mask = (y_val.sum(axis=1) == 1)
        if id_mask.any():
            loss = self.criterion(q_L_val[id_mask], q_U_val[id_mask], y_val[id_mask])
            self.log("val_loss", loss, prog_bar=True, batch_size=id_mask.sum(), on_step=False, on_epoch=True)

            val_labels  = torch.argmax(y_val[id_mask], dim=1)
            val_preds_U = torch.argmax(q_U_val[id_mask], dim=1)
            val_preds_L = torch.argmax(q_L_val[id_mask], dim=1)
        else:
            loss = None
            val_labels = torch.empty(0, dtype=torch.long, device=self.device)
            val_preds_U = torch.empty(0, dtype=torch.long, device=self.device)
            val_preds_L = torch.empty(0, dtype=torch.long, device=self.device)

        output = {
            "val_preds_U": val_preds_U.detach(),
            "val_preds_L": val_preds_L.detach(),
            "val_labels": val_labels.detach(),
        }
        if self.ood_in_val:
            TU, AU, EU = compute_uncertainties(q_L_val, q_U_val)
            output.update({
                "TU": TU.detach(),
                "AU": AU.detach(),
                "EU": EU.detach(),
                "targets": (1 - y_val.sum(axis=1)).long().detach(),
            })
        self._validation_outputs.append(output)
        return loss

    def on_validation_epoch_end(self):
        if not self._validation_outputs:
            return

        labels = torch.cat([o["val_labels"] for o in self._validation_outputs], dim=0)
        preds_U = torch.cat([o["val_preds_U"] for o in self._validation_outputs], dim=0)
        preds_L = torch.cat([o["val_preds_L"] for o in self._validation_outputs], dim=0)
        if labels.numel() > 0:
            self.log("val_f1_U", self._f1_score(preds_U, labels), prog_bar=True)
            self.log("val_f1_L", self._f1_score(preds_L, labels))

        if self.ood_in_val and "EU" in self._validation_outputs[0]:
            targets = torch.cat([o["targets"] for o in self._validation_outputs], dim=0)
            EU = torch.cat([o["EU"] for o in self._validation_outputs], dim=0)
            AU = torch.cat([o["AU"] for o in self._validation_outputs], dim=0)
            TU = torch.cat([o["TU"] for o in self._validation_outputs], dim=0)
            self.log("val_auroc_EU", self._auroc_score(EU, targets), prog_bar=True)
            self.log("val_auroc_AU", self._auroc_score(AU, targets))
            self.log("val_auroc_TU", self._auroc_score(TU, targets))

        self._validation_outputs.clear()

    def on_test_epoch_start(self):
        self._test_outputs = []

    def test_step(self, batch, batch_idx):
        if not hasattr(batch, "test_mask") and not (hasattr(batch, "batch_size") and hasattr(batch, "n_id")):
            raise ValueError("Test batch must include 'test_mask'.")
        test_mask = self._split_mask(batch, "test")
        if not torch.any(test_mask):
            raise ValueError("'test_mask' is empty or all False in test batch.")

        q_L, q_U = self(batch)
        q_L_test = q_L[test_mask].detach()
        q_U_test = q_U[test_mask].detach()
        y_test   = batch.y[test_mask].detach()

        TU, AU, EU = compute_uncertainties(q_L_test, q_U_test)
        targets = (1 - y_test.sum(axis=1)).long()

        id_mask = (y_test.sum(axis=1) == 1)
        id_labels  = torch.argmax(y_test[id_mask], dim=1)
        id_preds_U = torch.argmax(q_U_test[id_mask], dim=1)
        id_preds_L = torch.argmax(q_L_test[id_mask], dim=1)

        self._test_outputs.append({
            "TU": TU.detach(),
            "AU": AU.detach(),
            "EU": EU.detach(),
            "targets": targets.detach(),
            "id_labels": id_labels.detach(),
            "id_preds_U": id_preds_U.detach(),
            "id_preds_L": id_preds_L.detach(),
        })

        return EU

    def on_test_epoch_end(self):
        if not self._test_outputs:
            return

        targets = torch.cat([o["targets"] for o in self._test_outputs], dim=0)
        EU = torch.cat([o["EU"] for o in self._test_outputs], dim=0)
        AU = torch.cat([o["AU"] for o in self._test_outputs], dim=0)
        TU = torch.cat([o["TU"] for o in self._test_outputs], dim=0)
        self.log("test_auroc_EU", self._auroc_score(EU, targets))
        self.log("test_auroc_AU", self._auroc_score(AU, targets))
        self.log("test_auroc_TU", self._auroc_score(TU, targets))

        id_labels = torch.cat([o["id_labels"] for o in self._test_outputs], dim=0)
        id_preds_U = torch.cat([o["id_preds_U"] for o in self._test_outputs], dim=0)
        id_preds_L = torch.cat([o["id_preds_L"] for o in self._test_outputs], dim=0)
        if id_labels.numel() > 0:
            self.log("test_accuracy_U", (id_preds_U == id_labels).float().mean())
            self.log("test_accuracy_L", (id_preds_L == id_labels).float().mean())
            self.log("test_f1_U", self._f1_score(id_preds_U, id_labels))
            self.log("test_f1_L", self._f1_score(id_preds_L, id_labels))

        self._test_outputs.clear()

    # ---------------------------
    # Optimizer: only Credal layer params
    # ---------------------------
    def configure_optimizers(self):
        params = [p for p in self.credal_layer_model.parameters() if p.requires_grad]
        if len(params) == 0:
            raise RuntimeError("No trainable parameters found in credal head.")
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        return opt

    def on_fit_start(self):
        # keep backbone frozen (defensive)
        for p in self.backbone.parameters():
            p.requires_grad = False
