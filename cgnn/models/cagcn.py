"""CaGCN (Wang et al. 2021), post-hoc confidence calibration on a frozen VanillaGNN.

A GCN over the frozen logits predicts a per-node temperature
``t = log(exp(t_raw) + 1.1)``; calibrated logits are ``logits * t``. The scaler
is trained with NLL on the **ID validation nodes** (final-scaling stage of the
official repo); OOD score = negative MSP of the calibrated logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn.models import GCN

from cgnn.models.base import NodeUQModule, id_mask, ood_targets
from cgnn.models.vanilla import VanillaGNN


class CaGCNModule(NodeUQModule):
    def __init__(
        self,
        checkpoint_path: str,
        calib_hidden: int = 16,
        calib_layers: int = 2,
        lr: float = 1e-2,
        weight_decay: float = 5e-3,
        softplus_eps: float = 1.1,
        ood_in_val: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.base: VanillaGNN = VanillaGNN.load_from_checkpoint(checkpoint_path, map_location="cpu")
        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad = False

        self.C = self.base.C
        self.ood_in_val = ood_in_val
        self.softplus_eps = softplus_eps
        self.scaler = GCN(
            in_channels=self.C,
            hidden_channels=calib_hidden,
            num_layers=calib_layers,
            out_channels=1,
            act=F.sigmoid,
        )
        self.lr = lr
        self.weight_decay = weight_decay

    @torch.no_grad()
    def _base_logits(self, data) -> torch.Tensor:
        return self.base(data)

    def forward(self, data) -> torch.Tensor:
        logits = self._base_logits(data)
        t = torch.log(torch.exp(self.scaler(logits, data.edge_index)) + self.softplus_eps)
        return logits * t

    def training_step(self, batch, batch_idx):
        val_mask = self.split_mask(batch, "val")
        if not torch.any(val_mask):
            raise ValueError("'val_mask' is empty or all False.")
        logits_val = self(batch)[val_mask]
        y_val = batch.y[val_mask]
        is_id = id_mask(y_val)
        if not is_id.any():
            loss = logits_val.sum() * 0.0
        else:
            loss = F.cross_entropy(logits_val[is_id], torch.argmax(y_val[is_id], dim=1))
        self.log("train_loss", loss, prog_bar=True)
        self.log("val_nll", loss)
        return loss

    def _neg_msp(self, logits: torch.Tensor) -> torch.Tensor:
        return -F.softmax(logits, dim=1).max(dim=1).values

    def on_validation_epoch_start(self):
        self.buffer("val").clear()

    def validation_step(self, batch, batch_idx):
        mask = self.split_mask(batch, "val")
        logits = self(batch)[mask]
        self.buffer("val").add(scores=self._neg_msp(logits), targets=ood_targets(batch.y[mask]))

    def on_validation_epoch_end(self):
        buf = self.buffer("val")
        if buf:
            self.log("val_auroc", self.auroc(buf.cat("scores"), buf.cat("targets")), prog_bar=True)
        buf.clear()

    def on_test_epoch_start(self):
        self.buffer("test").clear()

    def test_step(self, batch, batch_idx):
        mask = self.split_mask(batch, "test")
        logits, y = self(batch)[mask], batch.y[mask]
        is_id = id_mask(y)
        self.buffer("test").add(
            scores=self._neg_msp(logits),
            targets=ood_targets(y),
            id_labels=torch.argmax(y[is_id], dim=1),
            id_preds=torch.argmax(logits[is_id], dim=1),
        )

    def on_test_epoch_end(self):
        buf = self.buffer("test")
        if not buf:
            return
        self.log("test_auroc", self.auroc(buf.cat("scores"), buf.cat("targets")), prog_bar=True)
        id_labels, id_preds = buf.cat("id_labels"), buf.cat("id_preds")
        if id_labels.numel() > 0:
            self.log("test_acc", (id_preds == id_labels).float().mean())
            self.log("test_f1", self.f1(id_preds, id_labels))
        self.log_test_extras({"": buf.cat("scores")}, buf.cat("targets"), id_preds, id_labels)
        buf.clear()

    def configure_optimizers(self):
        params = [p for p in self.scaler.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
