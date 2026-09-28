"""TEMPLATE — trainable model. Copy to cgnn/models/<name>.py and adapt.

Contract (checked by tests/test_runner.py):
- logs the sweep metric (`defaults.monitor`) on validation, once per epoch;
- logs `test_auroc{_key}` for every key in the MethodSpec's `score_keys`;
- higher score = more OOD; targets from `ood_targets(y)` (1 = OOD).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cgnn.models.backbones import build_backbone
from cgnn.models.base import TrainableUQModule, id_mask, ood_targets


class MyModel(TrainableUQModule):
    def __init__(
        self,
        gnn_type: str,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        alpha: float = 1.0,  # <- your hyper-parameters (add them to configs/methods/<name>.yaml)
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Construction order defines the RNG stream: keep it stable once results exist.
        self.gnn_model = build_backbone(
            gnn_type, in_channels, hidden_channels, num_layers, out_channels, **kwargs
        )
        self.C = out_channels
        self.lr, self.weight_decay, self.alpha = lr, weight_decay, alpha
        self.apply(self.weights_init)

    def forward(self, data) -> torch.Tensor:
        return self.gnn_model(data.x, data.edge_index)

    def ood_score(self, logits: torch.Tensor) -> torch.Tensor:
        return -F.softmax(logits, dim=1).max(dim=1).values  # replace with your uncertainty

    def training_step(self, batch, batch_idx):
        mask = self.split_mask(batch, "train")  # handles NeighborLoader seeds
        logits = self(batch)[mask]
        loss = F.cross_entropy(logits, batch.y[mask].argmax(1))
        self.log("train_loss", loss, batch_size=mask.sum())
        return loss

    # -- validation: collect per step, compute AUROC once per epoch -----------------------------
    def on_validation_epoch_start(self):
        self.buffer("val").clear()

    def validation_step(self, batch, batch_idx):
        mask = self.split_mask(batch, "val")
        logits, y = self(batch)[mask], batch.y[mask]
        self.buffer("val").add(scores=self.ood_score(logits), targets=ood_targets(y))
        is_id = id_mask(y)
        if is_id.any():
            self.log(
                "val_loss",
                F.cross_entropy(logits[is_id], y[is_id].argmax(1)),
                batch_size=is_id.sum(),
                on_step=False,
                on_epoch=True,
            )

    def on_validation_epoch_end(self):
        buf = self.buffer("val")
        if buf:
            self.log("val_auroc", self.auroc(buf.cat("scores"), buf.cat("targets")), prog_bar=True)
        buf.clear()

    # -- test ---------------------------------------------------------------------------------------
    def on_test_epoch_start(self):
        self.buffer("test").clear()

    def test_step(self, batch, batch_idx):
        mask = self.split_mask(batch, "test")
        logits, y = self(batch)[mask], batch.y[mask]
        is_id = id_mask(y)
        self.buffer("test").add(
            scores=self.ood_score(logits),
            targets=ood_targets(y),
            id_labels=y[is_id].argmax(1),
            id_preds=logits[is_id].argmax(1),
        )

    def on_test_epoch_end(self):
        buf = self.buffer("test")
        if not buf:
            return
        self.log("test_auroc", self.auroc(buf.cat("scores"), buf.cat("targets")))
        labels, preds = buf.cat("id_labels"), buf.cat("id_preds")
        if labels.numel():
            self.log("test_acc", (preds == labels).float().mean())
            self.log("test_f1", self.f1(preds, labels))
        buf.clear()


_ = nn  # keep the import if you add layers
