"""Standard GNN classifier; also the frozen backbone of every post-hoc method.

OOD score: negative maximum softmax probability (MSP).
Logged metrics: train_{loss,acc,f1}, val_{loss,acc,f1,auroc}, test_{auroc,acc,f1}.

Checkpoints (``checkpoints/``) load with ``VanillaGNN.load_from_checkpoint``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cgnn.models.backbones import build_backbone
from cgnn.models.base import TrainableUQModule, id_mask, ood_targets


class VanillaGNN(TrainableUQModule):
    def __init__(
        self,
        gnn_type: str,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        ood_in_val: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.gnn_model = build_backbone(
            gnn_type, in_channels, hidden_channels, num_layers, out_channels, **kwargs
        )
        self.C = out_channels
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.ood_in_val = ood_in_val
        self.apply(self.weights_init)

    def forward(self, data) -> torch.Tensor:
        return self.gnn_model(data.x, data.edge_index)  # logits [N, C]

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        train_mask = self.split_mask(batch, "train")
        logits_train = logits[train_mask]
        target = torch.argmax(batch.y[train_mask], dim=1)

        loss = self.criterion(logits_train, target)
        preds = torch.argmax(logits_train, dim=1)

        n = train_mask.sum()
        self.log("train_loss", loss, batch_size=n)
        self.log("train_acc", (preds == target).float().mean(), batch_size=n)
        self.log("train_f1", self.f1(preds, target), batch_size=n)
        return loss

    # -- validation -----------------------------------------------------------
    def on_validation_epoch_start(self):
        self.buffer("val").clear()

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        val_mask = self.split_mask(batch, "val")
        logits_val = logits[val_mask].detach()
        y_val = batch.y[val_mask].detach()

        is_id = id_mask(y_val)
        loss = None
        if is_id.any():
            id_logits = logits_val[is_id]
            id_labels = torch.argmax(y_val[is_id], dim=1)
            loss = self.criterion(id_logits, id_labels)
            id_preds = torch.argmax(id_logits, dim=1)
            n = is_id.sum()
            self.log("val_loss", loss, prog_bar=True, batch_size=n, on_step=False, on_epoch=True)
            self.log(
                "val_acc", (id_preds == id_labels).float().mean(), batch_size=n, on_step=False, on_epoch=True
            )
        else:
            id_labels = torch.empty(0, dtype=torch.long, device=self.device)
            id_preds = torch.empty(0, dtype=torch.long, device=self.device)

        buf = self.buffer("val")
        buf.add(id_labels=id_labels, id_preds=id_preds)
        if self.ood_in_val:
            msp = F.softmax(logits_val, dim=1).max(dim=1).values
            buf.add(ood_scores=-msp, ood_targets=ood_targets(y_val))
        return loss

    def on_validation_epoch_end(self):
        buf = self.buffer("val")
        if not buf:
            return
        id_labels = buf.cat("id_labels")
        if id_labels.numel() > 0:
            self.log("val_f1", self.f1(buf.cat("id_preds"), id_labels), prog_bar=True)
        if self.ood_in_val and "ood_scores" in buf:
            self.log("val_auroc", self.auroc(buf.cat("ood_scores"), buf.cat("ood_targets")), prog_bar=True)
        buf.clear()

    # -- test -----------------------------------------------------------------
    def on_test_epoch_start(self):
        self.buffer("test").clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        test_mask = self.split_mask(batch, "test")
        logits_test = logits[test_mask].detach()
        y_test = batch.y[test_mask].detach()

        msp = F.softmax(logits_test, dim=1).max(dim=1).values
        is_id = id_mask(y_test)
        self.buffer("test").add(
            ood_scores=-msp,
            ood_targets=ood_targets(y_test),
            id_labels=torch.argmax(y_test[is_id], dim=1),
            id_preds=torch.argmax(logits_test[is_id], dim=1),
        )
        return -msp

    def on_test_epoch_end(self):
        buf = self.buffer("test")
        if not buf:
            return
        self.log("test_auroc", self.auroc(buf.cat("ood_scores"), buf.cat("ood_targets")))
        id_labels, id_preds = buf.cat("id_labels"), buf.cat("id_preds")
        if id_labels.numel() > 0:
            self.log("test_acc", (id_preds == id_labels).float().mean())
            self.log("test_f1", self.f1(id_preds, id_labels))
        buf.clear()
