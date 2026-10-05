"""CGNN post train: credal head on the joint latent of a frozen, pre-trained VanillaGNN.

Joint = ``[x || act(z^1) || ... || z^L]`` (input features included, last layer
= backbone logits, i.e. no activation), matching the legacy
``models/credal_frozen_LJ.py``. Only the credal head is optimised.
"""

from __future__ import annotations

import torch
from torch import nn

from cgnn.models.backbones import joint_representation
from cgnn.models.base import NodeUQModule, id_mask, ood_targets
from cgnn.models.heads import CredalLayer
from cgnn.models.vanilla import VanillaGNN
from cgnn.uncertainty.entropy import credal_uncertainties
from cgnn.uncertainty.losses import CreNetLoss


class CredalFrozenJoint(NodeUQModule):
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

        self.backbone: VanillaGNN = VanillaGNN.load_from_checkpoint(checkpoint_path, map_location="cpu")
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        hp = self.backbone.hparams
        in_channels, hidden, num_layers, out = (
            int(hp["in_channels"]),
            int(hp["hidden_channels"]),
            int(hp["num_layers"]),
            int(hp["out_channels"]),
        )
        # x (in) + (L-1) hidden layers + last layer (logits, out_channels)
        joint_dim = in_channels + max(num_layers - 1, 0) * hidden + out

        self.C = out
        self.ood_in_val = ood_in_val
        self.credal_layer_model = CredalLayer(input_dim=joint_dim, C=self.C)
        self.credal_layer_model.apply(self.weights_init)

        self.criterion = CreNetLoss(delta=delta)
        self.lr = lr
        self.weight_decay = weight_decay

    @torch.no_grad()
    def _get_joint_embeddings(self, data) -> torch.Tensor:
        joint = joint_representation(
            self.backbone.gnn_model, data.x, data.edge_index, act_on_last=False, include_input=True
        )
        if joint.size(1) != self.credal_layer_model.input_dim:
            raise RuntimeError(
                f"Computed joint dim {joint.size(1)} != credal input_dim {self.credal_layer_model.input_dim}."
            )
        return joint

    def forward(self, data):
        return self.credal_layer_model(self._get_joint_embeddings(data))

    def training_step(self, batch, batch_idx):
        mask = self.split_mask(batch, "train")
        if not torch.any(mask):
            raise ValueError("'train_mask' is empty or all False in training batch.")
        q_L, q_U = self(batch)
        y, q_L, q_U = batch.y[mask], q_L[mask], q_U[mask]
        loss = self.criterion(q_L, q_U, y)
        labels = torch.argmax(y, dim=1)
        preds_U, preds_L = torch.argmax(q_U, dim=1), torch.argmax(q_L, dim=1)

        n = mask.sum()
        self.log("train_loss", loss, batch_size=n)
        self.log("train_acc_U", (preds_U == labels).float().mean(), batch_size=n)
        self.log("train_acc_L", (preds_L == labels).float().mean(), batch_size=n)
        self.log("train_f1_U", self.f1(preds_U, labels), batch_size=n)
        self.log("train_f1_L", self.f1(preds_L, labels), batch_size=n)
        return loss

    def on_validation_epoch_start(self):
        self.buffer("val").clear()

    def validation_step(self, batch, batch_idx):
        mask = self.split_mask(batch, "val")
        if not torch.any(mask):
            raise ValueError("'val_mask' is empty or all False in validation batch.")
        q_L, q_U = self(batch)
        q_L, q_U, y = q_L[mask].detach(), q_U[mask].detach(), batch.y[mask].detach()

        is_id = id_mask(y)
        loss = None
        empty = torch.empty(0, dtype=torch.long, device=self.device)
        labels = preds_U = preds_L = empty
        if is_id.any():
            loss = self.criterion(q_L[is_id], q_U[is_id], y[is_id])
            self.log("val_loss", loss, prog_bar=True, batch_size=is_id.sum(), on_step=False, on_epoch=True)
            labels = torch.argmax(y[is_id], dim=1)
            preds_U = torch.argmax(q_U[is_id], dim=1)
            preds_L = torch.argmax(q_L[is_id], dim=1)

        buf = self.buffer("val")
        buf.add(labels=labels, preds_U=preds_U, preds_L=preds_L)
        if self.ood_in_val:
            TU, AU, EU = credal_uncertainties(q_L, q_U)
            buf.add(TU=TU, AU=AU, EU=EU, targets=ood_targets(y))
        return loss

    def on_validation_epoch_end(self):
        buf = self.buffer("val")
        if not buf:
            return
        labels = buf.cat("labels")
        if labels.numel() > 0:
            self.log("val_f1_U", self.f1(buf.cat("preds_U"), labels), prog_bar=True)
            self.log("val_f1_L", self.f1(buf.cat("preds_L"), labels))
        if self.ood_in_val and "EU" in buf:
            targets = buf.cat("targets")
            self.log("val_auroc_EU", self.auroc(buf.cat("EU"), targets), prog_bar=True)
            self.log("val_auroc_AU", self.auroc(buf.cat("AU"), targets))
            self.log("val_auroc_TU", self.auroc(buf.cat("TU"), targets))
        buf.clear()

    def on_test_epoch_start(self):
        self.buffer("test").clear()

    def test_step(self, batch, batch_idx):
        mask = self.split_mask(batch, "test")
        if not torch.any(mask):
            raise ValueError("'test_mask' is empty or all False in test batch.")
        q_L, q_U = self(batch)
        q_L, q_U, y = q_L[mask].detach(), q_U[mask].detach(), batch.y[mask].detach()
        TU, AU, EU = credal_uncertainties(q_L, q_U)
        is_id = id_mask(y)
        self.buffer("test").add(
            TU=TU,
            AU=AU,
            EU=EU,
            targets=ood_targets(y),
            id_labels=torch.argmax(y[is_id], dim=1),
            id_preds_U=torch.argmax(q_U[is_id], dim=1),
            id_preds_L=torch.argmax(q_L[is_id], dim=1),
        )
        return EU

    def on_test_epoch_end(self):
        buf = self.buffer("test")
        if not buf:
            return
        targets = buf.cat("targets")
        self.log("test_auroc_EU", self.auroc(buf.cat("EU"), targets))
        self.log("test_auroc_AU", self.auroc(buf.cat("AU"), targets))
        self.log("test_auroc_TU", self.auroc(buf.cat("TU"), targets))
        id_labels = buf.cat("id_labels")
        if id_labels.numel() > 0:
            self.log("test_accuracy_U", (buf.cat("id_preds_U") == id_labels).float().mean())
            self.log("test_accuracy_L", (buf.cat("id_preds_L") == id_labels).float().mean())
            self.log("test_f1_U", self.f1(buf.cat("id_preds_U"), id_labels))
            self.log("test_f1_L", self.f1(buf.cat("id_preds_L"), id_labels))
        self.log_test_extras(
            {f"_{k}": buf.cat(k) for k in ("EU", "AU", "TU")}, targets, buf.cat("id_preds_U"), id_labels
        )
        buf.clear()

    def configure_optimizers(self):
        params = [p for p in self.credal_layer_model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters found in credal head.")
        return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

    def on_fit_start(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    @staticmethod
    def weights_init(m: nn.Module) -> None:
        NodeUQModule.weights_init(m)
