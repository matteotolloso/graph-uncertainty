"""Credal Graph Neural Networks (the paper's CGNN and its ablations).

One configurable module replaces the six pre-0.2 classes. The method registry
(`cgnn/methods/trainable.py`) maps every legacy method key to a configuration:

====================================================  ========  =====  ======  =========
method key (paper name)                               latent    heads  detach  lambda_cons
====================================================  ========  =====  ======  =========
credal_LJ_dual_head_detached   (**CGNN**)             joint     dual   yes     -
credal                         (CGNN last layer)      last      credal -       -
credal_LJ                      (CGNN only credal)     joint     credal -       -
credal_LJ_dual_head                                   joint     dual   no      -
credal_LJ_dual_head_constrained                       joint     dual   no      yes
credal_LJ_dual_head_constrained_detached              joint     dual   yes     yes
====================================================  ========  =====  ======  =========

Representation fed to the heads (``latent``):

- ``"last"``: the standard backbone output ``z^L`` (no activation on the last layer).
- ``"joint"``: ``[z^1 || ... || z^L]`` with the activation applied after *every*
  layer (legacy). **The paper's Eq. 14 also includes the input ``z^0``**; set
  ``joint_include_input=True`` to get that (default False = what produced the
  AAAI numbers, see docs/known_issues.md).

Losses: ``credal_loss`` (Eq. 9) + ``lambda_cls * CE(classifier)`` (dual head)
+ ``lambda_cons * consistency`` (constrained variants). With ``detach_credal``
the credal head sees ``stopgrad(z)`` (Eq. 7).

Initialisation order (backbone, credal head, classifier head, xavier re-init)
is identical to the legacy classes so seeded runs reproduce bit-for-bit.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from cgnn.models.backbones import build_backbone, joint_representation
from cgnn.models.base import TrainableUQModule, id_mask, ood_targets
from cgnn.models.heads import CredalLayer, mlp_classifier
from cgnn.uncertainty.entropy import credal_uncertainties
from cgnn.uncertainty.losses import CreNetLoss


class CredalGNN(TrainableUQModule):
    def __init__(
        self,
        gnn_type: str,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        delta: float = 0.5,
        ood_in_val: bool = True,
        latent: Literal["joint", "last"] = "joint",
        heads: Literal["dual", "credal"] = "dual",
        detach_credal: bool | None = None,  # None: True for the dual head, False otherwise
        lambda_cls: float = 1.0,
        lambda_cons: float | None = None,
        joint_include_input: bool = False,
        joint_act_on_last: bool = True,
        credal_mid_activation: str = "sigmoid",
        credal_half_activation: str = "sigmoid",
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if latent not in ("joint", "last"):
            raise ValueError(f"latent must be 'joint' or 'last', got {latent!r}")
        if heads not in ("dual", "credal"):
            raise ValueError(f"heads must be 'dual' or 'credal', got {heads!r}")
        if detach_credal is None:
            detach_credal = heads == "dual"
        if heads == "credal" and (detach_credal or lambda_cons is not None):
            raise ValueError(
                "detach_credal/lambda_cons require heads='dual' (nothing else would train the backbone)"
            )

        self.gnn_model = build_backbone(
            gnn_type, in_channels, hidden_channels, num_layers, hidden_channels, **kwargs
        )
        self.C = out_channels
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.latent = latent
        self.dual = heads == "dual"
        self.detach_credal = detach_credal
        self.joint_include_input = joint_include_input
        self.joint_act_on_last = joint_act_on_last

        if latent == "joint":
            latent_dim = hidden_channels * num_layers + (in_channels if joint_include_input else 0)
        else:
            latent_dim = hidden_channels
        self.credal_layer_model = CredalLayer(
            input_dim=latent_dim,
            C=out_channels,
            mid_activation=credal_mid_activation,
            half_activation=credal_half_activation,
        )
        if self.dual:
            self.classifier_head = mlp_classifier(latent_dim, out_channels)

        self.criterion = CreNetLoss(delta=delta)
        self.lr = lr
        self.weight_decay = weight_decay
        self.ood_in_val = ood_in_val
        self.lambda_cls = lambda_cls
        self.lambda_cons = lambda_cons
        self.apply(self.weights_init)

    # -- forward ----------------------------------------------------------------
    def representation(self, data) -> torch.Tensor:
        if self.latent == "last":
            return self.gnn_model(data.x, data.edge_index)
        return joint_representation(
            self.gnn_model,
            data.x,
            data.edge_index,
            num_layers=self.num_layers,
            act_on_last=self.joint_act_on_last,
            include_input=self.joint_include_input,
        )

    def forward(self, data):
        """Returns ``(q_L, q_U)`` or, with the dual head, ``(q_L, q_U, logits_cls)``."""
        z = self.representation(data)
        q_L, q_U = self.credal_layer_model(z.detach() if self.detach_credal else z)
        if not self.dual:
            return q_L, q_U
        return q_L, q_U, self.classifier_head(z)

    def _outputs(self, batch):
        out = self(batch)
        return out if self.dual else (*out, None)

    # -- losses -----------------------------------------------------------------
    def _consistency_loss(self, q_L, q_U, logits_cls) -> torch.Tensor:
        p_cls = F.softmax(logits_cls, dim=1)
        return (F.relu(q_L - p_cls) + F.relu(p_cls - q_U)).sum(dim=1).mean()

    def _losses(self, q_L, q_U, logits_cls, y) -> dict[str, torch.Tensor]:
        credal_loss = self.criterion(q_L, q_U, y)
        if not self.dual:
            return {"loss": credal_loss}
        ce_cls = F.cross_entropy(logits_cls, torch.argmax(y, dim=1))
        out = {"credal_loss": credal_loss, "ce_cls": ce_cls}
        loss = credal_loss + self.lambda_cls * ce_cls
        if self.lambda_cons is not None:
            out["consistency_loss"] = self._consistency_loss(q_L, q_U, logits_cls)
            loss = loss + self.lambda_cons * out["consistency_loss"]
        out["loss"] = loss
        return out

    # -- training -----------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        q_L, q_U, logits_cls = self._outputs(batch)
        mask = self.split_mask(batch, "train")
        y = batch.y[mask]
        q_L, q_U = q_L[mask], q_U[mask]
        logits_cls = logits_cls[mask] if self.dual else None
        losses = self._losses(q_L, q_U, logits_cls, y)
        labels = torch.argmax(y, dim=1)
        preds_U, preds_L = torch.argmax(q_U, dim=1), torch.argmax(q_L, dim=1)

        n = mask.sum()
        self.log("train_loss", losses["loss"], batch_size=n)
        if self.dual:
            preds_cls = torch.argmax(logits_cls, dim=1)
            self.log("train_credal_loss", losses["credal_loss"], batch_size=n)
            self.log("train_ce_cls", losses["ce_cls"], batch_size=n)
            if "consistency_loss" in losses:
                self.log("train_consistency_loss", losses["consistency_loss"], batch_size=n)
            self.log("train_acc_cls", (preds_cls == labels).float().mean(), batch_size=n)
            self.log("train_f1_cls", self.f1(preds_cls, labels), batch_size=n)
        self.log("train_acc_U", (preds_U == labels).float().mean(), batch_size=n)
        self.log("train_acc_L", (preds_L == labels).float().mean(), batch_size=n)
        self.log("train_f1_U", self.f1(preds_U, labels), batch_size=n)
        self.log("train_f1_L", self.f1(preds_L, labels), batch_size=n)
        return losses["loss"]

    # -- validation ---------------------------------------------------------------
    def on_validation_epoch_start(self):
        self.buffer("val").clear()

    def validation_step(self, batch, batch_idx):
        q_L, q_U, logits_cls = self._outputs(batch)
        mask = self.split_mask(batch, "val")
        q_L, q_U, y = q_L[mask].detach(), q_U[mask].detach(), batch.y[mask].detach()
        logits_cls = logits_cls[mask].detach() if self.dual else None

        is_id = id_mask(y)
        loss = None
        empty = torch.empty(0, dtype=torch.long, device=self.device)
        labels = preds_cls = preds_U = preds_L = empty
        if is_id.any():
            n = is_id.sum()
            losses = self._losses(q_L[is_id], q_U[is_id], logits_cls[is_id] if self.dual else None, y[is_id])
            loss = losses["loss"]
            labels = torch.argmax(y[is_id], dim=1)
            preds_U = torch.argmax(q_U[is_id], dim=1)
            preds_L = torch.argmax(q_L[is_id], dim=1)
            self.log("val_loss", loss, prog_bar=True, batch_size=n, on_step=False, on_epoch=True)
            if self.dual:
                preds_cls = torch.argmax(logits_cls[is_id], dim=1)
                self.log("val_credal_loss", losses["credal_loss"], batch_size=n, on_step=False, on_epoch=True)
                self.log("val_ce_cls", losses["ce_cls"], batch_size=n, on_step=False, on_epoch=True)
                if "consistency_loss" in losses:
                    self.log(
                        "val_consistency_loss",
                        losses["consistency_loss"],
                        batch_size=n,
                        on_step=False,
                        on_epoch=True,
                    )
                self.log(
                    "val_acc_cls",
                    (preds_cls == labels).float().mean(),
                    batch_size=n,
                    on_step=False,
                    on_epoch=True,
                )

        buf = self.buffer("val")
        buf.add(labels=labels, preds_U=preds_U, preds_L=preds_L)
        if self.dual:
            buf.add(preds_cls=preds_cls)
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
            if self.dual:
                self.log("val_f1_cls", self.f1(buf.cat("preds_cls"), labels), prog_bar=True)
            self.log("val_f1_U", self.f1(buf.cat("preds_U"), labels), prog_bar=not self.dual)
            self.log("val_f1_L", self.f1(buf.cat("preds_L"), labels))
        if self.ood_in_val and "EU" in buf:
            targets = buf.cat("targets")
            self.log("val_auroc_EU", self.auroc(buf.cat("EU"), targets), prog_bar=True)
            self.log("val_auroc_AU", self.auroc(buf.cat("AU"), targets))
            self.log("val_auroc_TU", self.auroc(buf.cat("TU"), targets))
        buf.clear()

    # -- test -----------------------------------------------------------------------
    def on_test_epoch_start(self):
        self.buffer("test").clear()

    def test_step(self, batch, batch_idx):
        q_L, q_U, logits_cls = self._outputs(batch)
        mask = self.split_mask(batch, "test")
        q_L, q_U, y = q_L[mask].detach(), q_U[mask].detach(), batch.y[mask].detach()

        TU, AU, EU = credal_uncertainties(q_L, q_U)
        is_id = id_mask(y)
        buf = self.buffer("test")
        buf.add(
            TU=TU,
            AU=AU,
            EU=EU,
            targets=ood_targets(y),
            id_labels=torch.argmax(y[is_id], dim=1),
            id_preds_U=torch.argmax(q_U[is_id], dim=1),
            id_preds_L=torch.argmax(q_L[is_id], dim=1),
        )
        if self.dual:
            buf.add(id_preds_cls=torch.argmax(logits_cls[mask].detach()[is_id], dim=1))
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
            heads = (["cls"] if self.dual else []) + ["U", "L"]
            for h in heads:
                self.log(f"test_accuracy_{h}", (buf.cat(f"id_preds_{h}") == id_labels).float().mean())
            for h in heads:
                self.log(f"test_f1_{h}", self.f1(buf.cat(f"id_preds_{h}"), id_labels))
        self.log_test_extras(
            {f"_{k}": buf.cat(k) for k in ("EU", "AU", "TU")},
            targets,
            buf.cat("id_preds_cls" if self.dual else "id_preds_U"),
            id_labels,
        )
        buf.clear()
