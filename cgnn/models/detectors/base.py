"""Base class for post-hoc OOD detectors on a frozen VanillaGNN backbone.

Subclasses implement :meth:`PostHocDetector.ood_scores` (``[N]``, higher =
more OOD, computed for *all* nodes of the batch) and optionally
:meth:`prepare` (fit statistics on the training nodes of the full graph).

Logged: ``val_auroc`` and ``test_auroc`` (epoch-level, exact AUROC), plus the extra test metrics of
``NodeUQModule.log_test_extras`` (misclassification detection uses the backbone's predictions).
"""

from __future__ import annotations

import copy

import torch

from cgnn.models.base import NodeUQModule, id_mask, ood_targets
from cgnn.models.vanilla import VanillaGNN


def load_frozen_backbone(path: str) -> VanillaGNN:
    backbone = VanillaGNN.load_from_checkpoint(path, map_location="cpu")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def shallow_to(data, device):
    """Move a PyG ``Data`` to ``device`` without rebinding the caller's object."""
    return copy.copy(data).to(device)


class PostHocDetector(NodeUQModule):
    needs_input_grad: bool = False  # ODIN/Mahalanobis perturb inputs -> Trainer(inference_mode=False)

    def __init__(self, backbone_ckpt_path: str) -> None:
        super().__init__()
        self.backbone = load_frozen_backbone(backbone_ckpt_path)
        self.C = self.backbone.C

    @property
    def backbone_device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def prepare(self, train_data) -> None:
        """Fit detector statistics on ``train_data.train_mask`` nodes (default: nothing)."""

    def ood_scores(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, batch) -> torch.Tensor:
        return self.ood_scores(batch)

    # -- evaluation -------------------------------------------------------------
    def _collect(self, batch, split: str) -> None:
        scores = self.ood_scores(batch).detach()
        mask = self.split_mask(batch, split).to(scores.device)
        y = batch.y.to(scores.device)[mask]
        self.buffer(split).add(scores=scores[mask], targets=ood_targets(y))

    @torch.no_grad()
    def _collect_id_preds(self, batch) -> None:
        """Backbone predictions on the ID test nodes, in buffer order (misclassification detection)."""
        logits = self.backbone(shallow_to(batch, self.backbone_device)).detach()
        mask = self.split_mask(batch, "test").to(logits.device)
        y = batch.y.to(logits.device)[mask]
        is_id = id_mask(y)
        self.buffer("test").add(
            id_labels=torch.argmax(y[is_id], dim=1), id_preds=torch.argmax(logits[mask][is_id], dim=1)
        )

    def _log_auroc(self, split: str) -> None:
        buf = self.buffer(split)
        if buf:
            self.log(f"{split}_auroc", self.auroc(buf.cat("scores"), buf.cat("targets")), prog_bar=True)
            if split == "test":
                id_preds = buf.cat("id_preds") if "id_preds" in buf else None
                id_labels = buf.cat("id_labels") if "id_labels" in buf else None
                self.log_test_extras({"": buf.cat("scores")}, buf.cat("targets"), id_preds, id_labels)
        buf.clear()

    def on_validation_epoch_start(self):
        self.buffer("val").clear()

    def validation_step(self, batch, batch_idx):
        self._collect(batch, "val")

    def on_validation_epoch_end(self):
        self._log_auroc("val")

    def on_test_epoch_start(self):
        self.buffer("test").clear()

    def test_step(self, batch, batch_idx):
        self._collect(batch, "test")
        self._collect_id_preds(batch)

    def on_test_epoch_end(self):
        self._log_auroc("test")

    def configure_optimizers(self):
        return None
