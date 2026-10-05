"""Ensembles of pre-trained VanillaGNNs.

One module produces both paper baselines:
- **Classical Ensemble**: ``TU = H(mean p)``, ``AU = mean H(p)``, ``EU = TU - AU`` -> ``*_classic``;
- **CGNN by Ensemble** (credal wrapper): credal set = class-wise [min, max] of the
  member predictions, generalized-entropy TU/AU/EU -> ``*_credal``.

Logged: ``{val,test}_auroc_{EU,AU,TU}_{credal,classic}``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cgnn.models.base import NodeUQModule, ood_targets
from cgnn.models.detectors.base import load_frozen_backbone
from cgnn.uncertainty.ensemble import ensemble_uncertainties

_KEYS = [f"{k}_{f}" for f in ("credal", "classic") for k in ("EU", "AU", "TU")]


class CredalEnsemble(NodeUQModule):
    def __init__(self, checkpoint_paths: list[str]) -> None:
        super().__init__()
        self.save_hyperparameters()
        if not checkpoint_paths:
            raise ValueError("checkpoint_paths cannot be empty.")
        self.models = nn.ModuleList([load_frozen_backbone(p) for p in checkpoint_paths])
        self.C = self.models[0].C

    def forward(self, data) -> torch.Tensor:
        """Stacked member probabilities ``[M, N, C]``."""
        return torch.stack([F.softmax(m(data), dim=1) for m in self.models], dim=0)

    def _collect(self, batch, split: str) -> None:
        mask = self.split_mask(batch, split)
        stacked = self(batch)[:, mask, :].detach()
        self.buffer(split).add(targets=ood_targets(batch.y[mask]), **ensemble_uncertainties(stacked))

    def _log(self, split: str) -> None:
        buf = self.buffer(split)
        if buf:
            targets = buf.cat("targets")
            for key in _KEYS:
                self.log(
                    f"{split}_auroc_{key}",
                    self.auroc(buf.cat(key), targets),
                    prog_bar=split == "val" and key == "EU_credal",
                )
        buf.clear()

    def on_validation_epoch_start(self):
        self.buffer("val").clear()

    def validation_step(self, batch, batch_idx):
        self._collect(batch, "val")

    def on_validation_epoch_end(self):
        self._log("val")

    def on_test_epoch_start(self):
        self.buffer("test").clear()

    def test_step(self, batch, batch_idx):
        self._collect(batch, "test")

    def on_test_epoch_end(self):
        self._log("test")

    def configure_optimizers(self):
        return None
