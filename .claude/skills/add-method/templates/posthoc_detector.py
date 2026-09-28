"""TEMPLATE — post-hoc detector on a frozen VanillaGNN. Add to cgnn/models/detectors/ and export it.

The base class loads/freezes the backbone (``self.backbone``), collects scores per step and logs
``val_auroc`` / ``test_auroc`` once per epoch. You only implement the score (+ optional statistics).
"""

from __future__ import annotations

import torch

from cgnn.models.detectors.base import PostHocDetector, shallow_to


class MyDetector(PostHocDetector):
    needs_input_grad = False  # True if you back-propagate to the input features (ODIN-style)
    requires_full_graph_eval = False  # True if scores need the whole graph (propagation over all edges)

    def __init__(self, backbone_ckpt_path: str, param1: float = 1.0):
        super().__init__(backbone_ckpt_path)
        self.save_hyperparameters()
        self.param1 = float(param1)
        self.stats = None

    def prepare(self, train_data) -> None:
        """Optional: fit statistics on the ID training nodes of the full graph."""
        data = shallow_to(train_data, self.backbone_device)  # never move the caller's Data in place
        with torch.no_grad():
            logits = self.backbone(data)[data.train_mask]
        self.stats = logits.mean(dim=0)

    def ood_scores(self, data) -> torch.Tensor:
        """``[N]`` scores for all nodes of the batch; higher = more OOD."""
        with torch.no_grad():
            logits = self.backbone(data.to(self.backbone_device))
        return (logits - self.stats).norm(dim=1) * self.param1
