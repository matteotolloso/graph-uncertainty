"""GEBM (Fuchsgruber et al. 2024) via ``graph_uq.gebm.GraphEBMWrapper`` (graph-ebm submodule).

Fit: logits + embeddings computed *with* edges on the full graph, training nodes.
Score: logits/embeddings computed *without* edges, energies diffused over the
full-graph ``edge_index`` stored at fit time. Needs full-graph evaluation
(the method spec forces full-batch val/test loaders).

Note: the embeddings passed to GEBM are the joint latent ``[x || z^1 .. z^L]``;
the original GEBM uses the penultimate layer.
"""

from __future__ import annotations

import enum
import typing

import torch
import torch.nn.functional as F

from cgnn.models.backbones import joint_representation
from cgnn.models.base import id_mask
from cgnn.models.detectors.base import PostHocDetector, shallow_to


def _python310_backports() -> None:
    """graph_uq uses ``enum.StrEnum`` and ``typing.Self`` (Python 3.11+)."""
    if not hasattr(enum, "StrEnum"):

        class StrEnum(str, enum.Enum):
            def __str__(self):
                return str(self.value)

        enum.StrEnum = StrEnum
    if not hasattr(typing, "Self"):
        from typing_extensions import Self

        typing.Self = Self


class GEBMDetector(PostHocDetector):
    requires_full_graph_eval = True

    def __init__(self, backbone_ckpt_path: str):
        super().__init__(backbone_ckpt_path)
        _python310_backports()
        from graph_uq.gebm import GraphEBMWrapper

        self.gebm = GraphEBMWrapper()
        self._fitted = False
        self._edge_index_cpu = None

    @torch.no_grad()
    def _embeddings(self, data) -> torch.Tensor:
        return joint_representation(
            self.backbone.gnn_model, data.x, data.edge_index, act_on_last=False, include_input=True
        )

    @torch.no_grad()
    def prepare(self, train_data) -> None:
        if not torch.any(train_data.train_mask):
            raise ValueError("'train_mask' is empty.")
        data = shallow_to(train_data, self.backbone_device)
        self.gebm.fit(
            self.backbone(data).detach().cpu(),
            self._embeddings(data).detach().cpu(),
            data.edge_index.detach().cpu(),
            torch.argmax(data.y, dim=1).detach().cpu(),
            data.train_mask.detach().cpu(),
        )
        self._edge_index_cpu = data.edge_index.detach().cpu()
        self._fitted = True

    @torch.no_grad()
    def ood_scores(self, batch) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("Call prepare(train_data) before evaluation.")
        no_edges = batch.clone()
        no_edges.edge_index = torch.empty((2, 0), dtype=batch.edge_index.dtype, device=batch.x.device)
        return self.gebm.get_uncertainty(
            logits_unpropagated=self.backbone(no_edges).detach().cpu(),
            embeddings_unpropagated=self._embeddings(no_edges).detach().cpu(),
            edge_index=self._edge_index_cpu,
        )

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self._collect(batch, "test")
        mask = self.split_mask(batch, "test").cpu()
        y = batch.y.detach().cpu()[mask]
        is_id = id_mask(y)
        if is_id.any():
            logits = self.backbone(batch).detach().cpu()[mask][is_id]
            self.buffer("test").add(
                id_labels=torch.argmax(y[is_id], dim=1),
                id_preds=torch.argmax(F.softmax(logits, dim=1), dim=1),
            )

    def on_test_epoch_end(self):
        buf = self.buffer("test")
        if buf and "id_labels" in buf:
            id_labels, id_preds = buf.cat("id_labels"), buf.cat("id_preds")
            self.log("test_acc", (id_preds == id_labels).float().mean())
            self.log("test_f1", self.f1(id_preds, id_labels))
        self._log_auroc("test")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self._collect(batch, "val")
