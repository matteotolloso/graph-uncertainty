"""Shared LightningModule machinery.

Every model in ``cgnn.models`` derives from :class:`NodeUQModule`, which provides:

- ``split_mask(batch, split)``: node mask for full-batch *and* NeighborLoader
  batches (in a sampled batch only the first ``batch.batch_size`` nodes are seeds);
- :class:`EpochBuffer`: collect per-batch outputs, compute metrics **once per
  epoch** on the concatenated tensors (never average per-batch AUROCs);
- metric helpers (``auroc``, ``f1``) and ``weights_init`` (xavier on ``nn.Linear``);
- ``cgnn_meta``: dataset/split metadata written into every checkpoint under
  ``checkpoint["cgnn"]`` (used to detect backbone/split mismatches).

Metric names are a public contract (sweeps optimise them, ``cgnn results``
reads them): see ``.claude/rules/metrics-contract.md`` before renaming anything.
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import lightning as L
import torch
from torch import nn

from cgnn import metrics


def is_sampled_batch(batch) -> bool:
    return hasattr(batch, "batch_size") and hasattr(batch, "n_id")


def split_mask(batch, split: str) -> torch.Tensor:
    if is_sampled_batch(batch):
        mask = torch.zeros(batch.num_nodes, dtype=torch.bool, device=batch.x.device)
        mask[: batch.batch_size] = True
        return mask
    return getattr(batch, f"{split}_mask")


def ood_targets(y: torch.Tensor) -> torch.Tensor:
    """1 for OOD nodes (all-zero label row), 0 for ID nodes."""
    return (1 - y.sum(axis=1)).long()


def id_mask(y: torch.Tensor) -> torch.Tensor:
    return y.sum(axis=1) == 1


class EpochBuffer:
    """Accumulates detached tensors across the steps of one epoch."""

    def __init__(self) -> None:
        self._items: dict[str, list[torch.Tensor]] = defaultdict(list)

    def add(self, **tensors: torch.Tensor) -> None:
        for key, value in tensors.items():
            self._items[key].append(value.detach())

    def cat(self, key: str) -> torch.Tensor:
        return torch.cat(self._items[key], dim=0)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __bool__(self) -> bool:
        return bool(self._items)

    def clear(self) -> None:
        self._items.clear()


class NodeUQModule(L.LightningModule):
    """Base class: see module docstring."""

    C: int  # number of ID classes

    def __init__(self) -> None:
        super().__init__()
        self.cgnn_meta: dict = {}
        self._buffers_by_stage: dict[str, EpochBuffer] = {}

    # -- masks & metrics ------------------------------------------------------
    @staticmethod
    def split_mask(batch, split: str) -> torch.Tensor:
        return split_mask(batch, split)

    @staticmethod
    def auroc(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return metrics.binary_auroc(scores, targets)

    def f1(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return metrics.multiclass_f1(preds, labels, self.C)

    def log_test_extras(
        self,
        scores: dict[str, torch.Tensor],
        targets: torch.Tensor,
        id_preds: torch.Tensor | None = None,
        id_labels: torch.Tensor | None = None,
    ) -> None:
        """Extra test metrics for every uncertainty score (``{suffix: [N]}``, suffix as in ``test_auroc{suffix}``).

        OOD detection: ``test_aupr*`` / ``test_fpr95*`` with OOD as the positive class (FPR = ID nodes
        flagged at 95% OOD recall) and ``test_auprin*`` / ``test_fpr95in*`` with ID positive (the convention
        of GNNSafe/GEBM: OOD nodes accepted at 95% ID recall); mean score on ID / OOD nodes
        (``test_mean*_id``, ``test_mean*_ood``). Misclassification detection on ID nodes:
        ``test_misc_auroc*`` (positive = wrong prediction). All exact, whatever ``auroc_impl`` is.
        ``id_preds``/``id_labels`` must follow the order of the ID nodes (``targets == 0``) in ``scores``.
        """
        is_id = targets == 0
        errors = None
        if id_preds is not None and id_labels is not None:
            if id_preds.numel() == int(is_id.sum()):
                errors = (id_preds != id_labels).long()
            else:
                warnings.warn(
                    f"{type(self).__name__}: {id_preds.numel()} ID predictions for {int(is_id.sum())} ID nodes; "
                    "skipping test_misc_auroc",
                    stacklevel=2,
                )
        for suffix, score in scores.items():
            score = score.reshape(-1)
            self.log(f"test_aupr{suffix}", metrics.binary_aupr(score, targets))
            self.log(f"test_fpr95{suffix}", metrics.fpr_at_tpr(score, targets))
            self.log(f"test_auprin{suffix}", metrics.binary_aupr(-score, 1 - targets))
            self.log(f"test_fpr95in{suffix}", metrics.fpr_at_tpr(-score, 1 - targets))
            self.log(f"test_mean{suffix}_id", score[is_id].float().mean())
            self.log(f"test_mean{suffix}_ood", score[~is_id].float().mean())
            if errors is not None and errors.numel() > 0:
                misc = metrics.exact_binary_auroc(score[is_id.to(score.device)], errors).float()
                self.log(f"test_misc_auroc{suffix}", misc)

    def buffer(self, stage: str) -> EpochBuffer:
        if stage not in self._buffers_by_stage:
            self._buffers_by_stage[stage] = EpochBuffer()
        return self._buffers_by_stage[stage]

    # -- init / checkpoints ---------------------------------------------------
    @staticmethod
    def weights_init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.cgnn_meta:
            checkpoint["cgnn"] = dict(self.cgnn_meta)


class TrainableUQModule(NodeUQModule):
    """Adds the Adam optimiser used by every trainable model."""

    lr: float
    weight_decay: float

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
