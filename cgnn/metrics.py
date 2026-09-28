"""Evaluation metrics shared by all models.

AUROC
-----
``binary_auroc`` is an exact rank-based (Mann-Whitney U) AUROC in float64 with
ties counted as 1/2. It is the default because ``torchmetrics``' binary AUROC
silently applies a sigmoid to scores outside ``[0, 1]``; in float32 that
saturates large scores (Mahalanobis distances, energies at high temperature)
into ties and drives the AUROC towards 0.5 (docs/known_issues.md).

The legacy behaviour is available with ``set_auroc_impl("torchmetrics")``
(config key ``auroc_impl``) to reproduce pre-0.2 numbers.

Conventions: ``targets`` are 1 for OOD, 0 for ID; higher score = more OOD.
"""

from __future__ import annotations

import torch
from torchmetrics.functional.classification import binary_auroc as _tm_binary_auroc
from torchmetrics.functional.classification import multiclass_f1_score

AUROC_IMPLS = ("exact", "torchmetrics")
_AUROC_IMPL = "exact"


def set_auroc_impl(name: str) -> None:
    global _AUROC_IMPL
    if name not in AUROC_IMPLS:
        raise ValueError(f"auroc_impl must be one of {AUROC_IMPLS}, got {name!r}")
    _AUROC_IMPL = name


def get_auroc_impl() -> str:
    return _AUROC_IMPL


def _average_ranks(x: torch.Tensor) -> torch.Tensor:
    """1-based ranks with ties replaced by their average rank (like scipy.stats.rankdata)."""
    sorted_x, order = torch.sort(x)
    _, inverse, counts = torch.unique_consecutive(sorted_x, return_inverse=True, return_counts=True)
    ends = torch.cumsum(counts, dim=0).to(torch.float64)
    starts = ends - counts.to(torch.float64) + 1
    ranks_sorted = ((starts + ends) / 2)[inverse]
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def exact_binary_auroc(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    scores = scores.detach().reshape(-1).to(torch.float64)
    positives = targets.detach().reshape(-1).to(scores.device) > 0
    n_pos = positives.sum().to(torch.float64)
    n_neg = (~positives).sum().to(torch.float64)
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(float("nan"), dtype=torch.float64)
    ranks = _average_ranks(scores)
    return (ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def binary_auroc(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """AUROC of ``scores`` for detecting ``targets == 1`` (OOD). Returns a 0-dim float tensor."""
    if _AUROC_IMPL == "torchmetrics":
        return _tm_binary_auroc(scores, targets.long())
    return exact_binary_auroc(scores, targets).to(torch.float32)


def multiclass_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Micro-averaged F1 (== accuracy for single-label data). Kept for continuity with logged ``*_f1``."""
    return multiclass_f1_score(preds, labels, num_classes=num_classes, average="micro")
