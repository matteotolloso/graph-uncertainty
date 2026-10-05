"""AUROC implementations (the exact one is the default)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from cgnn import metrics


def _reference_auroc(scores, targets):
    """O(n^2) definition: P(score_ood > score_id) + 0.5 P(tie)."""
    pos, neg = scores[targets == 1], scores[targets == 0]
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    return ((diff > 0).double().sum() + 0.5 * (diff == 0).double().sum()).item() / (len(pos) * len(neg))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exact_auroc_matches_definition_with_ties(seed):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randint(0, 20, (300,), generator=g).float()  # many ties
    targets = torch.randint(0, 2, (300,), generator=g)
    assert math.isclose(
        metrics.exact_binary_auroc(scores, targets).item(), _reference_auroc(scores, targets), abs_tol=1e-12
    )


def test_exact_auroc_is_invariant_to_score_scale():
    g = torch.Generator().manual_seed(0)
    targets = torch.randint(0, 2, (200,), generator=g)
    scores = torch.rand(200, generator=g, dtype=torch.float64) + targets * 0.3  # float64: no new ties
    base = metrics.exact_binary_auroc(scores, targets)
    for transform in (lambda s: 1000 * s + 100, lambda s: s - 1000):  # increasing maps
        assert torch.isclose(metrics.exact_binary_auroc(transform(scores), targets), base)


def test_torchmetrics_impl_saturates_large_scores():
    """Documents the legacy bug: torchmetrics sigmoids scores outside [0, 1]."""
    scores = torch.tensor([100.0, 200.0, 300.0, 400.0])
    targets = torch.tensor([0, 0, 1, 1])
    try:
        metrics.set_auroc_impl("torchmetrics")
        assert metrics.binary_auroc(scores, targets).item() == pytest.approx(0.5)
        metrics.set_auroc_impl("exact")
        assert metrics.binary_auroc(scores, targets).item() == pytest.approx(1.0)
    finally:
        metrics.set_auroc_impl("exact")


def test_auroc_degenerate_and_invalid_inputs():
    assert math.isnan(metrics.exact_binary_auroc(torch.rand(5), torch.zeros(5)).item())
    with pytest.raises(ValueError):
        metrics.set_auroc_impl("sklearn")


def test_micro_f1_equals_accuracy():
    preds, labels = torch.tensor([0, 1, 2, 2, 1]), torch.tensor([0, 2, 2, 2, 1])
    assert metrics.multiclass_f1(preds, labels, 3).item() == pytest.approx(0.8)


@pytest.mark.parametrize("ties", [False, True])
def test_aupr_and_fpr95_match_sklearn(ties):
    sk = pytest.importorskip("sklearn.metrics")
    g = torch.Generator().manual_seed(0)
    targets = (torch.rand(500, generator=g) < 0.3).long()
    scores = torch.randn(500, generator=g) + targets
    if ties:
        scores = (2 * scores).round()
    assert metrics.binary_aupr(scores, targets).item() == pytest.approx(
        sk.average_precision_score(targets.numpy(), scores.numpy()), abs=1e-6
    )
    fpr, tpr, _ = sk.roc_curve(targets.numpy(), scores.numpy())
    expected = fpr[np.searchsorted(tpr, 0.95, side="left")]
    assert metrics.fpr_at_tpr(scores, targets).item() == pytest.approx(expected, abs=1e-6)
