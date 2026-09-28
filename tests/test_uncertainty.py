"""Credal-set math: interval softmax, reachable bounds, entropy solvers, credal loss."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from cgnn.uncertainty import (
    CreNetLoss,
    classical_decomposition,
    credal_uncertainties,
    ensemble_uncertainties,
    entropy,
    interval_softmax,
    max_entropy_distribution,
    min_entropy_distribution,
    probs_cross_entropy,
    reachable_bounds,
)
from cgnn.uncertainty.entropy import set_min_entropy_method


def _intervals(n=400, C=4, scale=2.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = torch.sigmoid(scale * torch.randn(n, C, generator=g))
    h = torch.sigmoid(scale * torch.randn(n, C, generator=g))
    return reachable_bounds(*interval_softmax(m - h, m + h))


def _brute_force_min_entropy(lo, u):
    best = torch.full((lo.size(0),), float("inf"))
    for perm in itertools.permutations(range(lo.size(1))):
        p = lo.clone()
        rem = (1 - p.sum(1)).clamp_min(0)
        for j in perm:
            add = torch.minimum(rem, (u[:, j] - lo[:, j]).clamp_min(0))
            p[:, j] += add
            rem = rem - add
        best = torch.minimum(best, entropy(p))
    return best


def test_interval_softmax_bounds_are_ordered_probabilities():
    a_L = torch.randn(100, 5)
    q_L, q_U = interval_softmax(a_L, a_L + torch.rand(100, 5))
    assert torch.all(q_L <= q_U + 1e-7)
    assert torch.all((q_L > 0) & (q_U < 1))


def test_interval_softmax_degenerates_to_softmax():
    a = torch.randn(10, 3)
    q_L, q_U = interval_softmax(a, a)
    assert torch.allclose(q_L, torch.softmax(a, 1)) and torch.allclose(q_U, torch.softmax(a, 1))


def test_reachable_bounds_tighten_and_stay_consistent():
    a_L = torch.randn(200, 4)
    q_L, q_U = interval_softmax(a_L, a_L + 2 * torch.rand(200, 4))
    lo, u = reachable_bounds(q_L, q_U)
    assert torch.all(lo >= q_L - 1e-7) and torch.all(u <= q_U + 1e-7)
    assert torch.all(lo.sum(1) <= 1 + 1e-6) and torch.all(u.sum(1) >= 1 - 1e-6)
    ln, un = reachable_bounds(q_L.numpy(), q_U.numpy())
    assert np.allclose(ln, lo.numpy()) and np.allclose(un, u.numpy())


def test_max_entropy_is_uniform_when_feasible_and_feasible_otherwise():
    lo, u = torch.zeros(3, 4), torch.ones(3, 4)
    assert torch.allclose(entropy(max_entropy_distribution(lo, u)), torch.full((3,), 2.0))
    lo, u = _intervals(C=3)
    p = max_entropy_distribution(lo, u)
    assert torch.allclose(p.sum(1), torch.ones(len(p)), atol=1e-5)
    assert torch.all(p >= lo - 1e-6) and torch.all(p <= u + 1e-6)


def test_max_entropy_matches_grid_search_for_three_classes():
    lo, u = _intervals(n=30, C=3, seed=3)
    grid = torch.linspace(0, 1, 401)
    p1, p2 = torch.meshgrid(grid, grid, indexing="ij")
    cand = torch.stack([p1.flatten(), p2.flatten(), 1 - p1.flatten() - p2.flatten()], 1)
    for i in range(len(lo)):
        ok = torch.all((cand >= lo[i] - 1e-9) & (cand <= u[i] + 1e-9), 1)
        if ok.sum() == 0:
            continue
        grid_max = entropy(cand[ok]).max()
        assert entropy(max_entropy_distribution(lo[i : i + 1], u[i : i + 1]))[0] >= grid_max - 1e-3


@pytest.mark.parametrize("C", [3, 4, 5])
def test_min_entropy_methods_are_feasible_and_ordered(C):
    lo, u = _intervals(C=C, seed=C)
    exact = _brute_force_min_entropy(lo, u)
    h = {}
    for method in ("greedy", "multistart", "exact", "auto"):
        p = min_entropy_distribution(lo, u, method=method)
        assert torch.allclose(p.sum(1), torch.ones(len(p)), atol=1e-5), method
        assert torch.all(p >= lo - 1e-6) and torch.all(p <= u + 1e-6), method
        h[method] = entropy(p)
    assert torch.allclose(h["exact"], exact, atol=1e-6)
    assert torch.allclose(h["auto"], exact, atol=1e-6)
    assert torch.all(h["multistart"] <= h["greedy"] + 1e-6)
    assert torch.all(h["exact"] <= h["multistart"] + 1e-6)


def test_min_entropy_global_default_switch():
    lo, u = _intervals(C=4, seed=11)
    try:
        set_min_entropy_method("exact")
        assert torch.allclose(
            entropy(min_entropy_distribution(lo, u)), _brute_force_min_entropy(lo, u), atol=1e-6
        )
    finally:
        set_min_entropy_method("greedy")
    with pytest.raises(ValueError):
        set_min_entropy_method("nope")


def test_credal_uncertainties_properties_and_numpy_parity():
    a_L = torch.randn(300, 3)
    q_L, q_U = interval_softmax(a_L, a_L + torch.rand(300, 3))
    TU, AU, EU = credal_uncertainties(q_L, q_U)
    assert torch.all(EU >= -1e-6) and torch.allclose(TU - AU, EU)
    TUn, _, EUn = credal_uncertainties(q_L.numpy(), q_U.numpy())
    assert np.allclose(TUn, TU.cpu().numpy(), atol=1e-5) and np.allclose(EUn, EU.cpu().numpy(), atol=1e-5)


def test_credal_loss_formula():
    torch.manual_seed(0)
    q_L, q_U = interval_softmax(*(lambda a: (a, a + torch.rand(50, 3)))(torch.randn(50, 3)))
    y = torch.nn.functional.one_hot(torch.randint(0, 3, (50,)), 3).float()
    ce_U, ce_L = probs_cross_entropy(q_U, y), probs_cross_entropy(q_L, y)
    k = int(0.3 * 50)
    expected = ce_U.mean() + torch.topk(ce_L, k).values.mean()
    assert torch.allclose(CreNetLoss(0.3)(q_L, q_U, y), expected)
    assert torch.allclose(CreNetLoss(1.0)(q_L, q_U, y), ce_U.mean() + ce_L.mean())


def test_ensemble_of_identical_members_has_no_epistemic_uncertainty():
    p = torch.softmax(torch.randn(20, 4), 1)
    stacked = p.unsqueeze(0).repeat(5, 1, 1)
    _, _, EU = classical_decomposition(stacked)
    assert torch.allclose(EU, torch.zeros_like(EU), atol=1e-5)
    unc = ensemble_uncertainties(stacked)
    assert torch.allclose(unc["EU_credal"], torch.zeros(20), atol=1e-5)
