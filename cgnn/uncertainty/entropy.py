"""Generalized entropy of interval credal sets (paper Eq. 2).

``TU = max_{p in P} H(p)`` (upper entropy), ``AU = min_{p in P} H(p)`` (lower
entropy), ``EU = TU - AU``. Entropies are in bits (log2).

Solver notes:
- The maximum-entropy solution is exact: ``p_i = clamp(tau, l_i, u_i)`` with
  ``tau`` found by bisection (water filling).
- Minimum entropy over probability intervals is NP-hard in general. Entropy is
  concave, so the minimum is a vertex; every vertex is obtained by pouring the
  free mass ``1 - sum(l)`` into classes in some order. ``min_entropy_method``:

  * ``"greedy"`` (default, used for the paper): one order, by lower
    bound. Suboptimal on 30-80% of nodes (mean EU error ~4-14%).
  * ``"multistart"``: best of C+4 orders (~10x smaller error, C+4x the cost).
  * ``"exact"``: all C! orders; only for C <= 8.
  * ``"auto"``: exact for C <= 6, multistart otherwise.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch

from cgnn.uncertainty.interval import reachable_bounds

MIN_ENTROPY_METHODS = ("greedy", "multistart", "exact", "auto")
_DEFAULT_MIN_ENTROPY_METHOD = "greedy"


def set_min_entropy_method(name: str) -> None:
    """Process-wide AU solver used when ``min_entropy_method`` is not given (config key)."""
    global _DEFAULT_MIN_ENTROPY_METHOD
    if name not in MIN_ENTROPY_METHODS:
        raise ValueError(f"min_entropy_method must be one of {MIN_ENTROPY_METHODS}, got {name!r}")
    _DEFAULT_MIN_ENTROPY_METHOD = name


def get_min_entropy_method() -> str:
    return _DEFAULT_MIN_ENTROPY_METHOD


def _to_tensor(x, device=None) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _bounds_tolerance(num_classes: int, dtype: torch.dtype) -> float:
    """Slack allowed in the credal-set sanity checks, for rounding only.

    Reachable bounds add/subtract C probabilities, so when an interval collapses
    to a point (``q_L == q_U``, e.g. saturated half-width sigmoids) their sums
    miss 1 by up to ~C^2 ulp (~1e-5 for C=41 in float32). The tolerance only
    absorbs this rounding (it never changes values).
    At least float32 precision: the heads output float32, and casting to float64 keeps their rounding.
    """
    eps = torch.finfo(torch.float32).eps
    if dtype.is_floating_point:
        eps = max(eps, torch.finfo(dtype).eps)
    return min(1e-3, max(1e-6, num_classes * num_classes * eps))  # cap: half precision must not disable them


def entropy(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy in bits along the last dimension."""
    q_clipped = q.clamp_min(eps)
    return -torch.sum(q * torch.log2(q_clipped), dim=-1)


def max_entropy_distribution(
    lower_bound: torch.Tensor, upper_bound: torch.Tensor, num_iter: int = 64
) -> torch.Tensor:
    """Exact maximum-entropy member of ``{l <= q <= u, sum q = 1}`` (batched bisection)."""
    lo = lower_bound.min(dim=1, keepdim=True).values
    hi = upper_bound.max(dim=1, keepdim=True).values

    for _ in range(num_iter):
        mid = (lo + hi) / 2
        total = torch.clamp(mid, lower_bound, upper_bound).sum(dim=1, keepdim=True)
        lo = torch.where(total < 1.0, mid, lo)
        hi = torch.where(total >= 1.0, mid, hi)

    tau = (lo + hi) / 2
    return torch.clamp(tau, lower_bound, upper_bound)


def _fill_in_order(lower_bound: torch.Tensor, upper_bound: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Vertex reached by pouring the free mass into classes following ``order`` ``[N, C]``."""
    p = lower_bound.clone()
    remaining = (1.0 - p.sum(dim=1, keepdim=True)).clamp_min(0.0)
    capacity = (upper_bound - lower_bound).clamp_min(0.0)
    rows = torch.arange(lower_bound.size(0), device=lower_bound.device)

    for j in range(lower_bound.size(1)):
        cols = order[:, j]
        add = torch.minimum(remaining.squeeze(1), capacity[rows, cols])
        p[rows, cols] = p[rows, cols] + add
        remaining = (remaining - add.unsqueeze(1)).clamp_min(0.0)

    return p


def _candidate_orders(
    lower_bound: torch.Tensor, upper_bound: torch.Tensor, method: str
) -> list[torch.Tensor]:
    n, C = lower_bound.shape
    if method == "exact":
        if C > 8:
            raise ValueError(f"exact min-entropy enumerates C! orders; C={C} is too large (use 'multistart')")
        return [
            torch.tensor(perm, device=lower_bound.device).expand(n, C)
            for perm in itertools.permutations(range(C))
        ]
    # multistart: greedy order + a few natural orders + "class c first, rest by upper bound"
    by_upper = torch.argsort(upper_bound, dim=1, descending=True)
    orders = [
        torch.argsort(lower_bound + 1e-6 * upper_bound, dim=1, descending=True),
        by_upper,
        torch.argsort(lower_bound + upper_bound, dim=1, descending=True),
        torch.argsort(upper_bound - lower_bound, dim=1, descending=True),
    ]
    for c in range(C):
        first = torch.full((n, 1), c, device=lower_bound.device, dtype=by_upper.dtype)
        rest = by_upper[by_upper != c].view(n, C - 1)
        orders.append(torch.cat([first, rest], dim=1))
    return orders


def min_entropy_distribution(
    lower_bound: torch.Tensor, upper_bound: torch.Tensor, method: str | None = None
) -> torch.Tensor:
    """Low-entropy vertex of ``{l <= q <= u, sum q = 1}``; see module docstring for ``method``."""
    method = method or _DEFAULT_MIN_ENTROPY_METHOD
    if method not in MIN_ENTROPY_METHODS:
        raise ValueError(f"min_entropy method must be one of {MIN_ENTROPY_METHODS}, got {method!r}")
    if method == "auto":
        method = "exact" if lower_bound.size(1) <= 6 else "multistart"

    if method == "greedy":
        order = torch.argsort(lower_bound + 1e-6 * upper_bound, dim=1, descending=True)
        return _fill_in_order(lower_bound, upper_bound, order)

    best_p, best_h = None, None
    for order in _candidate_orders(lower_bound, upper_bound, method):
        p = _fill_in_order(lower_bound, upper_bound, order)
        h = entropy(p)
        if best_p is None:
            best_p, best_h = p, h
        else:
            better = h < best_h
            best_p = torch.where(better.unsqueeze(1), p, best_p)
            best_h = torch.minimum(h, best_h)
    return best_p


def interval_entropy_bounds(lower_bound, upper_bound, min_entropy_method: str | None = None):
    """Return ``(min_entropy, max_entropy)`` for batched probability intervals.

    Accepts tensors (result stays on the input device) or ndarrays (result is
    returned as ndarrays).
    """
    input_was_numpy = isinstance(lower_bound, np.ndarray) or isinstance(upper_bound, np.ndarray)

    lower_bound = _to_tensor(lower_bound)
    upper_bound = _to_tensor(upper_bound, device=lower_bound.device).to(
        device=lower_bound.device, dtype=lower_bound.dtype
    )

    assert lower_bound.shape == upper_bound.shape, "Lower and upper bounds must have the same shape"
    assert len(lower_bound.shape) == 2, "Lower and upper bounds must be 2D arrays"
    tol = _bounds_tolerance(lower_bound.size(1), lower_bound.dtype)
    assert torch.all(lower_bound <= upper_bound + tol), (
        "Lower bounds must be less than or equal to upper bounds"
    )
    assert torch.all(torch.sum(lower_bound, dim=1) <= 1.0 + tol), (
        "Sum of lower bounds for a node cannot exceed 1"
    )
    assert torch.all(torch.sum(upper_bound, dim=1) >= 1.0 - tol), (
        "Sum of upper bounds for a node must be at least 1"
    )

    min_entropy_values = entropy(
        min_entropy_distribution(lower_bound, upper_bound, method=min_entropy_method)
    )
    max_entropy_values = entropy(max_entropy_distribution(lower_bound, upper_bound))

    if input_was_numpy:
        return min_entropy_values.detach().cpu().numpy(), max_entropy_values.detach().cpu().numpy()
    return min_entropy_values, max_entropy_values


def credal_uncertainties(q_L, q_U, min_entropy_method: str | None = None):
    """Total, aleatoric and epistemic uncertainty of interval credal sets.

    Args:
        q_L, q_U: ``[N, C]`` lower/upper probabilities (tensors or ndarrays).
        min_entropy_method: AU solver (default: the process-wide setting, "greedy").

    Returns:
        ``(TU, AU, EU)`` with ``TU = max H``, ``AU = min H``, ``EU = TU - AU``,
        computed on the reachable bounds.
    """
    assert isinstance(q_L, (np.ndarray, torch.Tensor)) and isinstance(q_U, (np.ndarray, torch.Tensor)), (
        f"q_L and q_U must be arrays or tensors, but got {type(q_L)} and {type(q_U)}"
    )
    assert q_L.shape == q_U.shape, f"Shapes of q_L and q_U must match, but got {q_L.shape} and {q_U.shape}"
    assert len(q_L.shape) == 2, f"q_L and q_U must be 2D, but got shapes {q_L.shape} and {q_U.shape}"

    if isinstance(q_L, torch.Tensor):
        q_L = q_L.detach()
        q_U = q_U.detach()

    q_L_star, q_U_star = reachable_bounds(q_L, q_U)

    if isinstance(q_L_star, torch.Tensor):
        tol = _bounds_tolerance(q_L_star.shape[1], q_L_star.dtype)
        assert torch.all(q_L_star <= q_U_star + tol), (
            "Lower bounds must be less than or equal to upper bounds"
        )
    else:
        tol = _bounds_tolerance(q_L_star.shape[1], torch.float32)  # ndarrays are converted to float32 below
        assert np.all(q_L_star <= q_U_star + tol), "Lower bounds must be less than or equal to upper bounds"

    AU, TU = interval_entropy_bounds(q_L_star, q_U_star, min_entropy_method=min_entropy_method)
    EU = TU - AU
    return TU, AU, EU
