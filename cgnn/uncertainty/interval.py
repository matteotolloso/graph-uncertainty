"""Probability intervals: Interval SoftMax and reachability (paper Eq. 4).

Conventions (used everywhere in the package):
- ``a_L, a_U``: lower/upper interval *logits*, shape ``[N, C]``.
- ``q_L, q_U``: lower/upper class *probabilities*, shape ``[N, C]``.
- The credal set is ``{q : q_L <= q <= q_U, sum(q) = 1}``.
"""

from __future__ import annotations

import numpy as np
import torch


def interval_softmax(a_L: torch.Tensor, a_U: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Interval SoftMax (Wang et al. 2025), paper Eq. (4).

    ``q_L_i = exp(a_L_i) / (exp(a_L_i) + sum_{k != i} exp(mid_k))`` and the same
    for ``q_U`` with ``a_U``, where ``mid = (a_L + a_U) / 2``.
    """
    mid_points = (a_U + a_L) / 2

    exp_a_L = torch.exp(a_L)
    exp_a_U = torch.exp(a_U)
    exp_mid_points = torch.exp(mid_points)

    sum_exp_mid_points = torch.sum(exp_mid_points, dim=-1, keepdim=True)

    denom_q_L = exp_a_L + (sum_exp_mid_points - exp_mid_points)
    denom_q_U = exp_a_U + (sum_exp_mid_points - exp_mid_points)

    q_L = exp_a_L / denom_q_L
    q_U = exp_a_U / denom_q_U
    return q_L, q_U


def reachable_bounds(q_L, q_U):
    """Tighten interval bounds to the values actually reachable in the credal set.

    ``q_L*_i = max(q_L_i, 1 - sum_{k != i} q_U_k)`` and
    ``q_U*_i = min(q_U_i, 1 - sum_{k != i} q_L_k)``. Works on tensors and ndarrays.
    """
    if isinstance(q_L, torch.Tensor):
        sum_q_L = torch.sum(q_L, dim=-1, keepdim=True)
        sum_q_U = torch.sum(q_U, dim=-1, keepdim=True)
        q_L_star = torch.maximum(q_L, 1 - (sum_q_U - q_U))
        q_U_star = torch.minimum(q_U, 1 - (sum_q_L - q_L))
        return q_L_star, q_U_star

    sum_q_L = np.sum(q_L, axis=-1, keepdims=True)
    sum_q_U = np.sum(q_U, axis=-1, keepdims=True)
    q_L_star = np.maximum(q_L, 1 - (sum_q_U - q_U))
    q_U_star = np.minimum(q_U, 1 - (sum_q_L - q_L))
    return q_L_star, q_U_star
