"""Uncertainty from a set of predictive distributions (ensembles, reservoirs)."""

from __future__ import annotations

import torch

from cgnn.uncertainty.entropy import credal_uncertainties


def shannon_entropy(probs: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """Shannon entropy in bits along the last dimension."""
    probs_clipped = torch.clamp(probs, min=epsilon)
    return -torch.sum(probs * torch.log2(probs_clipped), dim=-1)


def credal_hull_bounds(stacked_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Interval bounds of the convex hull of ``M`` predictions.

    ``stacked_probs``: ``[M, N, C]``. Returns ``(q_L, q_U)`` as class-wise min/max
    ("CGNN by Ensemble" / credal wrapper, Wang et al. 2024b).
    """
    return torch.min(stacked_probs, dim=0).values, torch.max(stacked_probs, dim=0).values


def classical_decomposition(stacked_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bayesian-model-averaging decomposition: ``TU = H(mean)``, ``AU = mean(H)``, ``EU = TU - AU``."""
    mean_probs = torch.mean(stacked_probs, dim=0)
    TU = shannon_entropy(mean_probs)
    AU = torch.mean(shannon_entropy(stacked_probs), dim=0)
    return TU, AU, TU - AU


def ensemble_uncertainties(stacked_probs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Credal (hull) and classical uncertainties, keyed ``{EU,AU,TU}_{credal,classic}``."""
    q_L, q_U = credal_hull_bounds(stacked_probs)
    TU_c, AU_c, EU_c = credal_uncertainties(q_L, q_U)
    TU, AU, EU = classical_decomposition(stacked_probs)
    return {
        "TU_credal": TU_c,
        "AU_credal": AU_c,
        "EU_credal": EU_c,
        "TU_classic": TU,
        "AU_classic": AU,
        "EU_classic": EU,
    }
