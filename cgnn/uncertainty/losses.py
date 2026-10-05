"""Credal DRO objective (paper Eqs. 9-10)."""

from __future__ import annotations

import torch


def probs_cross_entropy(
    predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-9
) -> torch.Tensor:
    """Per-sample cross-entropy between *probabilities* and one-hot targets.

    Unlike ``F.cross_entropy`` this takes probabilities (it is applied to the
    unnormalised interval bounds ``q_U``/``q_L``). Returns shape ``[N]``.
    """
    assert predictions.shape == targets.shape, (
        f"Shapes of predictions and targets must match, but got {predictions.shape} and {targets.shape}"
    )
    return -torch.sum(targets * torch.log(predictions + epsilon), dim=-1)


class CreNetLoss(torch.nn.Module):
    """``L_credal = mean_n CE(q_U_n, y_n) + mean_{n in H_delta} CE(q_L_n, y_n)``.

    ``H_delta`` holds the ``max(1, int(delta * N))`` samples with the largest
    lower-bound cross-entropy (the DRO "hard" set).
    """

    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta

    def forward(self, q_L: torch.Tensor, q_U: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """``q_L``/``q_U``: ``[N, C]`` bounds; ``target``: ``[N, C]`` one-hot."""
        assert q_L.dim() == 2 and q_U.dim() == 2, f"q_L and q_U must be 2D, got {q_L.shape} and {q_U.shape}"
        assert q_L.shape == q_U.shape, (
            f"q_L and q_U must have the same shape, got {q_L.shape} and {q_U.shape}"
        )
        assert target.shape == q_L.shape, f"target must match q_L, got {target.shape} and {q_L.shape}"

        num_nodes = q_L.shape[0]
        vanilla_component = probs_cross_entropy(q_U, target).mean()
        dro_component = probs_cross_entropy(q_L, target)
        top_values, _ = torch.topk(input=dro_component, k=max(1, int(self.delta * num_nodes)))
        return vanilla_component + top_values.mean()
