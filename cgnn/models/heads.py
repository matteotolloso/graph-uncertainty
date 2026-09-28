"""Output heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cgnn.uncertainty.interval import interval_softmax

_ACTIVATIONS = {
    "sigmoid": torch.sigmoid,
    "softplus": F.softplus,
    "identity": lambda t: t,
}


class CredalLayer(nn.Module):
    """Credal head (paper Eq. 3-4): embedding -> interval probabilities ``(q_L, q_U)``.

    ``mh = MLP(z)`` gives midpoints ``m = g(.)`` and half-widths ``h = g'(.)``;
    ``[a_L, a_U] = [m - h - margin, m + h + margin]`` then Interval SoftMax.

    ``mid_activation``/``half_activation`` default to ``"sigmoid"`` (the version
    used for the paper). With sigmoids the interval logits live in ``(-1, 2)``,
    which caps ``q_U`` at ``e^2 / (e^2 + C - 1)`` (~0.20 for C=30); see
    docs/known_issues.md. ``"identity"``/``"softplus"`` lift this cap.
    """

    def __init__(
        self,
        input_dim: int,
        C: int,
        margin: float = 0.0,
        hidden_dim: int | None = None,
        mid_activation: str = "sigmoid",
        half_activation: str = "sigmoid",
    ):
        super().__init__()
        if half_activation == "identity":
            raise ValueError("half_activation must be non-negative (sigmoid or softplus)")
        self.C = C
        self.margin = margin
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim
        self.mid_activation = mid_activation
        self.half_activation = half_activation

        self.mh_layer = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(in_features=self.hidden_dim, out_features=2 * C),
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert z.dim() == 2, "Input must be a 2D tensor"
        assert z.shape[1] == self.input_dim, f"Input shape must be (num_nodes, {self.input_dim})"

        mh = self.mh_layer(z)
        m = _ACTIVATIONS[self.mid_activation](mh[:, : self.C])  # interval midpoint
        h = _ACTIVATIONS[self.half_activation](mh[:, self.C :])  # half-width >= 0

        a_L = m - h - self.margin
        a_U = m + h + self.margin
        assert torch.all(a_L <= a_U), "Lower bounds must be less than or equal to upper bounds"
        return interval_softmax(a_L, a_U)


def mlp_classifier(input_dim: int, num_classes: int) -> nn.Sequential:
    """Point-prediction head of the dual-head CGNN (Linear-Sigmoid-Linear)."""
    return nn.Sequential(nn.Linear(input_dim, input_dim), nn.Sigmoid(), nn.Linear(input_dim, num_classes))
