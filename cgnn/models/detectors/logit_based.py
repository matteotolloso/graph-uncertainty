"""Logit-based post-hoc detectors: Energy, ODIN, GNNSafe."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.utils import degree

from cgnn.models.detectors.base import PostHocDetector


class EnergyDetector(PostHocDetector):
    """Energy score (Liu et al. 2020): ``E = -T * logsumexp(logits / T)``; higher = more OOD."""

    def __init__(self, backbone_ckpt_path: str, temperature: float = 1.0):
        super().__init__(backbone_ckpt_path)
        self.save_hyperparameters()
        self.temperature = float(temperature)

    def ood_scores(self, data) -> torch.Tensor:
        with torch.no_grad():
            logits = self.backbone(data)
        return -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)


class ODINDetector(PostHocDetector):
    """ODIN (Liang et al. 2017): temperature scaling + input perturbation; score = -MSP."""

    needs_input_grad = True

    def __init__(self, backbone_ckpt_path: str, temperature: float = 1.0, noise_magnitude: float = 0.0):
        super().__init__(backbone_ckpt_path)
        self.temperature = temperature
        self.noise_magnitude = noise_magnitude
        self.save_hyperparameters(ignore=["backbone"])

    def id_scores(self, data) -> torch.Tensor:
        """Max softmax probability after the ODIN perturbation (higher = more ID)."""
        x_perturbed = data.x.clone().detach().requires_grad_(True)
        data_perturbed = data.clone()
        data_perturbed.x = x_perturbed
        with torch.enable_grad():
            logits = self.backbone(data_perturbed)
            max_score = F.softmax(logits / self.temperature, dim=1).max(dim=1).values
            gradient = torch.autograd.grad(max_score.sum(), x_perturbed, only_inputs=True)[0]

        data_perturbed.x = (data.x + self.noise_magnitude * gradient.sign()).detach()
        with torch.no_grad():
            final_logits = self.backbone(data_perturbed)
            return F.softmax(final_logits / self.temperature, dim=1).max(dim=1).values

    def ood_scores(self, data) -> torch.Tensor:
        return -self.id_scores(data)


class GNNSafeDetector(PostHocDetector):
    """GNNSafe without regularisation (Wu et al. 2023): energy + K-step belief propagation.

    ``e <- alpha * e + (1 - alpha) * mean_{j -> i} e_j`` repeated ``K`` times.
    """

    def __init__(self, backbone_ckpt_path: str, K: int = 2, alpha: float = 0.5):
        super().__init__(backbone_ckpt_path)
        self.save_hyperparameters()
        self.K = K
        self.alpha = alpha

    def _propagate(self, energy: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        from torch_sparse import SparseTensor, matmul

        e = energy.unsqueeze(1)
        N = e.size(0)
        row, col = edge_index
        d = degree(col, N).float()
        value = torch.nan_to_num(
            torch.ones_like(row, dtype=e.dtype) * (1.0 / d[col]), nan=0.0, posinf=0.0, neginf=0.0
        )
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        for _ in range(self.K):
            e = e * self.alpha + matmul(adj, e) * (1.0 - self.alpha)
        return e.squeeze(1)

    def ood_scores(self, data) -> torch.Tensor:
        with torch.no_grad():
            logits = self.backbone(data)
        energy = -torch.logsumexp(logits, dim=1)
        if self.K > 0:
            energy = self._propagate(energy, data.edge_index)
        return energy
