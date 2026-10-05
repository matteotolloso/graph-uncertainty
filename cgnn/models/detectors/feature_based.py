"""Feature-space post-hoc detectors: Mahalanobis, kNN, kNN on the joint latent (JLDE)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from cgnn.models.backbones import joint_representation
from cgnn.models.detectors.base import PostHocDetector, shallow_to


class MahalanobisDetector(PostHocDetector):
    """Lee et al. (2018) on the backbone logits: class means + shared covariance
    from training nodes; score = min squared Mahalanobis distance (optionally
    after the input pre-processing step with ``noise_magnitude``)."""

    needs_input_grad = True

    def __init__(self, backbone_ckpt_path: str, noise_magnitude: float = 0.0):
        super().__init__(backbone_ckpt_path)
        self.save_hyperparameters()
        self.noise_magnitude = float(noise_magnitude)
        self.num_classes = self.backbone.C
        self.class_means = None
        self.shared_covariance_inv = None

    def _get_features(self, data) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(data)

    def prepare(self, train_data) -> None:
        device = self.backbone_device
        data = shallow_to(train_data, device)
        feats = self._get_features(data)[data.train_mask]
        labels = data.y[data.train_mask]
        if labels.dim() > 1:
            labels = torch.argmax(labels, dim=1)

        dim = feats.size(1)
        class_means = torch.zeros(self.num_classes, dim, device=device)
        for c in range(self.num_classes):
            if (labels == c).sum() > 0:
                class_means[c] = feats[labels == c].mean(dim=0)
        cov = torch.zeros(dim, dim, device=device)
        for c in range(self.num_classes):
            if (labels == c).sum() > 0:
                diff = feats[labels == c] - class_means[c]
                cov += diff.t() @ diff
        cov /= max(feats.size(0), 1)
        cov = cov + 1e-6 * torch.eye(dim, device=device)
        self.class_means = class_means
        self.shared_covariance_inv = torch.linalg.pinv(cov)

    def _mahalanobis_sq(self, feats: torch.Tensor) -> torch.Tensor:
        if self.class_means is None:
            raise RuntimeError("Call prepare(train_data) before scoring.")
        diff = feats.unsqueeze(1) - self.class_means.unsqueeze(0)  # [N, C, D]
        left = torch.einsum("ncd,df->ncf", diff, self.shared_covariance_inv)
        return (left * diff).sum(dim=2)  # [N, C]

    def _preprocess(self, data):
        if self.noise_magnitude <= 0.0:
            return data
        data_adv = data.clone().to(self.backbone_device)
        x_orig = data_adv.x
        x_perturbed = x_orig.clone().detach().requires_grad_(True)
        data_adv.x = x_perturbed
        with torch.enable_grad():
            feats = self.backbone(data_adv)
            closest = self._mahalanobis_sq(feats).min(dim=1).indices
            diff = feats - self.class_means[closest]
            energy = torch.einsum("nd,df,nd->n", diff, self.shared_covariance_inv, diff)
            energy.sum().backward()
        with torch.no_grad():
            data_out = data.clone().to(self.backbone_device)
            data_out.x = x_orig - self.noise_magnitude * torch.sign(x_perturbed.grad)
        return data_out

    def ood_scores(self, data) -> torch.Tensor:
        data = data.to(self.backbone_device)
        feats = self._get_features(self._preprocess(data))
        return self._mahalanobis_sq(feats).min(dim=1).values


class _FaissKNN(PostHocDetector):
    """Deep-kNN detector (Sun et al. 2022): score = squared L2 distance to the
    k-th nearest *training* embedding (L2-normalised, FAISS flat index)."""

    def __init__(self, backbone_ckpt_path: str, k: int = 50):
        super().__init__(backbone_ckpt_path)
        self.k = int(k)
        self.faiss_index = None

    def embed(self, data) -> torch.Tensor:
        raise NotImplementedError

    def prepare(self, train_data) -> None:
        import faiss

        data = shallow_to(train_data, self.backbone_device)
        with torch.no_grad():
            train_emb = F.normalize(self.embed(data)[data.train_mask], p=2, dim=1)
        self.faiss_index = faiss.IndexFlatL2(train_emb.size(1))
        self.faiss_index.add(train_emb.detach().cpu().numpy().astype(np.float32))

    def ood_scores(self, data) -> torch.Tensor:
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not built. Call prepare(train_data) first.")
        with torch.no_grad():
            emb = F.normalize(self.embed(data.to(self.backbone_device)), p=2, dim=1)
        distances, _ = self.faiss_index.search(emb.detach().cpu().numpy().astype(np.float32), self.k)
        return torch.from_numpy(distances[:, -1]).float()


class KNNDetector(_FaissKNN):
    """kNN on the second-to-last backbone layer (after activation)."""

    def __init__(self, backbone_ckpt_path: str, k: int = 50):
        super().__init__(backbone_ckpt_path, k)
        self.save_hyperparameters()

    def embed(self, data) -> torch.Tensor:
        gnn = self.backbone.gnn_model
        if gnn.num_layers <= 1:
            return self.backbone(data)
        x = data.x
        for i in range(gnn.num_layers - 1):
            x = gnn.act(gnn.convs[i](x, data.edge_index))
        return x


class KNNJointDetector(_FaissKNN):
    """JLDE ablation (Fuchsgruber et al. 2025): kNN density on the joint latent
    ``[x || act(z^1) || ... || z^L]`` of the frozen backbone."""

    def __init__(self, backbone_ckpt_path: str, k: int = 50):
        super().__init__(backbone_ckpt_path, k)
        self.save_hyperparameters()

    def embed(self, data) -> torch.Tensor:
        return joint_representation(
            self.backbone.gnn_model, data.x, data.edge_index, act_on_last=False, include_input=True
        )
