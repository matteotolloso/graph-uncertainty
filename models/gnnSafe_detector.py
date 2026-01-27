# models/gnnSafe_detector.py

import torch
import lightning as L
from torchmetrics import AUROC
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree

from models.VanillaGNN import VanillaGNN


class GNNSafeDetector(L.LightningModule):
    """
    Post-hoc GNNSafe detector on a frozen VanillaGNN backbone.

    Theoretical core follows the official GNNSafe implementation:

      - Energy score (single-label multi-class):
          E(x) = - logsumexp(logits)
      - Energy belief propagation:
          e^{k} = alpha * e^{k-1} + (1 - alpha) * D^{-1} A e^{k-1}
      - OOD score = propagated energy (higher -> more likely OOD)

    We do not retrain the backbone or add the energy regularization term.
    """

    def __init__(self, backbone_ckpt_path: str, K: int = 2, alpha: float = 0.5):
        super().__init__()
        self.save_hyperparameters()

        # --- Load and freeze the pre-trained backbone (VanillaGNN) ---
        self.backbone = VanillaGNN.load_from_checkpoint(backbone_ckpt_path)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Propagation hyperparameters (match GNNSafe propagation)
        self.K = K
        self.alpha = alpha

        # Metrics
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

    def _propagate(self, energy_scores: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Energy belief propagation as in the official GNNSafe code.

        Args:
            energy_scores: Tensor of shape [N], initial energy per node.
            edge_index: LongTensor of shape [2, E] (PyG convention).

        Returns:
            Tensor of shape [N] with propagated energy scores.
        """
        e = energy_scores.unsqueeze(1)  # [N, 1]
        N = e.size(0)

        row, col = edge_index  # edge_index[0]: src, edge_index[1]: dst

        # Degree on destination nodes
        d = degree(col, N).float()

        # 1 / d_j for each edge (i -> j); same as official:
        #   d_norm = 1. / d[col]
        d_norm = 1.0 / d[col]
        value = torch.ones_like(row, dtype=e.dtype) * d_norm
        # Handle potential NaNs / infs (isolated nodes, etc.)
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

        # Build normalized adjacency: entries 1/d_j for edge i -> j
        # (note the swap row=col, col=row as in official gnnsafe.py)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))

        # Iterative propagation
        for _ in range(self.K):
            e = e * self.alpha + matmul(adj, e) * (1.0 - self.alpha)

        return e.squeeze(1)  # [N]

    def forward(self, data):
        """
        Compute GNNSafe OOD scores for all nodes.

        Steps:
          1. Get logits from the frozen backbone.
          2. Compute energy: E = -logsumexp(logits).
          3. Apply K-step energy belief propagation.
          4. Return propagated energy as OOD score (higher -> more OOD).
        """
        with torch.no_grad():
            logits = self.backbone(data)  # [N, C]

        # Energy (single-label multi-class) with T=1
        energy = -torch.logsumexp(logits, dim=1)  # [N]

        # Energy belief propagation (GNNSafe)
        if self.K > 0:
            energy = self._propagate(energy, data.edge_index)

        # OOD score: higher energy => more likely OOD
        ood_scores = energy  # [N]

        return ood_scores

    def validation_step(self, batch, batch_idx):
        """
        OOD detection AUROC on validation nodes.

        Targets:
          - y is one-hot / multi-hot as in your framework.
          - OOD nodes: y.sum(dim=1) == 0 -> label 1
          - ID nodes:  y.sum(dim=1) == 1 -> label 0
        """
        ood_scores_all = self(batch)  # [N]
        val_mask = batch.val_mask

        ood_scores_val = ood_scores_all[val_mask]
        y_val = batch.y[val_mask]

        targets = (y_val.sum(dim=1) == 0).long()  # 1 = OOD, 0 = ID

        self.val_auroc.update(ood_scores_val, targets)
        self.log(
            "val_auroc",
            self.val_auroc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        """
        OOD detection AUROC on test nodes.
        """
        ood_scores_all = self(batch)  # [N]
        test_mask = batch.test_mask

        ood_scores_test = ood_scores_all[test_mask]
        y_test = batch.y[test_mask]

        targets = (y_test.sum(dim=1) == 0).long()  # 1 = OOD, 0 = ID

        self.test_auroc.update(ood_scores_test, targets)
        self.log(
            "test_auroc",
            self.test_auroc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        # No training; purely post-hoc detector
        return None
