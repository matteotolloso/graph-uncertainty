# models/GEBMModule.py
import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics import AUROC, Accuracy, F1Score

# ---- Python 3.10 backports ----
import enum, typing as _typing
if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self): return str(self.value)
    enum.StrEnum = StrEnum

try:
    from typing import Self as _Self
except Exception:
    from typing_extensions import Self as _Self
    setattr(_typing, "Self", _Self)
# --------------------------------

from models.VanillaGNN import VanillaGNN

from graph_ebm.graph_uq.gebm import GraphEBMWrapper

class GEBMModule(L.LightningModule):
    """Post-hoc GEBM evaluator using joint embeddings and explicit train/test masks."""

    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.backbone: VanillaGNN = VanillaGNN.load_from_checkpoint(checkpoint_path)
        self.backbone.freeze()

        self.gebm = GraphEBMWrapper()
        self.C = self.backbone.C

        self.auroc_metric = AUROC(task="binary")
        self.acc_metric = Accuracy(task="multiclass", num_classes=self.C)
        self.f1_metric  = F1Score(task="multiclass", num_classes=self.C)

        self._fitted = False
        self._edge_index_cpu = None

    # ---------------------------
    # Backbone utilities
    # ---------------------------
    @torch.no_grad()
    def _compute_logits(self, data):
        return self.backbone(data)

    @staticmethod
    def _onehot_to_indices(y_onehot: torch.Tensor) -> torch.Tensor:
        return torch.argmax(y_onehot, dim=1)

    @torch.no_grad()
    def _get_joint_embeddings(self, data):
        """Concatenate embeddings from all layers of the frozen GNN backbone."""
        all_embeddings = [data.x]
        gnn = self.backbone.gnn_model
        if not hasattr(gnn, 'convs') or not hasattr(gnn, 'act'):
            raise NotImplementedError("Backbone GNN must expose 'convs' and 'act'.")
        x = data.x
        for i in range(gnn.num_layers):
            x = gnn.convs[i](x, data.edge_index)
            if i < gnn.num_layers - 1:
                x = gnn.act(x)
            all_embeddings.append(x)
        return torch.cat(all_embeddings, dim=1)

    # ---------------------------
    # Fit GEBM
    # ---------------------------
    @torch.no_grad()
    def fit_gebm(self, train_batch):
        """Fit GEBM on nodes selected by train_mask."""
        if not hasattr(train_batch, "train_mask"):
            raise ValueError("Training batch must include a 'train_mask'.")
        if not torch.any(train_batch.train_mask):
            raise ValueError("'train_mask' is empty or all False in training batch.")

        device = train_batch.x.device
        mask_train = train_batch.train_mask

        # WITH edges (network effects)
        logits_with_edges = self._compute_logits(train_batch)
        embeddings_with_edges = self._get_joint_embeddings(train_batch)
        y_indices = self._onehot_to_indices(train_batch.y)

        # move to CPU for GEBM
        self.gebm.fit(
            logits_with_edges.detach().cpu(),
            embeddings_with_edges.detach().cpu(),
            train_batch.edge_index.detach().cpu(),
            y_indices.detach().cpu(),
            mask_train.detach().cpu(),
        )
        self._fitted = True
        self._edge_index_cpu = train_batch.edge_index.detach().cpu()

    # ---------------------------
    # Test GEBM
    # ---------------------------
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        """Evaluate GEBM uncertainty on test nodes (requires test_mask)."""
        if not self._fitted:
            raise RuntimeError("Call fit_gebm(train_batch) before testing.")
        if not hasattr(batch, "test_mask"):
            raise ValueError("Test batch must include a 'test_mask'.")
        if not torch.any(batch.test_mask):
            raise ValueError("'test_mask' is empty or all False in test batch.")

        device = batch.x.device

        # Base model forward without edges
        data_no_edges = batch.clone()
        data_no_edges.edge_index = torch.empty((2, 0), dtype=batch.edge_index.dtype, device=device)
        logits_no_edges = self._compute_logits(data_no_edges)
        embeddings_no_edges = self._get_joint_embeddings(data_no_edges)

        uq_all = self.gebm.get_uncertainty(
            logits_unpropagated=logits_no_edges.detach().cpu(),
            embeddings_unpropagated=embeddings_no_edges.detach().cpu(),
            edge_index=self._edge_index_cpu,
        )

        # select test nodes
        test_mask_cpu = batch.test_mask.detach().cpu()
        y_all_cpu = batch.y.detach().cpu()
        y_test = y_all_cpu[test_mask_cpu]
        uq_test = uq_all[test_mask_cpu]
        ood_targets = 1 - y_test.sum(dim=1)

        auroc_gebm = self.auroc_metric(uq_test, ood_targets)
        self.log("test_auroc", auroc_gebm, prog_bar=True)

        # optional classification metrics
        id_mask = (y_test.sum(dim=1) == 1)
        if id_mask.any():
            logits_with_edges = self._compute_logits(batch).detach().cpu()
            logits_test_cpu = logits_with_edges[test_mask_cpu]
            id_logits = logits_test_cpu[id_mask]
            id_labels = torch.argmax(y_test[id_mask], dim=1)
            id_preds = torch.argmax(F.softmax(id_logits, dim=1), dim=1)
            self.log("test_acc", self.acc_metric(id_preds, id_labels))
            self.log("test_f1",  self.f1_metric(id_preds, id_labels))

        return auroc_gebm

    # No training/val phases
    def training_step(self, *args, **kwargs): pass
    def validation_step(self, *args, **kwargs): pass
    def configure_optimizers(self): return None
