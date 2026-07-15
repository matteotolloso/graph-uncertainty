import torch
import lightning as L
from torch_geometric.nn.models import GCN, GraphSAGE, GAT, GIN, EdgeCNN
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.VanillaGNN import VanillaGNN
from utils.math import compute_uncertainties
from torchmetrics import Accuracy, F1Score, AUROC

class credal_Ensemble(L.LightningModule):
    """
    A post-hoc, inference-only module that estimates two types of uncertainty
    from an ensemble of pre-trained VanillaGNN models:
    1. Credal Uncertainty (q_L, q_U) via min/max bounds.
    2. Classical Ensemble Uncertainty via entropy decomposition.
    """
    def __init__(
        self,
        model_class: L.LightningModule,
        checkpoint_paths: list[str],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.models = nn.ModuleList()
        print(f"Loading {len(checkpoint_paths)} models for the ensemble...")
        for path in checkpoint_paths:
            model = model_class.load_from_checkpoint(path)
            model.freeze() 
            self.models.append(model)
        
        if not self.models:
            raise ValueError("checkpoint_paths cannot be empty.")

        self.C = self.models[0].C
        self.accuracy_metric = Accuracy(task="multiclass", num_classes=self.C)
        self.f1_metric = F1Score(task="multiclass", num_classes=self.C)
    @staticmethod
    def _calculate_shannon_entropy(probs: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
        """Calculates Shannon entropy for a probability distribution."""
        # Clamp probabilities to avoid log(0)
        probs_clipped = torch.clamp(probs, min=epsilon)
        return -torch.sum(probs * torch.log2(probs_clipped), dim=-1)

    def _auroc_score(self, scores, targets):
        return AUROC(task="binary").to(scores.device)(scores, targets)

    def forward(self, data):
        """
        Performs a forward pass to compute credal bounds and stacked probabilities.
        Returns:
            q_L (torch.Tensor): Lower probability bounds.
            q_U (torch.Tensor): Upper probability bounds.
            stacked_probs (torch.Tensor): The raw probabilities from all models.
        """
        all_probs = []
        for model in self.models:
            logits = model(data)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)

        # Shape: [num_models, num_nodes, num_classes]
        stacked_probs = torch.stack(all_probs, dim=0)

        q_L = torch.min(stacked_probs, dim=0).values
        q_U = torch.max(stacked_probs, dim=0).values

        return q_L, q_U, stacked_probs

    def _split_mask(self, batch, split):
        if hasattr(batch, "batch_size") and hasattr(batch, "n_id"):
            mask = torch.zeros(batch.num_nodes, dtype=torch.bool, device=batch.x.device)
            mask[:batch.batch_size] = True
            return mask
        return getattr(batch, f"{split}_mask")

    def _shared_eval_step(self, batch, split: str):
        q_L, q_U, stacked_probs = self(batch)

        split_mask = self._split_mask(batch, split)
        q_L_eval = q_L[split_mask].detach()
        q_U_eval = q_U[split_mask].detach()
        stacked_probs_eval = stacked_probs[:, split_mask, :].detach()
        y_eval = batch.y[split_mask].detach()

        # --- 1. Credal Uncertainty Calculation (Your original method) ---
        TU, AU, EU = compute_uncertainties(q_L_eval, q_U_eval)
        targets = 1 - y_eval.sum(axis=1) # 1 for OOD, 0 for ID

        self.log(f"{split}_auroc_EU_credal", self._auroc_score(EU, targets), prog_bar=(split == "val"))
        self.log(f"{split}_auroc_AU_credal", self._auroc_score(AU, targets))
        self.log(f"{split}_auroc_TU_credal", self._auroc_score(TU, targets))

        # --- 2. Classical Ensemble Uncertainty Calculation (New method) ---
        # p_tilde: Mean of predictions across the ensemble
        # Shape: [num_test_nodes, num_classes]
        mean_probs = torch.mean(stacked_probs_eval, dim=0)

        # TU_classic: Total Uncertainty = Entropy of the mean prediction
        TU_classic = self._calculate_shannon_entropy(mean_probs)

        # AU_classic: Aleatoric Uncertainty = Mean of the individual entropies
        individual_entropies = self._calculate_shannon_entropy(stacked_probs_eval)
        AU_classic = torch.mean(individual_entropies, dim=0)

        # EU_classic: Epistemic Uncertainty = Disagreement among models
        EU_classic = TU_classic - AU_classic

        self.log(f"{split}_auroc_EU_classic", self._auroc_score(EU_classic, targets))
        self.log(f"{split}_auroc_AU_classic", self._auroc_score(AU_classic, targets))
        self.log(f"{split}_auroc_TU_classic", self._auroc_score(TU_classic, targets))

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, "test")


    # --- Non-Trainable Methods ---
    def training_step(self, batch, batch_idx): pass
    def configure_optimizers(self): return None
