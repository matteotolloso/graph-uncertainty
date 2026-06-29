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
        self.auroc_metric = AUROC(task="binary")

    @staticmethod
    def _calculate_shannon_entropy(probs: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
        """Calculates Shannon entropy for a probability distribution."""
        # Clamp probabilities to avoid log(0)
        probs_clipped = torch.clamp(probs, min=epsilon)
        return -torch.sum(probs * torch.log2(probs_clipped), dim=-1)

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

    def test_step(self, batch, batch_idx):
        q_L, q_U, stacked_probs = self(batch)

        # Isolate the test nodes for all tensors
        q_L_test = q_L[batch.test_mask].detach()
        q_U_test = q_U[batch.test_mask].detach()
        stacked_probs_test = stacked_probs[:, batch.test_mask, :].detach()
        y_test = batch.y[batch.test_mask].detach()

        # --- 1. Credal Uncertainty Calculation (Your original method) ---
        TU, AU, EU = compute_uncertainties(q_L_test, q_U_test) 
        targets = 1 - y_test.sum(axis=1) # 1 for OOD, 0 for ID
        
        print("\n--- Credal Uncertainty (Min/Max Bounds) ---")
        self.log("auroc_EU_credal", self.auroc_metric(EU, targets))
        self.log("auroc_AU_credal", self.auroc_metric(AU, targets))
        self.log("auroc_TU_credal", self.auroc_metric(TU, targets))

        # --- 2. Classical Ensemble Uncertainty Calculation (New method) ---
        # p_tilde: Mean of predictions across the ensemble
        # Shape: [num_test_nodes, num_classes]
        mean_probs = torch.mean(stacked_probs_test, dim=0)

        # TU_classic: Total Uncertainty = Entropy of the mean prediction
        TU_classic = self._calculate_shannon_entropy(mean_probs)

        # AU_classic: Aleatoric Uncertainty = Mean of the individual entropies
        individual_entropies = self._calculate_shannon_entropy(stacked_probs_test)
        AU_classic = torch.mean(individual_entropies, dim=0)

        # EU_classic: Epistemic Uncertainty = Disagreement among models
        EU_classic = TU_classic - AU_classic
        
        print("--- Classical Ensemble Uncertainty (Entropy Decomp.) ---")
        self.log("auroc_EU_classic", self.auroc_metric(EU_classic, targets))
        self.log("auroc_AU_classic", self.auroc_metric(AU_classic, targets))
        self.log("auroc_TU_classic", self.auroc_metric(TU_classic, targets))
        


    # --- Non-Trainable Methods ---
    def training_step(self, batch, batch_idx): pass
    def validation_step(self, batch, batch_idx): pass
    def configure_optimizers(self): return None
