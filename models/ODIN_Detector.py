import torch
import lightning as L
import torch.nn.functional as F
from torchmetrics import AUROC, Accuracy, F1Score
from models.VanillaGNN import VanillaGNN

class ODINDetector(L.LightningModule):
    def __init__(self, backbone_ckpt_path, temperature=1.0, noise_magnitude=0.0):
        super().__init__()

        # Load and freeze trained backbone
        self.backbone = VanillaGNN.load_from_checkpoint(backbone_ckpt_path)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # OOD method hyperparameters
        self.temperature = temperature
        self.noise_magnitude = noise_magnitude

        # Metrics
        self.auroc = AUROC(task="binary")

        # Save hyperparameters (wandb sweep will control these)
        self.save_hyperparameters(ignore=['backbone'])

    def forward(self, data): # TODO check
        # Clone input and enable gradients on x
        x_perturbed = data.x.clone().detach().requires_grad_(True)
        data_perturbed = data.clone()
        data_perturbed.x = x_perturbed

        # Get logits from the backbone
        logits = self.backbone(data_perturbed)  # shape: [num_nodes, C]

        # Apply temperature scaling
        logits_temp = logits / self.temperature
        probs = F.softmax(logits_temp, dim=1)

        # Get max probability (for loss-like target)
        max_score, _ = torch.max(probs, dim=1) 

        # Create fake "loss" to backprop the max confidence
        # We want to perturb x in the direction that increases softmax confidence
        # So we sum over the max score to simulate maximizing it
        score_sum = torch.sum(max_score)
        score_sum.backward()

        # Compute the perturbation: sign of gradient * noise magnitude
        gradient = x_perturbed.grad.data
        perturbation = self.noise_magnitude * gradient.sign()

        # Add perturbation to x
        x_final = data.x + perturbation
        data_perturbed.x = x_final.detach()  # no gradient needed now

        # Final forward with perturbed x
        final_logits = self.backbone(data_perturbed)
        final_logits_temp = final_logits / self.temperature
        final_probs = F.softmax(final_logits_temp, dim=1)

        # Return max softmax scores as OOD confidence
        ood_scores = torch.max(final_probs, dim=1).values  # higher = more in-distribution

        return ood_scores

    def _split_mask(self, batch, split):
        if hasattr(batch, "batch_size") and hasattr(batch, "n_id"):
            mask = torch.zeros(batch.num_nodes, dtype=torch.bool, device=batch.x.device)
            mask[:batch.batch_size] = True
            return mask
        return getattr(batch, f"{split}_mask")

    def _eval_step(self, batch, split: str):
        id_scores_all = self(batch)
        split_mask = self._split_mask(batch, split)
        id_scores = id_scores_all[split_mask]
        y_split = batch.y[split_mask]
        targets = (y_split.sum(dim=1) == 0).long()

        # AUROC expects higher scores for the positive class; target 1 is OOD.
        ood_scores = -id_scores
        auroc = AUROC(task="binary").to(ood_scores.device)(ood_scores, targets)
        self.log(f"{split}_auroc", auroc, prog_bar=True)
        return auroc

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, "test")

    def configure_optimizers(self):
        return None
