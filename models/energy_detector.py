# models/energy_detector.py

import torch
import lightning as L
from torchmetrics import AUROC
from models.VanillaGNN import VanillaGNN # Assuming your VanillaGG is here

class EnergyDetector(L.LightningModule):
    def __init__(self, backbone_ckpt_path: str):
        super().__init__()
        self.save_hyperparameters()

        # Load and freeze the pre-trained backbone
        self.backbone = VanillaGNN.load_from_checkpoint(backbone_ckpt_path)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
    def _split_mask(self, batch, split):
        if hasattr(batch, "batch_size") and hasattr(batch, "n_id"):
            mask = torch.zeros(batch.num_nodes, dtype=torch.bool, device=batch.x.device)
            mask[:batch.batch_size] = True
            return mask
        return getattr(batch, f"{split}_mask")

    def _auroc_score(self, scores, targets):
        return AUROC(task="binary").to(scores.device)(scores, targets)

    def forward(self, data):
        """
        Calculates the energy score for all nodes in the data object.
        This corresponds to Equation (4) in Liu et al. (2020).
        The OOD score is the energy score, where larger values are more OOD.
        """
        with torch.no_grad():
            # The energy score is based on the pre-softmax logits
            logits = self.backbone(data)

        # Energy score E(x) = -log(sum_c(exp(logit_c(x))))
        # A higher energy score means more likely to be OOD.
        energy_scores = -torch.logsumexp(logits, dim=1)

        # Higher energy means more OOD; AUROC targets use 1 = OOD.
        ood_scores = energy_scores

        return ood_scores

    def validation_step(self, batch, batch_idx):
        """
        Evaluates the detector on the validation set.
        This is used by the wandb sweep to find the best hyperparameters for other models,
        but for Energy, it's just for logging as there's nothing to tune.
        """
        ood_scores_all = self(batch).cpu()
        val_mask = self._split_mask(batch, "val").cpu()
        
        ood_scores_val = ood_scores_all[val_mask]
        y_val = batch.y.cpu()[val_mask]

        # OOD labels are 1, ID labels are 0
        targets = (y_val.sum(dim=1) == 0).long()

        self.log("val_auroc", self._auroc_score(ood_scores_val, targets), prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Evaluates the final model on the test set.
        """
        ood_scores_all = self(batch).cpu()
        test_mask = self._split_mask(batch, "test").cpu()

        ood_scores_test = ood_scores_all[test_mask]
        y_test = batch.y.cpu()[test_mask]
        
        targets = (y_test.sum(dim=1) == 0).long()

        self.log("test_auroc", self._auroc_score(ood_scores_test, targets), prog_bar=True)

    def configure_optimizers(self): return None
