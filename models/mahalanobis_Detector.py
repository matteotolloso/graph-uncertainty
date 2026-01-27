import torch
import lightning as L
from torchmetrics import AUROC

from models.VanillaGNN import VanillaGNN


class MahalanobisDetector(L.LightningModule):
    def __init__(self, backbone_ckpt_path: str, noise_magnitude: float = 0.0):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = VanillaGNN.load_from_checkpoint(backbone_ckpt_path)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.noise_magnitude = float(noise_magnitude)
        self.num_classes = self.backbone.C

        self.class_means = None
        self.shared_covariance_inv = None

        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

    # ---------------- feature extractor ----------------
    def _get_features(self, data):
        # pre-softmax logits as features
        with torch.no_grad():
            logits = self.backbone(data)
        return logits

    # ---------------- precompute statistics ----------------
    def precompute_statistics(self, train_data):
        if not hasattr(train_data, "train_mask"):
            raise RuntimeError("train_data must have a 'train_mask' attribute.")

        device = next(self.backbone.parameters()).device
        data = train_data.to(device)

        all_feats = self._get_features(data)       # [N, D]
        train_mask = data.train_mask
        train_feats = all_feats[train_mask]        # [N_train, D]
        train_labels = data.y[train_mask]

        if train_labels.dim() > 1:
            train_labels = torch.argmax(train_labels, dim=1)

        feature_dim = train_feats.size(1)
        class_means = torch.zeros(self.num_classes, feature_dim, device=device)

        for c in range(self.num_classes):
            class_mask = (train_labels == c)
            if class_mask.sum() > 0:
                class_means[c] = train_feats[class_mask].mean(dim=0)

        shared_cov = torch.zeros(feature_dim, feature_dim, device=device)
        total_count = train_feats.size(0)
        for c in range(self.num_classes):
            class_mask = (train_labels == c)
            if class_mask.sum() > 0:
                diff = train_feats[class_mask] - class_means[c]
                shared_cov += diff.t() @ diff

        shared_cov /= max(total_count, 1)
        eps = 1e-6
        shared_cov = shared_cov + eps * torch.eye(feature_dim, device=device)
        shared_cov_inv = torch.linalg.pinv(shared_cov)

        self.class_means = class_means
        self.shared_covariance_inv = shared_cov_inv

        print("Mahalanobis statistics pre-computation complete.")
        print(f"Feature dim: {feature_dim}, num_classes: {self.num_classes}")

    # ---------------- Mahalanobis distances ----------------
    def _mahalanobis_distances(self, feats):
        if self.class_means is None or self.shared_covariance_inv is None:
            raise RuntimeError("You must call precompute_statistics() before forward().")

        diff = feats.unsqueeze(1) - self.class_means.unsqueeze(0)   # [N, C, D]
        left = torch.einsum("ncd,df->ncf", diff, self.shared_covariance_inv)  # [N, C, D]
        mahalanobis_sq = (left * diff).sum(dim=2)  # [N, C]
        return mahalanobis_sq

    # ---------------- input pre-processing (fixed) ----------------
    def _apply_input_preprocessing(self, data):
        if self.noise_magnitude <= 0.0:
            return data

        device = next(self.backbone.parameters()).device
        data_adv = data.clone().to(device)
        x_orig = data_adv.x
        x_perturbed = x_orig.clone().detach().requires_grad_(True)
        data_adv.x = x_perturbed

        with torch.enable_grad():
            logits = self.backbone(data_adv)   # depends on x_perturbed
            feats = logits
            mahalanobis_sq = self._mahalanobis_distances(feats)
            _, closest_classes = mahalanobis_sq.min(dim=1)
            mu_selected = self.class_means[closest_classes]
            diff_sel = feats - mu_selected

            energy = torch.einsum("nd,df,nd->n", diff_sel, self.shared_covariance_inv, diff_sel)
            loss = energy.sum()
            loss.backward()

            grad = x_perturbed.grad

        with torch.no_grad():
            x_adv = x_orig - self.noise_magnitude * torch.sign(grad)
            data_out = data.clone().to(device)
            data_out.x = x_adv

        return data_out


    # ---------------- forward (OOD score) ----------------
    def forward(self, data):
        if self.class_means is None or self.shared_covariance_inv is None:
            raise RuntimeError("Mahalanobis statistics not computed. Call precompute_statistics() first.")

        device = next(self.backbone.parameters()).device
        data = data.to(device)

        # optional input pre-processing
        data_proc = self._apply_input_preprocessing(data)

        # features from possibly preprocessed input
        feats = self._get_features(data_proc)  # [N, D] (no_grad in _get_features is fine here)

        mahalanobis_sq = self._mahalanobis_distances(feats)  # [N, C]
        min_dist, _ = mahalanobis_sq.min(dim=1)              # [N]

        # higher distance => more OOD
        ood_scores = min_dist
        return ood_scores

    # ---------------- Lightning eval hooks ----------------
    def validation_step(self, batch, batch_idx):
        ood_scores_all = self(batch).cpu()
        val_mask = batch.val_mask

        ood_scores_val = ood_scores_all[val_mask]
        y_val = batch.y[val_mask]
        targets = (y_val.sum(dim=1) == 0).long()  # 1 = OOD, 0 = ID

        self.val_auroc.update(ood_scores_val, targets)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        ood_scores_all = self(batch).cpu()
        test_mask = batch.test_mask

        ood_scores_test = ood_scores_all[test_mask]
        y_test = batch.y[test_mask]
        targets = (y_test.sum(dim=1) == 0).long()

        self.test_auroc.update(ood_scores_test, targets)
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return None
