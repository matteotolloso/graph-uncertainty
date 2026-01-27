# trainers/mahalanobis_tester.py

import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import os
import sys
import torch

# --- Add project root to path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mahalanobis_Detector import MahalanobisDetector
from dataset_loader.dataset_loader import dataset_loader
from utils.model_manager import find_best_checkpoints


def mahalanobis_test(project_name, dataset_name, save_path, **kwargs):
    """
    Post-hoc Mahalanobis OOD detector (Lee et al. 2018) on top of a frozen VanillaGNN.
    We:
      1) Load the best vanilla backbone
      2) Pre-compute class means and shared covariance on training nodes
      3) Optionally use input pre-processing (noise_magnitude > 0)
      4) Evaluate AUROC for OOD detection on val and test masks
    """
    wandb.init(project=project_name, job_type="test_mahalanobis")
    config = wandb.config

    # --- 1. Find the Best Backbone Checkpoint ---
    vanilla_model_dir = os.path.join(save_path, f"vanilla_{dataset_name}")
    backbone_ckpt_path = find_best_checkpoints(dataset_name, 1)[0]
    if not backbone_ckpt_path:
        raise FileNotFoundError(f"No backbone checkpoint found in {vanilla_model_dir}")
    print(f"Using backbone checkpoint: {backbone_ckpt_path}")

    # --- 2. Instantiate Detector with Hyperparameters ---
    mahalanobis_model = MahalanobisDetector(
        backbone_ckpt_path=backbone_ckpt_path,
        noise_magnitude=config.get("noise_magnitude", 0.0),
    )

    # Decide device manually for precomputation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mahalanobis_model.to(device)

    # --- 3. Set up Trainer ---
    wandb_logger = WandbLogger(project=project_name)
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        logger=wandb_logger,
        enable_checkpointing=False,
        log_every_n_steps=1,
        inference_mode=False,   # <-- IMPORTANT: allow gradients in val/test
    )

    # --- 4. Load Datasets ---
    train_loader, val_loader, test_loader = dataset_loader(
        dataset_name,
        config,
    )

    # --- 5. Pre-compute Statistics on Training Data ---
    print(f"\n--- Pre-computing Mahalanobis statistics on {dataset_name} training data ---")
    # We assume the dataset is an InMemoryDataset with a single Data object
    train_data = train_loader.dataset[0]
    mahalanobis_model.precompute_statistics(train_data)
    print("Mahalanobis statistics pre-computation done.\n")

    # --- 6. Validation (hyperparameter tuning via wandb) ---
    print(f"--- Validating Mahalanobis on {dataset_name} ---")
    trainer.validate(mahalanobis_model, val_loader)

    # --- 7. Final Test ---
    print(f"\n--- Testing Mahalanobis on {dataset_name} ---")
    trainer.test(mahalanobis_model, test_loader)

    wandb.finish()
