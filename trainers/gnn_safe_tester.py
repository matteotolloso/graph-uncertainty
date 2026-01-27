# trainers/gnnsafe_tester.py

import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import os
import sys

# --- Add project root to path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import your custom modules ---
from models.gnnSafe_detector import GNNSafeDetector
from dataset_loader.dataset_loader import dataset_loader
from utils.model_manager import find_best_checkpoints


def gnnsafe_test(project_name, dataset_name, save_path, **kwargs):
    """
    Post-hoc GNNSafe evaluation on a frozen VanillaGNN backbone.
    This corresponds to the 'GNNSafe' (use_prop=True, use_reg=False) variant:
      - Energy score from logits
      - K-step energy propagation
      - No extra regularization / retraining
    """
    wandb.init(project=project_name, job_type="test_gnnsafe")
    config = wandb.config

    # --- 1. Find the Best Backbone Checkpoint (VanillaGNN) ---
    vanilla_model_dir = os.path.join(save_path, f"vanilla_{dataset_name}")
    backbone_ckpt_path = find_best_checkpoints(dataset_name, 1)[0]
    if not backbone_ckpt_path:
        raise FileNotFoundError(f"No backbone checkpoint found in {vanilla_model_dir}")
    print(f"Using backbone checkpoint: {backbone_ckpt_path}")

    # --- 2. Instantiate the GNNSafe Detector with Hyperparameters ---
    gnnsafe_model = GNNSafeDetector(
        backbone_ckpt_path=backbone_ckpt_path,
        K=config.get("K", 2),
        alpha=config.get("alpha", 0.5),
    )

    # --- 3. Set up Trainer ---
    wandb_logger = WandbLogger(project=project_name)
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        logger=wandb_logger,
        enable_checkpointing=False,
        log_every_n_steps=1,
    )

    # --- 4. Load Datasets ---
    train_loader, val_loader, test_loader = dataset_loader(dataset_name, config)

    # --- 5. Validation and Test (OOD AUROC on val/test masks) ---
    print(f"\n--- Validating GNNSafe on {dataset_name} ---")
    trainer.validate(gnnsafe_model, dataloaders=val_loader)

    print(f"\n--- Testing GNNSafe on {dataset_name} ---")
    trainer.test(gnnsafe_model, dataloaders=test_loader)

    wandb.finish()
