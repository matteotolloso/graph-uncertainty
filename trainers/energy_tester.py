# trainers/energy_tester.py

import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import os
import sys

# --- Add project root to path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import your custom modules ---
from models.energy_detector import EnergyDetector
from dataset_loader.dataset_loader import dataset_loader
from utils.model_manager import find_best_checkpoints

def energy_test(project_name, dataset_name, save_path, **kwargs):
    """
    Tests an EnergyDetector, a parameter-free post-hoc method for OOD detection.
    """
    wandb.init(project=project_name, job_type="test_energy")
    config = wandb.config 

    # --- 1. Find the Best Backbone Checkpoint ---
    vanilla_model_dir = os.path.join(save_path, f"vanilla_{dataset_name}")
    backbone_ckpt_path = find_best_checkpoints(dataset_name, num_models=1)[0]
    if not backbone_ckpt_path:
        raise FileNotFoundError(f"No backbone checkpoint found in {vanilla_model_dir}")
    print(f"Using backbone checkpoint: {backbone_ckpt_path}")

    # --- 2. Instantiate the Energy Detector ---
    energy_model = EnergyDetector(
        backbone_ckpt_path=backbone_ckpt_path
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
    _, val_loader, test_loader = dataset_loader(dataset_name, config)
    
    # --- 5. Run Validation and Test ---
    # No pre-computation step is needed for the Energy method.
    print(f"\n--- Validating Energy Detector on {dataset_name} ---")
    trainer.validate(energy_model, val_loader)

    print(f"\n--- Testing Energy Detector on {dataset_name} ---")
    trainer.test(energy_model, test_loader)

    wandb.finish()