# trainers/knn_LJ_tester.py

import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import os
import sys
from utils.model_manager import find_best_checkpoints

# --- Add project root to path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import your custom modules ---
from models.knn_LJ_detector import KNN_LJ_Detector
from dataset_loader.dataset_loader import dataset_loader
from utils.model_manager import search_best_model




def knn_LJ_test(project_name, dataset_name, save_path, **kwargs):
    wandb.init(project=project_name, job_type="test_knn_LJ")
    config = wandb.config

    # --- 1. Find the Best Backbone Checkpoint ---
    vanilla_model_dir = os.path.join(save_path, f"vanilla_{dataset_name}")
    backbone_ckpt_path = find_best_checkpoints(dataset_name, 1)[0]
    
    if not backbone_ckpt_path:
        raise FileNotFoundError(f"No backbone checkpoint found in {vanilla_model_dir}")
    print(f"Using backbone checkpoint: {backbone_ckpt_path}")

    
    # --- 4. Load ALL Datasets needed ---
    # The trainer needs access to the train_loader to pass it to the model's setup hook.
    train_loader, val_loader, test_loader = dataset_loader(dataset_name, config)

    # --- 2. Instantiate the KNN_LJ Detector ---
    knn_lj_model = KNN_LJ_Detector(
        backbone_ckpt_path=backbone_ckpt_path,
        k=config.get("k", 50),
        train_loader=train_loader  # Pass the train_loader to the model
    )

    # --- 3. Set up Trainer ---
    wandb_logger = WandbLogger(project=project_name)
    trainer = L.Trainer(
        devices="auto",
        accelerator="cpu",
        logger=wandb_logger,
        enable_checkpointing=False,
        log_every_n_steps=1,
    )


    
    # --- 5. Run Validation and Test ---
    # Lightning will automatically call .setup(stage='validate') before this.
    print(f"\n--- Validating KNN_LJ on {dataset_name} ---")
    trainer.validate(knn_lj_model, val_loader)

    # Lightning will automatically call .setup(stage='test') before this.
    print(f"\n--- Testing KNN_LJ Detector on {dataset_name} ---")
    trainer.test(knn_lj_model, test_loader)

    wandb.finish()
