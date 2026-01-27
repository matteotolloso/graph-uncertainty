# trainers/knn_tester.py

import os, sys, wandb, torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.knn_detector import KNNDetector
from dataset_loader.dataset_loader import dataset_loader
from utils.model_manager import find_best_checkpoints

def knn_test(project_name, dataset_name, save_path, **kwargs):
    wandb.init(project=project_name, job_type="test_knn")
    config = wandb.config

    vanilla_model_dir = os.path.join(save_path, f"vanilla_{dataset_name}")
    backbone_ckpt_path = find_best_checkpoints(dataset_name, num_models=1)[0]
    if not backbone_ckpt_path:
        raise FileNotFoundError(f"No backbone checkpoint found in {vanilla_model_dir}")
    print(f"Using backbone checkpoint: {backbone_ckpt_path}")

    model = KNNDetector(
        backbone_ckpt_path=backbone_ckpt_path,
        k=config.get("k", 50),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, val_loader, test_loader = dataset_loader(
        dataset_name,
        config,
    )

    # Precompute FAISS index from training graph
    train_data = train_loader.dataset[0]
    model.precompute_statistics(train_data)

    logger = WandbLogger(project=project_name)
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        logger=logger,
        enable_checkpointing=False,
        log_every_n_steps=1,
    )

    print(f"\n--- Validating kNN detector on {dataset_name} ---")
    trainer.validate(model, val_loader)

    print(f"\n--- Testing kNN detector on {dataset_name} ---")
    trainer.test(model, test_loader)

    wandb.finish()
