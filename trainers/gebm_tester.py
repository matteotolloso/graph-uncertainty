# trainers/gebm_tester.py

import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.GEBM_detector import GEBMModule
from dataset_loader.dataset_loader import dataset_loader
from utils.model_manager import find_best_checkpoints

def gebm_test(project_name, dataset_name, **kwargs):
    wandb.init(project=project_name, job_type="test_gebm")
    config = wandb.config

    # 1) Best VanillaGNN checkpoint
    ckpt_path = find_best_checkpoints(dataset_name, num_models=5)[config["seed"]]

    # 2) Build module
    model = GEBMModule(checkpoint_path=ckpt_path)

    # 3) Get loaders
    train_loader, _, test_loader = dataset_loader(dataset_name, config)

    # 4) FIT GEBM on TRAIN NODES (first train batch is enough for transductive full graph)
    train_batch = next(iter(train_loader))
    model.fit_gebm(train_batch)

    # 5) Lightning test
    wandb_logger = WandbLogger(project=project_name)
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        logger=wandb_logger,
        log_every_n_steps=1,
    )
    print(f"\n--- Testing GEBM on {dataset_name} ---")
    trainer.test(model, dataloaders=test_loader)

    wandb.finish()
    print("GEBM testing complete.")
