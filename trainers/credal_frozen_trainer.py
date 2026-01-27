import os
import sys
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.credal_frozen_LJ import CredalFrozenJoint
from dataset_loader.dataset_loader import dataset_loader
from utils.model_manager import find_best_checkpoints


def credal_frozen_joint_train(project_name, dataset_name, **kwargs):
    """
    Train/val/test the Credal-on-Frozen-Joint ablation:
    - Load a pretrained VanillaGNN checkpoint (selected by seed).
    - Freeze it and train ONLY the Credal layer on joint embeddings.
    - Evaluate on val and test splits.
    """
    wandb.init(project=project_name)
    config = wandb.config

    # hyperparameters
    delta = config.get("delta", 0.5)
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 0.0)

    # choose a pretrained checkpoint per seed index
    # IMPORTANT: uses num_models=5 and picks the index == config["seed"]
    ckpt_path = find_best_checkpoints(dataset_name, num_models=5)[config["seed"]]

    model = CredalFrozenJoint(
        checkpoint_path=ckpt_path,
        lr=lr,
        weight_decay=weight_decay,
        delta=delta,
        ood_in_val=True,
    )

    train_loader, val_loader, test_loader = dataset_loader(dataset_name, config)

    wandb_logger = WandbLogger(project=project_name)
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=config["patience"]),
        ],
    )

    print(f"\n--- Training Credal-on-Frozen-Joint on {dataset_name} ---")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"--- Testing on {dataset_name} ---")
    trainer.test(model, dataloaders=test_loader)

    wandb.finish()
    print("Credal-on-Frozen-Joint run complete.")
