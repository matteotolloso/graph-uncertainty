import gc
import os
import sys

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from models.graph_esn import GraphEchoStateNetwork

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset_loader.dataset_loader import dataset_loader


def graph_esn_train(project_name, dataset_name, save_path):
    wandb.init(project=project_name)
    config = wandb.config
    L.seed_everything(42, workers=True)

    wandb_logger = WandbLogger(project=project_name)
    monitor = "val_auroc_TU_credal"
    mode = "max"

    model = GraphEchoStateNetwork(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        hidden_channels=config["hidden_channels"],
        num_layers=config["num_layers"],
        lr=config.get("lr", 0.0),
        weight_decay=config.get("weight_decay", 0.0),
        ood_in_val=config.get("ood_in_val", True),
        spectral_radius=config.get("spectral_radius", 0.9),
        input_scaling=config.get("input_scaling", 1.0),
        leakage=config.get("leakage", 1.0),
        num_reservoirs=config.get("num_reservoirs", 1),
        readout_regularization=config.get("readout_regularization", 1e-3),
        bias=config.get("bias", False),
        pooling=config.get("pooling", None),
        fully=config.get("fully", False),
        max_iterations=config.get("max_iterations", None),
        epsilon=config.get("epsilon", 1e-6),
    )

    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        deterministic=True,
        max_epochs=1,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(
                monitor=monitor,
                mode=mode,
                save_top_k=1,
                save_last=False,
                dirpath=save_path,
                filename=f"{wandb.run.id}_{dataset_name}_{monitor}={{{monitor}:.4f}}",
                auto_insert_metric_name=False,
            ),
        ],
    )

    train_loader, val_loader, test_loader = dataset_loader(dataset_name, config)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    wandb.finish()

    del model
    del trainer
    del train_loader, val_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()
