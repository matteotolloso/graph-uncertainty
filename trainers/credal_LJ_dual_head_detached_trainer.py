import gc
import os
import sys

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from models.credal_gnn_lj_dual_head_detached import credal_GNN_LJ_DualHeadDetached

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset_loader.dataset_loader import dataset_loader


def credal_LJ_dual_head_detached_train(project_name, dataset_name, **kwargs):
    wandb.init(project=project_name)
    config = wandb.config
    L.seed_everything(42, workers=True)
    wandb_logger = WandbLogger(project=project_name)

    model = credal_GNN_LJ_DualHeadDetached(
        gnn_type=config["gnn_type"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        hidden_channels=config["hidden_channels"],
        num_layers=config["num_layers"],
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.0),
        delta=config["delta"],
        lambda_cls=config.get("lambda_cls", 1.0),
    )

    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        deterministic=True,
        num_sanity_val_steps=config.get("num_sanity_val_steps", 0),
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping(
                monitor=config.get("monitor", "val_f1_cls"),
                patience=config["patience"],
                mode=config.get("mode", "max"),
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
