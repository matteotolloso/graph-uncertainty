import os, sys, wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.CaGCN import CaGCNModule
from dataset_loader.dataset_loader import dataset_loader
from utils.model_manager import find_best_checkpoints


def cagcn_train(project_name, dataset_name, **kwargs):
    """
    CaGCN (post-hoc, final scaling stage):
      - pick a pretrained VanillaGNN checkpoint by seed index,
      - freeze it, train the calibration GCN on *validation* nodes,
      - evaluate OOD AUROC + ID metrics on test nodes.
    """
    wandb.init(project=project_name, job_type="train_cagcn")
    config = wandb.config

    # select a pretrained checkpoint by seed index
    ckpt_path = find_best_checkpoints(dataset_name, num_models=5)[config["seed"]]

    model = CaGCNModule(
        checkpoint_path=ckpt_path,
        calib_hidden = config.get("calib_hidden", 16),
        calib_layers = config.get("calib_layers", 2),
        lr          = config.get("lr", 1e-2),
        weight_decay= config.get("weight_decay", 5e-3),
        softplus_eps= config.get("softplus_eps", 1.1),
        ood_in_val  = True,
    )

    train_loader, val_loader, test_loader = dataset_loader(dataset_name, config)

    logger = WandbLogger(project=project_name)
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        logger=logger,
        log_every_n_steps=1,
        max_epochs=config.get("max_epochs", 200),
        callbacks=[
            EarlyStopping(monitor="val_nll", patience=config["patience"]),
        ],
    )

    print(f"\n--- Training CaGCN (final scaling) on {dataset_name} ---")
    # We still pass both loaders so Lightning can run its val loop.
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"--- Testing on {dataset_name} ---")
    trainer.test(model, dataloaders=test_loader)

    wandb.finish()
    print("CaGCN run complete.")
