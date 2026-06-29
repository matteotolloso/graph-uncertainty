import torch
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from models.VanillaGNN import VanillaGNN
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset_loader.dataset_loader import dataset_loader
import gc


def vanilla_train(project_name, dataset_name, save_path):
    
    # Initialize wandb at the start of the experiment
    wandb.init(project=project_name)

    config = wandb.config
    L.seed_everything(42, workers=True)

    wandb_logger = WandbLogger(project=project_name)
    monitor = config.get("monitor", "val_f1")
    mode = config.get("mode", "max")

    # Instantiate the model
    model = VanillaGNN(
        gnn_type=config["gnn_type"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        hidden_channels=config["hidden_channels"],
        num_layers=config["num_layers"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # Trainer setup
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        deterministic=True,
        num_sanity_val_steps=config.get("num_sanity_val_steps", 0),
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping(
                monitor=monitor,
                patience=config["patience"], 
                mode=mode,
            ),
            ModelCheckpoint(
                monitor=monitor,
                mode=mode,
                save_top_k=1,       
                save_last=False,         
                dirpath=save_path,         
                filename=f'{wandb.run.id}_{dataset_name}_{monitor}={{{monitor}:.4f}}',
                auto_insert_metric_name=False # We already included the metric in filename
            ),
        ],
    )

    # Load the dataset
    train_loader, val_loader, test_loader = dataset_loader(dataset_name, config)

    # Train and validate the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, test_loader)

    # Finalize the wandb run
    wandb.finish()

    del model
    del trainer
    del train_loader, val_loader, test_loader 
    gc.collect() 
    torch.cuda.empty_cache()
