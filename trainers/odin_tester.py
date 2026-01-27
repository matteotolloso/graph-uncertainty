import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from models.ODIN_Detector import ODINDetector
from dataset_loader.dataset_loader import dataset_loader
import os
from utils.model_manager import find_best_checkpoints



def odin_test(project_name, dataset_name, save_path, **kwargs):
    
    wandb.init(project=project_name)
    config = wandb.config
    wandb_logger = WandbLogger(project=project_name)

    # --- 1. Find the Best Backbone Checkpoint ---
    vanilla_model_dir = os.path.join(save_path, f"vanilla_{dataset_name}")
    backbone_ckpt_path = find_best_checkpoints(dataset_name, 1)[0]
    if not backbone_ckpt_path:
        raise FileNotFoundError(f"No backbone checkpoint found in {vanilla_model_dir}")
    print(f"Using backbone checkpoint: {backbone_ckpt_path}")

    # Instantiate the ODIN detector model with sweep parameters
    odin_model = ODINDetector(
        backbone_ckpt_path=backbone_ckpt_path,
        temperature=config.temperature,
        noise_magnitude=config.noise_magnitude
    )

    # Load the dataset
    _, OOD_val_loader, OOD_test_loader = dataset_loader(dataset_name, config)

    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=[]
    )

    trainer.fit(odin_model, OOD_val_loader)

    # Run only test
    trainer.test(odin_model, OOD_test_loader)

    wandb.finish()
