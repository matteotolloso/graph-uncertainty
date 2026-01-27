import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import os
import sys
import re # Using regex for robust filename parsing

# --- Add project root to path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import your custom modules ---
from models.credal_Ensemble import credal_Ensemble
from models.VanillaGNN import VanillaGNN # Important: The ensemble loads VanillaGNNs
from dataset_loader.dataset_loader import dataset_loader # Your unified loader
from utils.model_manager import find_best_checkpoints


def ensemble_tester(project_name, dataset_name, **kwargs):
    """
    Tests a CredalEnsemble model built from the best VanillaGNN checkpoints.
    This is a post-hoc, inference-only function.
    """
    # Initialize wandb for this test run
    wandb.init(project=project_name, job_type="test_ensemble")
    config = wandb.config

    # --- 1. Find the Best Checkpoints ---
    num_ensemble_models = config.get("M", 5) 
    checkpoint_paths = find_best_checkpoints(dataset_name, num_ensemble_models)

    # --- 2. Instantiate the Ensemble Model ---
    ensemble_model = credal_Ensemble(
        model_class=VanillaGNN,
        checkpoint_paths=checkpoint_paths
    )

    # --- 3. Set up Trainer and Logger ---
    wandb_logger = WandbLogger(project=project_name)
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        logger=wandb_logger,
        log_every_n_steps=1,
    )

    # --- 4. Load the Test Dataset ---
    # We only need the test_loader for this evaluation
    _, _, test_loader = dataset_loader(dataset_name, config)

    # --- 5. Run the Test ---
    # We DO NOT call trainer.fit(). This is inference only.
    print(f"\n--- Testing Credal Ensemble on {dataset_name} ---")
    trainer.test(ensemble_model, test_loader)

    # Finalize the wandb run
    wandb.finish()
    print("Ensemble testing complete.")
