import wandb
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trainers
import argparse
from functools import partial
import importlib
from utils.model_manager import search_best_model


# Set the dataset name with argument parsing
parser = argparse.ArgumentParser(description="Run a sweep for a specific dataset.")

parser.add_argument(
    "-d", 
    "--dataset",
    type=str,
    choices=["chameleon", "patents", "arxiv", "reddit2", "coauthor", "squirrel", "amazon_ratings", "cora", "roman_empire"],
    default="squirrel",
    help="Dataset to run the sweep on.",
)

parser.add_argument(
    "-s",
    "--sweep",
    type=str,
    default="",
    help="sweep id to run",
)

parser.add_argument(
    "-m",
    "--model",
    type=str,
    choices=[ "vanilla", "credal", "ensemble", "credal_LJ", "odin", "mahalanobis", "knn", "energy", "gnnsafe", "knn_LJ", "gebm", "frozen", "cagcn" ],
    default="vanilla",
    help="Model to run the sweep on.",
)

parser.add_argument(
    "-p",
    "--project_name",
    type=str,
    default="graph-uncertainty",
    help="Project name for wandb.",
)

parser.add_argument(
    "--save_path",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "checkpoints"),
    help="Path to save the model checkpoints.",
)

parser.add_argument(
    "-c",
    "--count",
    type=int,
    default=None,
    help="Number of runs to execute in the sweep.",
)

args = parser.parse_args()


try:
    # Dynamically construct the name of the sweep variable
    sweep_model_name = f"sweep_{args.model}"
    sweep_dataset_name = f"metadata_{args.dataset}"
    # Dynamically import the sweeps module
    sweeps_module = importlib.import_module("sweeps.sweeps")
    # Get the sweep dict by name
    sweep_model = getattr(sweeps_module, sweep_model_name)
    sweep_dataset = getattr(sweeps_module, sweep_dataset_name)
except AttributeError:
    raise ValueError(f"No sweep found for: {sweep_model_name} and {sweep_dataset_name}")


sweep = sweep_model.copy() 
sweep['parameters'].update(sweep_dataset)

sweep["name"] = f"{args.dataset}_{args.model}"


if args.model == "vanilla":
    train_func = trainers.vanilla_train
elif args.model == "odin":
    train_func = trainers.odin_test
elif args.model == "mahalanobis":
    train_func = trainers.mahalanobis_test 
elif args.model == "credal": # credal final
    train_func = trainers.credal_train
elif args.model == "ensemble":
    train_func = trainers.ensemble_tester
elif args.model == "credal_LJ":
    train_func = trainers.credal_LJ_train
elif args.model == "knn_LJ": 
    train_func = trainers.knn_LJ_test 
elif args.model == "knn": 
    train_func = trainers.knn_test 
elif args.model == "energy": 
    train_func = trainers.energy_test
elif args.model == "gnnsafe": 
    train_func = trainers.gnnsafe_test 
elif args.model == "gebm":
    train_func = trainers.gebm_test
elif args.model == "frozen":
    train_func = trainers.credal_frozen_joint_train
elif args.model == "cagcn":
    train_func = trainers.cagcn_train
else:
    raise ValueError(f"Unsupported model: {args.model}")


train_func = partial(
    train_func, 
    project_name=args.project_name, 
    dataset_name=args.dataset, 
    save_path=args.save_path
)

if args.sweep:
    sweep_id = args.sweep
else:
    sweep_id = wandb.sweep(sweep=sweep, project=args.project_name)


print(f"Running sweep with ID: {sweep_id}")

wandb.agent(
    sweep_id=sweep_id, 
    function=train_func, 
    project=args.project_name,
    count=args.count,
)

    
    
