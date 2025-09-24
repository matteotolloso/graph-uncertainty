

# Credal Graph Neural Network 

This repository contains the implementation and evaluation of various uncertainty quantification (UQ) methods for Graph Neural Networks (GNNs). Notably, this work introduces **Credal GNN**, a novel model for uncertainty-aware graph learning. The project is designed to benchmark different models, including our proposed method, on several graph-based datasets, using `wandb` for experiment tracking and hyperparameter sweeps.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running an Experiment Sweep](#running-an-experiment-sweep)
  - [Joining an Existing Sweep](#joining-an-existing-sweep)
  - [Command-Line Arguments](#command-line-arguments)
- [Implemented Models](#implemented-models)
- [Datasets](#datasets)
- [Configuration](#configuration)

## Project Structure
The project is organized into modular components for models, trainers, and data handling.

```

├── checkpoints/         # Saved model checkpoints
├── dataset/
├── dataset_loader/
├── docker/
├── models/               # GNN and detector model implementations
│   ├── __pycache__/
│   ├── credal_Ensemble.py
│   ├── ...
│   ├── knn_LJ_detector.py
│   └── VanillaGNN.py
├── sweeps/               # Wandb sweep configurations
│   └── sweeps.py
├── test/
├── trainers/             # Training and evaluation logic for each model
│   ├── __pycache__/
│   ├── __init__.py
│   ├── credal_trainer.py
│   ├── ...
│   └── vanilla_trainer.py
├── utils/                # Utility functions (e.g., model management)
├── wandb/                # Wandb local files
├── .gitignore
├── main.py               # Main script to run experiments and sweeps
├── readme.md             # This README file
└── run_all.sh            # Example script to run multiple experiments
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment and install dependencies:**
    It is recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should be created containing necessary packages like `wandb`, `torch`, `torch_geometric`, etc.)*

3.  **Log in to Weights & Biases:**
    You will need a `wandb` account to track experiments.
    ```bash
    wandb login
    ```

## Usage

The main entry point for all experiments is `main.py`. It uses `wandb` sweeps to perform hyperparameter optimization.

### Running an Experiment Sweep

To start a new sweep, you need to specify a model and a dataset. The script will generate a new sweep ID and start an agent to run the experiments.

**Example:** Start a new sweep for the `credal` model on the `squirrel` dataset.

```bash
python main.py --model credal --dataset squirrel
```

To limit the number of runs in the sweep, use the `-c` or `--count` argument:

```bash
python main.py --model vanilla --dataset chameleon --count 10
```

### Joining an Existing Sweep

If you have a `sweep_id` from a previously created sweep, you can make a new agent join it.

```bash
python main.py --model <model_name> --dataset <dataset_name> --sweep <your_sweep_id>
```

### Command-Line Arguments

The `main.py` script accepts the following arguments:

| Argument | Short | Description | Choices | Default |
|---|---|---|---|---|
| `--dataset` | `-d` | The dataset to run the sweep on. | `chameleon`, `patents`, `arxiv`, `reddit2`, `coauthor`, `squirrel` | `squirrel` |
| `--model` | `-m` | The model to run. | `vanilla`, `credal`, `ensemble`, `credal_LJ`, `odin`, `mahalanobis`, `knn`, `energy`, `gnnsafe`, `knn_LJ` | `vanilla` |
| `--sweep` | `-s`| An existing `wandb` sweep ID to join. | | `""` |
| `--project_name`| `-p`| The project name for `wandb`. | | `graph-uncertainty` |
| `--save_path` | | Path to save model checkpoints. | | `./checkpoints` |
| `--count` | `-c` | The number of runs to execute in the sweep. | | `None` (runs forever) |

## Implemented Models

The following models/methods are implemented. Each model has a corresponding trainer in the `trainers/` directory.

- `vanilla`: A standard GNN baseline.
- `credal`: A GNN based on Credal Self-Supervised Learning.
- `ensemble`: A deep ensemble of GNNs.
- `credal_LJ`: A variant of the Credal GNN.
- `odin`: ODIN for out-of-distribution detection.
- `mahalanobis`: A detector using Mahalanobis distance.
- `knn`: A k-Nearest Neighbors based detector.
- `energy`: An energy-based detector.
- `gnnsafe`: GnnSafe detector.
- `knn_LJ`: A variant of the KNN detector.

## Datasets
The framework is configured to run on the following datasets:
- `chameleon`
- `patents`
- `arxiv`
- `reddit2`
- `coauthor`
- `squirrel`

## Configuration

Hyperparameter search spaces are defined in `sweeps/sweeps.py`. The `main.py` script dynamically loads the appropriate configuration based on the selected `--model` and `--dataset`. You can modify this file to change search intervals, distributions, and other sweep parameters.