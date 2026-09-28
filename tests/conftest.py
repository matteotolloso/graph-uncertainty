"""Shared fixtures. Tests run on CPU with W&B disabled and never touch the real checkpoint dir."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["WANDB_MODE"] = "disabled"

import pytest
import torch

torch.set_num_threads(4)

from cgnn.paths import REPO_ROOT, Paths  # noqa: E402

FAST = {
    "accelerator": "cpu",
    "max_epochs": 3,
    "progress_bar": False,
    "model_summary": False,
    "patience": 50,
}


@pytest.fixture
def tmp_paths(tmp_path) -> Paths:
    return Paths(data_dir=REPO_ROOT / "dataset", ckpt_dir=tmp_path / "ckpt", output_dir=tmp_path / "out")


@pytest.fixture(scope="session")
def backbone_paths(tmp_path_factory) -> Paths:
    """Three VanillaGNN backbones trained on the synthetic dataset (split_seed=0)."""
    from cgnn.runner import run_experiment

    root = tmp_path_factory.mktemp("backbones")
    paths = Paths(data_dir=REPO_ROOT / "dataset", ckpt_dir=root / "ckpt", output_dir=root / "out")
    for i, (gnn, lr) in enumerate([("GCN", 0.01), ("SAGE", 0.01), ("GCN", 0.003)]):
        run_experiment(
            "vanilla",
            "synthetic",
            {**FAST, "gnn_type": gnn, "lr": lr, "seed": i},
            logger=False,
            run_id=f"bb{i}",
            paths=paths,
            write_summary=False,
        )
    return paths


def has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False
