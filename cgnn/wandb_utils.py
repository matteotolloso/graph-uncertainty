"""Weights & Biases integration: sweeps, agents, best-config lookup.

The runner itself is W&B-agnostic; this module adapts it to sweeps. Sweep
names are ``<dataset>_<method>`` (legacy convention relied upon by
``cgnn results``).
"""

from __future__ import annotations

import gc
import traceback
from collections.abc import Mapping
from typing import Any

DEFAULT_PROJECT = "graph-uncertainty"


def make_logger(project: str, mode: str, *, name: str | None = None, save_dir: str | None = None):
    """WandbLogger attached to the active run (creating one if needed); CSVLogger when W&B is disabled."""
    if mode == "disabled":
        from lightning.pytorch.loggers import CSVLogger

        from cgnn.paths import Paths

        return CSVLogger(save_dir=save_dir or str(Paths.from_env().output_dir), name=name or "csv")
    from lightning.pytorch.loggers import WandbLogger

    return WandbLogger(project=project)


def agent_function(method: str, dataset: str, project: str, overrides: Mapping[str, Any] | None = None):
    """Function executed by ``wandb.agent`` for each sweep run."""

    def _run() -> None:
        import torch

        import wandb
        from cgnn.runner import run_experiment

        run = wandb.init(project=project)
        cfg = {**dict(wandb.config), **dict(overrides or {})}
        failed = False
        try:
            run_experiment(
                method,
                dataset,
                cfg,
                logger=make_logger(project, "online"),
                run_id=run.id,
                config_callback=log_resolved_config,
            )
        except Exception:
            # Don't re-raise: the traceback's frames would keep the failed run's GPU tensors alive and
            # every following run of this agent would go out of memory too (seen after CUDA OOMs).
            failed = True
            traceback.print_exc()
        wandb.finish(exit_code=1 if failed else 0)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _run


def log_resolved_config(cfg: Mapping[str, Any]) -> None:
    """Record the fully resolved config (defaults included) and the code version on the active W&B run."""
    import wandb
    from cgnn import __version__

    if wandb.run is not None:
        wandb.config.update({**dict(cfg), "cgnn_version": __version__}, allow_val_change=True)


def fetch_best_config(
    sweep_id: str, project: str = DEFAULT_PROJECT, entity: str | None = None
) -> dict[str, Any]:
    """Config of the best run of a sweep (by the sweep's own metric)."""
    import wandb

    api = wandb.Api()
    entity = entity or api.default_entity
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    best = sweep.best_run()
    if best is None:
        raise RuntimeError(f"Sweep {sweep_id} has no finished runs.")
    print(f"Best run of sweep {sweep_id}: {best.id} ({sweep.config.get('metric', {})})")
    return {k: v for k, v in best.config.items() if not k.startswith("_")}
