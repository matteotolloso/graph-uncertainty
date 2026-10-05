"""The single generic experiment runner used by `cgnn run`, `cgnn sweep` and the tests.

Order of operations is deliberate (it fixes the RNG consumption, so seeded runs
reproduce the paper's experiments):

trainable:  resolve cfg -> seed -> build model -> trainer -> load data -> fit -> test
post-hoc:   resolve cfg -> seed -> select backbones -> build -> load data
            -> split check -> prepare(train graph) -> [fit on train|val] / validate -> test
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from cgnn import __version__, metrics
from cgnn.checkpoints import check_backbone_split, checkpoint_filename, select_backbones
from cgnn.config import resolve_config
from cgnn.data import GraphBundle, get_dataset, load_dataset
from cgnn.methods import MethodSpec, RunContext, get_method
from cgnn.paths import Paths
from cgnn.uncertainty.entropy import set_min_entropy_method


def runtime_device(cfg: Mapping[str, Any]) -> torch.device:
    if cfg.get("accelerator", "auto") != "cpu" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _callbacks(ctx: RunContext) -> list:
    cfg = ctx.cfg
    callbacks: list = []
    if cfg.get("early_stopping", True):
        callbacks.append(
            EarlyStopping(monitor=cfg["monitor"], patience=cfg["patience"], mode=cfg.get("mode", "max"))
        )
    if cfg.get("save_checkpoint", True):
        callbacks.append(
            ModelCheckpoint(
                monitor=cfg["monitor"],
                mode=cfg.get("mode", "max"),
                save_top_k=1,
                save_last=False,
                dirpath=str(ctx.paths.ckpt_dir / ctx.method.name),
                filename=checkpoint_filename(ctx.run_id, ctx.dataset.name, cfg["monitor"]),
                auto_insert_metric_name=False,
            )
        )
    return callbacks


def _trainer(ctx: RunContext, logger, *, callbacks: list, inference_mode: bool = True) -> L.Trainer:
    cfg = ctx.cfg
    accelerator = cfg.get("accelerator", "auto")
    return L.Trainer(
        accelerator=accelerator,
        devices=1 if accelerator == "cpu" else int(cfg.get("devices", 1)),
        deterministic=bool(cfg.get("deterministic", True)),
        num_sanity_val_steps=int(cfg.get("num_sanity_val_steps", 0)),
        max_epochs=cfg.get("max_epochs"),
        logger=logger if logger is not None else False,
        log_every_n_steps=1,
        callbacks=callbacks,
        enable_checkpointing=any(isinstance(c, ModelCheckpoint) for c in callbacks),
        inference_mode=inference_mode,
        default_root_dir=str(ctx.paths.output_dir / "lightning"),
        enable_progress_bar=bool(cfg.get("progress_bar", True)),
        enable_model_summary=bool(cfg.get("model_summary", True)),
    )


def _meta(ctx: RunContext, bundle: GraphBundle) -> dict[str, Any]:
    return {
        "cgnn_version": __version__,
        "method": ctx.method.name,
        "dataset": ctx.dataset.name,
        "run_id": ctx.run_id,
        "seed": ctx.cfg.get("seed"),
        "split_seed": bundle.split_seed,
        "split_fingerprint": bundle.fingerprint,
        "backbones": [str(r.path) for r in ctx.backbones],
    }


def _collect(trainer: L.Trainer, into: dict[str, float]) -> None:
    for key, value in trainer.callback_metrics.items():
        into[key] = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)


def _test_ckpt(ctx: RunContext) -> str | None:
    choice = ctx.cfg.get("test_ckpt", "last")
    if choice not in ("last", "best"):
        raise ValueError(f"test_ckpt must be 'last' or 'best', got {choice!r}")
    if choice == "best" and not ctx.cfg.get("save_checkpoint", True):
        raise ValueError("test_ckpt=best requires save_checkpoint=true")
    return None if choice == "last" else "best"


def run_trainable(ctx: RunContext, logger) -> tuple[dict[str, float], GraphBundle]:
    model = ctx.method.build(ctx)
    trainer = _trainer(ctx, logger, callbacks=_callbacks(ctx))
    bundle = load_dataset(ctx.dataset.name, ctx.cfg, paths=ctx.paths)
    model.cgnn_meta = _meta(ctx, bundle)

    results: dict[str, float] = {}
    trainer.fit(model, bundle.train_loader, bundle.val_loader)
    _collect(trainer, results)
    trainer.test(model, bundle.test_loader, ckpt_path=_test_ckpt(ctx))
    _collect(trainer, results)
    return results, bundle


def run_posthoc(ctx: RunContext, logger) -> tuple[dict[str, float], GraphBundle]:
    cfg, spec = ctx.cfg, ctx.method
    count = spec.num_backbones(cfg) if spec.num_backbones else 1
    rank = 0 if spec.num_backbones else int(cfg.get("backbone_rank", 0))
    ctx.backbones = select_backbones(ctx.dataset.name, count=count, rank=rank, paths=ctx.paths)
    print("Backbones: " + ", ".join(f"{r.path.name} ({r.metric}={r.score:.4f})" for r in ctx.backbones))

    model = spec.build(ctx)
    bundle = load_dataset(
        ctx.dataset.name,
        cfg,
        paths=ctx.paths,
        num_layers=int(ctx.backbones[0].hparams.get("num_layers", 2)),
        force_full_batch_eval=getattr(model, "requires_full_graph_eval", False),
    )
    check_backbone_split(
        ctx.backbones,
        ctx.dataset.name,
        ctx.dataset.split,
        bundle.fingerprint,
        cfg.get("backbone_split_check", "warn"),
    )
    model.cgnn_meta = _meta(ctx, bundle)

    if hasattr(model, "prepare"):
        model.to(runtime_device(cfg))
        model.prepare(bundle.data)

    results: dict[str, float] = {}
    if spec.fit_on is not None:
        trainer = _trainer(ctx, logger, callbacks=_callbacks(ctx))
        fit_loader = bundle.train_loader if spec.fit_on == "train" else bundle.val_loader
        trainer.fit(model, train_dataloaders=fit_loader, val_dataloaders=bundle.val_loader)
        _collect(trainer, results)
        trainer.test(model, dataloaders=bundle.test_loader, ckpt_path=_test_ckpt(ctx))
    else:
        needs_grad = getattr(model, "needs_input_grad", False)
        trainer = _trainer(ctx, logger, callbacks=[], inference_mode=not needs_grad)
        trainer.validate(model, bundle.val_loader)
        _collect(trainer, results)
        trainer.test(model, bundle.test_loader)
    _collect(trainer, results)
    return results, bundle


def write_run_summary(ctx: RunContext, results: Mapping[str, float], bundle: GraphBundle) -> Path:
    out_dir = ctx.paths.output_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{stamp}_{ctx.dataset.name}_{ctx.method.name}_{ctx.run_id}.json"
    payload = {
        "method": ctx.method.name,
        "paper_name": ctx.method.paper_name,
        "dataset": ctx.dataset.name,
        "run_id": ctx.run_id,
        "metrics": dict(results),
        "config": {k: v for k, v in ctx.cfg.items() if isinstance(v, (int, float, str, bool, type(None)))},
        "data": bundle.summary(),
        "backbones": [str(r.path) for r in ctx.backbones],
        "cgnn_version": __version__,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def run_experiment(
    method: str,
    dataset: str,
    overrides: Mapping[str, Any] | None = None,
    *,
    logger=None,
    run_id: str | None = None,
    paths: Paths | None = None,
    write_summary: bool = True,
    config_callback=None,
) -> dict[str, float]:
    """Run one experiment and return every logged metric (``{name: float}``).

    ``config_callback(cfg)`` is called with the fully resolved config (used to log it to W&B).
    """
    spec: MethodSpec = get_method(method)
    ds = get_dataset(dataset)
    cfg = resolve_config(
        method, ds.default_config(), overrides, seed_is_backbone_rank=spec.seed_is_backbone_rank
    )
    if config_callback is not None:
        config_callback({**cfg, "cgnn_method": method, "cgnn_dataset": dataset})
    metrics.set_auroc_impl(cfg.get("auroc_impl", "torchmetrics"))
    set_min_entropy_method(cfg.get("min_entropy_method", "greedy"))

    ctx = RunContext(
        method=spec,
        dataset=ds,
        cfg=cfg,
        paths=paths or Paths.from_env(),
        run_id=run_id or uuid.uuid4().hex[:8],
    )
    L.seed_everything(int(cfg["seed"]), workers=True)
    runner = run_trainable if spec.kind == "trainable" else run_posthoc
    results, bundle = runner(ctx, logger)
    if write_summary:
        path = write_run_summary(ctx, results, bundle)
        print(f"Run summary: {path}")
    return results
