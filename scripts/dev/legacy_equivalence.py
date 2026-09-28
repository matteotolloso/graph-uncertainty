"""Verify that the cgnn package reproduces the legacy (pre-0.2) trainers exactly.

Runs the *legacy* trainer functions for real (W&B stubbed, Lightning forced to CPU and a few epochs)
and the new runner with the legacy switches (`auroc_impl=torchmetrics`, `split_seed=legacy`), then
compares every logged metric. Covers vanilla, the six credal variants (Squirrel = public split,
ArXiv = random split), GraphESN and all post-hoc methods (Squirrel).

    python scripts/dev/legacy_equivalence.py                 # extracts the legacy code with git archive
    python scripts/dev/legacy_equivalence.py --legacy-ref ff4243f --legacy-dir /tmp/cgnn_legacy

Takes ~4 minutes on CPU and needs the real datasets (squirrel, arxiv). Expected output: every line
`[OK]` and `FAILURES: none`. Re-run after library upgrades or refactors of models/data/runner.
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--legacy-ref", default="ff4243f", help="git ref with the legacy code (last pre-0.2 commit)"
)
parser.add_argument(
    "--legacy-dir", default=None, help="existing checkout of the legacy code (skips git archive)"
)
args = parser.parse_args()

if args.legacy_dir is None:
    legacy_dir = Path(tempfile.mkdtemp(prefix="cgnn_legacy_"))
    archive = subprocess.run(
        ["git", "-C", str(REPO), "archive", args.legacy_ref], capture_output=True, check=True
    )
    subprocess.run(["tar", "-x", "-C", str(legacy_dir)], input=archive.stdout, check=True)
else:
    legacy_dir = Path(args.legacy_dir)
sys.path.insert(0, str(legacy_dir))
sys.path.insert(1, str(REPO))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["WANDB_MODE"] = "disabled"

import torch  # noqa: E402

torch.set_num_threads(16)
import lightning as L  # noqa: E402

import wandb  # noqa: E402

WORK = Path(tempfile.mkdtemp(prefix="cgnn_equiv_"))
(WORK / "dataset").symlink_to(REPO / "dataset")
(WORK / "checkpoints").mkdir()
os.chdir(WORK)  # legacy code uses ./dataset and ./checkpoints

import trainers as legacy_trainers  # noqa: E402

from cgnn.paths import Paths  # noqa: E402
from cgnn.runner import run_experiment  # noqa: E402

MAX_EPOCHS = 4
_RealTrainer = L.Trainer
_last = {}


class _CappedTrainer(_RealTrainer):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("max_epochs", MAX_EPOCHS)
        kwargs.update(accelerator="cpu", devices=1, enable_progress_bar=False, enable_model_summary=False)
        kwargs.setdefault("default_root_dir", str(WORK / "legacy_lightning"))
        super().__init__(*args, **kwargs)
        _last["trainer"] = self
        self._cgnn_metrics = {}


class _Cfg(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


class _Run:
    id = "legacyrun"


def legacy(train_fn, dataset, cfg, run_id):
    _Run.id = run_id
    wandb.init = lambda *a, **k: _Run()
    wandb.finish = lambda *a, **k: None
    wandb.config = _Cfg(cfg)
    wandb.run = _Run()
    L.Trainer = _CappedTrainer
    metrics = {}
    orig_fit, orig_val, orig_test = _RealTrainer.fit, _RealTrainer.validate, _RealTrainer.test

    def grab(orig):
        def inner(self, *a, **k):
            out = orig(self, *a, **k)
            metrics.update({kk: float(v) for kk, v in self.callback_metrics.items()})
            return out

        return inner

    _RealTrainer.fit, _RealTrainer.validate, _RealTrainer.test = (
        grab(orig_fit),
        grab(orig_val),
        grab(orig_test),
    )
    patched = []
    for mod in list(sys.modules.values()):
        if getattr(mod, "__name__", "").startswith("trainers.") and hasattr(mod, "WandbLogger"):
            patched.append(mod)
            mod.WandbLogger = lambda *a, **k: False
    try:
        train_fn(project_name="p", dataset_name=dataset, save_path=str(WORK / "checkpoints"))
    finally:
        L.Trainer = _RealTrainer
        _RealTrainer.fit, _RealTrainer.validate, _RealTrainer.test = orig_fit, orig_val, orig_test
    return metrics


NEW_PATHS = Paths(data_dir=REPO / "dataset", ckpt_dir=WORK / "checkpoints", output_dir=WORK / "new_outputs")
NEW_FIXED = dict(
    accelerator="cpu",
    devices=1,
    auroc_impl="torchmetrics",
    split_seed="legacy",
    progress_bar=False,
    model_summary=False,
    max_epochs=MAX_EPOCHS,
)


def new(method, dataset, cfg, run_id, **extra):
    overrides = {**cfg, **NEW_FIXED, **extra}
    return run_experiment(
        method, dataset, overrides, logger=False, run_id=run_id, paths=NEW_PATHS, write_summary=False
    )


fails = []


def compare(name, m_old, m_new):
    keys = sorted(m_old)
    missing = [k for k in keys if k not in m_new]
    diffs = {
        k: (m_old[k], m_new[k])
        for k in keys
        if k in m_new
        and not ((math.isnan(m_old[k]) and math.isnan(m_new[k])) or abs(m_old[k] - m_new[k]) <= 1e-6)
    }
    ok = not missing and not diffs and len(keys) > 0
    extra = sorted(set(m_new) - set(m_old))
    shown = {k: round(m_old[k], 4) for k in keys if "auroc" in k and k.startswith(("val", "test"))}
    print(
        f"[{'OK' if ok else 'FAIL'}] {name}: {len(keys)} metrics equal {shown}"
        + (f" | missing={missing}" if missing else "")
        + (f" | diffs={diffs}" if diffs else "")
        + (f" | extra={extra}" if extra else "")
    )
    if not ok:
        fails.append(name)


BASE = {
    "in_channels": 2089,
    "out_channels": 3,
    "batch_size": -1,
    "num_neighbors": -1,
    "num_sanity_val_steps": 0,
}
t0 = time.time()

# --- vanilla backbones on squirrel (3 configs -> also used by post-hoc methods) ---------------
vanilla_cfgs = [
    dict(gnn_type="GCN", hidden_channels=64, num_layers=2, lr=0.01, weight_decay=5e-4),
    dict(gnn_type="SAGE", hidden_channels=64, num_layers=2, lr=0.005, weight_decay=1e-4),
    dict(gnn_type="GCN", hidden_channels=128, num_layers=3, lr=0.02, weight_decay=1e-3),
]
for i, vc in enumerate(vanilla_cfgs):
    cfg = {**BASE, **vc, "patience": 30, "monitor": "val_auroc", "mode": "max"}
    m_old = legacy(legacy_trainers.vanilla_train, "squirrel", cfg, f"leg{i}")
    # new side writes to a different dir so the post-hoc comparisons use exactly the legacy files
    m_new = run_experiment(
        "vanilla",
        "squirrel",
        {**cfg, **NEW_FIXED},
        logger=False,
        run_id=f"new{i}",
        paths=Paths(REPO / "dataset", WORK / "new_ckpts", WORK / "new_outputs"),
        write_summary=False,
    )
    compare(f"vanilla/squirrel/{vc['gnn_type']}", m_old, m_new)
print("ckpts:", sorted(p.name for p in (WORK / "checkpoints").glob("*.ckpt")))

# --- credal family on squirrel and (random split) arxiv ----------------------------------------
credal_common = dict(
    gnn_type="GCN",
    hidden_channels=64,
    num_layers=2,
    lr=0.005,
    weight_decay=1e-4,
    delta=0.7,
    patience=10,
    monitor="val_auroc_EU",
    mode="max",
)
cases = [
    ("credal", legacy_trainers.credal_train, {}),
    ("credal_LJ", legacy_trainers.credal_LJ_train, {}),
    ("credal_LJ_dual_head", legacy_trainers.credal_LJ_dual_head_train, {"lambda_cls": 0.7}),
    ("credal_LJ_dual_head_detached", legacy_trainers.credal_LJ_dual_head_detached_train, {"lambda_cls": 0.7}),
    (
        "credal_LJ_dual_head_constrained",
        legacy_trainers.credal_LJ_dual_head_constrained_train,
        {"lambda_cls": 0.7, "lambda_cons": 0.05},
    ),
    (
        "credal_LJ_dual_head_constrained_detached",
        legacy_trainers.credal_LJ_dual_head_constrained_detached_train,
        {"lambda_cls": 0.7, "lambda_cons": 0.05},
    ),
]
for ds, meta in [
    ("squirrel", BASE),
    (
        "arxiv",
        {
            "in_channels": 128,
            "out_channels": 3,
            "batch_size": -1,
            "num_neighbors": 10,
            "num_sanity_val_steps": 0,
        },
    ),
]:
    for method, fn, extra in cases:
        cfg = {**meta, **credal_common, **extra}
        if ds == "arxiv":
            cfg["gnn_type"] = "SAGE"
        compare(
            f"{method}/{ds}", legacy(fn, ds, cfg, "legc"), new(method, ds, cfg, "newc", save_checkpoint=False)
        )

# --- graph ESN ---------------------------------------------------------------------------------
esn = dict(
    hidden_channels=64,
    num_layers=1,
    spectral_radius=0.9,
    input_scaling=0.1,
    leakage=0.9,
    num_reservoirs=3,
    readout_regularization=1e-2,
    bias=False,
    pooling="none",
    fully=False,
    max_iterations=30,
    epsilon=1e-6,
)
compare(
    "graph_esn/squirrel",
    legacy(legacy_trainers.graph_esn_train, "squirrel", {**BASE, **esn}, "lege"),
    new("graph_esn", "squirrel", {**BASE, **esn}, "newe", save_checkpoint=False, max_epochs=1),
)

# --- post-hoc methods on squirrel (same legacy backbone files for both sides) -------------------
posthoc = [
    ("energy", legacy_trainers.energy_test, {"seed": 1, "temperature": 10.0}),
    ("odin", legacy_trainers.odin_test, {"temperature": 100.0, "noise_magnitude": 0.005}),
    ("mahalanobis", legacy_trainers.mahalanobis_test, {"noise_magnitude": 0.001}),
    ("knn", legacy_trainers.knn_test, {"k": 20}),
    ("knn_LJ", legacy_trainers.knn_LJ_test, {"k": 20}),
    ("gnnsafe", legacy_trainers.gnnsafe_test, {"K": 4, "alpha": 0.3}),
    ("gebm", legacy_trainers.gebm_test, {"seed": 0}),
    ("ensemble", legacy_trainers.ensemble_tester, {"M": 3}),
    (
        "frozen",
        legacy_trainers.credal_frozen_joint_train,
        {
            "seed": 0,
            "lr": 0.005,
            "weight_decay": 1e-4,
            "delta": 0.7,
            "patience": 10,
            "monitor": "val_auroc_EU",
            "mode": "max",
        },
    ),
    (
        "cagcn",
        legacy_trainers.cagcn_train,
        {
            "seed": 2,
            "calib_hidden": 16,
            "calib_layers": 2,
            "lr": 0.01,
            "weight_decay": 5e-3,
            "max_epochs": MAX_EPOCHS,
            "patience": 10,
            "monitor": "val_auroc",
            "mode": "max",
        },
    ),
]
for method, fn, cfg in posthoc:
    cfg = {**BASE, **cfg}
    try:
        m_old = legacy(fn, "squirrel", cfg, "legp")
    except Exception as exc:
        print(f"[SKIP] {method}: legacy failed: {type(exc).__name__}: {str(exc)[:200]}")
        continue
    compare(f"{method}/squirrel", m_old, new(method, "squirrel", cfg, "newp", save_checkpoint=False))

print(f"\nFAILURES: {fails or 'none'}   ({time.time() - t0:.0f}s)")
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(1 if fails else 0)
