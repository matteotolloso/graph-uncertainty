"""Checkpoint naming, discovery and backbone selection.

Naming (unchanged from the legacy code so the existing files keep working)::

    <ckpt_dir>/<method>/<run_id>_<dataset>_<metric>=<score:.4f>.ckpt
    <ckpt_dir>/<run_id>_<dataset>_val_auroc=<score>.ckpt        # legacy vanilla layout (flat)

Post-hoc methods pick VanillaGNN backbones ranked by ``val_auroc`` (legacy
fallback: ``val_f1``). Ties are broken by file name, so the choice is
deterministic.

Every checkpoint written by the new runner carries ``checkpoint["cgnn"]`` with
the dataset, split seed and split fingerprint; :func:`check_backbone_split`
uses it to refuse backbones trained on a different split (which would leak
test nodes into the backbone's training set).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import torch

from cgnn.paths import Paths

_SCORE = r"(?P<score>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def checkpoint_filename(run_id: str, dataset: str, monitor: str) -> str:
    """Lightning ``ModelCheckpoint`` filename template (without ``.ckpt``)."""
    return f"{run_id}_{dataset}_{monitor}={{{monitor}:.4f}}"


@dataclass(frozen=True)
class CheckpointRecord:
    path: Path
    run_id: str
    dataset: str
    metric: str
    score: float

    @cached_property
    def _header(self) -> dict:
        ckpt = torch.load(self.path, map_location="cpu", weights_only=False)
        return {"cgnn": ckpt.get("cgnn"), "hyper_parameters": dict(ckpt.get("hyper_parameters", {}))}

    @property
    def meta(self) -> dict | None:
        """``checkpoint["cgnn"]`` (dataset, split fingerprint, ...) or ``None`` for legacy files."""
        return self._header["cgnn"]

    @property
    def hparams(self) -> dict:
        return self._header["hyper_parameters"]


def _scan(directory: Path, dataset: str, metric: str) -> list[CheckpointRecord]:
    if not directory.is_dir():
        return []
    pattern = re.compile(
        rf"^(?P<run>[A-Za-z0-9]+)_{re.escape(dataset)}_{re.escape(metric)}={_SCORE}(?:-v\d+)?\.ckpt$"
    )
    out = []
    for p in directory.iterdir():
        m = pattern.match(p.name)
        if m:
            out.append(CheckpointRecord(p, m["run"], dataset, metric, float(m["score"])))
    return out


def find_checkpoints(
    dataset: str,
    *,
    method: str = "vanilla",
    metric: str = "val_auroc",
    fallback_metrics: tuple[str, ...] = ("val_f1",),
    mode: str = "max",
    paths: Paths | None = None,
) -> list[CheckpointRecord]:
    """All checkpoints of ``method`` on ``dataset``, best first."""
    paths = paths or Paths.from_env()
    dirs = [paths.ckpt_dir / method]
    if method == "vanilla":
        dirs.append(paths.ckpt_dir)  # legacy flat layout
    for m in (metric, *fallback_metrics):
        records = [r for d in dirs for r in _scan(d, dataset, m)]
        if records:
            if m != metric:
                warnings.warn(
                    f"No '{metric}' checkpoints for '{dataset}'; falling back to '{m}'.", stacklevel=2
                )
            sign = -1.0 if mode == "max" else 1.0
            return sorted(records, key=lambda r: (sign * r.score, r.path.name))
    return []


def select_backbones(
    dataset: str, *, count: int = 1, rank: int = 0, paths: Paths | None = None
) -> list[CheckpointRecord]:
    """``count`` consecutive vanilla backbones starting at ``rank`` (0 = best)."""
    records = find_checkpoints(dataset, paths=paths)
    if not records:
        root = (paths or Paths.from_env()).ckpt_dir
        raise FileNotFoundError(
            f"No vanilla checkpoints for '{dataset}' in {root}/vanilla or {root}. "
            f"Train backbones first: `cgnn sweep -m vanilla -d {dataset}` (or `cgnn run -m vanilla -d {dataset}`)."
        )
    selected = records[rank : rank + count]
    if len(selected) < count:
        raise IndexError(
            f"Requested backbones {rank}..{rank + count - 1} for '{dataset}' but only {len(records)} exist."
        )
    return selected


def check_backbone_split(
    records: list[CheckpointRecord], dataset: str, split_type: str, fingerprint: str, policy: str
) -> None:
    """Verify that each backbone was trained on the split being evaluated.

    ``policy``: ``"error"`` | ``"warn"`` | ``"off"``.
    """
    if policy == "off":
        return
    problems = []
    for r in records:
        meta = r.meta
        if meta is None:
            if split_type == "random":
                problems.append(
                    f"{r.path.name}: legacy checkpoint without split metadata on a random-split dataset; "
                    "its training split is unknown (possible test leakage)."
                )
            continue
        if meta.get("dataset") != dataset:
            problems.append(f"{r.path.name}: trained on '{meta.get('dataset')}', not '{dataset}'.")
        elif meta.get("split_fingerprint") != fingerprint:
            problems.append(
                f"{r.path.name}: split fingerprint {meta.get('split_fingerprint')} != current {fingerprint} "
                f"(backbone split_seed={meta.get('split_seed')})."
            )
    if not problems:
        return
    message = "Backbone/split mismatch:\n  " + "\n  ".join(problems)
    if policy == "error" and any("fingerprint" in p or "trained on" in p for p in problems):
        raise RuntimeError(message + "\nSet backbone_split_check=warn to proceed anyway.")
    warnings.warn(message, stacklevel=2)
