#!/usr/bin/env python3
"""Link the K vanilla backbones with the best *validation accuracy* into a new checkpoint dir.

    python scripts/campaigns/link_backbones_by_accuracy.py squirrel arxiv ... \\
        --src checkpoints/v02 checkpoints --dst checkpoints/aistats/fshift_acc -k 5

Post-hoc methods normally rank backbones by validation OOD AUROC (file name). On several datasets the
top-ranked backbones are near-constant classifiers; experiments about how a *classifier's* uncertainty reacts
(e.g. the test-time feature shift) use this selection instead: point ``CGNN_CKPT_DIR`` at ``--dst``.
Only backbones trained on the current split are considered (same fingerprint; legacy files only on fixed-split
datasets). Accuracy = full-batch accuracy on the ID validation nodes. Existing files are never touched:
``--dst/vanilla`` gets relative symlinks with the original file names, plus ``selection_<dataset>.csv``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import os
from pathlib import Path

import torch

from cgnn.checkpoints import find_checkpoints
from cgnn.data import get_dataset, load_dataset
from cgnn.models.vanilla import VanillaGNN
from cgnn.paths import Paths


def candidates(dataset: str, sources: list[Path], fingerprint: str, split: str):
    seen = set()
    for src in sources:
        paths = dataclasses.replace(Paths.from_env(), ckpt_dir=src.resolve())
        for rec in find_checkpoints(dataset, paths=paths):
            if rec.path.resolve() in seen:
                continue
            meta = rec.meta
            if meta is None and split == "random":
                continue  # unknown training split
            if meta is not None and (
                meta.get("dataset") != dataset or meta.get("split_fingerprint") != fingerprint
            ):
                continue
            seen.add(rec.path.resolve())
            yield rec


@torch.no_grad()
def val_accuracy(path: Path, data) -> tuple[float, int]:
    model = VanillaGNN.load_from_checkpoint(path, map_location="cpu").eval()
    pred = model(data).argmax(dim=1)
    mask = data.val_mask & (data.y.sum(dim=1) == 1)
    y = data.y.argmax(dim=1)
    return float((pred[mask] == y[mask]).float().mean()), int(pred[mask].unique().numel())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="+")
    p.add_argument(
        "--src", nargs="+", type=Path, required=True, help="checkpoint dirs to search (CGNN_CKPT_DIR-like)"
    )
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()
    out = args.dst / "vanilla"
    out.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        bundle = load_dataset(ds, {"batch_size": 0})
        rows = []
        for rec in candidates(ds, args.src, bundle.fingerprint, get_dataset(ds).split):
            acc, distinct = val_accuracy(rec.path, bundle.data)
            rows.append(
                {
                    "file": rec.path.name,
                    "src": str(rec.path),
                    "val_acc": acc,
                    "val_auroc": rec.score,
                    "distinct_preds": distinct,
                }
            )
        rows.sort(key=lambda r: (-r["val_acc"], r["file"]))
        for r in rows[: args.k]:
            link = out / r["file"]
            if not link.exists():
                os.symlink(os.path.relpath(Path(r["src"]).resolve(), out.resolve()), link)
        with (args.dst / f"selection_{ds}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        top = rows[: args.k]
        print(
            f"{ds}: {len(rows)} candidates; top-{args.k} val acc "
            + ", ".join(f"{r['val_acc']:.3f}" for r in top)
            + f" (classes predicted {[r['distinct_preds'] for r in top]}; median of all {rows[len(rows) // 2]['val_acc']:.3f};"
            + f" OOD-ranked #1 has {max(rows, key=lambda r: r['val_auroc'])['val_acc']:.3f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
