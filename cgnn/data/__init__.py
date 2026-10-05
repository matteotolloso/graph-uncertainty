"""Datasets: registry, leave-out-class OOD protocol, loaders.

Typical use::

    from cgnn.data import load_dataset
    bundle = load_dataset("squirrel", {"split_seed": 0})
    bundle.data, bundle.train_loader, bundle.fingerprint
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from torch_geometric.data import Data

import cgnn.data.sources  # noqa: F401  (registers all datasets)
from cgnn.data.loaders import build_loaders, full_batch_loader, neighbor_loader
from cgnn.data.ood import apply_leave_out_classes, split_summary
from cgnn.data.perturb import perturb_training_labels
from cgnn.data.spec import DATASETS, DatasetSpec, RawGraph, register_dataset
from cgnn.data.splits import random_split_masks, split_fingerprint
from cgnn.paths import Paths

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "GraphBundle",
    "RawGraph",
    "get_dataset",
    "load_dataset",
    "register_dataset",
]

DEFAULT_SPLIT_SEED = 0


def get_dataset(name: str) -> DatasetSpec:
    return DATASETS.get(name)


@dataclass
class GraphBundle:
    spec: DatasetSpec
    data: Data
    train_loader: Any
    val_loader: Any
    test_loader: Any
    split_seed: int | None
    fingerprint: str
    loader_kwargs: dict[str, Any] = field(default_factory=dict)

    def test_loader_for(self, data: Data):
        """Test loader over another version of the graph (same batching as ``test_loader``)."""
        kw = self.loader_kwargs
        if kw["batch_size"] > 0 and kw["eval_neighbor_sampling"]:
            return neighbor_loader(
                data, "test", kw["batch_size"], kw["num_neighbors"], kw["num_layers"], False
            )
        return full_batch_loader(data)

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.spec.name,
            "num_nodes": int(self.data.num_nodes),
            "num_edges": int(self.data.edge_index.size(1)),
            "split_seed": self.split_seed,
            "split_fingerprint": self.fingerprint,
            **split_summary(self.data),
        }


_RAW_CACHE: dict[tuple[str, str], RawGraph] = {}


def _load_raw_cached(spec: DatasetSpec, paths: Paths, use_cache: bool) -> RawGraph:
    key = (spec.name, str(paths.data_dir))
    if use_cache and key in _RAW_CACHE:
        return _RAW_CACHE[key]
    raw = spec.load_raw(paths.data_dir, spec)
    if use_cache:
        _RAW_CACHE.clear()  # keep at most one (possibly huge) graph in memory
        _RAW_CACHE[key] = raw
    return raw


def resolve_split_seed(cfg: Mapping[str, Any]) -> int | None:
    """``split_seed`` from the config; ``None``/``"legacy"`` selects the legacy global-RNG split."""
    value = cfg.get("split_seed", DEFAULT_SPLIT_SEED)
    if value is None or (isinstance(value, str) and value.lower() in {"legacy", "none", "null"}):
        return None
    return int(value)


def load_dataset(
    name: str,
    cfg: Mapping[str, Any] | None = None,
    *,
    paths: Paths | None = None,
    use_cache: bool = True,
    num_layers: int | None = None,
    force_full_batch_eval: bool = False,
) -> GraphBundle:
    """Load a dataset with the leave-out-class OOD split and build its loaders.

    Config keys read: ``split_seed`` (random-split datasets only), ``batch_size``,
    ``num_neighbors``, ``num_layers`` (NeighborLoader depth), ``eval_neighbor_sampling``,
    ``train_fraction`` / ``label_noise`` / ``perturb_seed`` (``cgnn.data.perturb``).
    """
    cfg = dict(cfg or {})
    spec = get_dataset(name)
    paths = paths or Paths.from_env()
    raw = _load_raw_cached(spec, paths, use_cache)

    split_seed: int | None = None
    if spec.split == "random":
        split_seed = resolve_split_seed(cfg)
        train_ratio, val_ratio = spec.split_ratios
        base_train, base_val, base_test = random_split_masks(
            raw.num_nodes, train_ratio, val_ratio, split_seed
        )
    else:
        if raw.train_mask is None or raw.val_mask is None or raw.test_mask is None:
            raise ValueError(f"Dataset '{name}' declares a public split but its loader returned no masks")
        base_train, base_val, base_test = raw.train_mask, raw.val_mask, raw.test_mask

    data = apply_leave_out_classes(
        raw.x, raw.edge_index, raw.y, base_train, base_val, base_test, spec.id_classes, spec.ood_classes
    )

    perturbation = perturb_training_labels(data, cfg)

    batch_size = int(cfg.get("batch_size", spec.batch_size))
    eval_sampling = (
        bool(cfg.get("eval_neighbor_sampling", spec.eval_neighbor_sampling)) and not force_full_batch_eval
    )
    layers = int(cfg.get("num_layers", num_layers if num_layers is not None else 2))
    loader_kwargs = {
        "batch_size": batch_size,
        "num_neighbors": int(cfg.get("num_neighbors", spec.num_neighbors)),
        "num_layers": layers,
        "eval_neighbor_sampling": eval_sampling,
    }
    train_loader, val_loader, test_loader = build_loaders(data, **loader_kwargs)

    bundle = GraphBundle(
        spec=spec,
        data=data,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        split_seed=split_seed,
        fingerprint=split_fingerprint(data.train_mask, data.val_mask, data.test_mask, tag=perturbation),
        loader_kwargs=loader_kwargs,
    )
    s = bundle.summary()
    print(
        f"--- {spec.paper_name}: train(ID)={s['train_id']} | val ID/OOD={s['val_id']}/{s['val_ood']} | "
        f"test ID/OOD={s['test_id']}/{s['test_ood']} | split_seed={split_seed} fp={bundle.fingerprint}"
    )
    return bundle
