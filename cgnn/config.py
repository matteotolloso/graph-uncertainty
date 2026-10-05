"""Run-config resolution and W&B sweep construction.

Resolution order (later wins)::

    configs/defaults.yaml
    configs/methods/<method>.yaml  `defaults`
    dataset metadata               (DatasetSpec.default_config(): in/out_channels, batch_size, num_neighbors)
    overrides                      (W&B sweep values or `cgnn run --set key=value`)
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from cgnn.paths import CONFIG_DIR

MAX_GPUS = 2  # shared server policy: never more than 2 GPUs in total


@cache
def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def global_defaults() -> dict[str, Any]:
    return copy.deepcopy(_load_yaml(str(CONFIG_DIR / "defaults.yaml")))


def method_config_path(method: str) -> Path:
    return CONFIG_DIR / "methods" / f"{method}.yaml"


def method_config(method: str) -> dict[str, Any]:
    path = method_config_path(method)
    if not path.exists():
        raise FileNotFoundError(f"Missing config for method '{method}': {path}")
    doc = copy.deepcopy(_load_yaml(str(path)))
    doc.setdefault("defaults", {})
    doc.setdefault("sweep", None)
    return doc


def resolve_config(
    method: str,
    dataset_defaults: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
    *,
    seed_is_backbone_rank: bool = False,
) -> dict[str, Any]:
    overrides = dict(overrides or {})
    if seed_is_backbone_rank and "seed" in overrides and "backbone_rank" not in overrides:
        # Sweeps of energy/gebm/cagcn/frozen use `seed` to index the backbone ranking
        # while the global seed stays 42.
        overrides["backbone_rank"] = overrides.pop("seed")

    cfg = global_defaults()
    cfg.update(method_config(method)["defaults"])
    cfg.update(dataset_defaults)
    cfg.update(overrides)
    validate_hardware(cfg)
    return cfg


def validate_hardware(cfg: Mapping[str, Any]) -> None:
    devices = cfg.get("devices", 1)
    if not isinstance(devices, int) or isinstance(devices, bool):
        raise ValueError(f"`devices` must be an int (got {devices!r}); 'auto' would grab every visible GPU.")
    if devices > MAX_GPUS:
        raise ValueError(f"`devices={devices}` exceeds the shared-server limit of {MAX_GPUS} GPUs.")


def build_sweep(
    method: str, dataset: str, dataset_sweep_metadata: Mapping[str, Any], sweep_file: str | Path | None = None
) -> dict[str, Any]:
    """W&B sweep dict: the method's sweep + fixed dataset metadata, named ``<dataset>_<method>``."""
    if sweep_file is not None:
        doc = yaml.safe_load(Path(sweep_file).read_text())
        sweep = doc.get("sweep", doc)
    else:
        sweep = method_config(method)["sweep"]
        if sweep is None:
            raise ValueError(f"Method '{method}' has no sweep in {method_config_path(method)}")
    sweep = copy.deepcopy(sweep)
    sweep.setdefault("parameters", {}).update(copy.deepcopy(dict(dataset_sweep_metadata)))
    sweep["name"] = f"{dataset}_{method}"
    return sweep


def parse_value(text: str) -> Any:
    """YAML-typed value; also accepts ``1e-3`` (which PyYAML alone would keep as a string)."""
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            return float(value) if any(c in value for c in ".eE") else int(value)
        except ValueError:
            return value
    return value


def parse_overrides(items: Iterable[str] | None) -> dict[str, Any]:
    """``["lr=1e-3", "gnn_type=SAGE", "split_seed=null"]`` -> typed dict."""
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key=value")
        key, value = item.split("=", 1)
        out[key.strip()] = parse_value(value)
    return out
