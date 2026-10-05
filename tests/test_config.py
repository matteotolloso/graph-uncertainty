"""Registry <-> config consistency, sweep construction, override parsing, GPU cap."""

from __future__ import annotations

import pytest

from cgnn.config import (
    CONFIG_DIR,
    build_sweep,
    method_config,
    parse_overrides,
    resolve_config,
    validate_hardware,
)
from cgnn.data import get_dataset
from cgnn.methods import METHODS


def test_every_method_has_a_config_and_vice_versa():
    files = {p.stem for p in (CONFIG_DIR / "methods").glob("*.yaml")}
    assert files == set(METHODS.names())


@pytest.mark.parametrize("method", METHODS.names())
def test_method_config_is_well_formed(method):
    doc = method_config(method)
    sweep = doc["sweep"]
    assert sweep["method"] in ("bayes", "grid", "random")
    assert sweep["metric"]["goal"] in ("maximize", "minimize")
    for name, param in sweep["parameters"].items():
        assert isinstance(param, dict) and param, name
    if "monitor" in doc["defaults"]:
        # the metric the sweep optimises must be the one early stopping monitors
        assert doc["defaults"]["monitor"] == sweep["metric"]["name"]


def test_build_sweep_merges_dataset_metadata():
    ds = get_dataset("squirrel")
    sweep = build_sweep("credal_LJ_dual_head_detached", "squirrel", ds.sweep_metadata())
    assert sweep["name"] == "squirrel_credal_LJ_dual_head_detached"
    assert sweep["parameters"]["in_channels"] == {"values": [2089]}
    assert sweep["parameters"]["out_channels"] == {"values": [3]}
    # the stored config is not mutated
    assert "in_channels" not in method_config("credal_LJ_dual_head_detached")["sweep"]["parameters"]


def test_resolution_order():
    ds = get_dataset("reddit2")
    cfg = resolve_config("credal_LJ_dual_head_detached", ds.default_config(), {"lr": 0.123, "batch_size": 7})
    assert cfg["lr"] == 0.123 and cfg["batch_size"] == 7  # overrides win
    assert cfg["out_channels"] == 30 and cfg["num_neighbors"] == 8  # dataset metadata
    assert cfg["monitor"] == "val_auroc_EU"  # method defaults
    assert (
        cfg["split_seed"] is None and cfg["auroc_impl"] == "torchmetrics"
    )  # global defaults (paper protocol)


def test_seed_means_backbone_rank_for_some_posthoc_methods():
    cfg = resolve_config("energy", {}, {"seed": 3}, seed_is_backbone_rank=True)
    assert cfg["backbone_rank"] == 3 and cfg["seed"] == 42
    cfg = resolve_config("energy", {}, {"seed": 3, "backbone_rank": 1}, seed_is_backbone_rank=True)
    assert cfg["backbone_rank"] == 1 and cfg["seed"] == 3


@pytest.mark.parametrize("devices", [3, 8, "auto", True])
def test_gpu_cap(devices):
    with pytest.raises(ValueError):
        validate_hardware({"devices": devices})
    validate_hardware({"devices": 2})


def test_parse_overrides_types():
    out = parse_overrides(["lr=1e-3", "gnn_type=SAGE", "split_seed=null", "flag=true", "n=3", "x=null"])
    assert out == {"lr": 1e-3, "gnn_type": "SAGE", "split_seed": None, "flag": True, "n": 3, "x": None}
    with pytest.raises(ValueError):
        parse_overrides(["oops"])
