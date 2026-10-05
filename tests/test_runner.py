"""End-to-end runs of every registered method on the synthetic graph (CPU, a few epochs).

This is the contract test for new methods: each one must run through
``cgnn.runner.run_experiment`` and log the metric its sweep optimises plus
test AUROCs for every declared score key.
"""

from __future__ import annotations

import json

import pytest
import torch

from cgnn.checkpoints import find_checkpoints
from cgnn.config import method_config
from cgnn.methods import METHODS, get_method
from cgnn.runner import run_experiment
from tests.conftest import FAST, has_module

TRAINABLE = [m for m in METHODS.names() if get_method(m).kind == "trainable"]
POSTHOC = [m for m in METHODS.names() if get_method(m).kind == "posthoc"]

# cheap settings for methods whose defaults are expensive
PER_METHOD = {
    "ensemble": {"M": 3},
    "cagcn": {"max_epochs": 3},
}


def _skip_if_missing_deps(method: str) -> None:
    for req in get_method(method).requires:
        if not has_module(req):
            pytest.skip(f"{method} needs optional dependency '{req}'")


def _assert_contract(method: str, results: dict[str, float]) -> None:
    sweep_metric = method_config(method)["sweep"]["metric"]["name"]
    assert sweep_metric in results, f"{method} does not log its sweep metric {sweep_metric}"
    for key in get_method(method).score_keys:
        name = f"test_auroc_{key}" if key else "test_auroc"
        assert name in results, f"{method} did not log {name}"
        assert 0.0 <= results[name] <= 1.0


@pytest.mark.parametrize("method", TRAINABLE)
def test_trainable_methods_run(method, tmp_paths):
    _skip_if_missing_deps(method)
    results = run_experiment(
        method,
        "synthetic",
        {**FAST, **PER_METHOD.get(method, {})},
        logger=False,
        run_id="t0",
        paths=tmp_paths,
    )
    _assert_contract(method, results)
    summaries = list((tmp_paths.output_dir / "runs").glob("*.json"))
    assert summaries and json.loads(summaries[0].read_text())["method"] == method


@pytest.mark.parametrize("method", POSTHOC)
def test_posthoc_methods_run(method, backbone_paths):
    _skip_if_missing_deps(method)
    results = run_experiment(
        method,
        "synthetic",
        {**FAST, **PER_METHOD.get(method, {})},
        logger=False,
        run_id="p0",
        paths=backbone_paths,
        write_summary=False,
    )
    _assert_contract(method, results)


def test_vanilla_checkpoints_carry_split_metadata(backbone_paths):
    records = find_checkpoints("synthetic", paths=backbone_paths)
    assert len(records) == 3
    assert records == sorted(records, key=lambda r: (-r.score, r.path.name))
    meta = records[0].meta
    assert meta["dataset"] == "synthetic" and meta["split_seed"] == 0 and len(meta["split_fingerprint"]) == 12


def test_backbone_split_mismatch_is_refused(backbone_paths):
    with pytest.raises(RuntimeError, match="split fingerprint"):
        run_experiment(
            "energy",
            "synthetic",
            {**FAST, "split_seed": 1, "backbone_split_check": "error"},
            logger=False,
            paths=backbone_paths,
            write_summary=False,
        )
    # warn (default) proceeds
    results = run_experiment(
        "energy",
        "synthetic",
        {**FAST, "split_seed": 1, "backbone_split_check": "warn"},
        logger=False,
        paths=backbone_paths,
        write_summary=False,
    )
    assert "test_auroc" in results


def test_seeded_training_is_reproducible(tmp_paths):
    cfg = {**FAST, "save_checkpoint": False}
    a = run_experiment(
        "credal_LJ_dual_head_detached", "synthetic", cfg, logger=False, paths=tmp_paths, write_summary=False
    )
    b = run_experiment(
        "credal_LJ_dual_head_detached", "synthetic", cfg, logger=False, paths=tmp_paths, write_summary=False
    )
    assert a == b


def test_test_on_best_checkpoint(tmp_paths):
    results = run_experiment(
        "vanilla",
        "synthetic",
        {**FAST, "test_ckpt": "best"},
        logger=False,
        paths=tmp_paths,
        write_summary=False,
    )
    assert "test_auroc" in results
    with pytest.raises(ValueError):
        run_experiment(
            "vanilla",
            "synthetic",
            {**FAST, "test_ckpt": "best", "save_checkpoint": False},
            logger=False,
            paths=tmp_paths,
            write_summary=False,
        )


def test_runner_refuses_more_than_two_gpus(tmp_paths):
    with pytest.raises(ValueError, match="limit"):
        run_experiment("vanilla", "synthetic", {**FAST, "devices": 3}, logger=False, paths=tmp_paths)


def test_missing_backbones_error_is_actionable(tmp_paths):
    with pytest.raises(FileNotFoundError, match="cgnn sweep -m vanilla"):
        run_experiment("energy", "synthetic", FAST, logger=False, paths=tmp_paths)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU visible")
def test_gpu_smoke(tmp_paths):  # pragma: no cover - only with CUDA_VISIBLE_DEVICES set
    results = run_experiment(
        "credal_LJ_dual_head_detached",
        "synthetic",
        {**FAST, "accelerator": "gpu"},
        logger=False,
        paths=tmp_paths,
        write_summary=False,
    )
    assert "test_auroc_EU" in results
