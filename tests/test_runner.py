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
    "graph_esn": {"num_reservoirs": 2, "hidden_channels": 16, "max_iterations": 20, "max_epochs": 1},
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
        suffix = f"_{key}" if key else ""
        for extra in (
            "test_aupr",
            "test_fpr95",
            "test_auprin",
            "test_fpr95in",
            "test_misc_auroc",
        ):  # log_test_extras
            assert 0.0 <= results[f"{extra}{suffix}"] <= 1.0, f"{method} did not log {extra}{suffix}"
        assert f"test_mean{suffix}_id" in results and f"test_mean{suffix}_ood" in results


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
            {**FAST, "split_seed": 1},
            logger=False,
            paths=backbone_paths,
            write_summary=False,
        )
    # explicit opt-out still works
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


@pytest.mark.skipif(not has_module("faiss"), reason="faiss not installed")
def test_knn_torch_backend_matches_faiss(backbone_paths):
    import faiss
    import numpy as np
    import torch.nn.functional as F

    from cgnn.models.detectors import KNNDetector

    det = KNNDetector(
        str(find_checkpoints("synthetic", paths=backbone_paths)[0].path), k=7, knn_backend="torch"
    )
    det.TORCH_CHUNK_ELEMENTS = 5 * 300  # several query chunks
    g = torch.Generator().manual_seed(0)
    train = F.normalize(torch.randn(300, 16, generator=g), dim=1)
    query = F.normalize(torch.randn(101, 16, generator=g), dim=1)
    det.train_emb = train
    index = faiss.IndexFlatL2(16)
    index.add(train.numpy().astype(np.float32))
    expected = torch.from_numpy(index.search(query.numpy(), 7)[0][:, -1])
    torch.testing.assert_close(det._kth_distance_torch(query), expected, atol=1e-5, rtol=0)


@pytest.mark.skipif(not has_module("faiss"), reason="faiss not installed")
@pytest.mark.parametrize("method", ["knn", "knn_LJ"])
def test_knn_backends_give_same_auroc(method, backbone_paths):
    out = {
        backend: run_experiment(
            method,
            "synthetic",
            {**FAST, "knn_backend": backend},
            logger=False,
            paths=backbone_paths,
            write_summary=False,
        )
        for backend in ("faiss", "torch")
    }
    for key in ("val_auroc", "test_auroc"):
        assert out["torch"][key] == pytest.approx(out["faiss"][key], abs=1e-6)


@pytest.mark.parametrize("method", ["credal_LJ_dual_head_detached", "ensemble"])
def test_feature_shift_tests_leave_normal_test_unchanged(method, backbone_paths):
    cfg = {**FAST, **PER_METHOD.get(method, {}), "seed": 0}
    plain = run_experiment(method, "synthetic", cfg, logger=False, run_id="fs0", paths=backbone_paths)
    shift = run_experiment(
        method,
        "synthetic",
        {**cfg, "test_feature_noise": [0, 4]},
        logger=False,
        run_id="fs1",
        paths=backbone_paths,
    )
    for key, value in plain.items():  # the normal metrics are untouched
        if key.startswith("test_"):
            assert shift[key] == pytest.approx(value, nan_ok=True), key
    keys = [k for k in get_method(method).score_keys]
    for tag in ("fshift_0", "fshift_4"):
        for key in keys:
            name = f"{tag}_test_auroc_{key}" if key else f"{tag}_test_auroc"
            assert 0.0 <= shift[name] <= 1.0, name
    first = f"_{keys[0]}" if keys[0] else ""
    assert f"fshift_4_test_mean{first}_ood" in shift and not any(k.startswith("fshift_0p") for k in shift)
    assert not any(k.startswith("fshift_") and ("acc" in k or "misc" in k or "f1" in k) for k in shift)
