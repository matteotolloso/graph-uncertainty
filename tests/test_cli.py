"""CLI smoke tests (no network, no GPU)."""

from __future__ import annotations

import json
import sys

import pytest

from cgnn.cli import main


def test_list_json(capsys):
    assert main(["list", "methods", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert any(r["method"] == "credal_LJ_dual_head_detached" and r["paper_name"] == "CGNN" for r in rows)


def test_describe_method_and_dataset(capsys):
    assert main(["describe", "credal"]) == 0
    assert "CGNN last layer" in capsys.readouterr().out
    assert main(["describe", "reddit2"]) == 0
    assert "eval_neighbor_sampling: true" in capsys.readouterr().out
    assert main(["describe", "nope"]) == 2


def test_run_synthetic_without_wandb(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CGNN_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("CGNN_OUTPUT_DIR", str(tmp_path / "out"))
    code = main(
        [
            "run",
            "-m",
            "credal",
            "-d",
            "synthetic",
            "--cpu",
            "--seeds",
            "0",
            "1",
            "--set",
            "max_epochs=2",
            "progress_bar=false",
            "model_summary=false",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert out.count('"test_auroc_EU"') == 2


def test_legacy_main_shim_builds_a_sweep(monkeypatch):
    calls = {}
    import wandb

    monkeypatch.setattr(
        wandb, "sweep", lambda sweep, project, entity=None: calls.setdefault("sweep", sweep) and "sw1"
    )
    monkeypatch.setattr(wandb, "agent", lambda **kw: calls.setdefault("agent", kw))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    import importlib

    monkeypatch.setattr(
        sys, "argv", ["main.py", "-m", "credal_LJ_dual_head_detached", "-d", "squirrel", "-c", "3"]
    )
    shim = importlib.import_module("main")
    with pytest.raises(SystemExit) as exc:
        shim.main()
    assert exc.value.code == 0
    assert calls["sweep"]["name"] == "squirrel_credal_LJ_dual_head_detached"
    assert calls["agent"]["count"] == 3


def test_sweep_agent_function_runs_with_wandb_disabled(tmp_path, monkeypatch):
    """The function executed by `wandb.agent` resolves the config and runs the experiment."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("CGNN_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("CGNN_OUTPUT_DIR", str(tmp_path / "out"))
    from cgnn.wandb_utils import agent_function

    overrides = {"accelerator": "cpu", "max_epochs": 2, "progress_bar": False, "model_summary": False}
    agent_function("credal", "synthetic", "test-project", overrides)()
    summaries = list((tmp_path / "out" / "runs").glob("*_synthetic_credal_*.json"))
    assert len(summaries) == 1
