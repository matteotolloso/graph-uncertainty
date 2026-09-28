"""Model-level behaviour: shapes, gradient flow, checkpoint compatibility."""

from __future__ import annotations

import pytest
import torch

from cgnn.data import load_dataset
from cgnn.models import CredalGNN, CredalLayer, VanillaGNN
from cgnn.paths import REPO_ROOT


@pytest.fixture(scope="module")
def batch():
    return load_dataset("synthetic", {"split_seed": 0}).data


def _credal(**kw):
    base = dict(gnn_type="GCN", in_channels=16, hidden_channels=8, num_layers=2, out_channels=3, delta=0.5)
    return CredalGNN(**{**base, **kw})


@pytest.mark.parametrize(
    "kw,dim",
    [
        (dict(latent="joint", heads="dual"), 16),
        (dict(latent="joint", heads="dual", joint_include_input=True), 32),
        (dict(latent="last", heads="dual"), 8),
        (dict(latent="joint", heads="credal"), 16),
        (dict(latent="last", heads="credal"), 8),
    ],
)
def test_credal_variants_forward(batch, kw, dim):
    model = _credal(**kw)
    assert model.credal_layer_model.input_dim == dim
    out = model(batch)
    q_L, q_U = out[0], out[1]
    assert q_L.shape == q_U.shape == (batch.num_nodes, 3)
    assert torch.all(q_L <= q_U + 1e-7)
    assert len(out) == (3 if kw["heads"] == "dual" else 2)


def test_detached_credal_head_does_not_train_backbone(batch):
    model = _credal(latent="joint", heads="dual", detach_credal=True)
    q_L, q_U, _ = model(batch)
    model.criterion(q_L, q_U, batch.y).backward()
    backbone_grads = [p.grad for p in model.gnn_model.parameters()]
    assert all(g is None or torch.count_nonzero(g) == 0 for g in backbone_grads)
    assert any(p.grad is not None for p in model.credal_layer_model.parameters())


def test_attached_credal_head_trains_backbone(batch):
    model = _credal(latent="joint", heads="dual", detach_credal=False)
    q_L, q_U, _ = model(batch)
    model.criterion(q_L, q_U, batch.y).backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.gnn_model.parameters())


def test_invalid_credal_configurations():
    with pytest.raises(ValueError):
        _credal(heads="credal", detach_credal=True)
    with pytest.raises(ValueError):
        _credal(heads="credal", lambda_cons=0.1)
    with pytest.raises(ValueError):
        _credal(latent="middle")


@pytest.mark.parametrize(
    "mid,half", [("sigmoid", "sigmoid"), ("identity", "softplus"), ("identity", "sigmoid")]
)
def test_credal_layer_activations(mid, half):
    layer = CredalLayer(8, 30, mid_activation=mid, half_activation=half)
    q_L, q_U = layer(torch.randn(50, 8) * 10)
    assert torch.all(q_L <= q_U + 1e-7)
    if mid == "sigmoid" and half == "sigmoid":  # documented cap: q_U < e^2 / (e^2 + C - 1)
        cap = torch.exp(torch.tensor(2.0)) / (torch.exp(torch.tensor(2.0)) + 29)
        assert torch.all(q_U < cap)


def test_credal_layer_rejects_signed_half_width():
    with pytest.raises(ValueError):
        CredalLayer(8, 3, half_activation="identity")


def test_legacy_vanilla_checkpoint_loads():
    ckpts = sorted((REPO_ROOT / "checkpoints").glob("*_squirrel_val_auroc=*.ckpt"))
    if not ckpts:
        pytest.skip("no legacy squirrel checkpoints in ./checkpoints")
    model = VanillaGNN.load_from_checkpoint(ckpts[0], map_location="cpu")
    assert model.C == 3 and hasattr(model, "gnn_model")
