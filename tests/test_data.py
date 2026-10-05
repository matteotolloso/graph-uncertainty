"""Leave-out-class OOD protocol, splits, fingerprints, dataset registry."""

from __future__ import annotations

import pytest
import torch

from cgnn.data import DATASETS, load_dataset
from cgnn.data.ood import apply_leave_out_classes, split_summary
from cgnn.data.splits import random_split_masks, split_fingerprint
from cgnn.paths import Paths


def _toy(labels):
    y = torch.tensor(labels)
    n = len(y)
    x = torch.randn(n, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    ones = torch.ones(n, dtype=torch.bool)
    return x, edge_index, y, ones


def test_leave_out_classes_masks_and_labels():
    x, ei, y, ones = _toy([0, 1, 2, 3, 4, 2, 0])
    data = apply_leave_out_classes(x, ei, y, ones, ones, ones, id_classes=(2, 3, 4), ood_classes=(0, 1))
    is_ood = torch.isin(y, torch.tensor([0, 1]))
    assert torch.equal(data.train_mask, ~is_ood)  # OOD never in training
    assert torch.equal(data.val_mask, ones) and torch.equal(data.test_mask, ones)
    assert torch.all(data.y[is_ood] == 0)  # OOD -> all-zero row
    assert torch.equal(data.y[~is_ood].argmax(1), torch.tensor([0, 1, 2, 0]))  # remapped 2,3,4 -> 0,1,2


def test_leave_out_classes_non_contiguous_ids():
    x, ei, y, ones = _toy([5, 1, 9, 5, 3])
    data = apply_leave_out_classes(x, ei, y, ones, ones, ones, id_classes=(9, 5), ood_classes=(1, 3))
    assert data.y.shape == (5, 2)
    assert torch.equal(data.y[[0, 2, 3]].argmax(1), torch.tensor([0, 1, 0]))  # sorted: 5 -> 0, 9 -> 1


def test_leave_out_classes_drops_unassigned_labels():
    x, ei, y, ones = _toy([0, 2, 7])
    with pytest.warns(UserWarning, match="outside ID u OOD"):
        data = apply_leave_out_classes(x, ei, y, ones, ones, ones, id_classes=(2,), ood_classes=(0,))
    assert not data.val_mask[2] and not data.test_mask[2] and not data.train_mask[2]


def test_leave_out_classes_rejects_overlap():
    x, ei, y, ones = _toy([0, 1])
    with pytest.raises(ValueError):
        apply_leave_out_classes(x, ei, y, ones, ones, ones, id_classes=(0, 1), ood_classes=(1,))


def test_random_split_is_deterministic_disjoint_and_complete():
    a = random_split_masks(1000, 0.6, 0.2, split_seed=3)
    b = random_split_masks(1000, 0.6, 0.2, split_seed=3)
    c = random_split_masks(1000, 0.6, 0.2, split_seed=4)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    assert not torch.equal(a[0], c[0])
    tr, va, te = a
    assert (tr.int() + va.int() + te.int()).eq(1).all()
    assert (int(tr.sum()), int(va.sum()), int(te.sum())) == (600, 200, 200)


def test_seeded_split_ignores_global_rng():
    torch.manual_seed(0)
    a = random_split_masks(100, 0.6, 0.2, split_seed=1)
    torch.manual_seed(999)
    b = random_split_masks(100, 0.6, 0.2, split_seed=1)
    assert torch.equal(a[0], b[0])


def test_fingerprint_is_sensitive_to_masks():
    tr, va, te = random_split_masks(100, 0.6, 0.2, split_seed=0)
    fp = split_fingerprint(tr, va, te)
    assert fp == split_fingerprint(tr.clone(), va.clone(), te.clone())
    tr2 = tr.clone()
    tr2[0] = ~tr2[0]
    assert fp != split_fingerprint(tr2, va, te)


def test_synthetic_dataset_bundle():
    b = load_dataset("synthetic", {"split_seed": 0}, use_cache=False)
    s = b.summary()
    assert s["num_nodes"] == 240 and s["split_seed"] == 0
    assert s["train_id"] > 0 and s["val_ood"] > 0 and s["test_ood"] > 0
    assert b.fingerprint != load_dataset("synthetic", {"split_seed": 1}, use_cache=False).fingerprint
    legacy = load_dataset("synthetic", {"split_seed": "legacy"}, use_cache=False)
    assert legacy.split_seed is None


def test_registry_specs_are_consistent():
    for name in DATASETS:
        spec = DATASETS.get(name)
        assert not set(spec.id_classes) & set(spec.ood_classes), name
        assert spec.sweep_metadata()["out_channels"]["values"] == [len(spec.id_classes)]
        assert spec.split in ("public", "random")


@pytest.mark.slow
@pytest.mark.parametrize("name", ["squirrel", "amazon_ratings", "roman_empire", "cora", "chameleon"])
def test_real_small_datasets_match_their_spec(name):
    spec = DATASETS.get(name)
    if spec.missing_files(Paths.from_env().data_dir):
        pytest.skip(f"{name} not downloaded")
    b = load_dataset(name, use_cache=False)
    assert b.data.x.size(1) == spec.num_features
    assert b.data.y.size(1) == spec.num_id_classes
    assert split_summary(b.data)["train_id"] > 0


def test_training_label_perturbations():
    clean = load_dataset("synthetic", {})
    same = load_dataset("synthetic", {"train_fraction": 1.0, "label_noise": 0.0})
    assert same.fingerprint == clean.fingerprint and torch.equal(same.data.y, clean.data.y)

    few = load_dataset("synthetic", {"train_fraction": 0.5})
    n_train = int(clean.data.train_mask.sum())
    assert int(few.data.train_mask.sum()) == round(0.5 * n_train)
    assert not (few.data.train_mask & ~clean.data.train_mask).any() and few.fingerprint != clean.fingerprint

    noisy = load_dataset("synthetic", {"label_noise": 0.4})
    changed = (noisy.data.y != clean.data.y).any(dim=1)
    assert int(changed.sum()) == round(0.4 * n_train) and not (changed & ~clean.data.train_mask).any()
    assert torch.all(noisy.data.y.sum(dim=1)[clean.data.train_mask] == 1)  # still one ID class each
    assert (
        torch.equal(noisy.data.train_mask, clean.data.train_mask) and noisy.fingerprint != clean.fingerprint
    )
    again = load_dataset("synthetic", {"label_noise": 0.4})
    assert torch.equal(again.data.y, noisy.data.y)  # perturb_seed makes it deterministic


def test_csbm_controls_homophily_only():
    from cgnn.data.sources.csbm import make_csbm_graph

    low, high = make_csbm_graph(0.1), make_csbm_graph(0.9)
    assert torch.equal(low.x, high.x) and torch.equal(low.y, high.y)
    for graph, h in ((low, 0.1), (high, 0.9)):
        src, dst = graph.edge_index
        assert abs((graph.y[src] == graph.y[dst]).float().mean().item() - h) < 0.02
        assert not (src == dst).any()
    assert torch.equal(make_csbm_graph(0.5).edge_index, make_csbm_graph(0.5).edge_index)
    assert "csbm_h5" in DATASETS.names()


def test_test_feature_shift():
    from cgnn.data.perturb import shift_test_features

    data = load_dataset("synthetic", {}).data
    before = (data.x.clone(), data.y.clone(), data.test_mask.clone())
    shifted = shift_test_features(data, 2.0, fraction=0.5, seed=0)
    assert all(torch.equal(a, b) for a, b in zip(before, (data.x, data.y, data.test_mask), strict=True))

    is_id = data.y.sum(dim=1) == 1
    changed = (shifted.x != data.x).any(dim=1)
    assert torch.equal(shifted.test_mask, data.test_mask & is_id)  # real OOD nodes leave the test set
    assert torch.equal(changed, shifted.y.sum(dim=1) != data.y.sum(dim=1))  # shifted nodes = positives
    assert int(changed.sum()) == round(0.5 * int((data.test_mask & is_id).sum()))
    assert not (changed & ~shifted.test_mask).any()
    assert torch.equal(shifted.train_mask, data.train_mask) and torch.equal(shifted.val_mask, data.val_mask)

    control = shift_test_features(data, 0.0)
    assert torch.equal(control.x, data.x) and torch.equal(control.y, shifted.y)  # same nodes at every sigma
    weak = shift_test_features(data, 1.0)
    assert torch.allclose(2 * (weak.x - data.x), shifted.x - data.x, atol=1e-5)  # same direction, scaled
