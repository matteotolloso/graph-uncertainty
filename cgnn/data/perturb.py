"""Controlled perturbations for the uncertainty-disentanglement experiments.

Training labels (``perturb_training_labels``); config keys (defaults = no change, identical data and split
fingerprint):

- ``train_fraction`` (1.0): keep a random subset of this fraction of the ID training nodes;
- ``label_noise`` (0.0): give this fraction of the (kept) training nodes a uniformly random *other* ID class;
- ``perturb_seed`` (0): seed of both draws (own ``torch.Generator``, global RNG untouched).

Validation and test labels are never changed. Expected behaviour of a well-disentangled model: less
training data -> higher epistemic uncertainty (EU); noisy labels -> higher aleatoric uncertainty (AU).

Test-time feature shift (``shift_test_features``, used by the runner after the normal test when
``test_feature_noise`` is a non-empty list of noise levels): a fraction of the ID test nodes gets
Gaussian noise on its features. Training, validation and model selection never see it.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch_geometric.data import Data


def perturb_training_labels(data: Data, cfg: Mapping[str, Any]) -> str:
    """Apply ``train_fraction``/``label_noise`` in place; return a fingerprint tag ('' if nothing changed)."""
    fraction = float(cfg.get("train_fraction", 1.0))
    noise = float(cfg.get("label_noise", 0.0))
    seed = int(cfg.get("perturb_seed", 0))
    if not (0.0 < fraction <= 1.0) or not (0.0 <= noise < 1.0):
        raise ValueError(f"need 0 < train_fraction <= 1 and 0 <= label_noise < 1, got {fraction}, {noise}")
    if fraction == 1.0 and noise == 0.0:
        return ""

    g = torch.Generator().manual_seed(seed)
    train_idx = data.train_mask.nonzero().view(-1)
    if fraction < 1.0:
        keep = max(1, round(fraction * train_idx.numel()))
        train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)[:keep]].sort().values
        mask = torch.zeros_like(data.train_mask)
        mask[train_idx] = True
        data.train_mask = mask
    if noise > 0.0:
        num_classes = data.y.size(1)
        n = round(noise * train_idx.numel())
        noisy = train_idx[torch.randperm(train_idx.numel(), generator=g)[:n]]
        shift = torch.randint(1, num_classes, (n,), generator=g)
        new_labels = (data.y[noisy].argmax(dim=1) + shift) % num_classes
        y = data.y.clone()
        y[noisy] = 0.0
        y[noisy, new_labels] = 1.0
        data.y = y
    return f"frac={fraction:g},noise={noise:g},seed={seed}"


def shift_test_features(data: Data, sigma: float, fraction: float = 0.5, seed: int = 0) -> Data:
    """Copy of ``data`` whose test set is "clean vs feature-shifted ID nodes".

    A random ``fraction`` of the ID test nodes gets ``x + sigma * std * eps`` (``std``: per-feature std over
    the training nodes, ``eps ~ N(0, 1)``); these nodes become the positive class (all-zero label row, like
    OOD nodes) and the real OOD nodes leave the test mask. So ``test_auroc*`` measures the detection of
    shifted nodes and ``test_mean*_id`` / ``test_mean*_ood`` the mean score on clean / shifted nodes.
    Nodes and ``eps`` depend only on ``seed``: every ``sigma`` perturbs the same nodes in the same direction
    (``sigma = 0`` is the no-shift control, AUROC ~ 0.5 up to neighbourhood effects). Train/val masks,
    labels and features of all other nodes are unchanged (the shift still reaches their neighbours through
    message passing, as in any feature-perturbation shift on a graph).
    """
    if sigma < 0 or not (0.0 < fraction < 1.0):
        raise ValueError(f"need sigma >= 0 and 0 < fraction < 1, got {sigma}, {fraction}")
    is_id = data.y.sum(dim=1) == 1
    test_id = (data.test_mask & is_id).nonzero().view(-1)
    g = torch.Generator().manual_seed(seed)
    n = max(1, round(fraction * test_id.numel()))
    shifted = test_id[torch.randperm(test_id.numel(), generator=g)[:n]].sort().values
    eps = torch.randn((n, data.x.size(1)), generator=g)
    std = data.x[data.train_mask].float().std(dim=0)

    out = copy.copy(data)  # shallow: edge_index etc. shared, x/y/test_mask replaced below
    x = data.x.clone()
    x[shifted] = x[shifted] + (sigma * std * eps).to(x.dtype)
    y = data.y.clone()
    y[shifted] = 0
    test_mask = data.test_mask & is_id
    out.x, out.y, out.test_mask = x, y, test_mask
    return out
