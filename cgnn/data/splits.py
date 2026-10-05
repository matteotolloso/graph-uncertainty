"""Node splits and split fingerprints.

Random splits (arxiv, patents, coauthor) are drawn from a *dedicated*
``torch.Generator`` seeded with ``split_seed``, so every method sees the same
split regardless of how much global RNG it consumed before loading data.

``split_seed=None`` reproduces the legacy (pre-0.2) behaviour: the permutation
used the *global* RNG right after model construction, so the split depended
on the architecture and post-hoc methods silently evaluated on a different
split than their backbone was trained on (see docs/known_issues.md).
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch


def random_split_masks(
    num_nodes: int, train_ratio: float, val_ratio: float, split_seed: int | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if split_seed is None:
        indices = torch.randperm(num_nodes)  # legacy: global RNG
    else:
        generator = torch.Generator().manual_seed(int(split_seed))
        indices = torch.randperm(num_nodes, generator=generator)

    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size : train_size + val_size]] = True
    test_mask[indices[train_size + val_size :]] = True
    return train_mask, val_mask, test_mask


def split_fingerprint(
    train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor, tag: str = ""
) -> str:
    """Short, stable hash of the three node masks (stored in checkpoints).

    ``tag`` describes training-label perturbations (``cgnn.data.perturb``); empty = legacy hash.
    """
    h = hashlib.sha1()
    for mask in (train_mask, val_mask, test_mask):
        h.update(np.packbits(mask.detach().cpu().numpy().astype(np.uint8)).tobytes())
    if tag:
        h.update(tag.encode())
    return h.hexdigest()[:12]
