"""Label utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def one_hot_encode(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).float()


def even_quantile_labels(vals: np.ndarray, nclasses: int, verbose: bool = True) -> np.ndarray:
    """Partition ``vals`` into ``nclasses`` quantile bins (arxiv-year, patents-year).

    From Lim et al. (2021) / Ma et al. (2024, "Revisiting Score Propagation in
    Graph OOD Detection").
    """
    label = -1 * np.ones(vals.shape[0], dtype=int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print("Class Label Intervals:")
        for class_idx, interval in enumerate(interval_lst):
            print(f"Class {class_idx}: [{interval[0]}, {interval[1]})]")
    return label
