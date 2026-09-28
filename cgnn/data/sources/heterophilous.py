"""Amazon-Ratings and Roman-Empire (Platonov et al. 2023), public split 0 from the ``.npz``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cgnn.data.spec import DatasetSpec, RawGraph, register_dataset


def _load_platonov_npz(data_dir: Path, spec: DatasetSpec) -> RawGraph:
    z = np.load(data_dir / spec.extra["npz"], allow_pickle=True)
    x_np = np.asarray(z["node_features"], dtype=np.float32)  # (N, F)
    y_np = np.asarray(z["node_labels"]).reshape(-1).astype(np.int64)  # (N,)
    edges_np = np.asarray(z["edges"], dtype=np.int64)  # (E, 2)
    train_masks = np.asarray(z["train_masks"], dtype=bool)  # (S, N)
    val_masks = np.asarray(z["val_masks"], dtype=bool)
    test_masks = np.asarray(z["test_masks"], dtype=bool)

    N = x_np.shape[0]
    i = spec.public_split_index
    assert y_np.shape[0] == N and edges_np.ndim == 2 and edges_np.shape[1] == 2
    assert train_masks.shape[1] == N and val_masks.shape[1] == N and test_masks.shape[1] == N
    assert 0 <= i < train_masks.shape[0], "public_split_index out of range"

    return RawGraph(
        x=torch.from_numpy(x_np).float(),
        edge_index=torch.from_numpy(edges_np.T).long(),  # (E,2) -> (2,E)
        y=torch.from_numpy(y_np).long(),
        train_mask=torch.from_numpy(train_masks[i]).bool(),
        val_mask=torch.from_numpy(val_masks[i]).bool(),
        test_mask=torch.from_numpy(test_masks[i]).bool(),
    )


register_dataset(
    DatasetSpec(
        name="amazon_ratings",
        files=("amazon_ratings/amazon_ratings.npz",),
        paper_name="Amazon-Ratings",
        load_raw=_load_platonov_npz,
        id_classes=(2, 3, 4),
        ood_classes=(0, 1),
        num_features=300,
        homophily="heterophilic",
        split="public",
        extra={"npz": "amazon_ratings/amazon_ratings.npz"},
    )
)

register_dataset(
    DatasetSpec(
        name="roman_empire",
        files=("roman_empire/raw/roman_empire.npz",),
        paper_name="Roman Empire",
        load_raw=_load_platonov_npz,
        id_classes=tuple(range(5, 18)),
        ood_classes=tuple(range(0, 5)),
        num_features=300,
        homophily="heterophilic",
        split="public",
        extra={"npz": "roman_empire/raw/roman_empire.npz"},
    )
)
