"""DataLoaders for transductive node classification.

- Full batch (``batch_size <= 0``): each loader yields the whole graph once;
  models select nodes with ``data.{train,val,test}_mask``.
- Mini batch (``batch_size > 0``): ``NeighborLoader`` over the split's nodes.
  In a sampled batch only the first ``batch.batch_size`` nodes are seeds; models
  must use ``NodeUQModule.split_mask`` which handles both cases.
"""

from __future__ import annotations

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, NeighborLoader


def full_batch_loader(data: Data) -> DataLoader:
    return DataLoader([data], batch_size=1, shuffle=False)


def neighbor_loader(
    data: Data, split: str, batch_size: int, num_neighbors: int, num_layers: int, shuffle: bool
):
    return NeighborLoader(
        data,
        input_nodes=getattr(data, f"{split}_mask"),
        batch_size=batch_size,
        num_neighbors=[num_neighbors] * num_layers,
        shuffle=shuffle,
    )


def build_loaders(
    data: Data,
    *,
    batch_size: int,
    num_neighbors: int,
    num_layers: int,
    eval_neighbor_sampling: bool,
):
    """Return ``(train_loader, val_loader, test_loader)``."""
    if batch_size <= 0:
        train_loader = full_batch_loader(data)
    else:
        train_loader = neighbor_loader(data, "train", batch_size, num_neighbors, num_layers, shuffle=True)

    if batch_size > 0 and eval_neighbor_sampling:
        val_loader = neighbor_loader(data, "val", batch_size, num_neighbors, num_layers, shuffle=False)
        test_loader = neighbor_loader(data, "test", batch_size, num_neighbors, num_layers, shuffle=False)
    else:
        val_loader = full_batch_loader(data)
        test_loader = full_batch_loader(data)
    return train_loader, val_loader, test_loader
