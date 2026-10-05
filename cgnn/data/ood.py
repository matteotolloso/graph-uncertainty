"""Leave-out-class OOD protocol (paper "OOD Detection Setting").

Given a graph with original labels and base train/val/test node masks:

- ``train_mask = base_train & ID``  (OOD labels are never seen in training)
- ``val_mask   = base_val``          (ID + OOD, used for model selection)
- ``test_mask  = base_test``         (ID + OOD, final evaluation)
- ``y``: ``[N, C_id]`` one-hot over the (sorted, re-indexed) ID classes and the
  **all-zero row for OOD nodes**. Throughout the code ``y.sum(1) == 1`` means
  ID and ``1 - y.sum(1)`` is the OOD target (1 = OOD) for AUROC.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import torch
from torch_geometric.data import Data

from cgnn.data.labels import one_hot_encode


def apply_leave_out_classes(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    base_train: torch.Tensor,
    base_val: torch.Tensor,
    base_test: torch.Tensor,
    id_classes: Sequence[int],
    ood_classes: Sequence[int],
) -> Data:
    if set(id_classes) & set(ood_classes):
        raise ValueError(f"ID and OOD classes overlap: {sorted(set(id_classes) & set(ood_classes))}")

    id_node_mask = torch.isin(y, torch.tensor(list(id_classes), dtype=y.dtype))
    ood_node_mask = torch.isin(y, torch.tensor(list(ood_classes), dtype=y.dtype))

    unassigned = ~(id_node_mask | ood_node_mask)
    if unassigned.any():
        # Nodes whose label is neither ID nor OOD (e.g. unlabeled) would get an
        # all-zero row and be scored as OOD. Drop them from every split instead.
        warnings.warn(
            f"{int(unassigned.sum())} nodes have labels outside ID u OOD; removing them from all splits.",
            stacklevel=2,
        )
        base_train, base_val, base_test = (m & ~unassigned for m in (base_train, base_val, base_test))

    num_id = len(id_classes)
    remap = torch.full((int(y.max()) + 1,), -1, dtype=torch.long)
    for new_label, original in enumerate(sorted(id_classes)):
        remap[original] = new_label

    new_y = torch.zeros((y.size(0), num_id), dtype=torch.float)
    new_y[id_node_mask] = one_hot_encode(remap[y[id_node_mask]], num_id)

    data = Data(x=x, edge_index=edge_index, y=new_y)
    data.train_mask = base_train & id_node_mask
    data.val_mask = base_val
    data.test_mask = base_test
    return data


def split_summary(data: Data) -> dict[str, int]:
    """Node counts per split (the numbers in paper Table 1)."""
    is_id = data.y.sum(dim=1) == 1
    out = {"train_id": int(data.train_mask.sum())}
    for split in ("val", "test"):
        mask = getattr(data, f"{split}_mask")
        out[f"{split}_id"] = int((mask & is_id).sum())
        out[f"{split}_ood"] = int((mask & ~is_id).sum())
    return out
