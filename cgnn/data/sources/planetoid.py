"""Cora (Planetoid, ``split="full"``). Not in the paper; kept for quick checks."""

from __future__ import annotations

from pathlib import Path

from cgnn.data.spec import DatasetSpec, RawGraph, register_dataset


def _load_cora(data_dir: Path, spec: DatasetSpec) -> RawGraph:
    from torch_geometric.datasets import Planetoid

    data = Planetoid(root=str(data_dir), name="Cora", split="full")[0]
    return RawGraph(
        x=data.x,
        edge_index=data.edge_index,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
    )


register_dataset(
    DatasetSpec(
        name="cora",
        files=("Cora",),
        paper_name="Cora",
        load_raw=_load_cora,
        id_classes=(4, 5, 6),
        ood_classes=(0, 1, 2, 3),
        num_features=1433,
        homophily="homophilic",
        split="public",
        in_paper=False,
    )
)
