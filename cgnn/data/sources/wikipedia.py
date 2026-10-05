"""Squirrel (WikipediaNetwork, Rozemberczki et al. 2021).

Uses the geom-gcn preprocessing shipped with PyG and its public split
``public_split_index`` (of 10). Note: these graphs contain many duplicated
nodes (Platonov et al. 2023); the filtered versions are not used here.
"""

from __future__ import annotations

from pathlib import Path

from cgnn.data.spec import DatasetSpec, RawGraph, register_dataset


def _load_wikipedia(data_dir: Path, spec: DatasetSpec) -> RawGraph:
    from torch_geometric.datasets import WikipediaNetwork

    data = WikipediaNetwork(root=str(data_dir), name=spec.extra["pyg_name"])[0]
    i = spec.public_split_index
    return RawGraph(
        x=data.x,
        edge_index=data.edge_index,
        y=data.y,
        train_mask=data.train_mask[:, i],
        val_mask=data.val_mask[:, i],
        test_mask=data.test_mask[:, i],
    )


register_dataset(
    DatasetSpec(
        name="squirrel",
        files=("squirrel/geom_gcn",),
        paper_name="Squirrel",
        load_raw=_load_wikipedia,
        id_classes=(2, 3, 4),
        ood_classes=(0, 1),
        num_features=2089,
        homophily="heterophilic",
        split="public",
        extra={"pyg_name": "squirrel"},
    )
)
