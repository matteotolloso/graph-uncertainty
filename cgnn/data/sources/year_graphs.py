"""ArXiv-year and Patents-year (Lim et al. 2021; Ma et al. 2024).

Labels are 5 publication-year quantile bins. Both use a *random* 60/20/20
split (see ``cgnn.data.splits``). Edges are kept as provided (directed); the
LINKX reference setup symmetrises them.
"""

from __future__ import annotations

from pathlib import Path

import torch

from cgnn.data.labels import even_quantile_labels
from cgnn.data.spec import DatasetSpec, RawGraph, register_dataset


def _load_arxiv_year(data_dir: Path, spec: DatasetSpec) -> RawGraph:
    from ogb.nodeproppred import NodePropPredDataset

    graph = NodePropPredDataset(name="ogbn-arxiv", root=str(data_dir)).graph
    y = torch.tensor(even_quantile_labels(graph["node_year"].flatten(), 5, verbose=True), dtype=torch.long)
    return RawGraph(
        x=torch.as_tensor(graph["node_feat"]),
        edge_index=torch.as_tensor(graph["edge_index"]),
        y=y,
    )


def _load_patents_year(data_dir: Path, spec: DatasetSpec) -> RawGraph:
    import scipy.io

    fulldata = scipy.io.loadmat(f"{data_dir}/snap-patents.mat")
    y = torch.tensor(
        even_quantile_labels(fulldata["years"].flatten(), nclasses=5, verbose=False), dtype=torch.long
    )
    return RawGraph(
        x=torch.tensor(fulldata["node_feat"].todense(), dtype=torch.float),
        edge_index=torch.tensor(fulldata["edge_index"], dtype=torch.long),
        y=y,
    )


register_dataset(
    DatasetSpec(
        name="arxiv",
        files=("ogbn_arxiv",),
        paper_name="ArXiv",
        load_raw=_load_arxiv_year,
        id_classes=(2, 3, 4),
        ood_classes=(0, 1),
        num_features=128,
        homophily="heterophilic",
        split="random",
    )
)

register_dataset(
    DatasetSpec(
        name="patents",
        files=("snap-patents.mat",),
        paper_name="Patents",
        load_raw=_load_patents_year,
        id_classes=(2, 3, 4),
        ood_classes=(0, 1),
        num_features=269,
        homophily="heterophilic",
        split="random",
        batch_size=16384,
        num_neighbors=8,
    )
)
