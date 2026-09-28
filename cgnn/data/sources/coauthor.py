"""Coauthor-CS (Shchur et al. 2018), random 60/20/20 split.

The ``Coauthor`` dataset class is taken from the code of "Revisiting Score
Propagation in Graph Out-of-Distribution Detection" (Ma et al., NeurIPS 2024).
"""

from __future__ import annotations

import os.path as osp
from collections.abc import Callable
from pathlib import Path

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_npz

from cgnn.data.spec import DatasetSpec, RawGraph, register_dataset


# ----- begin: code from Ma et al. (NeurIPS 2024) -----------------------------
class Coauthor(InMemoryDataset):
    r"""The Coauthor CS and Coauthor Physics networks from the
    `"Pitfalls of Graph Neural Network Evaluation" <https://arxiv.org/abs/1811.05868>`_ paper."""

    url = "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/"

    def __init__(
        self,
        root: str,
        name: str,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        force_reload: bool = False,
    ) -> None:
        assert name.lower() in ["cs", "physics"]
        self.name = "CS" if name.lower() == "cs" else "Physics"
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> str:
        return f"ms_academic_{self.name[:3].lower()}.npz"

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self) -> None:
        download_url(self.url + self.raw_file_names, self.raw_dir)

    def process(self) -> None:
        data = read_npz(self.raw_paths[0], to_undirected=True)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])


# ----- end: code from Ma et al. ----------------------------------------------


def _load_coauthor_cs(data_dir: Path, spec: DatasetSpec) -> RawGraph:
    data = Coauthor(root=osp.join(str(data_dir), "coauthors"), name="CS")[0]
    return RawGraph(x=data.x, edge_index=data.edge_index, y=data.y)


register_dataset(
    DatasetSpec(
        name="coauthor",
        files=("coauthors/CS",),
        paper_name="Coauthor",
        load_raw=_load_coauthor_cs,
        id_classes=tuple(range(4, 15)),
        ood_classes=tuple(range(0, 4)),
        num_features=6805,
        homophily="homophilic",
        split="random",
    )
)
