"""Reddit2 (Hamilton et al. 2017) with the GraphSAINT public split.

The dataset class and downloader are taken from the code of "Revisiting Score
Propagation in Graph Out-of-Distribution Detection" (Ma et al., NeurIPS 2024).
Reddit2 is the only dataset evaluated with NeighborLoader by default
(``eval_neighbor_sampling=True``).
"""

from __future__ import annotations

import json
import os
import os.path as osp
import ssl
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

import fsspec
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs

from cgnn.data.spec import DatasetSpec, RawGraph, register_dataset


# ----- begin: code from Ma et al. (NeurIPS 2024) -----------------------------
def download_url(url: str, folder: str, log: bool = True, filename: str | None = None):
    if filename is None:
        filename = url.rpartition("/")[2]
        filename = filename if filename[0] == "?" else filename.split("?")[0]

    path = osp.join(folder, filename)
    if fs.exists(path):  # pragma: no cover
        if log and "pytest" not in sys.modules:
            print(f"Using existing file {filename}", file=sys.stderr)
        return path

    if log and "pytest" not in sys.modules:
        print(f"Downloading {url}", file=sys.stderr)

    os.makedirs(folder, exist_ok=True)
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)
    with fsspec.open(path, "wb") as f:
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return path


def download_google_url(id: str, folder: str, filename: str, log: bool = True):
    url = f"https://drive.usercontent.google.com/download?id={id}&confirm=t"
    return download_url(url, folder, log, filename)


class Reddit2(InMemoryDataset):
    adj_full_id = "1sncK996BM5lpuDf75lDFqCiDZyErc1c2"
    feats_id = "1ZsHaJ0ussP1W722krmEIp_8pwKAoi5b3"
    class_map_id = "1JF3Pjv9OboMNYs2aXRQGbJbc4t_nDd5u"
    role_id = "1nJIKd77lcAGU4j-kVNx_AIGEkveIKz3A"

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        return ["adj_full.npz", "feats.npy", "class_map.json", "role.json"]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self) -> None:
        download_google_url(self.adj_full_id, self.raw_dir, "adj_full.npz")
        download_google_url(self.feats_id, self.raw_dir, "feats.npy")
        download_google_url(self.class_map_id, self.raw_dir, "class_map.json")
        download_google_url(self.role_id, self.raw_dir, "role.json")

    def process(self) -> None:
        import scipy.sparse as sp

        f = np.load(osp.join(self.raw_dir, "adj_full.npz"))
        adj = sp.csr_matrix((f["data"], f["indices"], f["indptr"]), f["shape"]).tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        x = torch.from_numpy(np.load(osp.join(self.raw_dir, "feats.npy"))).to(torch.float)
        ys = [-1] * x.size(0)
        with open(osp.join(self.raw_dir, "class_map.json")) as fh:
            class_map = json.load(fh)
            for key, item in class_map.items():
                ys[int(key)] = item
        y = torch.tensor(ys)
        with open(osp.join(self.raw_dir, "role.json")) as fh:
            role = json.load(fh)
        train_mask = torch.zeros(x.size(0), dtype=torch.bool)
        train_mask[torch.tensor(role["tr"])] = True
        val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        val_mask[torch.tensor(role["va"])] = True
        test_mask = torch.zeros(x.size(0), dtype=torch.bool)
        test_mask[torch.tensor(role["te"])] = True
        data = Data(
            x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask
        )
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])


# ----- end: code from Ma et al. ----------------------------------------------


def _load_reddit2(data_dir: Path, spec: DatasetSpec) -> RawGraph:
    data = Reddit2(root=osp.join(str(data_dir), "reddit2"))[0]
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
        name="reddit2",
        files=("reddit2/raw",),
        paper_name="Reddit2",
        load_raw=_load_reddit2,
        id_classes=tuple(range(11, 41)),
        ood_classes=tuple(range(0, 11)),
        num_features=602,
        homophily="homophilic",
        split="public",
        batch_size=2**10,
        num_neighbors=8,
        eval_neighbor_sampling=True,
    )
)
