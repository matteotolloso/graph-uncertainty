import torch
import torch.nn.functional as F
import json
import os.path as osp
from typing import Callable, List, Optional
import numpy as np

import torch_geometric
from torch_geometric.data import Data, InMemoryDataset, download_google_url
from torch_geometric.loader import DataLoader, NeighborLoader

import os
import os.path as osp
import ssl
import sys
import urllib
from typing import Optional

import fsspec

from torch_geometric.io import fs


##### CODE FROM THE PAPER: Revisiting Score Propagation in Graph Out-of-Distribution Detection 
# Longfei Ma, Yiyou Sun, Kaize Ding, Zemin Liu, Fei Wu 
# 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

def download_url(
    url: str,
    folder: str,
    log: bool = True,
    filename: Optional[str] = None,
):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (str): The URL.
        folder (str): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
        filename (str, optional): The filename of the downloaded file. If set
            to :obj:`None`, will correspond to the filename given by the URL.
            (default: :obj:`None`)
    """
    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = osp.join(folder, filename)

    if fs.exists(path):  # pragma: no cover
        if log and 'pytest' not in sys.modules:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log and 'pytest' not in sys.modules:
        print(f'Downloading {url}', file=sys.stderr)

    os.makedirs(folder, exist_ok=True)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with fsspec.open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def download_google_url(
    id: str,
    folder: str,
    filename: str,
    log: bool = True,
):
    r"""Downloads the content of a Google Drive ID to a specific folder."""
    url = f'https://drive.usercontent.google.com/download?id={id}&confirm=t'
    return download_url(url, folder, log, filename)

class Reddit2(InMemoryDataset):
    adj_full_id = '1sncK996BM5lpuDf75lDFqCiDZyErc1c2'
    feats_id = '1ZsHaJ0ussP1W722krmEIp_8pwKAoi5b3'
    class_map_id = '1JF3Pjv9OboMNYs2aXRQGbJbc4t_nDd5u'
    role_id = '1nJIKd77lcAGU4j-kVNx_AIGEkveIKz3A'

    def __init__(
        self, root: str, transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None, force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['adj_full.npz', 'feats.npy', 'class_map.json', 'role.json']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        download_google_url(self.adj_full_id, self.raw_dir, 'adj_full.npz')
        download_google_url(self.feats_id, self.raw_dir, 'feats.npy')
        download_google_url(self.class_map_id, self.raw_dir, 'class_map.json')
        download_google_url(self.role_id, self.raw_dir, 'role.json')

    def process(self) -> None:
        import scipy.sparse as sp
        f = np.load(osp.join(self.raw_dir, 'adj_full.npz'))
        adj = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        x = torch.from_numpy(np.load(osp.join(self.raw_dir, 'feats.npy'))).to(torch.float)
        ys = [-1] * x.size(0)
        with open(osp.join(self.raw_dir, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        y = torch.tensor(ys)
        with open(osp.join(self.raw_dir, 'role.json')) as f:
            role = json.load(f)
        train_mask = torch.zeros(x.size(0), dtype=torch.bool)
        train_mask[torch.tensor(role['tr'])] = True
        val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        val_mask[torch.tensor(role['va'])] = True
        test_mask = torch.zeros(x.size(0), dtype=torch.bool)
        test_mask[torch.tensor(role['te'])] = True
        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])



##### END OF THE CODE FROM THE PAPER: Revisiting Score Propagation in Graph Out-of-Distribution Detection 


def one_hot_encode(labels, num_classes):
    """Simple one-hot encoder."""
    return F.one_hot(labels, num_classes=num_classes).float()


def load_reddit2(DATASET_STORAGE_PATH, config):
    """
    Loads the Reddit2 dataset and prepares it for transductive OOD detection,
    now including OOD samples in the validation set for hyperparameter tuning.

    Args:
        DATASET_STORAGE_PATH (str): Path to store the dataset.
        config (dict): A configuration dictionary.

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    # --- 1. Load the Full Graph Data ---
    dataset = Reddit2(root=osp.join(DATASET_STORAGE_PATH, 'reddit2'))
    data = dataset[0]

    # --- 2. Define Class and Node Splits ---
    OODclass = list(range(11))
    IDclass = list(range(11, 41))
    num_id_classes = len(IDclass)
    
    original_train_mask = data.train_mask
    original_val_mask = data.val_mask
    original_test_mask = data.test_mask

    ood_node_mask = torch.isin(data.y, torch.tensor(OODclass))
    id_node_mask = ~ood_node_mask

    # --- 3. Create OOD-Aware Validation and Test Masks ---

    # ID nodes for each set are taken from the original splits
    id_train_nodes = original_train_mask & id_node_mask
    id_val_nodes = original_val_mask & id_node_mask
    id_test_nodes = original_test_mask & id_node_mask

    # OOD nodes for each set are also taken from the original splits
    ood_train_nodes = original_train_mask & ood_node_mask
    ood_val_nodes = original_val_mask & ood_node_mask
    ood_test_nodes = original_test_mask & ood_node_mask
    
    # The final masks are attached to the data object
    data.train_mask = id_train_nodes
    data.val_mask = id_val_nodes | ood_val_nodes   # Validation includes ID and OOD nodes
    data.test_mask = id_test_nodes | ood_test_nodes   # Test set remains the final hold-out

    print("--- Reddit2 Dataset with OOD Validation ---")
    print(f"Nodes for training (ID only): {data.train_mask.sum().item()}")
    print(f"Nodes for validation (ID+OOD): {data.val_mask.sum().item()} -> {id_val_nodes.sum()} ID, {ood_val_nodes.sum()} OOD")
    print(f"Nodes for testing (ID+OOD): {data.test_mask.sum().item()} -> {id_test_nodes.sum()} ID, {ood_test_nodes.sum()} OOD")

    # --- 4. Prepare the Unified Label Tensor (y) ---
    new_y = torch.zeros((data.num_nodes, num_id_classes), dtype=torch.float)
    original_id_labels = data.y[id_node_mask]
    remapped_id_labels = original_id_labels - min(IDclass)
    new_y[id_node_mask] = one_hot_encode(remapped_id_labels, num_id_classes)
    data.y = new_y
    
    # ==============================================================================
    # --- 5. Create DataLoaders (with Mini-Batching Logic) ---
    # ==============================================================================
    if config.get("batch_size", -1) <= 0:
        print("Using DataLoader for full-batch training.")
        train_loader = DataLoader([data], batch_size=1, shuffle=False)
    else:
        batch_size = int(config["batch_size"])
        num_layers = int(config.get("num_layers", 2))
        num_neighbors = int(config.get("num_neighbors", 10))
        neighbor_sizes = [num_neighbors] * num_layers

        print(f"Using NeighborLoader for mini-batch training with batch size {batch_size}.")
        train_loader = NeighborLoader(
            data,
            input_nodes=data.train_mask, # Crucial: sample starting nodes from the train mask
            batch_size=batch_size,
            num_neighbors=neighbor_sizes,
            shuffle=True
        )

    val_loader = DataLoader([data], batch_size=1, shuffle=False)
    test_loader = DataLoader([data], batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
