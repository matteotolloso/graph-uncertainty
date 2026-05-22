import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_npz
import os.path as osp
from typing import Callable, Optional
from torch_geometric.loader import NeighborLoader, DataLoader

# =====================================================================================
##### CODE FROM THE PAPER: Revisiting Score Propagation in Graph Out-of-Distribution Detection 
# Longfei Ma, Yiyou Sun, Kaize Ding, Zemin Liu, Fei Wu 
# 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

# =====================================================================================

def one_hot_encode(labels, num_classes):
    """Simple one-hot encoder."""
    return F.one_hot(labels, num_classes=num_classes).float()

class Coauthor(InMemoryDataset):
    r"""The Coauthor CS and Coauthor Physics networks from the
    `"Pitfalls of Graph Neural Network Evaluation"
    <https://arxiv.org/abs/1811.05868>`_ paper."""
    url = 'https://github.com/shchur/gnn-benchmark/raw/master/data/npz/'

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        assert name.lower() in ['cs', 'physics']
        self.name = 'CS' if name.lower() == 'cs' else 'Physics'
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'ms_academic_{self.name[:3].lower()}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        download_url(self.url + self.raw_file_names, self.raw_dir)

    def process(self) -> None:
        data = read_npz(self.raw_paths[0], to_undirected=True)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])


##### END OF THE CODE FROM THE PAPER: Revisiting Score Propagation in Graph Out-of-Distribution Detection 


def load_coauthor_cs(DATASET_STORAGE_PATH, config):
    """
    Loads the Coauthor CS dataset and prepares it for transductive OOD detection,
    now including OOD samples in the validation set for hyperparameter tuning.

    Args:
        DATASET_STORAGE_PATH (str): Path to store the dataset.
        config (dict): A configuration dictionary.

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    # --- 1. Load the Full Graph Data ---
    dataset = Coauthor(root=osp.join(DATASET_STORAGE_PATH, 'coauthors'), name='CS')
    data = dataset[0]
    num_nodes = data.num_nodes

    # --- 2. Prepare Masks ---
    OODclass = list(range(4))  # Classes 0, 1, 2, 3
    IDclass = list(range(4, 15)) # Classes 4 to 14
    num_id_classes = len(IDclass)
    
    train_ratio = 0.6
    val_ratio = 0.2

    indices = torch.randperm(num_nodes)
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    
    # Create initial boolean masks for the random splits
    train_mask_split = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask_split = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask_split = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask_split[indices[:train_size]] = True
    val_mask_split[indices[train_size:train_size + val_size]] = True
    test_mask_split[indices[train_size + val_size:]] = True

    # --- 3. Create the Final Masks for the Data Object ---
    ood_node_mask = torch.isin(data.y, torch.tensor(OODclass))
    id_node_mask = ~ood_node_mask

    # The final train mask must ONLY include ID-class nodes
    data.train_mask = train_mask_split & id_node_mask
    
    # The validation mask should now include ALL nodes (ID and OOD) from its split
    data.val_mask = val_mask_split
    
    # The test mask also includes ALL nodes from its split
    data.test_mask = test_mask_split

    # --- Reporting for verification ---
    print("--- Coauthor CS Dataset with OOD Validation ---")
    id_val_nodes = data.val_mask & id_node_mask
    ood_val_nodes = data.val_mask & ood_node_mask
    id_test_nodes = data.test_mask & id_node_mask
    ood_test_nodes = data.test_mask & ood_node_mask
    
    print(f"Nodes for training (ID only): {data.train_mask.sum().item()}")
    print(f"Nodes for validation (ID+OOD): {data.val_mask.sum().item()} -> {id_val_nodes.sum()} ID, {ood_val_nodes.sum()} OOD")
    print(f"Nodes for testing (ID+OOD): {data.test_mask.sum().item()} -> {id_test_nodes.sum()} ID, {ood_test_nodes.sum()} OOD")
    
    # --- 4. Prepare the Unified Label Tensor (y) ---
    new_y = torch.zeros((num_nodes, num_id_classes), dtype=torch.float)
    original_id_labels = data.y[id_node_mask]
    remapped_id_labels = original_id_labels - min(IDclass)
    new_y[id_node_mask] = one_hot_encode(remapped_id_labels, num_id_classes)
    data.y = new_y
    
    # --- 5. Create DataLoaders ---
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
            input_nodes=data.train_mask,
            batch_size=batch_size,
            num_neighbors=neighbor_sizes,
            shuffle=True
        )

    val_loader = DataLoader([data], batch_size=1, shuffle=False)
    test_loader = DataLoader([data], batch_size=1, shuffle=False)
    
    return train_loader, val_loader, test_loader
