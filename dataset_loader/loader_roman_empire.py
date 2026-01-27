# --- roman_empire loader: train = ID only, val/test = ID+OOD ---
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, DataLoader as GeoDataLoader

def load_roman_empire(npz_path, config, split_idx=0, ood_classes=tuple(range(0,5)), id_classes=tuple(range(5,18))):
    z = np.load(npz_path + 'roman_empire/roman_empire.npz', allow_pickle=True)

    # 1) Core tensors
    x_np = np.asarray(z["node_features"], dtype=np.float32)             # (N,F)
    y_np = np.asarray(z["node_labels"]).reshape(-1).astype(np.int64)    # (N,)
    edges_np = np.asarray(z["edges"], dtype=np.int64)                    # (E,2)
    train_masks = np.asarray(z["train_masks"], dtype=bool)               # (S,N)
    val_masks   = np.asarray(z["val_masks"],   dtype=bool)               # (S,N)
    test_masks  = np.asarray(z["test_masks"],  dtype=bool)               # (S,N)

    N = x_np.shape[0]
    assert y_np.shape[0] == N and edges_np.ndim == 2 and edges_np.shape[1] == 2
    assert train_masks.shape[1] == N and val_masks.shape[1] == N and test_masks.shape[1] == N
    assert 0 <= split_idx < train_masks.shape[0]

    # PyG expects (2,E)
    edge_index = torch.from_numpy(edges_np.T).long()
    x = torch.from_numpy(x_np).float()
    y_orig = torch.from_numpy(y_np).long()
    data = Data(x=x, edge_index=edge_index, y=y_orig)

    # 2) Select split row -> (N,) masks
    train_mask_split = torch.from_numpy(train_masks[split_idx]).bool()
    val_mask_split   = torch.from_numpy(val_masks[split_idx]).bool()
    test_mask_split  = torch.from_numpy(test_masks[split_idx]).bool()

    # 3) OOD policy
    ood_node_mask = torch.isin(data.y, torch.tensor(list(ood_classes), dtype=torch.long))
    id_node_mask  = torch.isin(data.y, torch.tensor(list(id_classes),  dtype=torch.long))

    data.train_mask = train_mask_split & id_node_mask   # ID-only
    data.val_mask   = val_mask_split                    # ID + OOD
    data.test_mask  = test_mask_split                   # ID + OOD

    # 4) One-hot over ID classes (OOD -> all zeros)
    num_id = len(id_classes)
    new_y = torch.zeros((N, num_id), dtype=torch.float)
    id_min = min(id_classes)
    # remap ID labels to 0..num_id-1 using a table (safer than subtract if classes not contiguous)
    remap = {c:i for i, c in enumerate(sorted(id_classes))}
    id_labels_remapped = torch.tensor([remap[int(lbl)] for lbl in data.y[id_node_mask].tolist()], dtype=torch.long)
    new_y[id_node_mask] = F.one_hot(id_labels_remapped, num_classes=num_id).float()
    data.y = new_y

    # 5) Reporting
    id_val_nodes  = data.val_mask  & id_node_mask
    ood_val_nodes = data.val_mask  & ood_node_mask
    id_test_nodes = data.test_mask & id_node_mask
    ood_test_nodes= data.test_mask & ood_node_mask

    print("--- Roman Empire (ID train / ID+OOD val,test) ---")
    print(f"N={N}, F={data.x.size(-1)}, E={int(data.edge_index.size(1))}")
    print(f"Split idx: {split_idx} / {train_masks.shape[0]-1}")
    print(f"ID classes: {sorted(id_classes)}  | OOD classes: {sorted(ood_classes)}")
    print(f"Train(ID only): {int(data.train_mask.sum())}")
    print(f"Val(ID+OOD):    {int(data.val_mask.sum())} -> {int(id_val_nodes.sum())} ID, {int(ood_val_nodes.sum())} OOD")
    print(f"Test(ID+OOD):   {int(data.test_mask.sum())} -> {int(id_test_nodes.sum())} ID, {int(ood_test_nodes.sum())} OOD")

    # 6) Loaders
    batch_size   = int(config.get("batch_size", -1))
    num_neighbors= int(config.get("num_neighbors", 10))
    num_layers   = int(config.get("num_layers", 2))

    if batch_size <= 0:
        print("Using GeoDataLoader for full-batch training.")
        train_loader = GeoDataLoader([data], batch_size=1, shuffle=False)
    else:
        print(f"Using NeighborLoader with batch_size={batch_size}.")
        train_loader = NeighborLoader(
            data,
            input_nodes=data.train_mask,
            batch_size=batch_size,
            num_neighbors=[num_neighbors]*num_layers,
            shuffle=True
        )

    val_loader  = GeoDataLoader([data], batch_size=1, shuffle=False)
    test_loader = GeoDataLoader([data], batch_size=1, shuffle=False)
    return train_loader, val_loader, test_loader
