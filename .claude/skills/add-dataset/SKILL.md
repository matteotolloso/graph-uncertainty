---
name: add-dataset
description: Add a new graph dataset to the leave-out-class OOD benchmark (DatasetSpec + raw loader) so every method, sweep and `cgnn doctor` can use it. Use when the user wants to evaluate on a new graph/benchmark or change a dataset's ID/OOD class split.
---

# Add a dataset

A dataset = one file `cgnn/data/sources/<name>.py` with a raw loader returning a `RawGraph` and a
`register_dataset(DatasetSpec(...))` call. The OOD protocol (masks, one-hot ID labels, all-zero OOD rows),
random splits, fingerprints and loaders are generic (`cgnn/data/ood.py`, `splits.py`, `loaders.py`).

## 1. Raw loader
```python
from pathlib import Path
import torch
from cgnn.data.spec import DatasetSpec, RawGraph, register_dataset

def _load_mygraph(data_dir: Path, spec: DatasetSpec) -> RawGraph:
    from torch_geometric.datasets import SomeDataset          # heavy/optional imports stay inside
    data = SomeDataset(root=str(data_dir / "mygraph"))[0]
    return RawGraph(
        x=data.x.float(), edge_index=data.edge_index.long(), y=data.y.long(),   # ORIGINAL labels
        # public split: pass ONE split (index it yourself, e.g. [:, spec.public_split_index]);
        # random split: leave the masks as None and set split="random" below.
        train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
    )
```
Rules: never touch the global RNG (random splits are made by the framework from `split_seed`); keep
edges as the source provides them unless you document otherwise; files live under `CGNN_DATA_DIR`.

## 2. Spec (the paper's Table 1 row)
```python
register_dataset(DatasetSpec(
    name="mygraph",                 # CLI / W&B key (snake_case, never rename later)
    paper_name="MyGraph",
    load_raw=_load_mygraph,
    id_classes=(2, 3, 4),           # trained on; re-indexed 0..C_id-1 in sorted order
    ood_classes=(0, 1),             # never seen in training; all-zero label rows
    num_features=128,               # must equal x.size(1) (asserted by the tests / sweeps' in_channels)
    homophily="heterophilic",       # or "homophilic"
    split="public",                 # or "random" (60/20/20 via split_ratios, seeded by split_seed)
    files=("mygraph",),             # paths under CGNN_DATA_DIR checked by `cgnn doctor`
    batch_size=-1,                  # >0 -> NeighborLoader training with num_neighbors per layer
    num_neighbors=10,
    in_paper=False,
))
```
Choose ID/OOD classes so that both sides have enough val/test nodes; every label must be in exactly one set
(nodes with other labels are dropped from all splits with a warning).

## 3. Register and check
1. Import the module in `cgnn/data/sources/__init__.py`.
2. `cgnn list datasets` (FILES = ok), `cgnn data mygraph` (prints the Table-1 row: train ID, val/test ID/OOD, fingerprint).
3. Add the name to the `slow` parametrisation in `tests/test_data.py` if the data is small, then
   `python -m pytest -q tests/test_data.py tests/test_config.py`.
4. Smoke test a method: `cgnn run -m credal_LJ_dual_head_detached -d mygraph --wandb disabled --gpus 1 --set max_epochs=5`.
5. Post-hoc methods need backbones: `cgnn sweep -m vanilla -d mygraph -c 100` first.
6. Document it: dataset table in `README.md`, `docs/protocol.md` (split), `CHANGELOG.md`.
