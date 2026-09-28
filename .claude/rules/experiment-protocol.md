---
paths:
  - "cgnn/data/**"
  - "cgnn/runner.py"
  - "cgnn/methods/**"
  - "cgnn/checkpoints.py"
  - "cgnn/results.py"
  - "configs/**"
---

# Experimental protocol (leave-out-class OOD, transductive)

Full description: `docs/protocol.md`. The invariants below must hold for every dataset and method.

## Labels and masks
- `data.y` is `[N, C_id]`: one-hot over the re-indexed ID classes, **all-zero row for OOD nodes**.
  ID mask = `y.sum(1) == 1`; OOD target for AUROC = `1 - y.sum(1)` (1 = OOD).
- `train_mask` = base train ∩ ID (OOD labels are never seen); `val_mask`/`test_mask` = full base splits (ID+OOD).
- Nodes whose label is neither ID nor OOD must not appear in any split (handled in `cgnn/data/ood.py`).
- Mini-batch loaders: only the first `batch.batch_size` nodes of a NeighborLoader batch are seeds — always use
  `NodeUQModule.split_mask(batch, split)`, never raw masks.

## Splits, seeds, backbones
- Random-split datasets (arxiv, patents, coauthor) use `split_seed` (default 0) with a dedicated generator.
  `split_seed=legacy` reproduces the old architecture-dependent split — only for reproducing old runs.
- Every checkpoint stores `checkpoint["cgnn"]` with dataset, split seed and **split fingerprint**. Post-hoc
  methods must run on the backbone's split: `backbone_split_check=error` (default) enforces it.
- `seed` = global seed of the run. For post-hoc methods, which backbone is used is `backbone_rank`
  (0 = best by `val_auroc`); legacy sweeps of energy/gebm/cagcn/frozen called it `seed` (mapped automatically).

## Model selection and reporting
- Everything is selected on **validation** (early stopping, sweep metric, AU-vs-EU component, top-k runs).
  Never use test metrics to select hyper-parameters, epochs, components or runs.
- Validation contains OOD nodes (legacy protocol, as in the paper); say so when reporting.
- `test_ckpt=last` (legacy) tests the last epoch (best + patience); `test_ckpt=best` tests the best-val checkpoint.
  Don't mix the two within one table.
- Replicates: report mean ± std over ≥5 runs and state what varies (seeds, backbone ranks, or top-k sweep runs).

## Changing the protocol
Protocol changes alter every number. Add a config switch whose default keeps the current behaviour (unless the
user decides otherwise), document it in `docs/protocol.md` + `CHANGELOG.md`, and add a test.
