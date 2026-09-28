# Experimental protocol

## Task: transductive node-level OOD detection (leave-out-class)
For a graph `G = (V, E, X)` the classes are split into in-distribution (ID) and out-of-distribution (OOD)
classes (`DatasetSpec.id_classes` / `ood_classes`, paper Table 1). Nodes are split into train/val/test:

| split | nodes | labels available to the model |
|---|---|---|
| train | base train ∩ ID | ID labels (re-indexed 0..C_id−1) |
| val | base val (ID + OOD) | used for early stopping, sweeps, component and backbone selection |
| test | base test (ID + OOD) | final evaluation only |

The whole graph (all nodes and edges, including OOD nodes) is visible during message passing.
Labels are stored as `y ∈ {0,1}^{N×C_id}`: one-hot for ID nodes, **all-zero rows for OOD nodes**.
Metric: AUROC of the uncertainty score for detecting OOD nodes (OOD = positive class, higher score =
more OOD) on the test split; also ID accuracy of the point prediction.

Splits: public splits where the dataset provides one (Squirrel/Chameleon split 0, Amazon-Ratings/Roman-Empire
split 0, Reddit2 GraphSAINT roles, Cora "full"); random 60/20/20 for ArXiv, Patents, Coauthor with
`split_seed` (default 0). `cgnn data <dataset>` prints the counts and the split fingerprint.

## Methods and what is trained on what
- **Trainable** (vanilla, credal*, graph_esn): trained on the train split, early-stopped on the monitored
  validation metric (`patience`), tested at the last epoch (`test_ckpt=last`, legacy) or at the best
  validation checkpoint (`test_ckpt=best`).
- **Post-hoc** (energy, odin, mahalanobis, knn, knn_LJ, gnnsafe, gebm): computed from a frozen VanillaGNN
  backbone; statistics (class means, kNN index, GEBM) are fitted on the training nodes. Backbones are the
  vanilla checkpoints of the same dataset and split, ranked by `val_auroc` (MSP); `backbone_rank` picks one.
- **Ensembles** (ensemble): top-M vanilla backbones; `*_classic` = classical entropy decomposition,
  `*_credal` = credal hull (min/max) with generalized entropies.
- **Post-hoc with training**: `frozen` trains a credal head on the frozen backbone's joint latent (train
  split); `cagcn` trains a calibration GCN on the **ID validation nodes** (as in the CaGCN paper).

## Model selection and reporting (paper Table 2)
1. Hyper-parameters: W&B sweeps (`configs/methods/<method>.yaml`) maximising the validation metric.
2. Uncertainty component: for methods with several scores (EU/AU for credal methods; `*_credal`/`*_classic`
   for ensembles) the component with the best validation AUROC is reported.
3. Replicates: mean ± std over 5 runs. Recommended (0.2): take the best configuration and re-run it with 5
   seeds (`cgnn run --from-sweep <id> --seeds 0 1 2 3 4`, aggregate with `cgnn results --aggregate group`).
   Legacy tables used the top-5 validation runs of each sweep (`--aggregate topk`, all with seed 42).

Validation contains OOD nodes, so model selection has access to examples of the OOD classes
(common in graph OOD benchmarks, but it must be stated).

## Protocol switches (all logged to W&B)
| key | default | legacy value | effect |
|---|---|---|---|
| `split_seed` | 0 | `legacy` | random splits independent of model init; shared by all methods |
| `auroc_impl` | `exact` | `torchmetrics` | exact rank AUROC vs sigmoid-saturating implementation |
| `min_entropy_method` | `greedy` | `greedy` | AU solver (`auto` = exact for C ≤ 6) |
| `test_ckpt` | `last` | `last` | test at last epoch or best validation checkpoint |
| `backbone_split_check` | `error` | (none) | refuse post-hoc backbones trained on another split |
| `eval_neighbor_sampling` | per dataset | Reddit2: on | NeighborLoader evaluation for big graphs |

To reproduce pre-0.2 numbers exactly: `--set auroc_impl=torchmetrics split_seed=legacy`.
Recommended for new results: defaults + `min_entropy_method=auto` (+ `test_ckpt=best` if you re-run
everything), retrained backbones for random-split datasets.
