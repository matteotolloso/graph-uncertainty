# Credal Graph Neural Networks

Code for *Credal Graph Neural Networks* (CGNN): GNNs that output **credal sets** (probability intervals)
for node classification, with total / aleatoric / epistemic uncertainty from generalized entropies, a
dual-head architecture with a detached credal head on the **joint layer-wise latent**, and a benchmark against
uncertainty baselines for node-level OOD detection on homophilic and heterophilic graphs (paper Table 2).

## Install
```bash
conda env create -f environment.yml && conda activate gu     # or an existing env with torch + PyG
pip install -e ".[all]"                                      # the `cgnn` package, CLI, optional deps, tests
git submodule update --init graph-ebm && pip install -e graph-ebm   # only for the GEBM baseline
wandb login                                                  # sweeps and result tables use W&B
cgnn doctor                                                  # checks deps, data, checkpoints, GPUs
```
Datasets are stored under `dataset/` (override with `CGNN_DATA_DIR`) and downloaded by PyG/OGB on first use;
checkpoints go to `checkpoints/` (`CGNN_CKPT_DIR`), run summaries to `outputs/` (`CGNN_OUTPUT_DIR`).

## Quick start
```bash
cgnn list                                                        # methods (with paper names) and datasets
cgnn run -m credal_LJ_dual_head_detached -d synthetic --cpu --wandb disabled   # CGNN end-to-end in seconds
cgnn run -m credal_LJ_dual_head_detached -d arxiv --seeds 0 1 2  # single runs on a real dataset (1 GPU)
cgnn describe credal_LJ_dual_head_detached                       # defaults and sweep space
python -m pytest -q                                              # test suite, ~15 s on CPU
```

## Reproducing Table 2
```bash
bash scripts/reproduce_paper.sh                 # prints the full pipeline; EXECUTE=1 runs it
```
For every dataset, in this order:
1. `cgnn sweep -m vanilla -d <ds>`: GNN backbones. Their checkpoints, ranked by validation AUROC, are used by
   every post-hoc method, the ensembles and CGNN post train.
2. `cgnn sweep -m <method> -d <ds>` for the post-hoc baselines (grid sweeps), CaGCN and CGNN post train.
3. `cgnn sweep -m <method> -d <ds>` for CGNN and its trainable ablations.
4. `cgnn results --format latex`: for every method and dataset, the uncertainty component (e.g. EU or AU) with
   the best validation AUROC, reported as mean ± std of the test AUROC over the 5 runs with the best validation
   score of the sweep.

`configs/defaults.yaml` holds the protocol used for the paper (seed 42, `split_seed: null`,
`auroc_impl: torchmetrics`, `min_entropy_method: greedy`, `test_ckpt: last`); every key can be overridden
with `--set key=value`.

## Methods
| Paper | Key | Kind |
|---|---|---|
| **CGNN** (joint latent, dual head, detached credal head) | `credal_LJ_dual_head_detached` | trainable |
| CGNN last layer | `credal` | trainable |
| CGNN only credal | `credal_LJ` | trainable |
| CGNN post train | `frozen` | credal head trained on a frozen backbone |
| CGNN by ensemble / Classical ensemble | `ensemble` (`*_credal` / `*_classic` metrics) | post-hoc |
| Energy, ODIN, Mahalanobis, KNN | `energy`, `odin`, `mahalanobis`, `knn` | post-hoc |
| GNNSafe, GEBM, JLDE | `gnnsafe`, `gebm`, `knn_LJ` | post-hoc |
| CaGCN | `cagcn` | post-hoc, calibrated on the ID validation nodes |
| GNN backbone of the post-hoc methods (MSP) | `vanilla` | trainable |

## Datasets
| Dataset | Key | ID / OOD classes | Split |
|---|---|---|---|
| Squirrel | `squirrel` | {2,3,4} / {0,1} | public (split 0) |
| ArXiv (year) | `arxiv` | {2,3,4} / {0,1} | random 60/20/20 |
| Patents (year) | `patents` | {2,3,4} / {0,1} | random 60/20/20 |
| Amazon-Ratings | `amazon_ratings` | {2,3,4} / {0,1} | public (split 0) |
| Roman Empire | `roman_empire` | {5..17} / {0..4} | public (split 0) |
| Coauthor-CS | `coauthor` | {4..14} / {0..3} | random 60/20/20 |
| Reddit2 | `reddit2` | {11..40} / {0..10} | public (GraphSAINT) |

`synthetic` is a small random graph used by the tests.

## Protocol
Leave-out-class OOD detection, transductive: the whole graph is visible during message passing, the model is
trained on the ID training nodes, OOD nodes have all-zero label rows, and AUROC is computed with OOD as the
positive class (higher score = more OOD). Hyper-parameters, early stopping and the uncertainty component are
selected on a validation split containing ID and OOD nodes; the test split is used only for evaluation.
Configuration is resolved as `configs/defaults.yaml` → `configs/methods/<method>.yaml` → dataset metadata →
sweep values / `--set`.

## Layout
```
cgnn/uncertainty/  interval softmax, reachable bounds, min/max entropy (TU/AU/EU), credal loss, ensembles
cgnn/data/         dataset registry, leave-out-class OOD split, splits, loaders (one file per source)
cgnn/models/       LightningModules: vanilla GNN, CGNN and ablations, CGNN post train, CaGCN, detectors/
cgnn/methods/      method registry (method key -> builder)
cgnn/runner.py     generic runner (seed -> build -> data -> fit/validate/test -> metrics)
cgnn/cli.py        `cgnn` command line; results.py builds the tables from W&B
configs/           defaults.yaml and one YAML per method (defaults + W&B sweep space)
scripts/           reproduce_paper.sh
tests/             pytest suite (every method runs end-to-end on the synthetic graph)
graph-ebm/         GEBM baseline (git submodule)
```

## Hardware
Runs use one GPU by default and pin an idle one automatically (`--gpus N`, at most 2; `--cpu` for CPU).
Post-hoc methods on Reddit2 (full-graph inference, ~52 GB) and Squirrel with SAGE need `--cpu` or a large GPU.

## Acknowledgements
Dataset classes for Coauthor/Reddit2 and the year-based label construction follow Ma et al., "Revisiting
Score Propagation in Graph Out-of-Distribution Detection" (NeurIPS 2024). GEBM uses the official
`graph-ebm` implementation (git submodule).
