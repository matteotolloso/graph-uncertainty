# Credal Graph Neural Networks

Code for *Credal Graph Neural Networks* (CGNN): GNNs that output **credal sets** (probability intervals)
for node classification, with total / aleatoric / epistemic uncertainty from generalized entropies, a
dual-head architecture with a detached credal head on the **joint layer-wise latent**, and a benchmark against
9 uncertainty baselines (plus 4 ablations) for node-level OOD detection on homophilic and heterophilic graphs.

> Working with a coding agent? Start from [`AGENTS.md`](AGENTS.md) (Claude Code also reads `CLAUDE.md`
> and `.claude/`).

## Install
```bash
conda env create -f environment.yml && conda activate gu     # or an existing env with torch + PyG
pip install -e ".[dev]"                                      # the `cgnn` package, CLI and test tools
git submodule update --init graph-ebm && pip install -e graph-ebm   # only for the GEBM baseline
wandb login                                                  # experiment tracking
cgnn doctor                                                  # checks deps, data, checkpoints, GPUs
```
Datasets are expected under `dataset/` (override with `CGNN_DATA_DIR`); most are downloaded by PyG/OGB on
first use, see `cgnn list datasets`.

## Quick start
```bash
cgnn list                                                       # methods (with paper names) and datasets
cgnn run -m credal_LJ_dual_head_detached -d synthetic --cpu      # CGNN end-to-end in seconds, no W&B
cgnn run -m credal_LJ_dual_head_detached -d squirrel --seeds 0 1 2 3 4   # 5 seeds on Squirrel (1 GPU)
cgnn sweep -m credal_LJ_dual_head_detached -d arxiv -c 100      # W&B sweep (configs/methods/*.yaml)
cgnn results --format latex --out outputs/tables/table2.tex    # paper-style table from W&B
python -m pytest -q                                             # full test suite, ~15 s on CPU
```
Legacy interface still works: `python main.py -m <method> -d <dataset> -c <count>` (= `cgnn sweep`).
Full pipeline for Table 2 in dependency order: `scripts/reproduce_paper.sh`.

## Methods
| Paper | Key | Kind |
|---|---|---|
| **CGNN** (joint latent, dual head, detached credal head) | `credal_LJ_dual_head_detached` | trainable |
| CGNN last layer / only credal | `credal` / `credal_LJ` | trainable |
| CGNN post train | `frozen` | post-hoc + trained credal head |
| CGNN by ensemble / Classical ensemble | `ensemble` (`*_credal` / `*_classic`) | post-hoc |
| Energy, ODIN, Mahalanobis, KNN | `energy`, `odin`, `mahalanobis`, `knn` | post-hoc |
| GNNSafe, GEBM, JLDE, CaGCN | `gnnsafe`, `gebm`, `knn_LJ`, `cagcn` | post-hoc |
| (not in paper) dual head variants, Eq. 14 with `z^0`, clean last-layer ablation, GraphESN | `credal_LJ_dual_head*`, `credal_LJ0_dual_head_detached`, `credal_last_dual_head_detached`, `graph_esn` | trainable |

Post-hoc methods use the best VanillaGNN checkpoints (`vanilla`) of the same dataset and split.
Details: [`docs/paper_to_code.md`](docs/paper_to_code.md).

## Datasets
| Dataset | Key | Homophily | ID / OOD classes | Split |
|---|---|---|---|---|
| Squirrel | `squirrel` | heterophilic | {2,3,4} / {0,1} | public (split 0) |
| ArXiv (year) | `arxiv` | heterophilic | {2,3,4} / {0,1} | random 60/20/20 (`split_seed`) |
| Patents (year) | `patents` | heterophilic | {2,3,4} / {0,1} | random 60/20/20 |
| Amazon-Ratings | `amazon_ratings` | heterophilic | {2,3,4} / {0,1} | public (split 0) |
| Roman Empire | `roman_empire` | heterophilic | {5..17} / {0..4} | public (split 0) |
| Coauthor-CS | `coauthor` | homophilic | {4..14} / {0..3} | random 60/20/20 |
| Reddit2 | `reddit2` | homophilic | {11..40} / {0..10} | public (GraphSAINT) |
| Chameleon, Cora, synthetic | `chameleon`, `cora`, `synthetic` | – | see `cgnn describe` | not in the paper |

## Protocol and configuration
Leave-out-class OOD detection, transductive; model selection on a validation split containing ID and
OOD nodes; AUROC (OOD = positive). See [`docs/protocol.md`](docs/protocol.md). Configuration is resolved as
`configs/defaults.yaml` → `configs/methods/<method>.yaml` → dataset metadata → sweep values / `--set`.
Important switches: `split_seed`, `auroc_impl`, `min_entropy_method`, `test_ckpt`.
To reproduce the pre-0.2 (submission) numbers bit-for-bit: `--set auroc_impl=torchmetrics split_seed=legacy`.
Audit findings that affect some reported numbers: [`docs/known_issues.md`](docs/known_issues.md).

## Repository layout
```
cgnn/         package: uncertainty math, data, models, method registry, runner, CLI, results
configs/      defaults.yaml and one YAML per method (defaults + W&B sweep)
tests/        pytest suite (every method runs end-to-end on a synthetic graph)
docs/         protocol, paper<->code map, known issues, architecture
scripts/      reproduce_paper.sh, run_all.sh (legacy run log)
notebooks/    Figure 1 (credal simplex plots)
docker/       container build/run scripts
graph-ebm/    GEBM baseline (git submodule)
```
Adding a method or a dataset: [`.claude/skills/add-method/SKILL.md`](.claude/skills/add-method/SKILL.md),
[`.claude/skills/add-dataset/SKILL.md`](.claude/skills/add-dataset/SKILL.md) (plain-language procedures,
useful for humans too).

## Hardware note
Runs use one GPU by default (`--gpus N`, max 2) and pin an idle GPU automatically; post-hoc methods on
Reddit2 need `--cpu` (full-graph inference needs ~52 GB).

## Acknowledgements
Dataset classes for Coauthor/Reddit2 and the year-based label construction follow Ma et al., "Revisiting
Score Propagation in Graph Out-of-Distribution Detection" (NeurIPS 2024). GEBM uses the official
`graph-ebm` implementation (submodule).
