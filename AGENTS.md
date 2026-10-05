# AGENTS.md — Credal Graph Neural Networks (CGNN)

Research code for *Credal Graph Neural Networks* (AAAI submission): node-level OOD detection
with credal (interval-probability) predictions, 9 uncertainty baselines and 4 ablations, on 7 graphs.
This file is the single source of truth for coding agents (Claude Code, Codex, ...).
Humans: start from `README.md`.

## Environment

- Conda env **`gu`**: `conda activate gu` (or prefix commands with `conda run -n gu`).
  The package is installed editable (`pip install -e ".[dev]"`), so `cgnn` and `pytest` work from the repo root.
- Shared 8-GPU server. **Never use more than 2 GPUs in total**, across all processes you start
  (see `.claude/rules/gpu-etiquette.md`). `cgnn` pins one free GPU by default and refuses `devices > 2`.
- W&B project `graph-uncertainty` (sweeps/runs are shared with humans). Smoke tests: `--wandb disabled`.

## Commands

```bash
cgnn list                                   # methods (with paper names) and datasets (with file status)
cgnn describe <method|dataset>              # config defaults, sweep space, class split
cgnn doctor                                 # deps, data files, backbone checkpoints, GPUs, W&B login
cgnn run -m <method> -d <dataset> [--set k=v ...] [--seeds 0 1 2] [--wandb disabled] [--cpu]
cgnn sweep -m <method> -d <dataset> -c <count> [--sweep-id ID]   # W&B sweep + agent
cgnn results [-d ...] [-m ...] [--format md|latex|csv]           # W&B -> Table-2-style table
python -m pytest -q                         # ~15 s on CPU, runs every method end-to-end on a synthetic graph
ruff check cgnn tests && ruff format cgnn tests
python main.py -m <method> -d <dataset> -c N  # legacy interface == cgnn sweep
```

Fast end-to-end check of any method (CPU, seconds, no W&B):
`cgnn run -m credal_LJ_dual_head_detached -d synthetic --cpu --set max_epochs=3`

## Repository map

```
cgnn/
  uncertainty/   pure math: interval softmax (Eq.4), reachable bounds, min/max entropy -> TU/AU/EU (Eq.2),
                 credal DRO loss (Eq.9-10), ensemble decompositions
  data/          DatasetSpec registry (paper Table 1), leave-out-class OOD split, random splits + fingerprints,
                 loaders; one file per source in data/sources/ (+ `synthetic` for tests)
  models/        LightningModules on a common base (base.py): vanilla.py, credal.py (CGNN + ablations),
                 frozen.py (CGNN post train), cagcn.py, graph_esn.py, detectors/ (post-hoc baselines)
  methods/       MethodSpec registry: legacy method keys -> builders (trainable.py, posthoc.py)
  runner.py      THE generic runner (seed -> build -> data -> fit/validate/test -> metrics)
  config.py      config resolution + sweep construction; metrics.py (exact AUROC); checkpoints.py
  cli.py         `cgnn` CLI; results.py (W&B aggregation); hardware.py (GPU pinning); wandb_utils.py
configs/defaults.yaml            global switches (seed, split_seed, auroc_impl, test_ckpt, devices...)
configs/methods/<method>.yaml    per-method `defaults` (for `cgnn run`) + W&B `sweep`
tests/           pytest suite (unit + end-to-end contract tests for every method)
docs/            protocol.md, paper_to_code.md, known_issues.md, architecture.md;
                 status/ = ongoing work and handoff notes (READ FIRST when resuming: campaign_v02.md)
scripts/         reproduce_paper.sh (ordered pipeline), run_all.sh (legacy notes)
graph-ebm/       git submodule (GEBM baseline, `graph_uq`)
dataset/ checkpoints/ wandb/ outputs/ graph-uncertainty/   LARGE ARTIFACTS - gitignored, never delete
```

## Paper <-> code (method keys are legacy names, kept for W&B continuity)

| Paper                  | method key                       | Paper              | method key    |
|------------------------|----------------------------------|--------------------|---------------|
| **CGNN**               | `credal_LJ_dual_head_detached`   | Energy / ODIN      | `energy` / `odin` |
| CGNN last layer        | `credal`                         | Mahalanobis / KNN  | `mahalanobis` / `knn` |
| CGNN only credal       | `credal_LJ`                      | GNNSafe / GEBM     | `gnnsafe` / `gebm` |
| CGNN post train        | `frozen`                         | JLDE               | `knn_LJ` |
| CGNN by ensemble       | `ensemble` (`*_credal` metrics)  | CaGCN              | `cagcn` |
| Classical ensemble     | `ensemble` (`*_classic` metrics) | backbone of post-hoc methods | `vanilla` |

Details, equations and known paper/code mismatches: `docs/paper_to_code.md`.

## Non-negotiable rules (full text in `.claude/rules/`)

1. **GPUs**: at most 2 in total; pin with `CUDA_VISIBLE_DEVICES`; check `nvidia-smi` first; prefer CPU for tests.
2. **Artifacts**: never delete/overwrite `dataset/`, `checkpoints/`, `wandb/`, `graph-uncertainty/`, `outputs/runs/`.
   Never commit data, checkpoints or W&B files.
3. **Protocol**: labels of OOD nodes are all-zero rows (`y.sum(1)==0`); OOD target = 1; higher score = more OOD.
   Model selection uses validation only; never read test metrics to choose anything.
4. **Metric names are a contract** (`val_auroc_EU`, `test_auroc`, ...): sweeps optimise them and `cgnn results`
   reads them. Compute AUROC once per epoch on concatenated scores with `cgnn.metrics.binary_auroc`.
5. **Reproducibility**: the legacy numbers are reproducible bit-for-bit with
   `--set auroc_impl=torchmetrics split_seed=legacy`. Keep module construction order (RNG) when refactoring
   models; add/adjust tests; run `pytest` before declaring anything done.
6. **Anonymity**: this branch may be released anonymously. No names, e-mails, W&B entities, hostnames or
   absolute home paths in tracked files.
7. **Scope**: don't change scientific behaviour silently. New behaviour goes behind a config switch with the
   legacy value as default unless the user asks otherwise; document it in `docs/known_issues.md`/`CHANGELOG.md`.

## Extending (step-by-step procedures live in `.claude/skills/`, also exposed as `.agents/skills/`)

- New method -> `add-method` skill: builder in `cgnn/methods/`, model in `cgnn/models/`, `configs/methods/<name>.yaml`.
  `tests/test_runner.py` automatically runs it end-to-end and checks its metric contract.
- New dataset -> `add-dataset` skill: one file in `cgnn/data/sources/` with a `DatasetSpec`.
- Experiments -> `run-experiments`; tables -> `collect-results`; before finishing -> `verify-change`;
  paper consistency -> `paper-sync`; failures -> `debug-run`.

## Known issues you must keep in mind (see `docs/known_issues.md`)

- torchmetrics AUROC saturates scores outside [0,1] (legacy): affects Energy (T>=100), Mahalanobis, GEBM numbers.
- Legacy random splits (arxiv, patents, coauthor) depended on model-init RNG; post-hoc methods evaluated on a
  different split than their backbone was trained on. Fixed by `split_seed` + checkpoint split fingerprints.
- Paper Eq. 14 includes `z^0` in the joint latent; the CGNN code did not (`credal_LJ0_dual_head_detached` does).
- The AU (min-entropy) solver is a greedy heuristic (`min_entropy_method=greedy`); `auto` is exact for C<=6.
- In CGNN runs AU and EU are near mirror images (AUROC(EU)+AUROC(AU) ~ 1): don't present them as independent.
