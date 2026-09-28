# Changelog

## 0.2.0 — agent-ready refactor
Verified equivalence: with `--set auroc_impl=torchmetrics split_seed=legacy` every one of the 20 legacy
methods reproduces the legacy trainers' metrics exactly (Squirrel + ArXiv, CPU); all 9 dataset loaders
produce identical tensors and masks.

### Structure
- New installable package `cgnn/` (`pip install -e ".[dev]"`) replacing `models/`, `trainers/`,
  `dataset_loader/`, `utils/`, `sweeps/`: dataset and method registries, one generic runner, `cgnn` CLI
  (`list`, `describe`, `data`, `run`, `sweep`, `results`, `doctor`). `main.py` kept as a thin shim.
- Sweeps moved to `configs/methods/<method>.yaml` (identical search spaces) with `defaults` for single runs;
  global switches in `configs/defaults.yaml`.
- Six credal classes unified into `CredalGNN` (bit-identical); detectors share `PostHocDetector`.
- Tests (`pytest`, ~15 s CPU), ruff config, `environment.yml`, `.gitmodules` for `graph-ebm`, Docker update.
- Agent layer: `AGENTS.md`, `CLAUDE.md`, `.claude/{rules,skills,agents,hooks,settings.json}`.

### Behaviour changes (defaults; legacy values still selectable)
- `auroc_impl=exact`: rank-based AUROC (torchmetrics saturated large scores, docs/known_issues.md H1).
- `split_seed=0`: random splits no longer depend on model-init RNG; checkpoints store split fingerprints;
  post-hoc methods refuse mismatched backbones (H2).
- Epoch-level AUROC for all modules (post-hoc detectors averaged per-batch AUROCs on Reddit2, H3);
  NeighborLoader depth follows the backbone; GEBM always evaluates on the full graph.
- `devices=1` by default, hard cap 2 (Lightning `auto` used every visible GPU).
- Trainable methods save their best checkpoint to `$CGNN_CKPT_DIR/<method>/` (vanilla backbones:
  `checkpoints/vanilla/`; the legacy flat files are still found).
- `seed`-as-backbone-index of energy/gebm/cagcn/frozen sweeps renamed `backbone_rank` (legacy key mapped).

### New options / methods
- `min_entropy_method` (`greedy` | `multistart` | `exact` | `auto`) for AU.
- `CredalGNN` switches: `latent`, `heads`, `detach_credal`, `joint_include_input`, `joint_act_on_last`,
  `credal_mid_activation`, `credal_half_activation`.
- Methods `credal_last_dual_head_detached` (clean last-layer ablation) and `credal_LJ0_dual_head_detached`
  (paper Eq. 14 with `z^0`).
- `synthetic` dataset for tests/smoke runs; `test_ckpt=best`; run summaries in `outputs/runs/`.
