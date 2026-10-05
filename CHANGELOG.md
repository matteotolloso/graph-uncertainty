# Changelog

## Unreleased
- `scripts/gpu_queue.py`: job queue with dependencies, at most 2 concurrent GPU jobs, resumable state;
  `scripts/campaigns/v02.jobs`: re-run of the experiments affected by the audit (W&B `graph-uncertainty-v02`).
- Sweep agents no longer leak GPU memory after a crashed run (the failure is logged, the run marked failed).
- `cgnn results` skips runs tagged `superseded` in W&B.
- KNN/JLDE: `knn_backend` switch (`faiss` default = legacy; `torch` = exact chunked search on the backbone
  device, same scores, used for Patents on GPU).
- Credal-set sanity checks (`cgnn/uncertainty/entropy.py`) tolerate float32 rounding of collapsed intervals
  (`min(1e-3, max(1e-6, C^2 eps32))` instead of `1e-6`): fixes the `Sum of lower bounds for a node cannot exceed 1`
  crashes of `credal_LJ0_dual_head_detached` at high lr. Inputs that passed before give identical values.
- `scripts/campaigns/count_failed_runs.py` + top-up jobs in `v02.jobs` replacing the crashed sweep runs.
- Extra test metrics for every method (new names, existing ones unchanged): AUPR/FPR@95 with OOD and with ID as
  positive class, misclassification-detection AUROC, mean score on ID/OOD nodes (`NodeUQModule.log_test_extras`,
  `cgnn.metrics.binary_aupr`/`fpr_at_tpr`; docs/protocol.md).
- Training-label perturbations `train_fraction`, `label_noise`, `perturb_seed` (`cgnn/data/perturb.py`; defaults leave
  data and split fingerprint unchanged) and synthetic contextual-SBM datasets `csbm_h1..h9` for the homophily study.
- Test-time feature shift `test_feature_noise` / `test_shift_fraction` / `test_shift_seed` (default `[]` = off): extra
  tests after the normal one, logged as `fshift_<sigma>_test_*` (clean vs feature-shifted ID test nodes;
  docs/protocol.md). `test_*` metrics are unchanged; `cgnn results` treats the switches as non-hyper-parameters.
- `cgnn run --sweep-project`: read `--from-sweep` from another W&B project than the one logged to.
- `scripts/analysis/feature_sparsity.py` (node-feature sparsity per dataset, on top of `cgnn.data`); results in
  `docs/results/feature_sparsity/` (CSV, JSON, LaTeX).

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
