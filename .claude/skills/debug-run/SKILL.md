---
name: debug-run
description: Diagnose failing or suspicious runs (crashes, CUDA OOM, NaNs, AUROC stuck at 0.5, missing backbones, split-mismatch errors, W&B/sweep problems, results that don't match the paper). Use when a run/sweep errors out or produces odd numbers.
---

# Debug a run

Reproduce first, small and local: same method/dataset/config with `--wandb disabled`, `--set max_epochs=2`,
on CPU when possible. The run summary `outputs/runs/*.json` contains the resolved config and data split.

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: No vanilla checkpoints` | post-hoc method without backbones | `cgnn sweep -m vanilla -d <ds>`; check `cgnn doctor` |
| `RuntimeError: Backbone/split mismatch` | backbone trained on another split (`split_seed`) | use the same `split_seed`, retrain backbones, or `--set backbone_split_check=warn` knowingly |
| `IndexError: Requested backbones ...` | `backbone_rank`/`M` larger than available checkpoints | train more vanilla runs or lower it |
| CUDA OOM on Reddit2/Patents post-hoc | full-graph forward (Reddit2 ~52 GB) | `--cpu`; for training keep `batch_size>0` (NeighborLoader) |
| CUDA OOM on Squirrel/Chameleon with SAGE | 2089-dim messages on ~200K edges (>32 GB, deterministic) | `--cpu` (small graphs) or `gnn_type=GCN` |
| EU AUROC ~= 1 - AU AUROC | TU saturated / AU and EU mirror each other (known_issues M7) | report it; don't read it as disentanglement |
| AUROC exactly 0.5 | constant scores (e.g. TU saturated at log2 C) or `auroc_impl=torchmetrics` on large scores | check score spread; use `auroc_impl=exact` |
| TU AUROC 0.5 for credal models | uniform distribution inside every credal set -> TU = log2 C for all nodes | expected with wide intervals; select EU/AU |
| AUROC < 0.5 consistently | score sign inverted (higher must mean OOD) | negate the score |
| NaN losses | lr too high, `log(0)` in custom losses | lower lr; use `probs_cross_entropy`/`clamp_min` |
| `devices ... exceeds the shared-server limit` | `devices>2` or `devices: auto` | 1 (or 2) GPUs only |
| run uses a busy GPU | `CUDA_VISIBLE_DEVICES` pre-set | unset it (auto-pin picks idle GPUs) or choose from `nvidia-smi` |
| numbers differ from the paper | protocol differences | compare `auroc_impl`, `split_seed`, `test_ckpt`, `min_entropy_method`; see `docs/known_issues.md` |
| sweep agent runs wrong code | agent started before a code change | restart the agent (it imports code once) |
| W&B `permission denied`/offline | credentials | `wandb login`; `cgnn doctor` shows credential status |

## Useful probes
```bash
cgnn data <ds>                                  # split counts + fingerprint
cgnn describe <method>                          # effective defaults and sweep space
python -c "from cgnn.checkpoints import find_checkpoints as f; r=f('<ds>'); print(len(r), r[0].path, r[0].meta)"
python -m pytest -q tests/test_runner.py -k <method>
```
When you find a real bug: add a regression test, fix it behind a config switch if it changes results,
and record it in `docs/known_issues.md`.
