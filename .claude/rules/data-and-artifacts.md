# Data, checkpoints and experiment artifacts

These directories hold weeks of compute and are gitignored. Treat them as read-only unless the user asks:

| Path | Content | Notes |
|---|---|---|
| `dataset/` | raw + processed datasets (4.4 GB) | re-downloading some (Reddit2, Patents) is slow/fragile |
| `checkpoints/` | VanillaGNN backbones (`<run>_<ds>_val_auroc=<s>.ckpt`, legacy flat layout) and `checkpoints/<method>/` (new runs) | post-hoc methods depend on them |
| `wandb/` | local W&B run files (14 GB) | synced to the shared W&B project |
| `graph-uncertainty/` | legacy Lightning default checkpoints (12 GB) | produced by pre-0.2 credal trainers |
| `outputs/` | `runs/*.json` run summaries, `lightning/`, `csv/`, `tables/`, `logs/` | safe to add to, don't wipe |

- Never `rm -rf`, move or rewrite files in these directories; never `git add` them.
- New checkpoints go to `$CGNN_CKPT_DIR/<method>/` (default `checkpoints/<method>/`). Tests must use
  temporary directories (`tmp_paths` / `backbone_paths` fixtures), never the real ones.
- Paths are configurable: `CGNN_DATA_DIR`, `CGNN_CKPT_DIR`, `CGNN_OUTPUT_DIR` (see `cgnn/paths.py`).
  Code must not depend on the current working directory.
- W&B is shared with humans: use `--wandb disabled` (or `offline`) for smoke tests and debugging, and don't
  create sweeps "just to try". Never delete W&B runs or sweeps.
- Untracked files at the repo root (e.g. `compute_feature_sparsity.py`, `results/`) belong to the user:
  leave them alone.
