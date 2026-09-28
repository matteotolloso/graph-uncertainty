# Sourced by every job of a campaign (run from the repo root): keep temporary files and W&B caches on
# /raid instead of the small, shared root disk (it filled up during the v0.2 campaign).
export TMPDIR="$PWD/outputs/tmp"
export WANDB_CACHE_DIR="$TMPDIR/wandb-cache"
export WANDB_DATA_DIR="$TMPDIR/wandb-data"
mkdir -p "$TMPDIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
