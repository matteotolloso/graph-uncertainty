"""Print how many runs of a W&B sweep failed or crashed (used to top up sweeps after a fix).

Usage: python scripts/campaigns/count_failed_runs.py <project> <sweep_id>
The entity is the logged-in user's default one.
"""

from __future__ import annotations

import sys

import wandb


def main() -> int:
    project, sweep_id = sys.argv[1:3]
    api = wandb.Api()
    sweep = api.sweep(f"{api.default_entity}/{project}/{sweep_id}")
    print(sum(run.state in ("failed", "crashed") for run in sweep.runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
