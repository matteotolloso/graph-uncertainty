"""Filesystem locations used by the project.

Every location can be overridden with an environment variable so that the
code never depends on the current working directory:

- ``CGNN_DATA_DIR``   raw/processed datasets        (default: ``<repo>/dataset``)
- ``CGNN_CKPT_DIR``   model checkpoints             (default: ``<repo>/checkpoints``)
- ``CGNN_OUTPUT_DIR`` Lightning logs, result tables (default: ``<repo>/outputs``)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    ckpt_dir: Path
    output_dir: Path

    @classmethod
    def from_env(cls) -> Paths:
        return cls(
            data_dir=Path(os.environ.get("CGNN_DATA_DIR", REPO_ROOT / "dataset")),
            ckpt_dir=Path(os.environ.get("CGNN_CKPT_DIR", REPO_ROOT / "checkpoints")),
            output_dir=Path(os.environ.get("CGNN_OUTPUT_DIR", REPO_ROOT / "outputs")),
        )
