"""GPU selection on a shared multi-GPU server.

Policy: at most 2 GPUs per run.
`cgnn run`/`cgnn sweep` call :func:`pin_gpus` before CUDA is initialised: if
``CUDA_VISIBLE_DEVICES`` is unset they pick the least-loaded GPU(s) via
``nvidia-smi`` and pin the process to them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

from cgnn.config import MAX_GPUS


@dataclass(frozen=True)
class GPUStatus:
    index: int
    memory_used_mib: int
    memory_total_mib: int
    utilization: int

    @property
    def is_idle(self) -> bool:
        return self.memory_used_mib < 1024 and self.utilization < 10


def query_gpus() -> list[GPUStatus]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        idx, used, total, util = (int(float(v)) for v in line.split(","))
        gpus.append(GPUStatus(idx, used, total, util))
    return gpus


def pin_gpus(count: int) -> list[int] | None:
    """Pin the process to ``count`` free GPUs unless CUDA_VISIBLE_DEVICES is already set.

    Returns the chosen indices, or ``None`` when nothing was changed.
    """
    if count > MAX_GPUS:
        raise ValueError(f"Refusing to use {count} GPUs: the shared-server limit is {MAX_GPUS}.")
    if count <= 0 or "CUDA_VISIBLE_DEVICES" in os.environ:
        return None
    gpus = sorted(query_gpus(), key=lambda g: (not g.is_idle, g.memory_used_mib, g.utilization))
    if not gpus:
        return None
    chosen = [g.index for g in gpus[:count]]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, chosen))
    busy = [g.index for g in gpus[:count] if not g.is_idle]
    note = f" (warning: GPU {busy} already busy)" if busy else ""
    print(f"Pinned CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}{note}")
    return chosen
