#!/usr/bin/env python
"""Run a queue of shell jobs with at most ``--gpu-slots`` (<= 2) GPUs in use at any time.

Jobs file, one job per line (``#`` comments, blank lines ignored)::

    # name          needs                 device  command
    vanilla-arxiv   -                     gpu     cgnn sweep -m vanilla -d arxiv -c 50 -p my-project
    energy-arxiv    vanilla-arxiv         gpu     cgnn sweep -m energy -d arxiv -p my-project
    energy-reddit2  -                     cpu     cgnn sweep -m energy -d reddit2 --cpu -p my-project

- ``needs``: ``-`` or comma-separated job names that must have finished successfully first
  (a failed dependency marks the job ``blocked``).
- ``device``: ``gpu`` jobs get ``CUDA_VISIBLE_DEVICES=<one idle GPU>``; ``cpu`` jobs get
  ``CUDA_VISIBLE_DEVICES=""``. Every job gets ``OMP_NUM_THREADS``.
- The command runs with ``bash -c`` from the repo root (``VAR=value cmd`` prefixes work); its output
  goes to ``<log-dir>/<name>.log``.

The file is re-read every poll, so jobs can be appended while the queue runs. State is kept in
``<log-dir>/state.json``: restarting the queue skips finished jobs and re-runs interrupted ones.

    nohup python scripts/gpu_queue.py jobs.txt --log-dir outputs/logs/queue > outputs/logs/queue.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from cgnn.config import MAX_GPUS
from cgnn.hardware import query_gpus
from cgnn.paths import REPO_ROOT


@dataclass
class Job:
    name: str
    needs: tuple[str, ...]
    device: str
    command: str


def parse_jobs(path: Path) -> list[Job]:
    jobs, seen = [], set()
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip() if raw.lstrip().startswith("#") else raw.strip()
        if not line:
            continue
        parts = line.split(None, 3)
        if len(parts) < 4 or parts[2] not in ("gpu", "cpu"):
            raise ValueError(f"{path}:{lineno}: expected 'name needs gpu|cpu command', got {raw!r}")
        name, needs, device, command = parts
        if name in seen:
            raise ValueError(f"{path}:{lineno}: duplicate job name {name!r}")
        seen.add(name)
        jobs.append(Job(name, () if needs == "-" else tuple(needs.split(",")), device, command))
    return jobs


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


class Queue:
    def __init__(self, args):
        self.args = args
        self.jobs_file = Path(args.jobs).resolve()
        self.log_dir = Path(args.log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.log_dir / "state.json"
        self.state: dict[str, dict] = (
            json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        )
        for st in self.state.values():
            if st["status"] == "running":
                st["status"] = "interrupted"
        self.running: dict[str, tuple[subprocess.Popen, int | None]] = {}
        self.stopping = False

    def save(self) -> None:
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True))
        tmp.replace(self.state_file)

    def free_gpu(self) -> int | None:
        ours = {gpu for _, gpu in self.running.values() if gpu is not None}
        if len(ours) >= self.args.gpu_slots:
            return None
        preferred = self.args.prefer or []
        idle = [g.index for g in query_gpus() if g.is_idle and g.index not in ours]
        idle.sort(key=lambda i: (i not in preferred, preferred.index(i) if i in preferred else i))
        return idle[0] if idle else None

    def start(self, job: Job, gpu: int | None) -> None:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
        env.setdefault("OMP_NUM_THREADS", str(self.args.threads))
        env.setdefault("MKL_NUM_THREADS", str(self.args.threads))
        logfile = self.log_dir / f"{job.name}.log"
        fh = open(logfile, "a")
        fh.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} gpu={gpu} :: {job.command}\n")
        fh.flush()
        proc = subprocess.Popen(
            ["bash", "-c", job.command],
            cwd=REPO_ROOT,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        fh.close()
        self.running[job.name] = (proc, gpu)
        self.state[job.name] = {
            "status": "running",
            "device": job.device,
            "gpu": gpu,
            "pid": proc.pid,
            "start": time.strftime("%Y-%m-%d %H:%M:%S"),
            "log": str(logfile),
            "command": job.command,
        }
        log(f"START {job.name} ({'gpu ' + str(gpu) if gpu is not None else 'cpu'}) pid={proc.pid}")

    def reap(self) -> None:
        for name, (proc, _) in list(self.running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del self.running[name]
            st = self.state[name]
            st.update(status="done" if rc == 0 else "failed", rc=rc, end=time.strftime("%Y-%m-%d %H:%M:%S"))
            log(f"{'DONE ' if rc == 0 else 'FAIL '} {name} rc={rc}")

    def status(self, name: str) -> str:
        return self.state.get(name, {}).get("status", "pending")

    def step(self) -> bool:
        """One scheduling round; returns False when nothing is left to do."""
        self.reap()
        jobs = parse_jobs(self.jobs_file)
        names = {j.name for j in jobs}
        pending = []
        for job in jobs:
            if self.status(job.name) not in ("pending", "interrupted"):
                continue
            missing = [n for n in job.needs if n not in names]
            deps = [self.status(n) for n in job.needs]
            if missing or any(s in ("failed", "blocked") for s in deps):
                self.state[job.name] = {"status": "blocked", "reason": f"deps {job.needs} missing/failed"}
                log(f"BLOCK {job.name} (dependency failed or unknown)")
                continue
            if all(s == "done" for s in deps):
                pending.append(job)
        if not self.stopping:
            cpu_busy = sum(1 for n in self.running if self.state[n]["device"] == "cpu")
            for job in pending:
                if job.device == "cpu":
                    if cpu_busy < self.args.cpu_slots:
                        self.start(job, None)
                        cpu_busy += 1
                else:
                    gpu = self.free_gpu()
                    if gpu is not None:
                        self.start(job, gpu)
        self.save()
        waiting = any(self.status(j.name) in ("pending", "interrupted") for j in jobs)
        return bool(self.running) or (waiting and not self.stopping) or self.args.follow

    def stop(self, *_):
        if self.stopping:
            for proc, _ in self.running.values():
                os.killpg(proc.pid, signal.SIGTERM)
            log("second signal: terminated running jobs")
        else:
            log("stopping: no new jobs will start; running jobs continue (signal again to kill them)")
        self.stopping = True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jobs", help="jobs file")
    p.add_argument("--log-dir", default="outputs/logs/queue")
    p.add_argument("--gpu-slots", type=int, default=MAX_GPUS, help=f"concurrent GPU jobs (<= {MAX_GPUS})")
    p.add_argument("--prefer", type=int, nargs="*", help="preferred GPU indices, in order")
    p.add_argument("--cpu-slots", type=int, default=2)
    p.add_argument("--threads", type=int, default=16, help="OMP/MKL threads per job")
    p.add_argument("--poll", type=float, default=30.0, help="seconds between scheduling rounds")
    p.add_argument("--follow", action="store_true", help="keep running and wait for appended jobs")
    args = p.parse_args(argv)
    if not 0 <= args.gpu_slots <= MAX_GPUS:
        p.error(f"--gpu-slots must be between 0 and {MAX_GPUS} (shared-server limit)")
    queue = Queue(args)
    parse_jobs(queue.jobs_file)  # fail fast on syntax errors
    signal.signal(signal.SIGTERM, queue.stop)
    signal.signal(signal.SIGINT, queue.stop)
    log(f"queue {queue.jobs_file} gpu_slots={args.gpu_slots} cpu_slots={args.cpu_slots}")
    while queue.step():
        time.sleep(args.poll)
    log(
        "queue finished: "
        + json.dumps(
            {s: sum(v["status"] == s for v in queue.state.values()) for s in ("done", "failed", "blocked")}
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
