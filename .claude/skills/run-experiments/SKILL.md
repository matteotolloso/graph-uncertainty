---
name: run-experiments
description: Launch, resume and monitor experiments (single runs, repeated seeds, W&B sweeps, the full paper pipeline) on the shared GPU server within the 2-GPU limit. Use whenever the user asks to train, run, sweep, re-run or reproduce results.
---

# Run experiments

## Before launching
1. `cgnn doctor` — data files, backbones per dataset, GPU occupancy, W&B login.
2. Count the GPUs you already use (`nvidia-smi`, your background jobs). **Total must stay <= 2.**
3. Decide the protocol knobs and say them to the user (they change the numbers, see `docs/protocol.md`):
   `split_seed` (0; `legacy` only to reproduce old runs), `auroc_impl` (exact), `test_ckpt` (last|best),
   `min_entropy_method` (greedy|auto).
4. Smoke test the exact command on `synthetic` (`--cpu`) or with `--set max_epochs=2 --wandb disabled`.

## Commands
```bash
# one run (prints val/test AUROCs, writes outputs/runs/<...>.json); pins 1 idle GPU
cgnn run -m credal_LJ_dual_head_detached -d squirrel --set lr=0.003 delta=0.8
# replicates of a config (separate W&B runs in one group)
cgnn run -m credal_LJ_dual_head_detached -d squirrel --seeds 0 1 2 3 4
# re-run the best config of a finished sweep with 5 seeds
cgnn run -m credal_LJ_dual_head_detached -d squirrel --from-sweep <SWEEP_ID> --seeds 0 1 2 3 4
# post-hoc replicates over backbones
cgnn run -m energy -d squirrel --repeat backbone_rank=0,1,2,3,4 --set temperature=1
# sweep: create + agent (one agent = one GPU); join from another shell with --sweep-id
cgnn sweep -m credal_LJ_dual_head_detached -d arxiv -c 100
cgnn sweep -m credal_LJ_dual_head_detached -d arxiv -s <SWEEP_ID> -c 100
# everything for the paper, in dependency order (prints commands; EXECUTE=1 runs them)
DATASETS="squirrel" bash scripts/reproduce_paper.sh
```
Legacy `python main.py -m <method> -d <dataset> -c N [-s ID]` still works (= `cgnn sweep`).

## Dependencies between methods
- Post-hoc methods (`energy odin mahalanobis knn knn_LJ gnnsafe gebm ensemble cagcn frozen`) need
  `vanilla` checkpoints of the same dataset **and split** (they select by `val_auroc`; `ensemble` needs >= M).
  A backbone trained with another `split_seed` is refused (`backbone_split_check=error`).
- Old backbones in `checkpoints/` (flat files) have no split metadata: fine for public-split datasets,
  **leaky** for arxiv/patents/coauthor (you get a warning) — retrain them with `split_seed` set.
- Reddit2 post-hoc: run with `--cpu` (full-graph forward needs ~52 GB); GEBM and JLDE default to CPU.

## Running in the background and monitoring
```bash
mkdir -p outputs/logs
OMP_NUM_THREADS=16 nohup cgnn sweep -m vanilla -d arxiv -c 100 > outputs/logs/arxiv_vanilla.log 2>&1 &
tail -n 50 outputs/logs/arxiv_vanilla.log ; nvidia-smi
```
From Claude Code prefer `run_in_background` / Monitor over `sleep` loops. Report the sweep id and the
log path to the user. Stop an agent with `kill <pid>` (only your own processes).

## Many jobs: the queue
`scripts/gpu_queue.py` runs a jobs file (`name needs gpu|cpu command` per line) with at most 2 concurrent
GPU jobs (each pinned to one idle GPU), CPU slots, dependencies (e.g. post-hoc after `vanilla`), per-job
logs and a resumable `state.json`. The file is re-read every poll: appended jobs, and edits to jobs that
have not started yet (commands, `needs`), take effect without restarting the queue.
```bash
conda activate gu && mkdir -p outputs/logs
nohup python scripts/gpu_queue.py scripts/campaigns/v02.jobs --log-dir outputs/logs/v02 \
    > outputs/logs/v02_queue.log 2>&1 &
cat outputs/logs/v02/state.json            # status per job; logs in outputs/logs/v02/<job>.log
```
- Stop gracefully with one `kill <queue pid>` (no new jobs), a second one terminates running jobs.
  Restarting while jobs still run re-launches them (they are marked `interrupted`): wait for them first.
- A sweep job exits 0 even if some of its runs crashed: grep its log for `Traceback`/`OutOfMemoryError`.
- To redo a finished job, add it under a new name; tag the old sweep's runs `superseded` in W&B
  (`cgnn results` skips them). Never delete runs.
- Jobs `source scripts/campaigns/env.sh` to keep TMPDIR and W&B caches off the small root disk.
- 40 GB GPUs: Patents and Coauthor need `--set deterministic=false` (known_issues M8).

## After
- `cgnn results -d <ds> -m <method> --details` to check the new runs are picked up.
- Local summaries: `outputs/runs/*.json` (config, metrics, data split, backbones) even without W&B.
