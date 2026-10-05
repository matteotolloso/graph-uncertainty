# Campaign v0.2 — status and handoff

Re-run of every experiment the 0.2 audit found affected (see `docs/known_issues.md`). Written for the next
agent/human picking this up. Last updated: **2026-10-01 10:40 CEST**.

## What is running
- Queue: `scripts/gpu_queue.py` on `scripts/campaigns/v02.jobs` (92 jobs), started 2026-09-28 15:12, launched with
  `nohup conda run --no-capture-output -n gu python scripts/gpu_queue.py scripts/campaigns/v02.jobs --log-dir outputs/logs/v02 --prefer 6 7`.
  At most 2 GPUs (user's hard limit), 2 CPU slots.
- Monitor: `tail outputs/logs/v02_queue.log`; status per job `outputs/logs/v02/state.json`; job logs
  `outputs/logs/v02/<job>.log`. Is it alive: `pgrep -f gpu_queue.py`.
- W&B project **`graph-uncertainty-v02`** (separate from the legacy `graph-uncertainty` project, so sweeps with
  the same `<dataset>_<method>` names don't mix). New backbones: `checkpoints/v02/vanilla/`.
- Protocol: `auroc_impl=exact`, `split_seed=0`, `test_ckpt=last`, `min_entropy_method=greedy` (last two as in the
  paper). Squirrel/Reddit2 post-hoc use the paper's legacy backbones (`checkpoints/*.ckpt`, fixed splits).
  Patents and Coauthor GPU jobs use `deterministic=false` (40 GB GPUs, known_issues M8).

## Progress at last update
90/93 jobs done (the two `failed` entries are the old `knn-patents`/`knn_LJ-patents`, stopped on purpose); only
`credal_last_dual_head_detached-reddit2` is running (39/50 runs at 10:30, ~21 min/run -> ends ~14:30 on 2026-10-01).
- All Patents re-runs, the M1 ablation (all 7 datasets) and the M2 ablation (all but Reddit2) are done, with no
  OOM and no crashes except the LJ0 ones below. Top-ups ran with the fixed code: Coauthor LJ0 +1 run, Reddit2 LJ0
  +5 runs (the old sweep process crashed 5 times in total), Patents M2 needed none. No crash in any top-up run.
- LJ0 crashes (`Sum of lower bounds for a node cannot exceed 1`, Coauthor 1, Reddit2 3): investigated and fixed
  2026-09-30 (float32 rounding of collapsed credal intervals at high lr; known_issues M1). Running sweep processes
  keep the old code; the top-up jobs at the end of `v02.jobs` replace every failed run of the affected sweeps with
  the fixed code (no-op when none failed). Checked on the saved checkpoints: no selected top-5 run has
  rounding-level intervals; some are nearly collapsed (Patents/Amazon LJ0, Patents last-dual), see known_issues.
- **Still to do at the end:** tag the old sweeps `patents_odin`, `patents_mahalanobis`, `patents_knn`,
  `patents_knn_LJ` `superseded` in W&B, then export the final table (CSV + LaTeX) to `outputs/tables/`.
- Known losses (harmless): first Coauthor vanilla sweep lost 19 runs to OOM (SAGE, deterministic) -> replaced by
  `vanilla-coauthor-2`; its first Energy sweep was re-run as `energy-coauthor-2` and the old runs are tagged
  `superseded` in W&B (`cgnn results` skips them). 1 ArXiv vanilla run lost to a full root disk.

## Preliminary results (2026-09-28 23:20, test AUROC, top-5 by val; `outputs/tables/v02_partial_20260928.csv`)
| Method | squirrel | arxiv | patents | amazon_ratings | roman_empire | coauthor | reddit2 |
|---|---|---|---|---|---|---|---|
| Energy | 47.89 ± 0.45 | 69.69 ± 0.94 | - | 49.44 ± 0.09 | 74.86 ± 0.11 | 94.25 ± 1.58 | 69.83 ± 1.14 |
| KNN | - | 61.58 ± 0.13 | - | 49.47 ± 0.42 | 70.65 ± 0.40 | 83.58 ± 1.59 | - |
| ODIN | - | 72.67 ± 0.01 | - | 51.00 ± 0.03 | 75.13 ± 0.11 | 94.50 ± 0.03 | - |
| Mahalanobis | 59.55 ± 1.10 | 58.86 ± 4.02 | - | 48.14 ± 0.14 | 67.47 ± 3.37 | 87.23 ± 9.59 | 65.26 ± 0.04 |
| GNNSafe | - | 51.86 ± 9.33 | - | 49.38 ± 0.02 | 73.40 ± 1.29 | 94.50 ± 0.09 | - |
| JLDE | - | 55.00 ± 0.13 | - | 49.00 ± 0.30 | 66.68 ± 2.68 | 64.71 ± 0.16 | - |
| GEBM | 41.82 ± 0.33 | 56.29 ± 3.22 | - | 48.59 ± 0.24 | 55.08 ± 2.87 | 82.03 ± 9.56 | 52.28 ± 13.51 |
| CaGCN | - | 68.24 ± 0.23 | - | 52.86 ± 0.03 | 72.56 ± 1.24 | 93.70 ± 0.15 | - |
| Classical ensemble | - | 68.52 ± 0.14 | - | 49.45 ± 0.02 | 73.31 ± 0.61 | 96.50 ± 0.09 | - |
| CGNN | - | 70.38 ± 0.52 (EU) | 59.80 ± 8.29 | - | - | 79.88 ± 3.81 | - |
| CGNN last layer | - | 65.86 ± 0.51 | 59.19 ± 4.15 (partial) | - | - | 62.62 ± 7.99 | - |
| CGNN by ensemble | - | 67.88 ± 0.06 | - | 49.15 ± 0.02 | 73.32 ± 0.74 | 95.76 ± 0.12 | - |
| CGNN post train | - | 64.84 ± 2.58 | - | 51.13 ± 0.88 | 73.48 ± 0.84 | 65.60 ± 4.92 | - |
| CGNN only credal | - | 63.05 ± 2.03 | - | - | - | 69.60 ± 9.08 | - |

"-" = not re-run in this campaign (still valid in the legacy project: CGNN rows on the fixed-split datasets, the
unaffected baselines on Squirrel/Reddit2) or not finished yet. Legacy CGNN (project `graph-uncertainty`, via
`cgnn results`): Squirrel 73.96 ± 0.84, ArXiv 72.11 ± 0.40 (paper: 75.46 ± 2.31, 72.51 ± 0.94).

First observations (to confirm when complete):
- **ArXiv, same split for everybody: ODIN 72.67 > CGNN 70.38** (CGNN val-selected EU). The paper's ArXiv
  advantage may not survive the fixed protocol.
- Coauthor (homophilic): ensembles/ODIN/GNNSafe ~94-96 vs CGNN ~80, consistent with the paper's weaker
  homophilic results.
- Energy on Squirrel is 47.89 (legacy torchmetrics AUROC reported 50.00 = saturated ties); Mahalanobis on
  Reddit2 65.26 vs 62.90 legacy.

## Next steps
1. Let the queue finish; check with the commands above. If the queue died, relaunch the same command: finished
   jobs are skipped, interrupted ones restart (as new sweeps). If a job failed, read its log (`debug-run` skill).
   A sweep job exits 0 even when runs crash: grep logs for `Traceback`/`OutOfMemoryError`.
2. Build the final table: `cgnn results -p graph-uncertainty-v02 --details` (+ `--format latex`), combine with the
   still-valid legacy rows (Squirrel/Amazon/Roman/Reddit2 CGNN from project `graph-uncertainty`), and compare with
   the paper (`results-analyst` agent / `collect-results` skill). Also report CGNN with EU fixed a priori
   (known_issues M7) and the M1/M2 ablations (`-m credal_LJ0_dual_head_detached credal_last_dual_head_detached`).
3. Update `docs/known_issues.md` (H1/H2 with measured effects) and tell the user which paper claims change,
   especially ArXiv.
4. Optional follow-ups the user has not decided on: seed replicates of the best configs (`cgnn run --from-sweep
   <id> --seeds 0 1 2 3 4`), `test_ckpt=best`, `min_entropy_method=auto`, unbounded credal heads (M4).

## Environment notes
- Conda env `gu`; the shared root disk (`/`, `/tmp`, `/home`) is nearly full (other users). Jobs source
  `scripts/campaigns/env.sh` to keep TMPDIR/W&B caches on `/raid`.
- Git: `main` == `origin/main` at `16317ec` (pushed by the user; this agent cannot push: no credentials).
  The `AAAI` branch is the anonymous submission snapshot; don't touch it.
