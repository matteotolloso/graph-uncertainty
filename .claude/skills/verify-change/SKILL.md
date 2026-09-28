---
name: verify-change
description: Checklist to verify any code change in this repo before declaring it done — lint, unit/contract tests, CPU smoke runs, reproducibility of legacy numbers, and an ML-correctness review. Use after editing anything under cgnn/, configs/ or tests/, and before committing.
---

# Verify a change

Run in order; stop and fix at the first failure. Everything here is CPU-only and takes ~1 minute.

1. **Lint/format**: `ruff format cgnn tests && ruff check cgnn tests`
2. **Tests**: `python -m pytest -q` (all methods run end-to-end on the synthetic graph; contract checks on
   metric names, configs, GPU cap, split fingerprints, seeded reproducibility).
   With real data available, also `python -m pytest -q -m slow`.
3. **Smoke run of what you touched**:
   `cgnn run -m <method> -d synthetic --cpu --set max_epochs=3` (prints val/test AUROCs).
4. **Did numbers change?** If you touched `cgnn/uncertainty`, `cgnn/models`, `cgnn/data`, `cgnn/runner.py`
   or `configs/`, compare against the previous commit on a fixed config:
   ```bash
   cgnn run -m credal_LJ_dual_head_detached -d squirrel --cpu --wandb disabled \
       --set max_epochs=5 auroc_impl=torchmetrics split_seed=legacy > /tmp/after.txt
   git stash && (same command) > /tmp/before.txt && git stash pop && diff /tmp/before.txt /tmp/after.txt
   ```
   For refactors of models/data/runner, also run the full legacy check (~4 min, CPU):
   `python scripts/dev/legacy_equivalence.py` (every line must be `[OK]`).
   Identical output is expected unless the change is *meant* to alter results — then say so explicitly and
   document it (`CHANGELOG.md`, `docs/known_issues.md` or `docs/protocol.md`).
5. **Review**: delegate the diff to the `ml-reviewer` subagent (data leakage, split/backbone consistency,
   metric aggregation, RNG/init order, detach/gradient flow, paper consistency).
6. **Docs**: README/AGENTS tables if you added a method or dataset; config comments if you added keys.
7. Report: what changed, which checks ran (with their output), what was not verified (e.g. GPU, big datasets).
