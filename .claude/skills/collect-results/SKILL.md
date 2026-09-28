---
name: collect-results
description: Fetch runs from the W&B project and build paper-style result tables (Table 2: AUROC mean ± std, validation-selected AU/EU component), compare methods, or compare with the numbers in the paper. Use when the user asks for results, tables, LaTeX, "how did X do", or to check/update paper numbers.
---

# Collect results

`cgnn results` = `cgnn.results.fetch_runs` (read-only W&B API) + `paper_table` (pure, unit-tested) +
`format_table`. Runs are attributed to `(dataset, method)` through the sweep name `<dataset>_<method>` or,
for `cgnn run`, the config keys `cgnn_dataset`/`cgnn_method`.

```bash
cgnn results                                        # full Table 2 (7 datasets x 14 rows), markdown
cgnn results -d squirrel arxiv -m credal_LJ_dual_head_detached energy --details
cgnn results --format latex --out outputs/tables/table2.tex
cgnn results --aggregate group                      # best hyper-parameter group, mean over its seeds
```

## Protocol (what the numbers mean) — state it when you report
- `--aggregate topk -k 5` (default): the 5 runs with the best validation score of the component, test
  mean ± std over those runs (this mixes hyper-parameter configurations).
- `--aggregate group`: runs grouped by config minus replicate keys (`seed`, `backbone_rank`, `split_seed`);
  the group with the best mean val is reported. Use it after `cgnn run --seeds ...`.
- Component (EU vs AU; `*_classic` / `*_credal` for ensembles) is chosen on **validation** only.
- Values are AUROC x 100. Best test mean per column is bold.

## Pitfalls to flag to the user
- Runs made before cgnn 0.2 (no `cgnn_version` in their config) used the torchmetrics AUROC, which saturates
  large scores: Energy (T >= 100), Mahalanobis and GEBM values from those runs may be wrong (often exactly 0.5).
- Pre-0.2 post-hoc runs on arxiv/patents/coauthor evaluated on a different random split than their
  backbone was trained on (leakage); treat them as unreliable.
- Mixed protocols in one table (legacy vs new runs, `test_ckpt` last vs best) make columns incomparable:
  filter with `--details` and inspect the run ids.
- Crashed/killed runs are ignored; a cell `-` means no finished run with both val and test metrics.

## Custom analyses
Use the library directly (pandas optional):
```python
from cgnn.results import fetch_runs, summarize
from cgnn.data import DATASETS
rows = fetch_runs("graph-uncertainty", datasets=["squirrel"], known_datasets=DATASETS.names())
s = summarize(rows, "credal_LJ_dual_head_detached", "squirrel", ("EU", "AU"), aggregate="topk", k=5)
```
Save tables under `outputs/tables/`. Never write to W&B from analysis code.
