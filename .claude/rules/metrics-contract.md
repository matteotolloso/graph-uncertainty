---
paths:
  - "cgnn/models/**"
  - "cgnn/metrics.py"
  - "cgnn/results.py"
  - "cgnn/uncertainty/**"
  - "configs/methods/**"
---

# Metric contract

Metric names are an API: W&B sweeps optimise them (`sweep.metric.name`), early stopping monitors them
(`defaults.monitor`), and `cgnn results` reads them. Renaming one silently breaks sweeps and tables.

## Names (keep exactly)
| family | validation | test |
|---|---|---|
| single-score methods (vanilla, energy, odin, mahalanobis, knn, knn_LJ, gnnsafe, gebm, cagcn) | `val_auroc` | `test_auroc` |
| credal models (credal*, frozen) | `val_auroc_{EU,AU,TU}` | `test_auroc_{EU,AU,TU}` |
| ensembles (ensemble, graph_esn) | `val_auroc_{EU,AU,TU}_{credal,classic}` | `test_auroc_{...}_{credal,classic}` |
| ID classification | `val_f1`, `val_acc`, `val_f1_{cls,U,L}` | `test_acc`, `test_f1`, `test_accuracy_{cls,U,L}`, `test_f1_{cls,U,L}` |

- The `score_keys` of a `MethodSpec` must list every uncertainty suffix a method logs; `tests/test_runner.py`
  checks `test_auroc{_key}` for each and that the sweep metric is logged.
- `*_f1` is micro-F1 (= accuracy), kept for continuity. Add new metrics with new names; never change meaning.

## Computation
- Collect scores/targets per step into `self.buffer(stage)` and compute AUROC **once per epoch** on the
  concatenation (`on_*_epoch_end`). Never log per-batch AUROC (Lightning would average it across batches).
- Use `cgnn.metrics.binary_auroc` (exact, rank-based). Do **not** use `torchmetrics` AUROC directly: it applies
  a sigmoid to scores outside [0, 1], which saturates large scores into ties (AUROC -> 0.5).
  `auroc_impl=torchmetrics` exists only to reproduce legacy numbers.
- Score convention: higher = more OOD. Negate confidence scores (e.g. `-MSP`).
- Uncertainties from credal sets: `cgnn.uncertainty.credal_uncertainties` (TU = max entropy, AU = min entropy,
  EU = TU − AU, bits). The AU solver is selected globally by `min_entropy_method`.
