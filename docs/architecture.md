# Architecture

```
cgnn run/sweep ──► cgnn.config.resolve_config ──► cgnn.runner.run_experiment
                    defaults.yaml                    │
                    methods/<m>.yaml                 ├─ seed_everything(seed)
                    DatasetSpec metadata             ├─ MethodSpec.build(ctx) ──► LightningModule (cgnn.models)
                    overrides (W&B / --set)          ├─ cgnn.data.load_dataset ──► GraphBundle (data, loaders, fingerprint)
                                                     ├─ [post-hoc] select_backbones + check_backbone_split + prepare()
                                                     ├─ Lightning Trainer (EarlyStopping, ModelCheckpoint -> $CGNN_CKPT_DIR/<m>/)
                                                     └─ metrics dict + outputs/runs/<...>.json (+ W&B via WandbLogger)
```

## Layers (lower layers never import higher ones)
1. `cgnn.uncertainty`, `cgnn.metrics` — pure tensor math (no Lightning, no W&B).
2. `cgnn.data` — `DatasetSpec` registry, raw sources, OOD transform, splits, loaders.
3. `cgnn.models` — LightningModules on `NodeUQModule` (masks, epoch buffers, metrics, checkpoint metadata).
4. `cgnn.methods` — `MethodSpec` registry: how to build each method from a resolved config.
5. `cgnn.runner`, `cgnn.checkpoints`, `cgnn.config` — orchestration.
6. `cgnn.cli`, `cgnn.wandb_utils`, `cgnn.results`, `cgnn.hardware` — interfaces.

## Extension points
| Want to add | Touch | Discovered by |
|---|---|---|
| dataset | `cgnn/data/sources/<name>.py` + import in `sources/__init__.py` | CLI, sweeps, doctor, tests |
| method | model in `cgnn/models/`, `register_method` in `cgnn/methods/`, `configs/methods/<name>.yaml` | CLI, sweeps, `tests/test_runner.py` |
| CGNN variant | one tuple in `cgnn/methods/trainable.py::_CREDAL_METHODS` + YAML | same |
| global switch | `configs/defaults.yaml` + read it in the runner/model | logged to W&B automatically |
| table/analysis | `cgnn/results.py` (pure functions) | `cgnn results` |

## Invariants enforced by tests
- every registered method has a config, logs its sweep metric and `test_auroc{_key}` for each `score_keys`;
- `defaults.monitor == sweep.metric.name`;
- seeded runs are reproducible; checkpoints carry split metadata; mismatched backbones are refused;
- no more than 2 GPUs per run; OOD labels never reach training; AUROC is exact and scale-invariant.

## Checkpoint compatibility
`VanillaGNN` keeps the legacy hyper-parameter names and `gnn_model.*` state-dict keys, so every file in
`checkpoints/` loads. Legacy credal checkpoints (in `graph-uncertainty/<run>/checkpoints/`) load with
`CredalGNN.load_from_checkpoint(path, latent=..., heads=..., detach_credal=...)` using the table in
`cgnn/models/credal.py`.
