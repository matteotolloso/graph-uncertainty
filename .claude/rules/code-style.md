---
paths:
  - "**/*.py"
---

# Python conventions

- Python 3.10+, type hints on public functions, `from __future__ import annotations`. Line length 110.
  `ruff check` + `ruff format` must be clean (config in `pyproject.toml`).
- Match the surrounding code: short docstrings that say *what the paper equation / legacy behaviour is*,
  comments only where the code is not obvious.
- Imports: absolute (`from cgnn.x import y`). **No `sys.path` manipulation.** Optional dependencies
  (`faiss`, `ogb`, `graphesn`, `graph_uq`, `torch_sparse`, `pandas`) are imported lazily inside the function or
  constructor that needs them and declared in `MethodSpec.requires` / `pyproject.toml` extras.
- Registries, not if/elif chains: datasets via `register_dataset(DatasetSpec(...))`, methods via
  `register_method(MethodSpec(...))`. The CLI, sweeps, doctor and tests discover everything from them.
- Hyper-parameters come from the resolved config (`configs/defaults.yaml` -> `configs/methods/<m>.yaml` ->
  dataset metadata -> overrides). Don't hard-code them in trainers; new keys get a default in YAML.
- LightningModules derive from `cgnn.models.base.NodeUQModule` (or `TrainableUQModule`), call
  `save_hyperparameters()`, use `split_mask`, `EpochBuffer`, `self.auroc`, `self.f1`.
- **Reproducibility**: module construction order defines the RNG stream. When refactoring a model keep the
  order `backbone -> credal head -> classifier head -> weights_init`; run the seeded-reproducibility test.
- Keep state-dict attribute names (`gnn_model`, `credal_layer_model`, `classifier_head`, `backbone`, ...) so
  old checkpoints keep loading.
- Tests: every new behaviour gets a test in `tests/` (CPU, synthetic data, seconds). No network, no GPU, no
  real checkpoint dirs in tests.
- No new top-level scripts: add a `cgnn` subcommand or a file under `scripts/`. Throwaway analysis goes to the
  scratchpad or `notebooks/` (outputs cleared).
