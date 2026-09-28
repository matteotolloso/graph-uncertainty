---
name: add-method
description: Add a new uncertainty/OOD method (trainable model, post-hoc detector on the vanilla backbones, or a new CGNN variant/ablation) to the cgnn package so it works with `cgnn run`, W&B sweeps, `cgnn results` and the test suite. Use whenever the user asks to implement, try or benchmark a new model, baseline, ablation or head.
---

# Add a method

A method = a LightningModule (`cgnn/models/`) + a registered `MethodSpec` (`cgnn/methods/`) + a config
(`configs/methods/<name>.yaml`). The generic runner does seeding, data, trainer, checkpoints, metrics.

## 0. Decide the kind
| You want | Kind | Template |
|---|---|---|
| a CGNN variant (other latent, heads, detach, activations, loss weights) | trainable, **config only** | step 1C |
| a new model trained from scratch | `trainable` | `templates/trainable_model.py` |
| a score computed from a frozen pre-trained VanillaGNN | `posthoc` | `templates/posthoc_detector.py` |
| post-hoc + a trained head (like `frozen`, `cagcn`) | `posthoc` with `fit_on="train"/"val"` | copy `cgnn/models/frozen.py` |

Pick a **snake_case method key**; it becomes the CLI name, the YAML file name and the W&B sweep suffix
(`<dataset>_<method>`). Never reuse or rename existing keys (W&B history depends on them).

## 1A. Trainable model
1. Copy `templates/trainable_model.py` to `cgnn/models/<name>.py`. Derive from `TrainableUQModule`; call
   `save_hyperparameters()`; build submodules in a fixed order; use `self.split_mask`, `self.buffer(stage)`,
   `self.auroc`, `self.f1`; log metrics per the contract (`.claude/rules/metrics-contract.md`).
2. Builder + registration in `cgnn/methods/trainable.py` (lazy import of the model inside the builder):
   ```python
   def build_mymodel(ctx: RunContext):
       from cgnn.models.mymodel import MyModel
       cfg = ctx.cfg
       return MyModel(**_backbone_kwargs(cfg), weight_decay=cfg.get("weight_decay", 0.0), alpha=cfg["alpha"])

   register_method(MethodSpec(name="mymodel", kind="trainable", build=build_mymodel,
                              description="...", score_keys=("EU", "AU", "TU")))  # or ("",)
   ```
## 1B. Post-hoc detector
1. Copy `templates/posthoc_detector.py` into `cgnn/models/detectors/` (or add the class to
   `logit_based.py` / `feature_based.py`); implement `ood_scores(batch) -> [N]` (higher = more OOD) and,
   if it needs training-node statistics, `prepare(train_data)`. Set `needs_input_grad = True` if it
   back-propagates to the inputs; `requires_full_graph_eval = True` if it must see the whole graph.
   Export it in `cgnn/models/detectors/__init__.py`.
2. Register in `cgnn/methods/posthoc.py`:
   `MethodSpec("mydet", "posthoc", _detector("MyDetector", "param1", "param2"), description="...", requires=("faiss",))`.
   Backbone choice: `backbone_rank` (default 0 = best vanilla checkpoint by `val_auroc`); several backbones:
   `num_backbones=lambda cfg: int(cfg["M"])` and read `ctx.backbones`.
## 1C. CGNN variant (no new model code)
Add one tuple to `_CREDAL_METHODS` in `cgnn/methods/trainable.py`, e.g.
`("credal_LJ_wide_dual_head_detached", None, dict(latent="joint", heads="dual", detach_credal=True,
credal_mid_activation="identity", credal_half_activation="softplus"), "Unbounded interval logits.")`.
Options of `CredalGNN` (`cgnn/models/credal.py`): `latent` joint|last, `heads` dual|credal, `detach_credal`,
`joint_include_input`, `joint_act_on_last`, `credal_mid_activation`, `credal_half_activation`,
`lambda_cls`, `lambda_cons` (via `constrained=True`). For a quick try without registering:
`cgnn run -m credal_LJ_dual_head_detached -d synthetic --cpu --set credal_mid_activation=identity`.

## 2. Config `configs/methods/<name>.yaml`
Copy `templates/method.yaml`. `defaults` = a sensible single-run config (`cgnn run`); `sweep` = the W&B
search space. **`defaults.monitor` must equal `sweep.metric.name`** and be a metric the model logs on
validation. Dataset metadata (`in_channels`, `out_channels`, `batch_size`, `num_neighbors`) is merged
automatically — don't put it here. Post-hoc methods without training: `deterministic: false`.

## 3. Verify (all must pass)
```bash
python -m pytest -q tests/test_config.py tests/test_runner.py -k <name>   # contract + end-to-end on synthetic
cgnn run -m <name> -d synthetic --cpu --set max_epochs=3                  # prints the metrics
cgnn run -m <name> -d squirrel --wandb disabled --gpus 1                  # real data smoke test (1 GPU)
python -m pytest -q && ruff check cgnn tests
```
Post-hoc methods need vanilla backbones for the dataset *and split*: `cgnn doctor` lists them.

## 4. Document
- One line in the method table of `README.md` (and `AGENTS.md` if it is a paper method).
- If it corresponds to something in the paper: `docs/paper_to_code.md`.
- `CHANGELOG.md` entry. If it changes defaults or protocol: `docs/protocol.md`.
- Ask the `ml-reviewer` subagent to review the diff.
