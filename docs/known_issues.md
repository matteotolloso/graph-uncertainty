# Known issues and audit log

Findings of the 0.2 refactor audit (legacy = code at the `AAAI` branch head before the refactor).
Every legacy behaviour is still reproducible bit-for-bit with
`--set auroc_impl=torchmetrics split_seed=legacy` (verified: all 20 methods produce identical metrics).

Severity: **H** = can change reported numbers/conclusions, **M** = methodology or paper/code
consistency, **L** = engineering (fixed, no effect on numbers).

## H1 — AUROC saturation in torchmetrics (fixed, default `auroc_impl=exact`)
`torchmetrics` binary AUROC applies a sigmoid to any score outside `[0, 1]`. In float32 the sigmoid is
exactly 0 below about -104 and exactly 1 above about 17, so large scores become ties.
- Energy with `T >= 100`: `E ≈ -T·log C - mean(logits)` saturates for **every** node → AUROC exactly 0.5.
  Squirrel, top-1 legacy backbone: T=100 → 0.5000 (legacy) vs 0.4809 (true). Because 0.5 beats the real
  validation AUROC, legacy sweeps *prefer* the saturated temperatures.
- Mahalanobis (squared distances): Reddit2, top-1 legacy backbone, noise 0: test 0.6290 (legacy) vs
  0.6524 (true); Squirrel: 0.5888 vs 0.5891.
- GEBM: small shifts (Squirrel 0.4230 vs 0.4226). CGNN/MSP/ODIN/KNN/GNNSafe scores are small; no effect seen.
- Affected: paper Table 2 rows Energy, Mahalanobis (and possibly GEBM). Re-run these post-hoc sweeps
  (cheap) with the current code.

## H2 — Random-split datasets: post-hoc methods evaluated on a different split (fixed)
ArXiv, Patents and Coauthor use a random 60/20/20 split drawn with `torch.randperm` from the *global*
RNG right after model construction, so the split depended on the architecture of each run. Post-hoc
testers did not seed at all and drew a *fresh* split, so ~60% of their test ID nodes had been training
nodes of the backbone (optimistic ID confidence → inflated AUROC). Ensemble members were trained on
different splits.
- Affected: Table 2 columns ArXiv, Patents, Coauthor for Energy, KNN, ODIN, Mahalanobis, GNNSafe, JLDE,
  GEBM, CaGCN, Classical ensemble, CGNN by ensemble, CGNN post train. The trainable CGNN variants are
  not leaky but were each evaluated on a different random split.
- Fix: `split_seed` (default 0, dedicated generator), split fingerprint stored in every checkpoint,
  `backbone_split_check=error`. The 148 legacy ArXiv backbones have no split metadata (warning only):
  retrain backbones for these datasets.

## H3 — Per-batch AUROC on Reddit2 (fixed)
Post-hoc detectors and the ensemble logged `self.log("val_auroc", auroc(batch))` per step; with
NeighborLoader evaluation (Reddit2 only) Lightning averaged per-batch AUROCs instead of computing the
AUROC over all nodes. Credal models already aggregated per epoch. All modules now aggregate per epoch.
Related, also fixed: NeighborLoader depth for post-hoc methods was always 2 (now the backbone's depth);
GEBM mixed subgraph logits with the full-graph `edge_index` on sampled batches (now always full graph).

## M1 — Paper Eq. 14 vs code: `z^0` missing from the CGNN joint latent
The paper defines `z_joint = [z^0 || z^1 || ... || z^L]`; `credal_LJ_dual_head_detached` concatenates
only `z^1..z^L` (with the activation also applied after the last layer). `frozen`, `knn_LJ` (JLDE) and
`gebm` *do* include `z^0`. New method `credal_LJ0_dual_head_detached` implements Eq. 14 exactly
(`joint_include_input=True`). Either fix the paper text or re-run.

## M2 — "CGNN last layer" ablation changes two factors
`credal` = single credal head, end-to-end credal loss, last layer. Compared with CGNN it removes both the
joint latent and the dual (detached) head. The one-factor ablation is `credal_last_dual_head_detached`.

## M3 — AU (minimum entropy) is approximate
Minimum entropy over probability intervals is NP-hard in general; the solver pours free mass in a single
greedy order. On intervals produced by the credal layer it is suboptimal for 30–80% of nodes, mean EU
error ≈4–14% (C=3..6), correlation with exact AU 0.82–0.98. `min_entropy_method=auto` is exact for
C ≤ 6 (6 permutations for C=3) and uses a multistart heuristic (≈10× smaller error) above. Default stays
`greedy` for continuity; recommended `auto` for new results. Also affects EU (= TU − AU).

## M4 — Bounded interval logits in the credal head
`m = sigmoid(.)`, `h = sigmoid(.)` ⇒ interval logits in (-1, 2) ⇒ `q_U < e²/(e²+C−1)`: 0.79 (C=3),
0.43 (Coauthor, C=11), 0.38 (Roman Empire, C=13), **0.20 (Reddit2, C=30)**. The credal sets can never be
confident, TU often saturates at log2 C (TU AUROC = 0.5), and the CE on `q_U` has a floor. Plausible
contributor to the weaker results on many-class homophilic graphs. Try
`--set credal_mid_activation=identity credal_half_activation=softplus`.

## M5 — `lambda_cls` is (almost) a no-op for the detached dual head
With the credal head detached, `lambda_cls` scales the loss of a parameter set disjoint from the credal
head's; Adam is invariant to such scaling (up to `eps`) when `weight_decay=0` (the CGNN sweep has no
weight decay). Sweeping it spends budget on a flat dimension. Similarly `lambda_cal` in the CaGCN sweep is
not used by the model.

## M7 — AU and EU are (almost) the same score; component selection = sign selection
In the W&B CGNN runs (899 Squirrel, 300 ArXiv) the top runs have `AUROC(EU) + AUROC(AU) ≈ 1.000`
(0.995–1.002): EU and AU rank nodes in opposite order. Node-level check on the best Squirrel config
(0.2 code, best checkpoint): TU = log2 C for **100%** of test nodes (TU AUROC 0.5), Spearman(EU, AU) =
**−1.000**, EU 0.769 / AU 0.231. With unbounded heads (`identity`/`softplus`) TU saturates on 27.5% of
nodes but Spearman(EU, AU) is still −1.000. Which direction is informative depends on the
hyper-parameters: the top ArXiv runs by best component use AU (test 72.11 ± 0.40, their EU ≈ 0.28).
Selecting runs by validation **EU** instead (EU fixed a priori) still gives Squirrel 73.96 ± 0.84 and
ArXiv 69.21 ± 1.57 (`cgnn results`, top-5), i.e. CGNN stays ahead of the best Table 2 baseline there.
Consequences: (i) within a model AU/EU carry one signal, so the *disentanglement* claim is not supported
by these runs (the OOD-detection claim survives on Squirrel/ArXiv); (ii) choosing the component on
validation (which contains OOD nodes) is partly choosing the sign of that signal with OOD supervision, a
freedom the fixed-sign baselines do not get: Table 2 baselines far below 50 (GNNSafe 27.35 on Patents,
KNN 29.32 on Roman Empire, JLDE 31–37) would be close to CGNN with a sign choice (72.65 vs 75.63; 70.68
vs 71.81). Report EU fixed a priori as the main number, or give every method the same sign choice.

## M8 — Memory: Squirrel/Chameleon with SAGE on 40 GB GPUs
SAGEConv aggregates the raw 2089/2325-dim features over ~200K edges; with `deterministic=true` a
Squirrel SAGE run peaked above 32 GB and went OOM on a 40 GB A100 (the legacy runs used 80 GB GPUs).
Use `--cpu` (fast for these small graphs) or GCN.
Patents: with `deterministic=true` the sort-based deterministic scatter runs out of memory in the
full-graph validation forward on 40 GB GPUs (every architecture); with `deterministic=false` the peak is
13–24 GB. Run Patents with `--set deterministic=false` (GPU results are then not bit-reproducible).
Coauthor (6805 features): every SAGE configuration goes OOM with `deterministic=true`; fits with `false`.
Related bug (fixed): after a crashed run the W&B agent kept the run's GPU tensors alive through the
re-raised traceback, so every following run of that agent went OOM as well.

## M6 — Protocol points to state in the paper
- Validation contains OOD nodes and is used for early stopping, sweeps, component (AU/EU) selection and
  backbone ranking (OOD exposure during model selection).
- Trainable methods are tested at the *last* epoch (best + patience), not at the best checkpoint
  (`test_ckpt=last`; `best` available).
- Every legacy trainable run used seed 42; the "5 runs" of Table 2 are not seed replicates of one config.
  Use `cgnn run --seeds ...` / `cgnn results --aggregate group` for proper replicates.
- KNN uses the distance to the k-th neighbour (Sun et al. 2022), the text says "average distance".
- GEBM is given the joint latent as embeddings (original: penultimate layer); GNNSafe without regulariser.
- `*_f1` metrics are micro-F1 (= accuracy).
- Squirrel/Chameleon contain duplicated nodes (Platonov et al. 2023); ArXiv/Patents edges are directed
  (LINKX symmetrises them); Table 1 edge count for Squirrel (198,493) is the raw count, the PyG graph
  used has 217,073 directed edges.

## L — Engineering (fixed in 0.2, no effect on numbers)
Duplicated trainers/modules (21 trainer files → one runner), `sys.path` hacks, cwd-dependent paths
(backbones were searched in `./checkpoints` regardless of `--save_path`), `devices="auto"` grabbing all
GPUs (DDP) unless `CUDA_VISIBLE_DEVICES` was set, `docker --gpus all`, dead module `credal_GNN_i.py`
with a missing dependency, empty `utils/scores.py`, unused requirements (`torchdyn`, `ood-metrics`),
non-deterministic tie order when ranking backbones, `graph-ebm` gitlink without `.gitmodules`,
per-run PyG dataset re-loading (now cached in-process), no tests.
