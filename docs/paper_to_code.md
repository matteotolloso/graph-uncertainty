# Paper ↔ code map

Paper: *Credal Graph Neural Networks* (AAAI submission). Section/equation numbers refer to the submitted PDF.

## Credal learning
| Paper | Code | Notes |
|---|---|---|
| Credal set from probability intervals | `cgnn/uncertainty/interval.py` | `reachable_bounds` tightens `[q_L, q_U]` before entropies |
| Eq. 2: TU = max H, AU = min H, EU = TU − AU | `cgnn/uncertainty/entropy.py::credal_uncertainties` | bits; TU exact (water-filling); AU greedy unless `min_entropy_method` (docs/known_issues.md M3) |
| Eq. 3: `m = g(Wz+b)`, `h = g'(W'z+b')` | `cgnn/models/heads.py::CredalLayer` | 2-layer MLP (Linear–Sigmoid–Linear) produces `[m, h]`; `g = g' = sigmoid` (M4) |
| Eq. 4: Interval SoftMax | `cgnn/uncertainty/interval.py::interval_softmax` | other classes enter through their midpoints |
| Eq. 5: DRO objective | approximated by Eq. 9–10 | |
| Eq. 6: classifier head `softmax(W_c z + b_c)` | `cgnn/models/heads.py::mlp_classifier` | code: MLP (Linear–Sigmoid–Linear), not a single linear layer |
| Eq. 7: credal head on `stopgrad(z)` | `CredalGNN(detach_credal=True)` | `cgnn/models/credal.py::forward` |
| Eq. 8: CE of the classifier | `CredalGNN._losses` (`F.cross_entropy`) | |
| Eq. 9–10: `L_credal` (CE on `q_U` + top-δN CE on `q_L`) | `cgnn/uncertainty/losses.py::CreNetLoss` | `k = max(1, int(δN))`, δ swept in [0.5, 1] |
| Eq. 11: `L_total = L_cls + L_credal` | `CredalGNN._losses` | code: `L_credal + λ_cls·L_cls`; λ_cls is irrelevant when detached (M5) |
| Eq. 12: message passing | PyG `BasicGNN` (`GCN`, `GraphSAGE`), sigmoid activations | `cgnn/models/backbones.py` |
| Eq. 14: `z_joint = [z^0 ‖ z^1 ‖ … ‖ z^L]` | `joint_representation(..., include_input=?)` | **CGNN code omits `z^0`** (M1); `credal_LJ0_dual_head_detached` includes it |

## Methods (Table 2 rows)
| Paper row | Method key | Module | Notes |
|---|---|---|---|
| CGNN | `credal_LJ_dual_head_detached` | `CredalGNN(latent=joint, heads=dual, detach)` | |
| CGNN last layer | `credal` | `CredalGNN(latent=last, heads=credal)` | also drops the dual head (M2) → clean ablation: `credal_last_dual_head_detached` |
| CGNN only credal | `credal_LJ` | `CredalGNN(latent=joint, heads=credal)` | |
| CGNN post train | `frozen` | `cgnn/models/frozen.py` | joint latent of the frozen backbone *with* `z^0`, last layer = logits |
| CGNN by ensemble | `ensemble` → `*_credal` | `cgnn/models/detectors/ensemble.py` | hull = class-wise min/max of member softmaxes |
| Classical ensemble | `ensemble` → `*_classic` | same | `M` swept 2..15 |
| JLDE | `knn_LJ` | `KNNJointDetector` | k-th NN distance on `[x ‖ z^1 ‖ … ‖ z^L]` |
| Energy | `energy` | `EnergyDetector` | temperature swept (H1) |
| KNN | `knn` | `KNNDetector` | k-th NN distance, penultimate layer |
| ODIN | `odin` | `ODINDetector` | |
| Mahalanobis | `mahalanobis` | `MahalanobisDetector` | features = logits (H1) |
| GNNSafe | `gnnsafe` | `GNNSafeDetector` | energy propagation, no regulariser |
| GEBM | `gebm` | `GEBMDetector` | `graph_uq` from the `graph-ebm` submodule; joint-latent embeddings |
| CaGCN | `cagcn` | `CaGCNModule` | calibrated on ID validation nodes |

## Datasets (Table 1)
`cgnn list datasets` / `cgnn describe <dataset>` show the class split; `cgnn data <dataset>` prints the
node counts. Public-split datasets reproduce Table 1 exactly (e.g. Squirrel 1,515 / 982+682 / 622+419);
random-split datasets (ArXiv, Patents, Coauthor) depend on `split_seed`.

## Figures
Figure 1 (simplex plots): `notebooks/plots.ipynb` → `notebooks/figures/`.
