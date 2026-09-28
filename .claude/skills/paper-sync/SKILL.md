---
name: paper-sync
description: Keep the CGNN paper and the code consistent — map equations, methods, datasets and table numbers to code, audit discrepancies, and update docs/paper_to_code.md. Use when the user edits the paper, asks whether the code matches the paper, prepares the camera-ready/rebuttal, or changes a method's definition.
---

# Paper <-> code sync

Reference: `docs/paper_to_code.md` (equation-by-equation map + known mismatches). If the user provides a PDF,
extract its text (e.g. `pdfplumber`, column by column) into the scratchpad before comparing.

## Audit checklist
1. **Credal layer (Eq. 3-4)**: `cgnn/models/heads.py::CredalLayer` (MLP -> m, h; sigmoid/sigmoid by default)
   and `cgnn/uncertainty/interval.py::interval_softmax` (midpoint form). State g, g' used.
2. **Uncertainties (Eq. 2)**: `cgnn/uncertainty/entropy.py` — TU exact, AU greedy unless
   `min_entropy_method` says otherwise; entropies in bits; computed on reachable bounds.
3. **Loss (Eq. 9-11)**: `cgnn/uncertainty/losses.py::CreNetLoss` (+ `lambda_cls` on the classifier CE —
   the paper's Eq. 11 has no weight).
4. **Detach (Eq. 7)** and **joint latent (Eq. 14)**: `cgnn/models/credal.py` — the paper includes `z^0`;
   the AAAI code (`credal_LJ_dual_head_detached`) does not (`joint_include_input=False`); the joint latent
   applies the activation after the last layer; the classifier head is an MLP (paper: linear, Eq. 6).
5. **Ablations**: `CGNN last layer` = `credal` (single credal head, end-to-end) — differs from CGNN in two
   factors; the clean one-factor ablation is `credal_last_dual_head_detached`.
6. **Datasets (Table 1)**: `cgnn data <name>` prints train/val/test counts to compare with the table;
   random-split datasets depend on `split_seed`.
7. **Baselines**: GEBM uses the joint latent as embeddings; KNN uses the k-th neighbour distance
   (paper text says "average distance"); GNNSafe without the regulariser.
8. **Numbers (Table 2)**: `cgnn results --details` and compare cell by cell; flag cells produced by
   pre-0.2 runs affected by the AUROC saturation or the split leakage (`docs/known_issues.md`).

## When the definition changes
- Code follows paper: add a switch/new method key (don't silently change the old one), re-run, update the doc.
- Paper follows code: write the exact sentence/equation to put in the paper and list affected tables.
- Always update `docs/paper_to_code.md` and `CHANGELOG.md`.
