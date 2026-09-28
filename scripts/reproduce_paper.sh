#!/usr/bin/env bash
# Full pipeline for paper Table 2, in dependency order. Prints the commands; set EXECUTE=1 to run them.
#
#   DATASETS="squirrel arxiv" EXECUTE=1 bash scripts/reproduce_paper.sh
#
# Order matters: post-hoc baselines, ensembles and "CGNN post train" need the vanilla backbones
# (step 1) of the same dataset AND the same split (split_seed, see docs/protocol.md).
# Every command pins ONE free GPU (`--gpus 1`); run at most two of them in parallel (shared server).
set -euo pipefail
cd "$(dirname "$0")/.."

DATASETS=${DATASETS:-"squirrel arxiv patents amazon_ratings roman_empire coauthor reddit2"}
PROJECT=${PROJECT:-graph-uncertainty}
VANILLA_RUNS=${VANILLA_RUNS:-100}
CREDAL_RUNS=${CREDAL_RUNS:-100}
POSTTRAIN_RUNS=${POSTTRAIN_RUNS:-30}

run() {
    echo "+ $*"
    if [[ "${EXECUTE:-0}" == "1" ]]; then "$@"; fi
}

for ds in $DATASETS; do
    # 1) backbones (checkpoints ranked by val_auroc feed every post-hoc method)
    run cgnn sweep -m vanilla -d "$ds" -c "$VANILLA_RUNS" -p "$PROJECT" --gpus 1

    # 2) post-hoc baselines: grid sweeps, bounded by the grid size
    for m in energy odin mahalanobis knn knn_LJ gnnsafe gebm ensemble; do
        run cgnn sweep -m "$m" -d "$ds" -p "$PROJECT" --gpus 1
    done
    run cgnn sweep -m cagcn -d "$ds" -c "$POSTTRAIN_RUNS" -p "$PROJECT" --gpus 1
    run cgnn sweep -m frozen -d "$ds" -c "$POSTTRAIN_RUNS" -p "$PROJECT" --gpus 1

    # 3) CGNN and its trainable ablations
    for m in credal_LJ_dual_head_detached credal credal_LJ; do
        run cgnn sweep -m "$m" -d "$ds" -c "$CREDAL_RUNS" -p "$PROJECT" --gpus 1
    done
done

# 4) table (validation-selected component, mean +- std over the top-5 runs)
run cgnn results -p "$PROJECT" --format latex --out outputs/tables/table2.tex --details
