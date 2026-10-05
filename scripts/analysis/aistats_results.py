#!/usr/bin/env python3
"""Tables and figures of the AISTATS extra experiments (W&B, read-only).

    python scripts/analysis/aistats_results.py replicates  # seed replicates of the Table-2 configs
    python scripts/analysis/aistats_results.py csbm        # OOD AUROC vs homophily (synthetic CSBM)
    python scripts/analysis/aistats_results.py exp1        # EU/AU under fewer labels / label noise
    python scripts/analysis/aistats_results.py fshift      # EU/AU under a test-time node-feature shift

Projects: ``graph-uncertainty-aistats``, ``-csbm``, ``-exp1`` (see ``scripts/campaigns/aistats.jobs``), ``-fshift``
(``scripts/campaigns/aistats_fshift.jobs``).
Selection never looks at test metrics: components (EU/AU/...) are chosen by mean validation AUROC, as in
``cgnn results``; "CGNN (EU)" fixes EU a priori (docs/known_issues.md M7). Output: ``outputs/tables/aistats/``.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from cgnn.data import DATASETS
from cgnn.paths import Paths
from cgnn.results import PAPER_DATASETS, PAPER_ROWS, RunRow, fetch_runs, summarize

EXTRA = ("aupr", "fpr95", "auprin", "fpr95in", "misc_auroc")
ROWS = [*PAPER_ROWS, ("CGNN (EU)", "credal_LJ_dual_head_detached", ("EU",))]
CSBM_ROWS = [("MSP", "vanilla", ("",))] + [r for r in ROWS if r[1] not in ("credal", "frozen", "credal_LJ")]
CGNN = "credal_LJ_dual_head_detached"


def _out_dir() -> Path:
    out = Paths.from_env().output_dir / "tables" / "aistats"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _mean_std(values: list[float]) -> tuple[float, float, int]:
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return math.nan, math.nan, 0
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0, len(values)


def _accuracy(r: RunRow) -> float | None:
    for key in ("test_accuracy_cls", "test_acc", "test_accuracy_U"):
        if r.metric(key) is not None:
            return r.metric(key)
    return None


def _cell(rows, label, method, comps, ds):
    s = summarize(rows, method, ds, comps, aggregate="topk", k=5)
    if s is None:
        return None
    reps = [r for r in rows if r.run_id in set(s.run_ids)]
    suf = s.metric_suffix
    cell = {
        "label": label,
        "dataset": ds,
        "component": s.component or "-",
        "n": s.n,
        "val_auroc": s.val_mean,
        "test_auroc": s.test_mean,
        "test_auroc_std": s.test_std,
        "_aurocs": [r.metric(f"test_auroc{suf}") for r in reps],
    }
    for m in EXTRA:
        cell[m], cell[f"{m}_std"], _ = _mean_std([r.metric(f"test_{m}{suf}") for r in reps])
    cell["accuracy"], cell["accuracy_std"], _ = _mean_std([_accuracy(r) for r in reps])
    return cell


def _write_csv(path: Path, cells: list[dict]) -> None:
    keys = [k for k in cells[0] if not k.startswith("_")]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(cells)
    print(f"wrote {path}")


def cmd_replicates(args) -> None:
    from scipy.stats import ttest_ind

    rows = fetch_runs(args.project, known_datasets=DATASETS.names())
    cells = [
        c for label, m, comps in ROWS for ds in PAPER_DATASETS if (c := _cell(rows, label, m, comps, ds))
    ]
    by_ds = defaultdict(dict)
    for c in cells:
        by_ds[c["dataset"]][c["label"]] = c
    for group in by_ds.values():  # Welch t-test: CGNN (EU) vs every other row, on the 5 test AUROCs
        ref = group.get("CGNN (EU)")
        for c in group.values():
            c["p_vs_cgnn_eu"] = math.nan
            if ref and c is not ref and len(c["_aurocs"]) > 1 and len(ref["_aurocs"]) > 1:
                c["p_vs_cgnn_eu"] = float(ttest_ind(ref["_aurocs"], c["_aurocs"], equal_var=False).pvalue)
    out = _out_dir()
    _write_csv(out / "replicates_long.csv", cells)
    datasets = [ds for ds in PAPER_DATASETS if ds in by_ds]
    labels = [label for label, _, _ in ROWS]
    for metric, scale, better in (("test_auroc", 100, max), ("fpr95in", 100, min), ("misc_auroc", 100, max)):
        std_key = "test_auroc_std" if metric == "test_auroc" else f"{metric}_std"
        lines = [
            r"\begin{tabular}{l" + "c" * len(datasets) + "}",
            r"\toprule",
            "Method & " + " & ".join(d.replace("_", r"\_") for d in datasets) + r" \\",
            r"\midrule",
        ]
        best = {
            ds: better((c[metric] for c in by_ds[ds].values() if math.isfinite(c[metric])), default=None)
            for ds in datasets
        }
        for label in labels:
            row = []
            for ds in datasets:
                c = by_ds[ds].get(label)
                if c is None or not math.isfinite(c[metric]):
                    row.append("-")
                    continue
                txt = f"{scale * c[metric]:.2f} $\\pm$ {scale * c[std_key]:.2f}"
                if best[ds] is not None and c[metric] == best[ds]:
                    txt = r"\textbf{" + txt + "}"
                if metric == "test_auroc" and c.get("p_vs_cgnn_eu", math.nan) < 0.05:
                    txt += r"$^\dagger$"
                row.append(txt)
            lines.append(f"{label} & " + " & ".join(row) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        path = out / f"replicates_{metric}.tex"
        path.write_text("\n".join(lines) + "\n")
        print(f"wrote {path}")


def cmd_csbm(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = fetch_runs(args.project, known_datasets=DATASETS.names())
    levels = [(int(ds.removeprefix("csbm_h")) / 10, ds) for ds in DATASETS.names() if ds.startswith("csbm_h")]
    cells = [
        {**c, "homophily": h}
        for label, m, comps in CSBM_ROWS
        for h, ds in sorted(levels)
        if (c := _cell(rows, label, m, comps, ds))
    ]
    out = _out_dir()
    _write_csv(out / "csbm_long.csv", cells)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for label, _, _ in CSBM_ROWS:
        pts = sorted(
            (c["homophily"], c["test_auroc"], c["test_auroc_std"]) for c in cells if c["label"] == label
        )
        if not pts:
            continue
        hs, mu, sd = zip(*pts, strict=True)
        bold = label.startswith("CGNN")
        ax.errorbar(
            hs,
            [100 * m for m in mu],
            yerr=[100 * s for s in sd],
            label=label,
            capsize=2,
            lw=2.2 if bold else 1.0,
            alpha=1.0 if bold else 0.75,
            marker="o" if bold else ".",
        )
    ax.set_xlabel("edge homophily $h$")
    ax.set_ylabel("OOD AUROC (%)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"csbm_auroc.{ext}", dpi=200)
    print(f"wrote {out / 'csbm_auroc.pdf'}")


EXP1_SCORES = {  # method -> (uncertainty name, metric suffix)
    CGNN: [("CGNN EU", "_EU"), ("CGNN AU", "_AU"), ("CGNN TU", "_TU")],
    "ensemble": [
        ("Ens. EU", "_EU_classic"),
        ("Ens. AU", "_AU_classic"),
        ("CGNN-ens. EU", "_EU_credal"),
        ("CGNN-ens. AU", "_AU_credal"),
    ],
    "vanilla": [("MSP", "")],
}


def cmd_exp1(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in fetch_runs(args.project, known_datasets=DATASETS.names()) if r.state == "finished"]
    groups = defaultdict(list)
    for r in rows:
        cond = (float(r.config.get("train_fraction", 1.0)), float(r.config.get("label_noise", 0.0)))
        groups[(r.dataset, r.method, cond)].append(r)
    cells = []
    for (ds, method, (frac, noise)), rs in sorted(groups.items()):
        for name, suf in EXP1_SCORES.get(method, []):
            c = {
                "dataset": ds,
                "method": method,
                "score": name,
                "train_fraction": frac,
                "label_noise": noise,
                "n": len(rs),
            }
            for key in ("mean{}_id", "mean{}_ood", "auroc{}", "misc_auroc{}"):
                k = key.format(suf)
                c[k.replace(suf, "") if suf else k], c[(k.replace(suf, "") if suf else k) + "_std"], _ = (
                    _mean_std([r.metric(f"test_{k}") for r in rs])
                )
            c["accuracy"], c["accuracy_std"], _ = _mean_std([_accuracy(r) for r in rs])
            cells.append(c)
    out = _out_dir()
    _write_csv(out / "exp1_long.csv", cells)
    datasets = sorted({c["dataset"] for c in cells})
    fig, axes = plt.subplots(2, len(datasets), figsize=(3.4 * len(datasets), 5.6), squeeze=False)
    for j, ds in enumerate(datasets):
        for i, (axis_key, fixed) in enumerate(
            (("train_fraction", ("label_noise", 0.0)), ("label_noise", ("train_fraction", 1.0)))
        ):
            ax = axes[i][j]
            for name in ("CGNN EU", "CGNN AU", "Ens. EU", "Ens. AU"):
                pts = sorted(
                    (c[axis_key], c["mean_id"])
                    for c in cells
                    if c["dataset"] == ds and c["score"] == name and c[fixed[0]] == fixed[1]
                )
                ref = [v for x, v in pts if x == (1.0 if axis_key == "train_fraction" else 0.0)]
                if not pts or not ref or not ref[0]:
                    continue
                ax.plot(
                    [x for x, _ in pts],
                    [v / ref[0] for _, v in pts],
                    marker="o",
                    label=name,
                    ls="-" if "EU" in name else "--",
                )
            ax.set_xlabel("fraction of training labels" if i == 0 else "label noise")
            ax.set_ylabel("mean ID uncertainty / clean")
            ax.set_title(ds.replace("_", " "), fontsize=9)
            if axis_key == "train_fraction":
                ax.set_xscale("log")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"exp1_disentanglement.{ext}", dpi=200)
    print(f"wrote {out / 'exp1_disentanglement.pdf'}")


FSHIFT_SCORES = {  # method -> (uncertainty name, metric suffix)
    CGNN: [("CGNN EU", "_EU"), ("CGNN AU", "_AU"), ("CGNN TU", "_TU")],
    "vanilla": [("MSP", "")],
    "ensemble": [
        ("Ens. EU", "_EU_classic"),
        ("Ens. AU", "_AU_classic"),
        ("CGNN-ens. EU", "_EU_credal"),
        ("CGNN-ens. AU", "_AU_credal"),
    ],
}


def _sigma(tag: str) -> float:
    return float(tag.removeprefix("fshift_").replace("p", "."))


def cmd_fshift(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in fetch_runs(args.project, known_datasets=DATASETS.names()) if r.state == "finished"]
    groups = defaultdict(list)
    for r in rows:
        groups[(r.dataset, r.method)].append(r)
    cells = []
    for (ds, method), rs in sorted(groups.items()):
        tags = sorted(
            {k.split("_test_")[0] for r in rs for k in r.summary if k.startswith("fshift_")}, key=_sigma
        )
        for name, suf in FSHIFT_SCORES.get(method, []):
            for tag in tags:
                c = {"dataset": ds, "method": method, "score": name, "sigma": _sigma(tag), "n": len(rs)}
                pre = f"{tag}_test_"
                for out, key in (
                    ("auroc", f"auroc{suf}"),
                    ("mean_clean", f"mean{suf}_id"),
                    ("mean_shifted", f"mean{suf}_ood"),
                ):
                    c[out], c[f"{out}_std"], _ = _mean_std([r.metric(pre + key) for r in rs])
                ratios = [  # per run, then averaged: scales differ across seeds
                    r.metric(f"{pre}mean{suf}_ood") / r.metric(f"{pre}mean{suf}_id")
                    for r in rs
                    if r.metric(f"{pre}mean{suf}_id") and r.metric(f"{pre}mean{suf}_ood") is not None
                ]
                c["ratio"], c["ratio_std"], _ = _mean_std(ratios)
                c["ood_auroc"], _, _ = _mean_std([r.metric(f"test_auroc{suf}") for r in rs])
                cells.append(c)
    out = _out_dir()
    prefix = args.project.removeprefix("graph-uncertainty-aistats-").replace("-", "_")  # fshift, fshift_acc
    _write_csv(out / f"{prefix}_long.csv", cells)
    datasets = sorted({c["dataset"] for c in cells})
    names = ("CGNN EU", "CGNN AU", "Ens. EU", "Ens. AU", "CGNN-ens. EU")
    fig, axes = plt.subplots(2, len(datasets), figsize=(3.2 * len(datasets), 5.6), squeeze=False)
    for j, ds in enumerate(datasets):
        for i, (key, label) in enumerate(
            (("ratio", "mean unc. shifted / clean"), ("auroc", "AUROC shifted vs clean"))
        ):
            ax = axes[i][j]
            for name in names:
                pts = sorted(
                    (c["sigma"], c[key], c[f"{key}_std"])
                    for c in cells
                    if c["dataset"] == ds and c["score"] == name
                )
                if not pts:
                    continue
                xs, mu, sd = zip(*pts, strict=True)
                ax.errorbar(
                    [x + 0.05 for x in xs],
                    mu,
                    yerr=sd,
                    marker="o",
                    ms=3,
                    capsize=2,
                    label=name,
                    ls="-" if "EU" in name else "--",
                )
            ax.set_xscale("log")
            ax.set_xlabel(r"feature noise $\sigma$ (+0.05, training std units)")
            ax.set_ylabel(label)
            ax.axhline(1.0 if key == "ratio" else 0.5, color="grey", lw=0.6)
            ax.set_title(ds.replace("_", " "), fontsize=9)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{prefix}_disentanglement.{ext}", dpi=200)
    print(f"wrote {out / f'{prefix}_disentanglement.pdf'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, project in (
        ("replicates", cmd_replicates, "graph-uncertainty-aistats"),
        ("csbm", cmd_csbm, "graph-uncertainty-aistats-csbm"),
        ("exp1", cmd_exp1, "graph-uncertainty-aistats-exp1"),
        ("fshift", cmd_fshift, "graph-uncertainty-aistats-fshift"),
    ):
        s = sub.add_parser(name)
        s.add_argument("-p", "--project", default=project)
        s.set_defaults(fn=fn)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
