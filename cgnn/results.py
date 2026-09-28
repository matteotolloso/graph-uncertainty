"""Collect W&B runs and build result tables (paper Table 2 protocol).

Protocol implemented by :func:`summarize`:

1. Runs of ``(method, dataset)`` are those of the sweep named
   ``<dataset>_<method>`` (or single runs whose config has ``cgnn_method`` /
   ``cgnn_dataset``). Only finished runs with both val and test metrics count.
2. For every candidate uncertainty *component* (e.g. ``EU``/``AU`` for credal
   methods, ``""`` for single-score baselines) the replicate set is chosen on
   validation: ``aggregate="topk"`` = the ``k`` runs with the best val score;
   ``aggregate="group"`` = the hyper-parameter group (config minus seed-like
   keys) with the best mean val score.
3. The component with the best mean val score is selected (paper: "we select
   the uncertainty component using the validation set"), and its test
   mean/std over the replicate set is reported.

Everything except :func:`fetch_runs` is pure and unit-tested.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

# Keys that identify a replicate rather than a hyper-parameter configuration.
REPLICATE_KEYS = {"seed", "backbone_rank", "split_seed", "cgnn_version", "wandb_version", "_wandb"}

# (row label, method key, candidate components) -- rows of paper Table 2, in order.
PAPER_ROWS: list[tuple[str, str, tuple[str, ...]]] = [
    ("Energy", "energy", ("",)),
    ("KNN", "knn", ("",)),
    ("ODIN", "odin", ("",)),
    ("Mahalanobis", "mahalanobis", ("",)),
    ("GNNSafe", "gnnsafe", ("",)),
    ("JLDE", "knn_LJ", ("",)),
    ("GEBM", "gebm", ("",)),
    ("CaGCN", "cagcn", ("",)),
    ("Classical ensemble", "ensemble", ("EU_classic", "AU_classic")),
    ("CGNN", "credal_LJ_dual_head_detached", ("EU", "AU")),
    ("CGNN last layer", "credal", ("EU", "AU")),
    ("CGNN by ensemble", "ensemble", ("EU_credal", "AU_credal")),
    ("CGNN post train", "frozen", ("EU", "AU")),
    ("CGNN only credal", "credal_LJ", ("EU", "AU")),
]
PAPER_DATASETS = ["squirrel", "arxiv", "patents", "amazon_ratings", "roman_empire", "coauthor", "reddit2"]


@dataclass
class RunRow:
    run_id: str
    dataset: str
    method: str
    state: str
    config: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    sweep_id: str | None = None

    def metric(self, key: str) -> float | None:
        value = self.summary.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
        return None


@dataclass
class Summary:
    method: str
    dataset: str
    component: str
    n: int
    val_mean: float
    test_mean: float
    test_std: float
    run_ids: list[str]

    @property
    def metric_suffix(self) -> str:
        return f"_{self.component}" if self.component else ""


def split_sweep_name(name: str, datasets: Iterable[str]) -> tuple[str, str] | None:
    """``"roman_empire_credal_LJ"`` -> ``("roman_empire", "credal_LJ")`` using the known dataset names."""
    for ds in sorted(datasets, key=len, reverse=True):
        if name.startswith(ds + "_"):
            return ds, name[len(ds) + 1 :]
    return None


def _key(prefix: str, component: str) -> str:
    return f"{prefix}_auroc_{component}" if component else f"{prefix}_auroc"


def _config_signature(cfg: dict[str, Any]) -> tuple:
    return tuple(
        sorted((k, repr(v)) for k, v in cfg.items() if k not in REPLICATE_KEYS and not k.startswith("_"))
    )


def _replicates(rows: list[RunRow], component: str, aggregate: str, k: int) -> list[RunRow]:
    val_key = _key("val", component)
    scored = [
        r for r in rows if r.metric(val_key) is not None and r.metric(_key("test", component)) is not None
    ]
    if not scored:
        return []
    if aggregate == "topk":
        return sorted(scored, key=lambda r: (-r.metric(val_key), r.run_id))[:k]
    if aggregate == "group":
        groups: dict[tuple, list[RunRow]] = defaultdict(list)
        for r in scored:
            groups[_config_signature(r.config)].append(r)
        best = max(groups.values(), key=lambda g: (statistics.fmean(r.metric(val_key) for r in g), len(g)))
        return sorted(best, key=lambda r: r.run_id)
    raise ValueError(f"aggregate must be 'topk' or 'group', got {aggregate!r}")


def summarize(
    rows: Sequence[RunRow],
    method: str,
    dataset: str,
    components: Sequence[str] = ("",),
    *,
    aggregate: str = "topk",
    k: int = 5,
) -> Summary | None:
    rows = [r for r in rows if r.method == method and r.dataset == dataset and r.state == "finished"]
    best: Summary | None = None
    for component in components:
        reps = _replicates(rows, component, aggregate, k)
        if not reps:
            continue
        tests = [r.metric(_key("test", component)) for r in reps]
        s = Summary(
            method=method,
            dataset=dataset,
            component=component,
            n=len(reps),
            val_mean=statistics.fmean(r.metric(_key("val", component)) for r in reps),
            test_mean=statistics.fmean(tests),
            test_std=statistics.stdev(tests) if len(tests) > 1 else 0.0,
            run_ids=[r.run_id for r in reps],
        )
        if best is None or s.val_mean > best.val_mean:
            best = s
    return best


def paper_table(
    rows: Sequence[RunRow],
    datasets: Sequence[str] = PAPER_DATASETS,
    table_rows: Sequence[tuple[str, str, tuple[str, ...]]] = PAPER_ROWS,
    **kwargs,
) -> dict[str, dict[str, Summary | None]]:
    return {
        label: {ds: summarize(rows, method, ds, comps, **kwargs) for ds in datasets}
        for label, method, comps in table_rows
    }


def format_table(table: dict[str, dict[str, Summary | None]], fmt: str = "md", scale: float = 100.0) -> str:
    """Render ``paper_table`` output as markdown (``md``), ``csv`` or ``latex`` (best per column in bold)."""
    labels = list(table)
    datasets = list(next(iter(table.values()))) if table else []
    best = {
        ds: max((table[lab][ds].test_mean for lab in labels if table[lab][ds] is not None), default=None)
        for ds in datasets
    }

    def cell(s: Summary | None, ds: str, bold: tuple[str, str]) -> str:
        if s is None:
            return "-"
        text = f"{s.test_mean * scale:.2f} ± {s.test_std * scale:.2f}"
        if fmt == "latex":
            text = text.replace("±", r"$\pm$")
        if best[ds] is not None and s.test_mean == best[ds]:
            text = f"{bold[0]}{text}{bold[1]}"
        return text

    if fmt == "csv":
        lines = ["method," + ",".join(f"{d}_mean,{d}_std,{d}_component,{d}_n" for d in datasets)]
        for lab in labels:
            parts = []
            for ds in datasets:
                s = table[lab][ds]
                parts += (
                    ["", "", "", "0"]
                    if s is None
                    else [f"{s.test_mean:.6f}", f"{s.test_std:.6f}", s.component, str(s.n)]
                )
            lines.append(f"{lab}," + ",".join(parts))
        return "\n".join(lines)
    if fmt == "latex":
        head = "Method & " + " & ".join(datasets) + r" \\"
        body = [
            f"{lab} & " + " & ".join(cell(table[lab][ds], ds, (r"\textbf{", "}")) for ds in datasets) + r" \\"
            for lab in labels
        ]
        return "\n".join(
            [
                r"\begin{tabular}{l" + "c" * len(datasets) + "}",
                r"\toprule",
                head,
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabular}",
            ]
        )
    head = "| Method | " + " | ".join(datasets) + " |"
    sep = "|---|" + "---|" * len(datasets)
    body = [
        f"| {lab} | " + " | ".join(cell(table[lab][ds], ds, ("**", "**")) for ds in datasets) + " |"
        for lab in labels
    ]
    return "\n".join([head, sep, *body])


def _row_from_run(run, ds: str, method: str) -> RunRow:
    cfg = {k: v for k, v in run.config.items() if not k.startswith("_")}
    summary = {k: v for k, v in run.summary.items() if not k.startswith("_")}
    return RunRow(run.id, ds, method, run.state, cfg, summary, run.sweep_name)


def fetch_runs(
    project: str,
    entity: str | None = None,
    *,
    datasets: Iterable[str] | None = None,
    methods: Iterable[str] | None = None,
    known_datasets: Iterable[str] = (),
) -> list[RunRow]:
    """Download runs (read-only) and attach ``(dataset, method)``.

    Sweep runs are found through sweeps named ``<dataset>_<method>`` (only the matching sweeps are
    queried); single runs (``cgnn run``) through their ``cgnn_method``/``cgnn_dataset`` config keys.
    """
    import wandb

    api = wandb.Api(timeout=120)
    entity = entity or api.default_entity
    path = f"{entity}/{project}"
    want_ds = set(datasets) if datasets else None
    want_m = set(methods) if methods else None

    def wanted(ds: str, method: str) -> bool:
        return (want_ds is None or ds in want_ds) and (want_m is None or method in want_m)

    rows: list[RunRow] = []
    for sweep in api.project(project, entity=entity).sweeps(per_page=200):
        parsed = split_sweep_name(sweep.name or "", known_datasets)
        if parsed is None or not wanted(*parsed):
            continue
        for run in api.runs(path, filters={"sweep": sweep.id}, per_page=500, lazy=False):
            rows.append(_row_from_run(run, *parsed))

    single = {"config.cgnn_method": {"$exists": True}}
    for run in api.runs(path, filters=single, per_page=500, lazy=False):
        if run.sweep_name:
            continue  # already counted through its sweep
        ds, method = run.config.get("cgnn_dataset"), run.config.get("cgnn_method")
        if ds and method and wanted(ds, method):
            rows.append(_row_from_run(run, ds, method))
    return rows
