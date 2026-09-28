"""``cgnn`` command line.

    cgnn list [methods|datasets]              what exists (paper names, kinds, status)
    cgnn describe NAME                        method/dataset details, defaults and sweep
    cgnn data DATASET                         load a dataset and print its OOD split (Table 1 row)
    cgnn run -m METHOD -d DATASET [...]       one (or a few repeated) runs, no sweep
    cgnn sweep -m METHOD -d DATASET [...]     create/join a W&B sweep and run an agent
    cgnn results [...]                        W&B runs -> paper-style table
    cgnn doctor                               environment / data / checkpoint / GPU report

Run ``cgnn <command> -h`` for options. Heavy imports happen inside commands so
``CUDA_VISIBLE_DEVICES`` can still be pinned before CUDA initialises.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from cgnn.config import MAX_GPUS, build_sweep, method_config, method_config_path, parse_overrides

# ----------------------------------------------------------------------------- helpers


def _registries():
    from cgnn.data import DATASETS
    from cgnn.methods import METHODS

    return METHODS, DATASETS


def _pin(args) -> None:
    if getattr(args, "cpu", False):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        return
    from cgnn.hardware import pin_gpus

    pin_gpus(args.gpus)


def _wandb_mode(args) -> str:
    if args.wandb != "auto":
        return args.wandb
    return "disabled" if args.dataset == "synthetic" else "online"


# ----------------------------------------------------------------------------- commands


def cmd_list(args) -> int:
    METHODS, DATASETS = _registries()
    what = args.what
    if what in ("methods", "all"):
        rows = []
        for name in METHODS:
            s = METHODS.get(name)
            rows.append(
                {
                    "method": name,
                    "kind": s.kind,
                    "paper_name": s.paper_name or "-",
                    "description": s.description,
                }
            )
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print(f"{'METHOD':45s} {'KIND':10s} {'PAPER NAME':38s} DESCRIPTION")
            for r in rows:
                print(f"{r['method']:45s} {r['kind']:10s} {r['paper_name']:38s} {r['description']}")
    if what in ("datasets", "all"):
        from cgnn.paths import Paths

        data_dir = Paths.from_env().data_dir
        rows = []
        for name in DATASETS:
            s = DATASETS.get(name)
            rows.append(
                {
                    "dataset": name,
                    "paper_name": s.paper_name,
                    "homophily": s.homophily,
                    "split": s.split,
                    "id_classes": list(s.id_classes),
                    "ood_classes": list(s.ood_classes),
                    "in_paper": s.in_paper,
                    "available": not s.missing_files(data_dir),
                }
            )
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            if what == "all":
                print()
            print(
                f"{'DATASET':16s} {'PAPER NAME':16s} {'HOMOPHILY':13s} {'SPLIT':7s} {'IN PAPER':9s} {'FILES':6s} ID / OOD classes"
            )
            for r in rows:
                print(
                    f"{r['dataset']:16s} {r['paper_name']:16s} {r['homophily']:13s} {r['split']:7s} "
                    f"{'yes' if r['in_paper'] else 'no':9s} {'ok' if r['available'] else 'MISS':6s} "
                    f"{r['id_classes']} / {r['ood_classes']}"
                )
    return 0


def cmd_describe(args) -> int:
    import yaml

    METHODS, DATASETS = _registries()
    if args.name in METHODS:
        s = METHODS.get(args.name)
        info = {
            "method": s.name,
            "kind": s.kind,
            "paper_name": s.paper_name,
            "description": s.description,
            "score_keys": list(s.score_keys),
            "fit_on": s.fit_on,
            "requires": list(s.requires),
            "config_file": str(method_config_path(s.name)),
        }
        print(yaml.safe_dump(info, sort_keys=False))
        print(yaml.safe_dump(method_config(s.name), sort_keys=False))
        return 0
    if args.name in DATASETS:
        s = DATASETS.get(args.name)
        info = {k: v for k, v in s.__dict__.items() if k != "load_raw"}
        print(
            yaml.safe_dump(
                {k: list(v) if isinstance(v, tuple) else v for k, v in info.items()}, sort_keys=False
            )
        )
        return 0
    print(f"Unknown method or dataset '{args.name}'. See `cgnn list`.", file=sys.stderr)
    return 2


def cmd_data(args) -> int:
    from cgnn.data import load_dataset

    cfg = parse_overrides(args.set)
    if args.split_seed is not None:
        cfg["split_seed"] = args.split_seed
    bundle = load_dataset(args.dataset, cfg)
    print(json.dumps(bundle.summary(), indent=2))
    return 0


def cmd_run(args) -> int:
    _pin(args)
    from cgnn.runner import run_experiment
    from cgnn.wandb_utils import fetch_best_config, log_resolved_config, make_logger

    overrides = {}
    if args.from_sweep:
        overrides.update(fetch_best_config(args.from_sweep, args.project, args.entity))
    overrides.update(parse_overrides(args.set))
    if args.cpu:
        overrides["accelerator"] = "cpu"

    repeat_key, repeat_values = None, [None]
    if args.seeds:
        repeat_key, repeat_values = "seed", args.seeds
    elif args.repeat:
        key, _, values = args.repeat.partition("=")
        repeat_key, repeat_values = key, [parse_overrides([f"v={v}"])["v"] for v in values.split(",")]

    mode = _wandb_mode(args)
    all_results = []
    for value in repeat_values:
        cfg = dict(overrides)
        if repeat_key is not None:
            cfg[repeat_key] = value
        if mode == "disabled":
            results = run_experiment(args.method, args.dataset, cfg, logger=make_logger(args.project, mode))
        else:
            import wandb

            run = wandb.init(
                project=args.project,
                entity=args.entity,
                mode=mode,
                group=args.group or f"{args.dataset}_{args.method}",
                job_type="run",
                tags=[args.method, args.dataset, *(args.tags or [])],
                config={"cgnn_method": args.method, "cgnn_dataset": args.dataset},
            )
            try:
                results = run_experiment(
                    args.method,
                    args.dataset,
                    cfg,
                    logger=make_logger(args.project, mode),
                    run_id=run.id,
                    config_callback=log_resolved_config,
                )
            finally:
                wandb.finish()
        all_results.append({repeat_key or "run": value, **results})

    keys = sorted({k for r in all_results for k in r if k.startswith(("val_auroc", "test_auroc"))})
    print("\n=== results ===")
    for r in all_results:
        print(
            json.dumps(
                {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in r.items()
                    if k in keys or k in (repeat_key, "run")
                }
            )
        )
    return 0


def cmd_sweep(args) -> int:
    _pin(args)
    import wandb
    from cgnn.data import get_dataset
    from cgnn.methods import get_method
    from cgnn.wandb_utils import agent_function

    get_method(args.method)  # fail fast on typos
    ds = get_dataset(args.dataset)
    overrides = parse_overrides(args.set)
    if args.cpu:
        overrides["accelerator"] = "cpu"

    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        sweep = build_sweep(args.method, args.dataset, ds.sweep_metadata(), args.sweep_config)
        sweep_id = wandb.sweep(sweep=sweep, project=args.project, entity=args.entity)
    print(f"Running sweep with ID: {sweep_id}")
    if args.create_only:
        return 0
    wandb.agent(
        sweep_id=sweep_id,
        function=agent_function(args.method, args.dataset, args.project, overrides),
        project=args.project,
        entity=args.entity,
        count=args.count,
    )
    return 0


def cmd_results(args) -> int:
    from cgnn.data import DATASETS
    from cgnn.results import PAPER_DATASETS, PAPER_ROWS, fetch_runs, format_table, paper_table

    datasets = args.datasets or PAPER_DATASETS
    rows_spec = [r for r in PAPER_ROWS if not args.methods or r[1] in args.methods]
    if args.methods:
        known = {r[1] for r in rows_spec}
        from cgnn.methods import get_method

        for m in args.methods:
            if m not in known:
                spec = get_method(m)
                comps = tuple(k for k in spec.score_keys if not k.startswith("TU")) or ("",)
                rows_spec.append((spec.paper_name or m, m, comps))
    runs = fetch_runs(
        args.project,
        args.entity,
        datasets=datasets,
        methods=[r[1] for r in rows_spec],
        known_datasets=DATASETS.names(),
    )
    print(f"Fetched {len(runs)} runs.", file=sys.stderr)
    table = paper_table(runs, datasets, rows_spec, aggregate=args.aggregate, k=args.k)
    text = format_table(table, args.format)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"Wrote {args.out}", file=sys.stderr)
    print(text)
    if args.details:
        for label, per_ds in table.items():
            for ds, s in per_ds.items():
                if s is not None:
                    print(
                        f"{label:20s} {ds:15s} comp={s.component or '-':12s} n={s.n} val={s.val_mean:.4f} runs={','.join(s.run_ids)}"
                    )
    return 0


def cmd_doctor(args) -> int:
    import importlib
    import platform

    from cgnn.checkpoints import find_checkpoints
    from cgnn.hardware import query_gpus
    from cgnn.paths import Paths

    METHODS, DATASETS = _registries()
    paths = Paths.from_env()
    ok = True
    print(f"python {platform.python_version()}  ({sys.executable})")
    for mod in ("torch", "torch_geometric", "lightning", "torchmetrics", "wandb"):
        m = importlib.import_module(mod)
        print(f"  {mod:16s} {getattr(m, '__version__', '?')}")

    print("\nOptional dependencies:")
    needed = {}
    for name in METHODS:
        for req in METHODS.get(name).requires:
            needed.setdefault(req, []).append(name)
    for req in ("faiss", "ogb", "graphesn", "graph_uq", "torch_sparse", "pandas", "pytest", "ruff"):
        try:
            importlib.import_module(req)
            status = "ok"
        except Exception as exc:
            status = f"MISSING ({type(exc).__name__})"
        users = needed.get(req) or {"ogb": ["dataset arxiv"], "pytest": ["tests"], "ruff": ["lint"]}.get(
            req, []
        )
        print(f"  {req:14s} {status:28s} {', '.join(users)}")

    print(f"\nData dir: {paths.data_dir}")
    for name in DATASETS:
        missing = DATASETS.get(name).missing_files(paths.data_dir)
        print(f"  {name:16s} {'ok' if not missing else 'missing: ' + ', '.join(missing)}")

    print(f"\nVanilla backbones in {paths.ckpt_dir} (needed by post-hoc methods):")
    for name in DATASETS:
        recs = find_checkpoints(name, paths=paths)
        if recs:
            print(
                f"  {name:16s} {len(recs):4d} checkpoints, best {recs[0].metric}={recs[0].score:.4f} ({recs[0].path.name})"
            )
        else:
            print(f"  {name:16s}    0 checkpoints")

    print("\nMethod configs:")
    config_names = {p.stem for p in (method_config_path("x").parent).glob("*.yaml")}
    missing_cfg = [m for m in METHODS if m not in config_names]
    orphan_cfg = sorted(config_names - set(METHODS))
    print(
        f"  {len(METHODS)} methods; missing configs: {missing_cfg or 'none'}; orphan configs: {orphan_cfg or 'none'}"
    )
    ok &= not missing_cfg and not orphan_cfg

    gpus = query_gpus()
    print(f"\nGPUs (policy: at most {MAX_GPUS} in total):")
    for g in gpus:
        print(
            f"  [{g.index}] {g.memory_used_mib:6d}/{g.memory_total_mib} MiB  util {g.utilization:3d}%  {'idle' if g.is_idle else 'BUSY'}"
        )
    if not gpus:
        print("  none visible")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    netrc = Path.home() / ".netrc"
    has_wandb = bool(os.environ.get("WANDB_API_KEY")) or (
        netrc.exists() and "api.wandb.ai" in netrc.read_text()
    )
    print(f"\nW&B credentials: {'found' if has_wandb else 'NOT found (run `wandb login`)'}")
    return 0 if ok else 1


# ----------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    from cgnn.wandb_utils import DEFAULT_PROJECT

    p = argparse.ArgumentParser(
        prog="cgnn", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("list", help="list methods and/or datasets")
    s.add_argument("what", nargs="?", default="all", choices=["methods", "datasets", "all"])
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("describe", help="show a method's or dataset's configuration")
    s.add_argument("name")
    s.set_defaults(func=cmd_describe)

    s = sub.add_parser("data", help="load a dataset and print its split summary")
    s.add_argument("dataset")
    s.add_argument("--split-seed", default=None)
    s.add_argument("--set", nargs="*", metavar="KEY=VALUE")
    s.set_defaults(func=cmd_data)

    def hw(sp):
        sp.add_argument(
            "--gpus",
            type=int,
            default=1,
            help=f"GPUs to pin if CUDA_VISIBLE_DEVICES is unset (max {MAX_GPUS})",
        )
        sp.add_argument("--cpu", action="store_true", help="run on CPU")
        sp.add_argument("-p", "--project", default=DEFAULT_PROJECT)
        sp.add_argument("--entity", default=None)
        sp.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="config overrides (YAML-typed values)")

    s = sub.add_parser("run", help="run a single experiment (no sweep)")
    s.add_argument("-m", "--method", required=True)
    s.add_argument("-d", "--dataset", required=True)
    s.add_argument("--seeds", nargs="*", type=int, help="repeat the run for these global seeds")
    s.add_argument("--repeat", help="repeat over KEY=v1,v2,... (e.g. backbone_rank=0,1,2,3,4)")
    s.add_argument("--from-sweep", help="start from the best config of this W&B sweep id")
    s.add_argument(
        "--wandb",
        default="auto",
        choices=["auto", "online", "offline", "disabled"],
        help="auto = disabled for the synthetic dataset, online otherwise",
    )
    s.add_argument("--group", default=None)
    s.add_argument("--tags", nargs="*")
    hw(s)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep", help="create or join a W&B sweep and run an agent")
    s.add_argument("-m", "--method", required=True)
    s.add_argument("-d", "--dataset", required=True)
    s.add_argument("-s", "--sweep-id", default="", help="join an existing sweep")
    s.add_argument("-c", "--count", type=int, default=None, help="runs for this agent (default: unlimited)")
    s.add_argument(
        "--sweep-config", default=None, help="custom sweep YAML (default: configs/methods/<method>.yaml)"
    )
    s.add_argument("--create-only", action="store_true", help="only create the sweep and print its id")
    hw(s)
    s.set_defaults(func=cmd_sweep)

    s = sub.add_parser("results", help="fetch W&B runs and print a paper-style table")
    s.add_argument("-d", "--datasets", nargs="*")
    s.add_argument("-m", "--methods", nargs="*")
    s.add_argument("-p", "--project", default=DEFAULT_PROJECT)
    s.add_argument("--entity", default=None)
    s.add_argument("--aggregate", default="topk", choices=["topk", "group"])
    s.add_argument("-k", type=int, default=5, help="replicates for --aggregate topk")
    s.add_argument("--format", default="md", choices=["md", "csv", "latex"])
    s.add_argument("--out", default=None)
    s.add_argument("--details", action="store_true", help="also print selected components and run ids")
    s.set_defaults(func=cmd_results)

    s = sub.add_parser("doctor", help="check environment, data, checkpoints, GPUs")
    s.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    import warnings

    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")  # noise from ogb's deps
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
