"""Backward-compatible entry point: ``python main.py -m <method> -d <dataset> [-c N] [-s SWEEP_ID]``.

Equivalent to ``cgnn sweep ...`` (see ``cgnn --help``). Kept so existing
scripts and notes (``scripts/run_all.sh``) keep working.
"""

from __future__ import annotations

import argparse
import os

from cgnn.cli import main as cgnn_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a W&B sweep (legacy interface; prefer `cgnn sweep`).")
    parser.add_argument("-d", "--dataset", default="squirrel")
    parser.add_argument("-s", "--sweep", default="", help="existing sweep id to join")
    parser.add_argument("-m", "--model", default="vanilla", help="method name (see `cgnn list methods`)")
    parser.add_argument("-p", "--project_name", default="graph-uncertainty")
    parser.add_argument("--save_path", default=None, help="checkpoint dir (sets CGNN_CKPT_DIR)")
    parser.add_argument("-c", "--count", type=int, default=None)
    args = parser.parse_args()

    if args.save_path:
        os.environ["CGNN_CKPT_DIR"] = args.save_path
    argv = ["sweep", "-m", args.model, "-d", args.dataset, "-p", args.project_name]
    if args.sweep:
        argv += ["-s", args.sweep]
    if args.count is not None:
        argv += ["-c", str(args.count)]
    raise SystemExit(cgnn_main(argv))


if __name__ == "__main__":
    main()
