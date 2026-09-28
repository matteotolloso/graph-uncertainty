#!/usr/bin/env python3
"""Node-feature sparsity of the benchmark datasets (companion statistics for Table 1).

Graph sparsity concerns missing edges in the adjacency matrix; this script measures *node-feature
sparsity* instead: for the N-by-D feature matrix X consumed by the models, density = nnz(X) / (N * D)
and sparsity = 1 - density. Floating-point entries with |x| <= ``--zero-threshold`` count as zero.

Features come from the ``cgnn`` dataset registry (``cgnn.data.load_dataset``). For Patents, Reddit2
and Coauthor, memory-safe feature-only paths mirror their loaders without materialising the large dense
matrices. Output (CSV, JSON, LaTeX): ``docs/results/feature_sparsity/`` by default.

    python scripts/analysis/feature_sparsity.py [--datasets squirrel arxiv ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import scipy.sparse as sp
import torch

from cgnn.data import DATASETS, load_dataset
from cgnn.paths import REPO_ROOT, Paths

# Registry names; `synthetic` is a test fixture, not a benchmark dataset.
DEFAULT_DATASETS = [n for n in DATASETS.names() if n != "synthetic"]
CSV_COLUMNS = [
    "dataset",
    "num_nodes",
    "num_features",
    "total_entries",
    "nonzero_entries",
    "feature_density",
    "feature_sparsity",
    "mean_nnz_per_node",
    "std_nnz_per_node",
    "median_nnz_per_node",
    "min_nnz_per_node",
    "max_nnz_per_node",
    "mean_node_density",
    "median_node_density",
    "zero_feature_node_fraction",
    "storage_format",
    "dtype",
    "feature_source",
]


@dataclass
class LoadedFeatures:
    """A feature matrix plus metadata describing the model-facing form."""

    matrix: Any
    feature_source: str
    storage_format: str | None = None
    dtype: str | None = None
    analysis_cast_dtype: Any | None = None


FEATURE_SOURCES = {
    "patents": "snap-patents.mat['node_feat'] (the loader densifies to float32; analysed as the equivalent sparse source)",
    "coauthor": "Coauthor-CS NPZ attributes, float32 with positive values binarised to 1 as in PyG read_npz",
    "reddit2": "reddit2/raw/feats.npy cast to float32 by the loader (analysed row-wise from a memory map)",
}


def _display_name(name: str) -> str:
    return DATASETS.get(name).paper_name


def _load_via_registry(name: str) -> LoadedFeatures:
    """Features exactly as the models see them (``bundle.data.x``); the split does not affect X."""
    bundle = load_dataset(name, {"batch_size": -1, "num_neighbors": 10})
    return LoadedFeatures(matrix=bundle.data.x, feature_source=f"cgnn.data.load_dataset('{name}').data.x")


def _load_patents_features_safely(data_dir: Path) -> LoadedFeatures:
    """Mirror the Patents loader's feature path without its dense conversion."""
    path = data_dir / "snap-patents.mat"
    if not path.exists():
        raise FileNotFoundError(f"{path} is required by cgnn/data/sources/year_graphs.py")
    payload = scipy.io.loadmat(path, variable_names=["node_feat"])
    matrix = payload["node_feat"]
    if not sp.issparse(matrix):
        matrix = np.asarray(matrix)
    return LoadedFeatures(
        matrix=matrix,
        feature_source=FEATURE_SOURCES["patents"],
        storage_format="torch_dense",
        dtype="torch.float32",
        analysis_cast_dtype=np.float32,
    )


def _load_reddit2_features_safely(data_dir: Path) -> LoadedFeatures:
    """Mirror Reddit2's ``feats.npy`` load using a read-only memory map."""
    path = data_dir / "reddit2" / "raw" / "feats.npy"
    if not path.exists():
        return _load_via_registry("reddit2")  # lets the loader download/process it
    return LoadedFeatures(
        matrix=np.load(path, mmap_mode="r"),
        feature_source=FEATURE_SOURCES["reddit2"],
        storage_format="torch_dense",
        dtype="torch.float32",
        analysis_cast_dtype=np.float32,
    )


def _load_coauthor_features_safely(data_dir: Path) -> LoadedFeatures:
    """Mirror PyG ``read_npz`` feature preprocessing without densifying X."""
    path = data_dir / "coauthors" / "CS" / "raw" / "ms_academic_cs.npz"
    if not path.exists():
        return _load_via_registry("coauthor")  # lets the loader download it
    with np.load(path) as payload:
        matrix = sp.csr_matrix(
            (payload["attr_data"], payload["attr_indices"], payload["attr_indptr"]),
            shape=tuple(int(v) for v in payload["attr_shape"]),
        )
    # read_npz densifies, casts to float32, then sets x[x > 0] = 1: the value-only equivalent keeps the
    # ~125M-entry matrix sparse and gives the same X.
    matrix.sum_duplicates()
    matrix.data = matrix.data.astype(np.float32, copy=False)
    matrix.data[matrix.data > 0] = 1
    return LoadedFeatures(
        matrix=matrix,
        feature_source=FEATURE_SOURCES["coauthor"],
        storage_format="torch_dense",
        dtype="torch.float32",
    )


def load_features(name: str, data_dir: Path) -> LoadedFeatures:
    """Load the exact feature representation consumed by the models."""
    safe = {
        "patents": _load_patents_features_safely,
        "reddit2": _load_reddit2_features_safely,
        "coauthor": _load_coauthor_features_safely,
    }
    return safe[name](data_dir) if name in safe else _load_via_registry(name)


def _unwrap_data_x(matrix: Any) -> Any:
    """Accept a raw matrix or an object such as PyG ``Data`` with ``.x``."""

    if sp.issparse(matrix):
        return matrix
    if isinstance(matrix, torch.Tensor):
        return matrix
    if isinstance(matrix, np.ndarray):
        return matrix
    if getattr(matrix, "x", None) is not None:
        return matrix.x
    return matrix


def _torch_nonzero_mask(values: Any, threshold: float) -> Any:
    if values.is_floating_point() or values.is_complex():
        return values.abs() > threshold
    return values != 0


def _numpy_nonzero_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    if np.issubdtype(values.dtype, np.floating) or np.issubdtype(values.dtype, np.complexfloating):
        return np.abs(values) > threshold
    return values != 0


def _count_torch_dense(matrix: Any, threshold: float, target_chunk_entries: int) -> np.ndarray:
    num_nodes, num_features = map(int, matrix.shape)
    rows_per_chunk = max(1, target_chunk_entries // max(1, num_features))
    row_counts = np.empty(num_nodes, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, num_nodes, rows_per_chunk):
            stop = min(num_nodes, start + rows_per_chunk)
            mask = _torch_nonzero_mask(matrix[start:stop], threshold)
            counts = torch.count_nonzero(mask, dim=1)
            row_counts[start:stop] = counts.detach().cpu().numpy()
    return row_counts


def _count_torch_sparse(matrix: Any, threshold: float) -> np.ndarray:
    num_nodes = int(matrix.shape[0])
    if matrix.layout != torch.sparse_coo:
        matrix = matrix.to_sparse_coo()
    matrix = matrix.coalesce()
    indices = matrix.indices()
    values = matrix.values()
    if values.ndim != 1 or indices.shape[0] != 2:
        raise ValueError("Only two-dimensional scalar-valued sparse tensors are supported")
    keep = _torch_nonzero_mask(values, threshold)
    rows = indices[0, keep]
    return torch.bincount(rows, minlength=num_nodes).to(dtype=torch.int64, device="cpu").numpy()


def _canonical_scipy_sparse(matrix: Any) -> Any:
    if getattr(matrix, "has_canonical_format", True):
        return matrix
    matrix = matrix.copy()
    matrix.sum_duplicates()
    return matrix


def _count_scipy_sparse(
    matrix: Any,
    threshold: float,
    target_chunk_entries: int,
    cast_dtype: Any | None,
) -> np.ndarray:
    """Count sparse rows directly, ignoring explicit near-zero stored values."""

    matrix = _canonical_scipy_sparse(matrix)
    num_nodes, num_features = map(int, matrix.shape)
    row_counts = np.zeros(num_nodes, dtype=np.int64)

    if sp.isspmatrix_csr(matrix) or isinstance(matrix, getattr(sp, "csr_array", ())):
        rows_per_chunk = max(1, target_chunk_entries // max(1, num_features))
        for start in range(0, num_nodes, rows_per_chunk):
            stop = min(num_nodes, start + rows_per_chunk)
            first = int(matrix.indptr[start])
            last = int(matrix.indptr[stop])
            values = np.asarray(matrix.data[first:last], dtype=cast_dtype)
            keep = _numpy_nonzero_mask(values, threshold)
            prefix = np.empty(keep.size + 1, dtype=np.int64)
            prefix[0] = 0
            np.cumsum(keep, dtype=np.int64, out=prefix[1:])
            pointers = matrix.indptr[start : stop + 1] - first
            row_counts[start:stop] = prefix[pointers[1:]] - prefix[pointers[:-1]]
        return row_counts

    if sp.isspmatrix_csc(matrix) or isinstance(matrix, getattr(sp, "csc_array", ())):
        for column in range(num_features):
            first = int(matrix.indptr[column])
            last = int(matrix.indptr[column + 1])
            values = np.asarray(matrix.data[first:last], dtype=cast_dtype)
            keep = _numpy_nonzero_mask(values, threshold)
            np.add.at(row_counts, matrix.indices[first:last][keep], 1)
        return row_counts

    if sp.isspmatrix_coo(matrix) or isinstance(matrix, getattr(sp, "coo_array", ())):
        values = np.asarray(matrix.data, dtype=cast_dtype)
        keep = _numpy_nonzero_mask(values, threshold)
        np.add.at(row_counts, matrix.row[keep], 1)
        return row_counts

    return _count_scipy_sparse(matrix.tocsr(), threshold, target_chunk_entries, cast_dtype)


def _count_numpy_dense(
    matrix: np.ndarray,
    threshold: float,
    target_chunk_entries: int,
    cast_dtype: Any | None,
) -> np.ndarray:
    num_nodes, num_features = map(int, matrix.shape)
    rows_per_chunk = max(1, target_chunk_entries // max(1, num_features))
    row_counts = np.empty(num_nodes, dtype=np.int64)
    for start in range(0, num_nodes, rows_per_chunk):
        stop = min(num_nodes, start + rows_per_chunk)
        chunk = np.asarray(matrix[start:stop], dtype=cast_dtype)
        row_counts[start:stop] = np.count_nonzero(_numpy_nonzero_mask(chunk, threshold), axis=1)
    return row_counts


def _infer_storage_format(matrix: Any) -> str:
    if isinstance(matrix, torch.Tensor):
        if matrix.layout == torch.strided:
            return "torch_dense"
        return f"torch_{str(matrix.layout).replace('torch.', '')}"
    if sp.issparse(matrix):
        return f"scipy_{matrix.format}"
    if isinstance(matrix, np.memmap):
        return "numpy_memmap"
    if isinstance(matrix, np.ndarray):
        return "numpy_dense"
    raise TypeError(f"Unsupported feature matrix type: {type(matrix).__name__}")


def _infer_dtype(matrix: Any) -> str:
    if isinstance(matrix, torch.Tensor):
        return str(matrix.dtype)
    return str(matrix.dtype)


def compute_sparsity_statistics(
    matrix: Any,
    zero_threshold: float = 1e-12,
    *,
    storage_format: str | None = None,
    dtype: str | None = None,
    cast_dtype: Any | None = None,
    target_chunk_entries: int = 8_000_000,
) -> dict[str, Any]:
    """Return feature-sparsity statistics without densifying sparse inputs.

    Dense NumPy/PyTorch matrices are scanned in row chunks.  SciPy and PyTorch
    sparse matrices are counted from stored values after duplicate coordinates
    are coalesced, so explicit zeros and near-zero values are handled correctly.
    """

    if not math.isfinite(zero_threshold) or zero_threshold < 0:
        raise ValueError("zero_threshold must be a finite non-negative number")
    if target_chunk_entries <= 0:
        raise ValueError("target_chunk_entries must be positive")

    matrix = _unwrap_data_x(matrix)
    if not hasattr(matrix, "shape") or len(matrix.shape) != 2:
        raise ValueError("Feature matrix X must be two-dimensional")
    num_nodes, num_features = map(int, matrix.shape)
    if num_nodes <= 0 or num_features <= 0:
        raise ValueError(f"Feature matrix has invalid shape {matrix.shape}")

    if isinstance(matrix, torch.Tensor):
        if matrix.layout == torch.strided:
            row_counts = _count_torch_dense(matrix, zero_threshold, target_chunk_entries)
        else:
            row_counts = _count_torch_sparse(matrix, zero_threshold)
    elif sp.issparse(matrix):
        row_counts = _count_scipy_sparse(matrix, zero_threshold, target_chunk_entries, cast_dtype)
    elif isinstance(matrix, np.ndarray):
        row_counts = _count_numpy_dense(matrix, zero_threshold, target_chunk_entries, cast_dtype)
    else:
        raise TypeError(f"Unsupported feature matrix type: {type(matrix).__name__}")

    total_entries = num_nodes * num_features
    nonzero_entries = int(row_counts.sum(dtype=np.int64))
    density = nonzero_entries / total_entries
    node_densities = row_counts.astype(np.float64) / num_features

    return {
        "num_nodes": num_nodes,
        "num_features": num_features,
        "total_entries": total_entries,
        "nonzero_entries": nonzero_entries,
        "feature_density": float(density),
        "feature_sparsity": float(1.0 - density),
        "mean_nnz_per_node": float(np.mean(row_counts)),
        "std_nnz_per_node": float(np.std(row_counts)),
        "median_nnz_per_node": float(np.median(row_counts)),
        "min_nnz_per_node": int(np.min(row_counts)),
        "max_nnz_per_node": int(np.max(row_counts)),
        "mean_node_density": float(np.mean(node_densities)),
        "median_node_density": float(np.median(node_densities)),
        "zero_feature_node_fraction": float(np.mean(row_counts == 0)),
        "storage_format": storage_format or _infer_storage_format(matrix),
        "dtype": dtype or _infer_dtype(matrix),
    }


def run_internal_validation(zero_threshold: float) -> None:
    """Check dense, SciPy sparse, and PyTorch sparse implementations agree."""

    dense = np.array(
        [[0.0, 2.0, 1e-13, -3.0], [0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 5.0, 0.0]],
        dtype=np.float64,
    )
    rows, columns = np.nonzero(dense)
    indices = torch.tensor(np.vstack([rows, columns]), dtype=torch.long)
    values = torch.tensor(dense[rows, columns], dtype=torch.float64)
    representations: list[tuple[str, Any]] = [
        ("NumPy dense", dense),
        ("SciPy CSR", sp.csr_matrix(dense)),
        ("PyTorch sparse COO", torch.sparse_coo_tensor(indices, values, dense.shape)),
    ]

    comparable_keys = [
        key for key in CSV_COLUMNS if key not in {"dataset", "storage_format", "dtype", "feature_source"}
    ]
    reference = compute_sparsity_statistics(representations[0][1], zero_threshold=zero_threshold)
    for label, representation in representations[1:]:
        candidate = compute_sparsity_statistics(representation, zero_threshold=zero_threshold)
        for key in comparable_keys:
            if not math.isclose(
                float(reference[key]),
                float(candidate[key]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise AssertionError(
                    f"Internal validation failed for {label}: {key}={candidate[key]} != {reference[key]}"
                )
    print("Internal validation passed: " + ", ".join(label for label, _ in representations))


def _write_csv(path: Path, results: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(results)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _write_latex(path: Path, results: Sequence[dict[str, Any]], zero_threshold: float) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        (
            r"Dataset & $N$ & $D$ & $\mathrm{nnz}(X)$ & Density & Sparsity "
            r"& Mean nnz/node & Zero-node frac. \\"
        ),
        r"\midrule",
    ]
    for row in results:
        lines.append(
            "{} & {} & {} & {} & {:.3e} & {:.3e} & {:.2f} & {:.3e} \\\\".format(
                _latex_escape(str(row["dataset"])),
                row["num_nodes"],
                row["num_features"],
                row["nonzero_entries"],
                row["feature_density"],
                row["feature_sparsity"],
                row["mean_nnz_per_node"],
                row["zero_feature_node_fraction"],
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Node-feature (not graph-edge) sparsity. Floating-point "
                rf"entries satisfy $|x|>{zero_threshold:.1e}$ to count as nonzero.}}"
            ),
            r"\label{tab:feature-sparsity}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(results: Sequence[dict[str, Any]]) -> None:
    if not results:
        print("\nNo datasets were loaded successfully.")
        return
    headers = [
        "Dataset",
        "N",
        "D",
        "nnz(X)",
        "Mean nnz/node",
        "Density",
        "Sparsity",
        "Zero nodes",
        "Storage",
    ]
    rows = [
        [
            str(row["dataset"]),
            f"{row['num_nodes']:,}",
            f"{row['num_features']:,}",
            f"{row['nonzero_entries']:,}",
            f"{row['mean_nnz_per_node']:.2f}",
            f"{row['feature_density']:.3e}",
            f"{row['feature_sparsity']:.3e}",
            f"{row['zero_feature_node_fraction']:.3e}",
            str(row["storage_format"]),
        ]
        for row in results
    ]
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(len(headers))]
    print("\nNode-feature sparsity summary (sparsest to densest)")
    print("  ".join(value.ljust(widths[i]) for i, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparsity of the node-feature matrices of the benchmark.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=DEFAULT_DATASETS,
        metavar="NAME",
        help=f"registry names (default: all): {', '.join(DEFAULT_DATASETS)}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "results" / "feature_sparsity",
        help="directory for CSV, JSON and LaTeX outputs",
    )
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=1e-12,
        help="floating-point entries with abs(x) <= this value count as zero",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not math.isfinite(args.zero_threshold) or args.zero_threshold < 0:
        raise SystemExit("--zero-threshold must be a finite non-negative number")
    data_dir = Paths.from_env().data_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_internal_validation(args.zero_threshold)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for name in args.datasets:
        try:
            print(f"\nLoading {name} ...")
            loaded = load_features(name, data_dir)
            statistics = compute_sparsity_statistics(
                loaded.matrix,
                zero_threshold=args.zero_threshold,
                storage_format=loaded.storage_format,
                dtype=loaded.dtype,
                cast_dtype=loaded.analysis_cast_dtype,
            )
            results.append(
                {"dataset": _display_name(name), **statistics, "feature_source": loaded.feature_source}
            )
        except Exception as exc:  # continue and make every omission explicit
            errors.append({"dataset": name, "error_type": type(exc).__name__, "error": str(exc)})
            print(f"ERROR loading {name}: {type(exc).__name__}: {exc}")

    results.sort(key=lambda row: (row["feature_density"], row["dataset"]))
    _write_csv(args.output_dir / "feature_sparsity.csv", results)
    _write_json(args.output_dir / "feature_sparsity.json", results)
    _write_latex(args.output_dir / "feature_sparsity.tex", results, args.zero_threshold)
    if errors:
        _write_json(args.output_dir / "feature_sparsity_errors.json", errors)
    _print_summary(results)
    print(f"\nWrote outputs to {args.output_dir}")
    print(f"Datasets succeeded: {len(results)}; failed: {len(errors)}")
    return 0 if results and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
