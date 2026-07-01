#!/usr/bin/env python
"""Post-hoc z-regularization: smooth expression across adjacent z-levels to ensure coherent 3D stack.

Loads generated h5ad files from a directory, applies per-gene per-region
quadratic-penalty z-regularization, rescales spot-level counts, saves
regularized AnnData files and z-coherence metrics.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from FEAST.de_novo.z_regularize import (
    class_anchor_weight,
    regularize_mean_profiles,
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_generated_slices(input_dir: Path) -> list[tuple[str, Path, float]]:
    """Return (name, path, z) tuples for all generated *.h5ad files, sorted by z."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    entries: list[tuple[str, Path, float]] = []
    for path in sorted(input_dir.glob("*.h5ad")):
        adata = ad.read_h5ad(path, backed="r")
        try:
            if "z" not in adata.obs:
                raise KeyError(f"{path.name} is missing obs['z'].")
            z_vals = np.asarray(adata.obs["z"], dtype=float)
            if not np.all(np.isfinite(z_vals)):
                raise ValueError(f"{path.name} has invalid obs['z'] values.")
            if not np.allclose(z_vals, z_vals[0], rtol=0.0, atol=1e-8):
                raise ValueError(f"{path.name} contains multiple z values.")
            z = float(z_vals[0])
        finally:
            adata.file.close()
        entries.append((str(adata.uns.get("slice_name", path.stem)), path, z))

    if not entries:
        raise ValueError(f"No generated h5ad files found in {input_dir}.")

    entries.sort(key=lambda item: item[2])
    return entries


# ---------------------------------------------------------------------------
# Per-gene per-region mean extraction
# ---------------------------------------------------------------------------

def extract_label_mean_matrix(
    slices: Sequence[ad.AnnData],
    z_values: Sequence[float],
    gene_names: Sequence[str],
    label_key: str,
    min_spots: int,
) -> dict[str, tuple[pd.DataFrame, np.ndarray, list[int]]]:
    """For each region label, build a (z, gene) DataFrame of mean counts.

    Returns
    -------
    dict[label] -> (means_df, z_pass, n_spots_list)
      means_df : index=z, columns=genes
      z_pass : bool array, which z-levels pass min_spots
      n_spots_list : per-z-level spot counts (0 where z_pass is False)
    """
    all_labels: set[str] = set()
    for adata in slices:
        if label_key in adata.obs:
            all_labels.update(adata.obs[label_key].astype(str).unique().tolist())

    z_arr = np.asarray(z_values, dtype=float)
    nz = len(z_arr)
    gene_list = list(gene_names)

    result: dict[str, tuple[pd.DataFrame, np.ndarray, list[int]]] = {}
    for label in sorted(all_labels):
        means = np.full((nz, len(gene_list)), np.nan, dtype=np.float64)
        z_pass = np.zeros(nz, dtype=bool)
        n_spots_list = [0] * nz

        for zi, adata in enumerate(slices):
            if label_key not in adata.obs:
                continue
            mask = adata.obs[label_key].astype(str).to_numpy() == label
            n_spots = int(mask.sum())
            n_spots_list[zi] = n_spots
            if n_spots < int(min_spots):
                continue
            counts = _counts_for_genes(adata, gene_list)
            means[zi, :] = counts[mask, :].mean(axis=0)
            z_pass[zi] = True

        result[label] = (
            pd.DataFrame(means, index=pd.Index(z_arr, name="z"), columns=gene_list),
            z_pass,
            n_spots_list,
        )

    return result


# ---------------------------------------------------------------------------
# Quadratic-penalty smoothing and rescaling
# ---------------------------------------------------------------------------

def smooth_and_rescale(
    slices: list[ad.AnnData],
    label_means: dict[str, tuple[pd.DataFrame, np.ndarray, list[int]]],
    gene_names: Sequence[str],
    label_key: str,
    lambda_1: float,
    lambda_2: float,
    output_dir: Path,
) -> list[Path]:
    """Apply quadratic-penalty z-regularization and rescale counts per slice.

    Returns list of output paths for the regularized h5ad files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_list = list(gene_names)
    out_paths: list[Path] = []

    # ---- Phase 1: per-label, solve for regularized means across all z ----
    label_regularized: dict[str, pd.DataFrame] = {}
    for label, (means_df, z_pass, n_spots_list) in label_means.items():
        valid_idx = np.flatnonzero(z_pass)
        if len(valid_idx) < 2:
            # Not enough z-levels to regularize; leave means as-is
            label_regularized[label] = means_df
            continue

        n_nodes = int(len(valid_idx))
        z_valid = np.asarray(means_df.index[valid_idx], dtype=np.float64)

        # Build y_matrix (log1p of means) and weights
        y_matrix = np.zeros((n_nodes, len(gene_list)), dtype=np.float64)
        node_weights = np.zeros(n_nodes, dtype=np.float64)

        for local_idx, global_idx in enumerate(valid_idx):
            row_means = means_df.iloc[int(global_idx)].to_numpy(dtype=np.float64)
            y_matrix[local_idx, :] = np.log1p(np.clip(row_means, 0.0, None))
            node_weights[local_idx] = class_anchor_weight(n_spots_list[int(global_idx)])

        # Solve
        solved = regularize_mean_profiles(
            y_matrix=y_matrix,
            node_weights=node_weights,
            z_values=z_valid,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
        )  # (n_nodes, n_genes) in original space

        # Build a (z, gene) DataFrame matching the original means_df shape
        regularized_arr = np.full_like(means_df.to_numpy(dtype=float), np.nan, dtype=np.float64)
        for local_idx, global_idx in enumerate(valid_idx):
            regularized_arr[int(global_idx), :] = solved[local_idx, :]
        label_regularized[label] = pd.DataFrame(
            regularized_arr,
            index=means_df.index,
            columns=means_df.columns,
        )

    # ---- Phase 2: per-slice, rescale counts ----
    for zi, adata_in in enumerate(slices):
        adata = adata_in.copy()
        counts = _counts_for_genes(adata, gene_list)
        labels = adata.obs[label_key].astype(str).to_numpy() if label_key in adata.obs else None

        if labels is not None:
            for label, reg_df in label_regularized.items():
                label_mask = labels == label
                if not np.any(label_mask):
                    continue

                reg_row = reg_df.iloc[zi]
                reg_means = reg_row.to_numpy(dtype=np.float64)
                finite_mask = np.isfinite(reg_means)

                if not np.any(finite_mask):
                    continue

                for g, gene in enumerate(gene_list):
                    if not finite_mask[g]:
                        continue
                    desired_mu = float(reg_means[g])
                    current_mu = float(counts[label_mask, g].mean())
                    if current_mu <= 0.0 or desired_mu <= 0.0:
                        continue
                    scale = desired_mu / current_mu
                    counts[label_mask, g] = np.rint(np.clip(counts[label_mask, g] * scale, 0.0, None))

        # Write regularized counts back
        if "counts" in adata.layers:
            adata.layers["counts"] = _set_counts(adata.layers["counts"], counts, gene_list)
        adata.X = _set_counts(adata.X, counts, gene_list)

        # Stamp metadata
        adata.uns.setdefault("de_novo", {}).setdefault("experiment", {})["z_regularization"] = {
            "enabled": True,
            "method": "quadratic_penalty",
            "lambda_1": float(lambda_1),
            "lambda_2": float(lambda_2),
            "label_key": str(label_key),
        }

        out_path = output_dir / f"regularized_z{zi:03d}_{adata_in.obs['z'].iloc[0]:.2f}.h5ad"
        _sanitize_uns(adata)
        adata.write_h5ad(out_path)
        out_paths.append(out_path)
        del adata
        gc.collect()

    return out_paths


# ---------------------------------------------------------------------------
# Z-coherence metrics
# ---------------------------------------------------------------------------

def compute_z_coherence(
    slices: Sequence[ad.AnnData],
    z_values: Sequence[float],
    gene_names: Sequence[str],
    label_key: str,
    min_spots: int,
) -> pd.DataFrame:
    """Compute z-coherence metrics: adjacent-z Pearson correlations and per-gene lag-1 autocorrelation.

    Returns a DataFrame with columns:
        [metric, class, gene, value]
    """
    gene_list = list(gene_names)
    all_labels: set[str] = set()
    for adata in slices:
        if label_key in adata.obs:
            all_labels.update(adata.obs[label_key].astype(str).unique().tolist())

    z_arr = np.asarray(z_values, dtype=float)
    rows: list[dict[str, Any]] = []

    # --- Per-label adjacent-z Pearson correlation ---
    for label in sorted(all_labels):
        # Build (n_z, n_genes) mean matrix
        means_by_z: list[np.ndarray] = []
        valid_z: list[float] = []
        for zi, adata in enumerate(slices):
            if label_key not in adata.obs:
                continue
            mask = adata.obs[label_key].astype(str).to_numpy() == label
            if int(mask.sum()) < int(min_spots):
                continue
            counts = _counts_for_genes(adata, gene_list)
            means_by_z.append(counts[mask, :].mean(axis=0))
            valid_z.append(z_arr[zi])

        if len(valid_z) < 2:
            continue

        means_arr = np.stack(means_by_z, axis=0)  # (nz, ng)
        for g, gene in enumerate(gene_list):
            # Adjacent-z Pearson
            for a in range(len(valid_z) - 1):
                b = a + 1
                corr = _pearson_r(means_arr[a, :], means_arr[b, :])
                if np.isfinite(corr):
                    rows.append({
                        "metric": "adjacent_z_pearson",
                        "class": label,
                        "gene": gene,
                        "value": corr,
                        "z_a": valid_z[a],
                        "z_b": valid_z[b],
                    })

            # Per-gene z-autocorrelation at lag 1
            col = means_arr[:, g]
            finite = np.isfinite(col)
            if finite.sum() < 2:
                continue
            col_f = col[finite]
            if col_f.size < 2:
                continue
            autocorr = _lag_autocorr(col_f, lag=1)
            if np.isfinite(autocorr):
                rows.append({
                    "metric": "z_autocorr_lag1",
                    "class": label,
                    "gene": gene,
                    "value": autocorr,
                    "z_a": np.nan,
                    "z_b": np.nan,
                })

    if not rows:
        return pd.DataFrame(columns=["metric", "class", "gene", "value", "z_a", "z_b"])
    return pd.DataFrame(rows)


def summarize_z_coherence(df: pd.DataFrame) -> pd.DataFrame:
    """Produce summary statistics from raw z-coherence rows."""
    if df.empty:
        return pd.DataFrame(columns=["metric", "n", "mean", "median", "min", "max"])
    summary_rows = []
    for metric, group in df.groupby("metric"):
        values = group["value"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        summary_rows.append({
            "metric": metric,
            "n": int(len(finite)),
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "min": float(np.min(finite)) if finite.size else float("nan"),
            "max": float(np.max(finite)) if finite.size else float("nan"),
        })
    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _counts_for_genes(adata: ad.AnnData, gene_names: Sequence[str]) -> np.ndarray:
    """Extract a dense (n_obs, n_genes) count matrix."""
    view = adata[:, list(gene_names)]
    matrix = view.layers["counts"] if "counts" in view.layers else view.X
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    elif hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float64)


def _set_counts(
    target: Any,
    new_counts: np.ndarray,
    gene_names: Sequence[str],
) -> Any:
    """Write new integer counts back to a matrix-like object, subsetting to gene columns."""
    gene_list = list(gene_names)
    if sparse.issparse(target):
        full = target.toarray().astype(np.float64)
        return new_counts.astype(np.int32)
    arr = np.asarray(target, dtype=np.float64)
    n_genes_total = arr.shape[1]
    out = arr.copy()
    # Only update the gene columns we touched; keep others untouched
    for g, col_counts in enumerate(new_counts.T):
        col_idx = g  # assume same gene order
        if col_idx < n_genes_total:
            out[:, col_idx] = np.rint(np.clip(col_counts, 0.0, None)).astype(np.int32)
    return out


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.sqrt(np.sum(xc * xc) * np.sum(yc * yc))
    if denom <= 0:
        return float("nan")
    return float(np.sum(xc * yc) / denom)


def _lag_autocorr(x: np.ndarray, lag: int = 1) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size <= lag:
        return float("nan")
    return _pearson_r(x[:-lag], x[lag:])


def _sanitize_uns(adata: ad.AnnData) -> None:
    """Remove uns entries that can't be stored in h5ad."""
    uns_clean = {}
    for key, value in adata.uns.items():
        try:
            uns_clean[key] = _clean_value(value)
        except Exception:
            uns_clean[key] = str(value)
    adata.uns = uns_clean


def _clean_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc z-regularization of generated 3D stack h5ad files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", required=True, type=Path,
        help="Directory containing generated h5ad files.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory for regularized h5ad output.",
    )
    parser.add_argument(
        "--label-key", default="region", type=str,
        help="Observation column with region/domain labels.",
    )
    parser.add_argument(
        "--sigma", default=1.0, type=float,
        help="Smoothing strength (mapped to lambda_1=1/sigma, lambda_2=0.5/sigma).",
    )
    parser.add_argument(
        "--lambda1", default=None, type=float,
        help="First-derivative penalty strength (overrides --sigma mapping).",
    )
    parser.add_argument(
        "--lambda2", default=None, type=float,
        help="Second-derivative penalty strength (overrides --sigma mapping).",
    )
    parser.add_argument(
        "--min-spots", default=10, type=int,
        help="Minimum spots a label must have at a z-level to be included.",
    )
    parser.add_argument(
        "--metrics-output", default=None, type=Path,
        help="Path for z-coherence metrics CSV (default: <output-dir>/z_coherence_metrics.csv).",
    )
    parser.add_argument(
        "--summary-output", default=None, type=Path,
        help="Path for z-coherence summary CSV (default: <output-dir>/z_coherence_summary.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Discover slices
    print(f"Discovering generated slices in {args.input_dir}...")
    entries = discover_generated_slices(args.input_dir)
    z_values = [z for _, _, z in entries]
    print(f"  Found {len(entries)} slices, z range [{min(z_values):.2f}, {max(z_values):.2f}]")

    # Load all slices into memory
    slices: list[ad.AnnData] = []
    for name, path, z in entries:
        adata = ad.read_h5ad(path)
        if args.label_key not in adata.obs:
            print(f"  WARNING: {name} missing obs['{args.label_key}'], skipping.", file=sys.stderr)
            slices.append(adata)
            continue
        adata.uns["slice_name"] = name
        slices.append(adata)

    # Determine common gene set
    gene_sets = [set(str(g) for g in adata.var_names) for adata in slices]
    common_genes = sorted(gene_sets[0].intersection(*gene_sets[1:])) if len(gene_sets) > 1 else sorted(gene_sets[0])
    if not common_genes:
        raise ValueError("No common genes across all slices.")
    print(f"  Common genes: {len(common_genes)}")

    # 2. Extract per-label per-gene mean profiles
    print("Extracting per-label per-gene mean profiles...")
    label_means = extract_label_mean_matrix(
        slices=slices,
        z_values=z_values,
        gene_names=common_genes,
        label_key=args.label_key,
        min_spots=args.min_spots,
    )
    print(f"  {len(label_means)} labels with sufficient data")

    # Resolve lambda values from --sigma (or --lambda1/--lambda2 overrides)
    lambda_1 = args.lambda1 if args.lambda1 is not None else (1.0 / float(args.sigma))
    lambda_2 = args.lambda2 if args.lambda2 is not None else (0.5 / float(args.sigma))

    # 3. Apply quadratic-penalty regularization and rescale
    print(f"Applying quadratic-penalty z-regularization (lambda_1={lambda_1:.4f}, lambda_2={lambda_2:.4f})...")
    out_paths = smooth_and_rescale(
        slices=list(slices),
        label_means=label_means,
        gene_names=common_genes,
        label_key=args.label_key,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        output_dir=args.output_dir,
    )
    print(f"  Wrote {len(out_paths)} regularized h5ad files")

    # 4. Compute z-coherence metrics
    print("Computing z-coherence metrics...")
    coh_raw = compute_z_coherence(
        slices=list(slices),
        z_values=z_values,
        gene_names=common_genes,
        label_key=args.label_key,
        min_spots=args.min_spots,
    )
    coh_summary = summarize_z_coherence(coh_raw)

    metrics_path = args.metrics_output or args.output_dir / "z_coherence_metrics.csv"
    summary_path = args.summary_output or args.output_dir / "z_coherence_summary.csv"

    coh_raw.to_csv(metrics_path, index=False)
    coh_summary.to_csv(summary_path, index=False)

    print(f"  Wrote z-coherence metrics: {metrics_path}")
    print(f"  Wrote z-coherence summary: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
