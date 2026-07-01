#!/usr/bin/env python3
"""Benchmark FEAST conditional cross-slice transfer via simulate_single_slice.

Use the core simulator's hungarian x ot_spatial mode to generate a target
slice from source expression + target geometry/labels, and compare generated
counts against held-out real target counts.

v2 supports cross_slice and mask_complete modes.

Usage:
    # Cross-slice: source -> target
    PYTHONPATH=/maiziezhou_lab2/yiru/FEAST/src python run.py \\
        --mode cross_slice \\
        --data-dir /maiziezhou_lab2/yiru/Datasets/Processed/spatialLIBD_DLPFC_Visium/h5ad \\
        --annotation-key ground_truth \\
        --directions 151675:151676 151676:151675 \\
        --seed 2026 \\
        --overwrite

    # Mask-complete: split single slice, mask half, reconstruct
    PYTHONPATH=/maiziezhou_lab2/yiru/FEAST/src python run.py \\
        --mode mask_complete \\
        --data-dir /maiziezhou_lab2/yiru/Datasets/Processed/spatialLIBD_DLPFC_Visium/h5ad \\
        --annotation-key ground_truth \\
        --directions 151675 151676 \\
        --seed 2026 \\
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import ks_2samp, pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# FEAST import
# ---------------------------------------------------------------------------
FEAST_SRC = Path("/maiziezhou_lab2/yiru/FEAST/src")
if str(FEAST_SRC) not in sys.path:
    sys.path.insert(0, str(FEAST_SRC))

from FEAST.de_novo.conditional import fit_reference, simulate_from_reference
from FEAST import ReferenceFitConfig, SimulationConfig
from FEAST import SliceBlueprint
from FEAST import simulate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spatial_key(adata: ad.AnnData) -> str:
    """Return the obsm key FEAST will use for spatial coordinates."""
    if "spatial_3d" in adata.obsm:
        return "spatial_3d"
    if "spatial" in adata.obsm:
        return "spatial"
    raise KeyError("AnnData has neither obsm['spatial_3d'] nor obsm['spatial']")


def counts_matrix(adata: ad.AnnData):
    """Return the counts matrix, preferring layers['counts'] over .X."""
    return adata.layers["counts"] if "counts" in adata.layers else adata.X


def _sanitize_uns(uns):
    """Recursively convert uns values to h5py-compatible types."""
    if isinstance(uns, dict):
        return {str(k): _sanitize_uns(v) for k, v in uns.items()}
    if isinstance(uns, np.ndarray):
        if uns.dtype.kind in {"U", "S"}:
            return np.asarray(uns, dtype=str)
        if uns.dtype.kind == "O":
            return np.asarray([str(x) for x in uns.flat]).reshape(uns.shape)
        return uns
    if isinstance(uns, (list, tuple)):
        sanitized = [_sanitize_uns(v) for v in uns]
        if sanitized and any(not isinstance(v, (str, int, float, bool, type(None), np.generic)) for v in sanitized):
            return [str(v) for v in sanitized]
        return sanitized
    if isinstance(uns, (np.bool_, np.integer)):
        return int(uns)
    if isinstance(uns, (np.floating,)):
        return float(uns)
    if isinstance(uns, (str, int, float, bool, type(None))):
        return uns
    return str(uns)


def dense_counts(adata: ad.AnnData, *, dtype=np.float64) -> np.ndarray:
    """Extract a dense float64 array from an AnnData counts layer or X."""
    matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=dtype)


def git_output(repo: Path, args: list[str]) -> str | None:
    """Run a git command in *repo* and return stripped stdout, or None."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FEAST de novo conditional cross-slice transfer: "
            "fit on a source slice, generate a target slice from geometry + labels, "
            "and compare against held-out target counts."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("cross_slice", "mask_complete"),
        default="cross_slice",
        help="Experiment mode: cross_slice (source->target) or mask_complete (split, mask, reconstruct).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/maiziezhou_lab2/yiru/Datasets/Processed/spatialLIBD_DLPFC_Visium/h5ad"),
        help="Directory containing source/target .h5ad files.",
    )
    parser.add_argument(
        "--annotation-key",
        default="ground_truth",
        help="obs column used as the conditional label key. DLPFC uses 'ground_truth'; MERFISH uses 'class'.",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=["151675:151676", "151676:151675"],
        help="One or more source:target pairs. v1 expects one source per target.",
    )
    parser.add_argument(
        "--genes",
        default="",
        help="Comma-separated gene list. Empty string means auto-select.",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=0,
        help="Number of top genes to auto-select for Visium. 0 means use all common genes.",
    )
    parser.add_argument(
        "--assignment-randomness",
        type=float,
        default=0.3,
        help="Stochastic perturbation to break OT over-smoothing (0.0-1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducible simulation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/maiziezhou_lab2/yiru/Reproduce/Experiments/de_novo_conditional_transfer"),
        help="Output root directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing outputs for requested directions.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose FEAST generation output.",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.2,
        help="Hybrid assignment alpha passed to FEAST core when mask_complete uses simulate_single_slice.",
    )
    parser.add_argument(
        "--use-distributional-alteration",
        action="store_true",
        help="Enable FEAST core distributional alteration when mask_complete uses simulate_single_slice.",
    )
    parser.add_argument(
        "--ppf-method",
        choices=("interp", "exact"),
        default="interp",
        help="FEAST core PPF method when mask_complete uses simulate_single_slice.",
    )
    parser.add_argument(
        "--beta-n-jobs",
        type=int,
        default=1,
        help="FEAST core Beta mixture worker count when mask_complete uses simulate_single_slice.",
    )
    parser.add_argument(
        "--beta-early-stopping-patience",
        type=int,
        default=2,
        help="FEAST core Beta mixture early stopping patience.",
    )
    parser.add_argument(
        "--assignment-solver",
        choices=("scipy", "lapjv"),
        default="scipy",
        help="Hungarian assignment backend when mask_complete uses simulate_single_slice.",
    )
    parser.add_argument(
        "--assignment-n-jobs",
        type=int,
        default=1,
        help="Reserved FEAST core assignment worker count.",
    )
    parser.add_argument(
        "--assignment-blocks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use batched Hungarian assignment in FEAST core.",
    )
    parser.add_argument(
        "--assignment-block-size",
        type=int,
        default=None,
        help="Batched Hungarian block size; omit for FEAST default.",
    )
    parser.add_argument(
        "--assignment-block-multiplier",
        type=int,
        default=8,
        help="Candidate multiplier for batched Hungarian assignment.",
    )
    parser.add_argument(
        "--convert-n-jobs",
        type=int,
        default=1,
        help="FEAST core count-model conversion worker count.",
    )
    parser.add_argument(
        "--mask-axis",
        choices=("x", "y", "auto"),
        default="auto",
        help="Spatial axis to split on for mask_complete mode. 'auto' picks the axis with larger range.",
    )
    return parser


def feast_core_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "hybrid_alpha": float(args.hybrid_alpha),
        "use_distributional_alteration": bool(args.use_distributional_alteration),
        "ppf_method": str(args.ppf_method),
        "beta_n_jobs": int(args.beta_n_jobs),
        "beta_early_stopping_patience": int(args.beta_early_stopping_patience),
        "assignment_solver": str(args.assignment_solver),
        "assignment_n_jobs": int(args.assignment_n_jobs),
        "assignment_blocks": bool(args.assignment_blocks),
        "assignment_block_size": args.assignment_block_size,
        "assignment_block_multiplier": int(args.assignment_block_multiplier),
        "convert_n_jobs": int(args.convert_n_jobs),
    }


# ---------------------------------------------------------------------------
# Direction & file resolution
# ---------------------------------------------------------------------------


def resolve_h5ad(data_dir: Path, token: str) -> Path:
    """Resolve a direction token (stem or .h5ad filename) under *data_dir*."""
    token_path = Path(token)
    candidates = []
    if token_path.suffix == ".h5ad":
        candidates.append(data_dir / token_path.name)
    else:
        candidates.extend(
            [
                data_dir / f"{token}.h5ad",
                data_dir / f"DLPFC_{token}.h5ad",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not resolve h5ad token {token!r} under {data_dir}")


def parse_direction(direction: str) -> tuple[str, str]:
    """Split 'source:target' into (source, target)."""
    if direction.count(":") != 1:
        raise ValueError(
            f"Direction {direction!r} must contain exactly one ':' separator."
        )
    source, target = direction.split(":")
    if not source or not target:
        raise ValueError(f"Direction {direction!r} has empty source or target token.")
    if "," in source:
        raise NotImplementedError("multi-reference directions are deferred in v1")
    return source, target


def direction_name(source_id: str, target_id: str) -> str:
    return f"{source_id}_to_{target_id}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_slice(adata: ad.AnnData, annotation_key: str, path: Path) -> None:
    """Ensure the slice has the required annotation key and spatial coords."""
    if annotation_key not in adata.obs:
        raise KeyError(
            f"Slice {path} is missing obs[{annotation_key!r}]. "
            f"Available columns: {sorted(adata.obs.columns)}"
        )
    labels = adata.obs[annotation_key]
    if labels.isna().any():
        raise ValueError(
            f"Slice {path} has {labels.isna().sum()} NaN values in obs[{annotation_key!r}]."
        )
    s_key = _spatial_key(adata)
    coords = np.asarray(adata.obsm[s_key])
    if coords.ndim != 2 or coords.shape[1] not in {2, 3}:
        raise ValueError(
            f"Slice {path} spatial coords have shape {coords.shape}; expected (N, 2) or (N, 3)."
        )


def validate_coordinate_compatibility(source: ad.AnnData, target: ad.AnnData) -> None:
    s_key = _spatial_key(source)
    t_key = _spatial_key(target)
    s_dim = np.asarray(source.obsm[s_key]).shape[1]
    t_dim = np.asarray(target.obsm[t_key]).shape[1]
    if s_dim != t_dim:
        raise ValueError(
            f"Coordinate dimensionality mismatch: source={s_dim}D ({s_key}), "
            f"target={t_dim}D ({t_key})."
        )


# ---------------------------------------------------------------------------
# Gene selection
# ---------------------------------------------------------------------------


def detect_dataset_type(adatas: list[ad.AnnData]) -> str:
    max_n_vars = max(adata.n_vars for adata in adatas)
    if max_n_vars <= 2_000:
        return "merfish"
    return "visium"


def _common_genes(adatas: list[ad.AnnData]) -> list[str]:
    common = set(map(str, adatas[0].var_names))
    for adata in adatas[1:]:
        common &= set(map(str, adata.var_names))
    return sorted(common)


def _aggregate_stats(adatas: list[ad.AnnData], genes: list[str]) -> dict:
    """Compute per-gene aggregate mean, var, zero_prop across all observations."""
    total_n_obs = 0
    total_sum = {}
    total_sum_sq = {}
    total_nonzero = {}

    for adata in adatas:
        subset = adata[:, genes]
        matrix = counts_matrix(subset)
        n_obs = subset.n_obs
        total_n_obs += n_obs

        if sparse.issparse(matrix):
            sums = np.asarray(matrix.sum(axis=0)).ravel()
            sums_sq = np.asarray(matrix.power(2).sum(axis=0)).ravel()
            nz = np.asarray((matrix > 0).sum(axis=0)).ravel()
        else:
            sums = np.asarray(matrix.sum(axis=0)).ravel()
            sums_sq = np.asarray((matrix**2).sum(axis=0)).ravel()
            nz = np.asarray((matrix > 0).sum(axis=0)).ravel()

        for j, g in enumerate(genes):
            total_sum[g] = total_sum.get(g, 0.0) + float(sums[j])
            total_sum_sq[g] = total_sum_sq.get(g, 0.0) + float(sums_sq[j])
            total_nonzero[g] = total_nonzero.get(g, 0.0) + float(nz[j])

    stats = {}
    for g in genes:
        mean_g = total_sum[g] / total_n_obs
        var_g = (total_sum_sq[g] / total_n_obs) - mean_g**2
        var_g = max(var_g, 0.0)
        zero_prop_g = 1.0 - (total_nonzero[g] / total_n_obs)
        score_g = var_g * np.sqrt(max(mean_g, 1e-8)) * max(1.0 - zero_prop_g, 1e-8)
        stats[g] = {
            "mean": mean_g,
            "variance": var_g,
            "zero_prop": zero_prop_g,
            "score": score_g,
        }
    return stats


def select_genes(
    args: argparse.Namespace,
    unique_slices: dict[str, ad.AnnData],
) -> tuple[list[str], dict]:
    """Return (selected_genes, gene_panel_metadata)."""
    adatas = list(unique_slices.values())

    if args.genes:
        requested = [g.strip() for g in args.genes.split(",") if g.strip()]
        if not requested:
            raise ValueError("--genes was non-empty but produced no valid gene names.")
        for slice_id, adata in unique_slices.items():
            missing = [g for g in requested if g not in adata.var_names]
            if missing:
                raise ValueError(
                    f"Slice {slice_id!r} is missing {len(missing)} requested genes; "
                    f"first 10: {missing[:10]}"
                )
        return requested, {
            "selection_mode": "explicit",
            "n_selected_genes": len(requested),
            "user_genes": requested,
            "n_top_genes_arg": args.n_top_genes,
        }

    dataset_type = detect_dataset_type(adatas)
    common = _common_genes(adatas)
    if not common:
        raise ValueError("No common genes found across slices.")

    if dataset_type == "merfish":
        return common, {
            "selection_mode": "merfish_full_panel",
            "n_selected_genes": len(common),
            "n_top_genes_arg": args.n_top_genes,
            "max_n_vars": max(adata.n_vars for adata in adatas),
        }

    # Visium auto: scanpy highly_variable_genes with Seurat v3 flavor
    n_top = args.n_top_genes

    # Concatenate slices on common genes for unified HVG selection
    slices_common = [adata[:, common] for adata in adatas]
    combined = ad.concat(slices_common, join="inner", label="__slice__", keys=list(range(len(slices_common))))
    # Ensure raw counts are visible to scanpy
    if "counts" in combined.layers:
        combined.X = combined.layers["counts"].copy()

    if n_top > 0:
        sc.pp.highly_variable_genes(combined, flavor="seurat_v3", n_top_genes=n_top)
    else:
        sc.pp.highly_variable_genes(combined, flavor="seurat_v3")

    top_genes = list(combined.var_names[combined.var.highly_variable])
    sel_mode = "visium_seurat_v3" if n_top > 0 else "visium_seurat_v3_full"

    return top_genes, {
        "selection_mode": sel_mode,
        "n_selected_genes": len(top_genes),
        "n_common_genes": len(common),
        "n_top_genes_arg": args.n_top_genes,
        "max_n_vars": max(adata.n_vars for adata in adatas),
        "method": "scanpy.pp.highly_variable_genes",
        "flavor": "seurat_v3",
    }


# ---------------------------------------------------------------------------
# Overwrite protection
# ---------------------------------------------------------------------------


def prepare_output(args: argparse.Namespace, direction_names: list[str]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    collisions = []
    for filename in ("summary.csv", "metadata.json", "gene_panel.json"):
        path = args.output_dir / filename
        if path.exists():
            collisions.append(path)
    for dname in direction_names:
        ddir = args.output_dir / dname
        if ddir.exists() and any(ddir.iterdir()):
            collisions.append(ddir)
    if collisions and not args.overwrite:
        raise FileExistsError(
            "Output paths already exist. Re-run with --overwrite to replace them: "
            + ", ".join(str(p) for p in collisions[:10])
        )


# ---------------------------------------------------------------------------
# Moran's I
# ---------------------------------------------------------------------------


def neighbor_indices(coords: np.ndarray) -> list[np.ndarray]:
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    if n < 2:
        return []
    k = min(6, n)
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    raw = nbrs.kneighbors(coords, return_distance=False)
    out = []
    for i, row in enumerate(raw):
        neighbors = row[row != i]
        out.append(neighbors.astype(int, copy=False))
    return out


def moran_i(values: np.ndarray, neighbors: list[np.ndarray]) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    n = x.size
    if n < 3 or not neighbors:
        return float("nan")
    z = x - np.mean(x)
    denom = float(np.sum(z * z))
    if denom <= 0.0:
        return float("nan")
    numerator = 0.0
    weight_sum = 0.0
    for i, neigh in enumerate(neighbors):
        if len(neigh) == 0:
            continue
        numerator += float(np.sum(z[i] * z[neigh]))
        weight_sum += float(len(neigh))
    if weight_sum <= 0.0:
        return float("nan")
    return float((n / weight_sum) * (numerator / denom))


# ---------------------------------------------------------------------------
# Safe correlations
# ---------------------------------------------------------------------------


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    x = x[ok]
    y = y[ok]
    if np.var(x) <= 0.0 or np.var(y) <= 0.0:
        return float("nan")
    return float(pearsonr(x, y).statistic)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    x = x[ok]
    y = y[ok]
    if np.var(x) <= 0.0 or np.var(y) <= 0.0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


# ---------------------------------------------------------------------------
# Per-direction metrics
# ---------------------------------------------------------------------------


def compute_per_gene_metrics(
    generated: ad.AnnData,
    target_truth: ad.AnnData,
    common_genes: list[str],
    target_neighbors: list[np.ndarray],
) -> pd.DataFrame:
    gen_counts = dense_counts(generated)
    tgt_counts = dense_counts(target_truth)
    rows = []
    for j, g in enumerate(common_genes):
        gen = gen_counts[:, j]
        tgt = tgt_counts[:, j]
        rows.append(
            {
                "gene": g,
                "pearson": safe_pearson(gen, tgt),
                "spearman": safe_spearman(gen, tgt),
                "generated_mean": float(np.mean(gen)),
                "target_mean": float(np.mean(tgt)),
                "generated_variance": float(np.var(gen, ddof=0)),
                "target_variance": float(np.var(tgt, ddof=0)),
                "generated_zero_prop": float(np.mean(gen <= 0)),
                "target_zero_prop": float(np.mean(tgt <= 0)),
                "generated_moran_i": moran_i(gen, target_neighbors),
                "target_moran_i": moran_i(tgt, target_neighbors),
            }
        )
    return pd.DataFrame(rows)


def compute_panel_summary(
    per_gene: pd.DataFrame,
    direction: str,
    source_id: str,
    target_id: str,
    source_path: Path,
    target_path: Path,
    target_n_obs: int,
    n_input_genes: int,
    common_genes: list[str],
    generated: ad.AnnData,
    seed: int,
) -> dict:
    mean_corr = safe_pearson(
        np.log1p(per_gene["generated_mean"].to_numpy(float)),
        np.log1p(per_gene["target_mean"].to_numpy(float)),
    )
    var_corr = safe_pearson(
        per_gene["generated_variance"].to_numpy(float),
        per_gene["target_variance"].to_numpy(float),
    )
    moran_corr = safe_pearson(
        per_gene["generated_moran_i"].to_numpy(float),
        per_gene["target_moran_i"].to_numpy(float),
    )
    zero_ks = float(
        ks_2samp(
            per_gene["generated_zero_prop"].to_numpy(float),
            per_gene["target_zero_prop"].to_numpy(float),
        ).statistic
    )
    median_gene_pearson = float(np.nanmedian(per_gene["pearson"].to_numpy(float)))
    median_gene_spearman = float(np.nanmedian(per_gene["spearman"].to_numpy(float)))

    return {
        "direction": direction,
        "source": source_id,
        "target": target_id,
        "source_path": str(source_path),
        "target_path": str(target_path),
        "n_target_spots": int(target_n_obs),
        "n_input_genes": int(n_input_genes),
        "n_genes": int(len(common_genes)),
        "n_reference_genes": int(len(common_genes)),
        "mean_corr": mean_corr,
        "var_corr": var_corr,
        "moran_corr": moran_corr,
        "zero_ks": zero_ks,
        "median_gene_pearson": median_gene_pearson,
        "median_gene_spearman": median_gene_spearman,
        "decode_method": generated.uns.get("simulation_method"),
        "seed": int(seed),
    }


def compute_per_label_matrices(
    adata: ad.AnnData,
    annotation_key: str,
    common_genes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = adata.obs[annotation_key].astype(str)
    unique_labels = sorted(labels.unique())
    matrix = dense_counts(adata)
    gene_to_idx = {g: i for i, g in enumerate(common_genes)}

    mean_rows = []
    zero_rows = []
    for label in unique_labels:
        mask = (labels == label).to_numpy()
        if not mask.any():
            mean_rows.append({g: np.nan for g in common_genes})
            zero_rows.append({g: np.nan for g in common_genes})
            continue
        label_matrix = matrix[mask, :]
        mean_row = {"label": label}
        zero_row = {"label": label}
        for g in common_genes:
            j = gene_to_idx[g]
            mean_row[g] = float(np.mean(label_matrix[:, j]))
            zero_row[g] = float(np.mean(label_matrix[:, j] <= 0))
        mean_rows.append(mean_row)
        zero_rows.append(zero_row)

    mean_df = pd.DataFrame(mean_rows).set_index("label")
    zero_df = pd.DataFrame(zero_rows).set_index("label")
    mean_df.index.name = None
    zero_df.index.name = None
    return mean_df, zero_df


def compute_moran_metrics(
    per_gene: pd.DataFrame,
) -> pd.DataFrame:
    df = per_gene[["gene", "generated_moran_i", "target_moran_i"]].copy()
    df["moran_delta"] = df["generated_moran_i"] - df["target_moran_i"]
    return df


# ---------------------------------------------------------------------------
# Main per-direction pipeline
# ---------------------------------------------------------------------------


def run_direction(
    args: argparse.Namespace,
    source_id: str,
    target_id: str,
    source_path: Path,
    target_path: Path,
    selected_genes: list[str],
) -> dict:
    """Run a single cross_slice direction and return its panel_summary dict."""
    dname = direction_name(source_id, target_id)
    print(f"--- {dname} ---")

    # Load
    source = ad.read_h5ad(source_path)
    target = ad.read_h5ad(target_path)

    validate_slice(source, args.annotation_key, source_path)
    validate_slice(target, args.annotation_key, target_path)
    validate_coordinate_compatibility(source, target)

    # Ensure 'spatial' key exists (simulator's OT transport hardcodes obsm['spatial'])
    for adata_slice in (source, target):
        s_key = _spatial_key(adata_slice)
        if "spatial" not in adata_slice.obsm:
            adata_slice.obsm["spatial"] = np.asarray(adata_slice.obsm[s_key], dtype=float)

    # Subset to common labels so every target label exists in the source
    src_labels = set(source.obs[args.annotation_key].astype(str).unique())
    tgt_labels = set(target.obs[args.annotation_key].astype(str).unique())
    common_labels = src_labels & tgt_labels
    if common_labels != tgt_labels or common_labels != src_labels:
        dropped_src = src_labels - tgt_labels
        dropped_tgt = tgt_labels - src_labels
        msg_parts = []
        if dropped_tgt:
            n_drop = int((target.obs[args.annotation_key].astype(str).isin(dropped_tgt)).sum())
            msg_parts.append(f"{n_drop} target spots ({sorted(dropped_tgt)}) not in source")
        if dropped_src:
            n_drop = int((source.obs[args.annotation_key].astype(str).isin(dropped_src)).sum())
            msg_parts.append(f"{n_drop} source spots ({sorted(dropped_src)}) not in target")
        print(f"  Subsetting to {len(common_labels)} common labels: " + "; ".join(msg_parts))
        src_mask = source.obs[args.annotation_key].astype(str).isin(common_labels)
        tgt_mask = target.obs[args.annotation_key].astype(str).isin(common_labels)
        source = source[src_mask].copy()
        target = target[tgt_mask].copy()

    # Subset both source and target to common selected genes
    common_genes = [g for g in selected_genes if g in target.var_names]
    if not common_genes:
        raise RuntimeError(f"No common genes between source and target for {dname}")

    source = source[:, common_genes].copy()
    source.uns.setdefault("reference_name", source_id)

    target_truth = target[:, common_genes].copy()

    # Fit reference model on source, then generate via de_novo conditional pipeline
    print(
        f"  Generating {target_id} from {len(common_genes)} genes "
        f"({target.n_obs} spots, assignment_randomness={args.assignment_randomness})...")
    t0 = time.time()

    fit_cfg = ReferenceFitConfig(min_gene_spots=1, min_gene_mean=0.0, max_gene_zero_prop=1.0)
    t_fit0 = time.time()
    model = fit_reference(source, label_key=args.annotation_key, config=fit_cfg)
    fit_sec = time.time() - t_fit0

    # Subset to genes retained by fit_reference QC
    model_genes = list(model.gene_names)
    n_before = len(common_genes)
    common_genes = [g for g in common_genes if g in model_genes]
    source = source[:, common_genes].copy()
    target_truth = target_truth[:, common_genes].copy()
    print(f"    fit_reference QC retained {len(common_genes)}/{n_before} genes "
          f"({fit_sec:.0f}s)")

    target_coords = np.asarray(target.obsm["spatial"], dtype=float)
    target_labels = target.obs[args.annotation_key].astype(str).values
    blueprint = SliceBlueprint(
        coordinates=target_coords,
        domain_map=target_labels,
    )

    gen_cfg = SimulationConfig(assignment_randomness=args.assignment_randomness, verbose=args.verbose)
    generated = simulate_from_reference(
        model, blueprint,
        config=gen_cfg,
        random_seed=args.seed,
    )
    gen_sec = time.time() - t0

    # Copy target labels to generated AnnData (de_novo pipeline creates fresh obs)
    generated.obs[args.annotation_key] = target.obs[args.annotation_key].values

    # Verify generated integrity
    assert generated.shape == (target.n_obs, len(common_genes)), (
        f"Generated shape {generated.shape} != ({target.n_obs}, {len(common_genes)})"
    )
    assert list(generated.var_names.astype(str)) == common_genes, "Gene name mismatch"
    if "counts" not in generated.layers:
        generated.layers["counts"] = np.asarray(generated.X, dtype=np.int32)
    else:
        generated.layers["counts"] = np.asarray(generated.layers["counts"], dtype=np.int32)
    tgt_s_key = _spatial_key(target)
    if "spatial" not in generated.obsm and "spatial_3d" not in generated.obsm:
        generated.obsm[tgt_s_key] = np.asarray(target.obsm[tgt_s_key], dtype=float).copy()

    # Experiment metadata
    generated.uns.setdefault("experiment", {})
    generated.uns["experiment"] = {
        "script": str(Path(__file__).resolve()),
        "seed": args.seed,
        "direction": dname,
        "source": source_id,
        "target": target_id,
        "pipeline": "de_novo_conditional",
        "assignment_randomness": args.assignment_randomness,
    }

    # Output directory
    out_dir = args.output_dir / dname
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metrics
    print("  Computing metrics...")
    target_coords = np.asarray(target.obsm[tgt_s_key], dtype=float)
    target_nbrs = neighbor_indices(target_coords)

    per_gene = compute_per_gene_metrics(
        generated, target_truth, common_genes, target_nbrs
    )
    panel_summary = compute_panel_summary(
        per_gene,
        dname,
        source_id,
        target_id,
        source_path,
        target_path,
        target.n_obs,
        len(selected_genes),
        common_genes,
        generated,
        args.seed,
    )

    gen_mean_df, gen_zero_df = compute_per_label_matrices(
        generated, args.annotation_key, common_genes
    )
    tgt_mean_df, tgt_zero_df = compute_per_label_matrices(
        target_truth, args.annotation_key, common_genes
    )
    moran_df = compute_moran_metrics(per_gene)

    # Save outputs
    print("  Saving outputs...")
    generated.uns = _sanitize_uns(dict(generated.uns))
    generated.write_h5ad(out_dir / "generated.h5ad")
    per_gene.to_csv(out_dir / "per_gene_metrics.csv", index=False)
    with open(out_dir / "panel_summary.json", "w") as fh:
        json.dump(panel_summary, fh, indent=2, default=str)

    filtered_genes_json = {
        "direction": dname,
        "n_input_genes": len(selected_genes),
        "n_simulated_genes": len(common_genes),
        "simulated_genes": common_genes,
    }
    with open(out_dir / "filtered_genes.json", "w") as fh:
        json.dump(filtered_genes_json, fh, indent=2)

    gen_mean_df.to_csv(out_dir / "layer_mean_matrix_generated.csv")
    tgt_mean_df.to_csv(out_dir / "layer_mean_matrix_target.csv")
    gen_zero_df.to_csv(out_dir / "layer_zero_matrix_generated.csv")
    tgt_zero_df.to_csv(out_dir / "layer_zero_matrix_target.csv")
    moran_df.to_csv(out_dir / "moran_metrics.csv", index=False)

    panel_summary["_gen_seconds"] = gen_sec
    print(f"  Done. (gen {gen_sec:.1f}s)")
    return panel_summary


# ---------------------------------------------------------------------------
# Mask-complete pipeline
# ---------------------------------------------------------------------------


def run_mask_complete(
    args: argparse.Namespace,
    slice_id: str,
    slice_path: Path,
    selected_genes: list[str],
) -> dict:
    """Split a single slice by spatial median, mask half, reconstruct, compare."""
    dname = f"mask_{slice_id}"
    print(f"--- {dname} ---")

    adata = ad.read_h5ad(slice_path)
    validate_slice(adata, args.annotation_key, slice_path)

    s_key = _spatial_key(adata)
    coords = np.asarray(adata.obsm[s_key], dtype=float)
    n_spots = adata.n_obs
    all_labels = set(adata.obs[args.annotation_key].astype(str).unique())

    # Find a split axis that preserves label coverage in both halves
    if args.mask_axis == "auto":
        candidate_axes = [0, 1] if coords.shape[1] >= 2 else [0]
        # Prefer the axis with larger range
        ranges = coords.max(axis=0) - coords.min(axis=0)
        candidate_axes.sort(key=lambda a: -ranges[a])
    elif args.mask_axis == "x":
        candidate_axes = [0]
    else:
        candidate_axes = [1]

    best_axis = None
    best_median = None
    best_ref_mask = None
    best_tgt_mask = None
    best_missing = None

    for axis in candidate_axes:
        median = float(np.median(coords[:, axis]))
        ref_mask = coords[:, axis] <= median
        tgt_mask = coords[:, axis] > median
        ref_labels = set(adata.obs[args.annotation_key][ref_mask].astype(str).unique())
        tgt_labels = set(adata.obs[args.annotation_key][tgt_mask].astype(str).unique())
        missing_ref = tgt_labels - ref_labels
        missing_tgt = ref_labels - tgt_labels
        if not missing_ref and not missing_tgt:
            best_axis = axis
            best_median = median
            best_ref_mask = ref_mask
            best_tgt_mask = tgt_mask
            best_missing = None
            break
        if best_axis is None or len(missing_ref) < len(best_missing or []):
            best_axis = axis
            best_median = median
            best_ref_mask = ref_mask
            best_tgt_mask = tgt_mask
            best_missing = missing_ref

    if best_missing:
        common = all_labels - best_missing
        if len(common) < 2:
            raise RuntimeError(
                f"All splits on slice {slice_id} leave fewer than 2 common labels. "
                f"Cannot proceed with mask_complete."
            )
        print(
            f"  No split preserves all labels. Subsetting to {len(common)}/{len(all_labels)} "
            f"labels (dropping from reference: {sorted(best_missing)})."
        )
        keep_mask = adata.obs[args.annotation_key].astype(str).isin(common)
        n_before = adata.n_obs
        adata = adata[keep_mask].copy()
        coords = np.asarray(adata.obsm[s_key], dtype=float)
        print(f"  Subset slice from {n_before} to {adata.n_obs} spots.")

        # Recompute split on subset data
        for axis in candidate_axes:
            median = float(np.median(coords[:, axis]))
            ref_mask = coords[:, axis] <= median
            tgt_mask = coords[:, axis] > median
            if ref_mask.sum() >= 10 and tgt_mask.sum() >= 10:
                best_axis = axis
                best_median = median
                best_ref_mask = ref_mask
                best_tgt_mask = tgt_mask
                best_missing = None
                break

    axis = best_axis
    median = best_median
    ref_mask = best_ref_mask
    tgt_mask = best_tgt_mask

    n_ref = int(ref_mask.sum())
    n_tgt = int(tgt_mask.sum())
    if n_ref < 10 or n_tgt < 10:
        raise RuntimeError(
            f"Mask split produced {n_ref} ref / {n_tgt} tgt spots for {slice_id}; "
            f"need at least 10 each."
        )

    ref_adata = adata[ref_mask, :][:, selected_genes].copy()
    ref_adata.uns.setdefault("reference_name", f"{slice_id}_ref")
    ref_adata.obsm[s_key] = coords[ref_mask].copy()
    if "spatial" not in ref_adata.obsm:
        ref_adata.obsm["spatial"] = ref_adata.obsm[s_key].copy()

    tgt_adata = adata[tgt_mask, :].copy()
    tgt_adata.obsm[s_key] = coords[tgt_mask].copy()
    if "spatial" not in tgt_adata.obsm:
        tgt_adata.obsm["spatial"] = tgt_adata.obsm[s_key].copy()

    # Ensure label coverage: subset to labels present in both halves
    ref_labels = set(ref_adata.obs[args.annotation_key].astype(str).unique())
    tgt_labels = set(tgt_adata.obs[args.annotation_key].astype(str).unique())
    common_lbl = ref_labels & tgt_labels
    if common_lbl != ref_labels or common_lbl != tgt_labels:
        dropped_ref = ref_labels - tgt_labels
        dropped_tgt = tgt_labels - ref_labels
        msg_parts = []
        if dropped_tgt:
            n = int((tgt_adata.obs[args.annotation_key].astype(str).isin(dropped_tgt)).sum())
            msg_parts.append(f"{n} target spots ({sorted(dropped_tgt)}) not in reference")
        if dropped_ref:
            n = int((ref_adata.obs[args.annotation_key].astype(str).isin(dropped_ref)).sum())
            msg_parts.append(f"{n} reference spots ({sorted(dropped_ref)}) not in target")
        print(f"  Subsetting to {len(common_lbl)} common labels: " + "; ".join(msg_parts))
        ref_mask2 = ref_adata.obs[args.annotation_key].astype(str).isin(common_lbl)
        tgt_mask2 = tgt_adata.obs[args.annotation_key].astype(str).isin(common_lbl)
        ref_adata = ref_adata[ref_mask2].copy()
        tgt_adata = tgt_adata[tgt_mask2].copy()
        n_ref = ref_adata.n_obs
        n_tgt = tgt_adata.n_obs

    print(
        f"  Split {slice_id} along axis={axis} (median={median:.1f}): "
        f"{n_ref} ref spots, {n_tgt} target spots"
    )

    # Subset both halves to common selected genes
    common_genes = [g for g in selected_genes if g in tgt_adata.var_names]
    if not common_genes:
        raise RuntimeError(f"No common genes between reference and target halves for {dname}")

    if set(common_genes) != set(ref_adata.var_names):
        ref_adata = ref_adata[:, common_genes].copy()

    target_truth = tgt_adata[:, common_genes].copy()

    # Simulate on masked half using core simulator with hungarian × ot_spatial
    print(
        f"  Generating mask region from {len(common_genes)} genes "
        f"({n_tgt} spots)..."
    )
    t0 = time.time()
    generated = simulate(
        adata=ref_adata,
        annotation_key=args.annotation_key,
        parameter_mode='hungarian',
        spatial_mode='ot_spatial',
        target_adata=tgt_adata[:, common_genes].copy(),
        assignment_method='hybrid',
        seed=args.seed,
        verbose=args.verbose,
        **feast_core_options(args),
    )
    gen_sec = time.time() - t0
    fit_sec = 0.0  # No separate fit step in unified pipeline

    # Verify
    assert generated.shape == (n_tgt, len(common_genes)), (
        f"Generated shape {generated.shape} != ({n_tgt}, {len(common_genes)})"
    )
    assert list(generated.var_names.astype(str)) == common_genes, "Gene name mismatch"
    if "counts" not in generated.layers:
        generated.layers["counts"] = np.asarray(generated.X, dtype=np.int32)
    else:
        generated.layers["counts"] = np.asarray(generated.layers["counts"], dtype=np.int32)
    if "spatial" not in generated.obsm and "spatial_3d" not in generated.obsm:
        generated.obsm[s_key] = coords[tgt_mask].copy()

    # Metadata
    generated.uns.setdefault("experiment", {})
    generated.uns["experiment"] = {
        "script": str(Path(__file__).resolve()),
        "seed": args.seed,
        "direction": dname,
        "slice_id": slice_id,
        "mode": "mask_complete",
        "mask_axis": int(axis),
        "mask_median": float(median),
        "n_ref_spots": int(n_ref),
        "n_tgt_spots": int(n_tgt),
        "pipeline": "de_novo_conditional",
        "assignment_randomness": args.assignment_randomness,
        "feast_core_options": feast_core_options(args),
    }

    # Output directory
    out_dir = args.output_dir / dname
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metrics
    print("  Computing metrics...")
    tgt_coords = np.asarray(tgt_adata.obsm[s_key], dtype=float)
    tgt_nbrs = neighbor_indices(tgt_coords)

    per_gene = compute_per_gene_metrics(
        generated, target_truth, common_genes, tgt_nbrs
    )
    panel_summary = compute_panel_summary(
        per_gene,
        dname,
        f"{slice_id}_ref",
        f"{slice_id}_masked",
        slice_path,
        slice_path,
        n_tgt,
        len(selected_genes),
        common_genes,
        generated,
        args.seed,
    )

    gen_mean_df, gen_zero_df = compute_per_label_matrices(
        generated, args.annotation_key, common_genes
    )
    tgt_mean_df, tgt_zero_df = compute_per_label_matrices(
        target_truth, args.annotation_key, common_genes
    )
    moran_df = compute_moran_metrics(per_gene)

    # Save
    print("  Saving outputs...")
    generated.uns = _sanitize_uns(dict(generated.uns))
    generated.write_h5ad(out_dir / "generated.h5ad")
    per_gene.to_csv(out_dir / "per_gene_metrics.csv", index=False)
    with open(out_dir / "panel_summary.json", "w") as fh:
        json.dump(panel_summary, fh, indent=2, default=str)

    filtered_genes_json = {
        "direction": dname,
        "mode": "mask_complete",
        "n_input_genes": len(selected_genes),
        "n_simulated_genes": len(common_genes),
        "simulated_genes": common_genes,
    }
    with open(out_dir / "filtered_genes.json", "w") as fh:
        json.dump(filtered_genes_json, fh, indent=2)

    gen_mean_df.to_csv(out_dir / "layer_mean_matrix_generated.csv")
    tgt_mean_df.to_csv(out_dir / "layer_mean_matrix_target.csv")
    gen_zero_df.to_csv(out_dir / "layer_zero_matrix_generated.csv")
    tgt_zero_df.to_csv(out_dir / "layer_zero_matrix_target.csv")
    moran_df.to_csv(out_dir / "moran_metrics.csv", index=False)

    panel_summary["_gen_seconds"] = gen_sec
    print(f"  Done. (gen {gen_sec:.1f}s)")
    return panel_summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()

    if not args.directions:
        raise ValueError("At least one --directions entry is required.")

    if args.mode == "cross_slice":
        direction_pairs = [parse_direction(d) for d in args.directions]
        direction_names_list = [
            direction_name(src, tgt) for src, tgt in direction_pairs
        ]
        # Unique slice IDs for gene selection
        unique_slice_ids = sorted(set(
            token for pair in direction_pairs for token in pair
        ))
    else:  # mask_complete
        slice_ids = [d.strip() for d in args.directions]
        for sid in slice_ids:
            if not sid:
                raise ValueError("Empty slice ID in --directions.")
        direction_names_list = [f"mask_{sid}" for sid in slice_ids]
        unique_slice_ids = sorted(set(slice_ids))

    # Resolve h5ad paths
    resolved: dict[str, Path] = {}
    for sid in unique_slice_ids:
        resolved[sid] = resolve_h5ad(args.data_dir, sid)

    # Start timer
    start_time = time.time()
    start_utc = datetime.now(timezone.utc).isoformat()

    # Load unique slices for gene selection
    unique_slices: dict[str, ad.AnnData] = {}
    for sid in unique_slice_ids:
        unique_slices[sid] = ad.read_h5ad(resolved[sid])
        validate_slice(unique_slices[sid], args.annotation_key, resolved[sid])

    # Gene selection
    selected_genes, gene_panel_metadata = select_genes(args, unique_slices)
    if not selected_genes:
        raise ValueError("Gene selection produced an empty panel.")
    print(
        f"Selected {len(selected_genes)} genes "
        f"(mode: {gene_panel_metadata.get('selection_mode')})"
    )

    # Prepare output
    prepare_output(args, direction_names_list)

    # Save gene panel
    gene_panel_json = {
        **gene_panel_metadata,
        "selected_genes": selected_genes,
    }
    with open(args.output_dir / "gene_panel.json", "w") as fh:
        json.dump(gene_panel_json, fh, indent=2)

    # Run — core simulator handles gene usage internally from the source AnnData
    summaries = []
    if args.mode == "cross_slice":
        for (src_id, tgt_id) in direction_pairs:
            panel = run_direction(
                args,
                src_id,
                tgt_id,
                resolved[src_id],
                resolved[tgt_id],
                selected_genes,
            )
            summaries.append(panel)
    else:  # mask_complete
        for sid in slice_ids:
            panel = run_mask_complete(
                args,
                sid,
                resolved[sid],
                selected_genes,
            )
            summaries.append(panel)

    # End timer
    end_utc = datetime.now(timezone.utc).isoformat()
    elapsed_seconds = time.time() - start_time

    # Summary CSV
    summary_rows = []
    for s in summaries:
        row = {k: v for k, v in s.items() if not k.startswith("_")}
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output_dir / "summary.csv", index=False)

    # Metadata
    feast_repo = Path("/maiziezhou_lab2/yiru/FEAST")
    feast_commit_sha = git_output(feast_repo, ["rev-parse", "HEAD"])
    feast_git_dirty = bool(git_output(feast_repo, ["status", "--porcelain"]))

    input_files = {}
    for sid in unique_slice_ids:
        input_files[sid] = str(resolved[sid])

    if args.mode == "cross_slice":
        direction_metadata = []
        for i, (src, tgt) in enumerate(direction_pairs):
            direction_metadata.append(
                {
                    "direction": direction_name(src, tgt),
                    "source": src,
                    "target": tgt,
                    "source_path": str(resolved[src]),
                    "target_path": str(resolved[tgt]),
                    "n_genes": summaries[i]["n_genes"],
                }
            )
    else:
        direction_metadata = []
        for i, sid in enumerate(slice_ids):
            direction_metadata.append(
                {
                    "direction": f"mask_{sid}",
                    "slice": sid,
                    "path": str(resolved[sid]),
                    "n_genes": summaries[i]["n_genes"],
                }
            )

    metadata = {
        "script": str(Path(__file__).resolve()),
        "argv": sys.argv,
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "mode": args.mode,
        "data_dir": str(args.data_dir),
        "annotation_key": args.annotation_key,
        "directions": args.directions,
        "genes_arg": args.genes,
        "n_top_genes": args.n_top_genes,
        "seed": args.seed,
        "simulator_config": {
            "pipeline": "simulate_single_slice",
            "parameter_mode": "hungarian",
            "spatial_mode": "ot_spatial",
            "assignment_method": "hybrid",
            "verbose": args.verbose,
            **feast_core_options(args),
        },
        "output_dir": str(args.output_dir),
        "overwrite": args.overwrite,
        "python": {
            "version": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "conda": {
            "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"),
            "CONDA_PREFIX": os.environ.get("CONDA_PREFIX"),
        },
        "feast": {
            "repo": str(feast_repo),
            "commit": feast_commit_sha,
            "dirty": feast_git_dirty,
        },
        "timestamps": {
            "start_utc": start_utc,
            "end_utc": end_utc,
            "elapsed_seconds": elapsed_seconds,
        },
        "input_files": input_files,
        "gene_panel": gene_panel_metadata,
        "direction_summaries": direction_metadata,
    }
    with open(args.output_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    print(f"\nAll {len(summaries)} direction(s) complete ({elapsed_seconds:.1f}s total).")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
