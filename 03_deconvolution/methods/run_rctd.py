#!/usr/bin/env python3
"""Python wrapper for RCTD deconvolution via spacexr R package.

Prepares RCTD-compatible count, coordinate, and reference files from the
simulated low-resolution data and original MERFISH reference, calls the R
backend, and reformats the output to the standard proportions CSV format.

Usage:
    conda run -p /maiziezhou_lab2/yiru/envs/feast-py311-conda \\
    python scripts/methods/run_rctd.py \\
      --input outputs/simulations/007/resolution_0.1.h5ad \\
      --reference data/007.h5ad \\
      --cell-type-key cell_type \\
      --rscript /maiziezhou_lab2/yiru/envs/rctd_bioc/bin/Rscript \\
      --output outputs/rctd/007/resolution_0.1_proportions.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def prepare_rctd_inputs(
    spatial_adata: sc.AnnData,
    reference_adata: sc.AnnData,
    cell_type_key: str,
    work_dir: Path,
) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(spatial_adata.X, "toarray"):
        counts = spatial_adata.X.toarray()
    else:
        counts = spatial_adata.X
    counts = np.round(counts).astype(int)

    count_df = pd.DataFrame(
        counts.T,
        index=spatial_adata.var_names,
        columns=[f"spot_{i}" for i in range(spatial_adata.n_obs)],
    )
    counts_path = work_dir / "rctd_counts.csv"
    count_df.to_csv(counts_path)

    coords = spatial_adata.obsm["spatial"]
    n_umi = counts.sum(axis=0) if counts.shape[0] == spatial_adata.n_obs else counts.sum(axis=1)
    if counts.shape[0] == spatial_adata.n_obs:
        n_umi = counts.sum(axis=1)
    else:
        n_umi = counts.sum(axis=0)

    n_umi_vals = np.asarray(counts.sum(axis=1)).ravel()
    coords_df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "nUMI": n_umi_vals,
    }, index=[f"spot_{i}" for i in range(spatial_adata.n_obs)])
    coords_path = work_dir / "rctd_coords.csv"
    coords_df.to_csv(coords_path)

    shared_genes = reference_adata.var_names.intersection(
        spatial_adata.var_names)
    if len(shared_genes) == 0:
        raise RuntimeError("Zero shared genes between reference and spatial data")

    ref_sub = reference_adata[:, shared_genes].copy()

    # Filter rare cell types (< 50 cells) — spacexr auto-removes cells with
    # nUMI < 100, so we need a margin above the internal 25-cell minimum
    ct_counts = ref_sub.obs[cell_type_key].value_counts()
    keep_cts = ct_counts[ct_counts >= 50].index
    keep_mask = ref_sub.obs[cell_type_key].isin(keep_cts)
    ref_sub = ref_sub[keep_mask].copy()
    if ref_sub.n_obs == 0:
        raise RuntimeError("No cells remain after filtering for >= 25 cells per type")
    print(f"  Filtered to {ref_sub.n_obs} cells with >= 25 per type "
          f"({len(keep_cts)} cell types retained)")

    if hasattr(ref_sub.X, "toarray"):
        ref_X = ref_sub.X.toarray()
    else:
        ref_X = np.asarray(ref_sub.X)
    ref_X_int = np.round(ref_X).astype(int)

    # Write full single-cell reference (genes x cells) for SummarizedExperiment
    ref_df = pd.DataFrame(
        ref_X_int.T,
        index=shared_genes,
        columns=[f"ref_cell_{i}" for i in range(ref_sub.n_obs)],
    )
    ref_path = work_dir / "rctd_reference.csv"
    ref_df.to_csv(ref_path)

    # Write cell type labels per reference cell
    # spacexr::checkCellTypes() rejects "/" in cell type names, so sanitize
    # before passing to R. We keep a reverse mapping to restore original
    # names in the output CSV for compatibility with benchmark column matching.
    ct_series = ref_sub.obs[cell_type_key].astype(str)
    ct_sanitized = ct_series.str.replace("/", "_", regex=False)
    ct_to_original = dict(zip(ct_sanitized, ct_series))
    ct_sanitized.index = ref_df.columns
    ct_path = work_dir / "rctd_cell_types.csv"
    ct_sanitized.to_csv(ct_path, header=["cell_type"])

    n_cells = ref_sub.n_obs
    n_ct = ref_sub.obs[cell_type_key].nunique()

    return {
        "counts": str(counts_path),
        "coords": str(coords_path),
        "reference": str(ref_path),
        "cell_types": str(ct_path),
        "n_shared_genes": len(shared_genes),
        "n_ref_cells": n_cells,
        "n_cell_types": n_ct,
        "ct_sanitize_map": ct_to_original,
    }


def run_rctd_r_backend(
    inputs: dict,
    rscript_path: str,
    output_csv: Path,
    cache_dir: str = "/tmp/rctd_bioc_cache",
) -> int:
    backend_path = Path(__file__).parent / "run_rctd_backend.R"

    cmd = [
        rscript_path,
        str(backend_path),
        "--counts", inputs["counts"],
        "--coords", inputs["coords"],
        "--reference", inputs["reference"],
        "--cell-types", inputs["cell_types"],
        "--output", str(output_csv),
        "--cache-dir", cache_dir,
    ]

    env = dict(os.environ)
    env["RCTD_CACHE_DIR"] = cache_dir
    if "R_LIBS_USER" not in env:
        env["R_LIBS_USER"] = str(Path(cache_dir) / "R_libs")

    print(f"Running R backend: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env,
        timeout=7200,
    )

    log_dir = output_csv.parent.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = output_csv.stem
    parent_slice = output_csv.parent.name
    r_stdout = log_dir / f"rctd_{parent_slice}_{stem}_stdout.log"
    r_stderr = log_dir / f"rctd_{parent_slice}_{stem}_stderr.log"
    r_stdout.write_text(result.stdout)
    r_stderr.write_text(result.stderr)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"RCTD R backend failed (exit {result.returncode})", file=sys.stderr)
        print(result.stderr, file=sys.stderr)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to simulated low-resolution .h5ad")
    parser.add_argument("--reference", type=Path, required=True,
                        help="Path to single-cell resolution reference .h5ad")
    parser.add_argument("--cell-type-key", type=str, default="cell_type")
    parser.add_argument("--rscript", type=Path, required=True,
                        help="Path to Rscript binary")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path to output proportions CSV")
    parser.add_argument("--cache-dir", type=str,
                        default="/tmp/rctd_bioc_cache")
    parser.add_argument("--force", action="store_true",
                        help="Force rerun even if output exists")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        with open(args.output) as f:
            n_lines = sum(1 for _ in f)
        if n_lines >= 2:
            print(f"SKIP — output exists ({n_lines} lines): {args.output}")
            return 0
        else:
            print(f"Truncated output detected ({n_lines} lines), re-running: {args.output}")

    if not args.rscript.exists():
        print(f"ERROR: Rscript not found at {args.rscript}", file=sys.stderr)
        return 1

    print(f"Loading spatial data: {args.input}")
    spatial_adata = sc.read_h5ad(str(args.input))
    print(f"  {spatial_adata.n_obs} spots x {spatial_adata.n_vars} genes")

    print(f"Loading reference data: {args.reference}")
    reference_adata = sc.read_h5ad(str(args.reference))
    print(f"  {reference_adata.n_obs} cells x {reference_adata.n_vars} genes")

    if args.cell_type_key not in reference_adata.obs.columns:
        print(f"ERROR: '{args.cell_type_key}' not in reference.obs",
              file=sys.stderr)
        print(f"  Available: {list(reference_adata.obs.columns)}",
              file=sys.stderr)
        return 1

    if reference_adata.obs[args.cell_type_key].nunique() < 2:
        print("ERROR: RCTD requires at least 2 cell types", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="rctd_work_") as work_dir:
        work_path = Path(work_dir)
        print(f"Work dir: {work_dir}")

        try:
            inputs = prepare_rctd_inputs(
                spatial_adata, reference_adata,
                args.cell_type_key, work_path,
            )
        except Exception as e:
            print(f"ERROR preparing RCTD inputs: {e}", file=sys.stderr)
            return 1

        print(f"  {inputs['n_shared_genes']} shared genes, "
              f"{inputs['n_ref_cells']} ref cells, "
              f"{inputs['n_cell_types']} cell types")

        exit_code = run_rctd_r_backend(
            inputs, str(args.rscript), args.output, args.cache_dir,
        )

        if exit_code != 0:
            return exit_code

        # Restore original cell type names (with "/") in output for benchmark
        ct_map = inputs.get("ct_sanitize_map", {})
        if ct_map and args.output.exists():
            out_df = pd.read_csv(args.output, index_col=0)
            rename = {san: orig for san, orig in ct_map.items()
                      if san in out_df.columns}
            if rename:
                out_df.rename(columns=rename, inplace=True)
                out_df.to_csv(args.output)

    if args.output.exists() and args.output.stat().st_size > 0:
        print(f"Done: {args.output}")
        return 0
    else:
        print(f"ERROR: RCTD output not written: {args.output}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
