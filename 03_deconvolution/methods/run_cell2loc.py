#!/usr/bin/env python3
"""Run Cell2location deconvolution on FEAST-simulated low-resolution data.

Builds a reference signature from original single-cell-resolution MERFISH data
(via RegressionModel), then runs Cell2location on the simulated pseudo-bulk
spots to estimate cell-type proportions per spot.

Usage:
    conda run -p /maiziezhou_lab2/yiru/envs/cell2loc_env \\
    python methods/run_cell2loc.py \\
      --input outputs/05_deconvolution/007/resolution_0.1.h5ad \\
      --reference data/Zhuang-ABCA-1.007.h5ad \\
      --cell-type-key cell_type \\
      --output outputs/05_deconvolution/cell2location/007/resolution_0.1_proportions.csv \\
      --seed 2026
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

CELL_TYPE_FALLBACKS = ["cell_type", "CellType", "class", "annotation", "ground_truth"]


def _resolve_cell_type_key(adata_obj: sc.AnnData, requested: str) -> str:
    """Find a valid cell-type column in adata.obs, trying fallbacks."""
    if requested in adata_obj.obs.columns:
        return requested
    for key in CELL_TYPE_FALLBACKS:
        if key in adata_obj.obs.columns:
            print(f"Note: '{requested}' not found, using '{key}' as cell-type key")
            return key
    available = list(adata_obj.obs.columns)
    raise KeyError(
        f"No cell-type column found. Tried: {[requested] + CELL_TYPE_FALLBACKS}. "
        f"Available obs columns: {available}"
    )


def _check_gpu() -> bool:
    """Check if CUDA GPU is available via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def run_cell2location(
    spatial_adata: sc.AnnData,
    reference_adata: sc.AnnData,
    cell_type_key: str,
    seed: int,
    output_csv: Path,
    max_epochs: int = 25000,
    detection_alpha: float = 20.0,
    use_gpu: bool = False,
) -> dict:
    """Run the full Cell2location deconvolution pipeline.

    Steps:
      1. Build reference signatures from single-cell data via RegressionModel.
      2. Prepare spatial pseudo-bulk data.
      3. Run Cell2location to estimate cell-type proportions per spot.
      4. Export proportions as CSV.

    Returns a metadata dict with n_spots, n_cell_types, elapsed_seconds, etc.
    """
    import cell2location
    import scvi

    scvi.settings.seed = seed
    gpu_available = _check_gpu()
    # Pass accelerator to Lightning via scvi-tools train() kwargs.
    # NOTE: cell2location's RegressionModel uses PyroSviTrainMixin which takes
    # 'device' (singular), not 'devices' (plural).
    train_kwargs = {}
    if gpu_available:
        train_kwargs = {"accelerator": "gpu", "device": 1}
    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Build reference from single-cell data
    # ------------------------------------------------------------------
    print("Step 1/4: Building reference signatures from single-cell data ...")

    # Intersect genes: keep only genes present in BOTH reference and spatial
    shared_genes = reference_adata.var_names.intersection(spatial_adata.var_names)
    if len(shared_genes) == 0:
        raise RuntimeError(
            f"No shared genes between reference ({reference_adata.n_vars}) "
            f"and spatial ({spatial_adata.n_vars}) data."
        )
    print(f"  Shared genes: {len(shared_genes)}")

    ref = reference_adata[:, shared_genes].copy()
    spat = spatial_adata[:, shared_genes].copy()

    # Assign cell-type labels for RegressionModel
    ct_key = _resolve_cell_type_key(ref, cell_type_key)
    ref.obs["_cell_type"] = ref.obs[ct_key].astype(str)
    n_cell_types = ref.obs["_cell_type"].nunique()
    print(f"  Cell types: {n_cell_types} (key='{ct_key}')")

    # CRITICAL: RegressionModel expects RAW integer counts, NOT log-normalized.
    # Do NOT normalize or log-transform — GammaPoisson distribution requires
    # non-negative integer values.
    if hasattr(ref.X, "toarray"):
        ref.X = ref.X.toarray()
    # Ensure non-negative integer counts (clip negatives to 0, round to int)
    ref.X = np.maximum(np.round(ref.X).astype(np.int64), 0)

    # Add batch key if missing (required by scvi-tools setup_anndata)
    if "batch" not in ref.obs.columns:
        ref.obs["batch"] = "all"

    # Train RegressionModel to estimate per-cell-type expression signatures
    from cell2location.models import RegressionModel

    # scvi-tools >= 1.0 requires setup_anndata before model initialization
    RegressionModel.setup_anndata(ref, batch_key="batch", labels_key="_cell_type")
    mod_ref = RegressionModel(ref)
    mod_ref.train(
        max_epochs=250,
        batch_size=2500,
        train_size=1,
        lr=0.002,
        **train_kwargs,
    )

    # Export posterior
    ref = mod_ref.export_posterior(
        ref,
        sample_kwargs={"num_samples": 1000, "batch_size": 2500},
    )

    # Extract per-cell-type mean expression matrix
    if "means_per_cluster_mu_fg" not in ref.varm:
        raise RuntimeError(
            "RegressionModel.export_posterior did not produce "
            "'means_per_cluster_mu_fg' in varm."
        )

    factor_names = list(ref.uns["mod"]["factor_names"])
    # means_per_cluster_mu_fg columns are prefixed with "means_per_cluster_mu_fg_".
    # cell2location expects cell_state_df as (genes x cell_types):
    #   - index = gene names, columns = cell type names.
    # Keep original orientation, just rename columns to strip prefix.
    mdf = ref.varm["means_per_cluster_mu_fg"].iloc[:, :len(factor_names)].copy()
    mdf.columns = factor_names
    inf_aver = mdf
    print(f"  Reference signature: {inf_aver.shape[0]} genes x {inf_aver.shape[1]} cell types")

    # ------------------------------------------------------------------
    # Step 2: Prepare spatial data
    # ------------------------------------------------------------------
    print("Step 2/4: Preparing spatial pseudo-bulk data ...")
    # Cell2location expects raw integer counts — ensure integer type
    if hasattr(spat.X, "toarray"):
        spat.X = spat.X.toarray()
    spat.X = np.round(spat.X).astype(np.float32)

    n_genes_before = spat.n_vars
    sc.pp.filter_genes(spat, min_cells=1)
    if spat.n_vars < n_genes_before:
        print(f"  Filtered {n_genes_before - spat.n_vars} genes (min_cells=1), "
              f"{spat.n_vars} remaining")
        # Align reference signature to remaining spatial genes.
        # inf_aver: index=gene names, columns=cell type names
        shared_genes_after = spat.var_names.intersection(inf_aver.index)
        inf_aver = inf_aver.loc[shared_genes_after]
        spat = spat[:, shared_genes_after].copy()
        print(f"  After alignment: {spat.n_vars} genes in both")

    print(f"  Spatial: {spat.n_obs} spots x {spat.n_vars} genes")

    # Add batch key if missing (required by scvi-tools setup_anndata)
    if "batch" not in spat.obs.columns:
        spat.obs["batch"] = "all"

    # ------------------------------------------------------------------
    # Step 3: Run Cell2location
    # ------------------------------------------------------------------
    print(f"Step 3/4: Running Cell2location (max_epochs={max_epochs}) ...")
    from cell2location.models import Cell2location

    N_cells = float(spat.X.sum(axis=1).mean())
    print(f"  Mean total counts per spot: {N_cells:.1f}")

    Cell2location.setup_anndata(spat, batch_key="batch")
    mod_c2l = Cell2location(
        spat,
        cell_state_df=inf_aver,
        N_cells_per_location=N_cells,
        detection_alpha=detection_alpha,
    )

    mod_c2l.train(
        max_epochs=max_epochs,
        batch_size=None,
        train_size=1,
        **train_kwargs,
    )

    # ------------------------------------------------------------------
    # Step 4: Extract and export proportions
    # ------------------------------------------------------------------
    print("Step 4/4: Extracting cell-type proportions ...")
    spat = mod_c2l.export_posterior(
        spat,
        sample_kwargs={"num_samples": 1000, "batch_size": None},
    )

    # Extract proportions from posterior
    prop_key = None
    for candidate in [
        "means_cell_abundance_w_sf",
        "q05_cell_abundance_w_sf",
        "q50_cell_abundance_w_sf",
    ]:
        if candidate in spat.obsm:
            prop_key = candidate
            break

    if prop_key is None:
        available = list(spat.obsm.keys())
        raise RuntimeError(
            f"Cell2location did not produce abundance estimates. "
            f"Available obsm keys: {available}"
        )

    proportions = spat.obsm[prop_key]
    # Convert to numpy array — scvi/cell2location may store results as
    # pandas DataFrame, sparse matrix, or numpy array depending on version.
    if isinstance(proportions, pd.DataFrame):
        proportions = proportions.values
    if hasattr(proportions, "toarray"):
        proportions = proportions.toarray()

    # Normalize each row to sum to 1 (proportions)
    row_sums = proportions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid division by zero
    proportions = proportions / row_sums

    # Column names: cell type names from the reference signature
    cell_type_names = list(inf_aver.columns)

    # Build DataFrame and write CSV
    prop_df = pd.DataFrame(
        proportions,
        index=[f"spot_{i}" for i in range(proportions.shape[0])],
        columns=cell_type_names,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    prop_df.to_csv(output_csv)

    elapsed = time.time() - t_start
    print(f"  Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output: {output_csv}")

    return {
        "n_spots": spat.n_obs,
        "n_cell_types": len(cell_type_names),
        "n_shared_genes": len(shared_genes),
        "elapsed_seconds": round(elapsed, 2),
        "proportion_key_used": prop_key,
        "max_epochs": max_epochs,
        "detection_alpha": detection_alpha,
        "seed": seed,
        "use_gpu": use_gpu,
        "cell_type_key": ct_key,
        "status": "ok",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to simulated pseudo-bulk .h5ad file",
    )
    parser.add_argument(
        "--reference", type=Path, required=True,
        help="Path to original single-cell resolution .h5ad file (reference)",
    )
    parser.add_argument(
        "--cell-type-key", type=str, default="cell_type",
        help="Column in reference .obs with cell-type labels (default: cell_type)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Path for output proportions CSV",
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="Random seed (default: 2026)",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=25000,
        help="Cell2location training epochs (default: 25000)",
    )
    parser.add_argument(
        "--detection-alpha", type=float, default=20.0,
        help="Cell2location detection_alpha (default: 20)",
    )
    parser.add_argument(
        "--use-gpu", action="store_true",
        help="Enable GPU acceleration",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output",
    )
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        print(f"SKIP — output exists: {args.output}")
        return 0

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 1

    if not args.reference.exists():
        print(f"ERROR: reference file not found: {args.reference}", file=sys.stderr)
        return 1

    gpu_available = _check_gpu()
    use_gpu = args.use_gpu and gpu_available
    if args.use_gpu and not gpu_available:
        print("WARNING: --use-gpu requested but CUDA is not available. Falling back to CPU.")

    print(f"Cell2location deconvolution — {datetime.now(timezone.utc).isoformat()}")
    print(f"  Input:     {args.input}")
    print(f"  Reference: {args.reference}")
    print(f"  Output:    {args.output}")
    print(f"  Seed:      {args.seed}")
    print(f"  GPU:       {use_gpu}")
    print(f"  Max epochs: {args.max_epochs}")
    print()

    # Load data
    print("Loading data ...")
    spatial_adata = sc.read_h5ad(args.input)
    reference_adata = sc.read_h5ad(args.reference)
    spatial_adata.var_names_make_unique()
    reference_adata.var_names_make_unique()

    print(f"  Spatial:   {spatial_adata.n_obs} spots x {spatial_adata.n_vars} genes")
    print(f"  Reference: {reference_adata.n_obs} cells x {reference_adata.n_vars} genes")

    # Run deconvolution
    try:
        result = run_cell2location(
            spatial_adata=spatial_adata,
            reference_adata=reference_adata,
            cell_type_key=args.cell_type_key,
            seed=args.seed,
            output_csv=args.output,
            max_epochs=args.max_epochs,
            detection_alpha=args.detection_alpha,
            use_gpu=use_gpu,
        )
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        # Write a failure metadata file so the pipeline can continue
        meta_path = args.output.parent / f"{args.output.stem}_metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump({
                "status": "failed",
                "error": str(e),
                "input": str(args.input),
                "reference": str(args.reference),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)
        return 1

    # Write metadata
    meta_path = args.output.parent / f"{args.output.stem}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            **result,
            "input": str(args.input),
            "reference": str(args.reference),
            "output": str(args.output),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)

    print(f"\nOK — {result['n_spots']} spots, {result['n_cell_types']} cell types")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
