#!/usr/bin/env python3
"""Process raw DLPFC slices through the same HVG pipeline used for simulations,
producing "real_baseline" inputs for clustering methods.

This gives a real-data ceiling that the simulated baseline can be compared against.

Usage:
    conda run -p /maiziezhou_lab2/yiru/envs/feast-py311-conda \\
    python run_real_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc


def select_hvgs(adata: ad.AnnData, n_top_genes: int = 3000, flavor: str = "seurat_v3") -> ad.AnnData:
    """Replica of run_hvg.py:select_hvgs — HVG selection + normalize + log1p."""
    import numpy as np
    import scipy.sparse

    adata = adata.copy()

    # Raw DLPFC data is sparse CSR; convert to dense for scanpy compat
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.toarray().astype(np.float32)

    if adata.n_vars <= n_top_genes:
        print(f"    {adata.n_vars} genes <= {n_top_genes} HVG target, keeping all")
        adata.var["highly_variable"] = True
    else:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor)
        adata = adata[:, adata.var.highly_variable].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


def main():
    data_dir = Path("data")
    hvg_dir = Path("outputs/hvg_inputs")
    sim_dir = Path("outputs/simulations")
    slice_ids = ["151508", "151670", "151676"]

    hvg_rows = []
    sim_rows = []

    for slice_id in slice_ids:
        raw_path = data_dir / f"{slice_id}.h5ad"
        if not raw_path.exists():
            print(f"SKIP {slice_id}: raw data not found at {raw_path}")
            continue

        print(f"[{slice_id}] Loading raw data...", end=" ", flush=True)
        adata = ad.read_h5ad(str(raw_path))
        print(f"{adata.n_obs} spots x {adata.n_vars} genes")

        sim_id = "real_baseline"
        hvg_out = hvg_dir / slice_id / f"{sim_id}.h5ad"

        if hvg_out.exists():
            print(f"  SKIP HVG — {hvg_out} exists")
        else:
            print(f"  Running HVG selection (top 3000, seurat_v3)...", end=" ", flush=True)
            adata_hvg = select_hvgs(adata, n_top_genes=3000, flavor="seurat_v3")
            hvg_out.parent.mkdir(parents=True, exist_ok=True)
            adata_hvg.write_h5ad(str(hvg_out), compression="gzip")
            print(f"saved {adata_hvg.n_obs} spots x {adata_hvg.n_vars} genes")

        hvg_rows.append({
            "slice_id": slice_id,
            "simulation_id": sim_id,
            "hvg_file": str(hvg_out),
            "n_hvgs": 3000,
            "status": "ok",
        })

        sim_rows.append({
            "slice_id": slice_id,
            "simulation_id": sim_id,
            "alteration_type": "real_baseline",
            "fold_change": None,
            "file_path": str(raw_path),
            "n_spots": adata.n_obs,
            "n_genes": adata.n_vars,
            "elapsed_seconds": None,
            "status": "ok",
        })

    # Write HVG manifest
    hvg_manifest_path = hvg_dir / "real_baseline_hvg_manifest.csv"
    pd.DataFrame(hvg_rows).to_csv(hvg_manifest_path, index=False)
    print(f"\nHVG manifest: {hvg_manifest_path}")

    # Write simulation manifest (points to raw data for ground truth + spatial)
    sim_manifest_path = sim_dir / "real_baseline_manifest.csv"
    pd.DataFrame(sim_rows).to_csv(sim_manifest_path, index=False)
    print(f"Sim manifest: {sim_manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
