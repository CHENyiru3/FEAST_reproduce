#!/usr/bin/env python3
"""Select highly variable genes from simulated clustering data.

Reads each simulation .h5ad, selects HVGs, writes filtered copies for method input.
Never modifies source simulation files.

Usage:
    conda run -p /maiziezhou_lab2/yiru/envs/feast-py311-conda \\
    python scripts/run_hvg_clustering_refresh.py \\
      --simulation-manifest outputs/simulation_manifest.csv \\
      --output-dir outputs/hvg_inputs \\
      --n-top-genes 3000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def existing_hvg_count(output_path: Path, hvg_path: Path) -> int | None:
    """Return the existing HVG count without rewriting the output."""
    if hvg_path.exists():
        try:
            return len(pd.read_csv(hvg_path))
        except Exception:
            pass
    try:
        return ad.read_h5ad(str(output_path)).n_vars
    except Exception:
        return None


def select_hvgs(adata: ad.AnnData, n_top_genes: int = 3000, flavor: str = "seurat") -> ad.AnnData:
    """Select HVGs on count-like data, then normalize/log-transform for methods."""
    adata = adata.copy()

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-manifest", type=Path, required=True,
                        help="Path to simulation_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hvg_inputs"))
    parser.add_argument("--n-top-genes", type=int, default=3000)
    parser.add_argument("--flavor", type=str, default="seurat_v3")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.simulation_manifest.exists():
        print(f"ERROR: manifest not found: {args.simulation_manifest}", file=sys.stderr)
        return 1

    manifest = pd.read_csv(args.simulation_manifest)
    if "status" in manifest.columns:
        ok_rows = manifest[manifest["status"].isin(["ok", "skipped"])]
    else:
        ok_rows = manifest

    if len(ok_rows) == 0:
        print("ERROR: no usable simulations in manifest", file=sys.stderr)
        return 1

    hvg_rows = []
    for _, row in ok_rows.iterrows():
        slice_id = str(row["slice_id"])
        sim_id = str(row["simulation_id"])
        input_path = Path(row["file_path"])
        output_path = args.output_dir / slice_id / f"{sim_id}.h5ad"
        hvg_path = args.output_dir / slice_id / f"{sim_id}_hvg.csv"

        if output_path.exists() and not args.overwrite:
            print(f"SKIP {slice_id}/{sim_id} — exists")
            n_hvgs = existing_hvg_count(output_path, hvg_path)
            hvg_rows.append({"slice_id": slice_id, "simulation_id": sim_id,
                            "hvg_file": str(output_path), "n_hvgs": n_hvgs, "status": "ok"})
            continue

        if not input_path.exists():
            print(f"SKIP {slice_id}/{sim_id} — input missing: {input_path}")
            hvg_rows.append({"slice_id": slice_id, "simulation_id": sim_id,
                            "hvg_file": str(output_path), "n_hvgs": None, "status": "input_missing"})
            continue

        print(f"HVG  {slice_id}/{sim_id}")
        try:
            adata = ad.read_h5ad(str(input_path))
            adata_hvg = select_hvgs(adata, n_top_genes=args.n_top_genes, flavor=args.flavor)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            adata_hvg.write_h5ad(str(output_path), compression="gzip")

            hv_genes = adata_hvg.var_names.tolist()
            pd.DataFrame({"gene": hv_genes}).to_csv(hvg_path, index=False)

            hvg_rows.append({"slice_id": slice_id, "simulation_id": sim_id,
                            "hvg_file": str(output_path), "n_hvgs": len(hv_genes), "status": "ok"})
            print(f"  -> {len(hv_genes)} HVGs kept")
        except Exception as e:
            print(f"  -> FAILED: {e}", file=sys.stderr)
            hvg_rows.append({"slice_id": slice_id, "simulation_id": sim_id,
                            "hvg_file": str(output_path), "n_hvgs": None, "status": f"failed: {e}"})

    hvg_manifest_path = args.output_dir / "hvg_manifest.csv"
    pd.DataFrame(hvg_rows).to_csv(hvg_manifest_path, index=False)

    ok_count = sum(1 for r in hvg_rows if r["status"] == "ok")
    print(f"\nDone: {ok_count}/{len(hvg_rows)} ok")
    print(f"Manifest: {hvg_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
