#!/usr/bin/env python
"""GraphST vertical integration wrapper for FEAST batch-effect benchmarking.

Standard GraphST multi-slice integration: concatenate slices with their original
spatial coordinates, let GraphST build the spatial graph internally (edges only
within each slice since slice coordinates are disjoint), then train.

Reference: https://deepst-tutorials.readthedocs.io/en/latest/Tutorial%205_Vertical%20Integration.html

Usage:
    /maiziezhou_lab2/yiru/envs/GraphST/bin/python methods/graphst_batch.py \\
      --reference alpha_0.00.h5ad --query alpha_0.50.h5ad \\
      --output-dir outputs/methods/GraphST/shift_only/alpha_0.50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

if "R_HOME" not in os.environ:
    os.environ["R_HOME"] = "/maiziezhou_lab2/yiru/envs/GraphST/lib/R"

warnings.filterwarnings("ignore")


def run_graphst_batch_correction(
    adata_ref: sc.AnnData,
    adata_query: sc.AnnData,
    n_hvgs: int = 2000,
    device: str = "cuda",
    epochs: int = 600,
    seed: int = 41,
) -> tuple[sc.AnnData, dict]:
    """Run GraphST vertical integration on concatenated reference + query.

    Standard approach: concatenate slices keeping original spatial coordinates,
    so GraphST's internal graph construction naturally separates slices.
    """
    import torch
    from GraphST import GraphST as graphst_module

    torch_device = torch.device(device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    batch_key = "batch"

    t0 = time.time()

    # Tag samples
    adata_ref_copy = adata_ref.copy()
    adata_query_copy = adata_query.copy()
    adata_ref_copy.obs[batch_key] = "ref"
    adata_query_copy.obs[batch_key] = "query"

    # Concatenate — spatial coords from different slices are disjoint,
    # so GraphST's internal adjacency naturally groups by slice.
    adata = sc.concat([adata_ref_copy, adata_query_copy], join="inner")
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")

    # HVG selection
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_hvgs,
        flavor="seurat_v3",
        batch_key=batch_key,
    )
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()

    # Standard GraphST preprocessing
    sc.pp.normalize_total(adata_hvg, target_sum=1e4)
    sc.pp.log1p(adata_hvg)
    adata_hvg.var["highly_variable"] = True
    sc.pp.scale(adata_hvg, zero_center=False, max_value=10)

    # GraphST builds adjacency internally from obsm['spatial']
    model = graphst_module.GraphST(
        adata_hvg,
        device=torch_device,
        epochs=epochs,
        dim_input=adata_hvg.n_vars,
        random_seed=seed,
        datatype="10X",
    )
    adata_hvg = model.train()

    # Standard output: emb_pca from PCA on the embedding, or emb directly
    if "emb_pca" in adata_hvg.obsm:
        adata_hvg.obsm["X_GraphST"] = adata_hvg.obsm["emb_pca"]
    elif "emb" in adata_hvg.obsm:
        adata_hvg.obsm["X_GraphST"] = adata_hvg.obsm["emb"]
    else:
        raise RuntimeError("GraphST training did not produce 'emb' or 'emb_pca' in obsm")

    elapsed = time.time() - t0
    meta = {
        "method": "GraphST",
        "n_hvgs": n_hvgs,
        "epochs": epochs,
        "device": str(device),
        "n_genes_input": adata.n_vars,
        "n_genes_hvg": adata_hvg.n_vars,
        "n_spots_total": adata_hvg.n_obs,
        "embedding_key": "X_GraphST",
        "elapsed_seconds": round(elapsed, 2),
    }

    return adata_hvg, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-hvgs", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    if not args.reference.exists():
        print(f"ERROR: reference not found: {args.reference}", file=sys.stderr)
        return 1
    if not args.query.exists():
        print(f"ERROR: query not found: {args.query}", file=sys.stderr)
        return 1

    sc.settings.verbosity = 0

    adata_ref = sc.read_h5ad(str(args.reference))
    adata_query = sc.read_h5ad(str(args.query))
    print(f"GraphST: ref={adata_ref.n_obs}x{adata_ref.n_vars}, "
          f"query={adata_query.n_obs}x{adata_query.n_vars}")

    try:
        adata_out, meta = run_graphst_batch_correction(
            adata_ref,
            adata_query,
            n_hvgs=args.n_hvgs,
            device=args.device,
            epochs=args.epochs,
            seed=args.seed,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        fail_file = args.output_dir / "FAILED.txt"
        if fail_file.exists():
            fail_file.unlink()

        np.savez(
            args.output_dir / "embeddings.npz",
            embedding=adata_out.obsm["X_GraphST"],
            batch_labels=adata_out.obs["batch"].values,
        )

        adata_out.write_h5ad(args.output_dir / "result.h5ad", compression="gzip")

        with open(args.output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  -> {adata_out.n_obs} spots, latent={adata_out.obsm['X_GraphST'].shape[1]}d "
              f"in {meta['elapsed_seconds']:.1f}s")
        return 0

    except Exception as e:
        import traceback
        fail_file = args.output_dir / "FAILED.txt"
        fail_file.parent.mkdir(parents=True, exist_ok=True)
        with open(fail_file, "w") as f:
            f.write(traceback.format_exc())
        print(f"  -> FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
