#!/usr/bin/env python
"""scVI batch correction wrapper for FEAST batch-effect benchmarking.

Usage:
    /maiziezhou_lab2/yiru/miniconda3/envs/scvi-tools/bin/python methods/scvi_batch.py \\
      --reference alpha_0.00.h5ad --query alpha_0.50.h5ad \\
      --output-dir outputs/methods/scVI/shift_only/alpha_0.50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")


def run_scvi_batch_correction(
    adata_ref: sc.AnnData,
    adata_query: sc.AnnData,
    n_hvgs: int = 2000,
    n_latent: int = 10,
    n_layers: int = 1,
    n_hidden: int = 128,
    dropout_rate: float = 0.1,
    max_epochs: int = 400,
    seed: int = 42,
    batch_key: str = "batch",
) -> tuple[sc.AnnData, dict]:
    """Run scVI batch correction on concatenated reference + query.

    Uses scVI defaults: zinb likelihood, 1 hidden layer, 10-dim latent space.
    """
    import scvi

    t0 = time.time()

    # Concatenate reference and query with batch labels
    adata_ref_copy = adata_ref.copy()
    adata_query_copy = adata_query.copy()
    adata_ref_copy.obs[batch_key] = "ref"
    adata_query_copy.obs[batch_key] = "query"

    adata = sc.concat([adata_ref_copy, adata_query_copy], join="inner")
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")

    # Store raw counts (scVI requires integer counts)
    if hasattr(adata.X, "toarray"):
        adata.layers["counts"] = adata.X.copy()
    else:
        adata.layers["counts"] = adata.X.astype(np.float32).copy()

    # HVG selection
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_hvgs,
        flavor="seurat_v3",
        batch_key=batch_key,
        layer="counts",
    )
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()

    # Setup scVI
    scvi.model.SCVI.setup_anndata(
        adata_hvg,
        layer="counts",
        batch_key=batch_key,
    )

    model = scvi.model.SCVI(
        adata_hvg,
        n_latent=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
        dropout_rate=dropout_rate,
        gene_likelihood="zinb",
    )

    model.train(
        max_epochs=max_epochs,
        early_stopping=True,
        accelerator="auto",
        devices=1,
        plan_kwargs={"lr": 1e-3},
    )

    # Get latent representation
    adata_hvg.obsm["X_scVI"] = model.get_latent_representation()

    # Also get normalized expression (batch-corrected)
    adata_hvg.layers["scvi_normalized"] = model.get_normalized_expression(
        return_numpy=True,
    )

    elapsed = time.time() - t0
    model_history = {}
    for k, vs in model.history.items():
        try:
            model_history[k] = [float(v) if not (isinstance(v, str) or (isinstance(v, float) and (v != v))) else 0.0 for v in vs]
        except (ValueError, TypeError):
            model_history[k] = [str(v) for v in vs]

    meta = {
        "method": "scVI",
        "gene_likelihood": "zinb",
        "n_hvgs": n_hvgs,
        "n_latent": n_latent,
        "n_layers": n_layers,
        "n_hidden": n_hidden,
        "dropout_rate": dropout_rate,
        "max_epochs": max_epochs,
        "n_genes_input": adata.n_vars,
        "n_genes_hvg": adata_hvg.n_vars,
        "n_spots_total": adata_hvg.n_obs,
        "elapsed_seconds": round(elapsed, 2),
        "model_history": model_history,
    }

    return adata_hvg, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-hvgs", type=int, default=2000)
    parser.add_argument("--n-latent", type=int, default=10)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-hidden", type=int, default=128)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
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
    print(f"scVI: ref={adata_ref.n_obs}x{adata_ref.n_vars}, "
          f"query={adata_query.n_obs}x{adata_query.n_vars}")

    try:
        adata_out, meta = run_scvi_batch_correction(
            adata_ref,
            adata_query,
            n_hvgs=args.n_hvgs,
            n_latent=args.n_latent,
            n_layers=args.n_layers,
            n_hidden=args.n_hidden,
            dropout_rate=args.dropout_rate,
            max_epochs=args.max_epochs,
            seed=args.seed,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        fail_file = args.output_dir / "FAILED.txt"
        if fail_file.exists():
            fail_file.unlink()

        # Save embedding
        ref_mask = adata_out.obs["batch"] == "ref"
        query_mask = adata_out.obs["batch"] == "query"

        np.savez(
            args.output_dir / "embeddings.npz",
            embedding=adata_out.obsm["X_scVI"],
            batch_labels=adata_out.obs["batch"].values,
            ref_mask=ref_mask.values,
            query_mask=query_mask.values,
        )

        # Save corrected AnnData
        adata_out.write_h5ad(args.output_dir / "result.h5ad", compression="gzip")

        with open(args.output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  -> {adata_out.n_obs} spots, latent={meta['n_latent']}d "
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
