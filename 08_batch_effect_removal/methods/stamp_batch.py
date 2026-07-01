#!/usr/bin/env python
"""STAMP batch correction wrapper for FEAST batch-effect benchmarking.

Uses STAMP (Spatial Topic modeling for Annotation and Multi-sample integration)
with SGC mode and categorical_covariate_keys for batch correction.

Usage:
    /maiziezhou_lab2/yiru/miniconda3/envs/sctm/bin/python methods/stamp_batch.py \\
      --reference alpha_0.00.h5ad --query alpha_0.50.h5ad \\
      --output-dir outputs/methods/STAMP/shift_only/alpha_0.50
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


def _build_spatial_graph(adata, n_neighbors=6):
    """Build spatial neighbor graph for each batch separately."""
    from sklearn.neighbors import kneighbors_graph

    spatial = adata.obsm["spatial"]
    adj = kneighbors_graph(spatial, n_neighbors=n_neighbors, mode="connectivity")
    adata.obsp["spatial_connectivities"] = adj
    adata.obsp["spatial_distances"] = adj.copy()
    adata.uns["spatial_neighbors"] = {"n_neighbors": n_neighbors}
    return adata


def run_stamp_batch_correction(
    adata_ref: sc.AnnData,
    adata_query: sc.AnnData,
    n_topics: int = 10,
    n_hvgs: int = 2000,
    n_layers: int = 1,
    hidden_size: int = 128,
    batch_key: str = "batch",
    mode: str = "sgc",
    learning_rate: float = 0.01,
    seed: int = 42,
) -> tuple[sc.AnnData, dict]:
    """Run STAMP batch correction on concatenated reference + query.

    Returns (concatenated AnnData with cell_by_topic, metadata dict).
    """
    from sctm.stamp import STAMP

    t0 = time.time()

    # Concatenate reference and query with batch labels
    adata_ref_copy = adata_ref.copy()
    adata_query_copy = adata_query.copy()
    adata_ref_copy.obs[batch_key] = "ref"
    adata_query_copy.obs[batch_key] = "query"

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

    # Build spatial graph (per-batch)
    _build_spatial_graph(adata_hvg, n_neighbors=6)

    # Run STAMP
    model = STAMP(
        adata_hvg,
        n_topics=n_topics,
        n_layers=n_layers,
        hidden_size=hidden_size,
        categorical_covariate_keys=[batch_key],
        gene_likelihood="nb",
        mode=mode,
        verbose=False,
    )

    model.train(learning_rate=learning_rate)

    # Extract topic proportions (used as corrected embedding)
    cell_by_topic = model.get_cell_by_topic()
    feature_by_topic = model.get_feature_by_topic()

    adata_hvg.obsm["X_STAMP"] = cell_by_topic.values.astype(np.float32)

    # Store topic info
    adata_hvg.uns["STAMP_cell_by_topic"] = cell_by_topic
    adata_hvg.uns["STAMP_feature_by_topic"] = feature_by_topic

    elapsed = time.time() - t0
    meta = {
        "method": "STAMP",
        "n_topics": n_topics,
        "n_hvgs": n_hvgs,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "mode": mode,
        "gene_likelihood": "nb",
        "n_genes_input": adata.n_vars,
        "n_genes_hvg": adata_hvg.n_vars,
        "n_spots_total": adata_hvg.n_obs,
        "elapsed_seconds": round(elapsed, 2),
    }

    return adata_hvg, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-topics", type=int, default=10)
    parser.add_argument("--n-hvgs", type=int, default=2000)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--mode", type=str, default="sgc")
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
    print(f"STAMP: ref={adata_ref.n_obs}x{adata_ref.n_vars}, "
          f"query={adata_query.n_obs}x{adata_query.n_vars}")

    try:
        adata_out, meta = run_stamp_batch_correction(
            adata_ref,
            adata_query,
            n_topics=args.n_topics,
            n_hvgs=args.n_hvgs,
            n_layers=args.n_layers,
            hidden_size=args.hidden_size,
            mode=args.mode,
            seed=args.seed,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        fail_file = args.output_dir / "FAILED.txt"
        if fail_file.exists():
            fail_file.unlink()

        # Save embedding
        np.savez(
            args.output_dir / "embeddings.npz",
            embedding=adata_out.obsm["X_STAMP"],
            batch_labels=adata_out.obs["batch"].values,
        )

        # Save cell_by_topic
        adata_out.uns["STAMP_cell_by_topic"].to_csv(
            args.output_dir / "cell_by_topic.csv"
        )

        adata_out.write_h5ad(args.output_dir / "result.h5ad", compression="gzip")

        with open(args.output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  -> {adata_out.n_obs} spots, {meta['n_topics']} topics "
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
