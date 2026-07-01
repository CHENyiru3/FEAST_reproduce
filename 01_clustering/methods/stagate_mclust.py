#!/usr/bin/env python3
"""STAGATE + mclust clustering for Figure 2 clustering benchmark.

Usage:
    conda run -p /maiziezhou_lab2/yiru/envs/STAGATE \\
    python scripts/methods/clustering_STAGATE_mclust.py \\
      --input outputs/hvg_inputs/151670/mean_0.10.h5ad \\
      --output-dir outputs/methods/STAGATE_mclust/151670/mean_0.10 \\
      --rad-cutoff 150 \\
      --n-epochs 500 \\
      --n-clusters auto
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Point rpy2 at the conda env's R (which has mclust), not the system R
if "R_HOME" not in os.environ:
    os.environ["R_HOME"] = "/maiziezhou_lab2/yiru/envs/STAGATE/lib/R"

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse

# Disable TF eager execution before any TF import (required by STAGATE)
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

warnings.filterwarnings("ignore")


def run_stagate_mclust(
    adata: sc.AnnData,
    rad_cutoff: int = 150,
    n_epochs: int = 500,
    latent_key: str = "STAGATE",
    n_clusters: int | None = None,
    seed: int = 2026,
) -> tuple[np.ndarray, dict]:
    """Run STAGATE -> mclust pipeline. Returns (labels, metadata)."""
    import STAGATE

    t0 = time.time()

    if n_clusters is None:
        if "ground_truth" in adata.obs.columns:
            n_clusters = len(adata.obs["ground_truth"].unique())
        else:
            n_clusters = 7

    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)
    adata = STAGATE.train_STAGATE(
        adata,
        alpha=0,
        n_epochs=n_epochs,
        key_added=latent_key,
        random_seed=seed,
        save_attention=False,
        save_loss=False,
    )

    adata = STAGATE.mclust_R(
        adata,
        used_obsm=latent_key,
        num_cluster=n_clusters,
        random_seed=seed,
    )
    labels = adata.obs["mclust"].astype(int).values

    elapsed = time.time() - t0
    meta = {
        "method": "STAGATE_mclust",
        "rad_cutoff": rad_cutoff,
        "n_epochs": n_epochs,
        "latent_key": latent_key,
        "n_clusters_target": n_clusters,
        "n_clusters_mclust": len(np.unique(labels)),
        "mclust_model": "auto",
        "random_seed": seed,
        "input_preprocessing": "hvg_inputs_normalized_log1p",
        "elapsed_seconds": round(elapsed, 2),
    }

    return labels, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rad-cutoff", type=int, default=150)
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--latent-key", type=str, default="STAGATE")
    parser.add_argument("--n-clusters", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1

    sc.settings.verbosity = 0
    np.random.seed(args.seed)

    adata = sc.read_h5ad(str(args.input))
    if not scipy.sparse.issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
    print(f"STAGATE: {args.input.name} ({adata.n_obs} spots x {adata.n_vars} genes)")

    n_clusters = None if args.n_clusters == "auto" else int(args.n_clusters)

    try:
        labels, meta = run_stagate_mclust(
            adata, rad_cutoff=args.rad_cutoff,
            n_epochs=args.n_epochs, latent_key=args.latent_key,
            n_clusters=n_clusters, seed=args.seed,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        fail_file = args.output_dir / "FAILED.txt"
        if fail_file.exists():
            fail_file.unlink()

        adata.obs["predicted_cluster"] = labels.astype(str)
        adata.obs["predicted_cluster"] = adata.obs["predicted_cluster"].astype("category")

        clusters = pd.DataFrame({
            "spot_barcode": adata.obs.index,
            "predicted_cluster": labels,
        })
        clusters.to_csv(args.output_dir / "clusters.csv", index=False)

        # Clean uns of non-serializable keys (TF tensors, spatial nets, etc.)
        for k in list(adata.uns.keys()):
            if k not in ("log1p",):
                try:
                    del adata.uns[k]
                except Exception:
                    pass

        try:
            adata.write_h5ad(str(args.output_dir / "result.h5ad"), compression="gzip")
        except Exception:
            print(f"  -> WARNING: result.h5ad write failed, clusters.csv is fine")

        with open(args.output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  -> {meta['n_clusters_mclust']} clusters in {meta['elapsed_seconds']:.1f}s")
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
