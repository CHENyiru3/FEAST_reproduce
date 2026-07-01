#!/usr/bin/env python3
"""Leiden clustering with multi-resolution sweep for Figure 2 benchmark.

Sweeps 5 resolutions, picks the best by ARI against ground_truth.

Usage:
    conda run -p /maiziezhou_lab2/yiru/envs/feast-py311-conda \\
    python scripts/methods/clustering_leiden.py \\
      --input outputs/hvg_inputs/151670/mean_0.10.h5ad \\
      --output-dir outputs/methods/Leiden/151670/mean_0.10 \\
      --resolutions 0.2,0.4,0.6,0.8,1.0
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
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")


def run_leiden_sweep(
    adata: sc.AnnData,
    resolutions: list[float],
    n_pcs: int = 50,
    n_neighbors: int = 15,
    seed: int = 2026,
) -> tuple[np.ndarray, dict]:
    """Run Leiden at multiple resolutions, return best by ARI. Returns (labels, metadata)."""
    np.random.seed(seed)
    t0 = time.time()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=min(n_pcs, adata.n_obs - 1, adata.n_vars - 1))
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    sweep_results = []
    best_ari = -1.0
    best_labels = None
    best_resolution = None

    gt = adata.obs["ground_truth"].values if "ground_truth" in adata.obs else None

    for res in resolutions:
        sc.tl.leiden(adata, resolution=res, key_added=f"leiden_{res}")
        labels = adata.obs[f"leiden_{res}"].values
        n_clus = len(np.unique(labels))

        ari = adjusted_rand_score(gt, labels) if gt is not None else np.nan

        sweep_results.append({
            "resolution": res,
            "n_clusters": n_clus,
            "ari": float(ari) if not np.isnan(ari) else float("nan"),
        })

        if gt is not None and ari > best_ari:
            best_ari = ari
            best_labels = labels
            best_resolution = res

    if best_labels is None:
        best_labels = adata.obs[f"leiden_{resolutions[0]}"].values
        best_resolution = resolutions[0]

    elapsed = time.time() - t0
    meta = {
        "method": "Leiden",
        "n_pcs": n_pcs,
        "n_neighbors": n_neighbors,
        "best_resolution": best_resolution,
        "best_ari": float(best_ari) if not np.isnan(best_ari) else None,
        "n_clusters": len(np.unique(best_labels)),
        "sweep": sweep_results,
        "elapsed_seconds": round(elapsed, 2),
    }

    return best_labels, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolutions", type=str, default="0.2,0.5,0.8,1.0,1.5,2.0")
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1

    sc.settings.verbosity = 0
    adata = sc.read_h5ad(str(args.input))
    print(f"Leiden: {args.input.name} ({adata.n_obs} spots x {adata.n_vars} genes)")

    resolutions = [float(r.strip()) for r in args.resolutions.split(",")]

    try:
        labels, meta = run_leiden_sweep(
            adata, resolutions=resolutions,
            n_pcs=args.n_pcs, n_neighbors=args.n_neighbors, seed=args.seed,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        fail_file = args.output_dir / "FAILED.txt"
        if fail_file.exists():
            fail_file.unlink()

        adata.obs["predicted_cluster"] = labels
        adata.obs["predicted_cluster"] = adata.obs["predicted_cluster"].astype("category")

        clusters = pd.DataFrame({
            "spot_barcode": adata.obs.index,
            "predicted_cluster": labels,
        })
        clusters.to_csv(args.output_dir / "clusters.csv", index=False)

        # Clean uns of scanpy-internal keys that can't serialize to h5ad
        # (e.g., leiden_sweep with mixed-type numpy arrays)
        for k in list(adata.uns.keys()):
            if k.startswith("leiden") or k == "neighbors" or k == "pca":
                del adata.uns[k]

        try:
            adata.write_h5ad(str(args.output_dir / "result.h5ad"), compression="gzip")
        except Exception:
            print(f"  -> WARNING: result.h5ad write failed, clusters.csv is fine")

        with open(args.output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        best_ari_str = f"{meta['best_ari']:.4f}" if meta["best_ari"] is not None else "N/A"
        print(f"  -> best_res={meta['best_resolution']}, {meta['n_clusters']} clusters, "
              f"ARI={best_ari_str}, {meta['elapsed_seconds']:.1f}s")
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
