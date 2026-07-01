#!/usr/bin/env python3
"""Plot spatial maps of ground truth vs predicted clusters. Read-only.

Selects representative simulations (baseline + extremes) per slice and method.

Usage:
    conda run -p /maiziezhou_lab2/yiru/envs/feast-py311-conda \\
    python scripts/plot_spatial_maps.py \\
      --benchmark outputs/benchmarks/clustering_benchmark_results.csv \\
      --output-dir outputs/plots/spatial_maps
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

REPRESENTATIVE_SIMS = ["baseline", "mean_0_10", "mean_4", "sparsity_4"]


def plot_spatial(adata, color_key: str, title: str, output_path: Path):
    """Plot spatial coordinates colored by a categorical key."""
    if "spatial" not in adata.obsm:
        raise ValueError("missing obsm['spatial']")

    coords = np.asarray(adata.obsm["spatial"])
    labels = adata.obs[color_key].astype("category")
    codes = labels.cat.codes.to_numpy()
    categories = [str(c) for c in labels.cat.categories]
    cmap = plt.get_cmap("tab20", max(len(categories), 1))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(coords[:, 0], coords[:, 1], c=codes, cmap=cmap, s=8, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5,
               markerfacecolor=cmap(i), markeredgecolor="none", label=cat)
        for i, cat in enumerate(categories)
    ]
    if handles:
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, fontsize=7)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path,
                        default=Path("outputs/benchmarks/clustering_benchmark_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/plots/spatial_maps"))
    parser.add_argument("--simulation-manifest", type=Path,
                        default=Path("outputs/simulation_manifest.csv"))
    args = parser.parse_args()

    if not args.benchmark.exists():
        print(f"ERROR: benchmark not found: {args.benchmark}", file=sys.stderr)
        return 1

    results = pd.read_csv(args.benchmark)
    methods = [str(m) for m in results["method"].unique()]
    slices = [str(s) for s in results["slice_id"].unique()]

    written = 0
    failed = 0
    skipped = 0

    for sim_id in REPRESENTATIVE_SIMS:
        for method in methods:
            for slice_id in slices:
                result_h5ad = Path("outputs/methods") / method / slice_id / sim_id / "result.h5ad"
                if not result_h5ad.exists():
                    print(f"  SKIP {method}/{slice_id}/{sim_id} — result.h5ad missing")
                    skipped += 1
                    continue

                try:
                    adata = sc.read_h5ad(str(result_h5ad))

                    out_dir = args.output_dir / slice_id / sim_id
                    out_dir.mkdir(parents=True, exist_ok=True)

                    # Ground truth
                    if "ground_truth" in adata.obs.columns:
                        plot_spatial(
                            adata, "ground_truth",
                            f"{slice_id} {sim_id} — Ground Truth",
                            out_dir / f"{method}_gt.pdf",
                        )
                        written += 1

                    # Predicted
                    if "predicted_cluster" in adata.obs.columns:
                        plot_spatial(
                            adata, "predicted_cluster",
                            f"{slice_id} {sim_id} — {method}",
                            out_dir / f"{method}_pred.pdf",
                        )
                        written += 1

                    print(f"  OK   {method}/{slice_id}/{sim_id}")

                except Exception as e:
                    print(f"  FAIL {method}/{slice_id}/{sim_id}: {e}")
                    failed += 1

    print(f"\nSpatial maps written to {args.output_dir}: {written} panels "
          f"({failed} failed, {skipped} skipped)")
    return 1 if written == 0 or failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
