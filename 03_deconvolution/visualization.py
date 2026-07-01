#!/usr/bin/env python3
"""Generate Figure 2 deconvolution benchmark visualizations.

Reads saved benchmark results and proportion data. Does NOT rerun
any simulation or deconvolution method.

Usage:
    conda run -p /maiziezhou_lab2/yiru/envs/feast-py311-conda \\
    python scripts/visualization.py \\
      --benchmark-csv outputs/benchmarks/deconvolution_benchmark_results.csv \\
      --truth-dir outputs/ground_truth \\
      --prediction-dirs outputs/cell2location outputs/rctd \\
      --simulation-dir outputs/simulations \\
      --slices 007 050 100 \\
      --resolutions 0.1 0.25 \\
      --output-dir outputs/plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


def plot_metrics_overview(
    results_csv: Path,
    output_pdf: Path,
    output_png: Path,
) -> None:
    df = pd.read_csv(results_csv)
    df_ok = df[df["status"] == "ok"].copy()

    if len(df_ok) == 0:
        print("WARNING: No successful benchmarks to plot")
        return

    methods = sorted(df_ok["method"].unique())
    slices = sorted(df_ok["slice_id"].unique())
    resolutions = sorted(df_ok["resolution"].unique(), key=float)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(methods), 3)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: JSD by slice and method (averaged across resolutions)
    ax = axes[0, 0]
    x = np.arange(len(slices))
    width = 0.35
    for i, method in enumerate(methods):
        vals = []
        for s in slices:
            sub = df_ok[(df_ok["method"] == method) & (df_ok["slice_id"] == s)]
            vals.append(sub["jsd_mean"].mean() if len(sub) > 0 else np.nan)
        ax.bar(x + i * width, vals, width, label=method, color=colors[i])
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(slices)
    ax.set_ylabel("JSD (lower is better)")
    ax.set_title("Jensen-Shannon Divergence (mean across resolutions)")
    ax.legend(fontsize=8)

    # Panel 2: Pearson correlation
    ax = axes[0, 1]
    for i, method in enumerate(methods):
        vals = []
        for s in slices:
            sub = df_ok[(df_ok["method"] == method) & (df_ok["slice_id"] == s)]
            vals.append(sub["pearson_mean"].mean() if len(sub) > 0 else np.nan)
        ax.bar(x + i * width, vals, width, label=method, color=colors[i])
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(slices)
    ax.set_ylabel("Pearson r (higher is better)")
    ax.set_title("Pearson Correlation (mean across resolutions)")

    # Panel 3: Summed RMSE
    ax = axes[1, 0]
    for i, method in enumerate(methods):
        vals = []
        for s in slices:
            sub = df_ok[(df_ok["method"] == method) & (df_ok["slice_id"] == s)]
            vals.append(sub["summed_rmse"].mean() if len(sub) > 0 else np.nan)
        ax.bar(x + i * width, vals, width, label=method, color=colors[i])
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(slices)
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("Summed RMSE (mean across resolutions)")

    # Panel 4: Distance correlation (may be NaN if dcor unavailable)
    ax = axes[1, 1]
    for i, method in enumerate(methods):
        vals = []
        for s in slices:
            sub = df_ok[(df_ok["method"] == method) & (df_ok["slice_id"] == s)]
            v = sub["distance_correlation"].mean() if len(sub) > 0 else np.nan
            vals.append(v if not np.isnan(v) else np.nan)
        ax.bar(x + i * width, vals, width, label=method, color=colors[i])
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(slices)
    ax.set_ylabel("Distance correlation (higher is better)")
    ax.set_title("Distance Correlation (mean across resolutions)")

    fig.suptitle("Figure 2: Deconvolution Benchmark Metrics", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=150, bbox_inches="tight")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metrics plot: {output_pdf}")


def plot_proportion_maps(
    truth_dir: Path,
    prediction_dirs: list[Path],
    simulation_dir: Path,
    output_dir: Path,
    slices: list[str],
    resolutions: list[str],
) -> None:
    map_dir = output_dir / "proportion_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(slices) * len(resolutions) * (1 + len(prediction_dirs))

    method_sources = {"ground_truth": truth_dir}
    for pred_dir in prediction_dirs:
        method_sources[pred_dir.name] = pred_dir

    count = 0
    for slice_id in slices:
        slice_sim_dir = simulation_dir / slice_id
        if not slice_sim_dir.exists():
            print(f"WARNING: no simulation dir for slice {slice_id}")
            continue

        for resolution in resolutions:
            sim_path = slice_sim_dir / f"resolution_{resolution}.h5ad"
            if not sim_path.exists():
                print(f"WARNING: no simulation file for {slice_id}/resolution_{resolution}")
                continue

            try:
                adata = sc.read_h5ad(str(sim_path))
                coords = adata.obsm["spatial"]
            except Exception as e:
                print(f"WARNING: failed to load {sim_path}: {e}")
                continue

            for method_name, src_dir in method_sources.items():
                csv_path = src_dir / slice_id / f"resolution_{resolution}_proportions.csv"
                if not csv_path.exists():
                    print(f"WARNING: {csv_path} not found, skipping map")
                    count += 1
                    continue

                try:
                    proportions = pd.read_csv(csv_path, index_col=0)
                except Exception as e:
                    print(f"WARNING: failed to read {csv_path}: {e}")
                    count += 1
                    continue

                n_spots = min(len(coords), proportions.shape[0])
                if n_spots == 0:
                    count += 1
                    continue

                dominant_ct_idx = np.argmax(proportions.values[:n_spots], axis=1)
                unique_cts = list(proportions.columns)

                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                cmap = plt.cm.tab20

                for i, ct in enumerate(unique_cts):
                    mask = dominant_ct_idx == i
                    if mask.sum() == 0:
                        continue
                    ax.scatter(
                        coords[:n_spots, 0][mask],
                        coords[:n_spots, 1][mask],
                        c=[cmap(i % 20)], s=30, alpha=0.85,
                        label=ct,
                    )

                ax.set_title(f"{method_name} — Slice {slice_id}, Resolution {resolution}")
                ax.set_xlabel("Spatial X")
                ax.set_ylabel("Spatial Y")
                ax.set_aspect("equal")
                if len(unique_cts) <= 20:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left",
                              fontsize=5, title="Cell Type")

                fig.tight_layout()
                map_path = map_dir / f"{slice_id}_resolution_{resolution}_{method_name}.png"
                fig.savefig(map_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                count += 1
                print(f"  [{count}/{n_total}] Saved {map_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--prediction-dirs", type=Path, nargs="+", required=True,
                        help="Method prediction dirs (e.g. outputs/cell2location outputs/rctd)")
    parser.add_argument("--simulation-dir", type=Path, required=True,
                        help="Directory with simulation .h5ad files")
    parser.add_argument("--slices", type=str, nargs="+",
                        default=["007", "050", "100"])
    parser.add_argument("--resolutions", type=str, nargs="+",
                        default=["0.1", "0.25"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/plots"))
    parser.add_argument("--skip-maps", action="store_true",
                        help="Skip proportion map generation (metrics only)")
    args = parser.parse_args()

    print(f"Metrics plot: reading from {args.benchmark_csv}")
    if not args.benchmark_csv.exists():
        print(f"ERROR: benchmark CSV not found: {args.benchmark_csv}", file=sys.stderr)
        return 1

    plot_metrics_overview(
        args.benchmark_csv,
        args.output_dir / "deconvolution_metrics.pdf",
        args.output_dir / "deconvolution_metrics.png",
    )

    if not args.skip_maps:
        print(f"Proportion maps: {len(args.slices)} slices x "
              f"{len(args.resolutions)} resolutions x "
              f"{1 + len(args.prediction_dirs)} methods")
        plot_proportion_maps(
            args.truth_dir,
            args.prediction_dirs,
            args.simulation_dir,
            args.output_dir,
            args.slices,
            args.resolutions,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
