#!/usr/bin/env python
"""
Stage 4a: Plot alignment rotation error and spatial error metrics.

Reads benchmark CSV only. No rerunning of simulation or methods.

Usage:
    conda run -p ${FEAST_ENV} python scripts/plot_alignment_metrics.py \
      --benchmark-csv outputs/benchmarks/alignment_benchmark_results.csv \
      --output-dir outputs/plots
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot alignment benchmark metrics")
    parser.add_argument("--benchmark-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.benchmark_csv)
    if "rotation_error" not in df.columns and "recovered_rotation" in df.columns:
        df["rotation_error"] = df["recovered_rotation"].abs()
    ok = df[df["status"] == "ok"].copy()

    if ok.empty:
        print("No successful benchmark rows to plot.")
        return

    methods = sorted(ok["method"].unique())
    angles = sorted(ok["angle"].unique())

    colors = {"spateo": "#2196F3", "paste": "#2E7D32"}
    markers = {"spateo": "o", "paste": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Rotation error vs angle
    ax = axes[0]
    for method in methods:
        mdf = ok[ok["method"] == method].sort_values("angle")
        ax.plot(mdf["angle"], mdf["rotation_error"],
                color=colors.get(method, "gray"),
                marker=markers.get(method, "o"),
                linewidth=2, markersize=8, label=method.capitalize())
    ax.set_xlabel("True Rotation Angle (deg)")
    ax.set_ylabel("Rotation Error (deg)")
    ax.set_title("Rotation Recovery Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Spatial error vs angle
    ax = axes[1]
    for method in methods:
        mdf = ok[ok["method"] == method].sort_values("angle")
        ax.plot(mdf["angle"], mdf["mean_spatial_error"],
                color=colors.get(method, "gray"),
                marker=markers.get(method, "o"),
                linewidth=2, markersize=8, label=method.capitalize())
    ax.set_xlabel("True Rotation Angle (deg)")
    ax.set_ylabel("Mean Spatial Error (px)")
    ax.set_title("Spatial Alignment Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Pearson correlation vs angle
    ax = axes[2]
    for method in methods:
        mdf = ok[ok["method"] == method].sort_values("angle")
        avg_pearson = (mdf["pearson_x"] + mdf["pearson_y"]) / 2
        ax.plot(mdf["angle"], avg_pearson,
                color=colors.get(method, "gray"),
                marker=markers.get(method, "o"),
                linewidth=2, markersize=8, label=method.capitalize())
    ax.set_xlabel("True Rotation Angle (deg)")
    ax.set_ylabel("Mean Pearson r (x, y)")
    ax.set_title("Coordinate Correlation")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Alignment Benchmark — Rotation Simulation", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    for fmt in ["pdf", "png"]:
        path = out_dir / f"alignment_rotation_metrics.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
