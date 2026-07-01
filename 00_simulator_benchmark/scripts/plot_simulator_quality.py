#!/usr/bin/env python3
"""Generate QC plots from simulator quality metrics CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

METRIC_LABELS = {
    "mean_correlation": "Gene Mean Correlation (r)",
    "variance_correlation": "Gene Variance Correlation (r)",
    "zero_prop_ks": "Zero Proportion KS Distance",
    "spot_zero_prop_ks": "Spot Zero Prop. KS Distance",
    "relative_error_mean": "Mean Relative Error",
    "library_size_ks": "Library Size KS Distance",
    "gene_detection_ks": "Gene Detection KS Distance",
}

LOWER_IS_BETTER = {
    "zero_prop_ks", "spot_zero_prop_ks", "relative_error_mean",
    "library_size_ks", "gene_detection_ks",
}


def plot_metrics(metrics_path: Path, output_dir: Path) -> int:
    metrics = pd.read_csv(metrics_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    metric_cols = [c for c in METRIC_LABELS if c in metrics.columns]
    simulators = sorted(metrics["simulator"].unique())

    # Per-metric box plots
    fig, axes = plt.subplots(
        len(metric_cols), 1,
        figsize=(10, 3.5 * len(metric_cols)),
        squeeze=False,
    )

    for ax, col in zip(axes[:, 0], metric_cols):
        data = metrics[[col, "simulator"]].dropna()
        order = data.groupby("simulator")[col].median().sort_values().index.tolist()
        sns.boxplot(data=data, x="simulator", y=col, order=order, ax=ax,
                    palette="Set2", width=0.6)
        sns.stripplot(data=data, x="simulator", y=col, order=order, ax=ax,
                      color="black", alpha=0.4, size=4)
        ax.set_title(METRIC_LABELS.get(col, col))
        ax.set_xlabel("")
        if col in LOWER_IS_BETTER:
            ax.set_ylabel("lower is better")

    plt.tight_layout()
    fig.savefig(output_dir / "simulator_quality_boxplots.png", dpi=150)
    plt.close(fig)

    # Heatmap of median metrics by simulator
    pivot = metrics.groupby("simulator")[metric_cols].median()
    # Normalize per column for heatmap
    pivot_norm = (pivot - pivot.mean()) / (pivot.std() + 1e-10)

    fig, ax = plt.subplots(figsize=(10, max(3, len(simulators) * 0.5)))
    sns.heatmap(
        pivot_norm,
        annot=pivot.round(3).values,
        fmt="",
        cmap="RdYlGn",
        center=0,
        xticklabels=[METRIC_LABELS.get(c, c) for c in metric_cols],
        ax=ax,
    )
    ax.set_title("Median Metrics by Simulator (z-score normalized)")
    plt.tight_layout()
    fig.savefig(output_dir / "simulator_quality_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {output_dir}/")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True,
                        help="Path to simulator_quality_metrics.csv")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for plots")
    args = parser.parse_args()

    return plot_metrics(args.metrics, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
