#!/usr/bin/env python3
"""Plot clustering benchmark metrics from saved results. Read-only.

Usage:
    conda run -p /maiziezhou_lab2/yiru/envs/feast-py311-conda \\
    python scripts/plot_clustering_metrics.py \\
      --benchmark outputs/benchmarks/clustering_benchmark_results.csv \\
      --output-dir outputs/plots
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


def plot_heatmap(results: pd.DataFrame, metric: str, output_path: Path):
    """Plot heatmap: methods x alteration_type+fold_change for one metric."""
    ok = results[results["status"] == "ok"].copy()
    if ok.empty:
        return

    ok["alteration_label"] = ok["alteration_type"] + "_" + ok["fold_change"].astype(str)

    pivot = ok.pivot_table(
        index="method", columns="alteration_label",
        values=metric, aggfunc="mean",
    )

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(18, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlBu_r", ax=ax,
                vmin=0, vmax=1, linewidths=0.5)
    ax.set_title(f"{metric} — averaged across slices")
    plt.tight_layout()
    fig.savefig(str(output_path).replace(".pdf", f"_{metric}.pdf"), dpi=150)
    fig.savefig(str(output_path).replace(".pdf", f"_{metric}.png"), dpi=150)
    plt.close(fig)


def plot_method_comparison(results: pd.DataFrame, metric: str, output_path: Path):
    """Bar chart comparing methods per alteration type."""
    ok = results[results["status"] == "ok"].copy()
    if ok.empty:
        return

    fig, ax = plt.subplots(figsize=(16, 6))
    order = ["baseline", "mean", "variance", "sparsity"]
    sns.barplot(
        data=ok, x="alteration_type", y=metric, hue="method",
        order=[o for o in order if o in ok["alteration_type"].unique()],
        ci="sd", ax=ax,
    )
    ax.set_title(f"{metric} by method and alteration type (mean ± SD)")
    plt.tight_layout()
    fig.savefig(str(output_path).replace(".pdf", f"_{metric}.pdf"), dpi=150)
    fig.savefig(str(output_path).replace(".pdf", f"_{metric}.png"), dpi=150)
    plt.close(fig)


def plot_chaos_pas_scatter(results: pd.DataFrame, output_path: Path):
    """CHAOS vs PAS scatter colored by method and alteration_type."""
    ok = results[results["status"] == "ok"].copy()
    if ok.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, method in zip(axes, ok["method"].unique()):
        df = ok[ok["method"] == method]
        for alt_type in df["alteration_type"].unique():
            sub = df[df["alteration_type"] == alt_type]
            ax.scatter(sub["CHAOS"], sub["PAS"], label=alt_type, alpha=0.6, s=30)
        ax.set_xlabel("CHAOS")
        ax.set_ylabel("PAS")
        ax.set_title(method)
        ax.legend(fontsize=7)

    plt.suptitle("CHAOS vs PAS by method")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path,
                        default=Path("outputs/benchmarks/clustering_benchmark_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/plots"))
    args = parser.parse_args()

    if not args.benchmark.exists():
        print(f"ERROR: benchmark file not found: {args.benchmark}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(args.benchmark)
    print(f"Loaded {len(results)} benchmark rows")

    metrics = ["ARI", "NMI", "AMI", "Homogeneity", "Completeness", "V_measure"]
    heatmap_base = args.output_dir / "clustering_metrics_heatmap.pdf"
    comparison_base = args.output_dir / "method_comparison.pdf"

    for metric in metrics:
        plot_heatmap(results, metric, heatmap_base)
        print(f"  Heatmap: {metric}")

    for metric in metrics:
        plot_method_comparison(results, metric, comparison_base)
        print(f"  Comparison: {metric}")

    chaos_path = args.output_dir / "chaos_vs_pas_scatter.pdf"
    plot_chaos_pas_scatter(results, chaos_path)
    print(f"  CHAOS vs PAS scatter")

    print(f"Plots written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
