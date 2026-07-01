# %%
"""2D conditional transfer — AR sweep curves.

Plots key quality metrics against assignment randomness (AR) across
two DLPFC slice-pair directions. One panel per metric.
"""

# %%
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
EXPERIMENT_ROOT = Path("/maiziezhou_lab2/yiru/FEAST_experiments")
RESULTS_DIR = EXPERIMENT_ROOT / "04_2d_conditional_transfer" / "results"
OUTPUT_DIR = Path("/maiziezhou_lab2/yiru/Reproduce/Visualization/figures/2d_transfer")
OUTPUT_STEM = "2d_transfer_ar_sweep"

AR_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5]

METRIC_SPECS = [
    ("mean_corr", "Mean Correlation ↑", "Pearson r"),
    ("var_corr", "Variance Correlation ↑", "Pearson r"),
    ("moran_corr", "Moran Correlation ↑", "Moran I"),
    ("zero_ks", "Zero KS Distance ↓", "KS distance"),
    ("median_gene_pearson", "Median Gene Pearson ↑", "Pearson r"),
]

DIRECTION_PALETTE = {
    "151675→151676": "#4c72b0",
    "151676→151675": "#c44e52",
}


# %%
def load_all_summaries() -> pd.DataFrame:
    rows = []
    for ar in AR_VALUES:
        summary_path = RESULTS_DIR / f"ar_{ar}" / "summary.csv"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path)
        df["ar"] = float(ar)
        rows.append(df)
    combined = pd.concat(rows, ignore_index=True)
    combined["direction_label"] = combined["source"].astype(str) + "→" + combined["target"].astype(str)
    return combined


# %%
def draw_figure(data: pd.DataFrame, output_dir: Path, output_stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 9,
    })

    directions = sorted(data["direction_label"].unique())
    n_metrics = len(METRIC_SPECS)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.0 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax_idx, (metric_col, title, ylabel) in enumerate(METRIC_SPECS):
        ax = axes[ax_idx]
        for direction in directions:
            sub = data[data["direction_label"] == direction].sort_values("ar")
            if sub.empty:
                continue
            ax.plot(
                sub["ar"], sub[metric_col],
                marker="o", markersize=6, linewidth=1.8,
                color=DIRECTION_PALETTE[direction],
                label=direction,
            )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
        ax.set_xlabel("Assignment Randomness", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(-0.03, 0.53)
        ax.grid(True, axis="both", color="#d9d9d9", linewidth=0.6, alpha=0.55)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#555555")

    # Hide unused axes
    for extra_ax in axes[n_metrics:]:
        extra_ax.axis("off")

    axes[0].legend(fontsize=8, framealpha=0.9, edgecolor="#cccccc")

    plt.subplots_adjust(
        left=0.10, right=0.97, top=0.93, bottom=0.10,
        wspace=0.32, hspace=0.38,
    )

    for suffix, kwargs in {"pdf": {}, "svg": {}, "png": {"dpi": dpi}}.items():
        path = output_dir / f"{output_stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"Saved: {path}")

    plt.close(fig)


# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-stem", default=OUTPUT_STEM)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_all_summaries()
    draw_figure(data, args.output_dir, args.output_stem, args.dpi)
    return 0


# %%
if __name__ == "__main__":
    raise SystemExit(main())
