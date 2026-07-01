# %%
"""3D stack reconstruction — cross-density comparison.

Bar chart comparing quality metrics across three density levels
(dense_gap3, medium_gap5, sparse_gap10) from the archived cross-density summary.
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
CROSS_DENSITY_CSV = (
    EXPERIMENT_ROOT
    / "05_3d_stack"
    / "archive"
    / "archived_20260629_214105"
    / "outputs"
    / "cross_density_summary.csv"
)
OUTPUT_DIR = Path("/maiziezhou_lab2/yiru/Reproduce/Visualization/figures/3d_stack")
OUTPUT_STEM = "3d_stack_cross_density"

METRIC_SPECS = [
    ("mean_corr_mean", "Mean Correlation ↑", "Pearson r"),
    ("var_corr_mean", "Variance Correlation ↑", "Pearson r"),
    ("moran_corr_mean", "Moran Correlation ↑", "Moran I"),
    ("zero_ks_mean", "Zero KS Distance ↓", "KS distance"),
    ("z_coherence_mean", "Z Coherence ↑", "Spearman ρ"),
]

DENSITY_DISPLAY = {
    "dense_gap3": "Dense\n(gap=3)",
    "medium_gap5": "Medium\n(gap=5)",
    "sparse_gap10": "Sparse\n(gap=10)",
}
DENSITY_ORDER = ["dense_gap3", "medium_gap5", "sparse_gap10"]
DENSITY_PALETTE = ["#4c72b0", "#55a868", "#c44e52"]


# %%
def build_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["density_name"].isin(DENSITY_ORDER)].copy()
    return df


# %%
def draw_figure(summary: pd.DataFrame, output_dir: Path, output_stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 10,
    })

    n_metrics = len(METRIC_SPECS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 3.5))
    if n_metrics == 1:
        axes = [axes]

    bar_width = 0.55
    x = np.arange(len(DENSITY_ORDER))

    for ax_idx, (metric_col, title, ylabel) in enumerate(METRIC_SPECS):
        ax = axes[ax_idx]
        vals = []
        for density in DENSITY_ORDER:
            row = summary[summary["density_name"] == density]
            if row.empty:
                vals.append(0)
            else:
                vals.append(row[metric_col].values[0])

        bars = ax.bar(
            x, vals, bar_width,
            color=DENSITY_PALETTE,
            edgecolor="white", linewidth=0.8,
        )
        # Annotate bars with values
        for bar, val in zip(bars, vals):
            if abs(val) < 0.01:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5,
                fontweight="bold",
            )

        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([DENSITY_DISPLAY[d] for d in DENSITY_ORDER], fontsize=8)
        ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.55)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#555555")

    plt.subplots_adjust(
        left=0.08, right=0.98, top=0.88, bottom=0.18,
        wspace=0.35,
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
    summary = build_summary(CROSS_DENSITY_CSV)
    draw_figure(summary, args.output_dir, args.output_stem, args.dpi)
    return 0


# %%
if __name__ == "__main__":
    raise SystemExit(main())
