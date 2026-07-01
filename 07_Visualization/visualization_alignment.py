"""Alignment robustness curve — 2×3 multi-metric panel.

Reads alignment_alteration_metrics.csv (long format: method, angle, metric, score).
Top-left subplot reserved for manual schematic placement.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG_DIR = Path("/maiziezhou_lab2/yiru/Reproduce/Visualization/figures")

CSV_PATH = HERE / "alignment_alteration_metrics.csv"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df["angle"] = df["angle"].astype(float)

# ---------------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------------
methods = ["Spateo", "PASTE"]

palette = {
    "Spateo": "#e84b43",
    "PASTE": "#42b7d0",
}

linestyles = {
    "Spateo": "-",
    "PASTE": "--",
}

metric_positions = {
    "Precision": (0, 1),
    "F1 Score": (0, 2),
    "Accuracy": (1, 0),
    "Gene Expression Correlation": (1, 1),
    "Adjusted Region Accuracy": (1, 2),
}

metric_ylim = {
    "Accuracy": (-0.05, 1.05),
    "Precision": (-0.05, 1.05),
    "F1 Score": (-0.05, 1.05),
    "Gene Expression Correlation": (0.78, 1.005),
    "Adjusted Region Accuracy": (-0.05, 1.05),
}

x_ticks = [0, 10, 20, 30, 40, 50, 60]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.3), constrained_layout=False)

# Top-left reserved for manual schematic
axes[0, 0].axis("off")

# ---------------------------------------------------------------------------
# Plot each metric
# ---------------------------------------------------------------------------
for metric, pos in metric_positions.items():
    ax = axes[pos]
    sub_metric = df[df["metric"] == metric]

    for method in methods:
        sub = sub_metric[sub_metric["method"] == method].sort_values("angle")
        if sub.empty:
            continue

        ax.plot(
            sub["angle"],
            sub["score"],
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            linestyle=linestyles[method],
            color=palette[method],
            label=method,
        )

    ax.set_title(metric, fontsize=13.5, fontweight="bold", pad=4)
    ax.set_xlim(-2, 63)
    ax.set_xticks(x_ticks)

    if metric in metric_ylim:
        ax.set_ylim(*metric_ylim[metric])

    ax.grid(True, axis="both", color="#d9d9d9", linewidth=0.55, alpha=0.55)
    ax.set_axisbelow(True)

    ax.tick_params(axis="both", labelsize=8, width=0.7, length=3, color="#555555")

    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("#666666")

    if pos[0] == 1:
        ax.set_xlabel("Angles (Degree)", fontsize=11, fontweight="bold")
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    if metric == "Accuracy":
        ax.set_ylabel("Score", fontsize=11)
    else:
        ax.set_ylabel("")

# ---------------------------------------------------------------------------
# Legend in the blank top-left subplot
# ---------------------------------------------------------------------------
legend_ax = axes[0, 0]
for i, method in enumerate(methods):
    y = 0.35 - i * 0.12
    legend_ax.plot(
        [0.50, 0.62],
        [y, y],
        transform=legend_ax.transAxes,
        color=palette[method],
        linestyle=linestyles[method],
        linewidth=1.8,
        marker="o",
        markersize=5,
        clip_on=False,
    )
    legend_ax.text(
        0.67,
        y,
        method,
        transform=legend_ax.transAxes,
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

# ---------------------------------------------------------------------------
# Spacing and export
# ---------------------------------------------------------------------------
plt.subplots_adjust(
    left=0.07,
    right=0.985,
    top=0.91,
    bottom=0.12,
    wspace=0.28,
    hspace=0.32,
)

FIG_DIR.mkdir(parents=True, exist_ok=True)

for fmt, dpi in [("pdf", None), ("png", 600)]:
    path = FIG_DIR / f"alignment_alteration_curve_panel.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved {path}")

plt.show()
