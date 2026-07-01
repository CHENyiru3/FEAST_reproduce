# %%
"""Single-slice simulator benchmark boxplot figure.

This notebook-style script uses the final FEAST simulator benchmark metrics
from Subtask 06 and creates a 2 x 4 multi-panel boxplot figure:
seven metric panels plus a legend panel.
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
from matplotlib.patches import Patch


# %%
EXPERIMENT_ROOT = Path("/maiziezhou_lab2/yiru/FEAST_experiments")
METRICS_CSV = (
    EXPERIMENT_ROOT
    / "00_simulator_benchmark"
    / "outputs"
    / "benchmarks"
    / "simulator_quality_metrics.csv"
)
OUTPUT_DIR = Path("/maiziezhou_lab2/yiru/Reproduce/Visualization/figures/simulator_benchmark")
OUTPUT_STEM = "single_slice_simulator_benchmark_boxplot"


# %%
METHOD_LABELS = {
    "scCube": "scCube",
    "SRTsim": "SRTsim",
    "Splatter": "Splatter",
    "Splatter_Simple": "Splatter_Simple",
    "FEAST_Rank": "FEAST Rank",
    "FEAST_OT_Spatial": "FEAST OT Spatial",
}

PALETTE = {
    "scCube": "#e7c083",
    "SRTsim": "#e75a9c",
    "Splatter": "#f28e73",
    "Splatter_Simple": "#7ea6d8",
    "FEAST Rank": "#7b5ba7",
    "FEAST OT Spatial": "#9eb77d",
}

METRIC_SPECS = [
    {
        "column": "input_mean_corr",
        "label": "Mean Corr ↑",
        "higher_is_better": True,
    },
    {
        "column": "input_variance_corr",
        "label": "Var Corr ↑",
        "higher_is_better": True,
    },
    {
        "column": "moran_i_correlation",
        "label": "Moran I Corr ↑",
        "higher_is_better": True,
    },
    {
        "column": "zero_mask_jaccard",
        "label": "Zero Jaccard ↑",
        "higher_is_better": True,
    },
    {
        "column": "zero_preservation_score",
        "label": "Zero Preserv ↑",
        "higher_is_better": True,
    },
    {
        "column": "statistical_structure_score",
        "label": "Statistical Struct ↑",
        "higher_is_better": True,
    },
    {
        "column": "structured_novelty_score",
        "label": "Structured Novelty ↑",
        "higher_is_better": True,
    },
    {
        "column": "composite_quality_score",
        "label": "Composite Quality ↑",
        "higher_is_better": True,
    },
    {
        "column": "cosine_divergence",
        "label": "Cosine Div ↓",
        "higher_is_better": False,
    },
    {
        "column": "relative_error_mean",
        "label": "Rel Error Mean ↓",
        "higher_is_better": False,
        "yscale": "log",
    },
    {
        "column": "gene_zero_fraction_wasserstein",
        "label": "Zero Frac WDist ↓",
        "higher_is_better": False,
    },
    {
        "column": "gene_mean_wasserstein",
        "label": "Gene Mean WDist ↓",
        "higher_is_better": False,
        "yscale": "log",
    },
    {
        "column": "gene_variance_wasserstein",
        "label": "Gene Var WDist ↓",
        "higher_is_better": False,
        "yscale": "log",
    },
    {
        "column": "library_size_wasserstein",
        "label": "Library WDist ↓",
        "higher_is_better": False,
        "yscale": "log",
    },
]


# %%
def load_metrics(metrics_csv: Path) -> pd.DataFrame:
    """Load simulator metrics and apply publication labels."""
    metrics = pd.read_csv(metrics_csv)
    if "simulator" not in metrics.columns:
        raise ValueError(f"Missing required column 'simulator' in {metrics_csv}")

    metrics = metrics.copy()
    metrics["method"] = metrics["simulator"].map(METHOD_LABELS).fillna(metrics["simulator"])
    return metrics


def build_long_dataframe(metrics: pd.DataFrame) -> pd.DataFrame:
    """Convert wide benchmark metrics to long-form plotting data."""
    rows = []
    for spec in METRIC_SPECS:
        column = spec["column"]
        if column not in metrics.columns:
            continue

        sub = metrics[["method", "sample", column]].dropna(subset=[column])
        for row in sub.itertuples(index=False):
            rows.append(
                {
                    "metric": spec["label"],
                    "method": row.method,
                    "sample": row.sample,
                    "score": float(getattr(row, column)),
                    "source_column": column,
                    "higher_is_better": spec["higher_is_better"],
                }
            )

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise ValueError("No plottable metric values were found.")
    return long_df


def metric_order(long_df: pd.DataFrame, metric: str, higher_is_better: bool) -> list[str]:
    """Sort methods by per-metric median performance."""
    medians = long_df.loc[long_df["metric"] == metric].groupby("method")["score"].median()
    return medians.sort_values(ascending=not higher_is_better).index.tolist()


# %%
def draw_boxplot_panel(
    ax: plt.Axes,
    long_df: pd.DataFrame,
    spec: dict[str, object],
) -> None:
    metric = str(spec["label"])
    higher_is_better = bool(spec["higher_is_better"])
    sub = long_df.loc[long_df["metric"] == metric]
    order = metric_order(long_df, metric, higher_is_better)
    data = [sub.loc[sub["method"] == method, "score"].to_numpy() for method in order]

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        medianprops={"color": "#d97946", "linewidth": 1.2},
        boxprops={"linewidth": 0.8, "color": "#444444"},
        whiskerprops={"linewidth": 0.8, "color": "#444444"},
        capprops={"linewidth": 0.8, "color": "#444444"},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "white",
            "markeredgecolor": "#333333",
            "markeredgewidth": 0.6,
        },
    )

    for patch, method in zip(bp["boxes"], order):
        patch.set_facecolor(PALETTE.get(method, "#999999"))
        patch.set_alpha(0.85)

    # Jittered dot overlay — one dot per sample per method
    rng = np.random.default_rng(2026)
    for idx, method in enumerate(order):
        scores = data[idx]
        if len(scores) == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(scores))
        x_positions = np.full(len(scores), idx + 1) + jitter
        ax.scatter(
            x_positions, scores,
            s=18, c=PALETTE.get(method, "#999999"),
            edgecolors="white", linewidths=0.4,
            alpha=0.75, zorder=3,
        )

    ax.set_title(metric, fontsize=14, fontweight="bold", pad=9)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, axis="both", color="#dddddd", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#555555")

    scores = sub["score"].dropna()
    data_min = scores.min()
    data_max = scores.max()
    if spec.get("yscale") == "log":
        positive = scores[scores > 0]
        ax.set_yscale("log")
        ax.set_ylim(positive.min() * 0.65, positive.max() * 1.35)
    elif higher_is_better:
        span = max(data_max - data_min, abs(data_max) * 0.1, 1e-6)
        lower = data_min - 0.10 * span
        if data_min >= 0:
            lower = max(-0.02, lower)
        upper = data_max + 0.10 * span
        if data_max <= 1.0:
            upper = min(1.02, upper)
        ax.set_ylim(lower, upper)
    else:
        span = max(data_max - data_min, abs(data_max) * 0.1, 1e-6)
        lower = 0 if data_min >= 0 else data_min - 0.15 * span
        ax.set_ylim(lower, data_max + 0.15 * span)


def _style_params():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def draw_consolidated_figure(long_df: pd.DataFrame, output_dir: Path, output_stem: str, dpi: int) -> None:
    """Draw and save the 3 x 3 benchmark figure (8 metrics + legend)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _style_params()

    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    axes = axes.flatten()
    panel_axes = axes[:14]
    legend_ax = axes[14]
    axes[15].axis("off")

    for ax, spec in zip(panel_axes, METRIC_SPECS):
        draw_boxplot_panel(ax, long_df, spec)

    for row_start in [0, 4, 8, 12]:
        axes[row_start].set_ylabel("Score", fontsize=13)
    for axis_index in range(14):
        if axis_index not in {0, 4, 8, 12}:
            axes[axis_index].set_ylabel("")

    legend_ax.axis("off")
    plotted_methods = [m for m in PALETTE if m in set(long_df["method"])]
    legend_handles = [
        Patch(facecolor=PALETTE[method], edgecolor="none", label=method, alpha=0.85)
        for method in plotted_methods
    ]
    legend = legend_ax.legend(
        handles=legend_handles,
        loc="center left",
        frameon=True,
        fontsize=11,
        borderpad=0.6,
        handlelength=0.8,
        handletextpad=0.5,
        title="Method",
        title_fontsize=12,
    )
    legend.get_frame().set_edgecolor("#cccccc")
    legend.get_frame().set_linewidth(0.8)

    n_samples = long_df[["method", "sample"]].drop_duplicates().groupby("method").size()
    note = "Boxplots summarize datasets with available metrics."
    if not n_samples.empty:
        note += f"\nFEAST modes: n={int(n_samples.min())}-{int(n_samples.max())} datasets."
    legend_ax.text(0.0, 0.23, note, fontsize=9, color="#555555", va="top")

    fig.tight_layout(w_pad=2.6, h_pad=2.4)

    for suffix, kwargs in {"pdf": {}, "svg": {}, "png": {"dpi": dpi}}.items():
        path = output_dir / f"{output_stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"Saved: {path}")

    plt.close(fig)


def draw_individual_subplots(long_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Draw one standalone boxplot figure per metric (no legend)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in METRIC_SPECS:
        _style_params()
        fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.2))
        draw_boxplot_panel(ax, long_df, spec)
        ax.set_ylabel("Score", fontsize=12)

        safe_name = spec["label"].replace(" ", "_").replace("↑", "up").replace("↓", "down")
        for suffix, kwargs in {"pdf": {}, "svg": {}, "png": {"dpi": dpi}}.items():
            path = output_dir / f"subplot_{safe_name}.{suffix}"
            fig.savefig(path, bbox_inches="tight", **kwargs)
            print(f"Saved: {path}")

        plt.close(fig)


def draw_figure(long_df: pd.DataFrame, output_dir: Path, output_stem: str, dpi: int) -> None:
    draw_consolidated_figure(long_df, output_dir, output_stem, dpi)
    draw_individual_subplots(long_df, output_dir, dpi)


# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=METRICS_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-stem", default=OUTPUT_STEM)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    metrics = load_metrics(args.metrics_csv)
    long_df = build_long_dataframe(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.output_dir / f"{args.output_stem}_long_metrics.csv"
    summary_path = args.output_dir / f"{args.output_stem}_metric_medians.csv"

    long_df.to_csv(long_path, index=False)
    (
        long_df.groupby(["metric", "method"], as_index=False)["score"]
        .median()
        .rename(columns={"score": "median_score"})
        .to_csv(summary_path, index=False)
    )
    print(f"Saved: {long_path}")
    print(f"Saved: {summary_path}")

    draw_figure(long_df, args.output_dir, args.output_stem, args.dpi)
    return 0


# %%
if __name__ == "__main__":
    raise SystemExit(main())
