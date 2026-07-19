#!/usr/bin/env python3
"""Build publication panels for the clean Study 00 simulator benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_CSV = (
    REPOSITORY_ROOT
    / "00_simulator_benchmark"
    / "outputs"
    / "final_rerun_20260718_metrics_v2"
    / "simulator_quality_metrics.csv"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
DEFAULT_OUTPUT_STEM = "simulator_benchmark_boxplots"

METHOD_LABELS = {
    "FEAST_reference_rank": "FEAST Rank",
    "SRTsim": "SRTsim",
    "Splatter": "Splatter",
    "Splatter_Simple": "Splatter Simple",
    "scCube": "scCube",
}
METHOD_ORDER = ["FEAST Rank", "SRTsim", "Splatter", "Splatter Simple", "scCube"]
PALETTE = {
    "FEAST Rank": "#7B5BA7",
    "SRTsim": "#E75A9C",
    "Splatter": "#F28E73",
    "Splatter Simple": "#7EA6D8",
    "scCube": "#E7C083",
}

METRIC_SPECS = [
    {"column": "input_mean_corr", "label": "Mean Corr ↑", "higher_is_better": True},
    {"column": "input_variance_corr", "label": "Var Corr ↑", "higher_is_better": True},
    {"column": "moran_i_correlation", "label": "Moran I Corr ↑", "higher_is_better": True},
    {"column": "zero_mask_jaccard", "label": "Zero Jaccard ↑", "higher_is_better": True},
    {"column": "cosine_divergence", "label": "Cosine Div ↓", "higher_is_better": False},
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
        "yscale": "log",
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def load_metrics(metrics_csv: Path) -> pd.DataFrame:
    """Load the canonical atomic table and enforce its clean-run design."""
    if not metrics_csv.is_file():
        raise FileNotFoundError(f"Metric table does not exist: {metrics_csv}")

    metrics = pd.read_csv(metrics_csv)
    required = {"simulator", "sample", "identity_flag"} | {
        str(spec["column"]) for spec in METRIC_SPECS
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"Missing required columns in {metrics_csv}: {missing}")

    duplicate_mask = metrics.duplicated(["simulator", "sample"], keep=False)
    if duplicate_mask.any():
        duplicate_keys = (
            metrics.loc[duplicate_mask, ["simulator", "sample"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(f"Duplicate simulator/sample rows: {duplicate_keys}")

    observed_simulators = set(metrics["simulator"].dropna())
    expected_simulators = set(METHOD_LABELS)
    if observed_simulators != expected_simulators:
        missing_simulators = sorted(expected_simulators - observed_simulators)
        unexpected_simulators = sorted(observed_simulators - expected_simulators)
        raise ValueError(
            "Metric table does not match the clean five-simulator design: "
            f"missing={missing_simulators}, unexpected={unexpected_simulators}"
        )

    sample_counts = metrics.groupby("simulator")["sample"].nunique()
    if len(set(sample_counts)) != 1 or int(sample_counts.iloc[0]) != 12:
        raise ValueError(f"Expected 12 samples per simulator, found {sample_counts.to_dict()}")

    metrics = metrics.copy()
    metrics["method"] = metrics["simulator"].map(METHOD_LABELS)
    return metrics


def build_long_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in METRIC_SPECS:
        column = str(spec["column"])
        subset = metrics[["method", "sample", "identity_flag", column]].dropna(subset=[column])
        for row in subset.itertuples(index=False):
            rows.append(
                {
                    "metric": spec["label"],
                    "method": row.method,
                    "sample": row.sample,
                    "score": float(getattr(row, column)),
                    "identity_flag": bool(row.identity_flag),
                    "source_column": column,
                    "higher_is_better": bool(spec["higher_is_better"]),
                }
            )
    if not rows:
        raise ValueError("The metric table contains no plottable values.")
    return pd.DataFrame(rows)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "feast-study00-publication",
        }
    )


def draw_boxplot_panel(ax: plt.Axes, long_table: pd.DataFrame, spec: dict[str, object]) -> None:
    label = str(spec["label"])
    subset = long_table.loc[long_table["metric"] == label]
    method_rows = [
        subset.loc[subset["method"] == method].sort_values("sample")
        for method in METHOD_ORDER
    ]
    plotted = [
        (position, method, rows)
        for position, (method, rows) in enumerate(zip(METHOD_ORDER, method_rows), start=1)
        if not rows.empty
    ]

    boxplot = ax.boxplot(
        [rows["score"].to_numpy() for _, _, rows in plotted],
        positions=[position for position, _, _ in plotted],
        patch_artist=True,
        widths=0.52,
        showfliers=False,
        medianprops={"color": "#D97946", "linewidth": 1.2},
        boxprops={"linewidth": 0.8, "color": "#444444"},
        whiskerprops={"linewidth": 0.8, "color": "#444444"},
        capprops={"linewidth": 0.8, "color": "#444444"},
    )
    for patch, (_, method, _) in zip(boxplot["boxes"], plotted):
        patch.set_facecolor(PALETTE[method])
        patch.set_alpha(0.72)

    for position, method, rows in plotted:
        scores = rows["score"].to_numpy()
        jitter = np.zeros(1) if len(scores) == 1 else np.linspace(-0.13, 0.13, len(scores))
        x_positions = np.full(len(scores), position) + jitter
        identity = rows["identity_flag"].to_numpy(dtype=bool)
        ordinary = ~identity
        color = PALETTE[method]
        if ordinary.any():
            ax.scatter(
                x_positions[ordinary],
                scores[ordinary],
                s=20,
                c=color,
                edgecolors="white",
                linewidths=0.4,
                alpha=0.9,
                zorder=3,
            )
        if identity.any():
            ax.scatter(
                x_positions[identity],
                scores[identity],
                s=25,
                facecolors="none",
                edgecolors=color,
                marker="D",
                linewidths=1.0,
                zorder=4,
            )

    for position, rows in enumerate(method_rows, start=1):
        if rows.empty:
            ax.text(
                position,
                0.04,
                "NA",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#777777",
            )

    ax.set_xlim(0.5, len(METHOD_ORDER) + 0.5)
    ax.set_title(label, fontsize=11.5, fontweight="bold", pad=7)
    ax.set_xticks(range(1, len(METHOD_ORDER) + 1))
    ax.set_xticklabels(METHOD_ORDER, rotation=36, ha="right", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#555555")

    scores = subset["score"].dropna()
    data_min = float(scores.min())
    data_max = float(scores.max())
    if spec.get("yscale") == "log":
        positive = scores[scores > 0]
        if positive.empty:
            raise ValueError(f"Log-scale metric {label} has no positive values")
        ax.set_yscale("log")
        ax.set_ylim(float(positive.min()) * 0.65, float(positive.max()) * 1.35)
    elif bool(spec["higher_is_better"]):
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


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    paths: list[Path] = []
    for suffix, kwargs in {"pdf": {}, "svg": {}, "png": {"dpi": dpi}}.items():
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"Saved: {path}")
        paths.append(path)
    return paths


def draw_consolidated_figure(
    long_table: pd.DataFrame,
    output_dir: Path,
    output_stem: str,
    dpi: int,
) -> list[Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(3, 4, figsize=(16, 11.2))
    axes = axes.flatten()

    for panel_index, (ax, spec) in enumerate(zip(axes[:10], METRIC_SPECS)):
        draw_boxplot_panel(ax, long_table, spec)
        ax.text(
            -0.13,
            1.06,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    for row_start in (0, 4, 8):
        axes[row_start].set_ylabel("Metric value", fontsize=13)

    legend_ax = axes[10]
    legend_ax.axis("off")
    method_handles = [
        Patch(facecolor=PALETTE[method], edgecolor="none", label=method, alpha=0.85)
        for method in METHOD_ORDER
    ]
    method_legend = legend_ax.legend(
        handles=method_handles,
        loc="upper left",
        frameon=True,
        fontsize=9,
        borderpad=0.6,
        handlelength=0.8,
        handletextpad=0.5,
        title="Simulator",
        title_fontsize=10,
    )
    method_legend.get_frame().set_edgecolor("#CCCCCC")
    method_legend.get_frame().set_linewidth(0.8)

    status_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#777777",
            markeredgecolor="white",
            markersize=6,
            label="Ordinary",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#777777",
            markersize=6,
            label="Near identity",
        ),
    ]
    legend_ax.legend(
        handles=status_handles,
        loc="lower left",
        frameon=False,
        fontsize=8.5,
        title="Dataset point",
        title_fontsize=9.5,
    )
    legend_ax.add_artist(method_legend)

    notes_ax = axes[11]
    notes_ax.axis("off")
    n_datasets = long_table["sample"].nunique()
    notes_ax.text(
        0.02,
        0.98,
        f"{n_datasets} datasets per simulator\n\n"
        "Boxes summarize available values;\n"
        "each point is one dataset. NA marks\n"
        "an unavailable pairwise metric.\n\n"
        "Near identity requires all three:\n"
        "mean corr ≥ 0.995\n"
        "variance corr ≥ 0.95\n"
        "zero-mask Jaccard ≥ 0.95\n\n"
        "Historical FEAST OT is outside\n"
        "the clean rerun scope.",
        transform=notes_ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4F4F4F",
        linespacing=1.35,
    )

    fig.suptitle(
        f"Simulator benchmark across {n_datasets} spatial transcriptomics datasets",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98), w_pad=2.0, h_pad=2.2)
    paths = save_figure(fig, output_dir, output_stem, dpi)
    plt.close(fig)
    return paths


def draw_individual_panels(long_table: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    paths: list[Path] = []
    for spec in METRIC_SPECS:
        configure_matplotlib()
        fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.2))
        draw_boxplot_panel(ax, long_table, spec)
        ax.set_ylabel("Metric value", fontsize=12)
        safe_name = (
            str(spec["label"])
            .replace(" ", "_")
            .replace("↑", "up")
            .replace("↓", "down")
        )
        paths.extend(save_figure(fig, output_dir, f"panel_{safe_name}", dpi))
        plt.close(fig)
    return paths


def write_provenance(
    metrics_csv: Path,
    metrics: pd.DataFrame,
    output_dir: Path,
    output_paths: list[Path],
) -> Path:
    metrics_provenance = metrics_csv.parent / "provenance.json"
    payload = {
        "schema_version": 1,
        "figure_set": "study00_simulator_benchmark",
        "source": {
            "metrics_csv": repository_path(metrics_csv),
            "metrics_sha256": sha256(metrics_csv),
            "metrics_provenance": (
                repository_path(metrics_provenance) if metrics_provenance.is_file() else None
            ),
            "metrics_provenance_sha256": (
                sha256(metrics_provenance) if metrics_provenance.is_file() else None
            ),
            "plot_script": repository_path(Path(__file__)),
            "plot_script_sha256": sha256(Path(__file__)),
        },
        "design": {
            "atomic_rows": int(len(metrics)),
            "samples": sorted(metrics["sample"].unique().tolist()),
            "simulators": METHOD_ORDER,
            "historical_feast_ot_included": False,
        },
        "outputs": {
            path.name: sha256(path)
            for path in sorted(output_paths, key=lambda item: item.name)
        },
    }
    path = output_dir / "figure_provenance.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = load_metrics(args.metrics_csv)
    long_table = build_long_table(metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    long_path = args.output_dir / "plot_data_long.csv"
    medians_path = args.output_dir / "metric_medians.csv"
    long_table.to_csv(long_path, index=False)
    (
        long_table.groupby(["metric", "method"], as_index=False)["score"]
        .median()
        .rename(columns={"score": "median_score"})
        .to_csv(medians_path, index=False)
    )
    print(f"Saved: {long_path}")
    print(f"Saved: {medians_path}")

    output_paths = [long_path, medians_path]
    output_paths.extend(
        draw_consolidated_figure(long_table, args.output_dir, args.output_stem, args.dpi)
    )
    output_paths.extend(draw_individual_panels(long_table, args.output_dir, args.dpi))
    write_provenance(args.metrics_csv, metrics, args.output_dir, output_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
