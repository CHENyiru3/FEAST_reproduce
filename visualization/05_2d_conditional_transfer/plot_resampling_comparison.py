"""Render the Study 05 FEAST-versus-resampling metric comparison."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPRO_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = REPRO_ROOT / "05_2d_conditional_transfer"
METRIC_ROOT = STUDY_ROOT / "outputs" / "baselines" / "conditional_metric_comparison"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "candidate_figures" / "conditional_resampling"
)
METRICS = (
    (
        "conditional_expression_wasserstein",
        "Conditional expression\nWasserstein ↓",
    ),
    (
        "conditional_zero_fraction_wasserstein",
        "Conditional zero-fraction\nWasserstein ↓",
    ),
    ("moran_profile_correlation", "Moran I profile\ncorrelation ↑"),
    (
        "residual_moran_profile_correlation",
        "Label-residual Moran I\ncorrelation ↑",
    ),
)
GROUPS = (
    ("dlpfc", "cross_slice", "DLPFC\ncross-slice"),
    ("dlpfc", "mask_half", "DLPFC\nhalf-slice"),
    ("merfish", "cross_slice", "MERFISH\ncross-slice"),
    ("merfish", "mask_half", "MERFISH\nhalf-slice"),
)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def task_label(row: pd.Series) -> str:
    if row["dataset"] == "dlpfc":
        if row["mode"] == "mask_half":
            return f"D {str(row['source'])[-3:]} half"
        return f"D {str(row['source'])[-3:]}→{str(row['target'])[-3:]}"
    if row["mode"] == "mask_half":
        return f"M {str(row['source'])[-3:]} half"
    return f"M {str(row['source'])[-3:]}→{str(row['target'])[-3:]}"


def task_order(frame: pd.DataFrame) -> pd.DataFrame:
    tasks = frame[
        list(("mode", "dataset", "direction", "source", "target"))
    ].drop_duplicates()
    tasks["dataset_order"] = tasks["dataset"].map({"dlpfc": 0, "merfish": 1})
    tasks["mode_order"] = tasks["mode"].map({"cross_slice": 0, "mask_half": 1})
    tasks = tasks.sort_values(
        ["dataset_order", "mode_order", "source", "target", "direction"]
    ).reset_index(drop=True)
    tasks["task_index"] = np.arange(len(tasks))
    tasks["task_label"] = tasks.apply(task_label, axis=1)
    return tasks


def plot_metric(axis: plt.Axes, frame: pd.DataFrame, metric: str, title: str) -> None:
    metric_rows = frame[frame["metric"] == metric].sort_values("task_index")
    for group_index, (dataset, mode, _label) in enumerate(GROUPS):
        rows = metric_rows[
            (metric_rows["dataset"] == dataset) & (metric_rows["mode"] == mode)
        ]
        jitter = (
            np.zeros(1) if len(rows) == 1 else np.linspace(-0.055, 0.055, len(rows))
        )
        baseline_x = np.full(len(rows), group_index - 0.14) + jitter
        feast_x = np.full(len(rows), group_index + 0.14) + jitter
        for left_x, right_x, left_y, right_y in zip(
            baseline_x,
            feast_x,
            rows["baseline_mean"],
            rows["feast"],
            strict=True,
        ):
            axis.plot(
                [left_x, right_x],
                [left_y, right_y],
                color="#B9B9B9",
                linewidth=0.8,
                alpha=0.75,
                zorder=1,
            )
        axis.scatter(
            baseline_x,
            rows["baseline_mean"],
            s=30,
            facecolor="#999999",
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        axis.scatter(
            feast_x,
            rows["feast"],
            s=38,
            facecolor="#0072B2",
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        axis.hlines(
            np.median(rows["baseline_mean"]),
            group_index - 0.24,
            group_index - 0.04,
            color="#D97946",
            linewidth=1.3,
            zorder=4,
        )
        axis.hlines(
            np.median(rows["feast"]),
            group_index + 0.04,
            group_index + 0.24,
            color="#D97946",
            linewidth=1.3,
            zorder=4,
        )
    axis.set_title(title, fontsize=15, fontweight="bold", pad=9)
    axis.set_xticks(
        np.arange(len(GROUPS)), [label for _dataset, _mode, label in GROUPS]
    )
    axis.set_xlim(-0.5, len(GROUPS) - 0.5)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.65)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for spine_name in ("left", "bottom"):
        axis.spines[spine_name].set_linewidth(0.8)
        axis.spines[spine_name].set_color("#555555")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=METRIC_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "conditional_transfer_distribution_and_spatial_fidelity"
    targets = [stem.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png")]
    targets += [output / "plot_data.csv", output / "figure_provenance.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError("fresh-only Study 05 candidate figure already exists")

    comparison = pd.read_csv(
        args.input_dir / "feast_comparison.csv", dtype={"source": str, "target": str}
    )
    tasks = task_order(comparison)
    plot_data = comparison.merge(
        tasks.drop(columns=["dataset_order", "mode_order"]),
        on=["mode", "dataset", "direction", "source", "target"],
        validate="many_to_one",
    )
    plot_data.to_csv(output / "plot_data.csv", index=False)

    configure_matplotlib()
    figure, axes = plt.subplots(1, 4, figsize=(18.0, 5.4))
    for axis, (metric, title) in zip(axes, METRICS, strict=True):
        plot_metric(axis, plot_data, metric, title)
    handles = [
        Line2D([], [], marker="o", linestyle="None", color="#0072B2", label="FEAST"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            color="#999999",
            markerfacecolor="#999999",
            label="Conditional resampling mean",
        ),
    ]
    axes[0].set_ylabel("Metric value", fontsize=16)
    figure.suptitle(
        "Conditional transfer vs empirical resampling",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.025),
        frameon=False,
        handlelength=1.0,
        handletextpad=0.5,
    )
    figure.text(
        0.5,
        0.91,
        "Paired task values; each baseline point is the ten-run resampling mean and orange ticks mark group medians.",
        ha="center",
        va="top",
        fontsize=12,
        color="#4F4F4F",
    )
    figure.tight_layout(rect=(0.015, 0.17, 1, 0.80), w_pad=1.4)
    figure.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={
            "CreationDate": None,
            "ModDate": None,
            "Creator": "FEAST publication visualization",
        },
    )
    figure.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "FEAST publication visualization"},
    )
    figure.savefig(
        stem.with_suffix(".png"),
        dpi=args.dpi,
        bbox_inches="tight",
        metadata={"Software": "FEAST publication visualization"},
    )
    plt.close(figure)

    provenance = {
        "status": "candidate_verified_by_script",
        "study": "05_2d_conditional_transfer",
        "interpretation": "conditional distribution and spatial-profile similarity relative to whole-spot resampling",
        "exact_spot_metrics_displayed": False,
        "resampling_interval": "empirical_5th_to_95th_percentile_not_a_confidence_interval",
        "publication_canonical": False,
        "visual_style_reference": "visualization/00_simulator_benchmark/plot.py",
        "visual_layout": "one_by_four_paired_task_dots_grouped_by_dataset_and_transfer_mode",
        "inputs": [str(args.input_dir / "feast_comparison.csv")],
        "outputs": [path.name for path in targets[:-1]],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "figure_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote Study 05 candidate comparison figure to {output}")


if __name__ == "__main__":
    main()
