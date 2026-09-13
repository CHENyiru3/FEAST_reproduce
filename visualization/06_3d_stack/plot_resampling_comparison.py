"""Render the Study 06 FEAST-versus-resampling metric comparison."""

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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


REPRO_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = REPRO_ROOT / "06_3d_stack"
METRIC_ROOT = (
    STUDY_ROOT / "outputs" / "reference_density_rank_v1" / "baselines" / "conditional_metric_comparison"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "candidate_figures" / "conditional_resampling"
)
METRICS = (
    ("conditional_expression_wasserstein", "Conditional expression\nWasserstein ↓"),
    (
        "conditional_zero_fraction_wasserstein",
        "Conditional zero-fraction\nWasserstein ↓",
    ),
    ("moran_profile_correlation", "Moran I profile\ncorrelation ↑"),
    ("residual_moran_profile_correlation", "Label-residual Moran I\ncorrelation ↑"),
)
TARGET_KEYS = ("density_name", "gap", "target_index", "target_slice", "target_z")


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


def panel(axis: plt.Axes, frame: pd.DataFrame, metric: str, title: str) -> None:
    gaps = [3, 5, 10]
    positions = np.arange(len(gaps), dtype=float)
    colors = {"feast": "#0072B2", "conditional_whole_spot_resampling": "#999999"}
    feast = frame[frame["method"] == "feast"]
    baseline = (
        frame[frame["method"] == "conditional_whole_spot_resampling"]
        .groupby(list(TARGET_KEYS), as_index=False)[metric]
        .mean()
    )
    method_frames = {
        "conditional_whole_spot_resampling": baseline,
        "feast": feast,
    }
    for method, offset in (
        ("conditional_whole_spot_resampling", -0.14),
        ("feast", 0.14),
    ):
        data = [
            method_frames[method][method_frames[method]["gap"] == gap][metric].to_numpy(
                dtype=float
            )
            for gap in gaps
        ]
        box = axis.boxplot(
            data,
            positions=positions + offset,
            widths=0.22,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#D97946", "linewidth": 1.2},
            whiskerprops={"color": "#444444", "linewidth": 0.8},
            capprops={"color": "#444444", "linewidth": 0.8},
            boxprops={"edgecolor": "#444444", "linewidth": 0.8},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(colors[method])
            patch.set_alpha(0.72)
        for gap_index, values in enumerate(data):
            jitter = (
                np.zeros(1)
                if len(values) == 1
                else np.linspace(-0.07, 0.07, len(values))
            )
            axis.scatter(
                np.full(len(values), positions[gap_index] + offset) + jitter,
                values,
                s=20,
                c=colors[method],
                edgecolors="white",
                linewidths=0.4,
                alpha=0.88,
                zorder=3,
            )
    axis.set_title(title, fontsize=15, fontweight="bold", pad=9)
    axis.set_xticks(positions, ["3", "5", "10"])
    axis.set_xlabel("Reference gap")
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
    stem = output / "conditional_stack_distribution_and_spatial_fidelity"
    targets = [stem.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png")]
    targets += [output / "plot_data.csv", output / "figure_provenance.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError("fresh-only Study 06 candidate figure already exists")

    metrics = pd.read_csv(args.input_dir / "method_metrics.csv")
    metrics.to_csv(output / "plot_data.csv", index=False)
    configure_matplotlib()
    figure, axes = plt.subplots(1, 4, figsize=(18.0, 5.4))
    for axis, (metric, title) in zip(axes, METRICS, strict=True):
        panel(axis, metrics, metric, title)
    handles = [
        Patch(facecolor="#0072B2", edgecolor="none", alpha=0.85, label="FEAST"),
        Patch(
            facecolor="#999999",
            edgecolor="none",
            alpha=0.85,
            label="Conditional resampling",
        ),
    ]
    axes[0].set_ylabel("Metric value", fontsize=16)
    figure.suptitle(
        "Conditional 3D stack vs empirical resampling",
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
        handlelength=0.9,
        handletextpad=0.5,
    )
    figure.text(
        0.5,
        0.91,
        "Matched target distributions; each baseline point is the ten-run resampling mean and ordered z targets are not biological replicates.",
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
        "study": "06_3d_stack",
        "interpretation": "conditional distribution and spatial-profile similarity relative to whole-spot resampling",
        "exact_spot_metrics_displayed": False,
        "targets_are_biological_replicates": False,
        "publication_canonical": False,
        "visual_style_reference": "visualization/00_simulator_benchmark/plot.py",
        "visual_layout": "one_by_four_matched_target_boxplots_with_resampling_means",
        "resampling_display_unit": "per_target_mean_across_ten_runs",
        "inputs": [str(args.input_dir / "method_metrics.csv")],
        "outputs": [path.name for path in targets[:-1]],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "figure_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote Study 06 candidate comparison figure to {output}")


if __name__ == "__main__":
    main()
