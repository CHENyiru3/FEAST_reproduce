"""Render Study 07 coverage and FEAST-versus-resampling continuity."""

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
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from plot import EVALUATION_ROOT, STUDY_ROOT, coverage_for_plot, style_axis


BASELINE_ROOT = (
    STUDY_ROOT / "outputs" / "final" / "baselines" / "conditional_whole_spot_resampling"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "candidate_figures" / "conditional_resampling"
)
SOURCE_METRICS = {
    "gene_mean_discontinuity": "gene_mean_pearson",
    "region_gene_mean_discontinuity": "region_gene_mean_pearson",
}
REGION_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#666666",
    "#B9B9B9",
]
CONTINUITY_COLORS = {
    "gene_mean_discontinuity": "#0072B2",
    "region_gene_mean_discontinuity": "#E69F00",
}
CONTINUITY_LABELS = {
    "gene_mean_discontinuity": "Gene mean",
    "region_gene_mean_discontinuity": "Region–gene mean",
}


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


def spot_tick(value: float, _position: float) -> str:
    if value == 0:
        return "0"
    return f"{value / 1000:g}k"


def plot_coverage(
    axis: plt.Axes,
    frame: pd.DataFrame,
    region_order: list[str],
    age_summary: pd.Series,
) -> None:
    age = str(age_summary.name)
    age_frame = frame[frame["age"] == age]
    pivot = age_frame.pivot_table(
        index="z_world", columns="display_region", values="n_spots", fill_value=0
    ).sort_index()
    x = pivot.index.to_numpy(dtype=float)
    series = [
        pivot[region].to_numpy(dtype=float) if region in pivot else np.zeros(len(pivot))
        for region in region_order
    ]
    axis.stackplot(
        x,
        series,
        colors=REGION_COLORS[: len(region_order)],
        linewidth=0,
        alpha=0.97,
    )
    axis.plot(x, np.sum(np.vstack(series), axis=0), color="#333333", linewidth=0.8)
    axis.set_title(f"{age} regional support", fontsize=15, fontweight="bold", pad=9)
    axis.set_ylabel("Blueprint spots per z level")
    axis.set_xlabel("DevCCF z coordinate")
    axis.set_xlim(float(age_summary["z_start"]), float(age_summary["z_end"]))
    axis.set_ylim(bottom=0)
    axis.yaxis.set_major_formatter(FuncFormatter(spot_tick))
    axis.text(
        0.98,
        0.95,
        f"{int(age_summary['n_z_levels'])} levels  ·  {age_summary['n_spots'] / 1_000_000:.2f}M spots",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color="#4F4F4F",
    )
    style_axis(axis)


def baseline_summary(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    for output, source in SOURCE_METRICS.items():
        working[output] = np.maximum(1.0 - working[source].to_numpy(dtype=float), 1e-12)
    keys = ["age", "lower_z_index", "upper_z_index", "lower_z", "upper_z", "midpoint_z"]
    rows = []
    for values, group in working.groupby(keys, sort=False):
        row = dict(zip(keys, values, strict=True))
        for metric in SOURCE_METRICS:
            observed = group[metric].to_numpy(dtype=float)
            row[f"{metric}_median"] = float(np.median(observed))
            row[f"{metric}_p05"] = float(np.quantile(observed, 0.05))
            row[f"{metric}_p95"] = float(np.quantile(observed, 0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_continuity(
    axis: plt.Axes,
    feast: pd.DataFrame,
    baseline: pd.DataFrame,
    age_summary: pd.Series,
) -> None:
    age = str(age_summary.name)
    feast_age = feast[feast["age"] == age].sort_values("midpoint_z")
    baseline_age = baseline[baseline["age"] == age].sort_values("midpoint_z")
    for metric, linestyle in (
        ("gene_mean_discontinuity", "-"),
        ("region_gene_mean_discontinuity", (0, (3.2, 1.8))),
    ):
        color = CONTINUITY_COLORS[metric]
        x = baseline_age["midpoint_z"].to_numpy(dtype=float)
        axis.fill_between(
            x,
            baseline_age[f"{metric}_p05"],
            baseline_age[f"{metric}_p95"],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        axis.plot(
            x,
            baseline_age[f"{metric}_median"],
            color=color,
            linewidth=0.75,
            linestyle=linestyle,
            alpha=0.48,
        )
        axis.plot(
            feast_age["midpoint_z"],
            feast_age[metric],
            color=color,
            linewidth=1.15,
            linestyle=linestyle,
        )
    axis.set_title(
        f"{age} adjacent-z continuity", fontsize=15, fontweight="bold", pad=9
    )
    axis.set_ylabel("Discontinuity, Δ = 1 − Pearson r\n(log scale; lower is smoother)")
    axis.set_xlabel("Adjacent-slice midpoint z")
    axis.set_xlim(float(age_summary["z_start"]), float(age_summary["z_end"]))
    axis.set_yscale("log")
    displayed = np.concatenate([
        feast_age[list(SOURCE_METRICS)].to_numpy(dtype=float).ravel(),
        baseline_age[[f"{metric}_{stat}" for metric in SOURCE_METRICS for stat in ("p05", "p95")]].to_numpy(dtype=float).ravel(),
    ])
    positive = displayed[np.isfinite(displayed) & (displayed > 0)]
    axis.set_ylim(positive.min() * .5, positive.max() * 2)
    style_axis(axis)


def main() -> None:
    global EVALUATION_ROOT, BASELINE_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    EVALUATION_ROOT = args.input_root.resolve() / "evaluation"
    BASELINE_ROOT = args.baseline_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "full_axis_transfer_diagnostic"
    targets = [stem.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png")]
    targets += [
        output / "region_coverage_plot.csv",
        output / "feast_adjacent_z_plot.csv",
        output / "resampling_adjacent_z_plot.csv",
        output / "figure_provenance.json",
    ]
    if any(path.exists() for path in targets):
        raise FileExistsError("fresh-only Study 07 candidate figure already exists")

    coverage = pd.read_csv(EVALUATION_ROOT / "region_coverage.csv")
    feast = pd.read_csv(EVALUATION_ROOT / "adjacent_z_continuity.csv")
    age_summary = pd.read_csv(EVALUATION_ROOT / "age_summary.csv").set_index("age")
    baseline = baseline_summary(
        pd.read_csv(BASELINE_ROOT / "adjacent_z_continuity.csv")
    )
    coverage_plot, region_order = coverage_for_plot(coverage)
    for output_metric, source_metric in SOURCE_METRICS.items():
        feast[output_metric] = np.maximum(
            1.0 - feast[source_metric].to_numpy(dtype=float), 1e-12
        )
    coverage_plot.to_csv(output / "region_coverage_plot.csv", index=False)
    feast.to_csv(output / "feast_adjacent_z_plot.csv", index=False)
    baseline.to_csv(output / "resampling_adjacent_z_plot.csv", index=False)

    configure_matplotlib()
    figure = plt.figure(figsize=(14.5, 9.5))
    grid = figure.add_gridspec(
        2, 2, left=0.07, right=0.985, bottom=0.18, top=0.88, hspace=0.43, wspace=0.25
    )
    axes = np.asarray(
        [
            [figure.add_subplot(grid[row, column]) for column in range(2)]
            for row in range(2)
        ]
    )
    for row, age in enumerate(("E15.5", "E18.5")):
        plot_coverage(
            axes[row, 0],
            coverage_plot,
            region_order,
            age_summary.loc[age],
        )
        plot_continuity(axes[row, 1], feast, baseline, age_summary.loc[age])
    region_handles = [
        Patch(facecolor=color, edgecolor="none", label=region)
        for region, color in zip(
            region_order,
            REGION_COLORS,
            strict=False,
        )
    ]
    continuity_handles = [
        Line2D(
            [],
            [],
            color=CONTINUITY_COLORS[metric],
            linestyle=linestyle,
            linewidth=1.3,
            label=CONTINUITY_LABELS[metric],
        )
        for metric, linestyle in (
            ("gene_mean_discontinuity", "-"),
            ("region_gene_mean_discontinuity", (0, (3.2, 1.8))),
        )
    ]
    continuity_handles.append(
        Patch(
            facecolor="#999999",
            alpha=0.18,
            edgecolor="none",
            label="Conditional resampling 5–95% range",
        )
    )
    legend_one = figure.legend(
        handles=region_handles,
        loc="lower center",
        ncol=min(9, len(region_handles)),
        bbox_to_anchor=(0.5, 0.025),
        frameon=False,
        handlelength=0.9,
        handletextpad=0.45,
    )
    figure.add_artist(legend_one)
    figure.legend(
        handles=continuity_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.075),
        frameon=False,
        handletextpad=0.5,
    )
    figure.suptitle(
        "Full-axis transfer coverage and continuity control",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.955,
        "FEAST continuity is compared with ten age- and region-conditioned resampling runs; the expression-free blueprint cannot evaluate biological accuracy.",
        ha="center",
        va="top",
        fontsize=12,
        color="#4F4F4F",
    )
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
        "configuration_id": json.loads((EVALUATION_ROOT / "provenance.json").read_text())["configuration_id"],
        "inputs": {"evaluation": str(EVALUATION_ROOT), "conditional_resampling": str(BASELINE_ROOT)},
        "resampling_reuse": "existing ten-run all-eligible-reference control; checked against current target coverage, panel and regional reference populations",
        "status": "candidate_verified_by_script",
        "study": "07_3d_transfer",
        "interpretation": "regional coverage and descriptive continuity relative to conditional resampling",
        "target_expression_exists": False,
        "accuracy_claim_authorized": False,
        "resampling_interval": "empirical_5th_to_95th_percentile_not_a_confidence_interval",
        "publication_canonical": False,
        "visual_style_reference": "visualization/00_simulator_benchmark/plot.py",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "figure_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote Study 07 candidate continuity figure to {output}")


if __name__ == "__main__":
    main()
