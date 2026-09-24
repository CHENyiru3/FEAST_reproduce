"""Render the Study 06 FEAST, five-reference resampling and SpatialZ comparison."""

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
from plot_results import configure_style
from plot_spatialz_comparison import load_comparison, COLORS, NAMES, METHODS


REPRO_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = REPRO_ROOT / "06_3d_stack"
METRIC_ROOT = (
    STUDY_ROOT / "outputs" / "generative_five_reference_1.0.6_precision_v1"
    / "baselines" / "five_reference_conditional_resampling"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "figures" / "three_method_resampling_spatialz_v1"
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
    configure_style()


def panel(axis: plt.Axes, frame: pd.DataFrame, metric: str, title: str) -> None:
    gaps = [3, 5, 10]
    positions = np.arange(len(gaps), dtype=float)
    if metric.startswith('conditional_'):
        supported = frame.pivot(index=['gap','target_slice'], columns='method', values=metric).dropna().index
        frame = frame.set_index(['gap','target_slice']).loc[lambda x: x.index.isin(supported)].reset_index()
    method_frames = {method: frame[frame.method == method] for method in METHODS}
    for method, offset in zip(METHODS, (-.24, 0, .24)):
        data = [
            method_frames[method][method_frames[method]["gap"] == gap][metric].to_numpy(
                dtype=float
            )
            for gap in gaps
        ]
        box = axis.boxplot(
            data,
            positions=positions + offset,
            widths=0.19,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#24313F", "linewidth": 1.5},
            whiskerprops={"color": "#444444", "linewidth": 0.8},
            capprops={"color": "#444444", "linewidth": 0.8},
            boxprops={"edgecolor": "#444444", "linewidth": 0.8},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.20)
            patch.set_edgecolor(COLORS[method])
        for gap_index, values in enumerate(data):
            jitter = (
                np.zeros(1)
                if len(values) == 1
                else np.linspace(-0.07, 0.07, len(values))
            )
            axis.scatter(
                np.full(len(values), positions[gap_index] + offset) + jitter,
                values,
                s=12,
                c=COLORS[method],
                edgecolors="white",
                linewidths=0.2,
                alpha=0.55,
                zorder=3,
            )
    axis.set_title(title, fontsize=12, fontweight="bold", pad=12, loc='left')
    counts = method_frames['spatialz'].groupby('gap').size()
    axis.set_xticks(positions, [f'{gap}\nn={counts[gap]}' for gap in gaps])
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
    parser.add_argument('--spatialz-root', type=Path, default=STUDY_ROOT / 'outputs/spatialz_native_evaluation_cuda_v1')
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

    metrics, _ = load_comparison(args.spatialz_root / 'evaluation', args.input_dir / 'method_metrics.csv')
    metrics.to_csv(output / "plot_data.csv", index=False)
    configure_matplotlib()
    figure, axes = plt.subplots(1, 4, figsize=(14.0, 4.6))
    for letter, axis, (metric, title) in zip('ABCD', axes, METRICS, strict=True):
        panel(axis, metrics, metric, f'{letter}  {title}')
    handles = [Patch(facecolor=COLORS[m], edgecolor='none', alpha=.85, label=NAMES[m]) for m in METHODS]
    for axis, ylabel in zip(axes, ('Wasserstein distance', 'Wasserstein distance',
                                 "Moran's I profile correlation", 'Residual profile correlation')):
        axis.set_ylabel(ylabel)
    figure.suptitle(
        "FEAST, conditional resampling and SpatialZ",
        fontsize=17,
        fontweight="bold",
        y=0.985, x=0.06, ha='left',
    )
    figure.legend(
        handles=handles,
        loc="upper right",
        ncol=3,
        bbox_to_anchor=(0.985, 0.925),
        frameon=False,
        handlelength=0.9,
        handletextpad=0.5,
    )
    figure.text(
        0.06,
        0.90,
        "One point per target; resampling averaged over 10 repeats. Conditional panels use common supported targets.",
        ha="left",
        va="top",
        fontsize=10,
        color="#4F4F4F",
    )
    figure.tight_layout(rect=(0.005, 0.02, 1, 0.80), w_pad=1.4)
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
        "visual_style_reference": "visualization/06_3d_stack/plot_results.py",
        "visual_layout": "one_by_four_matched_target_boxplots_with_resampling_means",
        "resampling_display_unit": "per_target_mean_across_ten_runs",
        'conditional_population': 'identical supported targets across methods within each gap',
        'spatial_population': 'all 336 targets, independent native spatial graphs',
        "inputs": [str(args.input_dir / "method_metrics.csv"), str(args.spatialz_root / 'evaluation/metrics.csv')],
        "outputs": [path.name for path in targets[:-1]],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "figure_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote Study 06 candidate comparison figure to {output}")


if __name__ == "__main__":
    main()
