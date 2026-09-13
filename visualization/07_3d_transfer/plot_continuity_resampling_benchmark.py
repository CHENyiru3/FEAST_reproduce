#!/usr/bin/env python3
"""Compare Study 07 adjacent-z continuity with conditional resampling."""
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
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


REPRO_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = REPRO_ROOT / "07_3d_transfer"
FEAST_PATH = STUDY_ROOT / "outputs" / "final" / "evaluation" / "adjacent_z_continuity.csv"
BASELINE_ROOT = (
    STUDY_ROOT
    / "outputs"
    / "final"
    / "baselines"
    / "conditional_whole_spot_resampling"
)
BASELINE_PATH = BASELINE_ROOT / "adjacent_z_continuity.csv"
BASELINE_PROVENANCE_PATH = BASELINE_ROOT / "provenance.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "figures"
AGES = ("E15.5", "E18.5")
METRICS = (
    ("gene_mean_pearson", "Gene mean"),
    ("region_gene_mean_pearson", "Region–gene mean"),
)
PAIR_KEYS = (
    "age",
    "lower_z_index",
    "upper_z_index",
    "lower_z",
    "upper_z",
    "midpoint_z",
)
METHODS = ("conditional_resampling", "feast")
METHOD_LABELS = {
    "conditional_resampling": "Conditional resampling",
    "feast": "FEAST",
}
METHOD_COLORS = {"conditional_resampling": "#999999", "feast": "#0072B2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


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


def build_plot_data() -> tuple[pd.DataFrame, dict]:
    feast = pd.read_csv(FEAST_PATH)
    baseline = pd.read_csv(BASELINE_PATH)
    provenance = json.loads(BASELINE_PROVENANCE_PATH.read_text(encoding="utf-8"))
    if (
        provenance.get("status") != "complete"
        or provenance.get("replicates") != 10
        or provenance.get("target_expression_exists") is not False
    ):
        raise ValueError("approved Study 07 conditional-resampling baseline is required")

    if baseline.groupby(list(PAIR_KEYS)).size().nunique() != 1:
        raise ValueError("resampling replicate count differs across adjacent-z pairs")
    if int(baseline.groupby(list(PAIR_KEYS)).size().iloc[0]) != 10:
        raise ValueError("each adjacent-z pair must have ten resampling runs")

    baseline_mean = baseline.groupby(list(PAIR_KEYS), as_index=False)[
        [metric for metric, _label in METRICS]
    ].mean()
    feast_keys = feast[list(PAIR_KEYS)].sort_values(list(PAIR_KEYS)).reset_index(drop=True)
    baseline_keys = (
        baseline_mean[list(PAIR_KEYS)].sort_values(list(PAIR_KEYS)).reset_index(drop=True)
    )
    if not feast_keys.equals(baseline_keys):
        raise ValueError("FEAST and resampling adjacent-z keys do not match")

    frames = []
    for method, source in (("feast", feast), ("conditional_resampling", baseline_mean)):
        for metric, label in METRICS:
            values = source[metric].to_numpy(dtype=float)
            frames.append(
                source[list(PAIR_KEYS)].assign(
                    method=method,
                    metric=metric,
                    metric_label=label,
                    pearson_r=values,
                    discontinuity=np.maximum(1.0 - values, 1e-12),
                )
            )
    plot_data = pd.concat(frames, ignore_index=True).sort_values(
        ["age", "metric", "method", "lower_z_index"]
    )
    return plot_data, provenance


def panel(axis: plt.Axes, frame: pd.DataFrame, age: str, metric: str, title: str) -> None:
    subset = frame.loc[frame["age"].eq(age) & frame["metric"].eq(metric)]
    positions = {"conditional_resampling": -0.16, "feast": 0.16}
    for method in METHODS:
        values = subset.loc[subset["method"].eq(method), "discontinuity"].to_numpy(
            dtype=float
        )
        box = axis.boxplot(
            [values],
            positions=[positions[method]],
            widths=0.22,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#D97946", "linewidth": 1.25},
            whiskerprops={"color": "#444444", "linewidth": 0.8},
            capprops={"color": "#444444", "linewidth": 0.8},
            boxprops={"edgecolor": "#444444", "linewidth": 0.8},
        )
        box["boxes"][0].set_facecolor(METHOD_COLORS[method])
        box["boxes"][0].set_alpha(0.72)
        jitter = np.linspace(-0.065, 0.065, len(values))
        axis.scatter(
            np.full(len(values), positions[method]) + jitter,
            values,
            s=8,
            c=METHOD_COLORS[method],
            edgecolors="none",
            alpha=0.34,
            zorder=3,
            rasterized=True,
        )

    axis.set_title(f"{age}\n{title}", fontsize=14, fontweight="bold", pad=8)
    axis.set_xlim(-0.42, 0.42)
    axis.set_xticks([0], ["Adjacent z pairs"])
    axis.set_yscale("log")
    # Shared limits are set from all plotted values by render().
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.65)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for spine_name in ("left", "bottom"):
        axis.spines[spine_name].set_linewidth(0.8)
        axis.spines[spine_name].set_color("#555555")


def render(frame: pd.DataFrame, stem: Path, dpi: int) -> None:
    configure_matplotlib()
    figure, axes = plt.subplots(1, 4, figsize=(18.0, 5.4), sharey=True)
    panels = [
        (age, metric, label)
        for age in AGES
        for metric, label in METRICS
    ]
    for axis, (age, metric, label) in zip(axes, panels, strict=True):
        panel(axis, frame, age, metric, label)
    displayed = frame["discontinuity"].to_numpy(dtype=float)
    positive = displayed[np.isfinite(displayed) & (displayed > 0)]
    axes[0].set_ylim(positive.min() * .5, positive.max() * 2)
    axes[0].set_ylabel("Discontinuity, Δ = 1 − Pearson r\n(log scale; lower is smoother)")
    handles = [
        Patch(
            facecolor=METHOD_COLORS[method],
            edgecolor="none",
            alpha=0.85,
            label=METHOD_LABELS[method],
        )
        for method in ("feast", "conditional_resampling")
    ]
    figure.suptitle(
        "Full-axis continuity vs age/region-conditioned resampling",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.91,
        "Matched adjacent-z pairs; each resampling point is the ten-run mean and ordered z pairs are not biological replicates.",
        ha="center",
        va="top",
        fontsize=12,
        color="#4F4F4F",
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
        0.12,
        "Continuity control only: the expression-free DevCCF target cannot evaluate biological accuracy.",
        ha="center",
        fontsize=9.5,
        color="#666666",
    )
    figure.tight_layout(rect=(0.015, 0.21, 1, 0.80), w_pad=1.4)
    figure.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={
            "Title": "FEAST Study 07 continuity vs conditional resampling",
            "Creator": "FEAST publication visualization",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
        metadata={
            "Title": "FEAST Study 07 continuity vs conditional resampling",
            "Creator": "FEAST publication visualization",
            "Date": None,
        },
    )
    figure.savefig(
        stem.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        metadata={"Software": "FEAST Study 07 continuity benchmark plot"},
    )
    plt.close(figure)


def main() -> None:
    global FEAST_PATH, BASELINE_ROOT, BASELINE_PATH, BASELINE_PROVENANCE_PATH
    args = parse_args()
    FEAST_PATH = args.input_root.resolve() / "evaluation/adjacent_z_continuity.csv"
    BASELINE_ROOT = args.baseline_root.resolve()
    BASELINE_PATH = BASELINE_ROOT / "adjacent_z_continuity.csv"
    BASELINE_PROVENANCE_PATH = BASELINE_ROOT / "provenance.json"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "full_axis_continuity_vs_conditional_resampling"
    plot_data_path = output / "full_axis_continuity_vs_conditional_resampling_plot_data.csv"
    provenance_path = output / "full_axis_continuity_vs_conditional_resampling_provenance.json"
    targets = [stem.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png")]
    targets.extend([plot_data_path, provenance_path])
    if any(path.exists() for path in targets):
        raise FileExistsError("fresh-only continuity benchmark output already exists")

    plot_data, baseline_provenance = build_plot_data()
    expected_pairs = {"E15.5": 157, "E18.5": 201}
    observed_pairs = {
        age: int(
            plot_data.loc[
                plot_data["age"].eq(age)
                & plot_data["metric"].eq(METRICS[0][0])
                & plot_data["method"].eq("feast")
            ].shape[0]
        )
        for age in AGES
    }
    if observed_pairs != expected_pairs:
        raise ValueError(f"unexpected adjacent-z pair counts: {observed_pairs}")

    plot_data.to_csv(plot_data_path, index=False)
    render(plot_data, stem, int(args.dpi))
    provenance = {
        "configuration_id": json.loads((FEAST_PATH.parent / "provenance.json").read_text())["configuration_id"],
        "schema_version": 1,
        "study": "07_3d_transfer",
        "status": "candidate_author_review_required",
        "interpretation": "descriptive adjacent-z continuity relative to age-and-region-conditioned whole-spot resampling",
        "target_expression_exists": False,
        "target_expression_accuracy_claim_authorized": False,
        "methods": [METHOD_LABELS[method] for method in ("feast", "conditional_resampling")],
        "metrics": [metric for metric, _label in METRICS],
        "metric_display": "1 - Pearson correlation on a logarithmic axis; values are floored at 1e-12 only for display",
        "comparison_unit": "matched adjacent-z pair; resampling mean across ten runs",
        "adjacent_pairs": observed_pairs,
        "ordered_z_pairs_are_biological_replicates": False,
        "baseline": {
            "method": baseline_provenance["method"],
            "replicates": baseline_provenance["replicates"],
            "seeds": baseline_provenance["seeds"],
        },
        "inputs": {
            "feast": str(FEAST_PATH),
            "baseline": str(BASELINE_PATH),
            "baseline_provenance": str(BASELINE_PROVENANCE_PATH),
        },
        "outputs": [path.name for path in targets if path != provenance_path],
        "png_dpi": int(args.dpi),
        "editable_text": {"pdf_fonttype": 42, "svg_fonttype": "none"},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote Study 07 continuity-resampling benchmark to {output}")


if __name__ == "__main__":
    main()
