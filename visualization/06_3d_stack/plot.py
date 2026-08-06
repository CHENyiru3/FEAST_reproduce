#!/usr/bin/env python3
"""Render validated Study 06 fidelity and continuity diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPRO_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = REPRO_ROOT / "06_3d_stack"
FINAL_ROOT = STUDY_ROOT / "outputs" / "final"
EVALUATION_ROOT = FINAL_ROOT / "evaluation"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "figures"
GAP_ORDER = [3, 5, 10]
GAP_COLORS = {3: "#0072B2", 5: "#E69F00", 10: "#009E73"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPRO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "feast-study06-publication",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for key in ("left", "bottom"):
        axis.spines[key].set_color("#555555")
        axis.spines[key].set_linewidth(0.8)


def require_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    validation = json.loads((FINAL_ROOT / "validation_summary.json").read_text(encoding="utf-8"))
    if validation.get("status") != "passed" or validation.get("validated_outputs") != 93:
        raise ValueError("complete Study 06 validation is required")
    provenance = json.loads((EVALUATION_ROOT / "score_provenance.json").read_text(encoding="utf-8"))
    if provenance.get("status") != "complete" or provenance.get("metric_contract", {}).get("composite_score") != "prohibited":
        raise ValueError("Study 06 atomic metric contract changed")
    for name, expected in provenance["outputs"].items():
        if sha256(EVALUATION_ROOT / name) != expected:
            raise ValueError(f"Study 06 evaluation hash changed: {name}")
    metrics = pd.read_csv(EVALUATION_ROOT / "target_metrics.csv")
    continuity = pd.read_csv(EVALUATION_ROOT / "continuity_summary.csv")
    plan = json.loads((FINAL_ROOT / "plan.json").read_text(encoding="utf-8"))
    donor_support = {}
    for density in plan["densities"].values():
        for target in density["targets"]:
            donor_support[(str(target["density_name"]), int(target["target_slice"]))] = bool(target["donor_support"])
    metrics["donor_supported"] = [
        donor_support[(str(row.density_name), int(row.target_slice))] for row in metrics.itertuples(index=False)
    ]
    return metrics, continuity, provenance


def metric_panel(axis: plt.Axes, frame: pd.DataFrame, column: str, title: str, ylabel: str) -> None:
    values = [frame.loc[frame["gap"] == gap, column].dropna().to_numpy() for gap in GAP_ORDER]
    boxes = axis.boxplot(values, positions=np.arange(len(GAP_ORDER)), widths=0.55, showfliers=False, patch_artist=True, medianprops={"color": "#222222", "linewidth": 1.1})
    for patch, gap in zip(boxes["boxes"], GAP_ORDER):
        patch.set_facecolor(GAP_COLORS[gap])
        patch.set_alpha(0.22)
        patch.set_edgecolor(GAP_COLORS[gap])
    for position, gap in enumerate(GAP_ORDER):
        subset = frame.loc[frame["gap"] == gap].sort_values("target_slice")
        offsets = np.linspace(-0.18, 0.18, len(subset))
        for donor, marker in ((False, "o"), (True, "x")):
            mask = subset["donor_supported"] == donor
            axis.scatter(position + offsets[mask.to_numpy()], subset.loc[mask, column], s=13, marker=marker, color=GAP_COLORS[gap], linewidths=0.7, alpha=0.75)
    axis.set_title(title, fontweight="bold")
    axis.set_ylabel(ylabel)
    axis.set_xticks(np.arange(len(GAP_ORDER)), ["Gap 3", "Gap 5", "Gap 10"])
    style_axis(axis)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "conditional_stack_diagnostic"
    targets = [stem.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png")]
    targets += [output / "target_metric_plot_data.csv", output / "continuity_plot_data.csv", output / "figure_provenance.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError("fresh-only Study 06 figure output already exists")

    metrics, continuity, score_provenance = require_inputs()
    metrics.to_csv(output / "target_metric_plot_data.csv", index=False)
    continuity.to_csv(output / "continuity_plot_data.csv", index=False)
    configure_matplotlib()
    figure, axes = plt.subplots(2, 3, figsize=(11.2, 6.8))
    panels = [
        ("gene_mean_pearson", "Gene-mean agreement", "Pearson correlation"),
        ("gene_variance_pearson", "Gene-variance agreement", "Pearson correlation"),
        ("gene_zero_fraction_pearson", "Zero-fraction agreement", "Pearson correlation"),
        ("relative_error_gene_mean", "Relative gene-mean error", "Relative error"),
        ("zero_pattern_jaccard", "Zero-pattern overlap", "Jaccard index"),
    ]
    for axis, (column, title, ylabel) in zip(axes.ravel()[:5], panels):
        metric_panel(axis, metrics, column, title, ylabel)

    axis = axes.ravel()[5]
    width = 0.34
    positions = np.arange(len(GAP_ORDER))
    ordered = continuity.set_index("gap").loc[GAP_ORDER]
    axis.bar(positions - width / 2, ordered["generated_target_z_coherence_median"], width, color="#0072B2", label="Generated–target")
    axis.bar(positions + width / 2, ordered["real_split_half_z_coherence_median"], width, color="#999999", label="Real split-half")
    axis.set_title("Across-z coherence baseline", fontweight="bold")
    axis.set_ylabel("Median class–gene z correlation")
    axis.set_xticks(positions, ["Gap 3", "Gap 5", "Gap 10"])
    style_axis(axis)

    donor_handles = [
        mpl.lines.Line2D([], [], color="#555555", marker="o", linestyle="None", markersize=4, label="No external label donor"),
        mpl.lines.Line2D([], [], color="#555555", marker="x", linestyle="None", markersize=5, label="Donor-supported label(s)"),
    ]
    figure.legend(handles=donor_handles, loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False)
    axis.legend(loc="lower center", ncol=1, frameon=False)
    figure.subplots_adjust(top=0.91, bottom=0.09, left=0.08, right=0.98, hspace=0.34, wspace=0.26)
    figure.savefig(stem.with_suffix(".pdf"), metadata={"Title": "FEAST Study 06 conditional stack diagnostic", "Creator": "FEAST publication visualization"})
    figure.savefig(stem.with_suffix(".svg"), metadata={"Title": "FEAST Study 06 conditional stack diagnostic"})
    figure.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(figure)

    source_paths = {
        "validation_summary": FINAL_ROOT / "validation_summary.json",
        "score_provenance": EVALUATION_ROOT / "score_provenance.json",
        "target_metrics": EVALUATION_ROOT / "target_metrics.csv",
        "continuity_summary": EVALUATION_ROOT / "continuity_summary.csv",
        "plan": FINAL_ROOT / "plan.json",
    }
    output_paths = {
        "pdf": stem.with_suffix(".pdf"),
        "svg": stem.with_suffix(".svg"),
        "png": stem.with_suffix(".png"),
        "target_plot_data": output / "target_metric_plot_data.csv",
        "continuity_plot_data": output / "continuity_plot_data.csv",
    }
    record = {
        "schema_version": 1,
        "study": "06_3d_stack",
        "configuration_id": score_provenance["configuration_id"],
        "figure_promotion_authorized": False,
        "interpretation": "validated atomic fidelity and continuity diagnostic; target slices are ordered z levels, not biological replicates",
        "composite_score": "prohibited",
        "winner_ranking": "prohibited",
        "donor_support_marker_rule": "x marks targets requiring at least one declared out-of-bracket label donor",
        "inputs": {name: {"path": provenance_path(path), "sha256": sha256(path)} for name, path in source_paths.items()},
        "script_sha256": sha256(Path(__file__).resolve()),
        "outputs": {name: {"path": provenance_path(path), "sha256": sha256(path)} for name, path in output_paths.items()},
        "editable_text": {"pdf_fonttype": 42, "svg_fonttype": "none"},
    }
    (output / "figure_provenance.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote Study 06 diagnostic figure to {output}")


if __name__ == "__main__":
    main()
