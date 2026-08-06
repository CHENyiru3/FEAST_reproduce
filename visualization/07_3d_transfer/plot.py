#!/usr/bin/env python3
"""Render full-axis Study 07 coverage and adjacent-z continuity diagnostics."""

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
STUDY_ROOT = REPRO_ROOT / "07_3d_transfer"
EVALUATION_ROOT = STUDY_ROOT / "outputs" / "final" / "evaluation"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "figures"
AGE_COLORS = {"E15.5": "#0072B2", "E18.5": "#D55E00"}
REGION_COLORS = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
    "#D55E00", "#F0E442", "#000000", "#999999",
]


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
        "svg.hashsalt": "feast-study07-publication",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def style_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis, color="#DDDDDD", linewidth=0.5, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for key in ("left", "bottom"):
        axis.spines[key].set_color("#555555")
        axis.spines[key].set_linewidth(0.8)


def require_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    provenance_path = EVALUATION_ROOT / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError("validated Study 07 evaluation is required")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("status") != "complete" or provenance.get("accuracy_claim_authorized"):
        raise ValueError("Study 07 descriptive-only evaluation contract changed")
    for name, expected in provenance["outputs"].items():
        path = EVALUATION_ROOT / name
        if sha256(path) != expected:
            raise ValueError(f"Study 07 evaluation hash changed: {path}")
    coverage = pd.read_csv(EVALUATION_ROOT / "region_coverage.csv")
    adjacent = pd.read_csv(EVALUATION_ROOT / "adjacent_z_continuity.csv")
    summary = pd.read_csv(EVALUATION_ROOT / "age_summary.csv")
    return coverage, adjacent, summary, provenance


def coverage_for_plot(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    totals = frame.groupby("region", as_index=False)["n_spots"].sum().sort_values("n_spots", ascending=False)
    retained = totals.head(8)["region"].astype(str).tolist()
    result = frame.copy()
    result["display_region"] = result["region"].where(result["region"].isin(retained), "Other")
    result = result.groupby(["age", "z_index", "z_world", "display_region"], as_index=False)["n_spots"].sum()
    order = retained + (["Other"] if "Other" in set(result["display_region"]) else [])
    return result, order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "full_axis_transfer_diagnostic"
    targets = [stem.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png")]
    targets += [output / "region_coverage_plot.csv", output / "adjacent_z_plot.csv", output / "figure_provenance.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError("fresh-only Study 07 figure output already exists")

    coverage, adjacent, summary, evaluation_provenance = require_inputs()
    coverage_plot, region_order = coverage_for_plot(coverage)
    adjacent_plot = adjacent.copy()
    adjacent_plot["gene_mean_discontinuity"] = np.maximum(
        1.0 - adjacent_plot["gene_mean_pearson"].to_numpy(dtype=float), 1e-12
    )
    adjacent_plot["region_gene_mean_discontinuity"] = np.maximum(
        1.0 - adjacent_plot["region_gene_mean_pearson"].to_numpy(dtype=float), 1e-12
    )
    coverage_plot.to_csv(output / "region_coverage_plot.csv", index=False)
    adjacent_plot.to_csv(output / "adjacent_z_plot.csv", index=False)

    configure_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    for axis, age in zip(axes[0], ("E15.5", "E18.5")):
        age_frame = coverage_plot.loc[coverage_plot["age"] == age]
        pivot = age_frame.pivot_table(index="z_world", columns="display_region", values="n_spots", fill_value=0).sort_index()
        series = [pivot[region].to_numpy() if region in pivot else np.zeros(len(pivot)) for region in region_order]
        axis.stackplot(pivot.index.to_numpy(), series, labels=region_order, colors=REGION_COLORS[:len(region_order)], linewidth=0)
        axis.set_title(f"{age} full-axis regional support", fontweight="bold")
        axis.set_xlabel("DevCCF z coordinate")
        axis.set_ylabel("Blueprint spots")
        axis.set_xlim(float(pivot.index.min()), float(pivot.index.max()))
        style_axis(axis)

    for age in ("E15.5", "E18.5"):
        frame = adjacent_plot.loc[adjacent_plot["age"] == age].sort_values("midpoint_z")
        axes[1, 0].plot(frame["midpoint_z"], frame["gene_mean_discontinuity"], color=AGE_COLORS[age], linewidth=1.2, label=age)
        axes[1, 1].plot(frame["midpoint_z"], frame["region_gene_mean_discontinuity"], color=AGE_COLORS[age], linewidth=1.2, label=age)
    axes[1, 0].set_title("Adjacent-z gene-mean discontinuity", fontweight="bold")
    axes[1, 1].set_title("Adjacent-z region–gene discontinuity", fontweight="bold")
    for axis in axes[1]:
        axis.set_xlabel("Adjacent-slice midpoint z")
        axis.set_ylabel("1 − Pearson correlation\n(log scale; lower is smoother)")
        axis.set_yscale("log")
        style_axis(axis)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=min(5, len(labels)), frameon=False)
    axes[1, 0].legend(loc="lower center", ncol=2, frameon=False)
    figure.subplots_adjust(top=0.90, bottom=0.09, left=0.08, right=0.98, hspace=0.34, wspace=0.25)
    figure.savefig(stem.with_suffix(".pdf"), metadata={"Title": "FEAST Study 07 full-axis diagnostic", "Creator": "FEAST publication visualization"})
    figure.savefig(stem.with_suffix(".svg"), metadata={"Title": "FEAST Study 07 full-axis diagnostic"})
    figure.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(figure)

    source_paths = {
        "evaluation_provenance": EVALUATION_ROOT / "provenance.json",
        "region_coverage": EVALUATION_ROOT / "region_coverage.csv",
        "adjacent_z_continuity": EVALUATION_ROOT / "adjacent_z_continuity.csv",
        "age_summary": EVALUATION_ROOT / "age_summary.csv",
        "validation": STUDY_ROOT / "outputs" / "final" / "validation.json",
        "E15.5_manifest": STUDY_ROOT / "outputs" / "final" / "E15.5" / "manifest.csv",
        "E18.5_manifest": STUDY_ROOT / "outputs" / "final" / "E18.5" / "manifest.csv",
    }
    output_paths = {
        "pdf": stem.with_suffix(".pdf"),
        "svg": stem.with_suffix(".svg"),
        "png": stem.with_suffix(".png"),
        "coverage_plot_data": output / "region_coverage_plot.csv",
        "adjacent_plot_data": output / "adjacent_z_plot.csv",
    }
    record = {
        "schema_version": 1,
        "study": "07_3d_transfer",
        "configuration_id": evaluation_provenance["configuration_id"],
        "figure_promotion_authorized": False,
        "interpretation": "descriptive full-axis coverage and adjacent-z continuity; no target-expression accuracy claim",
        "composite_score": "prohibited",
        "continuity_display": "1 - Pearson correlation on a logarithmic axis; values are clipped only for display at 1e-12",
        "top_region_rule": "eight regions with greatest total blueprint spot support across both ages; remaining regions grouped as Other",
        "age_summary": summary.to_dict("records"),
        "inputs": {name: {"path": provenance_path(path), "sha256": sha256(path)} for name, path in source_paths.items()},
        "script_sha256": sha256(Path(__file__).resolve()),
        "outputs": {name: {"path": provenance_path(path), "sha256": sha256(path)} for name, path in output_paths.items()},
        "editable_text": {"pdf_fonttype": 42, "svg_fonttype": "none"},
    }
    (output / "figure_provenance.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote Study 07 diagnostic figure to {output}")


if __name__ == "__main__":
    main()
