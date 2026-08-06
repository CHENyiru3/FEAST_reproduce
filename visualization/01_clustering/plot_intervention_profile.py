#!/usr/bin/env python3
"""Render the realized simulator-intervention profile for Study 01.

This figure reads FEAST's recorded ``realized_stage_achieved_change``
diagnostics from all three registered DLPFC slices.  It describes the realized
relative mean, variance, and zero-fraction changes; it is not a clustering
score or a method comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
SIMULATION_DIR = (
    REPOSITORY_ROOT
    / "01_clustering"
    / "outputs"
    / "final_rerun_20260718"
    / "simulations"
)
SIMULATION_MANIFEST = SIMULATION_DIR / "simulation_manifest.csv"

SLICES = ("151508", "151670", "151676")
FACTOR_SPECS = {
    "mean": {
        "title": "Mean control",
        "x_label": "Nominal fold change",
        "neutral": 1.0,
        "levels": (
            (0.50, "mean_0_50"), (0.67, "mean_0_67"), (0.80, "mean_0_80"),
            (1.00, "baseline"), (1.25, "mean_1_25"), (1.50, "mean_1_50"),
            (2.00, "mean_2"),
        ),
    },
    "variance": {
        "title": "Variance control",
        "x_label": "Nominal fold change",
        "neutral": 1.0,
        "levels": (
            (0.50, "variance_0_50"), (0.67, "variance_0_67"), (0.80, "variance_0_80"),
            (1.00, "baseline"), (1.25, "variance_1_25"), (1.50, "variance_1_50"),
            (2.00, "variance_2"),
        ),
    },
    "sparsity": {
        "title": "Zero-fraction control",
        "x_label": "Nominal logit shift",
        "neutral": 0.0,
        "levels": (
            (-1.00, "sparsity_neg_1"), (-0.50, "sparsity_neg_0_50"),
            (-0.25, "sparsity_neg_0_25"), (0.00, "baseline"),
            (0.25, "sparsity_pos_0_25"), (0.50, "sparsity_pos_0_50"),
            (1.00, "sparsity_pos_1"),
        ),
    },
}
METRIC_SPECS = (
    ("mean", "Realized mean\n(relative to baseline)"),
    ("variance", "Realized variance\n(relative to baseline)"),
    ("zero_prop", "Realized zero fraction\n(relative to baseline)"),
)

SUMMARY_COLOR = "#0F4D92"
SLICE_COLOR = "#767676"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_matplotlib() -> None:
    """Apply the compact publication style from scientific-figure-making."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "feast-study01-intervention-profile-v1",
    })


def _manifest_index() -> pd.DataFrame:
    manifest = pd.read_csv(
        SIMULATION_MANIFEST,
        dtype={"slice_id": str, "simulation_id": str},
    )
    required = {"slice_id", "simulation_id", "file_path", "status"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Simulation manifest is missing columns: {sorted(missing)}")
    if not manifest["status"].eq("ok").all():
        raise ValueError("Simulation manifest contains unsuccessful jobs.")
    return manifest.set_index(["slice_id", "simulation_id"], verify_integrity=True)


def load_profile_data() -> tuple[pd.DataFrame, list[Path]]:
    manifest = _manifest_index()
    rows: list[dict[str, object]] = []
    input_paths: list[Path] = []
    cached_diagnostics: dict[tuple[str, str], dict[str, float]] = {}

    for factor, spec in FACTOR_SPECS.items():
        for nominal_level, simulation_id in spec["levels"]:
            for slice_id in SLICES:
                key = (slice_id, simulation_id)
                if key not in manifest.index:
                    raise ValueError(f"Missing registered simulation: {slice_id}/{simulation_id}")
                if key not in cached_diagnostics:
                    path = SIMULATION_DIR / str(manifest.loc[key, "file_path"])
                    if not path.is_file():
                        raise FileNotFoundError(f"Simulation file is missing: {path}")
                    data = ad.read_h5ad(path, backed="r")
                    try:
                        diagnostics = data.uns.get("alteration_diagnostics", {})
                        realized = diagnostics.get("realized_stage_achieved_change", {})
                        required_metrics = {metric for metric, _label in METRIC_SPECS}
                        missing_metrics = required_metrics - set(realized)
                        if missing_metrics:
                            raise ValueError(
                                f"{path} lacks realized diagnostics: {sorted(missing_metrics)}"
                            )
                        cached_diagnostics[key] = {
                            metric: float(realized[metric])
                            for metric in required_metrics
                        }
                    finally:
                        data.file.close()
                    input_paths.append(path)
                rows.append({
                    "slice_id": slice_id,
                    "factor": factor,
                    "simulation_id": simulation_id,
                    "nominal_level": nominal_level,
                    **cached_diagnostics[key],
                })

    data = pd.DataFrame(rows)
    expected_rows = len(SLICES) * sum(len(spec["levels"]) for spec in FACTOR_SPECS.values())
    if len(data) != expected_rows:
        raise ValueError(f"Expected {expected_rows} profile rows; found {len(data)}")
    return data, sorted(set(input_paths))


def _metric_ylim(data: pd.DataFrame, metric: str) -> tuple[float, float]:
    values = data[metric].to_numpy(dtype=float)
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    spread = high - low
    pad = 0.08 * spread if spread > 0 else 0.05
    return max(0.0, low - pad), high + pad


def _format_xticks(ax: plt.Axes, factor: str) -> None:
    spec = FACTOR_SPECS[factor]
    levels = [level for level, _sim in spec["levels"]]
    if factor in {"mean", "variance"}:
        ticks = [0.5, 1.0, 1.5, 2.0]
    else:
        ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ax.set_xticks(ticks)
    ax.set_xlim(min(levels) - 0.06 * (max(levels) - min(levels)),
                max(levels) + 0.06 * (max(levels) - min(levels)))


def draw_figure(data: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    configure_matplotlib()
    factors = tuple(FACTOR_SPECS)
    fig, axes = plt.subplots(len(METRIC_SPECS), len(factors), figsize=(13.0, 9.3))

    for row_idx, (metric, metric_label) in enumerate(METRIC_SPECS):
        y_lim = _metric_ylim(data, metric)
        for col_idx, factor in enumerate(factors):
            ax = axes[row_idx, col_idx]
            subset = data.loc[data["factor"] == factor].copy()
            for slice_id in SLICES:
                slice_data = subset.loc[subset["slice_id"] == slice_id].sort_values("nominal_level")
                ax.plot(
                    slice_data["nominal_level"],
                    slice_data[metric],
                    color=SLICE_COLOR,
                    linewidth=0.9,
                    alpha=0.35,
                    zorder=1,
                )
            summary = subset.groupby("nominal_level", as_index=False)[metric].mean()
            ax.plot(
                summary["nominal_level"],
                summary[metric],
                color=SUMMARY_COLOR,
                marker="o",
                markerfacecolor="white",
                markeredgewidth=1.1,
                markersize=5.5,
                linewidth=2.3,
                zorder=3,
            )
            ax.axhline(1.0, color="#4D4D4D", linewidth=0.9, linestyle="--", alpha=0.8, zorder=0)
            ax.axvline(
                FACTOR_SPECS[factor]["neutral"],
                color="#CFCECE",
                linewidth=0.8,
                linestyle=":",
                zorder=0,
            )
            _format_xticks(ax, factor)
            ax.set_ylim(*y_lim)
            ax.grid(axis="y", color="#E8E8E8", linewidth=0.55, alpha=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=9)
            if row_idx == len(METRIC_SPECS) - 1:
                ax.set_xlabel(FACTOR_SPECS[factor]["x_label"], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=10.5, fontweight="bold", labelpad=6)
            if row_idx == 0:
                ax.set_title(FACTOR_SPECS[factor]["title"], fontsize=12.5, fontweight="bold", pad=10)

    handles = [
        Line2D([0], [0], color=SLICE_COLOR, linewidth=1.2, alpha=0.6, label="Individual slice"),
        Line2D([0], [0], color=SUMMARY_COLOR, marker="o", markerfacecolor="white",
               markeredgewidth=1.1, linewidth=2.3, label="Three-slice mean"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.987),
               ncol=2, fontsize=10, handlelength=1.7, columnspacing=1.4)
    fig.text(0.08, 0.985, "Realized FEAST simulation intervention profile",
             ha="left", va="center", fontsize=14, fontweight="bold")
    fig.text(
        0.08,
        0.948,
        "Recorded simulator diagnostics; horizontal dashed line = baseline 1.0, vertical dotted line = neutral setting.",
        ha="left",
        va="center",
        fontsize=9.2,
        color="#666666",
    )
    fig.subplots_adjust(left=0.095, right=0.995, top=0.90, bottom=0.095, hspace=0.34, wspace=0.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "intervention_profile"
    paths: list[Path] = []
    for suffix, kwargs in {
        "pdf": {"metadata": {"Creator": "FEAST Study 01 intervention profile"}},
        "png": {"dpi": dpi, "metadata": {"Software": "FEAST Study 01 intervention profile"}},
        "svg": {"metadata": {"Creator": "FEAST Study 01 intervention profile"}},
    }.items():
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, **kwargs)
        print(f"Saved: {path}")
        paths.append(path)
    plt.close(fig)
    return paths


def write_provenance(
    output_dir: Path,
    outputs: list[Path],
    input_paths: list[Path],
) -> Path:
    payload = {
        "schema_version": 1,
        "figure_set": "study01_intervention_profile",
        "source": {
            "plot_script": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
            "plot_script_sha256": sha256(Path(__file__)),
            "simulation_manifest": str(SIMULATION_MANIFEST.relative_to(REPOSITORY_ROOT)),
            "simulation_manifest_sha256": sha256(SIMULATION_MANIFEST),
            "diagnostic_field": "uns.alteration_diagnostics.realized_stage_achieved_change",
            "n_simulation_files_read": len(input_paths),
        },
        "design": {
            "slices": list(SLICES),
            "factors": list(FACTOR_SPECS),
            "metrics": [metric for metric, _label in METRIC_SPECS],
            "baseline_reference": 1.0,
            "summary_encoding": "thin individual-slice traces; blue three-slice mean",
        },
        "outputs": {path.name: sha256(path) for path in sorted(outputs)},
        "scientific_disposition": {
            "figure_promotion_authorized": False,
            "supported_interpretation": (
                "Descriptive profile of FEAST-recorded realized simulation diagnostics "
                "across the registered three-slice cohort."
            ),
        },
    }
    output_path = output_dir / "intervention_profile_provenance.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data, input_paths = load_profile_data()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "intervention_profile_data.csv"
    data.to_csv(data_path, index=False, float_format="%.12f")
    outputs = [data_path, *draw_figure(data, args.output_dir, args.dpi)]
    print(f"Saved: {write_provenance(args.output_dir, outputs, input_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
