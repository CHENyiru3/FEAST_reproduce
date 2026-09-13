#!/usr/bin/env python3
"""Render a compact truth-versus-FEAST spatial-expression matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from plot_simulation_effect import (
    CASES,
    configure_matplotlib,
    generated_path,
    input_path,
    setup_spatial_axis,
    verify_evidence,
)


VIS_DIR = Path(__file__).resolve().parent
STUDY_DIR = VIS_DIR.parents[1] / "05_2d_conditional_transfer"
PER_GENE_METRICS = STUDY_DIR / "outputs" / "scores" / "per_gene_metrics.csv"
OUTPUT_STEM = "conditional_transfer_expression_matrix"


def dense_vector(values) -> np.ndarray:
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float64).reshape(-1)


def muted_expression_colors(
    values: np.ndarray,
    normalizer: mpl.colors.Normalize,
    *,
    color_fraction: float = 0.55,
) -> np.ndarray:
    """Blend viridis expression colors with gray for observed context."""
    colors = mpl.colormaps["viridis"](normalizer(np.clip(values, 0.0, 1.0)))
    gray = np.asarray(mpl.colors.to_rgb("#D8D8D8"), dtype=np.float64)
    colors[:, :3] = color_fraction * colors[:, :3] + (1.0 - color_fraction) * gray
    return colors


def select_reference_markers() -> tuple[dict[str, str], pd.DataFrame]:
    """Select one spatial marker per dataset without using FEAST performance."""
    metrics = pd.read_csv(PER_GENE_METRICS, low_memory=False)
    reference_observed = metrics["reference_observed"].eq(True) | metrics[
        "reference_observed"
    ].astype(str).str.lower().eq("true")
    candidates = metrics[
        metrics["mode"].eq("mask_half")
        & np.isclose(metrics["assignment_randomness"].astype(float), 0.3)
        & reference_observed
        & metrics["target_zero_prop"].between(0.05, 0.95)
        & np.isfinite(metrics["target_moran_i"])
    ].copy()

    selected: dict[str, str] = {}
    records: list[dict[str, object]] = []
    for dataset in ("dlpfc", "merfish"):
        subset = candidates[candidates["dataset"].eq(dataset)]
        required_directions = sum(case["dataset"] == dataset for case in CASES)
        ranked = (
            subset.groupby("gene", as_index=False)
            .agg(
                n_directions=("direction", "nunique"),
                median_target_moran_i=("target_moran_i", "median"),
                minimum_target_moran_i=("target_moran_i", "min"),
                mean_target_zero_prop=("target_zero_prop", "mean"),
            )
            .query("n_directions == @required_directions")
            .sort_values(
                ["median_target_moran_i", "minimum_target_moran_i", "gene"],
                ascending=[False, False, True],
            )
        )
        if ranked.empty:
            raise RuntimeError(f"no eligible common spatial marker for {dataset}")
        winner = ranked.iloc[0]
        gene = str(winner["gene"])
        selected[dataset] = gene
        records.append(
            {
                "dataset": dataset,
                "gene": gene,
                "n_directions": int(winner["n_directions"]),
                "median_target_moran_i": float(winner["median_target_moran_i"]),
                "minimum_target_moran_i": float(winner["minimum_target_moran_i"]),
                "mean_target_zero_prop": float(winner["mean_target_zero_prop"]),
                "selection_uses_generated_values": False,
            }
        )
    return selected, pd.DataFrame.from_records(records)


def load_case(case: dict, gene: str) -> dict:
    observed = ad.read_h5ad(input_path(case), backed="r")
    generated = ad.read_h5ad(generated_path(case))
    try:
        if gene not in observed.var_names or gene not in generated.var_names:
            raise RuntimeError(f"{gene} is absent from the shared panel for {case['slice']}")
        observed_indices = observed.obs_names.get_indexer(generated.obs_names)
        if np.any(observed_indices < 0):
            raise RuntimeError(f"generated spots are absent from {case['slice']}")

        coordinates = np.asarray(observed.obsm["spatial"], dtype=np.float64)[:, :2]
        target_coordinates = np.asarray(generated.obsm["spatial"], dtype=np.float64)[:, :2]
        np.testing.assert_allclose(
            coordinates[observed_indices], target_coordinates, rtol=0.0, atol=0.0
        )

        observed_truth = dense_vector(
            observed.layers["counts"][:, observed.var_names.get_loc(gene)]
        )
        truth = observed_truth[observed_indices]
        simulation = dense_vector(
            generated.layers["counts"][:, generated.var_names.get_loc(gene)]
        )
    finally:
        observed.file.close()

    observed_truth_log = np.log1p(observed_truth)
    truth_log = observed_truth_log[observed_indices]
    simulation_log = np.log1p(simulation)
    scale = float(
        np.nanpercentile(
            np.concatenate([observed_truth_log, simulation_log]),
            99.5,
        )
    )
    if not np.isfinite(scale) or scale <= 0:
        scale = float(
            np.nanmax(np.concatenate([observed_truth_log, simulation_log]))
        )
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"non-positive expression scale for {case['slice']} {gene}")

    return {
        "case": case,
        "gene": gene,
        "coordinates": coordinates,
        "target_coordinates": target_coordinates,
        "observed_truth": observed_truth_log / scale,
        "truth": truth_log / scale,
        "simulation": simulation_log / scale,
        "raw_scale_q995": scale,
        "split": float(np.median(coordinates[:, 0])),
    }


def draw_matrix(case_data: list[dict]) -> plt.Figure:
    configure_matplotlib()
    mpl.rcParams.update({"font.size": 9.0})
    width_ratios = [1.0 if data["case"]["dataset"] == "dlpfc" else 1.35 for data in case_data]
    fig = plt.figure(figsize=(11.2, 4.0))
    grid = fig.add_gridspec(
        2,
        len(case_data),
        width_ratios=width_ratios,
        left=0.075,
        right=0.99,
        bottom=0.20,
        top=0.78,
        wspace=0.06,
        hspace=0.05,
    )
    axes = np.empty((2, len(case_data)), dtype=object)
    normalizer = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    artist = None

    for column, data in enumerate(case_data):
        case = data["case"]
        point_size = float(case["point_size"]) * 1.25
        source_mask = data["coordinates"][:, 0] <= data["split"]
        source_colors = muted_expression_colors(
            data["observed_truth"][source_mask],
            normalizer,
        )
        for row, values in enumerate((data["truth"], data["simulation"])):
            ax = fig.add_subplot(grid[row, column])
            axes[row, column] = ax
            ax.scatter(
                data["coordinates"][:, 0],
                data["coordinates"][:, 1],
                s=point_size,
                color="#E7E7E7",
                linewidths=0,
                zorder=0,
                rasterized=True,
            )
            ax.scatter(
                data["coordinates"][source_mask, 0],
                data["coordinates"][source_mask, 1],
                s=point_size,
                color=source_colors,
                linewidths=0,
                zorder=1,
                rasterized=True,
            )
            artist = ax.scatter(
                data["target_coordinates"][:, 0],
                data["target_coordinates"][:, 1],
                s=point_size,
                c=np.clip(values, 0.0, 1.0),
                cmap="viridis",
                norm=normalizer,
                linewidths=0,
                zorder=2,
                rasterized=True,
            )
            ax.axvline(
                data["split"],
                color="#555555",
                linestyle=(0, (2, 2)),
                linewidth=0.65,
                zorder=3,
            )
            setup_spatial_axis(ax, data["coordinates"])

        short_slice = case["slice"].replace("Zhuang-ABCA-1.", "")
        top_axis = axes[0, column]
        top_axis.text(
            0.5,
            1.38,
            f"{case['dataset_label']}  {short_slice}",
            transform=top_axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
        )
        top_axis.plot(
            [0.22, 0.78],
            [1.31, 1.31],
            transform=top_axis.transAxes,
            color="#444444",
            linewidth=0.65,
            clip_on=False,
        )
        top_axis.text(
            0.5,
            1.12,
            data["gene"],
            transform=top_axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontstyle="italic",
        )

    for row, label in enumerate(("Held-out\ntruth", "FEAST\nsimulation")):
        bounds = axes[row, 0].get_position()
        fig.text(
            0.025,
            bounds.y0 + bounds.height / 2,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
        )

    colorbar_axis = fig.add_axes([0.37, 0.075, 0.30, 0.018])
    colorbar = fig.colorbar(artist, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.set_label(
        "Column-normalized log1p expression (value / q99.5 full truth + FEAST)",
        fontsize=7.3,
        labelpad=2,
    )
    colorbar.ax.tick_params(labelsize=6.8, length=2, width=0.5)
    return fig


def save_figure(figure: plt.Figure, output_dir: Path, dpi: int) -> list[Path]:
    paths: list[Path] = []
    for suffix, kwargs in {
        "pdf": {
            "metadata": {
                "Creator": "FEAST Study 05 conditional-transfer expression matrix",
                "CreationDate": None,
                "ModDate": None,
            }
        },
        "svg": {
            "metadata": {
                "Creator": "FEAST Study 05 conditional-transfer expression matrix",
                "Date": None,
            }
        },
        "png": {
            "dpi": dpi,
            "metadata": {
                "Software": "FEAST Study 05 conditional-transfer expression matrix"
            },
        },
    }.items():
        path = output_dir / f"{OUTPUT_STEM}.{suffix}"
        figure.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    plt.close(figure)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=VIS_DIR / "figures")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    score_provenance, _, _ = verify_evidence()
    markers, selection = select_reference_markers()
    case_data = [load_case(case, markers[case["dataset"]]) for case in CASES]

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}-expression-matrix-", dir=args.output_dir.parent)
    )
    try:
        output_paths = save_figure(draw_matrix(case_data), work_dir, int(args.dpi))
        summary = pd.DataFrame(
            [
                {
                    "dataset": data["case"]["dataset"],
                    "slice": data["case"]["slice"],
                    "direction": data["case"]["direction"],
                    "gene": data["gene"],
                    "expression_q99_5_log1p": data["raw_scale_q995"],
                    "observed_source_spots": int(
                        np.sum(data["coordinates"][:, 0] <= data["split"])
                    ),
                    "target_spots": len(data["target_coordinates"]),
                }
                for data in case_data
            ]
        )
        selection.to_csv(work_dir / f"{OUTPUT_STEM}_marker_selection.csv", index=False)
        summary.to_csv(work_dir / f"{OUTPUT_STEM}_summary.csv", index=False)
        provenance = {
            "status": "ok",
            "configuration_id": "study05-conditional-transfer-expression-matrix-v2",
            "score_configuration_id": score_provenance["configuration_id"],
            "design": {
                "rows": ["held-out truth", "FEAST simulation"],
                "columns": [case["slice"] for case in CASES],
                "observed_half": "ground-truth expression in 55% viridis blended with light gray",
                "target_half": "fully saturated held-out truth or FEAST expression",
                "assignment_randomness": 0.3,
                "normalization": "within-column log1p value divided by shared full-truth-plus-FEAST q99.5",
                "marker_selection": "highest median held-out target Moran I among genes observed in every dataset slice with 5-95% target zeros",
                "marker_selection_uses_generated_values": False,
                "selected_markers": markers,
            },
            "claim_limits": [
                "reference-selected markers illustrate spatial expression and are not a genome-wide performance summary",
                "target expression was used only after generation for evaluation and display",
                "each column has its own expression scale",
            ],
            "outputs": [path.name for path in output_paths]
            + [
                f"{OUTPUT_STEM}_marker_selection.csv",
                f"{OUTPUT_STEM}_summary.csv",
                f"{OUTPUT_STEM}_provenance.json",
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (work_dir / f"{OUTPUT_STEM}_provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        for path in sorted(work_dir.iterdir()):
            if path.is_file():
                os.replace(path, args.output_dir / path.name)
    finally:
        if work_dir.exists():
            work_dir.rmdir()

    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(args.output_dir),
                "selected_markers": markers,
                "slices": len(case_data),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
