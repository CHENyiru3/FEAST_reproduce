#!/usr/bin/env python3
"""Illustrate the fixed-plate rotation and support-loss stress test."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPOSITORY_ROOT / "02_alignment" / "data" / "local" / "151675.h5ad"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_STEM = "fixed_plate_rotation_illustration"
SPATEO_OUTPUT = REPOSITORY_ROOT / "02_alignment" / "outputs" / "fixed_plate_rerun_20260806_v1"
PASTE2_OUTPUT = REPOSITORY_ROOT / "02_alignment" / "outputs" / "paste2_rerun_20260807_v1"

ANGLES = (5.0, 15.0, 25.0, 35.0, 45.0, 60.0, 75.0)
SHOWCASE_ANGLES = (5.0, 45.0, 75.0)
PIVOT_FRACTION = 0.5

COLORS = {
    "retained": "#3775BA",
    "dropped": "#B64342",
    "low_error": "#0072B2",
    "elevated_error": "#E69F00",
    "high_error": "#D55E00",
    "plate": "#272727",
    "pivot": "#FFD700",
    "neutral": "#767676",
}


def read_spatial_coordinates(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = ad.read_h5ad(path, backed="r")
    try:
        if "spatial" not in data.obsm:
            raise KeyError(f"{path} does not contain .obsm['spatial']")
        coordinates = np.asarray(data.obsm["spatial"], dtype=np.float64)[:, :2].copy()
        names = np.asarray(data.obs_names, dtype=str)
    finally:
        data.file.close()
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"Invalid spatial coordinates: {coordinates.shape}")
    if not np.isfinite(coordinates).all():
        raise ValueError("Spatial coordinates contain non-finite values")
    return coordinates, names


def rotate_about(coordinates: np.ndarray, angle: float, pivot: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(angle)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    return (coordinates - pivot) @ rotation.T + pivot


def inside_plate(coordinates: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.all((coordinates >= lower) & (coordinates <= upper), axis=1)


def configure_matplotlib() -> None:
    arial_paths = (Path(sys.prefix) / "fonts" / "arial.ttf", Path(sys.prefix) / "fonts" / "arialbd.ttf")
    if not all((path.is_file() for path in arial_paths)):
        raise FileNotFoundError(f"Arial Regular and Bold are required for this figure: {arial_paths}")
    for path in arial_paths:
        font_manager.fontManager.addfont(path)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_exact_pair_errors(
    method: str,
    angle: float,
    reference_coordinates: np.ndarray,
    reference_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned and exact-reference coordinates with normalized pair errors."""
    angle_name = f"angle_{angle:g}"
    if method == "spateo":
        aligned_path = (
            SPATEO_OUTPUT
            / "methods"
            / "baseline"
            / "spateo"
            / angle_name
            / "aligned_coordinates.csv"
        )
        metrics_path = SPATEO_OUTPUT / "scores_v3" / "alignment_metrics.csv"
    elif method == "paste2":
        aligned_path = PASTE2_OUTPUT / "baseline" / angle_name / "aligned_coordinates.csv"
        metrics_path = PASTE2_OUTPUT / "scores_v3" / "alignment_metrics.csv"
    else:
        raise ValueError(f"Unsupported method: {method}")

    aligned_table = pd.read_csv(aligned_path)
    if aligned_table["spot_barcode"].duplicated().any():
        raise ValueError(f"Duplicate spot barcode in {aligned_path}")
    reference_index = pd.Index(reference_names)
    positions = reference_index.get_indexer(aligned_table["spot_barcode"].astype(str))
    if np.any(positions < 0):
        raise ValueError(f"Aligned spots are not a reference subset: {aligned_path}")
    metrics = pd.read_csv(metrics_path)
    metric_rows = metrics.loc[
        (metrics["method"] == method)
        & (metrics["alteration"] == "baseline")
        & (metrics["angle_degrees"] == angle)
    ]
    if len(metric_rows) != 1:
        raise ValueError(f"Expected one {method} baseline row at {angle:g}°")
    spacing = float(metric_rows.iloc[0]["reference_median_spot_spacing"])
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"Invalid reference spot spacing for {method} at {angle:g}°")

    aligned_coordinates = aligned_table[["x_aligned", "y_aligned"]].to_numpy(dtype=float)
    reference_subset = reference_coordinates[positions]
    errors = np.linalg.norm(aligned_coordinates - reference_subset, axis=1) / spacing
    return aligned_coordinates, reference_subset, errors


def build_figure(
    coordinates: np.ndarray,
    names: np.ndarray,
) -> tuple[plt.Figure, list[dict[str, float | int]]]:
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    center = (lower + upper) / 2.0
    half_extent = (upper - lower) / 2.0
    pivot = center + PIVOT_FRACTION * half_extent

    rotations: list[tuple[float, np.ndarray, np.ndarray]] = []
    summary: list[dict[str, float | int]] = []
    for angle in ANGLES:
        rotated = rotate_about(coordinates, angle, pivot)
        retained = inside_plate(rotated, lower, upper)
        retained_count = int(retained.sum())
        rotations.append((angle, rotated, retained))
        summary.append(
            {
                "angle_degrees": angle,
                "retained": retained_count,
                "dropped": int(len(coordinates) - retained_count),
                "retention_fraction": retained_count / len(coordinates),
            }
        )

    all_coordinates = np.vstack([coordinates, *(item[1] for item in rotations)])
    limits_lower = all_coordinates.min(axis=0)
    limits_upper = all_coordinates.max(axis=0)
    padding = 0.04 * np.maximum(limits_upper - limits_lower, 1.0)

    rotation_by_angle = {angle: (rotated, retained) for angle, rotated, retained in rotations}
    summary_by_angle = {float(row["angle_degrees"]): row for row in summary}

    result_data = {
        (method, angle): load_exact_pair_errors(method, angle, coordinates, names)
        for method in ("spateo", "paste2")
        for angle in SHOWCASE_ANGLES
    }
    fig = plt.figure(figsize=(15.5, 15.8))
    grid = fig.add_gridspec(
        4,
        3,
        height_ratios=(0.22, 1.0, 1.1, 1.1),
        hspace=0.18,
        wspace=0.12,
    )
    legend_axis = fig.add_subplot(grid[0, :])
    input_axes = [fig.add_subplot(grid[1, index]) for index in range(3)]
    result_axes = {
        method: [fig.add_subplot(grid[row, index], projection="3d") for index in range(3)]
        for method, row in (("spateo", 2), ("paste2", 3))
    }

    def decorate_spatial_axis(ax: plt.Axes, panel_label: str) -> None:
        ax.add_patch(
            Rectangle(
                lower,
                *(upper - lower),
                fill=False,
                edgecolor=COLORS["plate"],
                linewidth=1.8,
                zorder=4,
            )
        )
        ax.scatter(
            [pivot[0]],
            [pivot[1]],
            s=54,
            marker="D",
            facecolor=COLORS["pivot"],
            edgecolor=COLORS["plate"],
            linewidth=0.8,
            zorder=5,
        )
        ax.set_xlim(limits_lower[0] - padding[0], limits_upper[0] + padding[0])
        ax.set_ylim(limits_lower[1] - padding[1], limits_upper[1] + padding[1])
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.text(
            0.02,
            0.98,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
        )

    for axis, panel_label, angle in zip(input_axes, ("A", "B", "C"), SHOWCASE_ANGLES):
        rotated, retained = rotation_by_angle[angle]
        row = summary_by_angle[angle]
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=3.4,
            color="#D8D8D8",
            alpha=0.55,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        axis.scatter(
            rotated[~retained, 0],
            rotated[~retained, 1],
            s=4.2,
            color=COLORS["dropped"],
            alpha=0.72,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        axis.scatter(
            rotated[retained, 0],
            rotated[retained, 1],
            s=3.4,
            color=COLORS["retained"],
            alpha=0.88,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        decorate_spatial_axis(axis, panel_label)
        axis.set_title(
            f"{angle:g}° rotation and plate crop\n"
            f"{int(row['retained']):,} retained ({100 * float(row['retention_fraction']):.1f}%)",
            fontweight="bold",
            pad=9,
        )

    result_panel_labels = {"spateo": ("D", "E", "F"), "paste2": ("G", "H", "I")}
    for method, method_label in (("spateo", "Spateo"), ("paste2", "PASTE2")):
        for axis, panel_label, angle in zip(
            result_axes[method], result_panel_labels[method], SHOWCASE_ANGLES
        ):
            aligned, reference_subset, errors = result_data[(method, angle)]
            connector_indices = np.linspace(
                0, len(errors) - 1, min(120, len(errors)), dtype=int
            )
            error_colors = np.select(
                [errors <= 0.005, errors <= 0.01],
                [COLORS["low_error"], COLORS["elevated_error"]],
                default=COLORS["high_error"],
            )
            axis.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                np.zeros(len(coordinates)),
                s=1.8,
                color="#CFCFCF",
                alpha=0.34,
                depthshade=False,
                rasterized=True,
            )
            for connector in connector_indices:
                axis.plot(
                    [reference_subset[connector, 0], aligned[connector, 0]],
                    [reference_subset[connector, 1], aligned[connector, 1]],
                    [0.0, 1.0],
                    color="#555555",
                    alpha=0.16,
                    linewidth=0.4,
                )
            axis.scatter(
                aligned[:, 0],
                aligned[:, 1],
                np.ones(len(aligned)),
                s=3.4,
                color=error_colors,
                alpha=0.86,
                linewidths=0,
                depthshade=False,
                rasterized=True,
            )
            axis.set_xlim(float(lower[0]), float(upper[0]))
            axis.set_ylim(float(lower[1]), float(upper[1]))
            axis.set_zlim(-0.04, 1.04)
            axis.set_box_aspect((1.0, 1.0, 0.36))
            axis.view_init(elev=28, azim=-55)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_zticks([0.0, 1.0], ["target", "aligned"])
            axis.tick_params(axis="z", labelsize=7.5, pad=-1)
            axis.xaxis.pane.fill = False
            axis.yaxis.pane.fill = False
            axis.zaxis.pane.fill = False
            axis.grid(False)
            axis.text2D(
                0.02,
                0.97,
                panel_label,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=14,
                fontweight="bold",
            )
            mean_error = float(errors.mean())
            mean_label = "≈0" if mean_error < 5e-5 else f"{mean_error:.4f}"
            high_fraction = float((errors > 0.01).mean())
            axis.text2D(
                0.98,
                0.97,
                f"mean {mean_label}\n>0.01: {high_fraction:.1%}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color="#4D4D4D",
            )
            if axis is result_axes[method][0]:
                axis.text2D(
                    -0.18,
                    0.5,
                    f"{method_label}\nspot-to-spot alignment",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=11.5,
                    fontweight="bold",
                )

    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color="#D8D8D8", label="Original plate geometry"),
        Line2D([], [], marker="o", linestyle="", color=COLORS["retained"], label="Retained moving input"),
        Line2D([], [], marker="o", linestyle="", color=COLORS["dropped"], label="Excluded by plate crop"),
        Line2D([], [], marker="o", linestyle="", color=COLORS["low_error"], label="Low error ≤0.005"),
        Line2D([], [], marker="o", linestyle="", color=COLORS["elevated_error"], label="Elevated 0.005–0.01"),
        Line2D([], [], marker="o", linestyle="", color=COLORS["high_error"], label="High error >0.01"),
        Line2D(
            [], [], marker="D", linestyle="", markerfacecolor=COLORS["pivot"],
            markeredgecolor=COLORS["plate"], color="none", label="Fixed rotation pivot",
        ),
        Line2D([], [], color=COLORS["plate"], linewidth=1.8, label="Fixed plate boundary"),
    ]
    legend_axis.axis("off")
    legend_axis.legend(
        handles=legend_handles,
        loc="center",
        ncol=4,
        fontsize=9.5,
        handletextpad=0.5,
        columnspacing=1.35,
    )
    fig.suptitle(
        "Fixed-plate alignment: input geometry and spot-to-spot recovery",
        x=0.5,
        y=0.995,
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.968,
        "Columns show baseline DLPFC 151675 rotations. In the 3D rows, the whole gray lower plane is the target slice; colored upper-plane spots are the cropped moving slice after alignment, with color encoding exact-pair error. Thin lines sample exact barcode pairs; z is a visual layer only.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["neutral"],
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.94, bottom=0.055)
    return fig, summary


def write_summary(path: Path, summary: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    configure_matplotlib()
    coordinates, names = read_spatial_coordinates(args.input)
    figure, summary = build_figure(coordinates, names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_dir / OUTPUT_STEM
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output_stem.with_suffix(f".{suffix}"), dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    write_summary(output_stem.with_name(f"{OUTPUT_STEM}_data.csv"), summary)

    for row in summary:
        print(
            f"{float(row['angle_degrees']):g}°: {int(row['retained']):,} retained, "
            f"{int(row['dropped']):,} dropped "
            f"({100 * float(row['retention_fraction']):.2f}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
