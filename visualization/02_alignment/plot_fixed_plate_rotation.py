#!/usr/bin/env python3
"""Illustrate rotation-induced spot loss inside a fixed spatial plate."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPOSITORY_ROOT / "02_alignment" / "data" / "local" / "151675.h5ad"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_STEM = "fixed_plate_rotation_illustration"

ANGLES = (5.0, 15.0, 25.0, 35.0, 45.0, 60.0, 75.0)
PIVOT_FRACTION = 0.5

COLORS = {
    "retained": "#3775BA",
    "dropped": "#B64342",
    "plate": "#272727",
    "pivot": "#FFD700",
    "neutral": "#767676",
}


def read_spatial_coordinates(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = ad.read_h5ad(path, backed="r")
    try:
        if "spatial" not in data.obsm:
            raise KeyError(f"{path} does not contain .obsm['spatial']")
        coordinates = np.asarray(data.obsm["spatial"], dtype=np.float64)[:, :2].copy()
    finally:
        data.file.close()
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"Invalid spatial coordinates: {coordinates.shape}")
    if not np.isfinite(coordinates).all():
        raise ValueError("Spatial coordinates contain non-finite values")
    return coordinates


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
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
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
            "svg.hashsalt": "feast-fixed-plate-rotation-illustration",
        }
    )


def build_figure(
    coordinates: np.ndarray,
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

    fig = plt.figure(figsize=(20.0, 8.2))
    grid = fig.add_gridspec(2, len(ANGLES), height_ratios=(1.3, 0.8), hspace=0.28)
    spatial_axes = [fig.add_subplot(grid[0, index]) for index in range(len(ANGLES))]
    curve_axis = fig.add_subplot(grid[1, :])

    for index, (ax, (angle, rotated, retained), row) in enumerate(
        zip(spatial_axes, rotations, summary)
    ):
        dropped = ~retained
        ax.scatter(
            rotated[dropped, 0],
            rotated[dropped, 1],
            s=4.0,
            color=COLORS["dropped"],
            alpha=0.72,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        ax.scatter(
            rotated[retained, 0],
            rotated[retained, 1],
            s=3.0,
            color=COLORS["retained"],
            alpha=0.84,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        ax.add_patch(
            Rectangle(
                lower,
                *(upper - lower),
                fill=False,
                edgecolor=COLORS["plate"],
                linewidth=1.8,
                zorder=3,
            )
        )
        ax.scatter(
            [pivot[0]],
            [pivot[1]],
            s=48,
            marker="D",
            facecolor=COLORS["pivot"],
            edgecolor=COLORS["plate"],
            linewidth=0.8,
            zorder=4,
        )
        ax.set_xlim(limits_lower[0] - padding[0], limits_upper[0] + padding[0])
        ax.set_ylim(limits_lower[1] - padding[1], limits_upper[1] + padding[1])
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(
            f"{angle:g}°\n{int(row['retained']):,} retained "
            f"({100 * float(row['retention_fraction']):.1f}%)",
            fontweight="bold",
            pad=7,
        )
        ax.text(
            0.01,
            0.99,
            chr(ord("a") + index),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

    angles = np.asarray([float(row["angle_degrees"]) for row in summary])
    retention = np.asarray([100 * float(row["retention_fraction"]) for row in summary])
    curve_axis.plot(
        angles,
        retention,
        color=COLORS["retained"],
        marker="o",
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=1.0,
        linewidth=2.6,
        zorder=3,
    )
    curve_axis.fill_between(angles, 50.0, retention, color="#DDF3DE", alpha=0.48, zorder=0)
    for angle, value in zip(angles, retention):
        curve_axis.annotate(
            f"{value:.1f}%",
            (angle, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COLORS["plate"],
        )
    curve_axis.set_xlim(2, 78)
    curve_axis.set_ylim(50, 103)
    curve_axis.set_xticks(angles, [f"{angle:g}°" for angle in angles])
    curve_axis.set_yticks([50, 60, 70, 80, 90, 100])
    curve_axis.set_xlabel("Rotation angle")
    curve_axis.set_ylabel("Spots retained inside fixed plate (%)")
    curve_axis.grid(axis="y", color="#CFCECE", linewidth=0.8, alpha=0.65)
    curve_axis.set_title("Overlap range used for the alignment stress test", fontweight="bold", pad=10)

    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color=COLORS["retained"], label="Retained inside plate"),
        Line2D([], [], marker="o", linestyle="", color=COLORS["dropped"], label="Dropped outside plate"),
        Line2D(
            [], [], marker="D", linestyle="", markerfacecolor=COLORS["pivot"],
            markeredgecolor=COLORS["plate"], color="none", label="Fixed rotation pivot",
        ),
        Line2D([], [], color=COLORS["plate"], linewidth=1.8, label="Fixed plate boundary"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        fontsize=10,
        handletextpad=0.5,
        columnspacing=1.8,
    )
    fig.suptitle(
        "Fixed-plate rotation progressively reduces spatial overlap",
        x=0.5,
        y=0.995,
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.958,
        "DLPFC 151675 · plate = original coordinate bounds · "
        "pivot = centre + 0.5 × plate half-extent",
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["neutral"],
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.865, bottom=0.095)
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
    coordinates = read_spatial_coordinates(args.input)
    figure, summary = build_figure(coordinates)
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
