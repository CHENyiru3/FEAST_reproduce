#!/usr/bin/env python3
"""Render source-verified spatial alignment diagnostics for Study 02.

Every panel uses the same coordinate extent and the same reference-x colour
scale.  Light-gray points are the target reference geometry; coloured points
are the moving slice after the labelled transformation or alignment method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ["SOURCE_DATE_EPOCH"] = "0"

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_STEM = "alignment_spatial"
STUDY_OUTPUT = REPOSITORY_ROOT / "02_alignment" / "outputs" / "final_rerun_20260718_v2"
ROTATIONS_DIR = STUDY_OUTPUT / "rotations"
METHODS_DIR = STUDY_OUTPUT / "methods"
ROTATION_MANIFEST = ROTATIONS_DIR / "rotation_manifest.csv"
METHOD_MANIFEST = METHODS_DIR / "method_manifest.csv"

ANGLE = 45.0
ALTERATIONS = (
    ("baseline", "Baseline"),
    ("mean_0.5", "Mean ×0.5"),
    ("sparsity_0.5", "Sparsity ×0.5"),
    ("variance_2.0", "Variance ×2.0"),
)
METHODS = (("spateo", "Spateo"), ("paste", "PASTE"))

# These values are deliberately larger than the legacy map, while the dense
# point layer stays rasterized so vector exports retain editable text.
POINT_SIZE = 2.2
REFERENCE_POINT_SIZE = 0.75


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT))


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required manifest is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _verified_path(path_text: str, expected_hash: str, context: str) -> Path:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {context}: {path}")
    observed_hash = sha256(path)
    if observed_hash != expected_hash:
        raise ValueError(
            f"SHA-256 mismatch for {context}: expected {expected_hash}, observed {observed_hash}"
        )
    return path


def resolve_inputs() -> tuple[Path, dict[tuple[str, str], Path], dict[str, str]]:
    """Resolve and checksum-verify the 45° panels recorded by Study 02 manifests."""
    rotation_rows = _read_manifest(ROTATION_MANIFEST)
    method_rows = _read_manifest(METHOD_MANIFEST)
    required_alterations = {alteration for alteration, _label in ALTERATIONS}
    verified: dict[str, str] = {
        repository_path(ROTATION_MANIFEST): sha256(ROTATION_MANIFEST),
        repository_path(METHOD_MANIFEST): sha256(METHOD_MANIFEST),
    }

    reference_path: Path | None = None
    panels: dict[tuple[str, str], Path] = {}
    for alteration in required_alterations:
        matching = [
            row for row in rotation_rows
            if row.get("alteration") == alteration and float(row.get("angle_degrees", "nan")) == ANGLE
        ]
        if len(matching) != 1:
            raise ValueError(f"Expected one {alteration} rotation row at {ANGLE:g}°, found {len(matching)}")
        rotation = matching[0]
        rotated_path = _verified_path(
            rotation["output_path"], rotation["output_sha256"], f"{alteration} rotated input"
        )
        panels[(alteration, "rotated")] = rotated_path
        verified[repository_path(rotated_path)] = rotation["output_sha256"]

        source_path = _verified_path(
            rotation["source_path"], rotation["source_sha256"], f"{alteration} source geometry"
        )
        verified[repository_path(source_path)] = rotation["source_sha256"]
        if alteration == "baseline":
            reference_path = source_path

        for method, _label in METHODS:
            method_match = [
                row for row in method_rows
                if row.get("alteration") == alteration
                and row.get("method") == method
                and float(row.get("angle_degrees", "nan")) == ANGLE
                and row.get("status") == "ok"
            ]
            if len(method_match) != 1:
                raise ValueError(
                    f"Expected one successful {method} {alteration} row at {ANGLE:g}°, "
                    f"found {len(method_match)}"
                )
            artifacts = json.loads(method_match[0]["artifacts"])
            aligned = [item for item in artifacts if str(item.get("path", "")).endswith("/aligned.h5ad")]
            if len(aligned) != 1:
                raise ValueError(f"Missing aligned.h5ad artifact for {method} {alteration}")
            aligned_path = _verified_path(
                str(aligned[0]["path"]), str(aligned[0]["sha256"]),
                f"{method} {alteration} aligned coordinates",
            )
            panels[(alteration, method)] = aligned_path
            verified[repository_path(aligned_path)] = str(aligned[0]["sha256"])

    if reference_path is None:
        raise ValueError("Baseline source geometry was not resolved")
    return reference_path, panels, verified


def read_xy(path: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    data = ad.read_h5ad(path, backed="r")
    try:
        if key not in data.obsm:
            raise KeyError(f"{path} does not contain .obsm[{key!r}]")
        coords = np.asarray(data.obsm[key])[:, :2].astype(float)
        names = np.asarray(data.obs_names, dtype=str)
    finally:
        data.file.close()
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) != len(names):
        raise ValueError(f"Invalid spatial coordinate array in {path}")
    return coords, names


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "feast-study02-alignment-spatial",
    })


def _common_limits(reference_xy: np.ndarray, panel_coordinates: dict[tuple[str, str], np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    stacked = np.vstack([reference_xy, *panel_coordinates.values()])
    lower = stacked.min(axis=0)
    upper = stacked.max(axis=0)
    pad = 0.035 * np.maximum(upper - lower, 1.0)
    return (float(lower[0] - pad[0]), float(upper[0] + pad[0])), (float(lower[1] - pad[1]), float(upper[1] + pad[1]))


def write_plot_data(
    output_dir: Path,
    output_stem: str,
    reference_names: np.ndarray,
    reference_x: np.ndarray,
    panel_coordinates: dict[tuple[str, str], np.ndarray],
) -> Path:
    path = output_dir / f"{output_stem}_plot_data.csv"
    fieldnames = ["alteration", "panel", "spot_barcode", "x", "y", "reference_x"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for (alteration, panel), coords in panel_coordinates.items():
            for name, coordinate, x_ref in zip(reference_names, coords, reference_x):
                writer.writerow({
                    "alteration": alteration,
                    "panel": panel,
                    "spot_barcode": name,
                    "x": f"{coordinate[0]:.12g}",
                    "y": f"{coordinate[1]:.12g}",
                    "reference_x": f"{x_ref:.12g}",
                })
    return path


def draw_figure(
    output_dir: Path, output_stem: str, dpi: int,
) -> tuple[list[Path], Path, dict[str, str]]:
    configure_matplotlib()
    reference_path, panel_paths, verified_inputs = resolve_inputs()
    reference_xy, reference_names = read_xy(reference_path, "spatial")
    reference_x = reference_xy[:, 0].copy()

    panel_coordinates: dict[tuple[str, str], np.ndarray] = {}
    for alteration, _label in ALTERATIONS:
        for panel in ("rotated", *(method for method, _label in METHODS)):
            key = "spatial" if panel == "rotated" else "spatial_aligned"
            coords, names = read_xy(panel_paths[(alteration, panel)], key)
            if not np.array_equal(names, reference_names):
                raise ValueError(f"Spot order changed in the {alteration} {panel} panel")
            panel_coordinates[(alteration, panel)] = coords

    x_limits, y_limits = _common_limits(reference_xy, panel_coordinates)
    norm = Normalize(vmin=float(reference_x.min()), vmax=float(reference_x.max()))
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(len(ALTERATIONS), 1 + len(METHODS), figsize=(11.25, 12.1), squeeze=False)

    for row_index, (alteration, alteration_label) in enumerate(ALTERATIONS):
        for column_index, (panel, panel_label) in enumerate(
            (("rotated", f"Rotated input ({ANGLE:g}°)"), *METHODS)
        ):
            ax = axes[row_index, column_index]
            ax.scatter(
                reference_xy[:, 0], reference_xy[:, 1], s=REFERENCE_POINT_SIZE,
                color="#BDBDBD", alpha=0.48, marker="o", linewidths=0,
                rasterized=True, zorder=0,
            )
            xy = panel_coordinates[(alteration, panel)]
            ax.scatter(
                xy[:, 0], xy[:, 1], s=POINT_SIZE, c=reference_x,
                cmap=cmap, norm=norm, alpha=0.92, marker="o", linewidths=0,
                rasterized=True, zorder=1,
            )
            ax.set_xlim(*x_limits)
            ax.set_ylim(y_limits[1], y_limits[0])
            ax.set_aspect("equal")
            ax.set_axis_off()
            if row_index == 0:
                ax.set_title(panel_label, fontsize=11, fontweight="bold", pad=8)
        axes[row_index, 0].text(
            -0.16, 0.5, alteration_label,
            transform=axes[row_index, 0].transAxes,
            ha="right", va="center", fontsize=10, fontweight="bold",
        )

    fig.text(
        0.07, 0.982, f"Spatial alignment recovery — DLPFC 151675, {ANGLE:g}° rotation",
        ha="left", va="center", fontsize=14, fontweight="bold",
    )
    fig.text(
        0.07, 0.955,
        "Gray = target reference geometry. Colour = each spot's reference x-coordinate (shared scale). "
        "All panels use identical spatial limits.",
        ha="left", va="center", fontsize=8.2, color="#666666",
    )
    colorbar_axis = fig.add_axes((0.915, 0.145, 0.015, 0.69))
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axis)
    colorbar.set_label("Reference x-coordinate (shared scale)", fontsize=8.5, labelpad=7)
    colorbar.ax.tick_params(labelsize=7.5)
    fig.subplots_adjust(left=0.16, right=0.89, top=0.89, bottom=0.06, wspace=0.025, hspace=0.075)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / output_stem
    output_paths: list[Path] = []
    for suffix, kwargs in {
        "pdf": {"metadata": {"Creator": "FEAST Study 02 alignment spatial"}},
        "png": {"dpi": dpi, "metadata": {"Software": "FEAST Study 02 alignment spatial"}},
        "svg": {"metadata": {"Creator": "FEAST Study 02 alignment spatial"}},
    }.items():
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, **kwargs)
        output_paths.append(path)
    plt.close(fig)
    plot_data_path = write_plot_data(
        output_dir, output_stem, reference_names, reference_x, panel_coordinates
    )
    return output_paths, plot_data_path, verified_inputs


def write_provenance(
    output_dir: Path, output_stem: str, output_paths: list[Path],
    plot_data_path: Path, verified_inputs: dict[str, str],
) -> Path:
    payload = {
        "schema_version": 1,
        "figure_set": "study02_alignment_spatial_diagnostic",
        "promotion_status": "unpromoted_author_selection_required",
        "scientific_disposition": {
            "status": "author_scope_required",
            "figure_promotion_authorized": False,
            "aggregate_method_ranking_authorized": False,
            "winner_claim_authorized": False,
        },
        "source": {
            "plot_script": repository_path(Path(__file__)),
            "plot_script_sha256": sha256(Path(__file__)),
            "verified_inputs": verified_inputs,
        },
        "design": {
            "sample": "DLPFC 151675",
            "input_rotation_degrees": ANGLE,
            "alterations": [alteration for alteration, _label in ALTERATIONS],
            "columns": ["rotated_input", *(method for method, _label in METHODS)],
            "reference_encoding": "light-gray target geometry",
            "spot_encoding": "reference x-coordinate, shared viridis scale",
            "spatial_limits": "shared across every panel",
            "point_layer": "rasterized; text and annotations remain vector-editable",
        },
        "render_environment": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
        },
        "outputs": {
            path.name: sha256(path)
            for path in sorted([*output_paths, plot_data_path], key=lambda item: item.name)
        },
    }
    path = output_dir / f"{output_stem}_provenance.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-stem", default=OUTPUT_STEM)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_paths, plot_data_path, verified_inputs = draw_figure(
        args.output_dir, args.output_stem, args.dpi
    )
    provenance_path = write_provenance(
        args.output_dir, args.output_stem, output_paths, plot_data_path, verified_inputs
    )
    for path in [*output_paths, plot_data_path, provenance_path]:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
