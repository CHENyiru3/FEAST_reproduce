#!/usr/bin/env python3
"""Render the Study 03 Cell2location composition comparison for slice 100.

Fine labels are summed by their named ``class`` annotation before rendering.
Ground truth and Cell2location use identical named colour mapping, centres,
coordinate limits, and within-resolution pie radii.  Slice 100 has two
observed classes outside the 30-class reference legend; they are retained as
explicit 31st and 32nd entries rather than merged cosmetically.
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
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Wedge
from scipy.spatial import cKDTree


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PUBLICATION_MANIFEST = REPOSITORY_ROOT / "PUBLICATION_MANIFEST.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_STEM = "deconvolution_spatial"
STUDY_DIR = REPOSITORY_ROOT / "03_deconvolution" / "outputs" / "final_rerun_20260718_v2"
SIMULATION_MANIFEST = STUDY_DIR / "simulation_manifest.csv"
METHOD_MANIFEST = STUDY_DIR / "method_manifest.csv"

# The selected source is a manifest-verified, internally consistent 100 run.
SOURCE_CASE = "100"
RESOLUTIONS = ("0.25", "0.1")
SOURCE_LABELS = {
    "truth": "Ground truth",
    "cell2location": "Cell2location",
    "rctd": "RCTD",
}

# Fixed palette supplied for the reference-faithful 30-class colour system.
# Colours are retrieved by class name, never by a proportion-matrix position.
BASE_CELL_TYPES = (
    ("01 IT-ET Glut", "#436B92"),
    ("02 NP-CT-L6b Glut", "#A0B3CC"),
    ("03 OB-CR Glut", "#D77A47"),
    ("04 DG-IMN Glut", "#E6AE80"),
    ("05 OB-IMN GABA", "#58914E"),
    ("06 CTX-CGE GABA", "#95C28C"),
    ("07 CTX-MGE GABA", "#A83E3C"),
    ("08 CNU-MGE GABA", "#DC8E8B"),
    ("09 CNU-LGE GABA", "#786397"),
    ("10 LSX GABA", "#AD9FBB"),
    ("11 CNU-HYa GABA", "#73544E"),
    ("12 HY GABA", "#AA8E88"),
    ("13 CNU-HYa Glut", "#B2739E"),
    ("14 HY Glut", "#DBA8BC"),
    ("18 TH Glut", "#8A8A89"),
    ("19 MB Glut", "#C6C7C6"),
    ("20 MB GABA", "#A7AA55"),
    ("21 MB Dopa", "#C9CA90"),
    ("22 MB-HB Sero", "#5CADB2"),
    ("23 P Glut", "#9BC9CB"),
    ("24 MY Glut", "#D2D561"),
    ("26 P GABA", "#C1B598"),
    ("27 MY GABA", "#8BBF93"),
    ("28 CB GABA", "#4B4484"),
    ("29 CB Glut", "#AB719D"),
    ("30 Astro-Epen", "#68AF78"),
    ("31 OPC-Oligo", "#57676E"),
    ("32 OEC", "#74848B"),
    ("33 Vascular", "#8AB164"),
    ("34 Immune", "#D8E09F"),
)
ADDITIONAL_DISPLAY_TYPES = {
    "050": (("15 HY Gnrh1 Glut", "#496A81"),),
    "100": (("16 HY MM Glut", "#496A81"), ("25 Pineal Glut", "#7F6D4B")),
}
CELL_TYPES = BASE_CELL_TYPES
CELL_TYPE_ORDER = tuple(label for label, _colour in CELL_TYPES)
CELL_TYPE_COLORS = {label: colour for label, colour in CELL_TYPES}
SOURCE_CASE_SELECTION = "manifest-verified selected source slice; observed extra classes retained explicitly"

REFERENCE_POINT_SIZE = 0.36
FULL_RADIUS_SCALE = {"0.25": 0.34, "0.1": 0.18}
ROI_RADIUS_SCALE = 0.36
FULL_WEDGE_LINEWIDTH = 0.08
ROI_WEDGE_LINEWIDTH = 0.24
DISPLAY_THRESHOLD = 0.005
PROPORTION_EPSILON = 1e-9
ROI_X_FRACTION = (0.38, 0.60)
ROI_Y_FRACTION = (0.32, 0.60)


def configure_display_ontology(source_case: str, include_additional_classes: bool) -> None:
    """Select an exact ontology for a declared alternate-slice preview."""
    global SOURCE_CASE, CELL_TYPES, CELL_TYPE_ORDER, CELL_TYPE_COLORS, SOURCE_CASE_SELECTION
    if source_case not in {"007", "050", "100"}:
        raise ValueError(f"Unsupported registered source case: {source_case}")
    additional = ADDITIONAL_DISPLAY_TYPES.get(source_case, ())
    if additional and not include_additional_classes:
        labels = ", ".join(label for label, _color in additional)
        raise ValueError(
            f"Slice {source_case} contains classes outside the requested 30-class ontology: {labels}. "
            "Re-run with --include-additional-classes for a labeled preview rather than silently merging them."
        )
    SOURCE_CASE = source_case
    CELL_TYPES = BASE_CELL_TYPES + additional
    CELL_TYPE_ORDER = tuple(label for label, _colour in CELL_TYPES)
    CELL_TYPE_COLORS = {label: colour for label, colour in CELL_TYPES}
    if additional:
        SOURCE_CASE_SELECTION = (
            "registered alternate-slice preview with additional observed class(es) retained as explicit legend entries"
        )
    else:
        SOURCE_CASE_SELECTION = "only registered slice whose observed classes are all within the requested 30-class display ontology"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT))


def verified_path(path_text: str, expected_hash: str, context: str) -> Path:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {context}: {path}")
    observed_hash = sha256(path)
    if observed_hash != expected_hash:
        raise ValueError(
            f"SHA-256 mismatch for {context}: expected {expected_hash}, observed {observed_hash}"
        )
    return path


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_disposition() -> dict[str, object]:
    manifest = json.loads(PUBLICATION_MANIFEST.read_text())
    study = manifest["studies"]["03_deconvolution"]
    evidence = study["validated_evidence"]
    if study["status"] != "decision_required_headline_direction_changed":
        raise ValueError("Study 03 does not have the expected diagnostic-only disposition")
    for field in ("publication_claim_authorized", "aggregate_winner_ranking_authorized"):
        if evidence.get(field) is not False:
            raise ValueError(f"Study 03 disposition unexpectedly authorizes {field}")
    return study


def _manifest_row(rows: list[dict[str, str]], *, resolution: str, method: str | None = None) -> dict[str, str]:
    matches = [
        row for row in rows
        if row.get("slice_id") == SOURCE_CASE
        and float(row.get("resolution", "nan")) == float(resolution)
        and (method is None or row.get("method") == method)
        and row.get("status") == "ok"
    ]
    if len(matches) != 1:
        label = f"{SOURCE_CASE}, resolution {resolution}" if method is None else f"{SOURCE_CASE}, resolution {resolution}, {method}"
        raise ValueError(f"Expected one successful manifest row for {label}; found {len(matches)}")
    return matches[0]


def read_simulation_coordinates(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = ad.read_h5ad(path, backed="r")
    try:
        if "spatial" not in data.obsm:
            raise KeyError(f"{path} does not contain .obsm['spatial']")
        if "n_source_cells" not in data.obs:
            raise KeyError(f"{path} does not contain .obs['n_source_cells']")
        coords = np.asarray(data.obsm["spatial"])[:, :2].astype(float)
        names = np.asarray(data.obs_names, dtype=str)
        source_cells = data.obs["n_source_cells"].to_numpy(dtype=np.int64)
    finally:
        data.file.close()
    if (
        coords.ndim != 2 or coords.shape[1] != 2 or len(coords) != len(names)
        or source_cells.shape != (len(names),) or (source_cells < 0).any()
    ):
        raise ValueError(f"Invalid spatial coordinates in {path}")
    return coords, names, source_cells


def reference_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    data = ad.read_h5ad(path, backed="r")
    try:
        required = {"spatial"}
        if not required.issubset(data.obsm.keys()):
            raise KeyError(f"{path} lacks required .obsm['spatial']")
        required_obs = {"ground_truth", "class"}
        if not required_obs.issubset(data.obs.columns):
            raise KeyError(f"{path} lacks required observation fields: {sorted(required_obs)}")
        coords = np.asarray(data.obsm["spatial"])[:, :2].astype(float)
        names = np.asarray(data.obs_names, dtype=str)
        classes = np.asarray(data.obs["class"].astype(str), dtype=str)
        mapping_table = pd.DataFrame({
            "fine_label": data.obs["ground_truth"].astype(str),
            "display_class": data.obs["class"].astype(str),
        }).drop_duplicates()
    finally:
        data.file.close()
    if mapping_table["fine_label"].duplicated().any():
        raise ValueError("Fine labels do not map uniquely to display classes")
    if not set(classes).issubset(CELL_TYPE_COLORS):
        unknown = sorted(set(classes) - set(CELL_TYPE_COLORS))
        raise ValueError(f"Reference classes outside the requested display ontology: {unknown}")
    fine_to_class = dict(zip(mapping_table["fine_label"], mapping_table["display_class"], strict=True))
    return coords, names, classes, fine_to_class


def spot_id(name: str) -> str:
    value = str(name)
    return value if value.startswith("spot_") else f"spot_{value}"


def aggregate_proportions(
    path: Path,
    simulation_names: np.ndarray,
    fine_to_class: dict[str, str],
    source_cells: np.ndarray,
    *,
    allow_missing_empty_rows: bool = False,
    empty_row_expectation: str = "zero",
) -> np.ndarray:
    proportions = pd.read_csv(path, index_col=0)
    proportions.index = proportions.index.astype(str)
    expected_index = pd.Index([spot_id(name) for name in simulation_names])
    if proportions.index.duplicated().any():
        raise ValueError(f"Duplicate proportion rows: {path}")
    observed_index = pd.Index(proportions.index)
    extra_rows = observed_index.difference(expected_index)
    missing_rows = expected_index.difference(observed_index)
    if len(extra_rows):
        raise ValueError(f"Proportion rows are not simulation spots: {path}")
    if len(missing_rows):
        if not allow_missing_empty_rows:
            raise ValueError(f"Proportion rows do not exactly match simulation spots: {path}")
        source_by_id = dict(zip(expected_index, source_cells, strict=True))
        nonempty_missing = [name for name in missing_rows if source_by_id[name] > 0]
        if nonempty_missing:
            raise ValueError(f"Non-empty simulation spots missing from {path}: {nonempty_missing[:4]}")
    if proportions.columns.duplicated().any():
        raise ValueError(f"Duplicate fine-label columns in {path}")
    fine_labels = [str(column) for column in proportions.columns]
    unknown_labels = sorted(set(fine_labels) - set(fine_to_class))
    if unknown_labels:
        raise ValueError(f"Unmapped fine labels in {path}: {unknown_labels[:8]}")
    display_labels = [fine_to_class[label] for label in fine_labels]
    if not set(display_labels).issubset(CELL_TYPE_COLORS):
        raise ValueError(f"Display labels outside requested 30-class ontology: {sorted(set(display_labels) - set(CELL_TYPE_COLORS))}")
    aligned = proportions.reindex(expected_index, fill_value=0.0)
    result = np.zeros((len(aligned), len(CELL_TYPE_ORDER)), dtype=float)
    for class_index, cell_type in enumerate(CELL_TYPE_ORDER):
        columns = [column for column, display in zip(fine_labels, display_labels, strict=True) if display == cell_type]
        if columns:
            result[:, class_index] = aligned.loc[:, columns].to_numpy(dtype=float).sum(axis=1)
    if not np.isfinite(result).all() or (result < -1e-12).any():
        raise ValueError(f"Invalid proportions after aggregation: {path}")
    result = np.clip(result, 0.0, None)
    totals = result.sum(axis=1)
    positive = source_cells > 0
    if not np.allclose(totals[positive], 1.0, atol=1e-5):
        raise ValueError(f"Non-empty proportions do not sum to one: {path}")
    empty_totals = totals[~positive]
    if empty_row_expectation == "zero":
        if not np.allclose(empty_totals, 0.0, atol=1e-12):
            raise ValueError(f"Empty proportions are not zero: {path}")
    elif empty_row_expectation == "one":
        if not np.allclose(empty_totals, 1.0, atol=1e-5):
            raise ValueError(f"Empty proportions are not normalized: {path}")
    else:
        raise ValueError(f"Unsupported empty-row expectation: {empty_row_expectation}")
    if tuple(CELL_TYPE_ORDER) != tuple(label for label, _colour in CELL_TYPES):
        raise AssertionError("Display columns are not in the declared named order")
    return result


def nearest_neighbor_distance(coords: np.ndarray) -> float:
    if len(coords) < 2:
        raise ValueError("At least two spatial centres are required")
    distances, _ = cKDTree(coords).query(coords, k=2)
    return float(np.median(distances[:, 1]))


def verified_candidate_cell2location_path(
    candidate_dir: Path, simulation_row: dict[str, str], resolution: str,
) -> tuple[Path, Path]:
    """Verify a label-matched Cell2location corrective output before plotting."""
    path = (
        candidate_dir / "methods" / "cell2location" / SOURCE_CASE
        / f"resolution_{float(resolution):g}_proportions.csv"
    )
    metadata_path = path.with_name(path.stem + "_metadata.json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing corrective Cell2location artifacts for resolution ×{resolution}")
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("status") != "validated_success"
        or metadata.get("input_sha256") != simulation_row["simulation_sha256"]
        or metadata.get("reference_sha256") != simulation_row["reference_sha256"]
        or metadata.get("output_sha256") != sha256(path)
        or not metadata.get("reference_label_filter", {}).get("min_cells_per_type")
        or not metadata.get("reference_label_filter", {}).get("n_reference_cell_types_retained")
        or metadata.get("zero_library_policy", "").split(";")[0]
        != "excluded from training and reinserted as zero rows"
    ):
        raise ValueError(f"Corrective Cell2location metadata is not valid for resolution ×{resolution}")
    return path, metadata_path


def load_data(
    cell2location_candidate_dir: Path | None = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray],
    dict[tuple[str, str], np.ndarray], dict[str, str],
]:
    check_disposition()
    simulation_rows = read_manifest(SIMULATION_MANIFEST)
    method_rows = read_manifest(METHOD_MANIFEST)
    verified_inputs: dict[str, str] = {
        repository_path(PUBLICATION_MANIFEST): sha256(PUBLICATION_MANIFEST),
        repository_path(SIMULATION_MANIFEST): sha256(SIMULATION_MANIFEST),
        repository_path(METHOD_MANIFEST): sha256(METHOD_MANIFEST),
    }

    first_row = _manifest_row(simulation_rows, resolution=RESOLUTIONS[0])
    reference_path = verified_path(first_row["reference_path"], first_row["reference_sha256"], "MERFISH reference")
    verified_inputs[repository_path(reference_path)] = first_row["reference_sha256"]
    reference_xy, reference_names, reference_classes, fine_to_class = reference_data(reference_path)

    coordinates: dict[str, np.ndarray] = {}
    source_cells_by_resolution: dict[str, np.ndarray] = {}
    compositions: dict[tuple[str, str], np.ndarray] = {}
    for resolution in RESOLUTIONS:
        simulation_row = _manifest_row(simulation_rows, resolution=resolution)
        if simulation_row["reference_path"] != first_row["reference_path"]:
            raise ValueError("The two displayed resolutions do not share the same reference slice")
        simulation_path = verified_path(
            simulation_row["simulation_path"], simulation_row["simulation_sha256"],
            f"resolution ×{resolution} simulation",
        )
        truth_path = verified_path(
            simulation_row["truth_path"], simulation_row["truth_sha256"],
            f"resolution ×{resolution} ground truth",
        )
        verified_inputs[repository_path(simulation_path)] = simulation_row["simulation_sha256"]
        verified_inputs[repository_path(truth_path)] = simulation_row["truth_sha256"]
        xy, simulation_names, source_cells = read_simulation_coordinates(simulation_path)
        coordinates[resolution] = xy
        source_cells_by_resolution[resolution] = source_cells
        compositions[(resolution, "truth")] = aggregate_proportions(
            truth_path, simulation_names, fine_to_class, source_cells,
            empty_row_expectation="zero",
        )
        for method in ("cell2location", "rctd"):
            if method == "cell2location" and cell2location_candidate_dir is not None:
                output_path, metadata_path = verified_candidate_cell2location_path(
                    cell2location_candidate_dir, simulation_row, resolution,
                )
                verified_inputs[repository_path(output_path)] = sha256(output_path)
                verified_inputs[repository_path(metadata_path)] = sha256(metadata_path)
            else:
                method_row = _manifest_row(method_rows, resolution=resolution, method=method)
                if method_row["simulation_path"] != simulation_row["simulation_path"]:
                    raise ValueError(f"{method} does not reference the displayed simulation at resolution {resolution}")
                output_path = verified_path(
                    method_row["output_path"], method_row["output_sha256"],
                    f"{method} proportions at resolution ×{resolution}",
                )
                verified_inputs[repository_path(output_path)] = method_row["output_sha256"]
            compositions[(resolution, method)] = aggregate_proportions(
                output_path, simulation_names, fine_to_class, source_cells,
                allow_missing_empty_rows=(method == "rctd"),
                empty_row_expectation="zero" if (
                    method == "rctd" or cell2location_candidate_dir is not None and method == "cell2location"
                ) else "one",
            )
        expected_rows = len(xy)
        for source in ("truth", "cell2location", "rctd"):
            if compositions[(resolution, source)].shape != (expected_rows, len(CELL_TYPE_ORDER)):
                raise ValueError(f"Unexpected composition shape for {source}, resolution {resolution}")
    return (
        reference_xy, reference_names, reference_classes, coordinates,
        source_cells_by_resolution, compositions, verified_inputs,
    )


def common_limits(reference_xy: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    lower = reference_xy.min(axis=0)
    upper = reference_xy.max(axis=0)
    padding = 0.03 * (upper - lower)
    return (float(lower[0] - padding[0]), float(upper[0] + padding[0])), (float(lower[1] - padding[1]), float(upper[1] + padding[1]))


def roi_bounds(x_limits: tuple[float, float], y_limits: tuple[float, float]) -> tuple[float, float, float, float]:
    return (
        x_limits[0] + ROI_X_FRACTION[0] * (x_limits[1] - x_limits[0]),
        x_limits[0] + ROI_X_FRACTION[1] * (x_limits[1] - x_limits[0]),
        y_limits[0] + ROI_Y_FRACTION[0] * (y_limits[1] - y_limits[0]),
        y_limits[0] + ROI_Y_FRACTION[1] * (y_limits[1] - y_limits[0]),
    )


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9.5,
        "axes.titlesize": 11.2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "feast-study03-composition-comparison",
    })


def format_spatial_axis(ax: plt.Axes, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> None:
    ax.set_xlim(*x_limits)
    ax.set_ylim(y_limits[1], y_limits[0])
    ax.set_aspect("equal")
    ax.set_axis_off()


def pie_collection(
    xy: np.ndarray, proportions: np.ndarray, radius: float, *,
    crop: tuple[float, float, float, float] | None = None,
    linewidth: float = FULL_WEDGE_LINEWIDTH,
) -> PatchCollection:
    """Draw one named, clockwise pie glyph per location.

    The display threshold removes only wedges below 0.5% and the retained
    components are renormalized.  This makes sub-pixel slivers legible without
    changing the stored composition data or the order of the remaining wedges.
    """
    patches: list[Wedge] = []
    colors: list[str] = []
    if crop is None:
        selected = np.ones(len(xy), dtype=bool)
    else:
        x0, x1, y0, y1 = crop
        selected = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    for coordinate, values in zip(xy[selected], proportions[selected], strict=True):
        total = float(values.sum())
        if total <= PROPORTION_EPSILON:
            continue
        displayed = np.asarray(values, dtype=float) / total
        displayed[displayed < DISPLAY_THRESHOLD] = 0.0
        displayed_total = float(displayed.sum())
        if displayed_total <= PROPORTION_EPSILON:
            displayed[np.argmax(values)] = 1.0
        else:
            displayed /= displayed_total
        start_angle = 90.0
        for value, cell_type in zip(displayed, CELL_TYPE_ORDER, strict=True):
            if value <= PROPORTION_EPSILON:
                continue
            end_angle = start_angle - 360.0 * float(value)
            patches.append(Wedge(tuple(coordinate), radius, end_angle, start_angle))
            colors.append(CELL_TYPE_COLORS[cell_type])
            start_angle = end_angle
    return PatchCollection(
        patches, facecolor=colors, edgecolor="white", linewidth=linewidth,
        alpha=0.94, antialiased=True, rasterized=True, zorder=2,
    )


def add_composition_panel(
    ax: plt.Axes, xy: np.ndarray, proportions: np.ndarray, radius: float,
    x_limits: tuple[float, float], y_limits: tuple[float, float],
) -> None:
    format_spatial_axis(ax, x_limits, y_limits)
    ax.add_collection(pie_collection(xy, proportions, radius, linewidth=FULL_WEDGE_LINEWIDTH))


def add_roi_rectangle(parent: plt.Axes, roi: tuple[float, float, float, float]) -> None:
    """Mark the fixed ROI without annotating or obscuring the tissue."""
    x0, x1, y0, y1 = roi
    parent.add_patch(Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        fill=False, edgecolor="#D9534F", linewidth=0.8, zorder=5,
    ))


def add_roi_panel(
    ax: plt.Axes, xy: np.ndarray, proportions: np.ndarray, radius: float,
    roi: tuple[float, float, float, float],
) -> None:
    """Render an externally placed, identically scaled detailed crop."""
    x0, x1, y0, y1 = roi
    ax.set_facecolor("white")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.add_collection(pie_collection(
        xy, proportions, radius, crop=roi, linewidth=ROI_WEDGE_LINEWIDTH,
    ))


def draw_legend(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.985, "Cell Type (Proportion)", ha="left", va="top", fontsize=10.8, fontweight="bold")
    entries_per_column = int(np.ceil(len(CELL_TYPES) / 3))
    row_spacing = 0.073 if entries_per_column <= 11 else 0.067
    for index, (label, color) in enumerate(CELL_TYPES):
        column = index // entries_per_column
        row = index % entries_per_column
        x = (0.00, 0.335, 0.67)[column]
        y = 0.895 - row_spacing * row
        ax.add_patch(Rectangle((x, y - 0.017), 0.027, 0.034, facecolor=color, edgecolor="none"))
        ax.text(x + 0.036, y, label, ha="left", va="center", fontsize=7.15, color="#222222")


def write_plot_data(
    output_dir: Path,
    reference_xy: np.ndarray, reference_names: np.ndarray, reference_classes: np.ndarray,
    coordinates: dict[str, np.ndarray], source_cells_by_resolution: dict[str, np.ndarray],
    compositions: dict[tuple[str, str], np.ndarray], output_stem: str,
) -> tuple[Path, Path]:
    data_path = output_dir / f"{output_stem}_plot_data.csv"
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["panel", "resolution", "unit_id", "x", "y", "n_source_cells", "cell_type", "proportion"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for name, coordinate, cell_type in zip(reference_names, reference_xy, reference_classes, strict=True):
            writer.writerow({
                "panel": "single_cell_reference", "resolution": "single_cell", "unit_id": name,
                "x": f"{coordinate[0]:.12g}", "y": f"{coordinate[1]:.12g}",
                "n_source_cells": "1", "cell_type": cell_type, "proportion": "1",
            })
        for resolution in RESOLUTIONS:
            for source in ("truth", "cell2location", "rctd"):
                for spot_index, (coordinate, values) in enumerate(
                    zip(coordinates[resolution], compositions[(resolution, source)], strict=True)
                ):
                    for cell_type, value in zip(CELL_TYPE_ORDER, values, strict=True):
                        writer.writerow({
                            "panel": source, "resolution": resolution, "unit_id": f"spot_{spot_index}",
                            "x": f"{coordinate[0]:.12g}", "y": f"{coordinate[1]:.12g}",
                            "n_source_cells": int(source_cells_by_resolution[resolution][spot_index]),
                            "cell_type": cell_type, "proportion": f"{value:.12g}",
                        })
    palette_path = output_dir / f"{output_stem}_palette.csv"
    with palette_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["display_order", "cell_type", "hex_color", "present_in_source_slice"], lineterminator="\n")
        writer.writeheader()
        for order, (cell_type, color) in enumerate(CELL_TYPES, start=1):
            writer.writerow({
                "display_order": order, "cell_type": cell_type, "hex_color": color,
                "present_in_source_slice": cell_type in set(reference_classes),
            })
    return data_path, palette_path


def draw_figure(
    output_dir: Path, output_stem: str, dpi: int,
    cell2location_candidate_dir: Path | None = None,
) -> tuple[list[Path], tuple[Path, Path], dict[str, str]]:
    configure_matplotlib()
    (
        reference_xy, reference_names, reference_classes, coordinates,
        source_cells_by_resolution, compositions, verified_inputs,
    ) = load_data(cell2location_candidate_dir)
    x_limits, y_limits = common_limits(reference_xy)
    nearest_neighbour = {
        resolution: nearest_neighbor_distance(coordinates[resolution])
        for resolution in RESOLUTIONS
    }
    radii = {
        resolution: FULL_RADIUS_SCALE[resolution] * nearest_neighbour[resolution]
        for resolution in RESOLUTIONS
    }
    roi_radius = ROI_RADIUS_SCALE * nearest_neighbour["0.1"]
    roi = roi_bounds(x_limits, y_limits)
    # A zero-source-cell bin has no ground-truth composition.  Cell2location
    # nevertheless emits a normalized prediction there, whereas RCTD omits it.
    # Hide those bins for *all* sources so an apparent tissue-outline difference
    # cannot arise solely from this output-format convention.
    rendered_compositions: dict[tuple[str, str], np.ndarray] = {}
    for resolution in RESOLUTIONS:
        populated = source_cells_by_resolution[resolution] > 0
        for source in ("truth", "cell2location", "rctd"):
            displayed = compositions[(resolution, source)].copy()
            displayed[~populated] = 0.0
            rendered_compositions[(resolution, source)] = displayed

    # Main relationship: reference / ground truth in row 1, Cell2location in
    # row 2 and RCTD in row 3.  The last narrow column contains separate ROI
    # axes so none of the magnifications obscures a tissue panel.
    fig = plt.figure(figsize=(16.2, 10.2))
    grid = fig.add_gridspec(
        3, 4, left=0.025, right=0.99, top=0.965, bottom=0.040,
        width_ratios=(1.0, 1.0, 1.0, 0.34), wspace=0.075, hspace=0.150,
    )
    axes = {
        "reference": fig.add_subplot(grid[0, 0]),
        "truth_025": fig.add_subplot(grid[0, 1]),
        "truth_010": fig.add_subplot(grid[0, 2]),
        "truth_roi": fig.add_subplot(grid[0, 3]),
        "legend": fig.add_subplot(grid[1:, 0]),
        "cell2location_025": fig.add_subplot(grid[1, 1]),
        "cell2location_010": fig.add_subplot(grid[1, 2]),
        "cell2location_roi": fig.add_subplot(grid[1, 3]),
        "rctd_025": fig.add_subplot(grid[2, 1]),
        "rctd_010": fig.add_subplot(grid[2, 2]),
        "rctd_roi": fig.add_subplot(grid[2, 3]),
    }

    format_spatial_axis(axes["reference"], x_limits, y_limits)
    axes["reference"].scatter(
        reference_xy[:, 0], reference_xy[:, 1],
        s=REFERENCE_POINT_SIZE,
        c=[CELL_TYPE_COLORS[cell_type] for cell_type in reference_classes],
        marker="o", linewidths=0, alpha=0.88, rasterized=True, zorder=2,
    )
    axes["reference"].set_title("A  Single-cell MERFISH", loc="left", fontsize=11.2, fontweight="bold", pad=7)

    panel_specs = (
        ("truth_025", "truth", "0.25", "B  Ground truth — resolution ×0.25"),
        ("truth_010", "truth", "0.1", "C  Ground truth — resolution ×0.10"),
        ("cell2location_025", "cell2location", "0.25", "D  Cell2location — resolution ×0.25"),
        ("cell2location_010", "cell2location", "0.1", "E  Cell2location — resolution ×0.10"),
        ("rctd_025", "rctd", "0.25", "F  RCTD — resolution ×0.25"),
        ("rctd_010", "rctd", "0.1", "G  RCTD — resolution ×0.10"),
    )
    for axis_key, source, resolution, title in panel_specs:
        add_composition_panel(
            axes[axis_key], coordinates[resolution], rendered_compositions[(resolution, source)], radii[resolution], x_limits, y_limits
        )
        axes[axis_key].set_title(title, loc="left", fontsize=11.2, fontweight="bold", pad=7)
    for axis_key in ("truth_010", "cell2location_010", "rctd_010"):
        add_roi_rectangle(axes[axis_key], roi)
    for axis_key, source in (
        ("truth_roi", "truth"),
        ("cell2location_roi", "cell2location"),
        ("rctd_roi", "rctd"),
    ):
        add_roi_panel(axes[axis_key], coordinates["0.1"], rendered_compositions[("0.1", source)], roi_radius, roi)
        axes[axis_key].set_title("Enlarged ROI", fontsize=8.4, fontweight="bold", pad=4)
    draw_legend(axes["legend"])

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / output_stem
    outputs: list[Path] = []
    for suffix, kwargs in {
        "pdf": {"metadata": {"Creator": "FEAST Study 03 spatial deconvolution composition comparison"}},
        "png": {"dpi": dpi, "metadata": {"Software": "FEAST Study 03 spatial deconvolution composition comparison"}},
        "svg": {"metadata": {"Creator": "FEAST Study 03 spatial deconvolution composition comparison"}},
    }.items():
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    data_paths = write_plot_data(
        output_dir, reference_xy, reference_names, reference_classes, coordinates,
        source_cells_by_resolution, compositions, output_stem,
    )
    return outputs, data_paths, verified_inputs


def write_provenance(
    output_dir: Path, output_stem: str, output_paths: list[Path], data_paths: tuple[Path, Path], verified_inputs: dict[str, str]
) -> Path:
    payload = {
        "schema_version": 1,
        "figure_set": "study03_spatial_deconvolution_composition_diagnostic",
        "publication_status": "scientific_stop_headline_direction_changed",
        "scientific_disposition": {
            "scientific_stop": True,
            "publication_claim_authorized": False,
            "aggregate_winner_ranking_authorized": False,
            "figure_promotion_authorized": False,
        },
        "source": {
            "plot_script": repository_path(Path(__file__)),
            "plot_script_sha256": sha256(Path(__file__)),
            "verified_inputs": verified_inputs,
        },
        "design": {
            "source_slice": f"Zhuang-ABCA-1.{SOURCE_CASE}",
            "source_slice_selection": SOURCE_CASE_SELECTION,
            "resolutions": list(RESOLUTIONS),
            "methods": ["Cell2location", "RCTD"],
            "cell_type_order": list(CELL_TYPE_ORDER),
            "additional_display_classes": [
                label for label, _colour in ADDITIONAL_DISPLAY_TYPES.get(SOURCE_CASE, ())
            ],
            "composition_glyph": {
                "type": "clockwise pie wedges",
                "start_angle_degrees": 90,
                "cell_type_order": "fixed shared named order",
                "display_threshold": DISPLAY_THRESHOLD,
                "full_panel_radius_fraction_of_median_nearest_neighbour_distance": FULL_RADIUS_SCALE,
                "roi_radius_fraction_of_median_nearest_neighbour_distance": ROI_RADIUS_SCALE,
                "full_wedge_separator_linewidth": FULL_WEDGE_LINEWIDTH,
                "roi_wedge_separator_linewidth": ROI_WEDGE_LINEWIDTH,
            },
            "paired_glyph_contract": "Ground truth, Cell2location, and RCTD use the same centres, radii, colours, order, background, and coordinate limits within each resolution",
            "empty_location_display": "Locations with n_source_cells = 0 are hidden consistently from all three composition sources; Cell2location's normalized empty-location output is retained in the plot-data table but not rendered.",
            "roi": {
                "selection_rule": "fixed central-tissue fractions of the shared reference bounds; independent of prediction values",
                "x_fraction": list(ROI_X_FRACTION),
                "y_fraction": list(ROI_Y_FRACTION),
                "applied_to": [
                    "ground_truth_resolution_0.1",
                    "cell2location_resolution_0.1",
                    "rctd_resolution_0.1",
                ],
                "layout": "separate right-hand axes; no overlay or connector lines",
            },
            "point_layers": "rasterized; titles, legend, and ROI rectangles remain vector-editable",
        },
        "render_environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
        "outputs": {
            path.name: sha256(path)
            for path in sorted([*output_paths, *data_paths], key=lambda item: item.name)
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
    parser.add_argument("--source-case", choices=("007", "050", "100"), default=SOURCE_CASE)
    parser.add_argument(
        "--cell2location-candidate-dir", type=Path,
        help="Use a verified label-matched Cell2location corrective output root.",
    )
    parser.add_argument(
        "--include-additional-classes", action="store_true",
        help="For alternate-slice previews, add source classes that are absent from the 30-class primary ontology.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Slice 100 is the selected final source and has two biologically retained
    # classes beyond the reference's 30-class legend.  Other alternate previews
    # still require an explicit opt-in to retain their extra classes.
    include_additional_classes = args.include_additional_classes or args.source_case == "100"
    configure_display_ontology(args.source_case, include_additional_classes)
    output_paths, data_paths, verified_inputs = draw_figure(
        args.output_dir, args.output_stem, args.dpi, args.cell2location_candidate_dir,
    )
    provenance_path = write_provenance(args.output_dir, args.output_stem, output_paths, data_paths, verified_inputs)
    for path in [*output_paths, *data_paths, provenance_path]:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
