"""Render the Study 03 matched-support cell-class comparison for slice 100.

Every panel uses the exact evaluation ontology: the slice-specific classes
with at least 50 reference cells plus ``Other`` for rarer truth classes.
Ground truth, Cell2location, and RCTD therefore share the same named colours,
centres, coordinate limits, and within-resolution pie radii.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import platform
import sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ['SOURCE_DATE_EPOCH'] = '0'
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Wedge
from scipy.spatial import cKDTree
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / 'figures'
OUTPUT_STEM = 'deconvolution_spatial'
STUDY_DIR = REPOSITORY_ROOT / '03_deconvolution' / 'outputs' / 'cell_class_rerun_20260810_v1'
SIMULATION_MANIFEST = STUDY_DIR / 'simulation_manifest.csv'
METHOD_MANIFEST = STUDY_DIR / 'method_manifest.csv'
VALIDATION_PATH = STUDY_DIR / 'validation_cell_class.csv'
SUPPORT_AUDIT_PATH = STUDY_DIR / 'scores_cell_class' / 'support_audit.csv'
SOURCE_CASE = '100'
RESOLUTIONS = ('0.25', '0.1')
SOURCE_LABELS = {'truth': 'Ground truth', 'cell2location': 'Cell2location', 'rctd': 'RCTD'}
BASE_CELL_TYPES = (('01 IT-ET Glut', '#3B73A5'), ('02 NP-CT-L6b Glut', '#5264C3'), ('03 OB-CR Glut', '#D66F36'), ('04 DG-IMN Glut', '#D997CC'), ('05 OB-IMN GABA', '#4C8B4A'), ('06 CTX-CGE GABA', '#8BAE3E'), ('07 CTX-MGE GABA', '#B94945'), ('08 CNU-MGE GABA', '#C9686A'), ('09 CNU-LGE GABA', '#8066A8'), ('10 LSX GABA', '#7189C2'), ('11 CNU-HYa GABA', '#8C6352'), ('12 HY GABA', '#A76B78'), ('13 CNU-HYa Glut', '#B25B9F'), ('14 HY Glut', '#CC7A9A'), ('18 TH Glut', '#6C7072'), ('19 MB Glut', '#577F5E'), ('20 MB GABA', '#857526'), ('21 MB Dopa', '#C9A43A'), ('22 MB-HB Sero', '#3E9FA6'), ('23 P Glut', '#3E9FA6'), ('24 MY Glut', '#B9B43B'), ('26 P GABA', '#9361B0'), ('27 MY GABA', '#3B9C87'), ('28 CB GABA', '#586EAE'), ('29 CB Glut', '#A84C79'), ('30 Astro-Epen', '#32A36F'), ('31 OPC-Oligo', '#3F7181'), ('32 OEC', '#69758F'), ('33 Vascular', '#7A4E39'), ('34 Immune', '#A84C79'))
ADDITIONAL_DISPLAY_TYPES = {'050': (('15 HY Gnrh1 Glut', '#267F9C'),), '100': (('16 HY MM Glut', '#267F9C'), ('25 Pineal Glut', '#8F6D4B'))}
OTHER_CELL_CLASS = ('Other', '#8A8A8A')
FULL_CELL_TYPES = tuple(sorted(BASE_CELL_TYPES + tuple((item for additions in ADDITIONAL_DISPLAY_TYPES.values() for item in additions)), key=lambda item: int(item[0].split(' ', maxsplit=1)[0])))
FULL_CELL_TYPE_COLORS = {label: colour for label, colour in FULL_CELL_TYPES}
CELL_TYPES = BASE_CELL_TYPES
CELL_TYPE_ORDER = tuple((label for label, _colour in CELL_TYPES))
CELL_TYPE_COLORS = {label: colour for label, colour in CELL_TYPES}
RETAINED_CELL_CLASSES: tuple[str, ...] = ()
SOURCE_CASE_SELECTION = 'manifest-verified selected source slice with exact scored support'
REFERENCE_POINT_SIZE = 0.43
FULL_RADIUS_SCALE = {'0.25': 0.37, '0.1': 0.38}
ROI_RADIUS_SCALE = 0.41
FULL_WEDGE_LINEWIDTH = 0.0
ROI_WEDGE_LINEWIDTH = 0.18
DISPLAY_THRESHOLD = 0.005
PROPORTION_EPSILON = 1e-09
ROI_X_FRACTION = (0.5, 0.72)
ROI_Y_FRACTION = (0.22, 0.5)
FIGURE_SIZE_INCHES = (7.48, 4.55)
GRID_WIDTH_RATIOS = (0.9, 1.05, 1.05, 0.55)
GRID_HEIGHT_RATIOS = (1.0, 1.0, 1.0)

def read_retained_cell_classes(source_case: str) -> tuple[str, ...]:
    """Read and verify the exact method support recorded by the scorer."""
    audit = pd.read_csv(SUPPORT_AUDIT_PATH)
    selected = audit.loc[audit['slice_id'].astype(str).str.zfill(3) == source_case]
    if len(selected) != len(RESOLUTIONS) or set(selected['resolution'].astype(str)) != set(RESOLUTIONS) or set(selected['annotation_key']) != {'cell_class'} or (set(selected['min_cells_per_class'].astype(int)) != {50}) or (not selected['method_class_sets_exactly_equal'].astype(bool).all()):
        raise ValueError(f'Invalid retained-support audit for slice {source_case}')
    orders = [tuple(json.loads(value)) for value in selected['retained_cell_class_order_json']]
    if not orders or any((order != orders[0] for order in orders[1:])):
        raise ValueError(f'Retained class order differs by resolution for slice {source_case}')
    if len(orders[0]) != int(selected['n_retained_cell_classes'].iloc[0]):
        raise ValueError(f'Retained class count differs from its named support for slice {source_case}')
    return orders[0]

def configure_display_ontology(source_case: str, retained_cell_classes: tuple[str, ...]) -> None:
    """Select the exact scored ontology for a declared slice."""
    global SOURCE_CASE, CELL_TYPES, CELL_TYPE_ORDER, CELL_TYPE_COLORS
    global RETAINED_CELL_CLASSES, SOURCE_CASE_SELECTION
    if source_case not in {'007', '050', '100'}:
        raise ValueError(f'Unsupported registered source case: {source_case}')
    unknown = sorted(set(retained_cell_classes) - set(FULL_CELL_TYPE_COLORS))
    if unknown:
        raise ValueError(f'Retained classes lack registered colours: {unknown}')
    SOURCE_CASE = source_case
    RETAINED_CELL_CLASSES = retained_cell_classes
    CELL_TYPES = tuple(((label, FULL_CELL_TYPE_COLORS[label]) for label in retained_cell_classes)) + (OTHER_CELL_CLASS,)
    CELL_TYPE_ORDER = tuple((label for label, _colour in CELL_TYPES))
    CELL_TYPE_COLORS = {label: colour for label, colour in CELL_TYPES}
    SOURCE_CASE_SELECTION = 'registered slice with scorer-verified retained cell_class support plus Other'

def repository_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT))

def verified_path(path_text: str, context: str) -> Path:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f'Missing {context}: {path}')
    return path

def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing manifest: {path}')
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

def check_validation() -> None:
    validation = pd.read_csv(VALIDATION_PATH)
    if len(validation) != 1 or validation.loc[0, 'status'] != 'validated' or validation.loc[0, 'annotation_key'] != 'cell_class' or (not bool(validation.loc[0, 'exact_method_support_verified'])) or (not bool(validation.loc[0, 'cuda_execution_verified_for_all_cell2location_jobs'])):
        raise ValueError('Study 03 cell-class validation is incomplete')

def _manifest_row(rows: list[dict[str, str]], *, resolution: str, method: str | None=None) -> dict[str, str]:
    matches = [row for row in rows if row.get('slice_id') == SOURCE_CASE and float(row.get('resolution', 'nan')) == float(resolution) and (method is None or row.get('method') == method) and (row.get('status') == 'ok')]
    if len(matches) != 1:
        label = f'{SOURCE_CASE}, resolution {resolution}' if method is None else f'{SOURCE_CASE}, resolution {resolution}, {method}'
        raise ValueError(f'Expected one successful manifest row for {label}; found {len(matches)}')
    return matches[0]

def read_simulation_coordinates(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = ad.read_h5ad(path, backed='r')
    try:
        if 'spatial' not in data.obsm:
            raise KeyError(f"{path} does not contain .obsm['spatial']")
        if 'n_source_cells' not in data.obs:
            raise KeyError(f"{path} does not contain .obs['n_source_cells']")
        coords = np.asarray(data.obsm['spatial'])[:, :2].astype(float)
        names = np.asarray(data.obs_names, dtype=str)
        source_cells = data.obs['n_source_cells'].to_numpy(dtype=np.int64)
    finally:
        data.file.close()
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) != len(names) or (source_cells.shape != (len(names),)) or (source_cells < 0).any():
        raise ValueError(f'Invalid spatial coordinates in {path}')
    return (coords, names, source_cells)

def reference_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    data = ad.read_h5ad(path, backed='r')
    try:
        required = {'spatial'}
        if not required.issubset(data.obsm.keys()):
            raise KeyError(f"{path} lacks required .obsm['spatial']")
        required_obs = {'ground_truth', 'cell_class'}
        if not required_obs.issubset(data.obs.columns):
            raise KeyError(f'{path} lacks required observation fields: {sorted(required_obs)}')
        coords = np.asarray(data.obsm['spatial'])[:, :2].astype(float)
        names = np.asarray(data.obs_names, dtype=str)
        source_classes = np.asarray(data.obs['cell_class'].astype(str), dtype=str)
        mapping_table = pd.DataFrame({'fine_label': data.obs['ground_truth'].astype(str), 'display_class': data.obs['cell_class'].astype(str)}).drop_duplicates()
    finally:
        data.file.close()
    if mapping_table['fine_label'].duplicated().any():
        raise ValueError('Fine labels do not map uniquely to display classes')
    if not set(source_classes).issubset(FULL_CELL_TYPE_COLORS):
        unknown = sorted(set(source_classes) - set(FULL_CELL_TYPE_COLORS))
        raise ValueError(f'Reference classes outside the registered cell-class ontology: {unknown}')
    classes = np.asarray([label if label in RETAINED_CELL_CLASSES else 'Other' for label in source_classes], dtype=str)
    fine_to_class = dict(zip(mapping_table['fine_label'], mapping_table['display_class'], strict=True))
    return (coords, names, classes, fine_to_class)

def spot_id(name: str) -> str:
    value = str(name)
    return value if value.startswith('spot_') else f'spot_{value}'

def aggregate_proportions(path: Path, simulation_names: np.ndarray, fine_to_class: dict[str, str], source_cells: np.ndarray, *, allow_missing_empty_rows: bool=False, empty_row_expectation: str='zero') -> np.ndarray:
    proportions = pd.read_csv(path, index_col=0)
    proportions.index = proportions.index.astype(str)
    expected_index = pd.Index([spot_id(name) for name in simulation_names])
    if proportions.index.duplicated().any():
        raise ValueError(f'Duplicate proportion rows: {path}')
    observed_index = pd.Index(proportions.index)
    extra_rows = observed_index.difference(expected_index)
    missing_rows = expected_index.difference(observed_index)
    if len(extra_rows):
        raise ValueError(f'Proportion rows are not simulation spots: {path}')
    if len(missing_rows):
        if not allow_missing_empty_rows:
            raise ValueError(f'Proportion rows do not exactly match simulation spots: {path}')
        source_by_id = dict(zip(expected_index, source_cells, strict=True))
        nonempty_missing = [name for name in missing_rows if source_by_id[name] > 0]
        if nonempty_missing:
            raise ValueError(f'Non-empty simulation spots missing from {path}: {nonempty_missing[:4]}')
    if proportions.columns.duplicated().any():
        raise ValueError(f'Duplicate fine-label columns in {path}')
    fine_labels = [str(column) for column in proportions.columns]
    unknown_labels = sorted((label for label in set(fine_labels) if label not in fine_to_class and label not in FULL_CELL_TYPE_COLORS))
    if unknown_labels:
        raise ValueError(f'Unmapped fine labels in {path}: {unknown_labels[:8]}')
    source_labels = [label if label in FULL_CELL_TYPE_COLORS else fine_to_class[label] for label in fine_labels]
    display_labels = [label if label in RETAINED_CELL_CLASSES else 'Other' for label in source_labels]
    aligned = proportions.reindex(expected_index, fill_value=0.0)
    result = np.zeros((len(aligned), len(CELL_TYPE_ORDER)), dtype=float)
    for class_index, cell_type in enumerate(CELL_TYPE_ORDER):
        columns = [column for column, display in zip(fine_labels, display_labels, strict=True) if display == cell_type]
        if columns:
            result[:, class_index] = aligned.loc[:, columns].to_numpy(dtype=float).sum(axis=1)
    if not np.isfinite(result).all() or (result < -1e-12).any():
        raise ValueError(f'Invalid proportions after aggregation: {path}')
    result = np.clip(result, 0.0, None)
    totals = result.sum(axis=1)
    positive = source_cells > 0
    if not np.allclose(totals[positive], 1.0, atol=1e-05):
        raise ValueError(f'Non-empty proportions do not sum to one: {path}')
    empty_totals = totals[~positive]
    if empty_row_expectation == 'zero':
        if not np.allclose(empty_totals, 0.0, atol=1e-12):
            raise ValueError(f'Empty proportions are not zero: {path}')
    elif empty_row_expectation == 'one':
        if not np.allclose(empty_totals, 1.0, atol=1e-05):
            raise ValueError(f'Empty proportions are not normalized: {path}')
    else:
        raise ValueError(f'Unsupported empty-row expectation: {empty_row_expectation}')
    if tuple(CELL_TYPE_ORDER) != tuple((label for label, _colour in CELL_TYPES)):
        raise AssertionError('Display columns are not in the declared named order')
    return result

def nearest_neighbor_distance(coords: np.ndarray) -> float:
    if len(coords) < 2:
        raise ValueError('At least two spatial centres are required')
    distances, _ = cKDTree(coords).query(coords, k=2)
    return float(np.median(distances[:, 1]))

def verified_candidate_cell2location_path(candidate_dir: Path, simulation_row: dict[str, str], resolution: str) -> tuple[Path, Path]:
    """Verify a label-matched Cell2location corrective output before plotting."""
    path = candidate_dir / 'methods' / 'cell2location' / SOURCE_CASE / f'resolution_{float(resolution):g}_proportions.csv'
    metadata_path = path.with_name(path.stem + '_metadata.json')
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f'Missing corrective Cell2location artifacts for resolution ×{resolution}')
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('status') != 'validated_success' or not metadata.get('reference_label_filter', {}).get('min_cells_per_type') or (not metadata.get('reference_label_filter', {}).get('n_reference_cell_types_retained')) or (metadata.get('zero_library_policy', '').split(';')[0] != 'excluded from training and reinserted as zero rows'):
        raise ValueError(f'Corrective Cell2location metadata is not valid for resolution ×{resolution}')
    return (path, metadata_path)

def load_data(cell2location_candidate_dir: Path | None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[tuple[str, str], np.ndarray], dict[str, str]]:
    check_validation()
    simulation_rows = read_manifest(SIMULATION_MANIFEST)
    method_rows = read_manifest(METHOD_MANIFEST)
    verified_inputs: dict[str, str] = {repository_path(SIMULATION_MANIFEST): None, repository_path(METHOD_MANIFEST): None, repository_path(VALIDATION_PATH): None, repository_path(SUPPORT_AUDIT_PATH): None}
    first_row = _manifest_row(simulation_rows, resolution=RESOLUTIONS[0])
    reference_path = verified_path(first_row['reference_path'], 'MERFISH reference')
    verified_inputs[repository_path(reference_path)] = None
    reference_xy, reference_names, reference_classes, fine_to_class = reference_data(reference_path)
    coordinates: dict[str, np.ndarray] = {}
    source_cells_by_resolution: dict[str, np.ndarray] = {}
    compositions: dict[tuple[str, str], np.ndarray] = {}
    for resolution in RESOLUTIONS:
        simulation_row = _manifest_row(simulation_rows, resolution=resolution)
        if simulation_row['reference_path'] != first_row['reference_path']:
            raise ValueError('The two displayed resolutions do not share the same reference slice')
        simulation_path = verified_path(simulation_row['simulation_path'], f'resolution ×{resolution} simulation')
        truth_path = verified_path(simulation_row['truth_path'], f'resolution ×{resolution} ground truth')
        verified_inputs[repository_path(simulation_path)] = None
        verified_inputs[repository_path(truth_path)] = None
        xy, simulation_names, source_cells = read_simulation_coordinates(simulation_path)
        coordinates[resolution] = xy
        source_cells_by_resolution[resolution] = source_cells
        compositions[resolution, 'truth'] = aggregate_proportions(truth_path, simulation_names, fine_to_class, source_cells, empty_row_expectation='zero')
        for method in ('cell2location', 'rctd'):
            if method == 'cell2location' and cell2location_candidate_dir is not None:
                output_path, metadata_path = verified_candidate_cell2location_path(cell2location_candidate_dir, simulation_row, resolution)
                verified_inputs[repository_path(output_path)] = None
                verified_inputs[repository_path(metadata_path)] = None
            else:
                method_row = _manifest_row(method_rows, resolution=resolution, method=method)
                if method_row['simulation_path'] != simulation_row['simulation_path']:
                    raise ValueError(f'{method} does not reference the displayed simulation at resolution {resolution}')
                output_path = verified_path(method_row['output_path'], f'{method} proportions at resolution ×{resolution}')
                verified_inputs[repository_path(output_path)] = None
            compositions[resolution, method] = aggregate_proportions(output_path, simulation_names, fine_to_class, source_cells, allow_missing_empty_rows=method == 'rctd', empty_row_expectation='zero')
        expected_rows = len(xy)
        for source in ('truth', 'cell2location', 'rctd'):
            if compositions[resolution, source].shape != (expected_rows, len(CELL_TYPE_ORDER)):
                raise ValueError(f'Unexpected composition shape for {source}, resolution {resolution}')
    return (reference_xy, reference_names, reference_classes, coordinates, source_cells_by_resolution, compositions, verified_inputs)

def common_limits(reference_xy: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    lower = reference_xy.min(axis=0)
    upper = reference_xy.max(axis=0)
    padding = 0.018 * (upper - lower)
    return ((float(lower[0] - padding[0]), float(upper[0] + padding[0])), (float(lower[1] - padding[1]), float(upper[1] + padding[1])))

def roi_bounds(x_limits: tuple[float, float], y_limits: tuple[float, float]) -> tuple[float, float, float, float]:
    return (x_limits[0] + ROI_X_FRACTION[0] * (x_limits[1] - x_limits[0]), x_limits[0] + ROI_X_FRACTION[1] * (x_limits[1] - x_limits[0]), y_limits[0] + ROI_Y_FRACTION[0] * (y_limits[1] - y_limits[0]), y_limits[0] + ROI_Y_FRACTION[1] * (y_limits[1] - y_limits[0]))

def configure_matplotlib() -> None:
    arial_path = Path(sys.prefix) / 'fonts' / 'arial.ttf'
    if not arial_path.is_file():
        raise FileNotFoundError(f'Arial font is required for this figure: {arial_path}')
    font_manager.fontManager.addfont(arial_path)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8.0, 'axes.titlesize': 9.0, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def format_spatial_axis(ax: plt.Axes, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> None:
    ax.set_xlim(*x_limits)
    ax.set_ylim(y_limits[1], y_limits[0])
    ax.set_aspect('equal')
    ax.set_axis_off()

def pie_collection(xy: np.ndarray, proportions: np.ndarray, radius: float, *, crop: tuple[float, float, float, float] | None=None, linewidth: float=FULL_WEDGE_LINEWIDTH) -> PatchCollection:
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
    return PatchCollection(patches, facecolor=colors, edgecolor='white', linewidth=linewidth, alpha=1.0, antialiased=True, rasterized=True, zorder=2)

def add_composition_panel(ax: plt.Axes, xy: np.ndarray, proportions: np.ndarray, radius: float, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> None:
    format_spatial_axis(ax, x_limits, y_limits)
    ax.add_collection(pie_collection(xy, proportions, radius, linewidth=FULL_WEDGE_LINEWIDTH))

def add_roi_rectangle(parent: plt.Axes, roi: tuple[float, float, float, float]) -> None:
    """Mark the fixed ROI without annotating or obscuring the tissue."""
    x0, x1, y0, y1 = roi
    parent.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='#D95F4A', linewidth=0.68, zorder=5))

def add_roi_panel(ax: plt.Axes, xy: np.ndarray, proportions: np.ndarray, radius: float, roi: tuple[float, float, float, float]) -> None:
    """Render an externally placed, identically scaled detailed crop."""
    x0, x1, y0, y1 = roi
    ax.set_facecolor('white')
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.add_collection(pie_collection(xy, proportions, radius, crop=roi, linewidth=ROI_WEDGE_LINEWIDTH))

def draw_legend(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.985, 'Cell classes', ha='left', va='top', fontsize=8.4, fontweight='normal')
    entries_per_column = int(np.ceil(len(CELL_TYPES) / 2))
    row_spacing = 0.055
    for index, (label, color) in enumerate(CELL_TYPES):
        column = index // entries_per_column
        row = index % entries_per_column
        x = (0.0, 0.535)[column]
        y = 0.9 - row_spacing * row
        ax.add_patch(Rectangle((x, y - 0.013), 0.022, 0.026, facecolor=color, edgecolor='none'))
        ax.text(x + 0.028, y, label, ha='left', va='center', fontsize=6.65, color='#222222')

def write_plot_data(output_dir: Path, reference_xy: np.ndarray, reference_names: np.ndarray, reference_classes: np.ndarray, coordinates: dict[str, np.ndarray], source_cells_by_resolution: dict[str, np.ndarray], compositions: dict[tuple[str, str], np.ndarray], output_stem: str) -> tuple[Path, Path]:
    data_path = output_dir / f'{output_stem}_plot_data.csv'
    with data_path.open('w', newline='', encoding='utf-8') as handle:
        fieldnames = ['panel', 'resolution', 'unit_id', 'x', 'y', 'n_source_cells', 'cell_class', 'proportion']
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        for name, coordinate, cell_class in zip(reference_names, reference_xy, reference_classes, strict=True):
            writer.writerow({'panel': 'single_cell_reference', 'resolution': 'single_cell', 'unit_id': name, 'x': f'{coordinate[0]:.12g}', 'y': f'{coordinate[1]:.12g}', 'n_source_cells': '1', 'cell_class': cell_class, 'proportion': '1'})
        for resolution in RESOLUTIONS:
            for source in ('truth', 'cell2location', 'rctd'):
                for spot_index, (coordinate, values) in enumerate(zip(coordinates[resolution], compositions[resolution, source], strict=True)):
                    for cell_class, value in zip(CELL_TYPE_ORDER, values, strict=True):
                        writer.writerow({'panel': source, 'resolution': resolution, 'unit_id': f'spot_{spot_index}', 'x': f'{coordinate[0]:.12g}', 'y': f'{coordinate[1]:.12g}', 'n_source_cells': int(source_cells_by_resolution[resolution][spot_index]), 'cell_class': cell_class, 'proportion': f'{value:.12g}'})
    palette_path = output_dir / f'{output_stem}_palette.csv'
    with palette_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['display_order', 'cell_class', 'hex_color', 'present_in_source_slice'], lineterminator='\n')
        writer.writeheader()
        for order, (cell_class, color) in enumerate(CELL_TYPES, start=1):
            writer.writerow({'display_order': order, 'cell_class': cell_class, 'hex_color': color, 'present_in_source_slice': cell_class in set(reference_classes)})
    return (data_path, palette_path)

def draw_figure(output_dir: Path, output_stem: str, dpi: int, cell2location_candidate_dir: Path | None=None) -> tuple[list[Path], tuple[Path, Path], dict[str, str]]:
    configure_matplotlib()
    reference_xy, reference_names, reference_classes, coordinates, source_cells_by_resolution, compositions, verified_inputs = load_data(cell2location_candidate_dir)
    x_limits, y_limits = common_limits(reference_xy)
    nearest_neighbour = {resolution: nearest_neighbor_distance(coordinates[resolution]) for resolution in RESOLUTIONS}
    radii = {resolution: FULL_RADIUS_SCALE[resolution] * nearest_neighbour[resolution] for resolution in RESOLUTIONS}
    roi_radius = ROI_RADIUS_SCALE * nearest_neighbour['0.1']
    roi = roi_bounds(x_limits, y_limits)
    rendered_compositions: dict[tuple[str, str], np.ndarray] = {}
    for resolution in RESOLUTIONS:
        populated = source_cells_by_resolution[resolution] > 0
        for source in ('truth', 'cell2location', 'rctd'):
            displayed = compositions[resolution, source].copy()
            displayed[~populated] = 0.0
            rendered_compositions[resolution, source] = displayed
    fig = plt.figure(figsize=FIGURE_SIZE_INCHES)
    grid = fig.add_gridspec(3, 4, left=0.025, right=0.99, top=0.9, bottom=0.055, width_ratios=GRID_WIDTH_RATIOS, height_ratios=GRID_HEIGHT_RATIOS, wspace=0.028, hspace=0.085)
    axes = {'reference': fig.add_subplot(grid[0, 0]), 'truth_025': fig.add_subplot(grid[0, 1]), 'truth_010': fig.add_subplot(grid[0, 2]), 'truth_roi': fig.add_subplot(grid[0, 3]), 'legend': fig.add_subplot(grid[1:, 0]), 'cell2location_025': fig.add_subplot(grid[1, 1]), 'cell2location_010': fig.add_subplot(grid[1, 2]), 'cell2location_roi': fig.add_subplot(grid[1, 3]), 'rctd_025': fig.add_subplot(grid[2, 1]), 'rctd_010': fig.add_subplot(grid[2, 2]), 'rctd_roi': fig.add_subplot(grid[2, 3])}
    format_spatial_axis(axes['reference'], x_limits, y_limits)
    axes['reference'].scatter(reference_xy[:, 0], reference_xy[:, 1], s=REFERENCE_POINT_SIZE, c=[CELL_TYPE_COLORS[cell_type] for cell_type in reference_classes], marker='o', linewidths=0, alpha=0.95, rasterized=True, zorder=2)
    panel_specs = (('truth_025', 'truth', '0.25'), ('truth_010', 'truth', '0.1'), ('cell2location_025', 'cell2location', '0.25'), ('cell2location_010', 'cell2location', '0.1'), ('rctd_025', 'rctd', '0.25'), ('rctd_010', 'rctd', '0.1'))
    for axis_key, source, resolution in panel_specs:
        add_composition_panel(axes[axis_key], coordinates[resolution], rendered_compositions[resolution, source], radii[resolution], x_limits, y_limits)
    for axis_key in ('truth_010', 'cell2location_010', 'rctd_010'):
        add_roi_rectangle(axes[axis_key], roi)
    for axis_key, source in (('truth_roi', 'truth'), ('cell2location_roi', 'cell2location'), ('rctd_roi', 'rctd')):
        add_roi_panel(axes[axis_key], coordinates['0.1'], rendered_compositions['0.1', source], roi_radius, roi)
    draw_legend(axes['legend'])
    header_y = 0.962
    header_specs = (('reference', 'Single-cell MERFISH'), ('truth_025', 'Resolution ×0.25'), ('truth_010', 'Resolution ×0.10'), ('truth_roi', 'Enlarged ROI'))
    for axis_key, label in header_specs:
        bounds = axes[axis_key].get_position()
        fig.text(0.5 * (bounds.x0 + bounds.x1), header_y, label, ha='center', va='center', fontsize=8.6, fontweight='normal', color='#202020')
    for axis_key, label in (('truth_025', 'Ground truth'), ('cell2location_025', 'Cell2location'), ('rctd_025', 'RCTD')):
        bounds = axes[axis_key].get_position()
        fig.text(bounds.x0 - 0.015, 0.5 * (bounds.y0 + bounds.y1), label, ha='center', va='center', rotation=90, fontsize=8.6, fontweight='normal', color='#202020')
    fig.text(0.59, 0.018, f'All panels use the same {len(RETAINED_CELL_CLASSES)} S{SOURCE_CASE} cell_class categories (≥50 reference cells) + Other', ha='center', va='bottom', fontsize=6.3, color='#666666')
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / output_stem
    outputs: list[Path] = []
    for suffix, kwargs in {'pdf': {'dpi': dpi, 'metadata': {'Creator': 'FEAST Study 03 spatial deconvolution composition comparison', 'CreationDate': None, 'ModDate': None}}, 'png': {'dpi': dpi, 'metadata': {'Software': 'FEAST Study 03 spatial deconvolution composition comparison'}}, 'svg': {'dpi': dpi, 'metadata': {'Creator': 'FEAST Study 03 spatial deconvolution composition comparison', 'Date': '1970-01-01T00:00:00Z'}}}.items():
        path = stem.with_suffix(f'.{suffix}')
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    data_paths = write_plot_data(output_dir, reference_xy, reference_names, reference_classes, coordinates, source_cells_by_resolution, compositions, output_stem)
    return (outputs, data_paths, verified_inputs)

def write_provenance(output_dir: Path, output_stem: str, output_paths: list[Path], data_paths: tuple[Path, Path], verified_inputs: dict[str, str]) -> Path:
    payload = {'schema_version': 1, 'figure_set': 'study03_spatial_deconvolution_composition_diagnostic', 'publication_status': 'validated_cell_class_descriptive_comparison', 'scientific_disposition': {'scientific_stop': False, 'publication_claim_authorized': False, 'aggregate_winner_ranking_authorized': False, 'figure_promotion_authorized': True}, 'source': {'plot_script': repository_path(Path(__file__)), 'verified_inputs': verified_inputs}, 'design': {'source_slice': f'Zhuang-ABCA-1.{SOURCE_CASE}', 'source_slice_selection': SOURCE_CASE_SELECTION, 'resolutions': list(RESOLUTIONS), 'methods': ['Cell2location', 'RCTD'], 'cell_class_order': list(CELL_TYPE_ORDER), 'retained_cell_classes': list(RETAINED_CELL_CLASSES), 'rare_truth_class_policy': 'aggregate into Other', 'layout': {'matrix': 'reference/legend, resolution_x0.25, resolution_x0.10, enlarged_roi', 'row_labels': ['Ground truth', 'Cell2location', 'RCTD'], 'panel_letters': [], 'figure_size_inches': list(FIGURE_SIZE_INCHES), 'grid_width_ratios': list(GRID_WIDTH_RATIOS), 'grid_height_ratios': list(GRID_HEIGHT_RATIOS), 'column_headers_once': True, 'row_labels_once': True, 'legend_columns': 2, 'panel_indices_rendered': False}, 'palette': {'mapping': 'fixed named categorical mapping shared by every panel', 'design': 'medium-lightness opaque categories; high-co-occurrence abundant classes are assigned separated hues', 'near_white_cell_class_colours_allowed': False}, 'composition_glyph': {'type': 'clockwise pie wedges', 'start_angle_degrees': 90, 'cell_class_order': 'fixed shared named order', 'display_threshold': DISPLAY_THRESHOLD, 'full_panel_radius_fraction_of_median_nearest_neighbour_distance': FULL_RADIUS_SCALE, 'roi_radius_fraction_of_median_nearest_neighbour_distance': ROI_RADIUS_SCALE, 'full_wedge_separator_linewidth': FULL_WEDGE_LINEWIDTH, 'roi_wedge_separator_linewidth': ROI_WEDGE_LINEWIDTH}, 'paired_glyph_contract': 'Ground truth, Cell2location, and RCTD use the same centres, radii, colours, order, background, and coordinate limits within each resolution', 'empty_location_display': 'Locations with n_source_cells = 0 are hidden consistently from all three composition sources; the cell-class Cell2location output explicitly reinserts these locations as zero rows.', 'roi': {'selection_rule': 'fixed source-occupancy and ground-truth-diversity ROI selected independently of prediction values', 'x_fraction': list(ROI_X_FRACTION), 'y_fraction': list(ROI_Y_FRACTION), 'applied_to': ['ground_truth_resolution_0.1', 'cell2location_resolution_0.1', 'rctd_resolution_0.1'], 'layout': 'separate right-hand axes; no overlay or connector lines'}, 'point_layers': 'Single-cell dots and proportional pie wedges are rasterized at the requested export DPI; text, legend swatches, and ROI outlines remain vector in PDF/SVG.'}, 'typography': {'font_family': 'Arial', 'font_source': str(Path(sys.prefix) / 'fonts' / 'arial.ttf')}, 'render_environment': {'python': platform.python_version(), 'matplotlib': matplotlib.__version__}, 'outputs': {path.name: None for path in sorted([*output_paths, *data_paths], key=lambda item: item.name)}}
    path = output_dir / f'{output_stem}_provenance.json'
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--output-stem', default=OUTPUT_STEM)
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--source-case', choices=('007', '050', '100'), default=SOURCE_CASE)
    parser.add_argument('--cell2location-candidate-dir', type=Path, help='Use a verified label-matched Cell2location corrective output root.')
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    retained_cell_classes = read_retained_cell_classes(args.source_case)
    configure_display_ontology(args.source_case, retained_cell_classes)
    output_paths, data_paths, verified_inputs = draw_figure(args.output_dir, args.output_stem, args.dpi, args.cell2location_candidate_dir)
    provenance_path = write_provenance(args.output_dir, args.output_stem, output_paths, data_paths, verified_inputs)
    for path in [*output_paths, *data_paths, provenance_path]:
        print(f'Saved: {path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
