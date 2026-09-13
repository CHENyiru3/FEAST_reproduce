"""Render source-verified four-condition spatial alignment evidence for Study 02."""
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
from matplotlib import font_manager
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / 'figures'
OUTPUT_STEM = 'alignment_spatial'
STUDY_OUTPUT = REPOSITORY_ROOT / '02_alignment' / 'outputs' / 'fixed_plate_rerun_20260806_v1'
ROTATIONS_DIR = STUDY_OUTPUT / 'rotations'
METHODS_DIR = STUDY_OUTPUT / 'methods'
ROTATION_MANIFEST = ROTATIONS_DIR / 'rotation_manifest.csv'
METHOD_MANIFEST = METHODS_DIR / 'method_manifest.csv'
PASTE2_MANIFEST = REPOSITORY_ROOT / '02_alignment' / 'outputs' / 'paste2_rerun_20260807_v1' / 'method_manifest.csv'
ANGLE = 45.0
ALTERATIONS = (('baseline', 'Baseline'), ('mean_0.5', 'Mean ×0.5'), ('variance_2.0', 'Variance ×2.0'), ('sparsity_0.5', 'Sparsity ×0.5'))
METHODS = (('spateo', 'Spateo'), ('paste2', 'PASTE2'))
POINT_SIZE = 2.5
REFERENCE_POINT_SIZE = 0.75
ERROR_COLORS = {'low': '#0072B2', 'elevated': '#E69F00', 'high': '#D55E00'}

def repository_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT))

def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f'Required manifest is missing: {path}')
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

def _verified_path(path_text: str, context: str) -> Path:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f'Missing {context}: {path}')
    return path

def resolve_inputs() -> tuple[Path, dict[tuple[str, str], Path], dict[str, str]]:
    """Resolve the 45° panels recorded by Study 02 manifests."""
    rotation_rows = _read_manifest(ROTATION_MANIFEST)
    method_rows = _read_manifest(METHOD_MANIFEST)
    paste2_rows = _read_manifest(PASTE2_MANIFEST)
    required_alterations = {alteration for alteration, _label in ALTERATIONS}
    verified: dict[str, str] = {repository_path(ROTATION_MANIFEST): None, repository_path(METHOD_MANIFEST): None, repository_path(PASTE2_MANIFEST): None}
    reference_path: Path | None = None
    panels: dict[tuple[str, str], Path] = {}
    for alteration in required_alterations:
        matching = [row for row in rotation_rows if row.get('alteration') == alteration and float(row.get('angle_degrees', 'nan')) == ANGLE]
        if len(matching) != 1:
            raise ValueError(f'Expected one {alteration} rotation row at {ANGLE:g}°, found {len(matching)}')
        rotation = matching[0]
        rotated_path = _verified_path(rotation['output_path'], f'{alteration} rotated input')
        verified[repository_path(rotated_path)] = None
        source_path = _verified_path(rotation['source_path'], f'{alteration} source geometry')
        verified[repository_path(source_path)] = None
        if alteration == 'baseline':
            reference_path = source_path
        for method, _label in METHODS:
            rows = method_rows if method == 'spateo' else paste2_rows
            method_match = [row for row in rows if row.get('alteration') == alteration and row.get('method') == method and (float(row.get('angle_degrees', 'nan')) == ANGLE) and (row.get('status') == 'ok')]
            if len(method_match) != 1:
                raise ValueError(f'Expected one successful {method} {alteration} row at {ANGLE:g}°, found {len(method_match)}')
            artifacts = json.loads(method_match[0]['artifacts'])
            aligned = [item for item in artifacts if str(item.get('path', '')).endswith('/aligned.h5ad')]
            if len(aligned) != 1:
                raise ValueError(f'Missing aligned.h5ad artifact for {method} {alteration}')
            aligned_path = _verified_path(str(aligned[0]['path']), f'{method} {alteration} aligned coordinates')
            panels[alteration, method] = aligned_path
            verified[repository_path(aligned_path)] = None
    if reference_path is None:
        raise ValueError('Baseline source geometry was not resolved')
    return (reference_path, panels, verified)

def read_xy(path: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    data = ad.read_h5ad(path, backed='r')
    try:
        if key not in data.obsm:
            raise KeyError(f'{path} does not contain .obsm[{key!r}]')
        coords = np.asarray(data.obsm[key])[:, :2].astype(float)
        names = np.asarray(data.obs_names, dtype=str)
    finally:
        data.file.close()
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) != len(names):
        raise ValueError(f'Invalid spatial coordinate array in {path}')
    return (coords, names)

def configure_matplotlib() -> None:
    arial_paths = (Path(sys.prefix) / 'fonts' / 'arial.ttf', Path(sys.prefix) / 'fonts' / 'arialbd.ttf')
    if not all((path.is_file() for path in arial_paths)):
        raise FileNotFoundError(f'Arial Regular and Bold are required for this figure: {arial_paths}')
    for path in arial_paths:
        font_manager.fontManager.addfont(path)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'axes.titlesize': 12, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def _common_limits(reference_xy: np.ndarray, panel_coordinates: dict[tuple[str, str], np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    stacked = np.vstack([reference_xy, *panel_coordinates.values()])
    lower = stacked.min(axis=0)
    upper = stacked.max(axis=0)
    pad = 0.035 * np.maximum(upper - lower, 1.0)
    return ((float(lower[0] - pad[0]), float(upper[0] + pad[0])), (float(lower[1] - pad[1]), float(upper[1] + pad[1])))

def write_plot_data(output_dir: Path, output_stem: str, panel_coordinates: dict[tuple[str, str], np.ndarray], panel_names: dict[tuple[str, str], np.ndarray], panel_reference_x: dict[tuple[str, str], np.ndarray], panel_errors: dict[tuple[str, str], np.ndarray]) -> Path:
    path = output_dir / f'{output_stem}_plot_data.csv'
    fieldnames = ['alteration', 'panel', 'spot_barcode', 'x', 'y', 'reference_x', 'exact_pair_error_spot_spacings', 'residual_bin']
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        for key, coords in panel_coordinates.items():
            alteration, panel = key
            for name, coordinate, x_ref, error in zip(panel_names[key], coords, panel_reference_x[key], panel_errors[key]):
                if not np.isfinite(error):
                    residual_bin = ''
                    formatted_error = ''
                elif error <= 0.005:
                    residual_bin = 'low_le_0.005'
                    formatted_error = f'{error:.12g}'
                elif error <= 0.01:
                    residual_bin = 'elevated_0.005_to_0.01'
                    formatted_error = f'{error:.12g}'
                else:
                    residual_bin = 'high_gt_0.01'
                    formatted_error = f'{error:.12g}'
                writer.writerow({'alteration': alteration, 'panel': panel, 'spot_barcode': name, 'x': f'{coordinate[0]:.12g}', 'y': f'{coordinate[1]:.12g}', 'reference_x': f'{x_ref:.12g}', 'exact_pair_error_spot_spacings': formatted_error, 'residual_bin': residual_bin})
    return path

def draw_figure(output_dir: Path, output_stem: str, dpi: int) -> tuple[list[Path], Path, dict[str, str]]:
    configure_matplotlib()
    reference_path, panel_paths, verified_inputs = resolve_inputs()
    reference_xy, reference_names = read_xy(reference_path, 'spatial')
    reference_x = reference_xy[:, 0].copy()
    reference_index = pd.Index(reference_names)
    nearest_distances, _ = cKDTree(reference_xy).query(reference_xy, k=2)
    spot_spacing = float(np.median(nearest_distances[:, 1]))
    if not np.isfinite(spot_spacing) or spot_spacing <= 0:
        raise ValueError('Reference median spot spacing is invalid')
    panel_coordinates: dict[tuple[str, str], np.ndarray] = {}
    panel_names: dict[tuple[str, str], np.ndarray] = {}
    panel_reference_x: dict[tuple[str, str], np.ndarray] = {}
    panel_errors: dict[tuple[str, str], np.ndarray] = {}
    for alteration, _label in ALTERATIONS:
        for panel, _panel_label in METHODS:
            key = 'spatial_aligned'
            coords, names = read_xy(panel_paths[alteration, panel], key)
            positions = reference_index.get_indexer(names)
            if np.any(positions < 0) or np.any(np.diff(positions) <= 0):
                raise ValueError(f'Spots are not an ordered reference subset in the {alteration} {panel} panel')
            panel_key = (alteration, panel)
            panel_coordinates[panel_key] = coords
            panel_names[panel_key] = names
            panel_reference_x[panel_key] = reference_x[positions]
            panel_errors[panel_key] = np.linalg.norm(coords - reference_xy[positions], axis=1) / spot_spacing
    x_limits, y_limits = _common_limits(reference_xy, panel_coordinates)
    fig = plt.figure(figsize=(15.5, 15.8))
    grid = fig.add_gridspec(len(ALTERATIONS) + 1, len(METHODS), height_ratios=(0.22, *(1.0 for _ in ALTERATIONS)), hspace=0.1, wspace=0.12)
    legend_axis = fig.add_subplot(grid[0, :])
    axes = np.empty((len(ALTERATIONS), len(METHODS)), dtype=object)
    for row_index in range(len(ALTERATIONS)):
        for column_index in range(len(METHODS)):
            axes[row_index, column_index] = fig.add_subplot(grid[row_index + 1, column_index], projection='3d')
    for row_index, (alteration, alteration_label) in enumerate(ALTERATIONS):
        for column_index, (panel, panel_label) in enumerate(METHODS):
            ax = axes[row_index, column_index]
            xy = panel_coordinates[alteration, panel]
            errors = panel_errors[alteration, panel]
            point_colors = np.select([errors <= 0.005, errors <= 0.01], [ERROR_COLORS['low'], ERROR_COLORS['elevated']], default=ERROR_COLORS['high'])
            positions = reference_index.get_indexer(panel_names[alteration, panel])
            target_xy = reference_xy[positions]
            connector_indices = np.linspace(0, len(errors) - 1, min(120, len(errors)), dtype=int)
            ax.scatter(reference_xy[:, 0], reference_xy[:, 1], np.zeros(len(reference_xy)), s=1.8, color='#CFCFCF', alpha=0.34, depthshade=False, rasterized=True)
            for connector in connector_indices:
                ax.plot([target_xy[connector, 0], xy[connector, 0]], [target_xy[connector, 1], xy[connector, 1]], [0.0, 1.0], color='#555555', alpha=0.16, linewidth=0.4)
            ax.scatter(xy[:, 0], xy[:, 1], np.ones(len(xy)), s=3.4, color=point_colors, alpha=0.9, marker='o', linewidths=0, depthshade=False, rasterized=True)
            high_fraction = float((errors > 0.01).mean())
            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)
            ax.set_zlim(-0.04, 1.04)
            ax.set_box_aspect((1.0, 1.0, 0.36))
            ax.view_init(elev=28, azim=-55)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([0.0, 1.0], ['target', 'aligned'])
            ax.tick_params(axis='z', labelsize=7.5, pad=-1)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.grid(False)
            ax.text2D(0.02, 0.98, chr(ord('A') + row_index * len(METHODS) + column_index), transform=ax.transAxes, ha='left', va='top', fontsize=14, fontweight='bold')
            ax.text2D(0.98, 0.03, f'mean {errors.mean():.4f}\n>0.01: {high_fraction:.1%}', transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5, color='#4D4D4D', bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.86, 'pad': 1.2})
            if row_index == 0:
                ax.set_title(panel_label, fontsize=13, fontweight='bold', pad=9)
        axes[row_index, 0].text2D(-0.12, 0.5, alteration_label, transform=axes[row_index, 0].transAxes, ha='center', va='center', rotation=90, fontsize=12, fontweight='bold')
    fig.suptitle(f'Spatial alignment across four expression conditions — DLPFC 151675, {ANGLE:g}° rotation', x=0.5, y=0.995, fontsize=18, fontweight='bold')
    fig.text(0.5, 0.968, 'Each cell shows the complete gray target slice below the aligned moving slice.\nColored upper-plane spots encode exact barcode-pair residual; thin lines sample exact pairs; z is a visual layer only.', ha='center', va='center', fontsize=9.5, color='#767676')
    legend_handles = [Line2D([], [], marker='o', linestyle='', color='#BDBDBD', label='Full target slice'), Line2D([], [], marker='o', linestyle='', color=ERROR_COLORS['low'], label='Low residual'), Line2D([], [], marker='o', linestyle='', color=ERROR_COLORS['elevated'], label='Elevated residual'), Line2D([], [], marker='o', linestyle='', color=ERROR_COLORS['high'], label='High residual')]
    legend_axis.axis('off')
    legend_axis.legend(handles=legend_handles, loc='center', ncol=4, frameon=False, fontsize=9.5, columnspacing=1.35)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.94, bottom=0.055)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / output_stem
    output_paths: list[Path] = []
    for suffix, kwargs in {'pdf': {'metadata': {'Creator': 'FEAST Study 02 alignment spatial'}}, 'png': {'dpi': dpi, 'metadata': {'Software': 'FEAST Study 02 alignment spatial'}}, 'svg': {'metadata': {'Creator': 'FEAST Study 02 alignment spatial'}}}.items():
        path = stem.with_suffix(f'.{suffix}')
        fig.savefig(path, **kwargs)
        output_paths.append(path)
    plt.close(fig)
    plot_data_path = write_plot_data(output_dir, output_stem, panel_coordinates, panel_names, panel_reference_x, panel_errors)
    return (output_paths, plot_data_path, verified_inputs)

def write_provenance(output_dir: Path, output_stem: str, output_paths: list[Path], plot_data_path: Path, verified_inputs: dict[str, str]) -> Path:
    payload = {'schema_version': 1, 'figure_set': 'study02_alignment_spatial_diagnostic', 'promotion_status': 'unpromoted_author_selection_required', 'scientific_disposition': {'status': 'author_scope_required', 'figure_promotion_authorized': False, 'aggregate_method_ranking_authorized': False, 'winner_claim_authorized': False}, 'source': {'plot_script': repository_path(Path(__file__)), 'verified_inputs': verified_inputs}, 'design': {'sample': 'DLPFC 151675', 'input_rotation_degrees': ANGLE, 'alterations': [alteration for alteration, _label in ALTERATIONS], 'columns': [method for method, _label in METHODS], 'reference_encoding': 'light-gray complete target slice', 'result_encoding': 'two-plane exact barcode-pair view: complete gray target at z=0, colored aligned moving spots at z=1, with blue <=0.005, orange 0.005–0.01, red >0.01 reference spot spacings', 'spot_spacing_definition': 'median nearest-neighbor spacing of the complete reference slice', 'spatial_limits': 'shared across every panel', 'point_layer': 'rasterized; text and annotations remain vector-editable'}, 'render_environment': {'python': platform.python_version(), 'matplotlib': matplotlib.__version__}, 'outputs': {path.name: None for path in sorted([*output_paths, plot_data_path], key=lambda item: item.name)}}
    path = output_dir / f'{output_stem}_provenance.json'
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--output-stem', default=OUTPUT_STEM)
    parser.add_argument('--dpi', type=int, default=600)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    output_paths, plot_data_path, verified_inputs = draw_figure(args.output_dir, args.output_stem, args.dpi)
    provenance_path = write_provenance(args.output_dir, args.output_stem, output_paths, plot_data_path, verified_inputs)
    for path in [*output_paths, plot_data_path, provenance_path]:
        print(f'Saved: {path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
