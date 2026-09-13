"""Create the pre-Cell2location diagnostic for Zhuang-ABCA-1.120.

The script consumes only the fresh 120 FEAST aggregation outputs.  It does not
read predictions and must be reviewed before Cell2location is started.
"""
from __future__ import annotations
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ['SOURCE_DATE_EPOCH'] = '0'
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPOSITORY_ROOT / '03_deconvolution' / 'outputs' / 'diagnostic_120_20260806_v1'
REFERENCE = REPOSITORY_ROOT / '03_deconvolution' / 'data' / 'local' / 'Zhuang-ABCA-1.120.h5ad'
OUTPUT_DIR = Path(__file__).resolve().parent / 'figures'
OUTPUT_STEM = 'deconvolution_120_preflight'
RESOLUTIONS = ('0.25', '0.1')
CANONICAL_TYPES = ('01 IT-ET Glut', '02 NP-CT-L6b Glut', '03 OB-CR Glut', '04 DG-IMN Glut', '05 OB-IMN GABA', '06 CTX-CGE GABA', '07 CTX-MGE GABA', '08 CNU-MGE GABA', '09 CNU-LGE GABA', '10 LSX GABA', '11 CNU-HYa GABA', '12 HY GABA', '13 CNU-HYa Glut', '14 HY Glut', '18 TH Glut', '19 MB Glut', '20 MB GABA', '21 MB Dopa', '22 MB-HB Sero', '23 P Glut', '24 MY Glut', '26 P GABA', '27 MY GABA', '28 CB GABA', '29 CB Glut', '30 Astro-Epen', '31 OPC-Oligo', '32 OEC', '33 Vascular', '34 Immune')
PALETTE = {'01 IT-ET Glut': '#436B92', '02 NP-CT-L6b Glut': '#A0B3CC', '03 OB-CR Glut': '#D77A47', '04 DG-IMN Glut': '#E6AE80', '05 OB-IMN GABA': '#58914E', '06 CTX-CGE GABA': '#95C28C', '07 CTX-MGE GABA': '#A83E3C', '08 CNU-MGE GABA': '#DC8E8B', '09 CNU-LGE GABA': '#786397', '10 LSX GABA': '#AD9FBB', '11 CNU-HYa GABA': '#73544E', '12 HY GABA': '#AA8E88', '13 CNU-HYa Glut': '#B2739E', '14 HY Glut': '#DBA8BC', '18 TH Glut': '#8A8A89', '19 MB Glut': '#C6C7C6', '20 MB GABA': '#A7AA55', '21 MB Dopa': '#C9CA90', '22 MB-HB Sero': '#5CADB2', '23 P Glut': '#9BC9CB', '24 MY Glut': '#D2D561', '26 P GABA': '#C1B598', '27 MY GABA': '#8BBF93', '28 CB GABA': '#4B4484', '29 CB Glut': '#AB719D', '30 Astro-Epen': '#68AF78', '31 OPC-Oligo': '#57676E', '32 OEC': '#74848B', '33 Vascular': '#8AB164', '34 Immune': '#D8E09F'}

def checked_path(path_text: str, label: str) -> Path:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f'missing {label}: {path}')
    return path

def configure_matplotlib() -> None:
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 10, 'axes.titlesize': 11.5, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def canonical_proportions(table: pd.DataFrame, fine_to_class: dict[str, str], expected_index: pd.Index) -> pd.DataFrame:
    table.index = table.index.astype(str)
    if table.index.duplicated().any() or set(table.index) != set(expected_index):
        raise ValueError('ground-truth spot identifiers do not exactly match simulation locations')
    table = table.reindex(expected_index)
    if table.columns.duplicated().any():
        raise ValueError('ground-truth fine labels are not unique')
    columns = pd.Index(map(str, table.columns))
    if set(columns) != set(fine_to_class):
        missing = sorted(set(fine_to_class) - set(columns))
        extra = sorted(set(columns) - set(fine_to_class))
        raise ValueError(f'ground-truth fine-label support mismatch: missing={missing[:4]}, extra={extra[:4]}')
    result = pd.DataFrame(0.0, index=expected_index, columns=CANONICAL_TYPES)
    for fine_label, display_class in fine_to_class.items():
        result.loc[:, display_class] += table.loc[:, fine_label].to_numpy(dtype=float)
    if not np.isfinite(result.to_numpy()).all() or (result.to_numpy() < -1e-12).any():
        raise ValueError('canonical ground-truth proportions are invalid')
    if result.columns.tolist() != list(CANONICAL_TYPES):
        raise AssertionError('canonical display order changed')
    return result.clip(lower=0.0)

def select_roi(xy: np.ndarray, proportions: pd.DataFrame, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> tuple[float, float, float, float, dict[str, float | int | str]]:
    """Choose a reproducible, mixed-composition ROI from the x0.10 grid."""
    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    width = 0.23 * x_span
    height = 0.23 * y_span
    candidates_x = np.linspace(x_limits[0] + width / 2, x_limits[1] - width / 2, 19)
    candidates_y = np.linspace(y_limits[0] + height / 2, y_limits[1] - height / 2, 19)
    values = proportions.to_numpy(dtype=float)
    best: tuple[float, float, float, int, float] | None = None
    for center_x in candidates_x:
        for center_y in candidates_y:
            mask = (xy[:, 0] >= center_x - width / 2) & (xy[:, 0] <= center_x + width / 2) & (xy[:, 1] >= center_y - height / 2) & (xy[:, 1] <= center_y + height / 2)
            if mask.sum() < 25:
                continue
            mean = values[mask].mean(axis=0)
            mean /= mean.sum()
            entropy = float(-(mean[mean > 0] * np.log(mean[mean > 0])).sum())
            score = entropy * np.log1p(int(mask.sum()))
            if best is None or score > best[0]:
                best = (score, center_x, center_y, int(mask.sum()), entropy)
    if best is None:
        raise RuntimeError('could not find a populated ROI in the x0.10 grid')
    score, center_x, center_y, n_spots, entropy = best
    bounds = (center_x - width / 2, center_x + width / 2, center_y - height / 2, center_y + height / 2)
    return bounds + ({'selection': 'max_local_composition_entropy_times_log_spot_count_on_x0.10_ground_truth', 'x0.10_spot_count': n_spots, 'mean_composition_entropy_nats': entropy, 'selection_score': score},)

def format_axis(ax: plt.Axes, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> None:
    ax.set_xlim(*x_limits)
    ax.set_ylim(y_limits[1], y_limits[0])
    ax.set_aspect('equal')
    ax.set_axis_off()

def main() -> int:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    simulation_manifest_path = RUN_DIR / 'simulation_manifest.csv'
    simulation_provenance_path = RUN_DIR / 'simulation_provenance.json'
    if not simulation_provenance_path.is_file():
        raise FileNotFoundError(f'missing simulation-only provenance: {simulation_provenance_path}')
    manifest = pd.read_csv(simulation_manifest_path, dtype={'slice_id': str})
    rows = {}
    for resolution in RESOLUTIONS:
        matches = manifest.loc[(manifest['slice_id'] == '120') & np.isclose(manifest['resolution'].astype(float), float(resolution)) & (manifest['status'] == 'ok')]
        if len(matches) != 1:
            raise ValueError(f'expected one successful 120 row at x{resolution}; found {len(matches)}')
        rows[resolution] = matches.iloc[0].to_dict()
    reference = ad.read_h5ad(REFERENCE, backed='r')
    try:
        reference_xy = np.asarray(reference.obsm['spatial'], dtype=float)[:, :2]
        reference_classes = reference.obs['class'].astype(str).to_numpy()
        mapping = pd.DataFrame({'fine': reference.obs['ground_truth'].astype(str), 'display': reference.obs['class'].astype(str)}).drop_duplicates()
    finally:
        reference.file.close()
    if mapping['fine'].duplicated().any():
        raise ValueError('a fine label maps to multiple canonical display classes')
    fine_to_class = dict(zip(mapping['fine'], mapping['display'], strict=True))
    unknown_classes = sorted(set(reference_classes) - set(CANONICAL_TYPES))
    if unknown_classes:
        raise ValueError(f'reference contains non-canonical classes: {unknown_classes}')
    lower, upper = (reference_xy.min(axis=0), reference_xy.max(axis=0))
    padding = 0.03 * (upper - lower)
    x_limits = (float(lower[0] - padding[0]), float(upper[0] + padding[0]))
    y_limits = (float(lower[1] - padding[1]), float(upper[1] + padding[1]))
    simulations: dict[str, dict[str, object]] = {}
    summaries: list[dict[str, object]] = []
    average_rows: list[dict[str, object]] = []
    for resolution, row in rows.items():
        sim_path = checked_path(row['simulation_path'], f'x{resolution} simulation')
        truth_path = checked_path(row['truth_path'], f'x{resolution} ground truth')
        simulation = ad.read_h5ad(sim_path, backed='r')
        try:
            xy = np.asarray(simulation.obsm['spatial'], dtype=float)[:, :2]
            source_cells = np.asarray(simulation.obs['n_source_cells'], dtype=np.int64)
            names = pd.Index([f'spot_{name}' for name in simulation.obs_names.astype(str)])
        finally:
            simulation.file.close()
        if len(xy) != int(row['n_spots']) or source_cells.sum() != len(reference_xy):
            raise ValueError(f'x{resolution} cell-to-location accounting failed')
        truth = pd.read_csv(truth_path, index_col=0)
        canonical = canonical_proportions(truth, fine_to_class, names)
        positive = source_cells > 0
        canonical_sums = canonical.sum(axis=1).to_numpy(dtype=float)
        if not np.allclose(canonical_sums[positive], 1.0, atol=1e-08):
            raise ValueError(f'x{resolution} non-empty canonical truth rows do not sum to one')
        if not np.allclose(canonical_sums[~positive], 0.0, atol=1e-12):
            raise ValueError(f'x{resolution} empty canonical truth rows are not zero')
        mean = canonical.loc[positive].mean(axis=0)
        for display_order, cell_type in enumerate(CANONICAL_TYPES, start=1):
            average_rows.append({'resolution': resolution, 'display_order': display_order, 'cell_type': cell_type, 'mean_proportion_nonempty_locations': float(mean[cell_type]), 'present_in_reference': bool(cell_type in set(reference_classes))})
        summaries.append({'resolution': resolution, 'n_spatial_locations': int(len(xy)), 'n_nonempty_locations': int(positive.sum()), 'n_empty_locations': int((~positive).sum()), 'source_cells': int(source_cells.sum()), 'mean_source_cells_per_location': float(source_cells.mean()), 'median_source_cells_per_location': float(np.median(source_cells)), 'min_source_cells_per_location': int(source_cells.min()), 'max_source_cells_per_location': int(source_cells.max()), 'ground_truth_rows_equal_locations': True, 'ground_truth_columns_are_30_canonical_order': True, 'positive_row_sum_min': float(canonical_sums[positive].min()), 'positive_row_sum_max': float(canonical_sums[positive].max()), 'empty_row_sum_max': float(canonical_sums[~positive].max()) if (~positive).any() else 0.0})
        simulations[resolution] = {'xy': xy, 'canonical': canonical, 'positive': positive, 'simulation_path': sim_path, 'truth_path': truth_path}
    roi_x0, roi_x1, roi_y0, roi_y1, roi_details = select_roi(simulations['0.1']['xy'], simulations['0.1']['canonical'], x_limits, y_limits)
    roi = (roi_x0, roi_x1, roi_y0, roi_y1)
    for summary in summaries:
        resolution = str(summary['resolution'])
        xy = simulations[resolution]['xy']
        summary['roi_spatial_location_count'] = int(((xy[:, 0] >= roi_x0) & (xy[:, 0] <= roi_x1) & (xy[:, 1] >= roi_y0) & (xy[:, 1] <= roi_y1)).sum())
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.75), constrained_layout=True)
    format_axis(axes[0], x_limits, y_limits)
    for cell_type in CANONICAL_TYPES:
        mask = reference_classes == cell_type
        if mask.any():
            axes[0].scatter(reference_xy[mask, 0], reference_xy[mask, 1], s=0.32, c=PALETTE[cell_type], alpha=0.88, linewidths=0, rasterized=True)
    axes[0].set_title('A  Single-cell MERFISH — Zhuang-ABCA-1.120', fontweight='semibold', pad=8)
    for axis, resolution, panel in zip(axes[1:], RESOLUTIONS, ('B', 'C'), strict=True):
        xy = simulations[resolution]['xy']
        positive = simulations[resolution]['positive']
        format_axis(axis, x_limits, y_limits)
        axis.scatter(xy[~positive, 0], xy[~positive, 1], s=2.1, c='#D7D7D7', linewidths=0, rasterized=True)
        axis.scatter(xy[positive, 0], xy[positive, 1], s=2.1, c='#355C7D', linewidths=0, rasterized=True)
        axis.add_patch(Rectangle((roi_x0, roi_y0), roi_x1 - roi_x0, roi_y1 - roi_y0, fill=False, edgecolor='#D9534F', linewidth=0.9, zorder=3))
        axis.set_title(f'{panel}  Spatial centres — resolution ×{resolution}', fontweight='semibold', pad=8)
    fig.text(0.5, 0.015, 'Pre-Cell2location diagnostic: blue = non-empty aggregate location; grey = empty location; red = fixed ROI.', ha='center', va='bottom', fontsize=9, color='#4D4D4D')
    for suffix, dpi in (('pdf', 600), ('svg', 600), ('png', 600)):
        fig.savefig(OUTPUT_DIR / f'{OUTPUT_STEM}.{suffix}', dpi=dpi, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    pd.DataFrame(summaries).sort_values('resolution', ascending=False).to_csv(OUTPUT_DIR / f'{OUTPUT_STEM}_summary.csv', index=False)
    pd.DataFrame(average_rows).to_csv(OUTPUT_DIR / f'{OUTPUT_STEM}_average_proportions.csv', index=False)
    (OUTPUT_DIR / f'{OUTPUT_STEM}_roi.json').write_text(json.dumps({'source_slice': 'Zhuang-ABCA-1.120', 'coordinate_system': "reference .obsm['spatial']; common limits shared by all later spatial panels", 'x_min': roi_x0, 'x_max': roi_x1, 'y_min': roi_y0, 'y_max': roi_y1, 'selection_details': roi_details, 'x_limits': x_limits, 'y_limits': y_limits}, indent=2) + '\n')
    (OUTPUT_DIR / f'{OUTPUT_STEM}_alignment.json').write_text(json.dumps({'canonical_order': list(CANONICAL_TYPES), 'palette': PALETTE, 'reference_observed_classes': sorted(set(reference_classes)), 'canonical_type_count': len(CANONICAL_TYPES), 'reference_observed_class_count': len(set(reference_classes)), 'fine_label_count': len(fine_to_class), 'all_reference_classes_canonical': True, 'all_truth_tables_reindexed_to_canonical_order': True, 'cell2location_status': 'not_started_pending_preflight_review'}, indent=2) + '\n')
    provenance = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'stage': 'pre_cell2location_diagnostic', 'source_slice': 'Zhuang-ABCA-1.120', 'reference_path': str(REFERENCE.resolve()), 'simulation_run_dir': str(RUN_DIR.resolve()), 'inputs': {resolution: {'simulation': {'path': str(values['simulation_path'])}, 'ground_truth': {'path': str(values['truth_path'])}} for resolution, values in simulations.items()}, 'outputs': {name: None for name in (f'{OUTPUT_STEM}.pdf', f'{OUTPUT_STEM}.png', f'{OUTPUT_STEM}.svg', f'{OUTPUT_STEM}_summary.csv', f'{OUTPUT_STEM}_average_proportions.csv', f'{OUTPUT_STEM}_roi.json', f'{OUTPUT_STEM}_alignment.json')}, 'cell2location_started': False}
    (OUTPUT_DIR / f'{OUTPUT_STEM}_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'output_dir': str(OUTPUT_DIR), 'summary': summaries, 'roi': roi}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
