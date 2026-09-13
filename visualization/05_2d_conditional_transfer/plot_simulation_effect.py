"""Render the Study 05 simulation design and representative spatial effects."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path
import tempfile
import anndata as ad
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from scipy import sparse
VIS_DIR = Path(__file__).resolve().parent
REPRO_ROOT = VIS_DIR.parents[1]
STUDY_DIR = REPRO_ROOT / '05_2d_conditional_transfer'
CASES = ({'dataset': 'dlpfc', 'dataset_label': 'DLPFC', 'slice': '151670', 'direction': '151670_low_x_to_high_x', 'annotation_key': 'ground_truth', 'point_size': 3.4}, {'dataset': 'dlpfc', 'dataset_label': 'DLPFC', 'slice': '151675', 'direction': '151675_low_x_to_high_x', 'annotation_key': 'ground_truth', 'point_size': 3.4}, {'dataset': 'dlpfc', 'dataset_label': 'DLPFC', 'slice': '151676', 'direction': '151676_low_x_to_high_x', 'annotation_key': 'ground_truth', 'point_size': 3.4}, {'dataset': 'merfish', 'dataset_label': 'MERFISH', 'slice': 'Zhuang-ABCA-1.006', 'direction': 'Zhuang-ABCA-1.006_low_x_to_high_x', 'annotation_key': 'class', 'point_size': 0.6}, {'dataset': 'merfish', 'dataset_label': 'MERFISH', 'slice': 'Zhuang-ABCA-1.007', 'direction': 'Zhuang-ABCA-1.007_low_x_to_high_x', 'annotation_key': 'class', 'point_size': 0.6})
LABEL_COLORS = ('#0F4D92', '#42949E', '#D9A441', '#B64342', '#8B5FBF', '#5E8C61', '#D88F8A', '#767676')
EXPRESSION_CMAP = LinearSegmentedColormap.from_list('feast_expression', ('#F2F2F2', '#B7D7E8', '#42949E', '#0F4D92'))

def configure_matplotlib() -> None:
    arial_paths = (Path(sys.prefix) / 'fonts' / 'arial.ttf', Path(sys.prefix) / 'fonts' / 'arialbd.ttf')
    if not all((path.is_file() for path in arial_paths)):
        raise FileNotFoundError(f'Arial Regular and Bold are required for this figure: {arial_paths}')
    for path in arial_paths:
        font_manager.fontManager.addfont(path)
    mpl.rcParams.update({'font.family': 'Arial', 'font.size': 8.5, 'axes.titlesize': 8.5, 'axes.titleweight': 'bold', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.bottom': False, 'axes.spines.left': False, 'legend.frameon': False, 'text.color': '#222222', 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'feast-study05'})

def job_id(case: dict) -> str:
    return f"study05-two-dataset-conditional-half-slice-v2-mask_half-{case['dataset']}-{case['direction']}-ar-0.3"

def generated_path(case: dict) -> Path:
    return STUDY_DIR / 'outputs' / 'final' / 'mask_half' / case['dataset'] / case['direction'] / 'ar_0_3' / 'generated.h5ad'

def input_path(case: dict) -> Path:
    return STUDY_DIR / 'data' / 'local' / case['dataset'] / f"{case['slice']}.h5ad"

def verify_evidence() -> tuple[dict, pd.DataFrame, dict[str, dict]]:
    paths = {'input_manifest': STUDY_DIR / 'data' / 'input_checksums.csv', 'score_provenance': STUDY_DIR / 'outputs' / 'scores' / 'provenance.json'}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing Study 05 evidence: {missing}')
    provenance = json.loads(paths['score_provenance'].read_text(encoding='utf-8'))
    if not (provenance.get('status') == 'ok' and (provenance.get('target_expression_policy') == 'evaluation_only_never_loaded_by_generation')):
        raise RuntimeError('Study 05 score provenance does not validate the generation evidence')
    inputs = pd.read_csv(paths['input_manifest'], dtype={'slice_id': str})
    generation_support = {}
    for case in CASES:
        input_row = inputs[(inputs['dataset'] == case['dataset']) & (inputs['slice_id'] == case['slice'])]
        if len(input_row) != 1:
            raise RuntimeError(f"input manifest does not uniquely identify {case['slice']}")
        generation_record = provenance['generation_inputs'][job_id(case)]
        if not isinstance(generation_record, dict):
            raise RuntimeError(f"missing generation record for {case['direction']}")
        run_provenance_path = generated_path(case).with_name('provenance.json')
        run_provenance = json.loads(run_provenance_path.read_text(encoding='utf-8'))
        support = run_provenance.get('support', {})
        if not (run_provenance.get('status') == 'ok' and int(support.get('source_retained_spots', 0)) > 0 and (int(support.get('target_retained_spots', 0)) > 0) and (len(support.get('eligible_labels', [])) > 0)):
            raise RuntimeError(f"invalid generation support for {case['direction']}")
        generation_support[job_id(case)] = support
    return (provenance, inputs, generation_support)

def setup_spatial_axis(ax: plt.Axes, coordinates: np.ndarray) -> None:
    x_pad = 0.035 * float(np.ptp(coordinates[:, 0]))
    y_pad = 0.035 * float(np.ptp(coordinates[:, 1]))
    ax.set_xlim(coordinates[:, 0].min() - x_pad, coordinates[:, 0].max() + x_pad)
    ax.set_ylim(coordinates[:, 1].min() - y_pad, coordinates[:, 1].max() + y_pad)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

def background(ax: plt.Axes, data: dict) -> None:
    point_size = float(data['case']['point_size'])
    ax.scatter(data['coordinates'][:, 0], data['coordinates'][:, 1], s=point_size, color='#E5E5E5', linewidths=0, zorder=0)

def muted_library_colors(values: np.ndarray, normalizer: mpl.colors.Normalize, color_fraction: float=0.55) -> np.ndarray:
    """Blend viridis library-size colors with gray for observed context."""
    colors = mpl.colormaps['viridis'](normalizer(np.clip(values, -2.5, 2.5)))
    gray = np.asarray(mpl.colors.to_rgb('#D8D8D8'), dtype=np.float64)
    colors[:, :3] = color_fraction * colors[:, :3] + (1.0 - color_fraction) * gray
    return colors

def show_unsupported(ax: plt.Axes, data: dict) -> None:
    unsupported = (data['source_candidate'] | data['target_candidate']) & ~np.isin(data['labels'], data['eligible_labels'])
    if unsupported.any():
        point_size = float(data['case']['point_size'])
        ax.scatter(data['coordinates'][unsupported, 0], data['coordinates'][unsupported, 1], s=max(point_size * 2.4, 2.0), marker='x', color='#9A9A9A', linewidths=0.35, zorder=5)

def add_split_line(ax: plt.Axes, split: float) -> None:
    ax.axvline(split, color='#444444', linestyle=(0, (2, 2)), linewidth=0.75, zorder=6)

def add_masked_region_box(ax: plt.Axes, split: float) -> None:
    """Enclose the held-out half without obscuring its plotted values."""
    x_max = float(max(ax.get_xlim()))
    y_min, y_max = sorted(map(float, ax.get_ylim()))
    ax.add_patch(Rectangle((split, y_min), x_max - split, y_max - y_min, facecolor='#ECECEC', edgecolor='#666666', linewidth=0.7, alpha=0.22, zorder=0.5, clip_on=True))

def dense_rows(matrix, start: int, stop: int) -> np.ndarray:
    values = matrix[start:stop]
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float64)

def profile_metrics(truth_matrix, generated_matrix, chunk_size: int=128) -> tuple[np.ndarray, np.ndarray]:
    n_spots = truth_matrix.shape[0]
    pearson = np.empty(n_spots, dtype=np.float64)
    js_distance = np.empty(n_spots, dtype=np.float64)
    for start in range(0, n_spots, chunk_size):
        stop = min(start + chunk_size, n_spots)
        truth = dense_rows(truth_matrix, start, stop)
        generated = dense_rows(generated_matrix, start, stop)
        truth_log = np.log1p(truth)
        generated_log = np.log1p(generated)
        truth_centered = truth_log - truth_log.mean(axis=1, keepdims=True)
        generated_centered = generated_log - generated_log.mean(axis=1, keepdims=True)
        numerator = np.sum(truth_centered * generated_centered, axis=1)
        denominator = np.sqrt(np.sum(truth_centered ** 2, axis=1) * np.sum(generated_centered ** 2, axis=1))
        pearson[start:stop] = np.divide(numerator, denominator, out=np.full(stop - start, np.nan, dtype=np.float64), where=denominator > 0)
        truth_sum = truth.sum(axis=1, keepdims=True)
        generated_sum = generated.sum(axis=1, keepdims=True)
        if np.any(truth_sum <= 0) or np.any(generated_sum <= 0):
            raise RuntimeError('all-gene profile comparison encountered an empty spot')
        truth_fraction = truth / truth_sum
        generated_fraction = generated / generated_sum
        midpoint = 0.5 * (truth_fraction + generated_fraction)
        with np.errstate(divide='ignore', invalid='ignore'):
            truth_term = np.where(truth_fraction > 0, truth_fraction * np.log2(truth_fraction / midpoint), 0.0)
            generated_term = np.where(generated_fraction > 0, generated_fraction * np.log2(generated_fraction / midpoint), 0.0)
        divergence = 0.5 * (np.sum(truth_term, axis=1) + np.sum(generated_term, axis=1))
        js_distance[start:stop] = np.sqrt(np.clip(divergence, 0.0, 1.0))
    if not np.all(np.isfinite(pearson)) or not np.all(np.isfinite(js_distance)):
        raise RuntimeError('all-gene spatial profile metrics are not finite')
    return (pearson, js_distance)

def load_general_case(case: dict, support: dict) -> tuple[dict, pd.DataFrame, dict]:
    observed = ad.read_h5ad(input_path(case))
    generated = ad.read_h5ad(generated_path(case))
    try:
        coordinates = np.asarray(observed.obsm['spatial'], dtype=np.float64)[:, :2]
        generated_coordinates = np.asarray(generated.obsm['spatial'], dtype=np.float64)[:, :2]
        labels = observed.obs[case['annotation_key']].astype(str).to_numpy()
        eligible_labels = sorted(map(str, support['eligible_labels']))
        split = float(np.median(coordinates[:, 0]))
        source_candidate = coordinates[:, 0] <= split
        target_candidate = ~source_candidate
        source_retained = source_candidate & np.isin(labels, eligible_labels)
        target_retained = target_candidate & np.isin(labels, eligible_labels)
        generated_indices = observed.obs_names.get_indexer(generated.obs_names)
        if np.any(generated_indices < 0):
            raise RuntimeError('generated target identities are absent from the observed slice')
        if not np.array_equal(np.flatnonzero(target_retained), generated_indices):
            raise RuntimeError('generated target order differs from declared support')
        np.testing.assert_allclose(coordinates[generated_indices], generated_coordinates, rtol=0.0, atol=0.0)
        gene_indices = observed.var_names.get_indexer(generated.var_names)
        if np.any(gene_indices < 0):
            raise RuntimeError('generated gene panel is absent from the observed slice')
        observed_panel_matrix = observed.layers['counts'][:, gene_indices]
        truth_matrix = observed_panel_matrix[generated_indices]
        generated_matrix = generated.layers['counts']
        observed_library = np.asarray(observed_panel_matrix.sum(axis=1)).reshape(-1)
        truth_library = np.asarray(truth_matrix.sum(axis=1)).reshape(-1)
        generated_library = np.asarray(generated_matrix.sum(axis=1)).reshape(-1)
        observed_log_library = np.log1p(observed_library)
        truth_log_library = np.log1p(truth_library)
        generated_log_library = np.log1p(generated_library)
        truth_scale = float(np.std(truth_log_library, ddof=0))
        if not truth_scale > 0:
            raise RuntimeError('held-out log-library-size scale is zero')
        truth_center = float(np.mean(truth_log_library))
        observed_library_z = (observed_log_library - truth_center) / truth_scale
        truth_library_z = (truth_log_library - truth_center) / truth_scale
        generated_library_z = (generated_log_library - truth_center) / truth_scale
        spotwise_pearson, spotwise_js = profile_metrics(truth_matrix, generated_matrix)
    finally:
        observed.file.close() if observed.isbacked else None
        generated.file.close() if generated.isbacked else None
    data = {'case': case, 'coordinates': coordinates, 'generated_coordinates': generated_coordinates, 'labels': labels, 'eligible_labels': eligible_labels, 'split': split, 'source_candidate': source_candidate, 'target_candidate': target_candidate, 'source_retained': source_retained, 'target_retained': target_retained, 'observed_library_z': observed_library_z, 'truth_library_z': truth_library_z, 'generated_library_z': generated_library_z, 'spotwise_pearson': spotwise_pearson, 'spotwise_js': spotwise_js, 'n_genes': int(generated.n_vars)}
    plot_data = pd.DataFrame({'dataset': case['dataset'], 'slice': case['slice'], 'direction': case['direction'], 'target_obs_id': generated.obs_names.astype(str), 'x': generated_coordinates[:, 0], 'y': generated_coordinates[:, 1], 'truth_log_library_z': truth_library_z, 'generated_log_library_z': generated_library_z, 'spotwise_profile_pearson': spotwise_pearson, 'spotwise_profile_js_distance': spotwise_js, 'n_genes': int(generated.n_vars)})
    audit = {'dataset': case['dataset'], 'slice': case['slice'], 'direction': case['direction'], 'job_id': job_id(case), 'n_genes': int(generated.n_vars), 'source_candidate_spots': int(source_candidate.sum()), 'source_retained_spots': int(source_retained.sum()), 'target_candidate_spots': int(target_candidate.sum()), 'target_retained_spots': int(target_retained.sum()), 'target_retained_fraction': float(target_retained.sum() / target_candidate.sum()), 'median_spotwise_profile_pearson': float(np.median(spotwise_pearson)), 'median_spotwise_profile_js_distance': float(np.median(spotwise_js)), 'truth_generated_log_library_pearson': float(np.corrcoef(truth_log_library, generated_log_library)[0, 1])}
    return (data, plot_data, audit)

def condition_label_maps(case_data: list[dict]) -> dict[str, dict[str, str]]:
    maps = {}
    for dataset in ('dlpfc', 'merfish'):
        labels = sorted({label for data in case_data if data['case']['dataset'] == dataset for label in data['eligible_labels']})
        if len(labels) > len(LABEL_COLORS):
            raise RuntimeError(f'{dataset} has more condition labels than distinct display colors')
        maps[dataset] = {label: LABEL_COLORS[index] for index, label in enumerate(labels)}
    return maps

def build_general_spatial_effect(case_data: list[dict], label_maps: dict[str, dict[str, str]]) -> plt.Figure:
    configure_matplotlib()
    fig = plt.figure(figsize=(8.8, 1.55 * len(case_data) + 1.15))
    grid = fig.add_gridspec(len(case_data) + 1, 2, width_ratios=[5.0, 1.6], height_ratios=[*([1.0] * len(case_data)), 0.08], left=0.09, right=0.985, bottom=0.10, top=0.94, wspace=0.12, hspace=0.22)
    library_norm = mpl.colors.Normalize(vmin=-2.5, vmax=2.5)
    colorbar_grid = grid[-1, 0].subgridspec(1, 3, wspace=0.0)
    library_cax = fig.add_subplot(colorbar_grid[0, 1])
    dataset_rows = {dataset: [row for row, data in enumerate(case_data) if data['case']['dataset'] == dataset] for dataset in ('dlpfc', 'merfish')}
    legend_axes = {}
    for dataset, rows in dataset_rows.items():
        if rows:
            legend_axes[dataset] = fig.add_subplot(grid[min(rows):max(rows) + 1, 1])
            legend_axes[dataset].set_axis_off()
    library_artist = None
    for row, data in enumerate(case_data):
        if data['case']['dataset'] == 'dlpfc':
            map_grid = grid[row, 0].subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.8], wspace=0.16)
        else:
            map_grid = grid[row, 0].subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.2], wspace=0.16)
        axes = [fig.add_subplot(map_grid[0, index]) for index in range(3)]
        coordinates = data['coordinates']
        target_xy = data['generated_coordinates']
        point_size = float(data['case']['point_size'])
        support_mask = data['source_retained'] | data['target_retained']
        for ax in axes:
            background(ax, data)
            setup_spatial_axis(ax, coordinates)
            add_masked_region_box(ax, data['split'])
            add_split_line(ax, data['split'])
        label_map = label_maps[data['case']['dataset']]
        for label in data['eligible_labels']:
            mask = (data['labels'] == label) & support_mask
            axes[0].scatter(coordinates[mask, 0], coordinates[mask, 1], s=point_size, color=label_map[label], linewidths=0, zorder=2)
        show_unsupported(axes[0], data)
        observed_colors = muted_library_colors(data['observed_library_z'][data['source_retained']], library_norm)
        for ax in axes[1:]:
            ax.scatter(coordinates[data['source_retained'], 0], coordinates[data['source_retained'], 1], c=observed_colors, s=point_size, linewidths=0, zorder=1)
        library_artist = axes[1].scatter(target_xy[:, 0], target_xy[:, 1], c=np.clip(data['truth_library_z'], -2.5, 2.5), cmap='viridis', norm=library_norm, s=point_size, linewidths=0, zorder=2)
        axes[2].scatter(target_xy[:, 0], target_xy[:, 1], c=np.clip(data['generated_library_z'], -2.5, 2.5), cmap='viridis', norm=library_norm, s=point_size, linewidths=0, zorder=2)
        if row == 0:
            titles = ('Known condition\nlabels + XY', 'Held-out truth\nlog library size z', 'FEAST simulation\nlog library size z')
            for ax, title in zip(axes, titles):
                ax.set_title(title, fontsize=7.5, pad=5)
        short_slice = data['case']['slice'].replace('Zhuang-ABCA-1.', '')
        axes[0].text(-0.19, 0.5, f"{data['case']['dataset_label']}\n{short_slice}\n{data['n_genes']:,} genes", transform=axes[0].transAxes, ha='right', va='center', fontsize=7.5, fontweight='bold', linespacing=1.35)
    for dataset, label_map in label_maps.items():
        if dataset not in legend_axes:
            continue
        handles = [plt.Line2D([], [], marker='o', linestyle='none', markersize=4.2, color=color, label=label.replace('_', ' ') if dataset == 'dlpfc' else label) for label, color in label_map.items()]
        legend_axes[dataset].legend(handles=handles, title=f"{dataset.upper() if dataset == 'dlpfc' else 'MERFISH'} condition labels", loc='center left', ncol=1, handletextpad=0.35, fontsize=5.9, title_fontsize=6.6, borderaxespad=0.0)
    library_colorbar = fig.colorbar(library_artist, cax=library_cax, orientation='horizontal')
    library_colorbar.set_ticks([-2, -1, 0, 1, 2])
    library_colorbar.set_label('Log library size z (truth-referenced)', fontsize=6.3, labelpad=2)
    library_colorbar.ax.tick_params(labelsize=5.8, length=2, width=0.5)
    return fig

def schematic_points() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = np.meshgrid(np.linspace(-1.0, 1.0, 13), np.linspace(-0.8, 0.8, 9))
    points = np.column_stack([x.ravel(), y.ravel()])
    keep = (points[:, 0] / 1.05) ** 2 + (points[:, 1] / 0.85) ** 2 <= 1.0
    points = points[keep]
    labels = np.digitize(points[:, 1] + 0.18 * points[:, 0], (-0.25, 0.25))
    expression = np.clip(0.15 + 0.58 * (labels == 2) + 0.28 * (labels == 1) + 0.18 * (points[:, 0] + 1), 0, 1)
    return (points, labels, expression)

def draw_small_slice(ax: plt.Axes, center: tuple[float, float], size: tuple[float, float], *, mode: str, half: str | None=None) -> None:
    points, labels, expression = schematic_points()
    x = center[0] + points[:, 0] * size[0] / 2
    y = center[1] + points[:, 1] * size[1] / 2
    if mode == 'labels':
        colors = [LABEL_COLORS[index] for index in labels]
        ax.scatter(x, y, s=8, c=colors, linewidths=0, clip_on=False)
    elif mode == 'expression':
        ax.scatter(x, y, s=8, c=expression, cmap=EXPRESSION_CMAP, vmin=0, vmax=1, linewidths=0, clip_on=False)
    elif mode == 'masked':
        visible = points[:, 0] <= 0
        if half == 'right_truth':
            visible = ~visible
        ax.scatter(x, y, s=8, color='#E2E2E2', linewidths=0, clip_on=False)
        ax.scatter(x[visible], y[visible], s=8, c=expression[visible], cmap=EXPRESSION_CMAP, vmin=0, vmax=1, linewidths=0, clip_on=False)
        ax.plot([center[0], center[0]], [center[1] - size[1] * 0.48, center[1] + size[1] * 0.48], color='#555555', linestyle=(0, (2, 2)), linewidth=0.8)
    else:
        raise ValueError(f'unknown schematic slice mode: {mode}')

def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10, linewidth=1.0, color='#555555'))

def feast_box(ax: plt.Axes, center: tuple[float, float]) -> None:
    width, height = (0.12, 0.28)
    ax.add_patch(FancyBboxPatch((center[0] - width / 2, center[1] - height / 2), width, height, boxstyle='round,pad=0.018,rounding_size=0.018', facecolor='#E7F0FA', edgecolor='#0F4D92', linewidth=1.0))
    ax.text(center[0], center[1], 'FEAST', ha='center', va='center', fontweight='bold', color='#0F4D92')

def build_design_schematic() -> plt.Figure:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 3.35))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
    cross = axes[0]
    cross.text(0.0, 0.93, 'Cross-slice conditional transfer', ha='left', va='top', fontsize=9.5, fontweight='bold')
    draw_small_slice(cross, (0.12, 0.52), (0.16, 0.54), mode='expression')
    cross.text(0.12, 0.16, 'source expression\n+ source labels + XY', ha='center', va='top', fontsize=7.1)
    draw_small_slice(cross, (0.35, 0.52), (0.16, 0.54), mode='labels')
    cross.text(0.35, 0.16, 'target labels + XY\n(expression hidden)', ha='center', va='top', fontsize=7.1)
    arrow(cross, (0.44, 0.52), (0.54, 0.52))
    feast_box(cross, (0.61, 0.52))
    arrow(cross, (0.68, 0.52), (0.75, 0.52))
    draw_small_slice(cross, (0.83, 0.52), (0.16, 0.54), mode='expression')
    cross.text(0.83, 0.16, 'simulated target\nexpression', ha='center', va='top', fontsize=7.1)
    cross.text(0.83, 0.88, 'scored against held-out truth', ha='center', va='center', fontsize=7.1, color='#666666')
    cross.text(0.235, 0.52, '+', ha='center', va='center', fontsize=13, color='#666666')
    half = axes[1]
    half.text(0.0, 0.93, 'Half-slice conditional generation', ha='left', va='top', fontsize=9.5, fontweight='bold')
    draw_small_slice(half, (0.18, 0.52), (0.22, 0.58), mode='masked')
    half.text(0.18, 0.15, 'visible left expression\n+ labels + XY on both halves', ha='center', va='top', fontsize=7.1)
    arrow(half, (0.31, 0.52), (0.43, 0.52))
    feast_box(half, (0.5, 0.52))
    arrow(half, (0.57, 0.52), (0.65, 0.52))
    draw_small_slice(half, (0.75, 0.52), (0.22, 0.58), mode='expression')
    half.text(0.75, 0.15, 'completed slice', ha='center', va='top', fontsize=7.1)
    draw_small_slice(half, (0.92, 0.52), (0.13, 0.48), mode='masked', half='right_truth')
    half.text(0.92, 0.15, 'held-out right\nfor scoring only', ha='center', va='top', fontsize=7.1, color='#666666')
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.12, top=0.98, hspace=0.2)
    return fig

def save_figure(figure: plt.Figure, work_dir: Path, stem: str, dpi: int) -> None:
    figure.savefig(work_dir / f'{stem}.pdf', bbox_inches='tight', metadata={'Creator': 'FEAST Study 05 simulation-effect plotting script', 'CreationDate': None, 'ModDate': None})
    figure.savefig(work_dir / f'{stem}.svg', bbox_inches='tight', metadata={'Creator': 'FEAST Study 05 simulation-effect plotting script', 'Date': None})
    figure.savefig(work_dir / f'{stem}.png', dpi=dpi, bbox_inches='tight', metadata={'Software': 'FEAST Study 05 simulation-effect plotting script'})
    plt.close(figure)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=VIS_DIR / 'figures')
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_provenance, _, generation_support = verify_evidence()
    loaded = [load_general_case(case, generation_support[job_id(case)]) for case in CASES]
    case_data = [item[0] for item in loaded]
    spot_tables = [item[1] for item in loaded]
    audits = [item[2] for item in loaded]
    label_maps = condition_label_maps(case_data)
    main_case_keys = {('dlpfc', '151670'), ('merfish', 'Zhuang-ABCA-1.006')}
    main_cases = [data for data in case_data if (data['case']['dataset'], data['case']['slice']) in main_case_keys]
    supplementary_cases = [data for data in case_data if (data['case']['dataset'], data['case']['slice']) not in main_case_keys]
    work_dir = Path(tempfile.mkdtemp(prefix=f'.{args.output_dir.name}-simulation-effect-', dir=args.output_dir.parent))
    try:
        save_figure(build_design_schematic(), work_dir, 'conditional_transfer_simulation_design', int(args.dpi))
        save_figure(build_general_spatial_effect(main_cases, label_maps), work_dir, 'conditional_transfer_spatial_effect', int(args.dpi))
        save_figure(build_general_spatial_effect(supplementary_cases, label_maps), work_dir, 'conditional_transfer_spatial_effect_supplementary', int(args.dpi))
        pd.concat(spot_tables, ignore_index=True).to_csv(work_dir / 'simulation_effect_plot_data.csv', index=False)
        pd.DataFrame(audits).to_csv(work_dir / 'simulation_effect_summary.csv', index=False)
        outputs = {path.name: None for path in sorted(work_dir.iterdir()) if path.is_file()}
        provenance = {'status': 'ok', 'configuration_id': 'study05-conditional-transfer-simulation-effect-v6', 'scientific_disposition': 'validated_supporting_visualization_claim_reframe_required', 'figure_promotion_authorized': False, 'publication_claim_authorized': False, 'target_expression_loaded_by_generation': False, 'primary_assignment_randomness': 0.3, 'score_configuration_id': score_provenance['configuration_id'], 'general_spatial_scope': {'half_slice_cases': len(case_data), 'datasets': ['dlpfc', 'merfish'], 'slices': [audit['slice'] for audit in audits], 'main_figure_slices': [data['case']['slice'] for data in main_cases], 'supplementary_figure_slices': [data['case']['slice'] for data in supplementary_cases], 'single_gene_selection': False, 'all_fixed_panel_genes': True}, 'spatial_design': {'experiment': 'deterministic x-median half-slice masking', 'columns': ['known labels and XY', 'held-out truth log library size z', 'FEAST-generated log library size z'], 'condition_label_colors': label_maps, 'observed_half_library_size': 'ground-truth values in 55% viridis blended with light gray', 'target_half_library_size': 'fully saturated held-out truth or FEAST values', 'masked_region_box': 'light-gray translucent fill with solid gray outline from the x-median split to the right axis boundary', 'library_size_scale': 'truth-referenced within-slice z score, clipped to [-2.5, 2.5] for display', 'unsupported_spots_marked': True, 'panel_letters': False, 'vertical_value_jitter': False}, 'claim_limits': ['conditional on observed target XY and labels', 'target expression is used only after generation for scoring and display', 'library-size agreement does not establish accurate gene-level imputation', 'spotwise all-gene profile metrics remain available in the plotting-data table but are not displayed'], 'png_dpi': int(args.dpi), 'vector_text': {'pdf_fonttype': 42, 'svg_fonttype': 'none', 'adobe_illustrator_editable_text': True, 'deterministic_export_metadata': True}, 'outputs': outputs, 'created_at': datetime.now(timezone.utc).isoformat()}
        (work_dir / 'simulation_effect_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
        published = {}
        for path in sorted(work_dir.iterdir()):
            if path.is_file():
                destination = args.output_dir / path.name
                os.replace(path, destination)
                published[destination.name] = None
        print(json.dumps({'status': 'ok', 'output_dir': str(args.output_dir), 'half_slice_cases': len(audits), 'single_gene_selection': False, 'outputs': published}, indent=2))
    finally:
        if work_dir.exists():
            work_dir.rmdir()
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
