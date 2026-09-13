"""Render spatial clustering comparisons for the Study 01 representative slice.

Each result is shown with its own discrete clustering labels.  The panels are
for visual comparison of spatial partitioning; colours in predicted panels are
not remapped to reference domains or interpreted across methods.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('SOURCE_DATE_EPOCH', '0')
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / 'figures'
METHODS_DIR = REPOSITORY_ROOT / '01_clustering' / 'outputs' / 'final_rerun_20260718' / 'methods'
REGISTERED_METRICS_PATH = OUTPUT_DIR / 'alteration_plot_data.csv'
BASELINE_METRICS_PATH = OUTPUT_DIR / 'baseline_reference.csv'
SLICE = '151676'
METHOD_SPECS = (('GraphST', 'domain', 'GraphST'), ('STAGATE_mclust', 'mclust', 'STAGATE + mclust'), ('Leiden_unsupervised', 'predicted_cluster', 'Label-free Leiden'))
LEVEL_SPECS = {'mean': (('mean_0_50', 'FC\n0.50'), ('mean_0_67', 'FC\n0.67'), ('mean_0_80', 'FC\n0.80'), ('baseline', 'FEAST baseline\nFC 1.00'), ('mean_1_25', 'FC\n1.25'), ('mean_1_50', 'FC\n1.50'), ('mean_2', 'FC\n2.00')), 'variance': (('variance_0_50', 'FC\n0.50'), ('variance_0_67', 'FC\n0.67'), ('variance_0_80', 'FC\n0.80'), ('baseline', 'FEAST baseline\nFC 1.00'), ('variance_1_25', 'FC\n1.25'), ('variance_1_50', 'FC\n1.50'), ('variance_2', 'FC\n2.00')), 'sparsity': (('sparsity_neg_1', 'Shift\n−1.00'), ('sparsity_neg_0_50', 'Shift\n−0.50'), ('sparsity_neg_0_25', 'Shift\n−0.25'), ('baseline', 'FEAST baseline\nshift 0'), ('sparsity_pos_0_25', 'Shift\n+0.25'), ('sparsity_pos_0_50', 'Shift\n+0.50'), ('sparsity_pos_1', 'Shift\n+1.00'))}
CAT_PALETTE = ('#4C78A8', '#F58518', '#E45756', '#72B7B2', '#54A24B', '#B279A2', '#FF9DA6', '#9D755D', '#BAB0AC', '#D4A6C8')
POINT_SIZE = 8.2

def result_path(method: str, sim_id: str) -> Path:
    return METHODS_DIR / method / SLICE / sim_id / 'result.h5ad'

def read_panel(path: Path, label_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = ad.read_h5ad(path, backed='r')
    try:
        if label_key not in data.obs.columns:
            raise ValueError(f'{label_key!r} not in {path}')
        if 'ground_truth' not in data.obs.columns:
            raise ValueError(f"'ground_truth' not in {path}")
        if 'spatial' in data.obsm:
            coords = np.asarray(data.obsm['spatial'])
        elif {'x', 'y'}.issubset(data.obs.columns):
            coords = data.obs[['x', 'y']].to_numpy()
        else:
            raise ValueError(f'No spatial coordinates in {path}')
        return (coords[:, :2].astype(float), data.obs['ground_truth'].astype(str).to_numpy(), data.obs[label_key].astype(str).to_numpy())
    finally:
        data.file.close()

def ground_truth_order(labels: np.ndarray) -> list[str]:
    layer_order = [f'Layer_{i}' for i in range(1, 7)] + ['WM']
    present = set(labels)
    ordered = [label for label in layer_order if label in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered

def local_cluster_colors(labels: np.ndarray) -> np.ndarray:
    """Give the arbitrary labels of one clustering result a local palette."""
    unique_labels = sorted(set(labels), key=str)
    label_to_color = {label: CAT_PALETTE[idx % len(CAT_PALETTE)] for idx, label in enumerate(unique_labels)}
    return np.asarray([label_to_color[label] for label in labels], dtype=object)


def registered_ari_values(alter_type: str) -> dict[tuple[str, str], float]:
    """Return the registered ARI for each displayed method and simulation."""
    metrics = pd.read_csv(REGISTERED_METRICS_PATH)
    baseline = pd.read_csv(BASELINE_METRICS_PATH)
    alteration_type = 'sparsity_logit_shift' if alter_type == 'sparsity' else alter_type
    values: dict[tuple[str, str], float] = {}
    for method, _label_key, _method_label in METHOD_SPECS:
        for simulation_id, _level_label in LEVEL_SPECS[alter_type]:
            if simulation_id == 'baseline':
                matched = baseline[(baseline['slice_id'] == int(SLICE)) & (baseline['method'] == method)]
                metric_column = 'ARI_ref'
            else:
                matched = metrics[
                    (metrics['slice_id'] == int(SLICE))
                    & (metrics['method'] == method)
                    & (metrics['simulation_id'] == simulation_id)
                    & (metrics['alteration_type'] == alteration_type)
                ]
                metric_column = 'ARI'
            if len(matched) != 1:
                raise ValueError(f'Expected one registered ARI for {(SLICE, method, simulation_id)}, found {len(matched)}')
            values[(method, simulation_id)] = float(matched.iloc[0][metric_column])
    return values

def configure_matplotlib() -> None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def _draw_map(ax: plt.Axes, xy: np.ndarray, colors: np.ndarray) -> None:
    ax.scatter(xy[:, 0], xy[:, 1], s=POINT_SIZE, c=colors, marker='.', linewidths=0, rasterized=True)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')

def draw_one_figure(alter_type: str, output_dir: Path, dpi: int) -> list[Path]:
    if alter_type not in LEVEL_SPECS:
        raise ValueError(f'Unknown alteration type: {alter_type}')
    configure_matplotlib()
    reference_path = result_path('GraphST', 'baseline')
    reference_xy, reference_truth, _reference_prediction = read_panel(reference_path, 'domain')
    truth_order = ground_truth_order(reference_truth)
    truth_to_color = {label: CAT_PALETTE[idx % len(CAT_PALETTE)] for idx, label in enumerate(truth_order)}
    reference_colors = np.asarray([truth_to_color[label] for label in reference_truth], dtype=object)
    level_specs = LEVEL_SPECS[alter_type]
    ari_values = registered_ari_values(alter_type)
    fig, axes = plt.subplots(len(METHOD_SPECS), len(level_specs) + 1, figsize=(18.0, 9.3))
    for row_idx, (method, label_key, method_label) in enumerate(METHOD_SPECS):
        reference_ax = axes[row_idx, 0]
        _draw_map(reference_ax, reference_xy, reference_colors)
        reference_ax.text(-0.19, 0.5, method_label, transform=reference_ax.transAxes, ha='right', va='center', fontsize=11, fontweight='bold')
        for col_idx, (sim_id, _level_label) in enumerate(level_specs, start=1):
            ax = axes[row_idx, col_idx]
            path = result_path(method, sim_id)
            if not path.exists():
                ax.text(0.5, 0.5, 'missing', ha='center', va='center', transform=ax.transAxes, fontsize=10, color='#777777')
                ax.axis('off')
                continue
            xy, _truth, predicted = read_panel(path, label_key)
            colors = local_cluster_colors(predicted)
            _draw_map(ax, xy, colors)
            ax.text(0.5, -0.035, f'ARI = {ari_values[(method, sim_id)]:.3f}', transform=ax.transAxes,
                    ha='center', va='top', fontsize=9.0, fontweight='bold', color='#C62828')
    axes[0, 0].set_title('Reference\ndomains', fontsize=10.5, fontweight='bold', pad=9)
    for col_idx, (_sim_id, level_label) in enumerate(level_specs, start=1):
        axes[0, col_idx].set_title(level_label, fontsize=9.5, fontweight='bold', pad=9)
    alter_title = {'mean': 'Mean perturbation', 'variance': 'Variance perturbation', 'sparsity': 'Zero-fraction perturbation'}
    fig.suptitle(f'{alter_title[alter_type]} across all registered levels — DLPFC slice {SLICE}', fontsize=14, fontweight='bold', x=0.54, y=0.985)
    fig.text(0.54, 0.949, 'Columns show increasing alteration level (neutral FEAST baseline centred); predicted colours encode local cluster labels', ha='center', va='center', fontsize=9.3, color='#666666')
    fig.subplots_adjust(left=0.15, right=0.985, top=0.9, bottom=0.045, wspace=0.025, hspace=0.11)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f'alteration_clustering_spatial_{alter_type}'
    paths: list[Path] = []
    for suffix, kwargs in {'pdf': {'metadata': {'Creator': 'FEAST Study 01 clustering spatial'}}, 'png': {'dpi': dpi, 'metadata': {'Software': 'FEAST Study 01 clustering spatial'}}, 'svg': {'metadata': {'Creator': 'FEAST Study 01 clustering spatial'}}}.items():
        path = stem.with_suffix(f'.{suffix}')
        fig.savefig(path, **kwargs)
        print(f'Saved: {path}')
        paths.append(path)
    plt.close(fig)
    return paths

def write_provenance(output_dir: Path, outputs: list[Path]) -> Path:
    input_paths = [result_path(method, sim_id) for sim_id, _label in LEVEL_SPECS['mean'] for method, _key, _label in METHOD_SPECS]
    for alter_type in ('variance', 'sparsity'):
        input_paths.extend((result_path(method, sim_id) for sim_id, _label in LEVEL_SPECS[alter_type] for method, _key, _label in METHOD_SPECS))
    payload = {'schema_version': 1, 'figure_set': 'study01_spatial_clustering', 'display_slice': SLICE, 'predicted_label_display': 'Each predicted panel uses its own local categorical-label palette; no label remapping.', 'source': {'plot_script': str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)), 'inputs': {str(path.resolve().relative_to(REPOSITORY_ROOT)): None for path in sorted(set(input_paths))}}, 'outputs': {path.name: None for path in sorted(outputs)}, 'scientific_disposition': {'figure_promotion_authorized': False, 'supported_interpretation': 'Descriptive spatial comparison of a registered representative slice.'}}
    output_path = output_dir / 'spatial_clustering_provenance.json'
    output_path.write_text(json.dumps(payload, indent=2) + '\n')
    return output_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--dpi', type=int, default=600)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    outputs: list[Path] = []
    for alter_type in ('mean', 'variance', 'sparsity'):
        outputs.extend(draw_one_figure(alter_type, args.output_dir, args.dpi))
    print(f'Saved: {write_provenance(args.output_dir, outputs)}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
