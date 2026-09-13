"""Render baseline-relative partition change across Study 01 perturbations.

For each registered slice, method, and alteration level, this figure compares
the perturbed predicted partition with that method's FEAST-baseline partition.
ARI and NMI are invariant to a permutation of cluster labels. The figure shows
the intuitive transform ``1 - ARI``: zero means an identical partition and a
higher value means greater reorganization relative to that method's baseline.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('SOURCE_DATE_EPOCH', '0')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PUBLICATION_MANIFEST = REPOSITORY_ROOT / 'PUBLICATION_MANIFEST.json'
OUTPUT_DIR = Path(__file__).resolve().parent / 'figures'
METHODS_DIR = REPOSITORY_ROOT / '01_clustering' / 'outputs' / 'final_rerun_20260718' / 'methods'
SLICES = ('151508', '151670', '151676')
METHOD_SPECS = (('GraphST', 'GraphST', '#0072B2', 'o'), ('STAGATE_mclust', 'STAGATE + mclust', '#E69F00', 's'), ('Leiden_unsupervised', 'Label-free Leiden', '#009E73', '^'))
FACTOR_SPECS = {'mean': {'title': 'Mean control', 'x_label': 'Nominal fold change (neutral = 1)', 'neutral': 1.0, 'levels': ((0.5, 'mean_0_50'), (0.67, 'mean_0_67'), (0.8, 'mean_0_80'), (1.0, 'baseline'), (1.25, 'mean_1_25'), (1.5, 'mean_1_50'), (2.0, 'mean_2'))}, 'variance': {'title': 'Variance control', 'x_label': 'Nominal fold change (neutral = 1)', 'neutral': 1.0, 'levels': ((0.5, 'variance_0_50'), (0.67, 'variance_0_67'), (0.8, 'variance_0_80'), (1.0, 'baseline'), (1.25, 'variance_1_25'), (1.5, 'variance_1_50'), (2.0, 'variance_2'))}, 'sparsity': {'title': 'Zero-fraction control', 'x_label': 'Nominal logit shift (neutral = 0)', 'neutral': 0.0, 'levels': ((-1.0, 'sparsity_neg_1'), (-0.5, 'sparsity_neg_0_50'), (-0.25, 'sparsity_neg_0_25'), (0.0, 'baseline'), (0.25, 'sparsity_pos_0_25'), (0.5, 'sparsity_pos_0_50'), (1.0, 'sparsity_pos_1'))}}
PLOT_METRIC = 'partition_change_ari'

def configure_matplotlib() -> None:
    """Apply the trend-panel style from scientific-figure-making."""
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'axes.linewidth': 1.2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.facecolor': 'white', 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def registered_metric_source() -> Path:
    manifest = json.loads(PUBLICATION_MANIFEST.read_text())
    study = manifest['studies']['01_clustering']
    if study['status'] != 'validated_complete':
        raise ValueError('Study 01 is not registered as validated_complete.')
    candidate = study['canonical_candidate']
    return REPOSITORY_ROOT / candidate['metrics']

def verify_registered_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f'Registered input is missing: {path}')

def _load_run_manifests() -> dict[str, pd.DataFrame]:
    manifests: dict[str, pd.DataFrame] = {}
    for method, _label, _color, _marker in METHOD_SPECS:
        path = METHODS_DIR / method / 'run_manifest.csv'
        run = pd.read_csv(path, dtype={'slice_id': str, 'simulation_id': str})
        required = {'slice_id', 'simulation_id', 'status'}
        missing = required - set(run.columns)
        if missing:
            raise ValueError(f'{path} is missing columns: {sorted(missing)}')
        if not run['status'].eq('ok').all():
            raise ValueError(f'{path} includes an unsuccessful method run')
        manifests[method] = run.set_index(['slice_id', 'simulation_id'], verify_integrity=True)
    return manifests

def _clusters_path(method: str, slice_id: str, simulation_id: str) -> Path:
    return METHODS_DIR / method / slice_id / simulation_id / 'clusters.csv'

def _read_labels(path: Path) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(f'Cluster file is missing: {path}')
    table = pd.read_csv(path, dtype=str)
    required = {'spot_barcode', 'predicted_cluster'}
    missing = required - set(table.columns)
    if missing or table['spot_barcode'].duplicated().any():
        raise ValueError(f'Invalid cluster table: {path}')
    return table.set_index('spot_barcode')['predicted_cluster'].astype(str)

def load_stability_data() -> tuple[pd.DataFrame, dict[str, str], int]:
    metric_path = registered_metric_source()
    verify_registered_file(metric_path)
    run_manifests = _load_run_manifests()
    labels_cache: dict[tuple[str, str, str], pd.Series] = {}
    verified_paths: dict[str, str] = {}

    def get_labels(method: str, slice_id: str, simulation_id: str) -> pd.Series:
        key = (method, slice_id, simulation_id)
        if key not in labels_cache:
            manifest = run_manifests[method]
            manifest_key = (slice_id, simulation_id)
            if manifest_key not in manifest.index:
                raise ValueError(f'Missing registered method output: {method}/{slice_id}/{simulation_id}')
            path = _clusters_path(method, slice_id, simulation_id)
            labels_cache[key] = _read_labels(path)
            verified_paths[str(path.relative_to(REPOSITORY_ROOT))] = None
        return labels_cache[key]
    rows: list[dict[str, object]] = []
    for method, _label, _color, _marker in METHOD_SPECS:
        for slice_id in SLICES:
            baseline = get_labels(method, slice_id, 'baseline')
            for factor, spec in FACTOR_SPECS.items():
                for nominal_level, simulation_id in spec['levels']:
                    perturbed = get_labels(method, slice_id, simulation_id)
                    if not baseline.index.equals(perturbed.index):
                        if set(baseline.index) != set(perturbed.index):
                            raise ValueError(f'Spot set changed for {method}/{slice_id}/{simulation_id}')
                        perturbed = perturbed.reindex(baseline.index)
                    stability_ari = adjusted_rand_score(baseline, perturbed)
                    stability_nmi = normalized_mutual_info_score(baseline, perturbed)
                    rows.append({'slice_id': slice_id, 'method': method, 'factor': factor, 'simulation_id': simulation_id, 'nominal_level': nominal_level, 'stability_ari': stability_ari, 'stability_nmi': stability_nmi, 'partition_change_ari': 1.0 - stability_ari})
    data = pd.DataFrame(rows)
    expected_rows = len(SLICES) * len(METHOD_SPECS) * sum((len(spec['levels']) for spec in FACTOR_SPECS.values()))
    if len(data) != expected_rows:
        raise ValueError(f'Expected {expected_rows} stability rows; found {len(data)}')
    return (data, verified_paths, len(labels_cache))

def _format_xticks(ax: plt.Axes, factor: str) -> None:
    spec = FACTOR_SPECS[factor]
    levels = [level for level, _sim in spec['levels']]
    if factor in {'mean', 'variance'}:
        ax.set_xticks([0.5, 1.0, 1.5, 2.0])
    else:
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlim(min(levels) - 0.06 * (max(levels) - min(levels)), max(levels) + 0.06 * (max(levels) - min(levels)))

def draw_figure(data: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    configure_matplotlib()
    factors = tuple(FACTOR_SPECS)
    fig, axes = plt.subplots(1, len(factors), figsize=(13.0, 4.3))
    for col_idx, factor in enumerate(factors):
        ax = axes[col_idx]
        subset = data.loc[data['factor'] == factor]
        for method, _method_label, color, marker in METHOD_SPECS:
            method_data = subset.loc[subset['method'] == method]
            for slice_id in SLICES:
                slice_data = method_data.loc[method_data['slice_id'] == slice_id].sort_values('nominal_level')
                ax.plot(slice_data['nominal_level'], slice_data[PLOT_METRIC], color=color, linewidth=0.85, alpha=0.28, zorder=1)
            summary = method_data.groupby('nominal_level', as_index=False)[PLOT_METRIC].mean()
            ax.plot(summary['nominal_level'], summary[PLOT_METRIC], color=color, marker=marker, markerfacecolor='white', markeredgewidth=0.9, markersize=5.8, linewidth=2.4, zorder=3)
        ax.axhline(0.0, color='#4D4D4D', linewidth=0.9, linestyle='--', alpha=0.8, zorder=0)
        ax.axvline(FACTOR_SPECS[factor]['neutral'], color='#767676', linewidth=0.85, linestyle='--', alpha=0.8, zorder=0)
        ax.set_ylim(0.0, 1.02)
        _format_xticks(ax, factor)
        ax.grid(axis='y', color='#E8E8E8', linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', labelsize=9)
        ax.set_xlabel(FACTOR_SPECS[factor]['x_label'], fontsize=10)
        if col_idx == 0:
            ax.set_ylabel('Partition change from\nFEAST baseline (1 − ARI) ↑', fontsize=10.5, fontweight='bold', labelpad=6)
        ax.set_title(FACTOR_SPECS[factor]['title'], fontsize=12.5, fontweight='bold', pad=10)
    handles = [Line2D([0], [0], color=color, marker=marker, markerfacecolor='white', markeredgewidth=0.9, linewidth=2.25, markersize=6, label=label) for _method, label, color, marker in METHOD_SPECS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.55, 0.91), ncol=3, fontsize=10, handlelength=1.8, columnspacing=1.4)
    fig.text(0.08, 0.976, 'How much does the predicted partition change?', ha='left', va='center', fontsize=14, fontweight='bold')
    fig.text(0.08, 0.933, '1 − ARI versus the same-method FEAST baseline: 0 = identical partition; higher = stronger reorganization. Thin = slice; solid = mean.', ha='left', va='center', fontsize=8.8, color='#666666')
    fig.subplots_adjust(left=0.095, right=0.995, top=0.745, bottom=0.18, wspace=0.25)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / 'clustering_stability_vs_baseline'
    paths: list[Path] = []
    for suffix, kwargs in {'pdf': {'metadata': {'Creator': 'FEAST Study 01 clustering stability'}}, 'png': {'dpi': dpi, 'metadata': {'Software': 'FEAST Study 01 clustering stability'}}, 'svg': {'metadata': {'Creator': 'FEAST Study 01 clustering stability'}}}.items():
        path = stem.with_suffix(f'.{suffix}')
        fig.savefig(path, **kwargs)
        print(f'Saved: {path}')
        paths.append(path)
    plt.close(fig)
    return paths

def write_provenance(output_dir: Path, outputs: list[Path], verified_paths: dict[str, str], verified_cluster_count: int) -> Path:
    metric_path = registered_metric_source()
    payload = {'schema_version': 1, 'figure_set': 'study01_partition_change_from_baseline', 'source': {'plot_script': str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)), 'registered_metric_source': str(metric_path.relative_to(REPOSITORY_ROOT)), 'clusters_csv_verified': verified_cluster_count, 'cluster_paths': sorted(verified_paths)}, 'design': {'slices': list(SLICES), 'factors': list(FACTOR_SPECS), 'display_metric': '1 - stability_ari', 'data_metrics': ['stability_ari', 'stability_nmi'], 'reference_partition': 'same-method FEAST baseline', 'label_remapping': 'none; ARI and NMI are label-permutation-invariant'}, 'outputs': {path.name: None for path in sorted(outputs)}, 'scientific_disposition': {'figure_promotion_authorized': False, 'supported_interpretation': 'Descriptive within-method partition change across controlled simulation perturbations; no cross-method ranking is claimed.'}}
    output_path = output_dir / 'clustering_stability_provenance.json'
    output_path.write_text(json.dumps(payload, indent=2) + '\n')
    return output_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--dpi', type=int, default=600)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    data, verified_paths, verified_cluster_count = load_stability_data()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / 'clustering_stability_data.csv'
    data.to_csv(data_path, index=False, float_format='%.12f')
    outputs = [data_path, *draw_figure(data, args.output_dir, args.dpi)]
    print(f'Saved: {write_provenance(args.output_dir, outputs, verified_paths, verified_cluster_count)}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
