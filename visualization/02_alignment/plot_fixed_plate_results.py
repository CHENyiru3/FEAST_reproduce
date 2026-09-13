"""Render an unpromoted diagnostic for the fixed-plate Study 02 run."""
from __future__ import annotations
import argparse
import json
import os
import platform
import sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ['SOURCE_DATE_EPOCH'] = '0'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_OUTPUT = REPOSITORY_ROOT / '02_alignment' / 'outputs' / 'fixed_plate_rerun_20260806_v1'
PASTE2_METRICS = REPOSITORY_ROOT / '02_alignment' / 'outputs' / 'paste2_rerun_20260807_v1' / 'scores_v3' / 'alignment_metrics.csv'
SPATEO_METRICS = DEFAULT_STUDY_OUTPUT / 'scores_v3' / 'alignment_metrics.csv'
METRIC_VALIDATION = REPOSITORY_ROOT / '02_alignment' / 'outputs' / 'paste2_rerun_20260807_v1' / 'metrics_v3_validation.csv'
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'figures'
OUTPUT_STEM = 'fixed_plate_alignment_diagnostic'
METHODS = ('paste2', 'spateo')
METHOD_LABELS = {'paste2': 'PASTE2', 'spateo': 'Spateo'}
ALTERATIONS = ('baseline', 'mean_0.5', 'variance_2.0', 'sparsity_0.5')
ALTERATION_LABELS = {'baseline': 'Baseline', 'mean_0.5': 'Mean ×0.5', 'variance_2.0': 'Variance ×2.0', 'sparsity_0.5': 'Sparsity ×0.5'}
METHOD_COLORS = {'paste2': '#0F4D92', 'spateo': '#B64342'}
METHOD_MARKERS = {'paste2': 'o', 'spateo': 's'}
METHOD_MARKER_SIZES = {'paste2': 7.0, 'spateo': 5.0}
ERROR_FLOOR = 0.0001
ROTATION_ERROR_FLOOR = 0.0001
METRIC_SPECS = ({'column': 'mean_spatial_error_spot_units', 'label': 'Normalized spatial error\n(spot spacings; ↓)', 'scale': 'log', 'limits': (ERROR_FLOOR, 2.0), 'ticks': None, 'floor': ERROR_FLOOR}, {'column': 'rotation_recovery_error', 'label': 'Rotation recovery error\n(degrees; log scale; ↓)', 'scale': 'log', 'limits': (ROTATION_ERROR_FLOOR, 2.0), 'ticks': None, 'floor': ROTATION_ERROR_FLOOR})

def repository_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT))

def load_validated_data(study_output: Path) -> tuple[pd.DataFrame, dict]:
    metrics_path = study_output / 'scores_v3' / 'alignment_metrics.csv'
    rotations_path = study_output / 'rotations' / 'rotation_manifest.csv'
    config_path = REPOSITORY_ROOT / '02_alignment' / 'config.yaml'
    spateo_metrics = pd.read_csv(metrics_path)
    paste2_metrics = pd.read_csv(PASTE2_METRICS)
    metrics = pd.concat([spateo_metrics, paste2_metrics], ignore_index=True)
    validation = pd.read_csv(METRIC_VALIDATION)
    rotations = pd.read_csv(rotations_path)
    config = yaml.safe_load(config_path.read_text())
    angles = tuple((float(value) for value in config['angles']))
    expected = {(method, alteration, angle) for method in METHODS for alteration in ALTERATIONS for angle in angles}
    observed = set(zip(metrics['method'], metrics['alteration'], metrics['angle_degrees'].astype(float)))
    if len(metrics) != len(expected) or observed != expected or metrics['job_id'].duplicated().any() or (not metrics['status'].eq('ok').all()) or (not metrics['canonical_candidate'].all()) or (len(validation) != len(expected)) or (not validation['valid'].all()):
        raise RuntimeError('PASTE2 and Spateo metrics do not cover the exact fixed-plate design')
    support = rotations[['alteration', 'angle_degrees', 'input_n_spots', 'retained_n_spots', 'dropped_n_spots']]
    data = metrics.merge(support, on=['alteration', 'angle_degrees'], how='left', validate='many_to_one')
    if data['retained_fraction'].isna().any():
        raise RuntimeError('rotation support metadata did not join exactly')
    sources = {'spateo_metrics': metrics_path, 'paste2_metrics': PASTE2_METRICS, 'spateo_score_provenance': metrics_path.with_name('provenance.json'), 'paste2_score_provenance': PASTE2_METRICS.with_name('provenance.json'), 'metric_validation': METRIC_VALIDATION, 'rotations': rotations_path, 'config': config_path, 'angles': angles}
    return (data, sources)

def configure_matplotlib() -> None:
    arial_paths = (Path(sys.prefix) / 'fonts' / 'arial.ttf', Path(sys.prefix) / 'fonts' / 'arialbd.ttf')
    if not all((path.is_file() for path in arial_paths)):
        raise FileNotFoundError(f'Arial Regular and Bold are required for this figure: {arial_paths}')
    for path in arial_paths:
        font_manager.fontManager.addfont(path)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12, 'axes.linewidth': 1.6, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def draw_figure(data: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(len(METRIC_SPECS), len(ALTERATIONS), figsize=(14.5, 7.2), sharex=True, sharey='row', squeeze=False)
    angles = np.asarray(sorted(data['angle_degrees'].unique()), dtype=float)
    retention = data.groupby('angle_degrees')['retained_fraction'].first().reindex(angles)
    tick_labels = [f'{angle:g}°\n{100 * fraction:.0f}%' for angle, fraction in zip(angles, retention)]
    for row_index, spec in enumerate(METRIC_SPECS):
        for column_index, alteration in enumerate(ALTERATIONS):
            ax = axes[row_index, column_index]
            for method in METHODS:
                selected = data[(data['method'] == method) & (data['alteration'] == alteration)].sort_values('angle_degrees')
                raw_values = selected[spec['column']].to_numpy(dtype=float)
                plotted_values = raw_values
                if spec['floor'] is not None:
                    plotted_values = np.maximum(raw_values, spec['floor'])
                ax.plot(selected['angle_degrees'], plotted_values, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], markerfacecolor='none', markeredgewidth=1.1, markersize=METHOD_MARKER_SIZES[method], linewidth=1.9, zorder=3)
                if spec['floor'] is not None:
                    at_floor = raw_values < spec['floor']
                    if at_floor.any():
                        ax.scatter(selected.loc[at_floor, 'angle_degrees'], np.full(at_floor.sum(), spec['floor']), marker='v', s=28, color=METHOD_COLORS[method], zorder=4, clip_on=False)
            if spec['scale'] == 'log':
                ax.set_yscale('log')
            ax.set_ylim(*spec['limits'])
            if spec['ticks'] is not None:
                ax.set_yticks(spec['ticks'])
                if 'accuracy' in spec['column']:
                    ax.set_yticklabels([f'{100 * tick:.0f}' for tick in spec['ticks']])
            ax.set_xticks(angles, tick_labels)
            ax.set_xlim(float(angles.min()) - 3, float(angles.max()) + 3)
            ax.grid(axis='y', color='#CFCECE', linewidth=0.65, alpha=0.7)
            ax.set_axisbelow(True)
            ax.tick_params(axis='x', labelbottom=row_index == len(METRIC_SPECS) - 1, labelsize=10)
            if column_index == 0:
                ax.set_ylabel(spec['label'], labelpad=12)
            if row_index == 0:
                ax.set_title(ALTERATION_LABELS[alteration], fontweight='bold', pad=9)
                ax.text(0.02, 0.98, chr(ord('A') + column_index), transform=ax.transAxes, ha='left', va='top', fontsize=14, fontweight='bold')
    handles = [Line2D([0], [0], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], markerfacecolor='none', markeredgewidth=1.1, linewidth=2.2, label=METHOD_LABELS[method]) for method in METHODS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.91), ncol=2, handlelength=2.0, columnspacing=2.2)
    fig.suptitle('Fixed-plate alignment robustness', x=0.5, y=0.995, fontsize=18, fontweight='bold')
    fig.text(0.5, 0.945, 'Primary metrics use every retained moving spot after rigid alignment; lower values are better.', ha='center', va='center', fontsize=11.5, color='#4D4D4D')
    fig.text(0.5, 0.035, 'Input rotation angle and retained spots', ha='center')
    fig.text(0.99, 0.012, 'Downward triangles in log-scale rows mark values at the plotting floor.', ha='right', fontsize=10, color='#4D4D4D')
    fig.subplots_adjust(left=0.085, right=0.995, top=0.8, bottom=0.15, wspace=0.16, hspace=0.24)
    paths = []
    for suffix, metadata in (('png', {'Software': 'FEAST Study 02 fixed-plate diagnostic'}), ('pdf', {'CreationDate': None, 'ModDate': None}), ('svg', {'Date': None})):
        path = output_dir / f'{OUTPUT_STEM}.{suffix}'
        fig.savefig(path, dpi=dpi, bbox_inches='tight', metadata=metadata)
        paths.append(path)
    plt.close(fig)
    return paths

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-output', type=Path, default=DEFAULT_STUDY_OUTPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    data, sources = load_validated_data(args.study_output)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / f'{OUTPUT_STEM}_data.csv'
    data.to_csv(data_path, index=False)
    output_paths = [data_path, *draw_figure(data, args.output_dir, args.dpi)]
    provenance = {'schema_version': 1, 'figure_set': 'study02_fixed_plate_alignment_diagnostic', 'promotion_status': 'unpromoted_diagnostic', 'publication_figure_authorized': False, 'aggregate_method_ranking_authorized': False, 'winner_claim_authorized': False, 'validated_atomic_rows': int(len(data)), 'methods': list(METHODS), 'alterations': list(ALTERATIONS), 'angles_degrees': list(sources['angles']), 'source': {name: {'path': repository_path(path)} for name, path in sources.items() if isinstance(path, Path)}, 'plot_script': {'path': repository_path(Path(__file__))}, 'render_environment': {'python': platform.python_version(), 'matplotlib': matplotlib.__version__}, 'outputs': {path.name: None for path in sorted(output_paths, key=lambda item: item.name)}}
    provenance_path = args.output_dir / f'{OUTPUT_STEM}_provenance.json'
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    for path in [*output_paths, provenance_path]:
        print(f'Saved: {path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
