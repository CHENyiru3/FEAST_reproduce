"""Render the validated Study 03 cell-class deconvolution comparison."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/feast-reproduce-matplotlib')
os.environ.setdefault('SOURCE_DATE_EPOCH', '0')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPOSITORY_ROOT / '03_deconvolution' / 'outputs' / 'cell_class_rerun_20260810_v1'
DEFAULT_SCORES_DIR = DEFAULT_RUN_DIR / 'scores_cell_class'
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'figures'
METHOD_ORDER = ('RCTD', 'Cell2location')
METHOD_LABELS = {'rctd': 'RCTD', 'cell2location': 'Cell2location'}
COLORS = {'RCTD': '#365C8D', 'Cell2location': '#E07A3F'}
MARKERS = {'RCTD': 'o', 'Cell2location': 's'}
METRICS = (('jsd_mean', 'Composition divergence', 'Mean Jensen–Shannon divergence · lower is better'), ('pearson_mean', 'Composition correlation', 'Mean Pearson correlation · higher is better'), ('rmse', 'Composition error', 'RMSE · lower is better'))

def repository_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT))

def verified(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f'missing {label}: {path}')
    return path

def load_inputs(run_dir: Path, scores_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    provenance_path = scores_dir / 'scoring_provenance.json'
    validation_path = run_dir / 'validation_cell_class.csv'
    provenance = json.loads(provenance_path.read_text())
    if provenance.get('analysis') != 'study03_cell_class_exact_matched_support':
        raise ValueError('wrong Study 03 scoring layer')
    scores_path = verified(scores_dir / 'deconvolution_scores.csv', 'cell-class scores')
    support_path = verified(scores_dir / 'support_audit.csv', 'cell-class support audit')
    validation = pd.read_csv(validation_path)
    if len(validation) != 1 or validation.loc[0, 'status'] != 'validated' or (not bool(validation.loc[0, 'all_metrics_independently_reproduced'])) or (not bool(validation.loc[0, 'exact_method_support_verified'])) or (not bool(validation.loc[0, 'cuda_execution_verified_for_all_cell2location_jobs'])):
        raise ValueError('cell-class validation record is incomplete or stale')
    scores = pd.read_csv(scores_path, dtype={'slice_id': str})
    support = pd.read_csv(support_path, dtype={'slice_id': str})
    expected_scope = 'same_expression_exact_positive_spots_exact_retained_cell_classes_plus_other'
    if len(scores) != 12 or set(scores['method']) != set(METHOD_LABELS) or (not scores['status'].eq('ok').all()) or (not scores['annotation_key'].eq('cell_class').all()) or (not scores['comparison_scope'].eq(expected_scope).all()) or scores.duplicated(['pair_id', 'method']).any() or (len(support) != 6) or (set(support['pair_id']) != set(scores['pair_id'])):
        raise ValueError('cell-class plot inputs do not form the exact 12-row design')
    assertions = ('method_class_sets_exactly_equal', 'rctd_positive_spot_order_exact', 'cell2location_full_spot_order_exact', 'cell2location_zero_rows_zero', 'other_residual_included')
    if any((not support[column].astype(bool).all() for column in assertions)):
        raise ValueError('a support-audit assertion is false')
    sources = {'scores': scores_path, 'support_audit': support_path, 'scoring_provenance': provenance_path, 'validation': validation_path}
    return (scores, support, sources)

def build_plot_table(scores: pd.DataFrame) -> pd.DataFrame:
    pair_order = scores[['pair_id', 'slice_id', 'resolution']].drop_duplicates().sort_values(['slice_id', 'resolution'])['pair_id'].tolist()
    rows = []
    for score in scores.itertuples(index=False):
        for metric, title, description in METRICS:
            rows.append({'pair_id': score.pair_id, 'pair_order': pair_order.index(score.pair_id), 'slice_id': score.slice_id, 'resolution': float(score.resolution), 'method': METHOD_LABELS[score.method], 'metric': metric, 'panel_title': title, 'metric_description': description, 'value': float(getattr(score, metric)), 'annotation_key': 'cell_class', 'n_retained_cell_classes': int(score.n_retained_cell_classes), 'aggregate_method_ranking_authorized': False})
    return pd.DataFrame(rows).sort_values(['metric', 'pair_order', 'method'])

def configure_matplotlib() -> None:
    arial_paths = (Path(sys.prefix) / 'fonts' / 'arial.ttf', Path(sys.prefix) / 'fonts' / 'arialbd.ttf')
    if not all((path.is_file() for path in arial_paths)):
        raise FileNotFoundError(f'Arial Regular and Bold are required for this figure: {arial_paths}')
    for path in arial_paths:
        font_manager.fontManager.addfont(path)
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'font.family': 'Arial', 'font.size': 8, 'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.edgecolor': '#555555', 'axes.linewidth': 0.8, 'xtick.color': '#333333', 'ytick.color': '#333333', 'xtick.labelsize': 6.8, 'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def draw(plot_table: pd.DataFrame, support: pd.DataFrame, output_dir: Path) -> list[Path]:
    configure_matplotlib()
    ordered = support.sort_values(['slice_id', 'resolution']).reset_index(drop=True)
    x = [0.0, 1.0, 1.9, 2.9, 3.8, 4.8]
    separators = ((x[1] + x[2]) / 2, (x[3] + x[4]) / 2)
    group_centers = ((0.5, '007'), (2.4, '050'), (4.3, '100'))
    retained_by_slice = {str(slice_id): int(rows['n_retained_cell_classes'].iloc[0]) for slice_id, rows in ordered.groupby('slice_id', sort=True)}
    fig = plt.figure(figsize=(7.5, 4.3))
    grid = fig.add_gridspec(1, 3, left=0.075, right=0.985, bottom=0.235, top=0.655, wspace=0.38)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    for ax, (metric, title, description) in zip(axes, METRICS, strict=True):
        subset = plot_table[plot_table['metric'] == metric]
        for method in METHOD_ORDER:
            rows = subset[subset['method'] == method].sort_values('pair_order')
            if len(rows) != 6:
                raise ValueError(f'incomplete plotting matrix: {metric}, {method}')
            offset = -0.13 if method == 'RCTD' else 0.13
            for _slice_id, slice_rows in rows.groupby('slice_id', sort=False):
                slice_rows = slice_rows.sort_values('resolution')
                if len(slice_rows) != 2:
                    raise ValueError(f'expected two resolutions per slice: {metric}, {method}, {_slice_id}')
                ax.plot(np.asarray(x)[slice_rows['pair_order'].to_numpy(dtype=int)] + offset, slice_rows['value'].to_numpy(dtype=float), color=COLORS[method], linewidth=0.9, alpha=0.58, zorder=1.5)
            ax.scatter(np.asarray(x) + offset, rows['value'], s=26, color=COLORS[method], marker=MARKERS[method], edgecolor='white', linewidth=0.55, zorder=2.5)
        ax.set_title(title, loc='left', pad=23)
        ax.text(0, 1.02, description, transform=ax.transAxes, ha='left', va='bottom', fontsize=6.2, color='#666666')
        ax.set_xticks(x, [f'{value:.2f}' for value in ordered['resolution']])
        ax.set_xlim(x[0] - 0.42, x[-1] + 0.42)
        values = subset['value'].to_numpy(dtype=float)
        span = float(values.max() - values.min())
        margin = max(span * 0.08, 0.002)
        ax.set_ylim(float(values.min()) - margin, float(values.max()) + margin)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis='y', color='#DDDDDD', linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', width=0.75, length=3)
        for separator in separators:
            ax.axvline(separator, color='#D8D8D8', linewidth=0.65, linestyle=':')
        for center, slice_id in group_centers:
            ax.text(center, -0.22, f'S{slice_id}\nn = {retained_by_slice[slice_id]}', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=6.2, color='#555555', fontweight='bold')
    handles = [Line2D([], [], color=COLORS[method], marker=MARKERS[method], markeredgecolor='white', markeredgewidth=0.55, linestyle='-', linewidth=0.9, markersize=5.5, label=method) for method in METHOD_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=7.5, columnspacing=1.4, handletextpad=0.45, bbox_to_anchor=(0.53, 0.8))
    fig.suptitle('Deconvolution accuracy using cell-class annotations', x=0.53, y=0.97, fontsize=11.5, fontweight='bold')
    fig.text(0.53, 0.91, 'Same slice input, positive spots, and retained cell_class support for both methods (≥50 reference cells)', ha='center', fontsize=7.1, color='#666666')
    fig.text(0.5, 0.035, 'Lines connect ×0.10 and ×0.25 within each slice and method · no line crosses slices', ha='center', fontsize=6.6, color='#666666')
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / 'common_support_diagnostic'
    outputs = []
    for suffix, kwargs in {'.pdf': {'metadata': {'Creator': 'FEAST Study 03 plot_cell_class.py', 'CreationDate': None, 'ModDate': None}}, '.svg': {'metadata': {'Creator': 'FEAST Study 03 plot_cell_class.py', 'Date': '1970-01-01T00:00:00Z'}}, '.png': {'dpi': 600, 'metadata': {'Software': 'FEAST Study 03 plot_cell_class.py'}}}.items():
        path = stem.with_suffix(suffix)
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument('--scores-dir', type=Path, default=DEFAULT_SCORES_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    scores, support, sources = load_inputs(args.run_dir, args.scores_dir)
    plot_table = build_plot_table(scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_table_path = args.output_dir / 'common_support_scores_plot.csv'
    support_table_path = args.output_dir / 'common_support_audit_plot.csv'
    plot_table.to_csv(plot_table_path, index=False, float_format='%.12g')
    support.to_csv(support_table_path, index=False)
    outputs = draw(plot_table, support, args.output_dir)
    provenance = {'schema_version': 1, 'figure_set': 'study03_cell_class_deconvolution_comparison', 'publication_status': 'validated_descriptive_comparison', 'annotation_key': 'cell_class', 'aggregate_method_ranking_authorized': False, 'sources': {**{role: {'path': repository_path(path)} for role, path in sorted(sources.items())}, 'plot_script': {'path': repository_path(Path(__file__))}}, 'design': {'configuration_id': 'study03-cell-class-within-slice-resolution-lines-v2', 'methods': list(METHOD_ORDER), 'metrics': [metric for metric, _title, _description in METRICS], 'unconnected_case_points': False, 'within_slice_resolution_lines': True, 'cross_slice_lines': False, 'method_horizontal_dodge': 0.13, 'vertical_value_jitter': False, 'panel_indices_rendered': False, 'ranking_or_composite_used': False}, 'typography': {'font_family': 'Arial', 'font_source': str(Path(sys.prefix) / 'fonts' / 'arial.ttf')}, 'outputs': {path.name: None for path in [plot_table_path, support_table_path, *outputs]}}
    provenance_path = args.output_dir / 'figure_provenance.json'
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    for path in [plot_table_path, support_table_path, *outputs, provenance_path]:
        print(f'Saved: {path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
