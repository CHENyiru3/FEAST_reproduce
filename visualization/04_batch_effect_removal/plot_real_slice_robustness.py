"""Render Study 04B real-slice incremental-robustness diagnostics."""
from __future__ import annotations
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
from matplotlib.patches import Patch
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = REPOSITORY_ROOT / '04_batch_effect_removal' / 'real_slice_robustness' / 'outputs' / 'production_v1'
OUTPUT_DIR = Path(__file__).resolve().parent / 'figures' / 'two_slice'
METRICS_PATH = STUDY_ROOT / 'metrics' / 'atomic_metrics.csv'
METRICS_PROVENANCE_PATH = STUDY_ROOT / 'metrics' / 'provenance.json'
METRICS_SUMMARY_PATH = STUDY_ROOT / 'metrics' / 'summary.json'
QUALIFICATION_PATH = STUDY_ROOT / 'inputs' / 'qualification.json'
VALIDATION_PATH = STUDY_ROOT / 'validation' / 'complete_validation.json'
METHOD_ORDER = ['GraphST', 'STAMP', 'scVI']
METHOD_COLORS = {'GraphST': '#0072B2', 'STAMP': '#D55E00', 'scVI': '#009E73', 'Uncorrected': '#777777'}
METHOD_MARKERS = {'GraphST': 'o', 'STAMP': 's', 'scVI': 'D', 'Uncorrected': 'X'}
METHOD_REPRESENTATIONS = {'GraphST': 'pca10_randomized_seed42', 'STAMP': 'native_10d_topics', 'scVI': 'native_10d_latent'}
SOURCE_CONDITION_ORDER = ['raw', 'sim_0.00', 'sim_0.50', 'sim_1.00', 'sim_1.50']
CONDITION_ORDER = ['raw', 'sim_0.00', 'sim_0.50', 'sim_1.00']
CONDITION_X = {'raw': -0.55, 'sim_0.00': 0.0, 'sim_0.50': 0.5, 'sim_1.00': 1.0}
CONDITION_LABELS = ['Raw\nanchor', '0\ncontrol', '0.5', '1.0\nprimary']
SEED_JITTER = {42: -0.022, 43: 0.0, 44: 0.022}
BATCH_METRICS = [('conditional_batch_asw_macro', 'Layer-conditioned batch ASW', '↓'), ('conditional_batch_entropy_macro', 'Layer-conditioned batch entropy', '↑'), ('conditional_mmd2_macro', 'Layer-conditioned MMD²', '↓')]
PRESERVATION_METRICS = [('layer_asw', 'Layer separation ASW', '↑'), ('layer_purity_k', 'Layer purity (k=30)', '↑'), ('reference_to_query_layer_macro_f1', 'Reference→query layer macro-F1', '↑'), ('spatial_embedding_knn_overlap_k', 'Spatial–embedding kNN overlap', '↑'), ('within_section_knn_preservation_k', 'Within-section kNN preservation', '↑'), ('query_stability_knn_jaccard_k', 'Query kNN stability vs α=0', '↑'), ('query_stability_distance_spearman', 'Query distance stability vs α=0', '↑'), ('query_stability_linear_cka', 'Query linear CKA vs α=0', '↑')]
GRAPHST_SENSITIVITY_METRICS = [BATCH_METRICS[0], BATCH_METRICS[1], BATCH_METRICS[2], PRESERVATION_METRICS[1], PRESERVATION_METRICS[2], PRESERVATION_METRICS[-1]]
PLOT_METRICS = BATCH_METRICS + PRESERVATION_METRICS
METRIC_METADATA = {metric: {'metric_label': label, 'preferred_direction': direction} for metric, label, direction in PLOT_METRICS}
DISPLAY_TITLES = {
    'reference_to_query_layer_macro_f1': 'Reference→query layer\nmacro-F1',
    'spatial_embedding_knn_overlap_k': 'Spatial–embedding\nkNN overlap',
    'within_section_knn_preservation_k': 'Within-section kNN\npreservation',
    'query_stability_knn_jaccard_k': 'Query kNN stability\nvs α=0',
    'query_stability_distance_spearman': 'Query distance stability\nvs α=0',
    'query_stability_linear_cka': 'Query linear CKA\nvs α=0',
}
FONT_SCALE = 1.4
TICK_FONT_SCALE = 2.0

def repository_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT))

def configure_matplotlib() -> None:
    arial_paths = (Path(sys.prefix) / 'fonts' / 'arial.ttf', Path(sys.prefix) / 'fonts' / 'arialbd.ttf')
    if not all((path.is_file() for path in arial_paths)):
        raise FileNotFoundError(f'Arial Regular and Bold are required for this figure: {arial_paths}')
    for path in arial_paths:
        font_manager.fontManager.addfont(path)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9 * FONT_SCALE, 'axes.titlesize': 12 * FONT_SCALE, 'axes.labelsize': 10.5 * FONT_SCALE, 'xtick.labelsize': 8 * TICK_FONT_SCALE, 'ytick.labelsize': 8 * TICK_FONT_SCALE, 'legend.fontsize': 8.5 * FONT_SCALE, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def load_validated_inputs() -> tuple[pd.DataFrame, dict, dict]:
    required = [METRICS_PATH, METRICS_PROVENANCE_PATH, METRICS_SUMMARY_PATH, QUALIFICATION_PATH, VALIDATION_PATH]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing Study 04B inputs: {missing}')
    metrics_provenance = json.loads(METRICS_PROVENANCE_PATH.read_text())
    metrics_record = next((record for record in metrics_provenance.get('outputs', []) if Path(record['path']).name == METRICS_PATH.name), None)
    if metrics_record is None:
        raise ValueError('Study 04B atomic metrics are absent from provenance')
    validation = json.loads(VALIDATION_PATH.read_text())
    if validation.get('stage') != 'complete':
        raise ValueError('Study 04B complete validation record is not complete')
    if validation.get('prepared', {}).get('status') != 'passed':
        raise ValueError('Study 04B prepared validation did not pass')
    if validation.get('candidates', {}).get('status') != 'passed':
        raise ValueError('Study 04B candidate validation did not pass')
    if validation.get('candidates', {}).get('complete_candidates') != 45:
        raise ValueError('Study 04B does not contain all 45 candidate jobs')
    if validation.get('scores', {}).get('status') != 'passed':
        raise ValueError('Study 04B independent score reconstruction did not pass')
    if validation.get('scores', {}).get('independently_reconstructed_rows') != 65:
        raise ValueError('Study 04B does not contain all 65 reconstructed rows')
    qualification = json.loads(QUALIFICATION_PATH.read_text())
    if qualification.get('status') != 'passed':
        raise ValueError('Study 04B input qualification did not pass')
    if not qualification.get('added_deformation_exceeds_resampling_each_dimension'):
        raise ValueError('Added deformation did not exceed the resampling control')
    if qualification.get('cross_section_spot_pairing_declared') is not False:
        raise ValueError('Study 04B must not declare cross-section spot pairing')
    summary = json.loads(METRICS_SUMMARY_PATH.read_text())
    expected_summary = {'atomic_rows': 65, 'primary_method_rows': 45, 'uncorrected_rows': 5, 'graphst_representation_sensitivity_rows': 15, 'composite_score': 'prohibited_not_computed', 'winner_ranking': 'prohibited_not_computed', 'cross_section_exact_spot_retrieval': 'not_computed_not_a_batch_removal_endpoint'}
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise ValueError(f'Unexpected Study 04B summary field {field}: {summary.get(field)!r}')
    metrics = pd.read_csv(METRICS_PATH)
    required_columns = {'condition', 'condition_role', 'alpha', 'method', 'method_seed', 'representation', 'analysis_role', *(metric for metric, _, _ in PLOT_METRICS)}
    if (missing_columns := sorted(required_columns - set(metrics.columns))):
        raise ValueError(f'Study 04B metrics are missing columns: {missing_columns}')
    if len(metrics) != 65:
        raise ValueError(f'Expected 65 Study 04B metric rows, observed {len(metrics)}')
    primary = metrics.loc[metrics['analysis_role'].eq('primary')]
    uncorrected = metrics.loc[metrics['analysis_role'].eq('uncorrected_baseline')]
    sensitivity = metrics.loc[metrics['analysis_role'].eq('representation_sensitivity')]
    if len(primary) != 45 or len(uncorrected) != 5 or len(sensitivity) != 15:
        raise ValueError('Study 04B analysis-role row counts changed')
    if set(primary['method']) != set(METHOD_ORDER):
        raise ValueError('Study 04B primary method set changed')
    if set(primary['method_seed'].astype(int)) != set(SEED_JITTER):
        raise ValueError('Study 04B primary method seeds changed')
    if set(primary['condition']) != set(SOURCE_CONDITION_ORDER):
        raise ValueError('Study 04B primary condition set changed')
    for method, representation in METHOD_REPRESENTATIONS.items():
        observed = set(primary.loc[primary['method'].eq(method), 'representation'])
        if observed != {representation}:
            raise ValueError(f'Unexpected primary representation for {method}: {observed}')
    if primary[[metric for metric, _, _ in PLOT_METRICS]].isna().any().any():
        raise ValueError('Study 04B primary rows contain missing plotted metrics')
    return (metrics, validation, qualification)

def build_plot_data(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = metrics.loc[metrics['analysis_role'].isin(['primary', 'uncorrected_baseline']) & metrics['condition'].isin(CONDITION_ORDER)].copy()
    records: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        for metric, label, direction in PLOT_METRICS:
            records.append({'condition': row.condition, 'condition_role': row.condition_role, 'alpha': row.alpha, 'x_position': CONDITION_X[row.condition], 'method': row.method, 'method_seed': row.method_seed, 'representation': row.representation, 'analysis_role': row.analysis_role, 'metric': metric, 'metric_label': label, 'preferred_direction': direction, 'value': float(getattr(row, metric)), 'raw_anchor_connected_to_simulation_ladder': False, 'cross_section_exact_spot_pairing_used': False, 'composite_score_used': False, 'winner_ranking_authorized': False})
    long_data = pd.DataFrame(records).sort_values(['metric', 'method', 'x_position', 'method_seed'], na_position='last')
    seed_summary = long_data.groupby(['condition', 'condition_role', 'alpha', 'x_position', 'method', 'representation', 'analysis_role', 'metric', 'metric_label', 'preferred_direction'], dropna=False, sort=False)['value'].agg(seed_mean='mean', seed_sd='std', seed_min='min', seed_max='max', seed_n='size').reset_index().sort_values(['metric', 'method', 'x_position'])
    return (long_data, seed_summary)

def style_axis(ax: plt.Axes, metric: str, values: pd.Series) -> None:
    ax.grid(axis='y', color='#DDDDDD', linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.axvline(-0.275, color='#AAAAAA', linewidth=0.8, linestyle=':', zorder=1)
    ax.set_xlim(-0.72, 1.12)
    ax.set_xticks([CONDITION_X[condition] for condition in CONDITION_ORDER])
    ax.set_xticklabels(CONDITION_LABELS)
    ax.set_xlabel('Added FEAST perturbation α')
    for spine in ax.spines.values():
        spine.set_color('#555555')
        spine.set_linewidth(0.8)
    numeric = pd.to_numeric(values, errors='coerce').dropna()
    if numeric.empty:
        return
    lower = float(numeric.min())
    upper = float(numeric.max())
    span = max(upper - lower, abs(upper) * 0.08, 0.02)
    if metric in {'conditional_batch_asw_macro', 'conditional_mmd2_macro'}:
        ax.set_ylim(0.0, upper + 0.14 * span)
    elif metric in {'conditional_batch_entropy_macro', 'query_stability_linear_cka'}:
        ax.set_ylim(max(0.0, lower - 0.12 * span), min(1.03, upper + 0.1 * span))
    elif metric == 'within_section_knn_preservation_k':
        ax.set_ylim(0.0, 1.04)
    else:
        ax.set_ylim(max(0.0, lower - 0.13 * span), min(1.03, upper + 0.13 * span))

def draw_method_metric(ax: plt.Axes, metric: str, long_data: pd.DataFrame, seed_summary: pd.DataFrame) -> None:
    for method in METHOD_ORDER:
        summary = seed_summary.loc[seed_summary['metric'].eq(metric) & seed_summary['method'].eq(method)].sort_values('x_position')
        ladder = summary.loc[summary['condition'].ne('raw')]
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        ax.fill_between(ladder['x_position'].to_numpy(), ladder['seed_min'].to_numpy(), ladder['seed_max'].to_numpy(), color=color, alpha=0.12, linewidth=0, zorder=1)
        ax.plot(ladder['x_position'], ladder['seed_mean'], color=color, marker=marker, markersize=5.2, markeredgecolor='white', markeredgewidth=0.55, linewidth=1.7, zorder=3)
        raw = summary.loc[summary['condition'].eq('raw')]
        ax.scatter(raw['x_position'], raw['seed_mean'], color=color, marker=marker, s=35, edgecolor='white', linewidth=0.55, zorder=4)
        seeds = long_data.loc[long_data['metric'].eq(metric) & long_data['method'].eq(method) & long_data['analysis_role'].eq('primary')].copy()
        seed_x = seeds['x_position'].to_numpy() + seeds['method_seed'].map(SEED_JITTER).to_numpy()
        ax.scatter(seed_x, seeds['value'], s=10, marker=marker, facecolor=color, edgecolor='none', alpha=0.45, zorder=2)
    uncorrected = seed_summary.loc[seed_summary['metric'].eq(metric) & seed_summary['method'].eq('Uncorrected')].sort_values('x_position')
    uncorrected_ladder = uncorrected.loc[uncorrected['condition'].ne('raw')]
    ax.plot(uncorrected_ladder['x_position'], uncorrected_ladder['seed_mean'], color=METHOD_COLORS['Uncorrected'], marker=METHOD_MARKERS['Uncorrected'], markersize=5.2, linewidth=1.35, linestyle='--', zorder=3)
    raw_uncorrected = uncorrected.loc[uncorrected['condition'].eq('raw')]
    ax.scatter(raw_uncorrected['x_position'], raw_uncorrected['seed_mean'], color=METHOD_COLORS['Uncorrected'], marker=METHOD_MARKERS['Uncorrected'], s=32, zorder=4)

def legend_handles() -> list[Line2D | Patch]:
    handles: list[Line2D | Patch] = [Line2D([0], [0], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], markeredgecolor='white', markeredgewidth=0.5, linewidth=1.7, label=method) for method in METHOD_ORDER]
    handles.append(Line2D([0], [0], color=METHOD_COLORS['Uncorrected'], marker=METHOD_MARKERS['Uncorrected'], linestyle='--', linewidth=1.35, label='Uncorrected input'))
    return handles

def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    outputs: list[Path] = []
    formats = {'.pdf': {'metadata': {'Creator': 'FEAST Study 04B plot_real_slice_robustness.py', 'CreationDate': None, 'ModDate': None}}, '.svg': {'metadata': {'Creator': 'FEAST Study 04B plot_real_slice_robustness.py', 'Date': '1970-01-01T00:00:00Z'}}, '.png': {'dpi': 600, 'metadata': {'Software': 'FEAST Study 04B plot_real_slice_robustness.py'}}}
    for suffix, kwargs in formats.items():
        path = OUTPUT_DIR / f'{stem}{suffix}'
        fig.savefig(path, bbox_inches='tight', **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs

def draw_batch_metrics(long_data: pd.DataFrame, seed_summary: pd.DataFrame) -> list[Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.25))
    for ax, (metric, label, direction) in zip(axes, BATCH_METRICS, strict=True):
        draw_method_metric(ax, metric, long_data, seed_summary)
        display_label = DISPLAY_TITLES.get(metric, label)
        ax.set_title(f'{display_label} {direction}', fontweight='bold', loc='center', pad=8)
        style_axis(ax, metric, long_data.loc[long_data['metric'].eq(metric), 'value'])
    fig.legend(handles=legend_handles(), loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False, columnspacing=1.4, handletextpad=0.45)
    fig.text(0.5, 0.012, 'Raw is an isolated natural-data anchor; connected lines begin at the α=0 resampling control. Small points are seeds 42–44; bands show their min–max range.', ha='center', fontsize=8 * FONT_SCALE, color='#777777')
    fig.tight_layout(rect=(0.01, 0.09, 0.995, 0.9), w_pad=1.8)
    return save_figure(fig, 'real_slice_batch_removal_metrics')

def draw_preservation_metrics(long_data: pd.DataFrame, seed_summary: pd.DataFrame) -> list[Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 4, figsize=(15.6, 7.3))
    for index, (ax, (metric, label, direction)) in enumerate(zip(axes.flat, PRESERVATION_METRICS, strict=True)):
        draw_method_metric(ax, metric, long_data, seed_summary)
        display_label = DISPLAY_TITLES.get(metric, label)
        ax.set_title(f'{display_label} {direction}', fontweight='bold', loc='center', pad=8)
        style_axis(ax, metric, long_data.loc[long_data['metric'].eq(metric), 'value'])
        if index < 4:
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)
    fig.legend(handles=legend_handles(), loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False, columnspacing=1.4, handletextpad=0.45)
    fig.text(0.5, 0.01, 'Raw is isolated from the perturbation ladder. Query stability is relative to α=0; uncorrected within-section kNN preservation equals 1 by construction, not by comparative performance.', ha='center', fontsize=8 * FONT_SCALE, color='#777777')
    fig.tight_layout(rect=(0.01, 0.07, 0.995, 0.91), h_pad=1.45, w_pad=1.45)
    return save_figure(fig, 'real_slice_biological_preservation')

def build_graphst_sensitivity_data(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = metrics.loc[metrics['method'].eq('GraphST') & metrics['analysis_role'].isin(['primary', 'representation_sensitivity']) & metrics['condition'].isin(CONDITION_ORDER)].copy()
    rows['representation_label'] = np.where(rows['analysis_role'].eq('primary'), 'PCA-10 (primary)', 'PCA-20 (sensitivity)')
    records: list[dict[str, object]] = []
    for row in rows.itertuples(index=False):
        for metric, label, direction in PLOT_METRICS:
            records.append({'condition': row.condition, 'condition_role': row.condition_role, 'alpha': row.alpha, 'x_position': CONDITION_X[row.condition], 'method_seed': int(row.method_seed), 'representation': row.representation, 'representation_label': row.representation_label, 'analysis_role': row.analysis_role, 'metric': metric, 'metric_label': label, 'preferred_direction': direction, 'value': float(getattr(row, metric)), 'supplementary_sensitivity_only': True, 'winner_ranking_authorized': False})
    data = pd.DataFrame(records).sort_values(['metric', 'representation_label', 'x_position', 'method_seed'])
    expected_rows = 2 * len(SEED_JITTER) * len(CONDITION_ORDER) * len(PLOT_METRICS)
    if len(data) != expected_rows:
        raise ValueError(f'Expected {expected_rows} GraphST sensitivity values, observed {len(data)}')
    return data

def draw_graphst_sensitivity(data: pd.DataFrame) -> list[Path]:
    configure_matplotlib()
    rep_styles = {'PCA-10 (primary)': {'color': '#0072B2', 'marker': 'o', 'linestyle': '-'}, 'PCA-20 (sensitivity)': {'color': '#CC79A7', 'marker': '^', 'linestyle': '--'}}
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.1))
    for index, (ax, (metric, label, direction)) in enumerate(zip(axes.flat, GRAPHST_SENSITIVITY_METRICS, strict=True)):
        metric_data = data.loc[data['metric'].eq(metric)]
        for representation_label, style in rep_styles.items():
            subset = metric_data.loc[metric_data['representation_label'].eq(representation_label)]
            summary = subset.groupby(['condition', 'x_position'], sort=False)['value'].agg(seed_mean='mean', seed_min='min', seed_max='max').reset_index().sort_values('x_position')
            ladder = summary.loc[summary['condition'].ne('raw')]
            ax.fill_between(ladder['x_position'].to_numpy(), ladder['seed_min'].to_numpy(), ladder['seed_max'].to_numpy(), color=style['color'], alpha=0.12, linewidth=0)
            ax.plot(ladder['x_position'], ladder['seed_mean'], color=style['color'], marker=style['marker'], linestyle=style['linestyle'], linewidth=1.7, markersize=5.1, markeredgecolor='white', markeredgewidth=0.5, zorder=3)
            raw = summary.loc[summary['condition'].eq('raw')]
            ax.scatter(raw['x_position'], raw['seed_mean'], color=style['color'], marker=style['marker'], s=35, edgecolor='white', linewidth=0.5, zorder=4)
            seed_x = subset['x_position'].to_numpy() + subset['method_seed'].map(SEED_JITTER).to_numpy()
            ax.scatter(seed_x, subset['value'], color=style['color'], marker=style['marker'], s=10, alpha=0.45, edgecolor='none', zorder=2)
        ax.set_title(f'{label} {direction}', fontweight='bold', loc='center', pad=8)
        style_axis(ax, metric, metric_data['value'])
        if index < 3:
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)
    handles = [Line2D([0], [0], color=style['color'], marker=style['marker'], linestyle=style['linestyle'], linewidth=1.7, markeredgecolor='white', markeredgewidth=0.5, label=label) for label, style in rep_styles.items()]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False, columnspacing=1.5, handletextpad=0.45)
    fig.text(0.5, 0.011, 'GraphST representation sensitivity only; small points are seeds 42–44 and bands show min–max. It is not a cross-method ranking.', ha='center', fontsize=8 * FONT_SCALE, color='#777777')
    fig.tight_layout(rect=(0.01, 0.07, 0.995, 0.91), h_pad=1.45, w_pad=1.5)
    return save_figure(fig, 'real_slice_graphst_pca_sensitivity')

def write_provenance(output_paths: list[Path], validation: dict, qualification: dict) -> Path:
    provenance = {'schema_version': 1, 'configuration_id': 'study04b-real-slice-robustness-visualization-v2-alpha1-endpoint', 'inputs': [{'path': repository_path(path)} for path in [METRICS_PATH, METRICS_PROVENANCE_PATH, METRICS_SUMMARY_PATH, QUALIFICATION_PATH, VALIDATION_PATH]], 'outputs': [{'path': repository_path(path)} for path in sorted(output_paths)], 'source': {'path': repository_path(Path(__file__))}, 'validation': {'prepared_status': validation['prepared']['status'], 'complete_candidates': validation['candidates']['complete_candidates'], 'independently_reconstructed_rows': validation['scores']['independently_reconstructed_rows'], 'metric_columns_reconstructed': validation['scores']['metric_columns_reconstructed'], 'maximum_absolute_error': validation['scores']['maximum_absolute_error'], 'input_qualification_status': qualification['status'], 'identity_geometry_labels_exact': qualification['identity_geometry_labels_exact'], 'added_deformation_exceeds_resampling_each_dimension': qualification['added_deformation_exceeds_resampling_each_dimension'], 'sim1_layer_pseudobulk_correlation_to_raw_query': qualification['sim1_layer_pseudobulk_correlation_to_raw_query']}, 'figure_contract': {'study04a_role': 'controlled same-slice paired recovery; rendered separately', 'study04b_role': 'real-slice incremental robustness above natural section differences', 'raw_anchor_connected_to_simulation_ladder': False, 'displayed_alpha_maximum': 1.0, 'alpha_1_role': 'primary endpoint', 'method_seed_summary': 'mean with individual seeds and min-max band', 'uncorrected_replication': 'one deterministic fixed-panel PCA baseline', 'query_stability_reference': 'sim_0.00', 'uncorrected_within_section_knn_preservation': 'equals one by construction', 'cross_section_exact_spot_retrieval': 'prohibited_not_computed', 'composite_score': 'prohibited_not_computed', 'winner_ranking': 'prohibited_not_computed', 'figure_promotion_authorized': False}, 'claim_boundary': 'In one within-donor pair of real DLPFC sections, GraphST, STAMP, and scVI showed distinct robustness and biological-preservation trade-offs as a controlled FEAST marginal perturbation was added to one section.', 'not_established': ['batch-free ground truth', 'removal of all natural section differences', 'cross-section spot correspondence', 'generalization beyond this section pair', 'a universally best method']}
    path = OUTPUT_DIR / 'real_slice_figure_provenance.json'
    path.write_text(json.dumps(provenance, indent=2) + '\n')
    return path

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics, validation, qualification = load_validated_inputs()
    long_data, seed_summary = build_plot_data(metrics)
    graphst_sensitivity = build_graphst_sensitivity_data(metrics)
    long_path = OUTPUT_DIR / 'real_slice_atomic_metric_plot_data.csv'
    summary_path = OUTPUT_DIR / 'real_slice_seed_summary_plot_data.csv'
    sensitivity_path = OUTPUT_DIR / 'real_slice_graphst_pca_sensitivity_plot_data.csv'
    long_data.to_csv(long_path, index=False)
    seed_summary.to_csv(summary_path, index=False)
    graphst_sensitivity.to_csv(sensitivity_path, index=False)
    outputs: list[Path] = [long_path, summary_path, sensitivity_path]
    outputs.extend(draw_batch_metrics(long_data, seed_summary))
    outputs.extend(draw_preservation_metrics(long_data, seed_summary))
    outputs.extend(draw_graphst_sensitivity(graphst_sensitivity))
    provenance_path = write_provenance(outputs, validation, qualification)
    print(f'Validated Study 04B rows: {len(metrics)}')
    print(f'Plot-ready atomic values: {len(long_data)}')
    print(f'GraphST sensitivity values: {len(graphst_sensitivity)}')
    for path in outputs + [provenance_path]:
        print(path)
if __name__ == '__main__':
    main()
