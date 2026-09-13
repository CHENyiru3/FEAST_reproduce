"""Render validated Study 04 supplementary atomic-metric diagnostics."""
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
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / 'PUBLICATION_MANIFEST.json'
OUTPUT_DIR = Path(__file__).resolve().parent / 'figures' / 'same_slice'
METHOD_ORDER = ['GraphST', 'STAMP', 'scVI']
METHOD_COLORS = {'GraphST': '#0072B2', 'STAMP': '#D55E00', 'scVI': '#009E73'}
METHOD_MARKERS = {'GraphST': 'o', 'STAMP': 's', 'scVI': 'D'}
MODE_STYLES = {'shift_only': '-', 'diagonal_affine': '--'}
MODE_LABELS = {'shift_only': 'Shift only', 'diagonal_affine': 'Diagonal affine'}
ALL_METRICS = (('Batch alignment', (('batch_asw', 'Batch separation ASW ↓'), ('batch_mixing_entropy_k30', 'Batch mixing entropy (k=30) ↑'), ('centroid_distance', 'Centroid distance ↓'), ('covariance_distance', 'Covariance distance ↓'), ('mmd2_rbf_biased', 'Distribution MMD² ↓'))), ('Biological signal preservation', (('paired_retrieval_top1', 'Paired retrieval top-1 ↑'), ('paired_retrieval_median_rank', 'Paired retrieval median rank ↓ (log scale)'), ('domain_asw', 'Domain ASW ↑'), ('domain_purity_k30', 'Domain purity (k=30) ↑'), ('paired_expression_correlation', 'Paired expression correlation ↑\n(scVI only)'))))
PRIMARY_METRIC_GROUPS = (('Batch removal', (('batch_asw', 'Batch separation ASW ↓'), ('batch_mixing_entropy_k30', 'Batch mixing entropy (k=30) ↑'), ('mmd2_rbf_biased', 'Distribution MMD² ↓'))), ('Biological signal preservation', (('paired_retrieval_top1', 'Paired retrieval top-1 ↑'), ('domain_asw', 'Domain separation ASW ↑'), ('domain_purity_k30', 'Domain purity (k=30) ↑'))))
METRICS = [metric for _, group in ALL_METRICS for metric in group]
PRIMARY_METRICS = [metric for _, group in PRIMARY_METRIC_GROUPS for metric in group]
SENSITIVITY_METRICS = PRIMARY_METRICS
METRIC_PREFERENCE = {'batch_asw': -1.0, 'batch_mixing_entropy_k30': 1.0, 'mmd2_rbf_biased': -1.0, 'paired_retrieval_top1': 1.0, 'domain_asw': 1.0, 'domain_purity_k30': 1.0}
FONT_SCALE = 1.4
TICK_FONT_SCALE = 2.0
DISPLAY_MAX_ALPHA = 1.0

def repository_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT))

def selected_source(entry: dict[str, object], key: str) -> Path:
    path = REPOSITORY_ROOT / str(entry[key])
    if not path.is_file():
        raise FileNotFoundError(f'Manifest-selected source is missing: {path}')
    return path

def load_inputs() -> tuple[pd.DataFrame, dict[str, object], dict[str, object], dict[str, Path]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    study = manifest['studies']['04_batch_effect_removal']
    candidate = study['canonical_candidate']
    if study['status'] != 'validated_complete_author_review_required':
        raise ValueError('Study 04 is not in the validated author-review state')
    expected_false = ['composite_score_present', 'winner_ranking_authorized', 'figure_promotion_authorized']
    for field in expected_false:
        if candidate[field] is not False:
            raise ValueError(f'Study 04 governance field must be false: {field}')
    if candidate['active_manuscript_claim'] is not None:
        raise ValueError('Study 04 must not have an active manuscript claim')
    sources = {'atomic_metrics': selected_source(candidate, 'atomic_metrics'), 'validation_summary': selected_source(candidate, 'validation_summary'), 'stamp_numerical_retry_policy': selected_source(candidate, 'stamp_numerical_retry_policy')}
    metrics = pd.read_csv(sources['atomic_metrics'])
    validation = json.loads(sources['validation_summary'].read_text())
    required = {'method', 'representation', 'primary_representation', 'mode', 'alpha', 'alpha_stratum', 'support_variant', 'primary_support', 'analysis_role', *(column for column, _ in METRICS)}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f'Atomic metric table lacks columns: {missing}')
    if len(metrics) != 60 or int(candidate['atomic_metric_rows']) != 60:
        raise ValueError('Expected the selected 60-row Study 04 atomic table')
    if int(candidate['cross_batch_edges']) != 0 or int(validation['cross_batch_edges']) != 0:
        raise ValueError('Study 04 within-batch graph contract is not satisfied')
    if validation['active_manuscript_claim'] is not None:
        raise ValueError('Validation unexpectedly binds an active manuscript claim')
    if validation['decision'] != 'validated_fresh_supplementary_candidate_author_review_required':
        raise ValueError('Study 04 validation is not supplementary/author-review only')
    if not any(('PCA-20/PCA-10' in item for item in validation['limitations'])):
        raise ValueError('GraphST PCA sensitivity limitation is missing')
    if set(metrics['method']) != set(METHOD_ORDER):
        raise ValueError('Study 04 method set differs from the manifest selection')
    primary = metrics.loc[metrics['analysis_role'].eq('primary')].copy()
    if len(primary) != 36 or primary.groupby('method').size().to_dict() != {method: 12 for method in METHOD_ORDER}:
        raise ValueError('Expected 12 primary atomic rows per method')
    if not primary['primary_representation'].astype(bool).all():
        raise ValueError('Primary rows include a representation sensitivity')
    if not primary['primary_support'].astype(bool).all():
        raise ValueError('Primary rows include a support sensitivity')
    return (metrics, validation, candidate, sources)

def build_long_table(primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in primary.sort_values(['method', 'mode', 'alpha']).itertuples(index=False):
        for metric, label in METRICS:
            value = getattr(row, metric)
            rows.append({'method': row.method, 'mode': row.mode, 'alpha': float(row.alpha), 'alpha_stratum': row.alpha_stratum, 'metric': metric, 'metric_label': label, 'value': float(value) if pd.notna(value) else np.nan, 'analysis_role': 'primary', 'support_variant': row.support_variant, 'representation': row.representation, 'supplementary_diagnostic_only': True, 'active_manuscript_claim': False, 'winner_ranking_authorized': False, 'composite_score_used': False})
    return pd.DataFrame(rows)

def build_availability(long_table: pd.DataFrame) -> pd.DataFrame:
    return long_table.assign(available=lambda frame: frame['value'].notna()).groupby(['metric', 'metric_label', 'method'], sort=True, as_index=False).agg(n_expected=('available', 'size'), n_available=('available', 'sum'))

def build_graphst_sensitivity(metrics: pd.DataFrame) -> pd.DataFrame:
    graphst = metrics.loc[metrics['method'].eq('GraphST')].copy()
    keys = ['mode', 'alpha', 'alpha_stratum', 'support_variant']
    primary_rep = graphst.loc[graphst['representation'].eq('pca20_randomized_seed42')]
    sensitivity_rep = graphst.loc[graphst['representation'].eq('pca10_randomized_seed42_sensitivity')]
    merged = primary_rep.merge(sensitivity_rep, on=keys, suffixes=('_pca20', '_pca10'), validate='one_to_one')
    if len(merged) != 8:
        raise ValueError('Expected eight displayed GraphST PCA-20/PCA-10 sensitivity rows through alpha 1.0')
    rows: list[dict[str, object]] = []
    for row in merged.sort_values(keys).itertuples(index=False):
        for metric, label in METRICS:
            pca20 = getattr(row, f'{metric}_pca20')
            pca10 = getattr(row, f'{metric}_pca10')
            rows.append({'mode': row.mode, 'alpha': float(row.alpha), 'alpha_stratum': row.alpha_stratum, 'support_variant': row.support_variant, 'metric': metric, 'metric_label': label, 'pca20_value': float(pca20) if pd.notna(pca20) else np.nan, 'pca10_value': float(pca10) if pd.notna(pca10) else np.nan, 'pca10_minus_pca20': float(pca10 - pca20) if pd.notna(pca10) and pd.notna(pca20) else np.nan, 'supplementary_sensitivity_only': True, 'winner_ranking_authorized': False})
    return pd.DataFrame(rows)

def build_graphst_representation_effects(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """Orient and scale matched PCA-10/PCA-20 differences within each metric."""
    rows: list[dict[str, object]] = []
    for metric, label in SENSITIVITY_METRICS:
        subset = sensitivity.loc[sensitivity['metric'].eq(metric)].dropna(subset=['pca20_value', 'pca10_value'])
        if len(subset) != 8:
            raise ValueError(f'Expected eight displayed GraphST rows through alpha 1.0 for {metric}')
        values = pd.concat([subset['pca20_value'], subset['pca10_value']])
        metric_span = float(values.max() - values.min())
        if not np.isfinite(metric_span) or metric_span <= 0:
            raise ValueError(f'GraphST PCA comparison has no usable span for {metric}')
        preference = METRIC_PREFERENCE[metric]
        for row in subset.sort_values(['mode', 'alpha', 'support_variant']).itertuples(index=False):
            raw_delta = float(row.pca10_value - row.pca20_value)
            rows.append({'mode': row.mode, 'alpha': float(row.alpha), 'alpha_stratum': row.alpha_stratum, 'support_variant': row.support_variant, 'metric': metric, 'metric_label': label, 'pca20_value': float(row.pca20_value), 'pca10_value': float(row.pca10_value), 'pca10_minus_pca20': raw_delta, 'metric_preference_direction': preference, 'observed_metric_span': metric_span, 'pca10_advantage_span_units': preference * raw_delta / metric_span, 'supplementary_sensitivity_only': True, 'winner_ranking_authorized': False})
    effects = pd.DataFrame(rows)
    if len(effects) != 48:
        raise ValueError('Expected 48 displayed six-metric GraphST representation-effect rows')
    return effects

def configure_matplotlib() -> None:
    arial_paths = (Path(sys.prefix) / 'fonts' / 'arial.ttf', Path(sys.prefix) / 'fonts' / 'arialbd.ttf')
    if not all((path.is_file() for path in arial_paths)):
        raise FileNotFoundError(f'Arial Regular and Bold are required for this figure: {arial_paths}')
    for path in arial_paths:
        font_manager.fontManager.addfont(path)
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'font.family': 'Arial', 'font.size': 9 * FONT_SCALE, 'xtick.labelsize': 9 * TICK_FONT_SCALE, 'ytick.labelsize': 9 * TICK_FONT_SCALE, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    base = output_dir / stem
    outputs: list[Path] = []
    formats = {'.pdf': {'metadata': {'Creator': 'FEAST reproduction Study 04 plot.py', 'CreationDate': None, 'ModDate': None}}, '.svg': {'metadata': {'Creator': 'FEAST reproduction Study 04 plot.py', 'Date': '1970-01-01T00:00:00Z'}}, '.png': {'dpi': 600, 'metadata': {'Software': 'FEAST reproduction Study 04 plot.py'}}}
    for suffix, kwargs in formats.items():
        path = base.with_suffix(suffix)
        fig.savefig(path, bbox_inches='tight', **kwargs)
        outputs.append(path)
    return outputs

def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis='y', color='#DDDDDD', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#666666')
        spine.set_linewidth(0.75)

def draw_primary_figure(long_table: pd.DataFrame, output_dir: Path) -> list[Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 9.5))
    axes = axes.ravel()
    for ax, (metric, label) in zip(axes, PRIMARY_METRICS):
        subset = long_table.loc[long_table['metric'].eq(metric)]
        for method in METHOD_ORDER:
            for mode, linestyle in MODE_STYLES.items():
                rows = subset.loc[subset['method'].eq(method) & subset['mode'].eq(mode)].sort_values('alpha')
                rows = rows.dropna(subset=['value'])
                if rows.empty:
                    continue
                ax.plot(rows['alpha'], rows['value'], color=METHOD_COLORS[method], linestyle=linestyle, marker=METHOD_MARKERS[method], markersize=4.1, markeredgecolor='white', markeredgewidth=0.45, linewidth=1.55)
        ax.set_title(label, fontsize=11.5 * FONT_SCALE, fontweight='bold', pad=8, loc='center')
        ax.set_xticks([0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis='x', labelrotation=0)
        style_axis(ax)
    method_handles = [Line2D([0], [0], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], markeredgecolor='white', markeredgewidth=0.45, lw=1.8, label=method) for method in METHOD_ORDER]
    mode_handles = [Line2D([0], [0], color='#444444', lw=1.5, ls=style, label=MODE_LABELS[mode]) for mode, style in MODE_STYLES.items()]
    fig.legend(handles=method_handles + mode_handles, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 0.905), frameon=False, fontsize=9 * FONT_SCALE)
    fig.suptitle('Batch-effect removal — atomic diagnostics', fontsize=14 * FONT_SCALE, fontweight='bold', y=0.99)
    fig.text(0.5, 0.948, 'Six complementary metrics; primary representation and all-spot support; no composite score or ranking.', ha='center', fontsize=10 * FONT_SCALE, color='#666666')
    fig.text(0.5, 0.837, PRIMARY_METRIC_GROUPS[0][0], ha='center', fontsize=11.5 * FONT_SCALE, fontweight='bold')
    fig.text(0.5, 0.435, PRIMARY_METRIC_GROUPS[1][0], ha='center', fontsize=11.5 * FONT_SCALE, fontweight='bold')
    fig.text(0.5, 0.006, 'Four additional atomic metrics remain in the companion CSV.', ha='center', fontsize=8 * FONT_SCALE, color='#777777')
    fig.supxlabel('Perturbation strength α', fontsize=11.5 * FONT_SCALE, y=0.035)
    fig.tight_layout(rect=(0.015, 0.065, 0.995, 0.815), w_pad=1.6, h_pad=5.0)
    return save_figure(fig, output_dir, 'atomic_metric_diagnostics')

def draw_sensitivity_figure(effects: pd.DataFrame, output_dir: Path) -> list[Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.6))
    axes = axes.ravel()
    mode_markers = {'shift_only': 'o', 'diagonal_affine': 's'}
    mode_colors = {'shift_only': '#0F4D92', 'diagonal_affine': '#42949E'}
    for ax, (metric, label) in zip(axes, SENSITIVITY_METRICS):
        subset = effects.loc[effects['metric'].eq(metric)]
        for mode, linestyle in MODE_STYLES.items():
            all_spots = subset.loc[subset['mode'].eq(mode) & subset['support_variant'].eq('all_spots_primary')].sort_values('alpha')
            ax.plot(all_spots['alpha'], all_spots['pca10_advantage_span_units'], color=mode_colors[mode], linestyle=linestyle, marker=mode_markers[mode], markersize=4.5, markeredgecolor='white', markeredgewidth=0.5, linewidth=1.7)
        ax.axhline(0.0, color='#4D4D4D', linewidth=0.85, zorder=-1)
        ax.set_ylim(-0.6, 1.0)
        ax.set_yticks([-0.5, 0.0, 0.5, 1.0])
        ax.set_xticks([0.25, 0.5, 0.75, 1.0])
        ax.set_title(label, fontsize=11.5 * FONT_SCALE, fontweight='bold', pad=8, loc='center')
        style_axis(ax)
    legend_handles = [Line2D([0], [0], marker=marker, color=mode_colors[mode], markerfacecolor=mode_colors[mode], markeredgecolor='white', markeredgewidth=0.5, linestyle=MODE_STYLES[mode], linewidth=1.7, label=f'{MODE_LABELS[mode]} (all spots)') for mode, marker in mode_markers.items()]
    fig.legend(handles=legend_handles, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 0.875), frameon=False, fontsize=9 * FONT_SCALE)
    fig.suptitle('GraphST representation sensitivity', fontsize=14 * FONT_SCALE, fontweight='bold', y=0.99)
    fig.text(0.5, 0.925, 'Direction-aware PCA-10 minus PCA-20 effect across eight matched rows; positive values favor PCA-10.', ha='center', fontsize=10 * FONT_SCALE, color='#666666')
    fig.text(0.5, 0.012, 'Each metric is normalized to its own observed PCA-10/PCA-20 span; no cross-metric aggregate is used.', ha='center', fontsize=8 * FONT_SCALE, color='#777777')
    fig.supxlabel('Perturbation strength α', fontsize=11.5 * FONT_SCALE, y=0.04)
    fig.supylabel('PCA-10 advantage over PCA-20 (within-metric span units)', fontsize=11 * FONT_SCALE, x=0.012)
    fig.tight_layout(rect=(0.035, 0.1, 0.995, 0.82), w_pad=1.7, h_pad=2.0)
    return save_figure(fig, output_dir, 'graphst_pca_sensitivity')

def write_provenance(sources: dict[str, Path], validation: dict[str, object], candidate: dict[str, object], long_table: pd.DataFrame, sensitivity: pd.DataFrame, representation_effects: pd.DataFrame, output_paths: list[Path]) -> Path:
    display_limitations = [item for item in validation['limitations'] if 'above 1.00' not in item]
    payload = {'schema_version': 1, 'figure_set': 'study04_supplementary_atomic_diagnostic_candidate', 'publication_status': 'supplementary_unpromoted_author_review_required', 'scientific_disposition': {'active_manuscript_claim': None, 'supplementary_candidate_only': True, 'composite_score_present': False, 'winner_ranking_authorized': False, 'figure_promotion_authorized': False}, 'sources': {'publication_manifest': {'path': repository_path(MANIFEST_PATH)}, **{role: {'path': repository_path(path)} for role, path in sorted(sources.items())}, 'plot_script': {'path': repository_path(Path(__file__))}}, 'design': {'source_atomic_rows': int(candidate['atomic_metric_rows']), 'source_primary_rows': 36, 'displayed_alpha_maximum': DISPLAY_MAX_ALPHA, 'displayed_primary_rows': 24, 'primary_long_rows': int(len(long_table)), 'graphst_pca_sensitivity_pairs': 8, 'graphst_pca_sensitivity_long_rows': int(len(sensitivity)), 'graphst_pca_representation_effect_rows': int(len(representation_effects)), 'methods': METHOD_ORDER, 'available_atomic_metrics': [column for column, _ in METRICS], 'primary_figure_groups': {name: [column for column, _ in group] for name, group in PRIMARY_METRIC_GROUPS}, 'graphst_sensitivity_displayed_metrics': [column for column, _ in SENSITIVITY_METRICS], 'graphst_representation_effect': {'formula': 'preference_direction * (pca10 - pca20) / observed_metric_span', 'positive_interpretation': 'PCA-10 has the preferred metric direction', 'normalization_scope': 'each metric separately over its matched PCA-10/PCA-20 values'}, 'cross_batch_edges': int(candidate['cross_batch_edges']), 'ranking_or_composite_used': False}, 'limitations': display_limitations, 'external_numpy_limitation': candidate['external_numpy_limitation'], 'outputs': {path.name: None for path in sorted(output_paths, key=lambda item: item.name)}}
    path = OUTPUT_DIR / 'figure_provenance.json'
    path.write_text(json.dumps(payload, indent=2) + '\n')
    return path

def main() -> int:
    metrics, validation, candidate, sources = load_inputs()
    primary = metrics.loc[metrics['analysis_role'].eq('primary') & metrics['alpha'].le(DISPLAY_MAX_ALPHA)].copy()
    long_table = build_long_table(primary)
    availability = build_availability(long_table)
    sensitivity = build_graphst_sensitivity(metrics.loc[metrics['alpha'].le(DISPLAY_MAX_ALPHA)].copy())
    representation_effects = build_graphst_representation_effects(sensitivity)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table_paths = [OUTPUT_DIR / 'atomic_metrics_primary_long.csv', OUTPUT_DIR / 'atomic_metric_availability.csv', OUTPUT_DIR / 'graphst_pca_sensitivity_plot.csv', OUTPUT_DIR / 'graphst_pca_representation_effects.csv']
    long_table.to_csv(table_paths[0], index=False, na_rep='NA', float_format='%.12g')
    availability.to_csv(table_paths[1], index=False)
    sensitivity.to_csv(table_paths[2], index=False, na_rep='NA', float_format='%.12g')
    representation_effects.to_csv(table_paths[3], index=False, na_rep='NA', float_format='%.12g')
    figure_paths = draw_primary_figure(long_table, OUTPUT_DIR)
    figure_paths.extend(draw_sensitivity_figure(representation_effects, OUTPUT_DIR))
    write_provenance(sources, validation, candidate, long_table, sensitivity, representation_effects, table_paths + figure_paths)
    for path in table_paths + figure_paths + [OUTPUT_DIR / 'figure_provenance.json']:
        print(f'Saved: {path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
