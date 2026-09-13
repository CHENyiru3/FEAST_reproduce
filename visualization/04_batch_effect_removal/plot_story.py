"""Render the integrated Study 04 removal–preservation story figure."""
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
from plot import load_inputs as load_study04a_inputs
from plot_real_slice_robustness import build_plot_data as build_study04b_plot_data, load_validated_inputs as load_study04b_inputs
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STUDY04B_ROOT = REPOSITORY_ROOT / '04_batch_effect_removal' / 'real_slice_robustness' / 'outputs' / 'production_v1'
DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'figures' / 'supplementary'
METHOD_ORDER = ['GraphST', 'STAMP', 'scVI']
METHOD_COLORS = {'GraphST': '#0072B2', 'STAMP': '#D55E00', 'scVI': '#009E73', 'Uncorrected': '#777777'}
METHOD_MARKERS = {'GraphST': 'o', 'STAMP': 's', 'scVI': 'D', 'Uncorrected': 'X'}
MODE_ORDER = ['shift_only', 'diagonal_affine']
MODE_STYLES = {'shift_only': '-', 'diagonal_affine': (0, (3.2, 1.8))}
MODE_LABELS = {'shift_only': 'Shift only', 'diagonal_affine': 'Diagonal affine'}
CONTROLLED_ALPHAS = [0.25, 0.5, 0.75, 1.0]
REAL_CONDITIONS = ['raw', 'sim_0.00', 'sim_0.50', 'sim_1.00']
CONDITION_X = {'raw': -0.55, 'sim_0.00': 0.0, 'sim_0.50': 0.5, 'sim_1.00': 1.0}
CONDITION_LABELS = ['Raw', '0', '0.5', '1.0']
SEED_JITTER = {42: -0.022, 43: 0.0, 44: 0.022}
BATCH_METRICS = [('conditional_batch_asw_macro', 'Batch ASW  ↓', 'Layer-conditioned ASW'), ('conditional_batch_entropy_macro', 'Batch entropy  ↑', 'Layer-conditioned entropy'), ('conditional_mmd2_macro', 'Batch MMD²  ↓', 'Layer-conditioned MMD²')]
PRESERVATION_METRICS = [('reference_to_query_layer_macro_f1', 'Layer macro-F1  ↑', 'Layer macro-F1'), ('spatial_embedding_knn_overlap_k', 'Spatial overlap  ↑', 'Spatial–embedding overlap'), ('query_stability_linear_cka', 'Linear CKA  ↑', 'Linear CKA vs α=0')]
SUPPLEMENTARY_BATCH_METRICS = BATCH_METRICS[1:]
SUPPLEMENTARY_PRESERVATION_METRICS = PRESERVATION_METRICS[:2]
SUPPLEMENTARY_REAL_METRICS = SUPPLEMENTARY_BATCH_METRICS + SUPPLEMENTARY_PRESERVATION_METRICS
STUDY04B_INPUTS = {'atomic_metrics': STUDY04B_ROOT / 'metrics' / 'atomic_metrics.csv', 'metrics_provenance': STUDY04B_ROOT / 'metrics' / 'provenance.json', 'metrics_summary': STUDY04B_ROOT / 'metrics' / 'summary.json', 'input_qualification': STUDY04B_ROOT / 'inputs' / 'qualification.json', 'complete_validation': STUDY04B_ROOT / 'validation' / 'complete_validation.json'}

def repository_path(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()

def configure_matplotlib() -> None:
    arial_paths = (Path(sys.prefix) / 'fonts' / 'arial.ttf', Path(sys.prefix) / 'fonts' / 'arialbd.ttf')
    if not all((path.is_file() for path in arial_paths)):
        raise FileNotFoundError(f'Arial Regular and Bold are required for this figure: {arial_paths}')
    for path in arial_paths:
        font_manager.fontManager.addfont(path)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8.0, 'axes.titlesize': 11.0, 'axes.titleweight': 'bold', 'axes.labelsize': 10.0, 'axes.labelcolor': '#222222', 'axes.edgecolor': '#555555', 'axes.linewidth': 0.8, 'xtick.labelsize': 7.0, 'ytick.labelsize': 7.0, 'xtick.color': '#333333', 'ytick.color': '#333333', 'legend.fontsize': 7.0, 'legend.frameon': False, 'text.color': '#222222', 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def style_axis(axis: plt.Axes, grid_axis: str='both') -> None:
    axis.grid(axis=grid_axis, color='#E2E2E2', linewidth=0.5, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    for key in ('left', 'bottom'):
        axis.spines[key].set_color('#555555')
        axis.spines[key].set_linewidth(0.8)
    axis.tick_params(length=3, width=0.7, color='#555555')

def validate_and_select_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    study04a_metrics, _, _, study04a_sources = load_study04a_inputs()
    for name, path in STUDY04B_INPUTS.items():
        observed = None
    study04b_metrics, _, _ = load_study04b_inputs()
    controlled = study04a_metrics.loc[study04a_metrics['analysis_role'].eq('primary') & study04a_metrics['alpha'].isin(CONTROLLED_ALPHAS), ['method', 'mode', 'alpha', 'alpha_stratum', 'representation', 'support_variant', 'batch_mixing_entropy_k30', 'paired_retrieval_top1', 'domain_purity_k30']].copy()
    expected_controlled = {(method, mode, alpha) for method in METHOD_ORDER for mode in MODE_ORDER for alpha in CONTROLLED_ALPHAS}
    observed_controlled = set(zip(controlled['method'], controlled['mode'], controlled['alpha'], strict=True))
    if len(controlled) != 24 or observed_controlled != expected_controlled:
        raise ValueError('Study 04A controlled α≤1 matrix changed')
    real_long, real_summary = build_study04b_plot_data(study04b_metrics)
    real_metric_names = [metric for metric, _, _ in SUPPLEMENTARY_REAL_METRICS]
    real_long = real_long.loc[real_long['condition'].isin(REAL_CONDITIONS) & real_long['metric'].isin(real_metric_names)].copy()
    real_summary = real_summary.loc[real_summary['condition'].isin(REAL_CONDITIONS) & real_summary['metric'].isin(real_metric_names)].copy()
    if len(real_long) != 160 or len(real_summary) != 64:
        raise ValueError(f'Study 04B integrated-story plot matrix changed: long={len(real_long)}, summary={len(real_summary)}')
    source_paths = {'study04a_atomic_metrics': study04a_sources['atomic_metrics'], 'study04a_validation_summary': study04a_sources['validation_summary'], 'study04a_stamp_retry_policy': study04a_sources['stamp_numerical_retry_policy'], **{f'study04b_{name}': path for name, path in STUDY04B_INPUTS.items()}}
    return (controlled, real_long, real_summary, source_paths)

def add_margin(axis: plt.Axes, x: pd.Series, y: pd.Series) -> None:
    x_min, x_max = (float(x.min()), float(x.max()))
    y_min, y_max = (float(y.min()), float(y.max()))
    x_span = max(x_max - x_min, 0.05)
    y_span = max(y_max - y_min, 0.05)
    axis.set_xlim(x_min - 0.08 * x_span, x_max + 0.08 * x_span)
    axis.set_ylim(max(0.0, y_min - 0.1 * y_span), min(1.02, y_max + 0.1 * y_span))

def draw_controlled_tradeoff(axis: plt.Axes, controlled: pd.DataFrame, y_metric: str, title: str) -> None:
    for method in METHOD_ORDER:
        for mode in MODE_ORDER:
            frame = controlled.loc[controlled['method'].eq(method) & controlled['mode'].eq(mode)].sort_values('alpha')
            axis.plot(frame['batch_mixing_entropy_k30'], frame[y_metric], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], markersize=4.0, markeredgecolor='white', markeredgewidth=0.45, linewidth=1.25, linestyle=MODE_STYLES[mode], alpha=0.96, zorder=3)
            endpoint = frame.loc[np.isclose(frame['alpha'], 1.0)]
            axis.scatter(endpoint['batch_mixing_entropy_k30'], endpoint[y_metric], s=34, marker=METHOD_MARKERS[method], facecolor=METHOD_COLORS[method], edgecolor='#222222', linewidth=0.75, zorder=5)
    axis.set_title(title, loc='center', pad=7)
    add_margin(axis, controlled['batch_mixing_entropy_k30'], controlled[y_metric])
    style_axis(axis)

def real_metric_limits(axis: plt.Axes, metric: str, values: pd.Series) -> None:
    numeric = pd.to_numeric(values, errors='coerce').dropna()
    lower, upper = (float(numeric.min()), float(numeric.max()))
    span = max(upper - lower, abs(upper) * 0.08, 0.02)
    if metric in {'conditional_batch_asw_macro', 'conditional_mmd2_macro', 'spatial_embedding_knn_overlap_k'}:
        axis.set_ylim(0.0, upper + 0.14 * span)
    elif metric in {'conditional_batch_entropy_macro', 'query_stability_linear_cka'}:
        axis.set_ylim(max(0.0, lower - 0.12 * span), min(1.03, upper + 0.1 * span))
    else:
        axis.set_ylim(max(0.0, lower - 0.13 * span), min(1.03, upper + 0.13 * span))

def draw_real_metric(axis: plt.Axes, metric: str, title: str, real_long: pd.DataFrame, real_summary: pd.DataFrame) -> None:
    for method in METHOD_ORDER:
        summary = real_summary.loc[real_summary['metric'].eq(metric) & real_summary['method'].eq(method)].sort_values('x_position')
        ladder = summary.loc[summary['condition'].ne('raw')]
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        axis.fill_between(ladder['x_position'].to_numpy(), ladder['seed_min'].to_numpy(), ladder['seed_max'].to_numpy(), color=color, alpha=0.11, linewidth=0, zorder=1)
        axis.plot(ladder['x_position'], ladder['seed_mean'], color=color, marker=marker, markersize=4.3, markeredgecolor='white', markeredgewidth=0.45, linewidth=1.35, zorder=3)
        raw = summary.loc[summary['condition'].eq('raw')]
        axis.scatter(raw['x_position'], raw['seed_mean'], color=color, marker=marker, s=27, edgecolor='white', linewidth=0.45, zorder=4)
        seeds = real_long.loc[real_long['metric'].eq(metric) & real_long['method'].eq(method) & real_long['analysis_role'].eq('primary')]
        seed_x = seeds['x_position'].to_numpy() + seeds['method_seed'].map(SEED_JITTER).to_numpy()
        axis.scatter(seed_x, seeds['value'], s=8, marker=marker, facecolor=color, edgecolor='none', alpha=0.43, zorder=2)
    axis.axvline(-0.275, color='#AAAAAA', linewidth=0.7, linestyle=':', zorder=1)
    axis.axvspan(0.94, 1.06, color='#F0F0F0', alpha=0.75, zorder=0)
    axis.set_xlim(-0.72, 1.12)
    axis.set_xticks([CONDITION_X[condition] for condition in REAL_CONDITIONS])
    axis.set_xticklabels(CONDITION_LABELS)
    axis.set_title(title, loc='center', pad=7)
    visible_values = real_long.loc[real_long['metric'].eq(metric) & real_long['method'].isin(METHOD_ORDER), 'value']
    real_metric_limits(axis, metric, visible_values)
    style_axis(axis, grid_axis='y')

def legend_handles() -> list[Line2D]:
    method_handles = [Line2D([], [], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], markeredgecolor='white', markeredgewidth=0.45, linewidth=1.35, label=method) for method in METHOD_ORDER]
    method_handles.extend(Line2D([], [], color='#555555', linestyle=MODE_STYLES[mode], linewidth=1.2, label=f'04A {MODE_LABELS[mode].lower()}') for mode in MODE_ORDER)
    return method_handles

def save_figure(figure: plt.Figure, stem: Path, dpi: int) -> dict[str, Path]:
    outputs = {'pdf': stem.with_suffix('.pdf'), 'svg': stem.with_suffix('.svg'), 'png': stem.with_suffix('.png')}
    figure.savefig(outputs['pdf'], metadata={'Title': 'FEAST Study 04 removal-preservation story', 'Creator': 'FEAST Study 04 plot_story.py', 'CreationDate': None, 'ModDate': None})
    figure.savefig(outputs['svg'], metadata={'Title': 'FEAST Study 04 removal-preservation story', 'Creator': 'FEAST Study 04 plot_story.py', 'Date': None})
    figure.savefig(outputs['png'], dpi=dpi, metadata={'Software': 'FEAST Study 04 plot_story.py'})
    plt.close(figure)
    return outputs

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = output / 'batch_removal_preservation_story'
    controlled_path = output / 'batch_removal_story_controlled.csv'
    real_long_path = output / 'batch_removal_story_real_slice_seed_values.csv'
    real_summary_path = output / 'batch_removal_story_real_slice_summary.csv'
    provenance_path = output / 'batch_removal_story_provenance.json'
    targets = [stem.with_suffix('.pdf'), stem.with_suffix('.svg'), stem.with_suffix('.png'), controlled_path, real_long_path, real_summary_path, provenance_path]
    if any((path.exists() for path in targets)):
        raise FileExistsError('fresh-only Study 04 story output already exists')
    controlled, real_long, real_summary, source_paths = validate_and_select_inputs()
    controlled.to_csv(controlled_path, index=False)
    real_long.to_csv(real_long_path, index=False)
    real_summary.to_csv(real_summary_path, index=False)
    configure_matplotlib()
    figure = plt.figure(figsize=(7.5, 8.7))
    outer = figure.add_gridspec(11, 1, height_ratios=[0.18, 0.18, 1.18, 0.15, 0.18, 0.16, 0.98, 0.16, 0.98, 0.15, 0.10], left=0.065, right=0.985, bottom=0.045, top=0.985, hspace=0.44)
    legend_axis = figure.add_subplot(outer[0])
    controlled_header = figure.add_subplot(outer[1])
    controlled_xlabel = figure.add_subplot(outer[3])
    real_header = figure.add_subplot(outer[4])
    batch_header = figure.add_subplot(outer[5])
    preservation_header = figure.add_subplot(outer[7])
    real_xlabel = figure.add_subplot(outer[9])
    caption_note = figure.add_subplot(outer[10])
    for axis in [legend_axis, controlled_header, controlled_xlabel, real_header, batch_header, preservation_header, real_xlabel, caption_note]:
        axis.set_axis_off()
    top_grid = outer[2].subgridspec(1, 2, wspace=0.27)
    middle_grid = outer[6].subgridspec(1, 2, wspace=0.27)
    bottom_grid = outer[8].subgridspec(1, 2, wspace=0.27)
    top_axes = [figure.add_subplot(top_grid[0, index]) for index in range(2)]
    middle_axes = [figure.add_subplot(middle_grid[0, index]) for index in range(2)]
    bottom_axes = [figure.add_subplot(bottom_grid[0, index]) for index in range(2)]
    legend_axis.legend(handles=legend_handles(), loc='center', ncol=5, columnspacing=1.15, handletextpad=0.35, borderaxespad=0.0)
    controlled_header.text(0.5, 0.5, 'Controlled same-slice trade-offs', ha='center', va='center', fontsize=11.5, fontweight='bold')
    controlled_xlabel.text(0.5, 0.5, 'Batch mixing entropy (k=30)  ↑', ha='center', va='center', fontsize=10.0)
    real_header.text(0.5, 0.5, 'Real two-slice robustness', ha='center', va='center', fontsize=11.5, fontweight='bold')
    batch_header.text(0.5, 0.5, 'Batch removal', ha='center', va='center', fontsize=10.5, fontweight='bold')
    preservation_header.text(0.5, 0.5, 'Biological preservation', ha='center', va='center', fontsize=10.5, fontweight='bold')
    real_xlabel.text(0.5, 0.5, 'Added FEAST perturbation α', ha='center', va='center', fontsize=10.0)
    caption_note.text(0.5, 0.5, 'Raw is a separate natural-data anchor; lines begin at the α=0 resampling control.', ha='center', va='center', fontsize=7.0, color='#666666')
    draw_controlled_tradeoff(top_axes[0], controlled, 'paired_retrieval_top1', 'A  Paired retrieval top-1  ↑')
    draw_controlled_tradeoff(top_axes[1], controlled, 'domain_purity_k30', 'B  Domain purity (k=30)  ↑')
    for letter, axis, (metric, title, _label) in zip('CD', middle_axes, SUPPLEMENTARY_BATCH_METRICS, strict=True):
        draw_real_metric(axis, metric, f'{letter}  {title}', real_long, real_summary)
    for letter, axis, (metric, title, _label) in zip('EF', bottom_axes, SUPPLEMENTARY_PRESERVATION_METRICS, strict=True):
        draw_real_metric(axis, metric, f'{letter}  {title}', real_long, real_summary)
    figure_outputs = save_figure(figure, stem, int(args.dpi))
    output_paths = {**figure_outputs, 'controlled_plot_data': controlled_path, 'real_slice_seed_plot_data': real_long_path, 'real_slice_summary_plot_data': real_summary_path}
    record = {'schema_version': 1, 'study': '04_batch_effect_removal', 'figure_configuration_id': 'study04-integrated-removal-preservation-supplement-v7-alpha1-endpoint', 'publication_status': 'supplementary_candidate_author_review_required', 'figure_promotion_authorized': False, 'active_manuscript_claim': None, 'winner_ranking_authorized': False, 'composite_score_used': False, 'new_experiments_used': False, 'panel_indices_rendered': True, 'claim_boundary': 'controlled FEAST perturbations reveal method-specific batch-removal and biological-preservation trade-offs; no universally best method is established', 'scope': {'study04a': 'controlled same-slice paired recovery, α=0.25 through 1.00', 'study04b': 'one within-donor real DLPFC section pair, raw anchor plus α=0.00 through 1.00', 'displayed_alpha_maximum': 1.0, 'cross_section_exact_spot_pairing': 'prohibited_not_computed'}, 'metric_selection': {'study04a': ['batch_mixing_entropy_k30', 'paired_retrieval_top1', 'domain_purity_k30'], 'study04b_rendered': [metric for metric, _, _ in SUPPLEMENTARY_REAL_METRICS], 'study04b_omitted_as_redundant_or_secondary': ['conditional_batch_asw_macro', 'query_stability_linear_cka']}, 'study04b_seed_display': 'individual seeds 42-44, seed mean, and min-max band; not biological replicates', 'uncorrected_baseline': {'rendered': False, 'retained_in_task_specific_plot_data_and_diagnostics': True}, 'text_policy': {'figure_title_and_subtitle': 'moved_to_caption', 'preferred_direction_prose': 'removed_as_redundant_with_arrows', 'condition_tick_labels': CONDITION_LABELS, 'raw_anchor_explanation': 'one concise footer', 'endpoint_seed_and_limit_explanations': 'moved_to_caption'}, 'png_dpi': int(args.dpi), 'inputs': {name: {'path': repository_path(path)} for name, path in source_paths.items()}, 'outputs': {name: {'path': repository_path(path)} for name, path in output_paths.items()}, 'editable_text': {'pdf_fonttype': 42, 'svg_fonttype': 'none', 'deterministic_export_metadata': True}}
    provenance_path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(f'wrote Study 04 integrated story figure to {output}')
if __name__ == '__main__':
    main()
