"""Render validated Study 06 fidelity and continuity diagnostics."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('SOURCE_DATE_EPOCH', '0')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
REPRO_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = REPRO_ROOT / '06_3d_stack'
DEFAULT_INPUT_ROOT = STUDY_ROOT / 'outputs' / 'generative_five_reference_v1'
DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'figures' / 'generative_v1'
GAP_ORDER = [3, 5, 10]
GAP_COLORS = {3: '#0072B2', 5: '#E69F00', 10: '#009E73'}
GAP_LABELS: dict[int, str] = {}
GENERATED_COLOR = '#9A4D8E'
BASELINE_COLOR = '#A7A7A7'

def provenance_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPRO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()

def configure_matplotlib() -> None:
    arial_paths = (Path(sys.prefix) / 'fonts' / 'arial.ttf', Path(sys.prefix) / 'fonts' / 'arialbd.ttf')
    if not all((path.is_file() for path in arial_paths)):
        raise FileNotFoundError(f'Arial Regular and Bold are required for this figure: {arial_paths}')
    for path in arial_paths:
        font_manager.fontManager.addfont(path)
    mpl.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold', 'axes.labelsize': 8.5, 'axes.labelcolor': '#222222', 'axes.edgecolor': '#555555', 'axes.linewidth': 0.8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'xtick.color': '#333333', 'ytick.color': '#333333', 'legend.fontsize': 7.2, 'legend.frameon': False, 'text.color': '#222222', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})

def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis='y', color='#E3E3E3', linewidth=0.5, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    for key in ('left', 'bottom'):
        axis.spines[key].set_color('#555555')
        axis.spines[key].set_linewidth(0.8)
    axis.tick_params(length=3, width=0.7, color='#555555')

def require_inputs(input_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    evaluation_root = input_root / 'evaluation'
    input_paths = {'validation_summary': input_root / 'validation_summary.json', 'score_provenance': evaluation_root / 'score_provenance.json', 'target_metrics': evaluation_root / 'target_metrics.csv', 'continuity_summary': evaluation_root / 'continuity_summary.csv', 'plan': input_root / 'plan.json'}
    validation = json.loads(input_paths['validation_summary'].read_text(encoding='utf-8'))
    if validation.get('status') != 'passed' or validation.get('positions_per_stack') != 144 or validation.get('stack_geometries_identical') is not True:
        raise ValueError('complete Study 06 validation is required')
    provenance = json.loads(input_paths['score_provenance'].read_text(encoding='utf-8'))
    if provenance.get('status') != 'complete' or provenance.get('metric_contract', {}).get('composite_score') != 'prohibited':
        raise ValueError('Study 06 atomic metric contract changed')
    metrics = pd.read_csv(input_paths['target_metrics'])
    required_metrics = {'input_mean_corr', 'input_variance_corr', 'gene_zero_fraction_wasserstein', 'relative_error_mean', 'zero_mask_jaccard'}
    missing_metrics = sorted(required_metrics - set(metrics.columns))
    if missing_metrics:
        raise ValueError(f'Study 06 is missing Study 00-equivalent metrics: {missing_metrics}')
    continuity = pd.read_csv(input_paths['continuity_summary'])
    plan = json.loads(input_paths['plan'].read_text(encoding='utf-8'))
    donor_support = {}
    for density in plan['densities'].values():
        for target in density['targets']:
            donor_support[str(target['density_name']), int(target['target_slice'])] = bool(target['donor_support'])
    metrics['donor_supported'] = [donor_support[str(row.density_name), int(row.target_slice)] for row in metrics.itertuples(index=False)]
    return (metrics, continuity, provenance, plan)

def metric_panel(axis: plt.Axes, frame: pd.DataFrame, column: str, title: str, ylabel: str, ylim: tuple[float, float] | None, yticks: list[float] | None, yscale: str='linear') -> None:
    values = [frame.loc[frame['gap'] == gap, column].dropna().to_numpy() for gap in GAP_ORDER]
    if yscale == 'log' and any((np.any(values_for_gap <= 0) for values_for_gap in values)):
        raise ValueError(f'{column} requires strictly positive values for a log axis')
    boxes = axis.boxplot(values, positions=np.arange(len(GAP_ORDER)), widths=0.52, showfliers=False, patch_artist=True, medianprops={'color': '#222222', 'linewidth': 1.1})
    for patch, gap in zip(boxes['boxes'], GAP_ORDER):
        patch.set_facecolor(GAP_COLORS[gap])
        patch.set_alpha(0.18)
        patch.set_edgecolor(GAP_COLORS[gap])
        patch.set_linewidth(0.9)
    for line, gap in zip(boxes['whiskers'], np.repeat(GAP_ORDER, 2)):
        line.set_color(GAP_COLORS[gap])
        line.set_alpha(0.75)
        line.set_linewidth(0.8)
    for line, gap in zip(boxes['caps'], np.repeat(GAP_ORDER, 2)):
        line.set_color(GAP_COLORS[gap])
        line.set_alpha(0.75)
        line.set_linewidth(0.8)
    for position, gap in enumerate(GAP_ORDER):
        subset = frame.loc[frame['gap'] == gap].sort_values('target_slice')
        offsets = np.linspace(-0.16, 0.16, len(subset))
        for donor, marker in ((False, 'o'), (True, 'x')):
            mask = subset['donor_supported'] == donor
            axis.scatter(position + offsets[mask.to_numpy()], subset.loc[mask, column], s=10, marker=marker, color=GAP_COLORS[gap], linewidths=0.65, alpha=0.78, zorder=3)
    axis.set_title(title, loc='left', pad=6)
    axis.set_ylabel(ylabel)
    axis.set_xticks(np.arange(len(GAP_ORDER)), ['3', '5', '10'])
    axis.set_xlabel('Reference gap')
    axis.set_yscale(yscale)
    if ylim is not None:
        axis.set_ylim(*ylim)
    if yticks is not None:
        axis.set_yticks(yticks)
    style_axis(axis)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = output / 'conditional_stack_diagnostic'
    targets = [stem.with_suffix(suffix) for suffix in ('.pdf', '.svg', '.png')]
    targets += [output / 'target_metric_plot_data.csv', output / 'continuity_plot_data.csv', output / 'figure_provenance.json']
    if any((path.exists() for path in targets)):
        raise FileExistsError('fresh-only Study 06 figure output already exists')
    input_root = args.input_root.resolve()
    metrics, continuity, score_provenance, plan = require_inputs(input_root)
    GAP_LABELS.update({int(density['spec']['gap']): f"Gap {int(density['spec']['gap'])} ({len(density['reference_pool'])} real / {len(density['target_pool'])} simulated)" for density in plan['densities'].values()})
    metrics.to_csv(output / 'target_metric_plot_data.csv', index=False)
    continuity.to_csv(output / 'continuity_plot_data.csv', index=False)
    configure_matplotlib()
    figure = plt.figure(figsize=(7.5, 5.6))
    grid = figure.add_gridspec(3, 3, height_ratios=[0.26, 1.0, 1.0], left=0.085, right=0.985, bottom=0.085, top=0.985, hspace=0.50, wspace=0.42)
    legend_axis = figure.add_subplot(grid[0, :])
    legend_axis.set_axis_off()
    axes = np.asarray([[figure.add_subplot(grid[1, index]) for index in range(3)], [figure.add_subplot(grid[2, index]) for index in range(3)]])
    panels = [('input_mean_corr', 'A  Mean Corr ↑', 'Metric value', (0.79, 1.005), [0.8, 0.9, 1.0], 'linear'), ('input_variance_corr', 'B  Var Corr ↑', 'Metric value', (0.71, 1.005), [0.75, 0.85, 0.95], 'linear'), ('gene_zero_fraction_wasserstein', 'C  Zero Frac WDist ↓', 'Metric value', None, None, 'log'), ('relative_error_mean', 'D  Real Error Mean ↓', 'Metric value', (0.0, 2.9), [0.0, 1.0, 2.0], 'linear'), ('zero_mask_jaccard', 'E  Zero Jaccard ↑', 'Metric value', (0.84, 0.922), [0.85, 0.88, 0.91], 'linear')]
    for axis, (column, title, ylabel, ylim, yticks, yscale) in zip(axes.ravel()[:5], panels):
        metric_panel(axis, metrics, column, title, ylabel, ylim, yticks, yscale)
    axis = axes.ravel()[5]
    width = 0.32
    positions = np.arange(len(GAP_ORDER))
    ordered = continuity.set_index('gap').loc[GAP_ORDER]
    generated = ordered['reconstructed_original_z_coherence_median'].to_numpy(float)
    baseline = ordered['real_split_half_z_coherence_median'].to_numpy(float)
    normalized = ordered['normalized_z_coherence'].to_numpy(float)
    axis.bar(positions - width / 2, generated, width, color=GENERATED_COLOR, label='Reconstructed–original')
    axis.bar(positions + width / 2, baseline, width, color=BASELINE_COLOR, label='Real split-half')
    for position, generated_value, baseline_value, ratio in zip(positions, generated, baseline, normalized):
        axis.text(position, max(generated_value, baseline_value) + 0.025, f'{ratio:.2f}×', ha='center', va='bottom', fontsize=6.8, fontweight='bold', color=GENERATED_COLOR)
    axis.set_title('F  Across-z coherence', loc='left', pad=6)
    axis.set_ylabel('Median z correlation  ↑')
    axis.set_xticks(positions, ['3', '5', '10'])
    axis.set_xlabel('Reference gap')
    axis.set_ylim(0.0, 0.82)
    axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    style_axis(axis)
    density_handles = [mpl.lines.Line2D([], [], color=GAP_COLORS[gap], marker='s', linestyle='None', markersize=6, label=GAP_LABELS[gap]) for gap in GAP_ORDER]
    donor_handles = [mpl.lines.Line2D([], [], color='#555555', marker='o', linestyle='None', markersize=4, label='Bracket-only labels'), mpl.lines.Line2D([], [], color='#555555', marker='x', linestyle='None', markersize=5, label='Uses label donor'), Patch(facecolor=GENERATED_COLOR, edgecolor='none', label='Full-stack reconstructed–original coherence'), Patch(facecolor=BASELINE_COLOR, edgecolor='none', label='Real split-half baseline')]
    density_legend = legend_axis.legend(handles=density_handles, loc='center', bbox_to_anchor=(0.5, 0.74), ncol=3, columnspacing=1.25, handletextpad=0.4, fontsize=7.0)
    legend_axis.add_artist(density_legend)
    legend_axis.legend(handles=donor_handles, loc='center', bbox_to_anchor=(0.5, 0.20), ncol=4, columnspacing=0.9, handletextpad=0.35, fontsize=6.2)
    figure.savefig(stem.with_suffix('.pdf'), metadata={'Title': 'FEAST Study 06 conditional stack diagnostic', 'Creator': 'FEAST publication visualization', 'CreationDate': None, 'ModDate': None})
    figure.savefig(stem.with_suffix('.svg'), metadata={'Title': 'FEAST Study 06 conditional stack diagnostic', 'Creator': 'FEAST publication visualization', 'Date': None})
    figure.savefig(stem.with_suffix('.png'), dpi=int(args.dpi), metadata={'Software': 'FEAST Study 06 plotting script'})
    plt.close(figure)
    source_paths = {'validation_summary': input_root / 'validation_summary.json', 'score_provenance': input_root / 'evaluation' / 'score_provenance.json', 'target_metrics': input_root / 'evaluation' / 'target_metrics.csv', 'continuity_summary': input_root / 'evaluation' / 'continuity_summary.csv', 'plan': input_root / 'plan.json'}
    output_paths = {'pdf': stem.with_suffix('.pdf'), 'svg': stem.with_suffix('.svg'), 'png': stem.with_suffix('.png'), 'target_plot_data': output / 'target_metric_plot_data.csv', 'continuity_plot_data': output / 'continuity_plot_data.csv'}
    record = {'schema_version': 1, 'study': '06_3d_stack', 'figure_configuration_id': 'study06-conditional-stack-figure-v2', 'configuration_id': score_provenance['configuration_id'], 'figure_promotion_authorized': False, 'publication_claim_authorized': False, 'png_dpi': int(args.dpi), 'interpretation': 'validated atomic fidelity and continuity diagnostic; target slices are ordered z levels, not biological replicates', 'study00_equivalent_panels': ['input_mean_corr', 'input_variance_corr', 'gene_zero_fraction_wasserstein', 'relative_error_mean', 'zero_mask_jaccard'], 'composite_score': 'prohibited', 'winner_ranking': 'prohibited', 'donor_support_marker_rule': 'x marks targets requiring at least one declared out-of-bracket label donor', 'inputs': {name: {'path': provenance_path(path)} for name, path in source_paths.items()}, 'outputs': {name: {'path': provenance_path(path)} for name, path in output_paths.items()}, 'editable_text': {'pdf_fonttype': 42, 'svg_fonttype': 'none', 'deterministic_export_metadata': True}}
    (output / 'figure_provenance.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'wrote Study 06 diagnostic figure to {output}')
if __name__ == '__main__':
    main()
