"""Render full-axis Study 07 coverage and adjacent-z continuity diagnostics."""
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
REPRO_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = REPRO_ROOT / '07_3d_transfer'
INPUT_ROOT = STUDY_ROOT / 'outputs' / 'generative_transfer_github_a79cf22_estimated_ar_v1'
EVALUATION_ROOT = INPUT_ROOT / 'evaluation'
DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'figures' / 'generative_v1'
REGION_COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442', '#666666', '#B9B9B9']
CONTINUITY_COLORS = {'gene_mean_discontinuity': '#7A3E9D', 'region_gene_mean_discontinuity': '#008B8B'}
CONTINUITY_LABELS = {'gene_mean_discontinuity': 'Gene mean', 'region_gene_mean_discontinuity': 'Region–gene mean'}

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

def style_axis(axis: plt.Axes, grid_axis: str='y') -> None:
    axis.grid(axis=grid_axis, color='#DDDDDD', linewidth=0.5, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    for key in ('left', 'bottom'):
        axis.spines[key].set_color('#555555')
        axis.spines[key].set_linewidth(0.8)
    axis.tick_params(length=3, width=0.7, color='#555555')

def require_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    input_paths = {'evaluation_provenance': EVALUATION_ROOT / 'provenance.json', 'region_coverage': EVALUATION_ROOT / 'region_coverage.csv', 'adjacent_z_continuity': EVALUATION_ROOT / 'adjacent_z_continuity.csv', 'age_summary': EVALUATION_ROOT / 'age_summary.csv', 'validation': INPUT_ROOT / 'validation.json', 'E15.5_manifest': INPUT_ROOT / 'E15.5' / 'manifest.csv', 'E18.5_manifest': INPUT_ROOT / 'E18.5' / 'manifest.csv'}
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'validated Study 07 inputs are required: {missing}')
    provenance = json.loads(input_paths['evaluation_provenance'].read_text(encoding='utf-8'))
    if provenance.get('status') != 'complete' or provenance.get('accuracy_claim_authorized'):
        raise ValueError('Study 07 descriptive-only evaluation contract changed')
    for name in provenance['outputs']:
        if not (EVALUATION_ROOT / name).is_file():
            raise FileNotFoundError(f'evaluation output listed in provenance is missing: {name}')
    coverage = pd.read_csv(input_paths['region_coverage'])
    adjacent = pd.read_csv(input_paths['adjacent_z_continuity'])
    summary = pd.read_csv(input_paths['age_summary'])
    return (coverage, adjacent, summary, provenance)

def coverage_for_plot(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    totals = frame.groupby('region', as_index=False)['n_spots'].sum().sort_values('n_spots', ascending=False)
    retained = totals.head(8)['region'].astype(str).tolist()
    result = frame.copy()
    result['display_region'] = result['region'].where(result['region'].isin(retained), 'Other')
    result = result.groupby(['age', 'z_index', 'z_world', 'display_region'], as_index=False)['n_spots'].sum()
    order = retained + (['Other'] if 'Other' in set(result['display_region']) else [])
    return (result, order)

def spot_tick(value: float, _position: float) -> str:
    if value == 0:
        return '0'
    return f'{value / 1000:g}k'

def plot_coverage(axis: plt.Axes, frame: pd.DataFrame, region_order: list[str], age_summary: pd.Series, panel: str) -> None:
    age = str(age_summary.name)
    age_frame = frame.loc[frame['age'] == age]
    pivot = age_frame.pivot_table(index='z_world', columns='display_region', values='n_spots', fill_value=0).sort_index()
    x = pivot.index.to_numpy(dtype=float)
    series = [pivot[region].to_numpy(dtype=float) if region in pivot else np.zeros(len(pivot)) for region in region_order]
    axis.stackplot(x, series, colors=REGION_COLORS[:len(region_order)], linewidth=0, alpha=0.97)
    total = np.sum(np.vstack(series), axis=0)
    axis.plot(x, total, color='#333333', linewidth=0.8, zorder=3)
    axis.set_title(f'{panel}  {age} regional support', loc='left', pad=6)
    axis.set_ylabel('Blueprint spots per z level')
    axis.set_xlabel('DevCCF z coordinate')
    axis.set_xlim(float(age_summary['z_start']), float(age_summary['z_end']))
    axis.set_ylim(bottom=0)
    axis.yaxis.set_major_formatter(FuncFormatter(spot_tick))
    axis.text(0.98, 0.95, f"{int(age_summary['n_z_levels'])} levels  ·  {age_summary['n_spots'] / 1000000.0:.2f}M spots", transform=axis.transAxes, ha='right', va='top', fontsize=7.0, color='#555555', bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.82, 'pad': 1.7})
    style_axis(axis)

def plot_continuity(axis: plt.Axes, frame: pd.DataFrame, age_summary: pd.Series, panel: str) -> None:
    age = str(age_summary.name)
    age_frame = frame.loc[frame['age'] == age].sort_values('midpoint_z')
    for metric, linestyle in (('gene_mean_discontinuity', '-'), ('region_gene_mean_discontinuity', (0, (3.2, 1.8)))):
        axis.plot(age_frame['midpoint_z'], age_frame[metric], color=CONTINUITY_COLORS[metric], linewidth=1.05, linestyle=linestyle, alpha=0.96)
    maxima = {metric: int(age_frame[metric].idxmax()) for metric in CONTINUITY_COLORS}
    maximum_metric = max(maxima, key=lambda metric: float(age_frame.loc[maxima[metric], metric]))
    maximum_row = age_frame.loc[maxima[maximum_metric]]
    maximum_value = float(maximum_row[maximum_metric])
    axis.scatter([float(maximum_row['midpoint_z'])], [maximum_value], s=23, facecolor='white', edgecolor=CONTINUITY_COLORS[maximum_metric], linewidth=1.0, zorder=5)
    axis.annotate(f"Largest observed step\n{CONTINUITY_LABELS[maximum_metric]}, z = {float(maximum_row['midpoint_z']):.2f}", xy=(float(maximum_row['midpoint_z']), maximum_value), xycoords='data', xytext=(0.97, 0.96), textcoords='axes fraction', ha='right', va='top', fontsize=6.7, color='#555555', bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.86, 'pad': 1.2}, arrowprops={'arrowstyle': '-', 'color': CONTINUITY_COLORS[maximum_metric], 'linewidth': 0.7, 'shrinkA': 2, 'shrinkB': 3})
    axis.set_title(f'{panel}  {age} adjacent-z continuity', loc='left', pad=6)
    axis.set_ylabel('Discontinuity, Δ = 1 − Pearson r\n(log scale; lower is smoother)')
    axis.set_xlabel('Adjacent-slice midpoint z')
    axis.set_xlim(float(age_summary['z_start']), float(age_summary['z_end']))
    axis.set_yscale('log')
    displayed = age_frame[list(CONTINUITY_COLORS)].to_numpy(dtype=float)
    finite = displayed[np.isfinite(displayed) & (displayed > 0)]
    axis.set_ylim(float(finite.min()) * .5, float(finite.max()) * 2)
    axis.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=6))
    style_axis(axis)

def main() -> None:
    global INPUT_ROOT, EVALUATION_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=INPUT_ROOT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    INPUT_ROOT = args.input_root.resolve()
    EVALUATION_ROOT = INPUT_ROOT / "evaluation"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = output / 'full_axis_transfer_diagnostic'
    targets = [stem.with_suffix(suffix) for suffix in ('.pdf', '.svg', '.png')]
    targets += [output / 'region_coverage_plot.csv', output / 'adjacent_z_plot.csv', output / 'figure_provenance.json']
    if any((path.exists() for path in targets)):
        raise FileExistsError('fresh-only Study 07 figure output already exists')
    coverage, adjacent, summary, evaluation_provenance = require_inputs()
    coverage_plot, region_order = coverage_for_plot(coverage)
    adjacent_plot = adjacent.copy()
    adjacent_plot['gene_mean_discontinuity'] = np.maximum(1.0 - adjacent_plot['gene_mean_pearson'].to_numpy(dtype=float), 1e-12)
    adjacent_plot['region_gene_mean_discontinuity'] = np.maximum(1.0 - adjacent_plot['region_gene_mean_pearson'].to_numpy(dtype=float), 1e-12)
    coverage_plot.to_csv(output / 'region_coverage_plot.csv', index=False)
    adjacent_plot.to_csv(output / 'adjacent_z_plot.csv', index=False)
    configure_matplotlib()
    figure = plt.figure(figsize=(7.5, 6.2))
    grid = figure.add_gridspec(3, 2, height_ratios=[0.25, 1.0, 1.0], width_ratios=[1.42, 1.0], left=0.09, right=0.985, bottom=0.105, top=0.985, hspace=0.56, wspace=0.34)
    legend_axis = figure.add_subplot(grid[0, :])
    legend_axis.set_axis_off()
    axes = np.asarray([[figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])], [figure.add_subplot(grid[2, 0]), figure.add_subplot(grid[2, 1])]])
    summary_by_age = summary.set_index('age')
    for row, (age, coverage_panel, continuity_panel) in enumerate((('E15.5', 'A', 'B'), ('E18.5', 'C', 'D'))):
        plot_coverage(axes[row, 0], coverage_plot, region_order, summary_by_age.loc[age], coverage_panel)
        plot_continuity(axes[row, 1], adjacent_plot, summary_by_age.loc[age], continuity_panel)
    region_handles = [Patch(facecolor=REGION_COLORS[index], edgecolor='none', label=region) for index, region in enumerate(region_order)]
    continuity_handles = [Line2D([], [], color=CONTINUITY_COLORS[metric], linestyle=linestyle, linewidth=1.3, label=CONTINUITY_LABELS[metric]) for metric, linestyle in (('gene_mean_discontinuity', '-'), ('region_gene_mean_discontinuity', (0, (3.2, 1.8))))]
    region_legend = legend_axis.legend(handles=region_handles, loc='upper center', ncol=min(8, len(region_handles)), columnspacing=1.15, handletextpad=0.38, borderaxespad=0.0)
    legend_axis.add_artist(region_legend)
    legend_axis.legend(handles=continuity_handles, loc='lower center', ncol=2, columnspacing=1.8, handlelength=2.3, handletextpad=0.5, borderaxespad=0.0)
    figure.text(0.5, 0.025, 'Ordered z levels are not biological replicates. Continuity is descriptive; target-expression accuracy is not evaluated.', ha='center', va='bottom', fontsize=6.7, color='#666666')
    figure.savefig(stem.with_suffix('.pdf'), metadata={'Title': 'FEAST Study 07 full-axis diagnostic', 'Creator': 'FEAST publication visualization', 'CreationDate': None, 'ModDate': None})
    figure.savefig(stem.with_suffix('.svg'), metadata={'Title': 'FEAST Study 07 full-axis diagnostic', 'Creator': 'FEAST publication visualization', 'Date': None})
    figure.savefig(stem.with_suffix('.png'), dpi=int(args.dpi), metadata={'Software': 'FEAST Study 07 plotting script'})
    plt.close(figure)
    source_paths = {'evaluation_provenance': EVALUATION_ROOT / 'provenance.json', 'region_coverage': EVALUATION_ROOT / 'region_coverage.csv', 'adjacent_z_continuity': EVALUATION_ROOT / 'adjacent_z_continuity.csv', 'age_summary': EVALUATION_ROOT / 'age_summary.csv', 'validation': INPUT_ROOT / 'validation.json', 'E15.5_manifest': INPUT_ROOT / 'E15.5' / 'manifest.csv', 'E18.5_manifest': INPUT_ROOT / 'E18.5' / 'manifest.csv'}
    output_paths = {'pdf': stem.with_suffix('.pdf'), 'svg': stem.with_suffix('.svg'), 'png': stem.with_suffix('.png'), 'coverage_plot_data': output / 'region_coverage_plot.csv', 'adjacent_plot_data': output / 'adjacent_z_plot.csv'}
    record = {'schema_version': 1, 'study': '07_3d_transfer', 'figure_configuration_id': 'study07-full-axis-figure-v2', 'configuration_id': evaluation_provenance['configuration_id'], 'figure_promotion_authorized': False, 'publication_claim_authorized': False, 'target_expression_accuracy_claim_authorized': False, 'png_dpi': int(args.dpi), 'interpretation': 'descriptive full-axis coverage and adjacent-z continuity; no target-expression accuracy claim', 'composite_score': 'prohibited', 'continuity_display': '1 - Pearson correlation on a logarithmic axis; values are clipped only for display at 1e-12', 'top_region_rule': 'eight regions with greatest total blueprint spot support across both ages; remaining regions grouped as Other', 'panel_layout': 'one age per row, aligning full-axis regional support with adjacent-z continuity over the same declared z extent', 'annotation_rule': 'label the largest displayed discontinuity across the two atomic continuity metrics within each age', 'age_summary': summary.to_dict('records'), 'inputs': {name: {'path': provenance_path(path)} for name, path in source_paths.items()}, 'outputs': {name: {'path': provenance_path(path)} for name, path in output_paths.items()}, 'editable_text': {'pdf_fonttype': 42, 'svg_fonttype': 'none', 'deterministic_export_metadata': True}}
    (output / 'figure_provenance.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'wrote Study 07 diagnostic figure to {output}')
if __name__ == '__main__':
    main()
