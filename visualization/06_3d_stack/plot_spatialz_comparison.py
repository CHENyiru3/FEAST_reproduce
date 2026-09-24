"""Render the complete three-method Study 06 comparison from existing metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

from plot_results import configure_style

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('feast', 'resampling', 'spatialz')
NAMES = {'spatialz': 'SpatialZ', 'feast': 'FEAST', 'resampling': 'Resampling'}
COLORS = {'spatialz': '#B64342', 'feast': '#0072B2', 'resampling': '#777777'}
GAPS = (3, 5, 10)
KEYS = ['gap', 'target_slice']
MAIN = [
    ('input_mean_corr', 'Gene-mean correlation ↑'),
    ('input_variance_corr', 'Gene-variance correlation ↑'),
    ('gene_zero_fraction_wasserstein', 'Gene zero-fraction Wasserstein ↓'),
    ('expression_wasserstein', 'Expression Wasserstein ↓'),
    ('library_size_wasserstein', 'Library-size Wasserstein ↓'),
    ('detected_genes_wasserstein', 'Detected-gene-count Wasserstein ↓'),
    ('moran_profile_correlation', 'Moran-profile correlation ↑'),
    ('residual_moran_profile_correlation', 'Residual Moran correlation ↑'),
    ('relative_error_mean', 'Relative gene-mean error ↓'),
]
CONDITIONAL = ['conditional_expression_wasserstein', 'conditional_zero_fraction_wasserstein']
GEOMETRY = ['cell_count_relative_error', 'class_proportion_total_variation',
            'missing_target_class_mass', 'extra_generated_class_mass']
SHARED = [m for m, _ in MAIN if m != 'expression_wasserstein'] + CONDITIONAL + ['gene_zero_fraction_pearson']


def summarize(frame, metrics):
    rows = []
    for (gap, method), group in frame.groupby(['gap', 'method']):
        for metric in metrics:
            values = group.loc[np.isfinite(group[metric]), metric]
            rows.append(dict(gap=int(gap), method=method, metric=metric,
                             n_targets=len(group), n_defined=len(values),
                             median=values.median(), q25=values.quantile(.25), q75=values.quantile(.75)))
    return pd.DataFrame(rows)


def load_comparison(evaluation, baseline):
    metadata = json.loads((evaluation / 'evaluation.json').read_text())
    if metadata['status'] != 'completed' or metadata['targets_evaluated'] != 336:
        raise ValueError('the complete 336-target evaluation is required')
    current = pd.read_csv(evaluation / 'metrics.csv')
    previous = pd.read_csv(baseline)
    left = current[current.method == 'feast'].set_index(KEYS).sort_index()
    right = previous[previous.method == 'feast'].set_index(KEYS).sort_index()
    pd.testing.assert_index_equal(left.index, right.index)
    np.testing.assert_allclose(left[SHARED], right[SHARED], rtol=1e-9, atol=1e-10, equal_nan=True)
    draws = previous[previous.method == 'conditional_whole_spot_resampling']
    if draws.duplicated(KEYS + ['replicate']).any() or not draws.groupby(KEYS).size().eq(10).all():
        raise ValueError('expected ten unique resampling repeats per target')
    averages = draws.groupby(KEYS)[SHARED].mean().join(left[['target_z', 'target_index', 'density_name']])
    pd.testing.assert_index_equal(left.index, averages.index)
    averages['method'] = 'resampling'
    combined = pd.concat([current, averages.reset_index()], ignore_index=True)
    for method in METHODS:
        counts = combined[combined.method == method].groupby('gap').size().to_dict()
        if counts != {3: 95, 5: 114, 10: 127}:
            raise ValueError(f'{method}: unexpected target coverage {counts}')
    conditional_rows = []
    for metric in CONDITIONAL:
        wide = combined.pivot(index=KEYS, columns='method', values=metric).dropna()
        for gap, group in wide.groupby(level='gap'):
            for method in METHODS:
                values = group[method]
                conditional_rows.append(dict(gap=int(gap), method=method, metric=metric,
                                             n_targets=len(group), n_defined=len(group),
                                             median=values.median(), q25=values.quantile(.25), q75=values.quantile(.75)))
    return combined, pd.DataFrame(conditional_rows)


def panel(axis, summary, metric, title, methods=METHODS, labels=None):
    for method, offset in zip(methods, np.linspace(-.15, .15, len(methods)) if len(methods) > 1 else [0]):
        data = summary[(summary.method == method) & (summary.metric == metric)].set_index('gap').reindex(GAPS)
        if not data.n_defined.fillna(0).gt(0).any():
            continue
        med = data['median'].to_numpy(float)
        axis.errorbar(np.arange(3) + offset, med,
                      yerr=np.vstack([med-data.q25, data.q75-med]),
                      color=COLORS[method], marker='o', markersize=5,
                      linewidth=1.6, capsize=3, elinewidth=1.1)
    axis.set_title(title, loc='left', pad=10)
    axis.set_xticks(range(3), labels if labels else ['3', '5', '10'])
    axis.set_xlabel('Reference gap')
    axis.set_xlim(-.4, 2.4)
    axis.yaxis.set_major_locator(MaxNLocator(5))
    axis.grid(axis='y', color='#E5E8EC', linewidth=.65)
    axis.set_axisbelow(True)


def save(fig, destination, stem, dpi=300):
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(destination / f'{stem}.{extension}', dpi=dpi, facecolor='white')
    plt.close(fig)


def figures(summary, conditional, destination, dpi=300):
    configure_style()
    handles = [Line2D([], [], color=COLORS[m], marker='o', label=NAMES[m], linewidth=1.6) for m in METHODS]
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 11.3))
    for i, ((metric, title), axis) in enumerate(zip(MAIN, axes.flat)):
        panel(axis, summary, metric, f'{chr(65+i)}  {title}')
        if metric == 'expression_wasserstein':
            axis.text(.98, .96, 'Resampling: not computed', transform=axis.transAxes,
                      ha='right', va='top', fontsize=9, color='#666666')
    fig.suptitle('Study 06: expression fidelity and spatial-autocorrelation profiles', fontsize=17, y=.98)
    fig.legend(handles=handles, ncol=3, loc='upper center', bbox_to_anchor=(.5, .95))
    fig.text(.07, .025, 'Points: slice medians; bars: interquartile ranges (not confidence intervals). N = 95 / 114 / 127 slices for gaps 3 / 5 / 10.\nResampling: average of 10 repeats per slice. SpatialZ infers target cells; FEAST and resampling receive target geometry and labels.', fontsize=10)
    fig.subplots_adjust(left=.07, right=.98, bottom=.12, top=.88, hspace=.53, wspace=.30)
    save(fig, destination, 'overview', dpi)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.3))
    counts = conditional[conditional.method == 'spatialz'].drop_duplicates('gap').set_index('gap').n_targets
    labels = [f'{gap}\nn={int(counts[gap])}' for gap in GAPS]
    panel(axes[0, 0], conditional, CONDITIONAL[0], 'A  Conditional expression Wasserstein ↓', labels=labels)
    panel(axes[0, 1], conditional, CONDITIONAL[1], 'B  Conditional zero-fraction W1 ↓', labels=labels)
    covered = [int(counts[g]) for g in GAPS]
    totals = [95, 114, 127]
    ax = axes[0, 2]
    ax.bar(range(3), np.array(covered)/totals, color=COLORS['spatialz'], width=.55)
    for i, (n, total) in enumerate(zip(covered, totals)):
        ax.text(i, n/total+.018, f'{n}/{total}', ha='center', fontsize=11)
    ax.set(title='C  SpatialZ: full conditional score available', xlabel='Reference gap', ylim=(0, .25))
    ax.set_xticks(range(3), ['3', '5', '10'])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    for ax, metric, title in zip(axes[1], GEOMETRY[:3], [
        'D  SpatialZ: cell-count error ↓', 'E  SpatialZ: class-proportion TV ↓', 'F  SpatialZ: missing observed class mass ↓']):
        panel(ax, summary, metric, title, methods=('spatialz',))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=3 if metric == 'missing_target_class_mass' else 1))
    fig.suptitle('Conditional expression and native cell recovery', fontsize=17, y=.98)
    fig.legend(handles=handles, ncol=3, loc='upper center', bbox_to_anchor=(.5, .94))
    fig.text(.07, .025, 'A–B: identical supported target subsets for all methods; this is not the full set of slices. D–F: medians and interquartile ranges.\nFEAST/resampling receive target cell counts and labels, so they do not perform the cell-recovery task shown in D–F.', fontsize=10)
    fig.subplots_adjust(left=.07, right=.98, bottom=.15, top=.83, hspace=.64, wspace=.30)
    save(fig, destination, 'conditional_and_cell_recovery', dpi)


def report(summary, conditional, destination):
    lines = ['# Complete Study 06 comparison', '',
             'All 336 held-out targets and all 1,122 genes. Values below are medians across slices. Resampling uses the mean of ten repeats per slice before aggregation. ↑ higher is favorable; ↓ lower is favorable.', '',
             '![Overview](overview.png)', '']
    display = MAIN + [('gene_zero_fraction_pearson', 'Gene zero-fraction correlation ↑')]
    for gap, total in zip(GAPS, (95, 114, 127)):
        table = summary[summary.gap == gap].pivot(index='metric', columns='method', values='median')
        lines += [f'## Gap {gap}: {total} targets', '', '| Metric | FEAST | Resampling | SpatialZ |', '|---|---:|---:|---:|']
        for metric, title in display:
            cells = [f'{table.loc[metric,m]:.5g}' if pd.notna(table.loc[metric,m]) else '—' for m in METHODS]
            lines.append('| ' + ' | '.join([title] + cells) + ' |')
        lines.append('')
    lines += ['## Conditional expression on common supported targets', '',
              'A whole conditional score is undefined when a generated slice lacks any observed class. Comparisons below use the same supported slices for every method within a gap; they do not summarize all 336 targets. Supported subsets differ across gaps, so the conditional curves do not isolate an effect of reference density.', '',
              '| Gap | Supported / total | Expression W1: FEAST / resampling / SpatialZ | Zero-fraction W1: FEAST / resampling / SpatialZ |', '|---|---:|---|---|']
    for gap, total in zip(GAPS, (95, 114, 127)):
        block = conditional[conditional.gap == gap]
        table = block.pivot(index='metric', columns='method', values='median')
        cells = [' / '.join(f'{table.loc[metric,m]:.5g}' for m in METHODS) for metric in CONDITIONAL]
        lines.append(f'| {gap} | {int(block.n_targets.iloc[0])}/{total} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## SpatialZ cell recovery', '',
              '| Gap | Cell-count error | Class-proportion total variation | Missing observed class mass |', '|---|---:|---:|---:|']
    for gap in GAPS:
        table = summary[(summary.gap == gap) & (summary.method == 'spatialz')].set_index('metric')['median']
        lines.append(f'| {gap} | {table[GEOMETRY[0]]:.2%} | {table[GEOMETRY[1]]:.2%} | {table[GEOMETRY[2]]:.4%} |')
    lines += ['', '![Conditional expression and cell recovery](conditional_and_cell_recovery.png)', '',
              '## Interpretation and limits', '',
              '- Moran-profile correlation measures agreement in the gene-wise strength of spatial autocorrelation. It does not test anatomical placement or absolute agreement of Moran values; a correlated profile can still be shifted or scaled.',
              '- Residual Moran subtracts each method’s own within-class gene mean. It still uses each slice’s own coordinates and neighborhood graph.',
              '- FEAST and resampling use observed target cell counts, coordinates, and class labels, with five primary references and declared supporting donors. SpatialZ infers cells, coordinates, and labels from two bracketing references. These information inputs differ.',
              '- Unconditional expression Wasserstein was not computed in the existing resampling run; the missing values are not zero.',
              '- Serial sections are not independent biological replicates. Interquartile ranges describe slice variation; no significance tests, composite score, or overall winner is assigned.',
              '- Anatomical expression maps and a three-method whole-stack continuity comparison are not part of these slice metrics.', '',
              'All shared FEAST metrics were checked against the stored resampling-comparison FEAST scores for all 336 targets and agree numerically. Plotting recomputes no gene-expression or spatial metric.', '',
              'Files: `summary.csv`, `conditional_common_targets.csv`, `plot_data.csv`, PDF/SVG/PNG figures, and `sources.json`.', '']
    (destination / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, default=ROOT / '06_3d_stack/outputs/spatialz_native_evaluation_cuda_v1/evaluation')
    parser.add_argument('--baseline', type=Path, default=ROOT / '06_3d_stack/outputs/generative_five_reference_1.0.6_precision_v1/baselines/five_reference_conditional_resampling/method_metrics.csv')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'figures/spatialz_full_comparison_v1')
    args = parser.parse_args()
    combined, conditional = load_comparison(args.evaluation, args.baseline)
    summary = summarize(combined, [m for m, _ in MAIN] + CONDITIONAL + GEOMETRY + ['gene_zero_fraction_pearson'])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    combined.to_csv(args.output_dir / 'plot_data.csv', index=False)
    summary.to_csv(args.output_dir / 'summary.csv', index=False)
    conditional.to_csv(args.output_dir / 'conditional_common_targets.csv', index=False)
    figures(summary, conditional, args.output_dir)
    report(summary, conditional, args.output_dir)
    (args.output_dir / 'sources.json').write_text(json.dumps({
        'entry_point': str(Path(__file__).resolve()), 'evaluation': str(args.evaluation.resolve()),
        'baseline': str(args.baseline.resolve()), 'target_counts': {'3': 95, '5': 114, '10': 127},
        'resampling_aggregation': 'mean of ten repeats within each target, then median and IQR across targets',
        'conditional_comparison': 'identical supported target subset across methods per gap',
        'shared_feast_metric_agreement': 'passed on all 336 targets',
    }, indent=2) + '\n')
    print(args.output_dir / 'REPORT.md')
    print(summary.pivot(index=['gap','metric'], columns='method', values='median').to_string())
    print('Conditional scores on common targets:')
    print(conditional[['gap','method','metric','n_targets','median']].to_string(index=False))


if __name__ == '__main__':
    main()
