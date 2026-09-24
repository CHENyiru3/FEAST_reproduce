"""Study 06 figures including FEAST, conditional resampling and native SpatialZ."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from functools import lru_cache

os.environ.setdefault('MPLCONFIGDIR', '/tmp/study06-results-matplotlib')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

from plot import require_inputs

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / '06_3d_stack/outputs/generative_five_reference_1.0.6_precision_v1'
DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'figures/publication_with_spatialz_v1'
DEFAULT_SPATIALZ = ROOT / '06_3d_stack/outputs/spatialz_native_evaluation_cuda_v1'
GAPS = (3, 5, 10)
COLORS = {3: '#0072B2', 5: '#E69F00', 10: '#009E73'}
INK = '#24313F'
GRAY = '#8A9097'


@lru_cache(maxsize=1)
def benchmark_definitions():
    modules = []
    for name, path in (
        ('study00_plot_specs', ROOT / 'visualization/00_simulator_benchmark/plot.py'),
        ('study00_score_functions', ROOT / '00_simulator_benchmark/score.py'),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def add_benchmark_metrics(data, input_root):
    """Reuse the six shared scores; calculate the two missing benchmark scores."""
    import h5py
    from anndata.experimental import read_elem
    from scipy.stats import wasserstein_distance

    cache = input_root / 'evaluation/benchmark_additional_metrics.csv'
    if cache.exists():
        additional = pd.read_csv(cache)
    else:
        _, scoring = benchmark_definitions()
        plan = json.loads((input_root / 'plan.json').read_text())

        def read_counts(path):
            with h5py.File(path, 'r') as handle:
                obs = np.asarray(read_elem(handle['obs'][handle['obs'].attrs['_index']]))
                genes = np.asarray(read_elem(handle['var'][handle['var'].attrs['_index']]))
                matrix = read_elem(handle['layers/counts'])
            return matrix, obs, genes

        records = []
        for index, row in enumerate(data.itertuples(index=False), start=1):
            name = f'Zhuang-ABCA-1.{int(row.target_slice):03d}'
            real, real_ids, real_genes = read_counts(Path(plan['data_dir']) / f'{name}.h5ad')
            simulated, sim_ids, sim_genes = read_counts(
                input_root / 'targets' / row.density_name / name / 'generated.h5ad')
            if not np.array_equal(real_ids, sim_ids) or not np.array_equal(real_genes, sim_genes):
                raise ValueError(f'{name}: matched spot and gene identities are required')
            records.append({'gap': int(row.gap), 'target_slice': int(row.target_slice),
                'cosine_divergence': scoring.cosine_divergence(real, simulated),
                'library_size_wasserstein': float(wasserstein_distance(
                    np.asarray(real.sum(axis=1)).ravel(), np.asarray(simulated.sum(axis=1)).ravel()))})
            if index % 24 == 0 or index == len(data):
                print(f'Benchmark metrics: {index}/{len(data)} targets', flush=True)
        additional = pd.DataFrame(records)
        additional.to_csv(cache, index=False)
    merged = data.merge(additional, on=['gap', 'target_slice'], how='left', validate='one_to_one')
    if not np.isfinite(merged[['cosine_divergence', 'library_size_wasserstein']]).all().all():
        raise ValueError('missing or non-finite additional benchmark scores')
    return merged


def configure_style():
    for filename in ('arial.ttf', 'arialbd.ttf'):
        font_manager.fontManager.addfont(Path(sys.prefix) / 'fonts' / filename)
    mpl.rcParams.update({
        'font.family': 'Arial', 'font.size': 11,
        'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.labelsize': 11,
        'axes.labelcolor': INK, 'text.color': INK, 'axes.edgecolor': '#626C76',
        'axes.linewidth': .85, 'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'xtick.color': INK, 'ytick.color': INK,
        'legend.fontsize': 10, 'legend.frameon': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    })


def style_axis(axis):
    axis.set_axisbelow(True)
    axis.grid(axis='y', color='#E5E8EC', linewidth=.65)
    axis.tick_params(length=3, width=.8)


def title(axis, letter, text):
    axis.set_title(text, loc='left', pad=12)
    axis.text(-.18, 1.055, letter, transform=axis.transAxes, fontsize=16,
              fontweight='bold', va='bottom', ha='left')


def metric_limits(axis, values, kind):
    if kind == 'benchmark_log':
        positive = values[values > 0]
        if not len(positive):
            raise ValueError('benchmark log-scale metric has no positive values')
        axis.set_yscale('log')
        axis.set_ylim(float(positive.min()) * .65, float(positive.max()) * 1.35)
    elif kind == 'benchmark_linear':
        lo, hi = float(values.min()), float(values.max())
        span = max(hi - lo, abs(hi) * .1, 1e-6)
        lower = max(-.02, lo - .1 * span) if lo >= 0 else lo - .1 * span
        upper = min(1.02, hi + .1 * span) if hi <= 1 else hi + .1 * span
        axis.set_ylim(lower, upper)
        axis.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 5, 10]))
    elif kind == 'error':
        axis.set_yscale('log')
        axis.set_ylim(min(10., float(values.min()) * .8), max(300., float(values.max()) * 1.1))
        axis.yaxis.set_major_locator(FixedLocator([10, 30, 100, 300]))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        axis.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    elif kind == 'spot':
        axis.set_ylim(min(0., float(values.min()) - .01), max(.24, float(values.max()) + .02))
        axis.yaxis.set_major_locator(mpl.ticker.MultipleLocator(.1))
    else:
        span = float(values.max() - values.min())
        margin = max(.015, span * .09)
        axis.set_ylim(max(-1.02, float(values.min()) - margin), min(1.02, float(values.max()) + margin))
        axis.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 5, 10]))


def distribution(axis, data, column, letter, label, ylabel, kind='correlation'):
    multiplier = 100. if kind == 'error' else 1.
    values = [data.loc[data.gap == gap, column].to_numpy(float) * multiplier for gap in GAPS]
    boxes = axis.boxplot(values, positions=np.arange(3), widths=.48, patch_artist=True,
        showfliers=False, medianprops={'color': INK, 'linewidth': 1.6},
        whiskerprops={'color': '#737B83', 'linewidth': .9},
        capprops={'color': '#737B83', 'linewidth': .9})
    rng = np.random.default_rng(2026)
    for position, (gap, observed, box) in enumerate(zip(GAPS, values, boxes['boxes'])):
        box.set_facecolor(mpl.colors.to_rgba(COLORS[gap], .15))
        box.set_edgecolor(COLORS[gap])
        box.set_linewidth(1.)
        axis.scatter(position + rng.uniform(-.15, .15, len(observed)), observed,
                     s=15, c=COLORS[gap], alpha=.56, linewidths=0, zorder=3)
    axis.set_xticks(np.arange(3), [str(g) for g in GAPS])
    axis.set_xlabel('Reference gap')
    axis.set_ylabel(ylabel)
    metric_limits(axis, np.concatenate(values), kind)
    style_axis(axis)
    title(axis, letter, label)


def save(figure, output, stem, dpi):
    for extension in ('pdf', 'svg', 'png'):
        metadata = ({'CreationDate': None, 'ModDate': None} if extension == 'pdf'
                    else {'Date': None} if extension == 'svg' else {})
        figure.savefig(output / f'{stem}.{extension}', dpi=dpi, metadata=metadata)
    plt.close(figure)


def reference_coverage(plan, inventory, output, dpi):
    # One quarter of the original overview width (11.6 / 4 = 2.9 inches).
    # Place row descriptions above their tracks so the short width stays readable.
    with mpl.rc_context({'font.size': 6, 'axes.labelsize': 6,
                         'xtick.labelsize': 5.5, 'axes.linewidth': .5}):
        fig = plt.figure(figsize=(2.9, 1.5))
        axis = fig.add_axes((.085, .26, .885, .55))
        z = inventory['z'].to_numpy(float)
        ids = inventory['slice_id'].to_numpy(int)
        for y, gap in zip((2.3, 1.3, .3), GAPS):
            density = next(d for d in plan['densities'].values() if int(d['spec']['gap']) == gap)
            mask = np.isin(ids, density['reference_pool'])
            axis.hlines(y, z.min(), z.max(), color='#E2E6EA', linewidth=.4, zorder=0)
            axis.scatter(z[~mask], np.full((~mask).sum(), y), s=1.3,
                         facecolors='white', edgecolors=GRAY, linewidths=.25, zorder=2)
            axis.scatter(z[mask], np.full(mask.sum(), y), s=13, marker='|',
                         c=COLORS[gap], linewidths=.65, zorder=3)
            axis.text(z.min(), y + .24,
                      f'Gap {gap}  ·  {mask.sum()} real / {(~mask).sum()} generated',
                      fontsize=6, va='bottom', color=INK)
        axis.set_yticks([])
        axis.spines['left'].set_visible(False)
        axis.set_ylim(-.05, 3.05)
        axis.set_xlim(z.min() - .08, z.max() + .08)
        axis.set_xticks([0, 4, 8, 12])
        axis.tick_params(axis='x', length=2, width=.5, pad=1.5)
        axis.set_xlabel('Slice z', labelpad=1)
        fig.text(.045, .965, 'A  Reference coverage', fontsize=8,
                 fontweight='bold', va='top')
        handles = [Line2D([], [], marker='|', color=INK, linestyle='none',
                          markersize=5, markeredgewidth=.7, label='Reference'),
                   Line2D([], [], marker='o', color=GRAY, markerfacecolor='white',
                          linestyle='none', markersize=2, label='Generated')]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.54, .005),
                   ncol=2, fontsize=5.5, handletextpad=.25, columnspacing=1.2,
                   handlelength=.8, borderpad=0, borderaxespad=.2)
        save(fig, output, 'study06_reference_coverage', dpi)


def overview(data, output, dpi):
    reference, _ = benchmark_definitions()
    fig = plt.figure(figsize=(11.6, 6.5))
    grid = fig.add_gridspec(2, 4, left=.085, right=.985, bottom=.105,
                          top=.875, hspace=.70, wspace=.36)
    fig.suptitle('Study 06  |  Local 3D reconstruction', x=.085, y=.985,
                 ha='left', fontsize=18, fontweight='bold')
    for index, spec in enumerate(reference.METRIC_SPECS):
        axis = fig.add_subplot(grid[index // 4, index % 4])
        kind = 'benchmark_log' if spec.get('yscale') == 'log' else 'benchmark_linear'
        distribution(axis, data, spec['column'], chr(ord('B') + index),
                     spec['label'], 'Metric value' if index % 4 == 0 else '', kind)
    save(fig, output, 'study06_overview', dpi)


def z_coherence(continuity, output, dpi):
    fig, axis = plt.subplots(figsize=(4.0, 3.3))
    fig.subplots_adjust(left=.20, right=.97, bottom=.20, top=.85)
    ordered = continuity.set_index('gap').loc[list(GAPS)]
    values = ordered.reconstructed_original_z_coherence_median.to_numpy(float)
    baseline = ordered.real_split_half_z_coherence_median.to_numpy(float)
    axis.plot(range(3), values, color='#C4CAD0', linewidth=1.3, zorder=1)
    axis.plot(range(3), baseline, color=GRAY, linestyle='--', linewidth=1.3, label='Real split-half')
    for position, gap, value in zip(range(3), GAPS, values):
        axis.scatter(position, value, s=68, color=COLORS[gap], zorder=3)
        axis.annotate(f'{value:.2f}', (position, value), xytext=(0, 8),
                      textcoords='offset points', ha='center', fontsize=11, fontweight='bold')
    axis.set_xticks(range(3), [str(g) for g in GAPS])
    axis.set_xlim(-.35, 2.35)
    axis.set_ylim(0, max(float(values.max()), float(baseline.max())) + .15)
    axis.set_xlabel('Reference gap')
    axis.set_ylabel('Median class–gene correlation')
    axis.text(.03, .94, '144 slices; real anchors included', transform=axis.transAxes,
              fontsize=9, color='#626C76', va='top')
    axis.legend(loc='lower left', fontsize=9)
    style_axis(axis)
    axis.set_title('FEAST full-stack z coherence ↑', loc='left', pad=12)
    save(fig, output, 'study06_z_coherence', dpi)


def z_profiles(data, output, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.5), sharex=True)
    fig.subplots_adjust(left=.085, right=.98, bottom=.09, top=.84, wspace=.27, hspace=.42)
    fig.suptitle('Reconstruction fidelity along z', x=.085, y=.98,
                 ha='left', fontsize=18, fontweight='bold')
    specs = [('input_mean_corr', 'Gene means', 'Pearson r (log1p mean)', 'correlation'),
             ('mean_per_gene_spot_pearson', 'Spotwise agreement', 'Mean gene-wise Pearson r', 'spot'),
             ('moran_i_correlation', 'Spatial autocorrelation', "Moran's I profile correlation", 'correlation'),
             ('relative_error_mean', 'Gene-mean error', 'Mean relative error (%)', 'error')]
    for axis, letter, (column, label, ylabel, kind) in zip(axes.flat, 'ABCD', specs):
        for gap, marker, linestyle in zip(GAPS, ('o', 's', '^'), ('-', '--', ':')):
            frame = data[data.gap == gap].sort_values('target_z')
            axis.plot(frame.target_z, frame[column] * (100 if kind == 'error' else 1),
                      color=COLORS[gap], marker=marker, markersize=2.3,
                      linewidth=.95, linestyle=linestyle, alpha=.83,
                      label=f'Gap {gap} ({len(frame)} targets)')
        metric_limits(axis, data[column].to_numpy(float) * (100 if kind == 'error' else 1), kind)
        axis.set_ylabel(ylabel)
        axis.set_xlabel('Target z coordinate')
        axis.tick_params(labelbottom=True)
        style_axis(axis)
        title(axis, letter, label)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.54, .94), ncol=3, columnspacing=2.)
    save(fig, output, 'study06_z_profiles', dpi)


def marker_maps(plan, input_root, output, dpi, spatialz_root=None, slice_id=None):
    import anndata as ad
    from plot_3d_gene_distribution import GENES, positive_norm

    shared = set.intersection(*(set(map(int, d['target_pool'])) for d in plan['densities'].values()))
    slice_id = min(shared) if slice_id is None else slice_id
    if slice_id not in shared:
        raise ValueError('marker-map slice must be held out in all density arms')
    name = f'Zhuang-ABCA-1.{slice_id:03d}'
    paths = [('Observed', 0, Path(plan['data_dir']) / f'{name}.h5ad')]
    for gap in GAPS:
        density_name = next(k for k, d in plan['densities'].items() if int(d['spec']['gap']) == gap)
        paths.append(('FEAST', gap, input_root / 'targets' / density_name / name / 'generated.h5ad'))
        if spatialz_root is not None:
            paths.append(('SpatialZ', gap, spatialz_root / 'targets' / density_name / name / 'generated.h5ad'))
    frames, coordinates, obs_names = [], None, None
    for method, gap, path in paths:
        data = ad.read_h5ad(path)
        xy = np.asarray(data.obsm['spatial'], dtype=float)
        if coordinates is None:
            coordinates, obs_names = xy.copy(), data.obs_names.copy()
        elif method == 'FEAST' and (not np.array_equal(coordinates, xy) or not obs_names.equals(data.obs_names)):
            raise ValueError('spatial gene maps require identical target positions and row order')
        indices = [data.var_names.get_loc(gene) for gene in GENES]
        expression = data.layers['counts'][:, indices]
        values = expression.toarray() if hasattr(expression, 'toarray') else np.asarray(expression)
        frame = pd.DataFrame({'method': method, 'gap': gap, 'target_slice': slice_id,
                              'spot_index': np.arange(data.n_obs), 'x': xy[:, 0], 'y': xy[:, 1]})
        for index, gene in enumerate(GENES):
            frame[gene] = values[:, index]
        frames.append(frame)
        del data
    joined = pd.concat(frames, ignore_index=True)
    stem = 'study06_marker_maps' if slice_id == min(shared) else f'study06_marker_maps_{slice_id:03d}'
    joined.to_csv(output / f'{stem}_plot_data.csv.gz', index=False)
    columns = len(paths)
    fig = plt.figure(figsize=(2.5 * columns + 1, 10.0))
    grid = fig.add_gridspec(4, columns + 1, width_ratios=[1] * columns + [.045],
        left=.055, right=.94, bottom=.055, top=.895, wspace=.08, hspace=.12)
    fig.suptitle(f'Observed and reconstructed expression  |  Slice {slice_id}',
                 x=.055, y=.98, ha='left', fontsize=18, fontweight='bold')
    fig.text(.055, .946, 'All cells displayed on each method’s native XY coordinates  ·  Shared axes and color scale within each gene',
             fontsize=10, color='#626C76')
    limits = [(joined[key].min(), joined[key].max()) for key in ('x', 'y')]
    upper_limits = {}
    for row, gene in enumerate(GENES):
        _, norm, upper = positive_norm(joined[gene].to_numpy(float))
        upper_limits[gene] = upper
        for column, ((method, gap, _), frame) in enumerate(zip(paths, frames)):
            axis = fig.add_subplot(grid[row, column])
            display = np.log1p(frame[gene].to_numpy(float))
            order = np.argsort(display, kind='stable')
            positive = order[display[order] > 0]
            axis.scatter(frame.x, frame.y, s=1.3, color='#D6DCE2', alpha=.45,
                         linewidths=0, rasterized=True)
            axis.scatter(frame.x.to_numpy()[positive], frame.y.to_numpy()[positive],
                         c=display[positive], s=1.6, cmap='viridis', norm=norm,
                         linewidths=0, alpha=.95, rasterized=True)
            for setter, (low, high) in zip((axis.set_xlim, axis.set_ylim), limits):
                margin = .025 * (high - low)
                setter(low - margin, high + margin)
            axis.set_aspect('equal')
            axis.set_axis_off()
            if row == 0:
                label = 'Observed' if gap == 0 else f'{method} · gap {gap}'
                axis.set_title(f'{label}\n{len(frame):,} cells',
                               color='#B64342' if method == 'SpatialZ' else INK, pad=9)
            if column == 0:
                axis.text(-.04, .5, gene, transform=axis.transAxes, rotation=90,
                          va='center', ha='right', fontsize=12, fontstyle='italic', fontweight='bold')
        color_axis = fig.add_subplot(grid[row, columns])
        bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=color_axis)
        bar.set_label('log1p counts', fontsize=10)
        bar.ax.tick_params(labelsize=9)
        bar.outline.set_linewidth(.6)
    fig.text(.5, .02, 'Gray points: zero counts. Color limits follow the existing pooled positive-count 99th-percentile display rule.',
             ha='center', fontsize=9, color='#626C76')
    save(fig, output, stem, dpi)
    return {'slice': slice_id, 'selection': 'common held-out slice selected by position',
            'genes': list(GENES), 'gene_selection_source': 'plot_3d_gene_distribution.py::GENES',
            'positions_displayed_per_panel': {f'{m}_gap{g}': len(f) for (m,g,_),f in zip(paths,frames)},
            'spatialz_root': str(spatialz_root) if spatialz_root is not None else None,
            'spot_subsampling': False,
            'color_transform': 'log1p raw counts', 'color_limits': upper_limits,
            'color_limit_rule': 'pooled positive-value 99th percentile per gene across all displayed columns'}


def comparison_z_profiles(frame, output, dpi):
    from plot_spatialz_comparison import METHODS, NAMES, COLORS as METHOD_COLORS
    specs = [('input_mean_corr', 'Gene-mean correlation ↑'),
             ('expression_wasserstein', 'Expression Wasserstein ↓'),
             ('residual_moran_profile_correlation', 'Residual Moran correlation ↑')]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex='col')
    fig.subplots_adjust(left=.07, right=.98, bottom=.09, top=.86, hspace=.35, wspace=.28)
    for row, (metric, label) in enumerate(specs):
        for col, gap in enumerate(GAPS):
            axis = axes[row, col]
            for method in METHODS:
                selected = frame[(frame.method == method) & (frame.gap == gap)].sort_values('target_z')
                selected = selected[np.isfinite(selected[metric])]
                if not len(selected):
                    continue
                axis.plot(selected.target_z, selected[metric], color=METHOD_COLORS[method],
                          linewidth=.9, alpha=.9, label=NAMES[method])
            if row == 0: axis.set_title(f'Gap {gap}')
            if col == 0: axis.set_ylabel(label)
            if row == 2: axis.set_xlabel('Target z coordinate')
            style_axis(axis)
    fig.suptitle('Reconstruction fidelity along z: FEAST, resampling and SpatialZ', fontsize=17, x=.07, ha='left', y=.98)
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc='upper center', ncol=3, bbox_to_anchor=(.5,.95))
    fig.text(.07, .02, 'Raw slice scores; no smoothing. Target sets differ between gaps. Resampling unconditional expression Wasserstein is not available.', fontsize=10)
    save(fig, output, 'study06_z_profiles', dpi)


def main():
    from plot_spatialz_comparison import load_comparison, summarize, figures, MAIN, CONDITIONAL, GEOMETRY
    parser = argparse.ArgumentParser(description='Study 06 figures including native SpatialZ and conditional resampling')
    parser.add_argument('--input-root', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--spatialz-root', type=Path, default=DEFAULT_SPATIALZ)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    _, continuity, provenance, plan = require_inputs(args.input_root)
    baseline = args.input_root / 'baselines/five_reference_conditional_resampling/method_metrics.csv'
    combined, conditional = load_comparison(args.spatialz_root / 'evaluation', baseline)
    summary = summarize(combined, [m for m,_ in MAIN] + CONDITIONAL + GEOMETRY + ['gene_zero_fraction_pearson'])
    inventory = pd.read_csv(args.input_root / 'input_manifest.csv').sort_values('z')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    configure_style()
    reference_coverage(plan, inventory, args.output_dir, args.dpi)
    figures(summary, conditional, args.output_dir, args.dpi)
    z_coherence(continuity, args.output_dir, args.dpi)
    comparison_z_profiles(combined, args.output_dir, args.dpi)
    maps = marker_maps(plan, args.input_root, args.output_dir, args.dpi, args.spatialz_root)
    combined.to_csv(args.output_dir / 'target_plot_data.csv', index=False)
    summary.to_csv(args.output_dir / 'summary.csv', index=False)
    conditional.to_csv(args.output_dir / 'conditional_common_targets.csv', index=False)
    continuity.to_csv(args.output_dir / 'feast_continuity_plot_data.csv', index=False)
    record = {'input_root': str(args.input_root.resolve()), 'spatialz_root': str(args.spatialz_root.resolve()),
              'metrics_recomputed': False, 'methods': ['FEAST','resampling','SpatialZ'],
              'conditional_population': 'identical supported targets within each gap',
              'continuity_population': 'FEAST only; all 144 positions including retained real anchors',
              'marker_maps': maps, 'font': 'Arial', 'png_dpi': args.dpi}
    (args.output_dir / 'figure_provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    (args.output_dir / 'CAPTIONS.md').write_text(
        '# Study 06: FEAST, resampling and SpatialZ\n\n'
        '**Overview.** Existing slice metrics for all 336 targets; medians and interquartile ranges. '
        'Resampling is averaged over its ten repeats within each slice. No cellwise matching is used for SpatialZ. '
        'Conditional scores use only common supported slices within each gap; the displayed counts specify their coverage.\n\n'
        '**Along-z profiles.** Raw scores connected in actual z order, without smoothing. '
        'Reference-density arms have different target sets. No inferential tests treat slices as biological replicates.\n\n'
        '**Marker maps.** The existing four genes on the first common held-out slice, all cells displayed. '
        'Observed, FEAST and SpatialZ use their own coordinates and shared axes/color scales. '
        'Log1p counts saturate only in display at the pooled positive 99th percentile.\n\n'
        '**FEAST z coherence.** Retained as a clearly labeled FEAST-only panel because the completed SpatialZ evaluation '
        'does not contain a matched whole-stack continuity metric. Paired cellwise metrics are absent from the three-method overview.\n')
    print(f'Wrote three-method Study 06 figures to {args.output_dir}')


if __name__ == '__main__':
    main()
