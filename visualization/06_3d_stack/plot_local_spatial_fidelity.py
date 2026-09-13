"""Render Study 06 local spatial-pattern fidelity diagnostics."""
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

REPRO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPRO_ROOT / '06_3d_stack' / 'outputs' / 'reference_density_rank_v1'
OUTPUT = Path(__file__).resolve().parent / 'figures'
GAPS = (3, 5, 10)
COLORS = {3: '#0072B2', 5: '#E69F00', 10: '#009E73'}
LABELS: dict[int, str] = {}
METRICS = (
    ('mean_per_gene_spot_pearson', 'A  Mean gene spatial Pearson ↑', 'D  Mean spatial Pearson vs z'),
    ('median_per_gene_spot_pearson', 'B  Median gene spatial Pearson ↑', 'E  Median spatial Pearson vs z'),
    ('moran_i_correlation', 'C  Moran I Corr ↑', 'F  Moran I Corr vs z'),
)


def configure_matplotlib() -> None:
    arial_paths = (Path(sys.prefix) / 'fonts' / 'arial.ttf', Path(sys.prefix) / 'fonts' / 'arialbd.ttf')
    if not all((path.is_file() for path in arial_paths)):
        raise FileNotFoundError(f'Arial Regular and Bold are required for this figure: {arial_paths}')
    for path in arial_paths:
        font_manager.fontManager.addfont(path)
    mpl.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold', 'axes.labelsize': 8.5, 'axes.edgecolor': '#555555', 'axes.linewidth': 0.8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.0, 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis='y', color='#E3E3E3', linewidth=0.5, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.tick_params(length=3, width=0.7, color='#555555')


def require_inputs(input_root: Path) -> tuple[pd.DataFrame, dict, dict]:
    validation = json.loads((input_root / 'validation_summary.json').read_text(encoding='utf-8'))
    if validation.get('status') != 'passed' or validation.get('validated_outputs') != 336:
        raise ValueError('complete Study 06 validation is required')
    provenance = json.loads((input_root / 'evaluation' / 'score_provenance.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(input_root / 'evaluation' / 'target_metrics.csv')
    required = {name for name, _, _ in METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'missing local spatial metrics: {missing}')
    plan = json.loads((input_root / 'plan.json').read_text(encoding='utf-8'))
    return frame, provenance, plan


def limits(values: np.ndarray) -> tuple[float, float]:
    lower, upper = (float(np.nanmin(values)), float(np.nanmax(values)))
    margin = max(0.02, (upper - lower) * 0.08)
    return (max(-1.0, lower - margin), min(1.0, upper + margin))


def box_panel(axis: plt.Axes, frame: pd.DataFrame, column: str, title: str, ylim: tuple[float, float]) -> None:
    values = [frame.loc[frame['gap'] == gap, column].dropna().to_numpy() for gap in GAPS]
    boxes = axis.boxplot(values, positions=np.arange(len(GAPS)), widths=0.52, showfliers=False, patch_artist=True, medianprops={'color': '#222222', 'linewidth': 1.1})
    for patch, gap in zip(boxes['boxes'], GAPS):
        patch.set_facecolor(COLORS[gap])
        patch.set_alpha(0.18)
        patch.set_edgecolor(COLORS[gap])
    for position, gap in enumerate(GAPS):
        subset = frame.loc[frame['gap'] == gap].sort_values('target_z')
        offsets = np.linspace(-0.16, 0.16, len(subset))
        axis.scatter(position + offsets, subset[column], s=10, color=COLORS[gap], alpha=0.8, linewidths=0, zorder=3)
    axis.set_title(title, loc='left', pad=6)
    axis.set_ylabel('Metric value')
    axis.set_xlabel('Reference gap')
    axis.set_xticks(np.arange(len(GAPS)), [str(gap) for gap in GAPS])
    axis.set_ylim(*ylim)
    style_axis(axis)


def trajectory_panel(axis: plt.Axes, frame: pd.DataFrame, column: str, title: str, ylim: tuple[float, float]) -> None:
    for gap in GAPS:
        subset = frame.loc[frame['gap'] == gap].sort_values('target_z')
        axis.plot(subset['target_z'], subset[column], marker='o', markersize=2.6, linewidth=0.9, color=COLORS[gap], alpha=0.85, label=LABELS[gap])
    axis.set_title(title, loc='left', pad=6)
    axis.set_ylabel('Metric value')
    axis.set_xlabel('Target z')
    axis.set_ylim(*ylim)
    style_axis(axis)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = output / 'conditional_stack_local_spatial_fidelity'
    paths = [stem.with_suffix(suffix) for suffix in ('.pdf', '.svg', '.png')]
    paths += [output / 'local_spatial_fidelity_plot_data.csv', output / 'local_spatial_fidelity_provenance.json']
    if any(path.exists() for path in paths):
        raise FileExistsError('fresh-only local spatial-fidelity outputs already exist')
    frame, score_provenance, plan = require_inputs(input_root)
    LABELS.update({int(density['spec']['gap']): f"Gap {int(density['spec']['gap'])} ({len(density['reference_pool'])} real / {len(density['target_pool'])} simulated)" for density in plan['densities'].values()})
    frame.to_csv(output / 'local_spatial_fidelity_plot_data.csv', index=False)
    configure_matplotlib()
    figure = plt.figure(figsize=(8.5, 5.7))
    grid = figure.add_gridspec(3, 3, height_ratios=[0.14, 1.0, 1.0], left=0.075, right=0.99, bottom=0.085, top=0.98, hspace=0.46, wspace=0.30)
    legend_axis = figure.add_subplot(grid[0, :])
    legend_axis.set_axis_off()
    axes = np.asarray([[figure.add_subplot(grid[row, column]) for column in range(3)] for row in (1, 2)])
    for column_index, (column, title, z_title) in enumerate(METRICS):
        ylim = limits(frame[column].to_numpy(float))
        box_panel(axes[0, column_index], frame, column, title, ylim)
        trajectory_panel(axes[1, column_index], frame, column, z_title, ylim)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    legend_axis.legend(handles, labels, loc='center', ncol=3, columnspacing=1.25, handletextpad=0.4)
    for suffix, kwargs in {'.pdf': {'metadata': {'Title': 'FEAST Study 06 local spatial fidelity', 'Creator': 'FEAST publication visualization', 'CreationDate': None, 'ModDate': None}}, '.svg': {'metadata': {'Title': 'FEAST Study 06 local spatial fidelity', 'Creator': 'FEAST publication visualization', 'Date': None}}, '.png': {'dpi': 600, 'metadata': {'Software': 'FEAST Study 06 plotting script'}}}.items():
        figure.savefig(stem.with_suffix(suffix), **kwargs)
    plt.close(figure)
    record = {'schema_version': 1, 'study': '06_3d_stack', 'configuration_id': score_provenance['configuration_id'], 'figure_promotion_authorized': False, 'publication_claim_authorized': False, 'interpretation': 'descriptive local spatial-pattern fidelity across corrected simulated targets; no biological-replicate inference', 'metrics': [name for name, _, _ in METRICS], 'inputs': {'target_metrics': str(input_root / 'evaluation' / 'target_metrics.csv'), 'score_provenance': str(input_root / 'evaluation' / 'score_provenance.json')}, 'outputs': {path.name: None for path in paths[:-1]}, 'editable_text': {'pdf_fonttype': 42, 'svg_fonttype': 'none'}, 'png_dpi': 600}
    (output / 'local_spatial_fidelity_provenance.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
