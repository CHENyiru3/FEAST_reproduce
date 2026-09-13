"""Build the validated Study 05 publication figure and plotting data."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path
import tempfile
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
VIS_DIR = Path(__file__).resolve().parent
REPRO_ROOT = VIS_DIR.parents[1]
STUDY_DIR = REPRO_ROOT / '05_2d_conditional_transfer'
COLORS = {'DLPFC cross-slice, within donor': '#0F4D92', 'DLPFC cross-slice, cross donor': '#9A4D8E', 'DLPFC half-slice': '#D88F8A', 'MERFISH cross-slice': '#42949E', 'MERFISH half-slice': '#D9A441'}
MARKERS = {'DLPFC cross-slice, within donor': 'o', 'DLPFC cross-slice, cross donor': 's', 'DLPFC half-slice': '^', 'MERFISH cross-slice': 'D', 'MERFISH half-slice': 'v'}
GROUP_ORDER = list(COLORS)

def configure_matplotlib() -> None:
    """Apply the FEAST manuscript style before creating any axes."""
    arial_paths = (Path(sys.prefix) / 'fonts' / 'arial.ttf', Path(sys.prefix) / 'fonts' / 'arialbd.ttf')
    if not all((path.is_file() for path in arial_paths)):
        raise FileNotFoundError(f'Arial Regular and Bold are required for this figure: {arial_paths}')
    for path in arial_paths:
        font_manager.fontManager.addfont(path)
    mpl.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold', 'axes.labelsize': 8.5, 'axes.labelcolor': '#222222', 'axes.edgecolor': '#555555', 'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'xtick.color': '#333333', 'ytick.color': '#333333', 'legend.fontsize': 7.5, 'legend.frameon': False, 'text.color': '#222222', 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'feast-study05'})

def style_axis(ax: plt.Axes, *, grid_axis: str) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color='#E3E3E3', linewidth=0.5, alpha=0.75)
    ax.tick_params(length=3, width=0.7, color='#555555')

def group_label(row: pd.Series) -> str:
    if row['dataset'] == 'dlpfc' and row['mode'] == 'cross_slice':
        suffix = 'within donor' if row['donor_stratum'] == 'within_donor' else 'cross donor'
        return f'DLPFC cross-slice, {suffix}'
    if row['dataset'] == 'dlpfc':
        return 'DLPFC half-slice'
    if row['mode'] == 'cross_slice':
        return 'MERFISH cross-slice'
    return 'MERFISH half-slice'

def short_direction(row: pd.Series) -> str:
    direction = str(row['direction'])
    if row['dataset'] == 'merfish':
        direction = direction.replace('Zhuang-ABCA-1.', '')
    return direction.replace('_low_x_to_high_x', ' half').replace('_to_', '→')

def verify_inputs() -> tuple[pd.DataFrame, dict[str, Path], dict]:
    paths = {'summary': STUDY_DIR / 'outputs' / 'scores' / 'summary.csv', 'score_provenance': STUDY_DIR / 'outputs' / 'scores' / 'provenance.json', 'validation': STUDY_DIR / 'outputs' / 'study05_complete_validation.json', 'comparison': STUDY_DIR / 'outputs' / 'old_vs_new_metrics.csv', 'comparison_provenance': STUDY_DIR / 'outputs' / 'old_vs_new_metrics_provenance.json', 'decision': STUDY_DIR / 'PUBLICATION_DECISION.md'}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing Study 05 evidence: {missing}')
    score_provenance = json.loads(paths['score_provenance'].read_text(encoding='utf-8'))
    validation = json.loads(paths['validation'].read_text(encoding='utf-8'))
    comparison_provenance = json.loads(paths['comparison_provenance'].read_text(encoding='utf-8'))
    if not (score_provenance.get('status') == 'ok' and validation.get('status') == 'ok' and (int(validation.get('validated_jobs', -1)) == 45) and (validation.get('scores', {}).get('valid') is True) and (comparison_provenance.get('status') == 'ok')):
        raise RuntimeError('Study 05 evidence or validation status does not agree')
    summary = pd.read_csv(paths['summary'], dtype={'source': str, 'target': str})
    if not (len(summary) == 45 and summary['job_id'].is_unique and (int(summary['primary_setting'].sum()) == 13) and (set(summary['donor_stratum']) == {'within_donor', 'cross_donor', 'within_slice', 'donor_not_declared'})):
        raise RuntimeError('Study 05 summary is not the declared 45-row stratified table')
    return (summary, paths, validation)

def prepare_plotting_data(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = summary[summary['primary_setting']].copy()
    primary['group'] = primary.apply(group_label, axis=1)
    primary['direction_label'] = primary.apply(short_direction, axis=1)
    primary['zero_fidelity'] = 1.0 - primary['zero_ks']
    primary['reference_observed_fraction'] = primary['n_reference_observed_genes'] / primary['n_genes']
    primary['evaluation_scope'] = np.where(primary['target_retained_fraction'] < 0.999999, 'filtered_supported_target', 'complete_target')
    sensitivity = summary[summary['mode'] == 'cross_slice'].copy()
    sensitivity['group'] = sensitivity.apply(group_label, axis=1)
    sensitivity['direction_label'] = sensitivity.apply(short_direction, axis=1)
    ar_summary = sensitivity.groupby(['group', 'assignment_randomness'], sort=False).agg(moran_corr_mean=('moran_corr', 'mean'), moran_corr_min=('moran_corr', 'min'), moran_corr_max=('moran_corr', 'max'), direction_count=('direction', 'size')).reset_index()
    support = primary[['job_id', 'dataset', 'mode', 'group', 'direction', 'direction_label', 'donor_stratum', 'target_retained_fraction', 'reference_observed_fraction', 'n_target_spots', 'n_genes', 'n_reference_observed_genes', 'evaluation_scope']].copy()
    return (primary, ar_summary, support)

def plot_fidelity(ax: plt.Axes, primary: pd.DataFrame) -> None:
    metrics = ['mean_corr', 'var_corr', 'moran_corr', 'zero_fidelity']
    labels = ['Mean\nprofile r', 'Variance\nprofile r', 'Moran I\nprofile r', '1 − zero\nKS']
    offsets = np.linspace(-0.24, 0.24, len(GROUP_ORDER))
    for offset, group in zip(offsets, GROUP_ORDER):
        rows = primary[primary['group'] == group]
        for index, metric in enumerate(metrics):
            values = rows[metric].to_numpy(float)
            jitter = np.linspace(-0.035, 0.035, len(values)) if len(values) > 1 else np.zeros(1)
            if len(values) > 1:
                ax.plot([index + offset, index + offset], [float(np.min(values)), float(np.max(values))], color=COLORS[group], linewidth=0.8, alpha=0.45, zorder=1)
            ax.scatter(index + offset + jitter, values, s=15, marker=MARKERS[group], facecolor='white', edgecolor=COLORS[group], linewidth=0.8, alpha=0.9, zorder=2)
            ax.scatter(index + offset, float(np.mean(values)), s=38, marker=MARKERS[group], color=COLORS[group], edgecolor='white', linewidth=0.6, zorder=3)
    ax.set_xticks(range(len(metrics)), labels)
    ax.set_xlim(-0.48, len(metrics) - 0.52)
    ax.set_ylim(0.7, 1.007)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel('Fidelity  ↑')
    ax.set_title('A  Aggregate profile fidelity', loc='left', pad=7)
    ax.text(0.02, 0.035, 'Panels A–B: open = direction/slice; filled = group mean', transform=ax.transAxes, fontsize=6.5, color='#666666')
    style_axis(ax, grid_axis='y')

def plot_spotwise(ax: plt.Axes, primary: pd.DataFrame) -> None:
    metrics = ['median_gene_pearson', 'median_gene_spearman']
    labels = ['Median gene\nPearson', 'Median gene\nSpearman']
    offsets = np.linspace(-0.24, 0.24, len(GROUP_ORDER))
    for offset, group in zip(offsets, GROUP_ORDER):
        rows = primary[primary['group'] == group]
        for index, metric in enumerate(metrics):
            values = rows[metric].to_numpy(float)
            jitter = np.linspace(-0.035, 0.035, len(values)) if len(values) > 1 else np.zeros(1)
            ax.scatter(index + offset + jitter, values, s=15, marker=MARKERS[group], facecolor='white', edgecolor=COLORS[group], linewidth=0.8, alpha=0.9)
            ax.scatter(index + offset, float(np.mean(values)), s=38, marker=MARKERS[group], color=COLORS[group], edgecolor='white', linewidth=0.6, zorder=3)
    ax.axhline(0.0, color='#555555', linewidth=0.8, linestyle='--')
    ax.set_xticks(range(len(metrics)), labels)
    ax.set_xlim(-0.42, len(metrics) - 0.58)
    ax.set_ylim(-0.01, 0.055)
    ax.set_yticks([-0.01, 0.0, 0.02, 0.04])
    ax.set_ylabel('Median per-gene correlation')
    ax.set_title('B  Spot-level correspondence', loc='left', pad=7)
    ax.text(0.02, 0.94, 'DLPFC medians cluster near zero', transform=ax.transAxes, ha='left', va='top', fontsize=6.8, color='#666666')
    style_axis(ax, grid_axis='y')

def plot_ar(ax: plt.Axes, ar_summary: pd.DataFrame) -> None:
    groups = ['DLPFC cross-slice, within donor', 'DLPFC cross-slice, cross donor', 'MERFISH cross-slice']
    for group in groups:
        rows = ar_summary[ar_summary['group'] == group].sort_values('assignment_randomness')
        x = rows['assignment_randomness'].to_numpy(float)
        y = rows['moran_corr_mean'].to_numpy(float)
        ax.fill_between(x, rows['moran_corr_min'].to_numpy(float), rows['moran_corr_max'].to_numpy(float), color=COLORS[group], alpha=0.15, linewidth=0)
        ax.plot(x, y, color=COLORS[group], marker=MARKERS[group], markersize=4.5, linewidth=1.7, label=group)
    ax.axvline(0.3, color='#555555', linewidth=0.8, linestyle='--')
    ax.text(0.307, 0.575, 'primary', rotation=90, ha='left', va='bottom', fontsize=6.8, color='#555555')
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.5])
    ax.set_xlabel('Assignment randomness')
    ax.set_ylabel('Moran profile correlation')
    ax.set_ylim(0.55, 0.94)
    ax.set_title('C  Spatial sensitivity to randomness', loc='left', pad=7)
    ax.text(0.98, 0.035, 'band = min–max across directions', transform=ax.transAxes, ha='right', fontsize=6.5, color='#666666')
    style_axis(ax, grid_axis='both')

def plot_support(ax: plt.Axes, support: pd.DataFrame) -> None:
    order = support.copy()
    order['group'] = pd.Categorical(order['group'], categories=GROUP_ORDER, ordered=True)
    order = order.sort_values(['group', 'direction']).reset_index(drop=True)
    y_positions = []
    separators = []
    cursor = 0.0
    previous_group = None
    for group in order['group'].astype(str):
        if previous_group is not None and group != previous_group:
            cursor += 0.55
            separators.append(cursor - 0.275)
        y_positions.append(cursor)
        cursor += 1.0
        previous_group = group
    y = np.asarray(y_positions)
    for index, row in order.iterrows():
        ax.plot([row['target_retained_fraction'], row['reference_observed_fraction']], [y[index], y[index]], color='#CFCFCF', linewidth=0.8, zorder=1)
    ax.scatter(order['target_retained_fraction'], y, s=24, color='#333333', marker='o', label='Retained target spots', zorder=3)
    ax.scatter(order['reference_observed_fraction'], y, s=24, facecolor='white', edgecolor='#B64342', linewidth=1.1, marker='s', label='Globally observed reference genes', zorder=3)
    for separator in separators:
        ax.axhline(separator, color='#E6E6E6', linewidth=0.6, zorder=0)
    ax.set_yticks(y, order['direction_label'].tolist(), fontsize=6.7)
    ax.set_xlim(0.8, 1.005)
    ax.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
    ax.set_xlabel('Fraction of declared support')
    ax.set_title('D  Evaluation support', loc='left', pad=7)
    ax.set_ylim(-0.65, y.max() + 0.65)
    ax.invert_yaxis()
    ax.legend(loc='lower left', fontsize=6.5, handletextpad=0.5, borderaxespad=0.25)
    style_axis(ax, grid_axis='x')

def build_figure(primary: pd.DataFrame, ar_summary: pd.DataFrame, support: pd.DataFrame):
    configure_matplotlib()
    fig = plt.figure(figsize=(7.4, 5.8))
    grid = fig.add_gridspec(3, 2, height_ratios=[0.16, 1.0, 1.0], left=0.095, right=0.985, bottom=0.085, top=0.985, wspace=0.34, hspace=0.48)
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.set_axis_off()
    axes = np.asarray([[fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])], [fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])]])
    plot_fidelity(axes[0, 0], primary)
    plot_spotwise(axes[0, 1], primary)
    plot_ar(axes[1, 0], ar_summary)
    plot_support(axes[1, 1], support)
    handles = [plt.Line2D([], [], color=COLORS[group], marker=MARKERS[group], linestyle='none', markersize=6, label=group) for group in GROUP_ORDER]
    legend_ax.legend(handles=handles, loc='center', ncol=3, columnspacing=1.4, handletextpad=0.45, fontsize=7.3)
    return fig

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=VIS_DIR / 'figures')
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    summary, input_paths, validation = verify_inputs()
    primary, ar_summary, support = prepare_plotting_data(summary)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f'.{args.output_dir.name}-fidelity-and-limits-', dir=args.output_dir.parent))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        primary.to_csv(work_dir / 'primary_metrics.csv', index=False)
        ar_summary.to_csv(work_dir / 'ar_sensitivity.csv', index=False)
        support.to_csv(work_dir / 'support_and_panel.csv', index=False)
        figure = build_figure(primary, ar_summary, support)
        stem = 'conditional_transfer_fidelity_and_limits'
        figure.savefig(work_dir / f'{stem}.pdf', bbox_inches='tight', metadata={'Creator': 'FEAST Study 05 plotting script', 'CreationDate': None, 'ModDate': None})
        figure.savefig(work_dir / f'{stem}.svg', bbox_inches='tight', metadata={'Creator': 'FEAST Study 05 plotting script', 'Date': None})
        figure.savefig(work_dir / f'{stem}.png', dpi=int(args.dpi), bbox_inches='tight', metadata={'Software': 'FEAST Study 05 plotting script'})
        plt.close(figure)
        outputs = {path.name: None for path in sorted(work_dir.iterdir()) if path.is_file()}
        provenance = {'status': 'ok', 'configuration_id': 'study05-conditional-transfer-figure-v2', 'scientific_disposition': 'validated_evidence_claim_reframe_required', 'figure_promotion_authorized': False, 'publication_claim_authorized': False, 'primary_assignment_randomness': 0.3, 'png_dpi': int(args.dpi), 'rows': {'score_summary': len(summary), 'primary': len(primary), 'ar_summary': len(ar_summary), 'support': len(support)}, 'claim_limits': ['conditional on observed target XY and labels', 'aggregate distribution/spatial-statistic fidelity is distinct from spotwise recovery', 'DLPFC spotwise median gene correlations are near zero', '151670 outgoing directions evaluate only supported target labels', 'MERFISH donor identity is not declared'], 'vector_text': {'pdf_fonttype': 42, 'svg_fonttype': 'none', 'adobe_illustrator_editable_text': True, 'deterministic_export_metadata': True}, 'outputs': outputs, 'created_at': datetime.now(timezone.utc).isoformat()}
        (work_dir / 'figure_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
        published = {}
        for path in sorted(work_dir.iterdir()):
            if path.is_file():
                destination = args.output_dir / path.name
                os.replace(path, destination)
                published[destination.name] = None
        print(json.dumps({'status': 'ok', 'output_dir': str(args.output_dir), 'outputs': published}, indent=2))
    finally:
        if work_dir.exists():
            work_dir.rmdir()
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
