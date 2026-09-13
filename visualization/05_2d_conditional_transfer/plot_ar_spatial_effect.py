"""Render best-case cross-slice gene effects across assignment randomness."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path
import tempfile
import anndata as ad
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import sparse
VIS_DIR = Path(__file__).resolve().parent
REPRO_ROOT = VIS_DIR.parents[1]
STUDY_DIR = REPRO_ROOT / '05_2d_conditional_transfer'
AR_VALUES = (0.0, 0.1, 0.2, 0.3, 0.5)
DATASET_STYLE = {'dlpfc': {'label': 'DLPFC', 'annotation_key': 'ground_truth', 'point_size': 4.0}, 'merfish': {'label': 'MERFISH', 'annotation_key': 'class', 'point_size': 0.75}}
EXPRESSION_CMAP = LinearSegmentedColormap.from_list('feast_expression', ('#F2F2F2', '#B7D7E8', '#42949E', '#0F4D92'))

def configure_matplotlib() -> None:
    arial_paths = (Path(sys.prefix) / 'fonts' / 'arial.ttf', Path(sys.prefix) / 'fonts' / 'arialbd.ttf')
    if not all((path.is_file() for path in arial_paths)):
        raise FileNotFoundError(f'Arial Regular and Bold are required for this figure: {arial_paths}')
    for path in arial_paths:
        font_manager.fontManager.addfont(path)
    mpl.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.titlesize': 8.2, 'axes.titleweight': 'bold', 'axes.labelsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.75, 'legend.frameon': False, 'text.color': '#222222', 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'feast-study05'})

def matrix_column(matrix, index: int) -> np.ndarray:
    values = matrix[:, index]
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float64).reshape(-1)

def ar_dir_name(value: float) -> str:
    return f'ar_{value:.1f}'.replace('.', '_')

def generated_path(dataset: str, direction: str, ar_value: float) -> Path:
    return STUDY_DIR / 'outputs' / 'final' / 'cross_slice' / dataset / direction / ar_dir_name(ar_value) / 'generated.h5ad'

def input_path(dataset: str, slice_id: str) -> Path:
    return STUDY_DIR / 'data' / 'local' / dataset / f'{slice_id}.h5ad'

def verify_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    paths = {'input_manifest': STUDY_DIR / 'data' / 'input_checksums.csv', 'summary': STUDY_DIR / 'outputs' / 'scores' / 'summary.csv', 'per_gene_metrics': STUDY_DIR / 'outputs' / 'scores' / 'per_gene_metrics.csv', 'score_provenance': STUDY_DIR / 'outputs' / 'scores' / 'provenance.json'}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing Study 05 evidence: {missing}')
    provenance = json.loads(paths['score_provenance'].read_text(encoding='utf-8'))
    if provenance.get('status') != 'ok':
        raise RuntimeError('Study 05 score provenance does not validate the inputs')
    summary = pd.read_csv(paths['summary'], dtype={'source': str, 'target': str})
    per_gene = pd.read_csv(paths['per_gene_metrics'], dtype={'source': str, 'target': str}, low_memory=False)
    manifest = pd.read_csv(paths['input_manifest'], dtype={'slice_id': str})
    return (summary, per_gene, provenance, manifest)

def choose_best_case(dataset: str, summary: pd.DataFrame, per_gene: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    cross = summary[(summary['dataset'] == dataset) & (summary['mode'] == 'cross_slice')]
    ar_means = cross.groupby('assignment_randomness', as_index=False).agg(mean_moran_profile=('moran_corr', 'mean')).sort_values(['mean_moran_profile', 'assignment_randomness'], ascending=[False, True])
    best_ar = float(ar_means.iloc[0]['assignment_randomness'])
    best_direction_row = cross[cross['assignment_randomness'] == best_ar].sort_values(['moran_corr', 'direction'], ascending=[False, True]).iloc[0]
    direction = str(best_direction_row['direction'])
    best_job_id = str(best_direction_row['job_id'])
    genes = per_gene[(per_gene['job_id'] == best_job_id) & per_gene['reference_observed'] & per_gene['target_zero_prop'].between(0.05, 0.95) & np.isfinite(per_gene['pearson']) & np.isfinite(per_gene['target_moran_i'])].copy()
    moran_cutoff = float(genes['target_moran_i'].quantile(0.95))
    spatial_genes = genes[genes['target_moran_i'] >= moran_cutoff].copy()
    selected = spatial_genes.sort_values(['pearson', 'target_moran_i', 'gene'], ascending=[False, False, True]).iloc[0]
    gene = str(selected['gene'])
    gene_trajectory = per_gene[(per_gene['dataset'] == dataset) & (per_gene['mode'] == 'cross_slice') & (per_gene['direction'] == direction) & (per_gene['gene'] == gene)].sort_values('assignment_randomness')
    if not np.array_equal(gene_trajectory['assignment_randomness'].to_numpy(float), np.asarray(AR_VALUES)):
        raise RuntimeError(f'incomplete AR trajectory for {dataset} {direction} {gene}')
    case = {'dataset': dataset, 'dataset_label': DATASET_STYLE[dataset]['label'], 'annotation_key': DATASET_STYLE[dataset]['annotation_key'], 'point_size': DATASET_STYLE[dataset]['point_size'], 'source': str(best_direction_row['source']), 'target': str(best_direction_row['target']), 'direction': direction, 'best_ar': best_ar, 'best_direction_moran_profile': float(best_direction_row['moran_corr']), 'best_ar_dataset_mean_moran_profile': float(ar_means.iloc[0]['mean_moran_profile']), 'gene': gene, 'gene_selection_moran_quantile': 0.95, 'gene_selection_moran_cutoff': moran_cutoff, 'gene_selection_candidate_count': int(len(spatial_genes))}
    return (case, gene_trajectory)

def verify_case_files(case: dict, trajectory: pd.DataFrame, score_provenance: dict, input_manifest: pd.DataFrame) -> dict[float, dict]:
    input_row = input_manifest[(input_manifest['dataset'] == case['dataset']) & (input_manifest['slice_id'] == case['target'])]
    if len(input_row) != 1:
        raise RuntimeError(f"target input is not unique for {case['target']}")
    run_support = {}
    for _, row in trajectory.iterrows():
        ar_value = float(row['assignment_randomness'])
        job_id = str(row['job_id'])
        generation_record = score_provenance['generation_inputs'][job_id]
        path = generated_path(case['dataset'], case['direction'], ar_value)
        run_provenance_path = path.with_name('provenance.json')
        run_provenance = json.loads(run_provenance_path.read_text(encoding='utf-8'))
        if not (run_provenance.get('status') == 'ok' and run_provenance.get('target_expression_loaded_by_generation') is False and (int(run_provenance.get('support', {}).get('target_retained_spots', 0)) > 0)):
            raise RuntimeError(f'invalid generation support for {job_id}')
        run_support[ar_value] = run_provenance['support']
    return run_support

def load_case(case: dict, trajectory: pd.DataFrame, run_support: dict[float, dict]) -> dict:
    target = ad.read_h5ad(input_path(case['dataset'], case['target']))
    gene = case['gene']
    try:
        if gene not in target.var_names:
            raise RuntimeError(f'selected gene {gene} is absent from target input')
        coordinates = np.asarray(target.obsm['spatial'], dtype=np.float64)[:, :2]
        labels = target.obs[case['annotation_key']].astype(str).to_numpy()
        truth_all = matrix_column(target.layers['counts'], target.var_names.get_loc(gene))
        generated_log = {}
        target_indices = None
        target_coordinates = None
        eligible_labels = None
        for ar_value in AR_VALUES:
            generated = ad.read_h5ad(generated_path(case['dataset'], case['direction'], ar_value))
            try:
                if gene not in generated.var_names:
                    raise RuntimeError(f'selected gene {gene} is absent at AR {ar_value}')
                indices = target.obs_names.get_indexer(generated.obs_names)
                if np.any(indices < 0):
                    raise RuntimeError(f'generated identities are absent at AR {ar_value}')
                generated_coordinates = np.asarray(generated.obsm['spatial'], dtype=np.float64)[:, :2]
                np.testing.assert_allclose(coordinates[indices], generated_coordinates, rtol=0.0, atol=0.0)
                support_labels = sorted(map(str, run_support[ar_value]['eligible_labels']))
                if target_indices is None:
                    target_indices = indices
                    target_coordinates = generated_coordinates
                    eligible_labels = support_labels
                elif not (np.array_equal(target_indices, indices) and eligible_labels == support_labels):
                    raise RuntimeError('target support changes across AR')
                values = matrix_column(generated.layers['counts'], generated.var_names.get_loc(gene))
                generated_log[ar_value] = np.log1p(values)
            finally:
                generated.file.close() if generated.isbacked else None
    finally:
        target.file.close() if target.isbacked else None
    truth_log = np.log1p(truth_all[target_indices])
    expression_limit = float(np.quantile(np.concatenate([truth_log, *[generated_log[value] for value in AR_VALUES]]), 0.99))
    expression_limit = max(expression_limit, float(np.log(2.0)))
    retained_mask = np.zeros(len(coordinates), dtype=bool)
    retained_mask[target_indices] = True
    return {**case, 'coordinates': coordinates, 'labels': labels, 'target_indices': target_indices, 'target_coordinates': target_coordinates, 'retained_mask': retained_mask, 'eligible_labels': eligible_labels, 'truth_log': truth_log, 'generated_log': generated_log, 'expression_limit': expression_limit, 'trajectory': trajectory.reset_index(drop=True)}

def setup_map_axis(ax: plt.Axes, coordinates: np.ndarray) -> None:
    x_pad = 0.035 * float(np.ptp(coordinates[:, 0]))
    y_pad = 0.035 * float(np.ptp(coordinates[:, 1]))
    ax.set_xlim(coordinates[:, 0].min() - x_pad, coordinates[:, 0].max() + x_pad)
    ax.set_ylim(coordinates[:, 1].min() - y_pad, coordinates[:, 1].max() + y_pad)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def draw_background(ax: plt.Axes, case: dict) -> None:
    ax.scatter(case['coordinates'][:, 0], case['coordinates'][:, 1], s=float(case['point_size']), color='#E4E4E4', linewidths=0, zorder=0)

def build_spatial_figure(cases: list[dict]) -> plt.Figure:
    configure_matplotlib()
    fig = plt.figure(figsize=(15.0, 4.8))
    rows = fig.add_gridspec(2, 1, height_ratios=[1, 1], left=0.12, right=0.955, bottom=0.10, top=0.94, hspace=0.34)
    shared_norm = mpl.colors.Normalize(vmin=0.0, vmax=max(float(case['expression_limit']) for case in cases))
    shared_artist = None
    for row, case in enumerate(cases):
        # Both rows span the figure width so the DLPFC series does not leave an
        # empty half-canvas and the two datasets remain directly comparable.
        row_spec = rows[row, 0]
        grid = row_spec.subgridspec(1, 6, wspace=0.18)
        map_axes = [fig.add_subplot(grid[0, index]) for index in range(6)]
        for ax in map_axes:
            draw_background(ax, case)
            setup_map_axis(ax, case['coordinates'])
        point_size = float(case['point_size'])
        truth_artist = map_axes[0].scatter(case['target_coordinates'][:, 0], case['target_coordinates'][:, 1], c=case['truth_log'], cmap=EXPRESSION_CMAP, norm=shared_norm, s=point_size, linewidths=0, zorder=2)
        shared_artist = truth_artist
        target_moran = float(case['trajectory'].iloc[0]['target_moran_i'])
        map_axes[0].set_title(f'Held-out truth\nMoran I = {target_moran:.3f}', fontsize=8.5, pad=4)
        for index, ar_value in enumerate(AR_VALUES, start=1):
            metric = case['trajectory'][case['trajectory']['assignment_randomness'] == ar_value].iloc[0]
            map_axes[index].scatter(case['target_coordinates'][:, 0], case['target_coordinates'][:, 1], c=case['generated_log'][ar_value], cmap=EXPRESSION_CMAP, norm=shared_norm, s=point_size, linewidths=0, zorder=2)
            is_best = np.isclose(ar_value, case['best_ar'])
            map_axes[index].set_title(f"AR {ar_value:.1f}\nMoran I = {float(metric['generated_moran_i']):.3f}", color='#0F4D92' if is_best else '#222222', fontsize=8.5, pad=4)
        source_label = case['source'].replace('Zhuang-ABCA-1.', '')
        target_label = case['target'].replace('Zhuang-ABCA-1.', '')
        map_axes[0].text(-0.20, 0.5, f"{case['dataset_label']}\n{source_label} → {target_label}\n{case['gene']}", transform=map_axes[0].transAxes, ha='right', va='center', fontsize=9.5, fontweight='bold', linespacing=1.35)
    colorbar_ax = fig.add_axes((0.972, 0.28, 0.010, 0.44))
    colorbar = fig.colorbar(shared_artist, cax=colorbar_ax)
    colorbar.set_label('log1p(count)', fontsize=7.3, labelpad=2)
    colorbar.ax.tick_params(labelsize=6.4, length=1.8, width=0.5)
    return fig

def build_trajectory_figure(cases: list[dict]) -> plt.Figure:
    configure_matplotlib()
    fig = plt.figure(figsize=(7.55, 2.75))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.35], wspace=0.32)
    axes = [fig.add_subplot(grid[0, index]) for index in range(2)]
    legend_ax = fig.add_subplot(grid[0, 2])
    legend_ax.set_axis_off()
    for ax, case in zip(axes, cases):
        trajectory = case['trajectory']
        x = trajectory['assignment_randomness'].to_numpy(float)
        target_moran = float(trajectory.iloc[0]['target_moran_i'])
        ax.plot(x, trajectory['pearson'], color='#0F4D92', marker='o', markersize=4.0, linewidth=1.7, label='Pearson r')
        ax.plot(x, trajectory['generated_moran_i'], color='#D9A441', marker='s', markersize=3.8, linewidth=1.7, label='Generated Moran I')
        ax.axhline(target_moran, color='#777777', linestyle=(0, (2, 2)), linewidth=0.9, label='Truth Moran I')
        ax.axvline(case['best_ar'], color='#0F4D92', linestyle=':', linewidth=1.0)
        all_values = np.concatenate([trajectory['pearson'].to_numpy(float), trajectory['generated_moran_i'].to_numpy(float), np.asarray([target_moran])])
        margin = max(float(np.ptp(all_values)) * 0.18, 0.04)
        ax.set_ylim(max(0.0, all_values.min() - margin), min(1.0, all_values.max() + margin))
        ax.set_xticks(AR_VALUES)
        ax.set_xlabel('Assignment randomness')
        ax.set_ylabel('Gene-level value')
        source = case['source'].replace('Zhuang-ABCA-1.', '')
        target = case['target'].replace('Zhuang-ABCA-1.', '')
        ax.set_title(f"{case['dataset_label']}  {source} → {target}  ·  {case['gene']}", pad=7)
        ax.grid(axis='y', color='#E2E2E2', linewidth=0.6)
        ax.tick_params(length=3, width=0.7)
        ax.text(case['best_ar'], ax.get_ylim()[1], 'best aggregate AR', color='#0F4D92', fontsize=6.3, ha='center', va='bottom')
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='center left', fontsize=7.0, handlelength=1.7, borderaxespad=0.0)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.2, top=0.86)
    return fig

def save_figure(figure: plt.Figure, work_dir: Path, stem: str, dpi: int) -> None:
    figure.savefig(work_dir / f'{stem}.pdf', bbox_inches='tight', metadata={'Creator': 'FEAST Study 05 AR spatial-effect plotting script', 'CreationDate': None, 'ModDate': None})
    figure.savefig(work_dir / f'{stem}.svg', bbox_inches='tight', metadata={'Creator': 'FEAST Study 05 AR spatial-effect plotting script', 'Date': None})
    figure.savefig(work_dir / f'{stem}.png', dpi=dpi, bbox_inches='tight', metadata={'Software': 'FEAST Study 05 AR spatial-effect plotting script'})
    plt.close(figure)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=VIS_DIR / 'figures')
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, per_gene, score_provenance, manifest = verify_inputs()
    selected = [choose_best_case(dataset, summary, per_gene) for dataset in ('dlpfc', 'merfish')]
    case_data = []
    plot_rows = []
    for case, trajectory in selected:
        run_support = verify_case_files(case, trajectory, score_provenance, manifest)
        case_data.append(load_case(case, trajectory, run_support))
        for _, row in trajectory.iterrows():
            plot_rows.append({'dataset': case['dataset'], 'source': case['source'], 'target': case['target'], 'direction': case['direction'], 'gene': case['gene'], 'assignment_randomness': float(row['assignment_randomness']), 'best_aggregate_ar': case['best_ar'], 'is_best_aggregate_ar': bool(np.isclose(row['assignment_randomness'], case['best_ar'])), 'dataset_mean_moran_profile_at_best_ar': case['best_ar_dataset_mean_moran_profile'], 'direction_moran_profile_at_best_ar': case['best_direction_moran_profile'], 'gene_selection_target_moran_quantile': case['gene_selection_moran_quantile'], 'gene_selection_target_moran_cutoff': case['gene_selection_moran_cutoff'], 'gene_selection_candidate_count': case['gene_selection_candidate_count'], 'pearson': float(row['pearson']), 'spearman': float(row['spearman']), 'target_moran_i': float(row['target_moran_i']), 'generated_moran_i': float(row['generated_moran_i'])})
    work_dir = Path(tempfile.mkdtemp(prefix=f'.{args.output_dir.name}-ar-spatial-effect-', dir=args.output_dir.parent))
    try:
        save_figure(build_spatial_figure(case_data), work_dir, 'conditional_transfer_ar_spatial_effect', int(args.dpi))
        save_figure(build_trajectory_figure(case_data), work_dir, 'conditional_transfer_ar_trajectory', int(args.dpi))
        pd.DataFrame(plot_rows).to_csv(work_dir / 'ar_spatial_effect_plot_data.csv', index=False)
        outputs = {path.name: None for path in sorted(work_dir.iterdir()) if path.is_file()}
        provenance = {'status': 'ok', 'configuration_id': 'study05-conditional-transfer-ar-spatial-effect-v2', 'scientific_disposition': 'validated_best_case_supporting_visualization_claim_reframe_required', 'figure_promotion_authorized': False, 'publication_claim_authorized': False, 'target_expression_loaded_by_generation': False, 'score_configuration_id': score_provenance['configuration_id'], 'selection': {'best_ar': 'maximum dataset mean cross-slice Moran-profile correlation', 'direction': 'maximum direction Moran-profile correlation at best AR', 'gene': 'maximum Pearson r among reference-observed genes in the top 5% target Moran I with 5-95% target zeros', 'selected': {case['dataset']: {'best_ar': case['best_ar'], 'direction': case['direction'], 'gene': case['gene']} for case in case_data}, 'best_case_label_required': False}, 'design': {'assignment_randomness_values': list(AR_VALUES), 'same_target_gene_and_expression_scale_across_ar': True, 'merfish_slice_width_relative_to_dlpfc': 2, 'best_aggregate_ar_highlighted': 'blue title only', 'spatial_map_annotations': 'Moran I only', 'gene_level_pearson_and_moran_trajectory': 'separate figure', 'panel_letters': False}, 'claim_limits': ['best-case gene illustration, not a typical-gene estimate', 'conditional on observed target XY and labels', 'target expression used only after generation for scoring and display', "aggregate best AR need not maximize the selected gene's coordinate-wise Pearson correlation"], 'png_dpi': int(args.dpi), 'outputs': outputs, 'created_at': datetime.now(timezone.utc).isoformat()}
        (work_dir / 'ar_spatial_effect_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
        published = {}
        for path in sorted(work_dir.iterdir()):
            if path.is_file():
                destination = args.output_dir / path.name
                os.replace(path, destination)
                published[destination.name] = None
        print(json.dumps({'status': 'ok', 'output_dir': str(args.output_dir), 'selected': provenance['selection']['selected'], 'outputs': published}, indent=2))
    finally:
        if work_dir.exists():
            work_dir.rmdir()
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
