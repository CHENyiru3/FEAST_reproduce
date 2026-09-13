"""Render all-arm 3D and z-wise spot-profile fidelity diagnostics for Study 06."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('SOURCE_DATE_EPOCH', '0')

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.colors import TwoSlopeNorm

REPRO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPRO_ROOT / '06_3d_stack' / 'outputs' / 'reference_density_rank_v1'
OUTPUT = Path(__file__).resolve().parent / 'figures'
GAPS = (3, 5, 10)
COLORS = {3: '#0072B2', 5: '#E69F00', 10: '#009E73'}
LABELS: dict[int, str] = {}
SAMPLE_CAP = 1500
SAMPLE_SEED = 2026
BLOCK_SIZE = 512


def configure_matplotlib() -> None:
    arial_paths = (Path(sys.prefix) / 'fonts' / 'arial.ttf', Path(sys.prefix) / 'fonts' / 'arialbd.ttf')
    if not all((path.is_file() for path in arial_paths)):
        raise FileNotFoundError(f'Arial Regular and Bold are required for this figure: {arial_paths}')
    for path in arial_paths:
        font_manager.fontManager.addfont(path)
    mpl.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold', 'axes.labelsize': 8.5, 'axes.edgecolor': '#555555', 'axes.linewidth': 0.8, 'xtick.labelsize': 7.0, 'ytick.labelsize': 7.0, 'legend.fontsize': 7.0, 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})


def dense(matrix: object) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def count_matrix(adata: ad.AnnData) -> object:
    return adata.layers['counts'] if 'counts' in adata.layers else adata.X


def profile_correlations(generated: object, target: object, n_obs: int) -> np.ndarray:
    values = np.full(n_obs, np.nan, dtype=float)
    for start in range(0, n_obs, BLOCK_SIZE):
        stop = min(start + BLOCK_SIZE, n_obs)
        left = dense(generated[start:stop]).astype(float, copy=False)
        right = dense(target[start:stop]).astype(float, copy=False)
        left -= left.mean(axis=1, keepdims=True)
        right -= right.mean(axis=1, keepdims=True)
        denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
        valid = denominator > 0
        values[start:stop][valid] = np.sum(left[valid] * right[valid], axis=1) / denominator[valid]
    return values


def require_inputs(input_root: Path) -> tuple[dict, dict, dict]:
    validation = json.loads((input_root / 'validation_summary.json').read_text(encoding='utf-8'))
    if validation.get('status') != 'passed' or validation.get('validated_outputs') != 336 or validation.get('positions_per_stack') != 144 or validation.get('stack_geometries_identical') is not True:
        raise ValueError('complete Study 06 validation is required')
    score = json.loads((input_root / 'evaluation' / 'score_provenance.json').read_text(encoding='utf-8'))
    plan = json.loads((input_root / 'plan.json').read_text(encoding='utf-8'))
    return plan, score, validation


def collect_profile_data(plan: dict, input_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    data_dir = Path(plan['data_dir'])
    prefix = 'Zhuang-ABCA-1'
    for density_name, density in plan['densities'].items():
        for target in sorted(density['targets'], key=lambda item: int(item['target_index'])):
            target_slice = int(target['target_slice'])
            name = f'{prefix}.{target_slice:03d}'
            generated = ad.read_h5ad(input_root / 'targets' / density_name / name / 'generated.h5ad')
            observed = ad.read_h5ad(data_dir / f'{name}.h5ad')
            try:
                if generated.shape != observed.shape or not generated.obs_names.equals(observed.obs_names) or not generated.var_names.equals(observed.var_names):
                    raise ValueError(f'{density_name} {name} does not retain validated target identity')
                coordinates = np.asarray(generated.obsm['spatial_3d'], dtype=float)
                if coordinates.shape != (generated.n_obs, 3) or not np.array_equal(coordinates, np.asarray(observed.obsm['spatial_3d'], dtype=float)):
                    raise ValueError(f'{density_name} {name} does not retain target 3D coordinates')
                correlation = profile_correlations(count_matrix(generated), count_matrix(observed), generated.n_obs)
                valid = np.isfinite(correlation)
                if not valid.any():
                    raise ValueError(f'{density_name} {name} has no valid per-spot profile correlations')
                valid_values = correlation[valid]
                summaries.append({'density_name': density_name, 'gap': int(target['gap']), 'target_index': int(target['target_index']), 'target_slice': target_slice, 'target_z': float(target['target_z']), 'n_spots': int(generated.n_obs), 'n_valid_spots': int(valid.sum()), 'valid_spot_fraction': float(valid.mean()), 'profile_pearson_q25': float(np.quantile(valid_values, 0.25)), 'profile_pearson_median': float(np.median(valid_values)), 'profile_pearson_q75': float(np.quantile(valid_values, 0.75))})
                valid_indices = np.flatnonzero(valid)
                rng = np.random.default_rng(SAMPLE_SEED + int(target['gap']) * 10000 + int(target['target_index']))
                selected = np.sort(rng.choice(valid_indices, size=min(SAMPLE_CAP, len(valid_indices)), replace=False))
                for index in selected:
                    samples.append({'density_name': density_name, 'gap': int(target['gap']), 'target_index': int(target['target_index']), 'target_slice': target_slice, 'target_z': float(target['target_z']), 'spot_index': int(index), 'x': float(coordinates[index, 0]), 'y': float(coordinates[index, 1]), 'z': float(coordinates[index, 2]), 'profile_pearson': float(correlation[index]), 'source_kind': 'simulated'})
            finally:
                del generated, observed
        for stack_index, reference_slice in enumerate(map(int, density['reference_pool'])):
            name = f'{prefix}.{reference_slice:03d}'
            reference = ad.read_h5ad(data_dir / f'{name}.h5ad', backed='r')
            try:
                coordinates = np.asarray(reference.obsm['spatial_3d'], dtype=float)
                rng = np.random.default_rng(SAMPLE_SEED + int(density['spec']['gap']) * 10000 + reference_slice)
                selected = np.sort(rng.choice(reference.n_obs, size=min(SAMPLE_CAP, reference.n_obs), replace=False))
                for index in selected:
                    samples.append({'density_name': density_name, 'gap': int(density['spec']['gap']), 'target_index': stack_index, 'target_slice': reference_slice, 'target_z': float(coordinates[index, 2]), 'spot_index': int(index), 'x': float(coordinates[index, 0]), 'y': float(coordinates[index, 1]), 'z': float(coordinates[index, 2]), 'profile_pearson': float('nan'), 'source_kind': 'real'})
            finally:
                reference.file.close()
    return (pd.DataFrame(summaries).sort_values(['gap', 'target_index']), pd.DataFrame(samples).sort_values(['gap', 'target_index', 'spot_index']))


def spatial_limits(samples: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    result = []
    for column in ('x', 'y', 'z'):
        lower, upper = (float(samples[column].min()), float(samples[column].max()))
        margin = max((upper - lower) * 0.03, 0.01)
        result.append((lower - margin, upper + margin))
    return tuple(result)  # type: ignore[return-value]


def style_3d(axis: plt.Axes, xyz_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]], elev: float, azim: float) -> None:
    axis.set_xlim(*xyz_limits[0])
    axis.set_ylim(*xyz_limits[1])
    axis.set_zlim(*xyz_limits[2])
    axis.set_xlabel('x', labelpad=-6)
    axis.set_ylabel('y', labelpad=-6)
    axis.set_zlabel('z', labelpad=-4)
    axis.set_box_aspect(tuple(limit[1] - limit[0] for limit in xyz_limits))
    axis.view_init(elev=elev, azim=azim)
    axis.xaxis.pane.fill = False
    axis.yaxis.pane.fill = False
    axis.zaxis.pane.fill = False
    axis.grid(False)


def render_3d(samples: pd.DataFrame, stem: Path) -> None:
    configure_matplotlib()
    figure = plt.figure(figsize=(10.2, 6.4))
    grid = figure.add_gridspec(2, 3, left=0.045, right=0.89, bottom=0.06, top=0.93, hspace=0.12, wspace=0.04)
    limits = spatial_limits(samples)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    scatter = None
    for row, (view_name, elev, azim) in enumerate((('Oblique view', 25.0, -55.0), ('Lateral view', 12.0, 0.0))):
        for column, gap in enumerate(GAPS):
            axis = figure.add_subplot(grid[row, column], projection='3d')
            subset = samples.loc[samples['gap'] == gap]
            real = subset.loc[subset['source_kind'] == 'real']
            simulated = subset.loc[subset['source_kind'] == 'simulated'].sort_values('profile_pearson')
            axis.scatter(real['x'], real['y'], real['z'], color='#777777', marker='^', s=0.7, linewidths=0, alpha=0.35, depthshade=False, rasterized=True)
            scatter = axis.scatter(simulated['x'], simulated['y'], simulated['z'], c=simulated['profile_pearson'], cmap='RdBu_r', norm=norm, marker='o', s=0.45, linewidths=0, alpha=0.82, depthshade=False, rasterized=True)
            style_3d(axis, limits, elev, azim)
            if row == 0:
                axis.set_title(LABELS[gap], pad=5)
            if column == 0:
                axis.text2D(-0.15, 0.5, view_name, transform=axis.transAxes, rotation=90, ha='center', va='center', fontsize=8.5, fontweight='bold')
    assert scatter is not None
    colorbar = figure.colorbar(scatter, ax=figure.axes, shrink=0.70, pad=0.025, location='right')
    colorbar.set_label('Per-spot all-gene Pearson r')
    figure.legend(handles=[mpl.lines.Line2D([], [], color='#777777', marker='^', linestyle='None', label='Retained real geometry'), mpl.lines.Line2D([], [], color='#777777', marker='o', linestyle='None', label='FEAST-simulated fidelity')], loc='lower center', ncol=2, frameon=False)
    figure.suptitle('Study 06 whole-stack spot-profile fidelity', y=0.985, fontsize=12, fontweight='bold')
    for suffix, kwargs in {'.pdf': {'metadata': {'Title': 'FEAST Study 06 3D spot-profile fidelity', 'Creator': 'FEAST publication visualization', 'CreationDate': None, 'ModDate': None}}, '.svg': {'metadata': {'Title': 'FEAST Study 06 3D spot-profile fidelity', 'Creator': 'FEAST publication visualization', 'Date': None}}, '.png': {'dpi': 600, 'metadata': {'Software': 'FEAST Study 06 plotting script'}}}.items():
        figure.savefig(stem.with_suffix(suffix), **kwargs)
    plt.close(figure)


def render_z_summary(summary: pd.DataFrame, stem: Path) -> None:
    configure_matplotlib()
    figure, axes = plt.subplots(1, 3, figsize=(9.3, 3.2), sharey=True)
    low = float(summary['profile_pearson_q25'].min())
    high = float(summary['profile_pearson_q75'].max())
    margin = max(0.02, (high - low) * 0.08)
    for axis, gap in zip(axes, GAPS):
        subset = summary.loc[summary['gap'] == gap].sort_values('target_z')
        axis.fill_between(subset['target_z'], subset['profile_pearson_q25'], subset['profile_pearson_q75'], color=COLORS[gap], alpha=0.18, linewidth=0)
        axis.plot(subset['target_z'], subset['profile_pearson_median'], color=COLORS[gap], marker='o', markersize=3, linewidth=1.0)
        axis.set_title(LABELS[gap], pad=6)
        axis.set_xlabel('Target z')
        axis.set_ylim(max(-1.0, low - margin), min(1.0, high + margin))
        axis.grid(axis='y', color='#E3E3E3', linewidth=0.5, alpha=0.75)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    axes[0].set_ylabel('Per-spot all-gene Pearson r')
    figure.suptitle('Study 06 spot-profile fidelity across ordered target z', y=0.985, fontsize=11, fontweight='bold')
    figure.subplots_adjust(left=0.07, right=0.995, bottom=0.18, top=0.84, wspace=0.07)
    for suffix, kwargs in {'.pdf': {'metadata': {'Title': 'FEAST Study 06 profile fidelity by z', 'Creator': 'FEAST publication visualization', 'CreationDate': None, 'ModDate': None}}, '.svg': {'metadata': {'Title': 'FEAST Study 06 profile fidelity by z', 'Creator': 'FEAST publication visualization', 'Date': None}}, '.png': {'dpi': 600, 'metadata': {'Software': 'FEAST Study 06 plotting script'}}}.items():
        figure.savefig(stem.with_suffix(suffix), **kwargs)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stack_stem = output / 'conditional_stack_3d_profile_fidelity'
    z_stem = output / 'conditional_stack_profile_fidelity_by_z'
    outputs = [path for stem in (stack_stem, z_stem) for path in (stem.with_suffix('.pdf'), stem.with_suffix('.svg'), stem.with_suffix('.png'))]
    outputs += [output / 'profile_fidelity_target_summary.csv', output / 'profile_fidelity_stack_samples.csv', output / 'profile_fidelity_provenance.json']
    if any(path.exists() for path in outputs):
        raise FileExistsError('fresh-only spot-profile outputs already exist')
    plan, score_provenance, validation = require_inputs(input_root)
    LABELS.update({int(density['spec']['gap']): f"Gap {int(density['spec']['gap'])} ({len(density['reference_pool'])} real / {len(density['target_pool'])} simulated)" for density in plan['densities'].values()})
    summary, samples = collect_profile_data(plan, input_root)
    summary.to_csv(output / 'profile_fidelity_target_summary.csv', index=False)
    samples.to_csv(output / 'profile_fidelity_stack_samples.csv', index=False)
    render_3d(samples, stack_stem)
    render_z_summary(summary, z_stem)
    record = {'schema_version': 1, 'study': '06_3d_stack', 'configuration_id': score_provenance['configuration_id'], 'figure_promotion_authorized': False, 'publication_claim_authorized': False, 'interpretation': 'descriptive per-spot fidelity on simulated targets within complete 144-position real-plus-simulated geometry; no biological-replicate inference', 'profile_metric': {'name': 'per_spot_all_gene_pearson', 'genes': 1122, 'constant_profiles': 'excluded'}, 'sampling': {'seed': SAMPLE_SEED, 'maximum_valid_spots_per_slice': SAMPLE_CAP, 'scope': 'real anchors provide geometry context; fidelity is computed only for simulated targets'}, 'inputs': {'validation_summary': str(input_root / 'validation_summary.json'), 'plan': str(input_root / 'plan.json'), 'score_provenance': str(input_root / 'evaluation' / 'score_provenance.json'), 'validated_generated_outputs': int(validation['validated_outputs'])}, 'outputs': {path.name: None for path in outputs}, 'editable_text': {'pdf_fonttype': 42, 'svg_fonttype': 'none'}, 'png_dpi': 600}
    (output / 'profile_fidelity_provenance.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
