"""Build transparent, data-first components for the two FEAST 3D workflows."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import struct
from pathlib import Path
from typing import Any

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('SOURCE_DATE_EPOCH', '0')

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

REPRO_ROOT = Path(__file__).resolve().parents[2]
STUDY06 = REPRO_ROOT / '06_3d_stack'
STUDY07 = REPRO_ROOT / '07_3d_transfer'
DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'figures' / 'illustration_components'
DEVCCF_ROOT = REPRO_ROOT.parent / 'Datasets' / 'Processed' / 'DevCCFv1_figshare_26377171' / 'coordinate_system' / 'GSE269617_region_merged'
SOURCE_ROOT = REPRO_ROOT.parent / 'Datasets' / 'Processed' / 'GSE269617' / 'h5ad_region_annotated'

INK = '#25313A'
MUTED = '#AEB8C0'
MEASURED = '#4D6574'
GENERATED = '#0072B2'
REGION_COLORS = {
    'DorsalVZ': '#cc4c02', 'IZ': '#756bb1', 'CP': '#2b8cbe', 'VentralVZ': '#fd8d3c',
    'GE': '#31a354', 'SeptalVZ': '#e6550d', 'Septum': '#636363', 'BT': '#de2d26',
}


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
        'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'figure.facecolor': 'none', 'axes.facecolor': 'none',
        'savefig.facecolor': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'text.color': INK,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--study06-target', type=int, default=83)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--dpi', type=int, default=600)
    return parser.parse_args()


def dense_counts(data: ad.AnnData) -> np.ndarray:
    matrix = data.layers['counts'] if 'counts' in data.layers else data.X
    if hasattr(matrix, 'toarray'):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def load_h5ad(path: Path, *, counts: bool) -> dict[str, Any]:
    data = ad.read_h5ad(path, backed=None if counts else 'r')
    try:
        spatial = np.asarray(data.obsm['spatial'], dtype=float)
        record = {
            'path': path, 'obs_names': data.obs_names.astype(str).to_numpy(),
            'xy': spatial,
            'xyz': np.asarray(data.obsm['spatial_3d'], dtype=float) if 'spatial_3d' in data.obsm else np.column_stack([spatial, np.zeros(len(spatial))]),
            'obs': data.obs.copy(),
        }
        if counts:
            record['library'] = np.log1p(dense_counts(data).sum(axis=1))
        return record
    finally:
        if data.isbacked:
            data.file.close()


def sampled_indices(size: int, cap: int, seed: int) -> np.ndarray:
    if size <= cap:
        return np.arange(size)
    return np.sort(np.random.default_rng(seed).choice(size, size=cap, replace=False))


def normalise(xy: np.ndarray, limits: tuple[tuple[float, float], tuple[float, float]] | None=None) -> tuple[np.ndarray, tuple[tuple[float, float], tuple[float, float]]]:
    if limits is None:
        limits = ((float(xy[:, 0].min()), float(xy[:, 0].max())), (float(xy[:, 1].min()), float(xy[:, 1].max())))
    width = max(limits[0][1] - limits[0][0], 1e-12)
    height = max(limits[1][1] - limits[1][0], 1e-12)
    values = np.column_stack(((xy[:, 0] - limits[0][0]) / width, (xy[:, 1] - limits[1][0]) / height))
    return values, limits


def finish_axis(axis: plt.Axes) -> None:
    axis.set_aspect('equal')
    axis.set_axis_off()
    axis.set_xlim(-0.03, 1.03)
    axis.set_ylim(-0.03, 1.03)


def export(figure: plt.Figure, output: Path, stem: str, dpi: int) -> list[Path]:
    paths = []
    for suffix, kwargs in {
        '.pdf': {'metadata': {'Title': stem, 'Creator': 'FEAST Study 08 component builder', 'CreationDate': None, 'ModDate': None}},
        '.svg': {'metadata': {'Title': stem, 'Creator': 'FEAST Study 08 component builder', 'Date': None}},
        '.png': {'metadata': {'Software': 'FEAST Study 08 component builder'}},
    }.items():
        path = output / f'{stem}{suffix}'
        figure.savefig(path, dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=0.02, **kwargs)
        paths.append(path)
    plt.close(figure)
    return paths


def class_colors(labels: np.ndarray) -> dict[str, Any]:
    values = sorted(set(map(str, labels)))
    palette = list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('Set3').colors)
    return {label: palette[index % len(palette)] for index, label in enumerate(values)}


def plot_identity(axis: plt.Axes, xy: np.ndarray, labels: np.ndarray, palette: dict[str, Any], seed: int, *, hollow: bool=False) -> None:
    values, _ = normalise(xy)
    index = sampled_indices(len(values), 18000, seed)
    colors = [palette[str(label)] for label in labels[index]]
    axis.scatter(values[index, 0], values[index, 1], s=1.0, c='none' if hollow else colors,
                 edgecolors=colors if hollow else 'none', linewidths=0.24 if hollow else 0,
                 alpha=0.72 if hollow else 0.84, rasterized=True)
    finish_axis(axis)


def plot_expression(axis: plt.Axes, xy: np.ndarray, values: np.ndarray, norm: Normalize, seed: int) -> Any:
    xy_values, _ = normalise(xy)
    index = sampled_indices(len(xy_values), 22000, seed)
    artist = axis.scatter(xy_values[index, 0], xy_values[index, 1], s=1.0, c=values[index], cmap='viridis', norm=norm,
                          linewidths=0, alpha=0.86, rasterized=True)
    finish_axis(axis)
    return artist


def find_study06_case(target_slice: int) -> tuple[dict[str, Any], Path, Path, Path]:
    final = STUDY06 / 'outputs' / 'final'
    plan = json.loads((final / 'plan.json').read_text(encoding='utf-8'))
    matches = [target for density in plan['densities'].values() for target in density['targets'] if int(target['target_slice']) == target_slice]
    matches = [target for target in matches if not target['donor_support']]
    if len(matches) != 1:
        raise ValueError(f'expected exactly one bracket-only Study 06 case for target {target_slice}, found {len(matches)}')
    target = matches[0]
    data_dir = Path(plan['data_dir'])
    prefix = 'Zhuang-ABCA-1'
    generated = final / 'targets' / str(target['density_name']) / f'{prefix}.{target_slice:03d}' / 'generated.h5ad'
    lower = data_dir / f"{prefix}.{int(target['lower_ref_slice']):03d}.h5ad"
    upper = data_dir / f"{prefix}.{int(target['upper_ref_slice']):03d}.h5ad"
    for path in (generated, lower, upper):
        if not path.is_file():
            raise FileNotFoundError(path)
    return target, lower, upper, generated


def study06_assets(output: Path, target_slice: int, seed: int, dpi: int) -> tuple[list[Path], dict[str, Any]]:
    target, lower_path, upper_path, generated_path = find_study06_case(target_slice)
    lower, upper, generated = (load_h5ad(path, counts=True) for path in (lower_path, upper_path, generated_path))
    if not np.array_equal(generated['xyz'][:, :2], generated['xy']) or not np.allclose(generated['xyz'][:, 2], float(target['target_z'])):
        raise ValueError('Study 06 generated target geometry does not match the frozen plan')
    blueprint = load_h5ad(Path(json.loads((STUDY06 / 'outputs' / 'final' / 'plan.json').read_text())['data_dir']) / f'Zhuang-ABCA-1.{target_slice:03d}.h5ad', counts=False)
    if not np.array_equal(blueprint['obs_names'], generated['obs_names']) or not np.array_equal(blueprint['xyz'], generated['xyz']):
        raise ValueError('Study 06 blueprint and generated target identities differ')
    palette = class_colors(np.concatenate([lower['obs']['class'].astype(str), blueprint['obs']['class'].astype(str), upper['obs']['class'].astype(str)]))
    library_values = np.concatenate([item['library'] for item in (lower, upper, generated)])
    norm = Normalize(vmin=float(np.quantile(library_values, 0.02)), vmax=float(np.quantile(library_values, 0.98)))
    outputs: list[Path] = []
    figure, axes = plt.subplots(1, 2, figsize=(4.8, 2.25))
    for axis, item, local_seed in zip(axes, (lower, upper), (seed + 1, seed + 2)):
        plot_expression(axis, item['xy'], item['library'], norm, local_seed)
    outputs += export(figure, output, 'study06_reference_pair_expression', dpi)
    figure, axis = plt.subplots(figsize=(2.5, 2.5))
    plot_identity(axis, blueprint['xy'], blueprint['obs']['class'].astype(str).to_numpy(), palette, seed + 3, hollow=True)
    outputs += export(figure, output, 'study06_target_blueprint_identity', dpi)
    figure, axis = plt.subplots(figsize=(2.5, 2.5))
    artist = plot_expression(axis, generated['xy'], generated['library'], norm, seed + 4)
    figure.colorbar(artist, ax=axis, fraction=0.045, pad=0.02).set_label('log1p total counts')
    outputs += export(figure, output, 'study06_generated_target_expression', dpi)
    for name, expression in (('study06_native_stack_identity', False), ('study06_native_stack_expression', True)):
        figure = plt.figure(figsize=(3.4, 3.3))
        axis = figure.add_subplot(111, projection='3d')
        items = (lower, generated, upper)
        for index, item in enumerate(items):
            xy, _ = normalise(item['xy'])
            selected = sampled_indices(len(xy), 9000, seed + 10 + index)
            if expression:
                axis.scatter(xy[selected, 0], xy[selected, 1], item['xyz'][selected, 2], c=item['library'][selected], cmap='viridis', norm=norm, s=0.45, alpha=0.78, linewidths=0, depthshade=False, rasterized=True)
            else:
                labels = item['obs']['class'].astype(str).to_numpy()[selected]
                axis.scatter(xy[selected, 0], xy[selected, 1], item['xyz'][selected, 2], c=[palette[label] for label in labels], s=0.45, alpha=0.75, linewidths=0, depthshade=False, rasterized=True)
        axis.set_axis_off(); axis.set_box_aspect((1, 1, 0.75)); axis.view_init(elev=23, azim=-55)
        outputs += export(figure, output, name, dpi)
    figure, axis = plt.subplots(figsize=(4.6, 1.5))
    axis.set_axis_off(); axis.set_xlim(0, 1); axis.set_ylim(0, 1)
    z0, zt, z1 = (float(target['z0']), float(target['target_z']), float(target['z1']))
    positions = np.interp([z0, zt, z1], [z0, z1], [0.14, 0.86])
    axis.plot([positions[0], positions[2]], [0.45, 0.45], color=MUTED, linewidth=1.4)
    axis.scatter(positions, [0.45] * 3, s=[65, 86, 65], c=[MEASURED, GENERATED, MEASURED], edgecolors='white', linewidths=1.0, zorder=3)
    axis.text(positions[0], 0.18, f"z = {z0:.3f}", ha='center', fontsize=8)
    axis.text(positions[1], 0.18, f"z = {zt:.3f}", ha='center', fontsize=8)
    axis.text(positions[2], 0.18, f"z = {z1:.3f}", ha='center', fontsize=8)
    axis.text((positions[0] + positions[1]) / 2, 0.66, f"1 − τ = {1 - float(target['tau']):.3f}", ha='center', color=MEASURED, fontsize=8)
    axis.text((positions[1] + positions[2]) / 2, 0.66, f"τ = {float(target['tau']):.3f}", ha='center', color=MEASURED, fontsize=8)
    outputs += export(figure, output, 'study06_native_z_bracket', dpi)
    return outputs, {'target_slice': target_slice, 'density_name': target['density_name'], 'reference_slices': [int(target['lower_ref_slice']), int(target['upper_ref_slice'])], 'target_z': zt, 'tau': float(target['tau']), 'target_expression_read_for_blueprint': False}


def load_schema() -> dict[str, str]:
    frame = pd.read_csv(DEVCCF_ROOT / 'gse269617_region_schema.tsv', sep='\t')
    return dict(zip(frame['region_label'].astype(str), frame['hex_color'].astype(str)))


def nifti_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with gzip.open(path, 'rb') as handle:
        raw = handle.read()
    endian = '<' if struct.unpack('<i', raw[:4])[0] == 348 else '>'
    dim = struct.unpack(endian + '8h', raw[40:56]); shape = tuple(int(value) for value in dim[1:1 + int(dim[0])])
    datatype = int(struct.unpack(endian + 'h', raw[70:72])[0]); dtype = {2: np.dtype('u1'), 4: np.dtype('i2'), 8: np.dtype('i4'), 16: np.dtype('f4'), 512: np.dtype('u2')}[datatype].newbyteorder(endian)
    offset = int(round(struct.unpack(endian + 'f', raw[108:112])[0]))
    data = np.frombuffer(raw[offset:offset + int(np.prod(shape)) * dtype.itemsize], dtype=dtype).reshape(shape, order='F')
    affine = np.eye(4, dtype=float)
    affine[0] = struct.unpack(endian + '4f', raw[280:296]); affine[1] = struct.unpack(endian + '4f', raw[296:312]); affine[2] = struct.unpack(endian + '4f', raw[312:328])
    return data, affine


def generated_path(age: str, index: int) -> Path:
    manifest = pd.read_csv(STUDY07 / 'outputs' / 'final' / age / 'manifest.csv')
    row = manifest.loc[manifest['z_index'].eq(index)]
    if len(row) != 1:
        raise ValueError(f'{age} z index {index} is not unique')
    return STUDY07 / 'outputs' / 'final' / age / str(row.iloc[0]['filename'])


def plot_region(axis: plt.Axes, item: dict[str, Any], colors: dict[str, str], seed: int) -> None:
    labels = item['obs']['region'].astype(str).to_numpy(); xy, _ = normalise(item['xy']); index = sampled_indices(len(xy), 24000, seed)
    axis.scatter(xy[index, 0], xy[index, 1], c=[colors[label] for label in labels[index]], s=1.0, alpha=0.86, linewidths=0, rasterized=True)
    finish_axis(axis)


def study07_assets(output: Path, seed: int, dpi: int) -> tuple[list[Path], dict[str, Any]]:
    colors = load_schema(); outputs: list[Path] = []
    age_specs = {'E15.5': {'central': 78, 'levels': [0, 39, 78, 118, 157], 'volume': 'E15.5_broad_region_annotations.nii.gz', 'source_glob': '*E14*.h5ad'}, 'E18.5': {'central': 100, 'levels': [0, 50, 100, 151, 201], 'volume': 'E18.5_broad_region_annotations.nii.gz', 'source_glob': '*E18M*.h5ad'}}
    for age_index, (age, spec) in enumerate(age_specs.items()):
        central = load_h5ad(generated_path(age, int(spec['central'])), counts=True)
        if not np.allclose(central['xyz'][:, :2], central['xy']) or not np.allclose(central['xyz'][:, 2], central['obs']['z'].to_numpy(float)):
            raise ValueError(f'{age} generated coordinates are inconsistent')
        if set(central['obs']['region'].astype(str)) - set(colors):
            raise ValueError(f'{age} contains a region absent from the schema')
        norm = Normalize(vmin=float(np.quantile(central['library'], 0.02)), vmax=float(np.quantile(central['library'], 0.98)))
        figure, axis = plt.subplots(figsize=(2.55, 2.55)); plot_region(axis, central, colors, seed + age_index * 100)
        outputs += export(figure, output, f'study07_{age.replace(".", "_")}_central_region', dpi)
        figure, axis = plt.subplots(figsize=(2.55, 2.55)); artist = plot_expression(axis, central['xy'], central['library'], norm, seed + age_index * 100 + 1)
        figure.colorbar(artist, ax=axis, fraction=0.045, pad=0.02).set_label('log1p total counts')
        outputs += export(figure, output, f'study07_{age.replace(".", "_")}_central_expression', dpi)
        figure = plt.figure(figsize=(3.45, 3.25)); axis = figure.add_subplot(111, projection='3d')
        for level_index, z_index in enumerate(spec['levels']):
            item = load_h5ad(generated_path(age, int(z_index)), counts=False); xy, _ = normalise(item['xy']); chosen = sampled_indices(len(xy), 12000, seed + age_index * 100 + 10 + level_index); labels = item['obs']['region'].astype(str).to_numpy()[chosen]
            axis.scatter(xy[chosen, 0], xy[chosen, 1], item['xyz'][chosen, 2], c=[colors[label] for label in labels], s=0.42, alpha=0.78, linewidths=0, depthshade=False, rasterized=True)
        axis.set_axis_off(); axis.set_box_aspect((1, 1, 0.9)); axis.view_init(elev=23, azim=-55)
        outputs += export(figure, output, f'study07_{age.replace(".", "_")}_generated_volume', dpi)
        volume, _ = nifti_data(DEVCCF_ROOT / str(spec['volume']))
        if volume.ndim != 3 or not np.any(volume > 0):
            raise ValueError(f'{age} DevCCF volume is not a populated 3D atlas')
        figure = plt.figure(figsize=(3.45, 3.25)); axis = figure.add_subplot(111, projection='3d')
        for level_index, z_index in enumerate(spec['levels']):
            item = load_h5ad(generated_path(age, int(z_index)), counts=False); xy, _ = normalise(item['xy']); chosen = sampled_indices(len(xy), 12000, seed + age_index * 100 + 30 + level_index)
            axis.scatter(xy[chosen, 0], xy[chosen, 1], item['xyz'][chosen, 2], s=0.42, c=MUTED, alpha=0.46, linewidths=0, depthshade=False, rasterized=True)
        axis.set_axis_off(); axis.set_box_aspect((1, 1, 0.9)); axis.view_init(elev=23, azim=-55)
        outputs += export(figure, output, f'study07_{age.replace(".", "_")}_expression_free_coordinate_space', dpi)
        source_paths = sorted(SOURCE_ROOT.glob(str(spec['source_glob'])))
        figure, axes = plt.subplots(2, 5, figsize=(6.0, 2.4)); axes = axes.ravel()
        for axis, path in zip(axes, source_paths):
            source = load_h5ad(path, counts=False); plot_region(axis, source, colors, seed + age_index * 100 + len(source_paths));
        for axis in axes[len(source_paths):]: axis.set_axis_off()
        outputs += export(figure, output, f'study07_{age.replace(".", "_")}_source_cohort', dpi)
    coverage = pd.read_csv(STUDY07 / 'outputs' / 'final' / 'evaluation' / 'region_coverage.csv')
    for age in age_specs:
        frame = coverage.loc[coverage['age'].eq(age)]; pivot = frame.pivot_table(index='z_world', columns='region', values='n_spots', aggfunc='sum', fill_value=0).sort_index(); regions = [region for region in REGION_COLORS if region in pivot]
        figure, axis = plt.subplots(figsize=(4.8, 1.75)); axis.stackplot(pivot.index.to_numpy(float), [pivot[region].to_numpy(float) for region in regions], colors=[colors[region] for region in regions], linewidth=0, alpha=0.96); axis.set_axis_off()
        outputs += export(figure, output, f'study07_{age.replace(".", "_")}_axis_region_support', dpi)
    return outputs, {'central_indices': {age: int(spec['central']) for age, spec in age_specs.items()}, 'volume_indices': {age: spec['levels'] for age, spec in age_specs.items()}, 'target_expression_accuracy_claim': False}


def contact_sheet(output: Path, stems: list[str], dpi: int) -> list[Path]:
    columns = 3
    rows = int(np.ceil(len(stems) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(8.0, rows * 2.0))
    for axis, stem in zip(axes.ravel(), stems):
        axis.imshow(plt.imread(output / f'{stem}.png'))
        axis.set_title(stem.replace('_', ' '), fontsize=6.5, pad=3)
        axis.set_axis_off()
    for axis in axes.ravel()[len(stems):]:
        axis.set_axis_off()
    figure.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015, hspace=0.24, wspace=0.06)
    return export(figure, output, 'component_contact_sheet', dpi)


def main() -> None:
    args = parse_args(); output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'use a fresh output directory: {output}')
    output.mkdir(parents=True, exist_ok=False)
    configure_matplotlib()
    outputs06, record06 = study06_assets(output, int(args.study06_target), int(args.seed), int(args.dpi))
    outputs07, record07 = study07_assets(output, int(args.seed), int(args.dpi))
    stems = sorted({path.stem for path in outputs06 + outputs07})
    contact = contact_sheet(output, stems, int(args.dpi))
    record = {'schema_version': 1, 'study': '08_workflow', 'purpose': 'manual-assembly illustration assets; not a promoted manuscript figure', 'study06': record06, 'study07': record07, 'rendering': {'seed': int(args.seed), 'png_dpi': int(args.dpi), 'pdf_fonttype': 42, 'svg_fonttype': 'none', 'point_layers': 'rasterized'}, 'outputs': [path.name for path in sorted(outputs06 + outputs07 + contact)], 'source_paths': {'study06_plan': str(STUDY06 / 'outputs' / 'final' / 'plan.json'), 'study07_evaluation': str(STUDY07 / 'outputs' / 'final' / 'evaluation' / 'region_coverage.csv'), 'devccf_schema': str(DEVCCF_ROOT / 'gse269617_region_schema.tsv')}}
    (output / 'component_manifest.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'wrote {len(stems)} component sets to {output}')


if __name__ == '__main__':
    main()
