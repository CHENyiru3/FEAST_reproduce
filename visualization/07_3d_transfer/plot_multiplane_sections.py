#!/usr/bin/env python3
"""Show native voxel cross-sections through five directions of the Study 07 volume."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from anndata.experimental import read_elem
import plot_3d_region_four_views as regions
from plot_axis_cross_sections import GENES, EXPRESSION_CMAP, configure_matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / '07_3d_transfer/outputs/generative_transfer_two_reference_float32_v1'
DEFAULT_OUTPUT = Path(__file__).parent / 'figures/multiplane_sections'
THICKNESS = .02
XZ_OBLIQUE_OFFSET = .4
S = 1 / np.sqrt(2)
# In-plane horizontal and vertical unit vectors, in stored world coordinates.
PLANES = [('XY', [1, 0, 0], [0, 1, 0]), ('XZ', [1, 0, 0], [0, 0, 1]),
          ('YZ', [0, 1, 0], [0, 0, 1]),
          ('Oblique XZ 45°', [S, 0, -S], [0, 1, 0]),
          ('Oblique YZ 45°', [1, 0, 0], [0, S, -S])]


def collect(input_root, manifests):
    sections, contexts, records = {}, {}, []
    for age in regions.AGES:
        blocks = []
        for row in manifests[age].itertuples(index=False):
            with h5py.File(input_root / age / row.filename) as h:
                xyz = h['obsm/spatial_3d'][:]
                assert xyz.shape == (row.n_spots, 3)
                assert np.allclose(xyz[:, 2], row.z_world)
                labels = np.asarray(read_elem(h['obs/region'])).astype(str)
            blocks.append(pd.DataFrame(dict(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                region=labels, z_index=row.z_index, spot_index=np.arange(len(xyz)))))
        frame = pd.concat(blocks, ignore_index=True)
        xyz = frame[['x', 'y', 'z']].to_numpy()
        levels = [np.unique(xyz[:, j]) for j in range(3)]
        center = np.array([values[len(values)//2] for values in levels])
        context = []
        for index, block in enumerate(blocks):
            rng = np.random.default_rng(2026 + regions.AGES.index(age)*100000 + index)
            ids = np.arange(len(block)) if len(block) <= 700 else rng.choice(len(block), 700, replace=False)
            context.append(block.iloc[ids])
        contexts[age] = (pd.concat(context), regions.spatial_limits(frame))
        masks = []
        centered = xyz - center
        for name, horizontal, vertical in PLANES:
            normal = np.cross(horizontal, vertical)
            plane_offset = XZ_OBLIQUE_OFFSET if name == 'Oblique XZ 45°' else 0.
            plane_center = center + plane_offset * normal
            mask = np.abs(centered @ normal - plane_offset) < THICKNESS/2 + 1e-8
            assert mask.any(), (age, name)
            masks.append(mask)
            records.append(dict(age=age, plane=name, center=plane_center.tolist(), offset_from_central_plane=plane_offset,
                horizontal=horizontal, vertical=vertical, normal=normal.tolist(),
                thickness=THICKNESS, n_spots=int(mask.sum())))
        selected = np.logical_or.reduce(masks)
        expression = np.full((len(frame), len(GENES)), np.nan, dtype=np.float32)
        offset = 0
        for row, block in zip(manifests[age].itertuples(index=False), blocks, strict=True):
            local = np.flatnonzero(selected[offset:offset+len(block)])
            if len(local):
                with h5py.File(input_root / age / row.filename) as h:
                    names = list(np.asarray(read_elem(h['var'][h['var'].attrs['_index']])).astype(str))
                    indices = np.array([names.index(gene) for gene in GENES])
                    order = np.argsort(indices)
                    values = h['layers/counts'][:, indices[order]]
                    expression[offset+local] = values[local][:, np.argsort(order)]
            offset += len(block)
        assert np.isfinite(expression[selected]).all()
        for (name, u, v), mask in zip(PLANES, masks, strict=True):
            section = frame.loc[mask].copy()
            section['u'], section['v'] = centered[mask] @ u, centered[mask] @ v
            for j, gene in enumerate(GENES):
                section[gene] = expression[mask, j]
            sections[age, name] = section
        print(f'{age}: extracted five planes from {len(frame):,} native voxels', flush=True)
    return sections, contexts, records


def render(age, sections, contexts, records, norms, output, dpi):
    fig = plt.figure(figsize=(12, 10.8))
    grid = fig.add_gridspec(6, 5, height_ratios=[1]*5+[.07], left=.025,
        right=.99, bottom=.10, top=.86, hspace=.10, wspace=.025)
    context, limits = contexts[age]
    extent = max(max(abs(section[k]).max() for k in ['u', 'v']) for section in sections.values())*1.04
    for row, (name, u, v) in enumerate(PLANES):
        section = sections[age, name]
        record = next(r for r in records if r['age'] == age and r['plane'] == name)
        locator = fig.add_subplot(grid[row, 0], projection='3d')
        locator.scatter(context.x, context.y, context.z, color='#939CA5', s=.12,
            alpha=.07, linewidths=0, rasterized=True, depthshade=False)
        uu, vv = np.meshgrid([section.u.min(), section.u.max()], [section.v.min(), section.v.max()])
        surface = np.asarray(record['center']) + uu[..., None]*u + vv[..., None]*v
        locator.plot_surface(surface[..., 0], surface[..., 1], surface[..., 2], color='#0072B2', alpha=.35, shade=False)
        regions.style_3d(locator, limits, 25, -55, 0)
        label = name.replace('Oblique ', '')
        if record['offset_from_central_plane']:
            label += f" · offset {record['offset_from_central_plane']:+.1f}"
        locator.set_title(label, fontsize=12, pad=1)
        locator.set_xlabel('x', labelpad=-12, fontsize=11)
        locator.set_ylabel('y', labelpad=-12, fontsize=11)
        locator.set_zlabel('z', labelpad=-12, fontsize=11)
        for col, gene in enumerate(GENES, start=1):
            ax = fig.add_subplot(grid[row, col])
            values = np.log1p(section[gene].to_numpy())
            order = np.argsort(values, kind='stable')
            positive = order[values[order] > 0]
            ax.scatter(section.u, section.v, color='#E2E2E2', s=1.6, marker='s', linewidths=0, rasterized=True)
            ax.scatter(section.u.to_numpy()[positive], section.v.to_numpy()[positive], c=values[positive],
                norm=norms[gene], cmap=EXPRESSION_CMAP, s=1.6, marker='s', linewidths=0, rasterized=True)
            ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect='equal')
            ax.set_axis_off()
            if row == 0:
                ax.set_title(gene, fontsize=15, pad=3)
    for col, gene in enumerate(GENES, start=1):
        slot = grid[5, col].get_position(fig)
        cax = fig.add_axes([slot.x0 + slot.width*.22, slot.y0, slot.width*.56, slot.height*.70])
        bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norms[gene], cmap=EXPRESSION_CMAP),
            cax=cax, orientation='horizontal')
        bar.ax.tick_params(labelsize=11, length=2, pad=2)
        bar.locator = mpl.ticker.MaxNLocator(3)
        bar.update_ticks()
        bar.set_label('log1p counts', fontsize=12, labelpad=3)
    fig.suptitle(f'{age} gene expression across five cutting planes',
        x=.045, ha='left', y=.98, fontsize=19, fontweight='bold')
    fig.text(.045, .94, 'Blue planes locate sections. All section voxels shown; no interpolation or smoothing.', fontsize=12)
    fig.text(.045, .91, 'Slab thickness: 0.02. Shared gene and spatial scales. XZ 45° cut shifted by +0.4.', fontsize=12)
    fig.text(.5, .02, 'Generated expression. Planes selected independently by age; not homologous cross-age sections.', ha='center', fontsize=11)
    stem = output / f"{age.replace('.', '_')}_multiplane_sections"
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(stem.with_suffix('.'+extension), dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    regions.FINAL_ROOT = args.input_root.resolve()
    validation, manifests, _ = regions.require_inputs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    sections, contexts, records = collect(args.input_root, manifests)
    norms = {}
    for gene in GENES:
        values = np.log1p(np.concatenate([section[gene].to_numpy() for section in sections.values()]))
        positive = values[values > 0]
        norms[gene] = mpl.colors.Normalize(0, float(np.quantile(positive, .99)) if len(positive) else 1, clip=True)
    for age in regions.AGES:
        render(age, sections, contexts, records, norms, args.output_dir, args.dpi)
        print(f'Rendered {age}', flush=True)
    pd.concat([frame.assign(age=age, plane=plane) for (age, plane), frame in sections.items()]).to_csv(
        args.output_dir / 'section_plot_data.csv.gz', index=False)
    (args.output_dir/'provenance.json').write_text(json.dumps(dict(
        input_root=str(args.input_root.resolve()), entry_point=str(Path(__file__).resolve()),
        configuration_id=validation['configuration_id'], genes=list(GENES), planes=records,
        panels='neutral 3D plane locator and four gene-expression panels; no region or cell-type coloring',
        center_rule='middle unique native coordinate independently on each axis within each age; XZ oblique translated +0.4 along its unit normal',
        selection='absolute signed distance to plane < 0.01 + 1e-8; all selected voxels shown',
        interpolation=False, smoothing=False, expression='log1p raw generated counts',
        color_rule='pooled positive 99th percentile per gene over displayed sections, shared across ages and planes',
        color_limits={gene:norm.vmax for gene,norm in norms.items()},
        context_sampling='up to 700 voxels per original z slice; seed 2026 + age_index * 100000 + z_index',
        png_dpi=args.dpi), indent=2)+'\n')
    print(args.output_dir, flush=True)


if __name__ == '__main__':
    main()
