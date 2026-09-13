"""Show the saved uncorrected same-slice input ladder, one PDF page per mode."""
from __future__ import annotations

import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/feast-numba-cache')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import json
from pathlib import Path

import plot_additional as context
import plot_parameter_cloud as parameters
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

MAX_ALPHA = 1.5
STEM = 'uncorrected_embedding_alpha_ladder'
PARAMETER_LIMITS = {**parameters.DISPLAY_LIMITS, 'log_variance': (-3.9, -0.5)}


def load_parameter_data(alphas):
    reference, _ = parameters.load_cloud('shift_only', 0.0)
    selected = np.logical_and.reduce([
        reference[coordinate].between(lower, upper).to_numpy()
        for coordinate, (lower, upper) in parameters.REFERENCE_DISPLAY_REGION.items()
    ])
    genes = reference.index[selected]
    records = []
    reference_columns = [f'{name}_reference' for name in PARAMETER_LIMITS]
    for mode in context.MODE_ORDER:
        for alpha in alphas:
            cloud, _ = parameters.load_cloud(mode, alpha)
            if not reference[reference_columns].equals(cloud[reference_columns]):
                raise ValueError(f'Reference gene parameters differ: {mode}, {alpha}')
            frame = cloud.loc[genes, list(PARAMETER_LIMITS)].copy()
            for name, (lower, upper) in PARAMETER_LIMITS.items():
                if not frame[name].between(lower, upper).all():
                    raise ValueError(f'Parameter limits would clip {name}: {mode}, {alpha}')
            frame.insert(0, 'gene', frame.index.astype(str))
            frame.insert(0, 'alpha', float(alpha))
            frame.insert(0, 'mode', mode)
            records.append(frame.reset_index(drop=True))
    return pd.concat(records, ignore_index=True)


def parameter_metadata():
    return {
        'source': context.repository_path(Path(parameters.__file__)),
        'reference_selection': parameters.REFERENCE_DISPLAY_REGION,
        'display_limits': PARAMETER_LIMITS,
        'view': {'elevation': 12, 'azimuth': -45},
        'coordinates': ['log mean', 'log variance', 'zero proportion'],
        'meaning': 'Stored gene-level simulation parameters; the same reference-selected genes in every panel, distinct from spots in UMAP.',
        'limit_change': 'Shared log-variance lower limit expanded to -3.9 to include the alpha=1.5 stress tests without clipping.',
    }


def load_input(path, genes):
    data = ad.read_h5ad(path, backed='r')
    try:
        counts = context.dense(data[:, genes].X).astype(np.float32, copy=False)
        ids = data.obs_names.astype(str).to_numpy(copy=True)
        layers = data.obs['ground_truth'].astype(str).to_numpy(copy=True)
    finally:
        data.file.close()
    return counts, ids, layers


def draw_mode(mode, frames, alphas, parameter_data):
    fig = plt.figure(figsize=(3.5 * len(alphas), 13.2))
    grid = fig.add_gridspec(3, len(alphas), height_ratios=[1, 1, 1.12])
    axes = np.array([[fig.add_subplot(grid[row, col]) for col in range(len(alphas))]
                     for row in range(2)])
    for column, (alpha, frame) in enumerate(zip(alphas, frames, strict=True)):
        order = np.random.default_rng(context.PUBLIC_SEED).permutation(len(frame))
        coordinates = frame[['umap_1', 'umap_2']].to_numpy()
        lower, upper = coordinates.min(axis=0), coordinates.max(axis=0)
        center = (lower + upper) / 2
        # Equal square viewports with a uniform scale in both dimensions:
        # every cloud fills the same longest extent without changing its shape.
        half_span = float((upper - lower).max()) * 0.54
        for row, key, palette in [(0, 'batch', context.BATCH_COLORS),
                                   (1, 'ground_truth', context.DOMAIN_COLORS)]:
            ax = axes[row, column]
            ax.scatter(frame['umap_1'].to_numpy()[order], frame['umap_2'].to_numpy()[order],
                       c=frame[key].map(palette).to_numpy()[order], s=2.3,
                       alpha=0.88, linewidths=0, rasterized=True)
            ax.set_xlim(center[0] - half_span, center[0] + half_span)
            ax.set_ylim(center[1] - half_span, center[1] + half_span)
            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')
        subtitle = ' '
        if alpha == 0:
            subtitle = 'Identical inputs'
        elif alpha > 1:
            subtitle = 'Stress test'
        title = f'α = {alpha:g}\n{subtitle}'
        axes[0, column].set_title(title, fontsize=17, fontweight='bold', pad=10)
        cloud = parameter_data.loc[parameter_data['mode'].eq(mode)
                                   & parameter_data['alpha'].eq(alpha)]
        ax = fig.add_subplot(grid[2, column], projection='3d')
        ax.scatter(cloud['log_mean'], cloud['log_variance'], cloud['zero_proportion'],
                   color=context.BATCH_COLORS['query'], s=2.2, alpha=0.28,
                   depthshade=False, linewidths=0, rasterized=True)
        ax.set(xlim=PARAMETER_LIMITS['log_mean'], ylim=PARAMETER_LIMITS['log_variance'],
               zlim=PARAMETER_LIMITS['zero_proportion'])
        ax.set_xticks([-4, -3, -2])
        ax.set_yticks([-3.5, -2.5, -1.5])
        ax.set_zticks([0.8, 0.9, 1.0])
        ax.tick_params(labelsize=10, pad=0)
        ax.set_xlabel('log mean', fontsize=12, labelpad=3)
        ax.set_ylabel('log variance', fontsize=12, labelpad=3)
        # A top label remains readable in a seven-column grid; a conventional
        # right-side z label would overlap the neighboring 3D panel.
        ax.set_zlabel('')
        ax.text2D(0.5, 0.98, 'z: zero proportion', transform=ax.transAxes,
                  ha='center', fontsize=12, color='#555555')
        ax.view_init(elev=12, azim=-45)
        ax.set_box_aspect((1, 1, 0.85))
        if column == 0:
            ax.text2D(-0.13, 0.5, 'Gene parameters', transform=ax.transAxes,
                      rotation=90, va='center', ha='right', fontsize=17, fontweight='bold')
    for row, title in enumerate(['Batch', 'Cortical layer']):
        axes[row, 0].text(-0.12, 0.5, title, transform=axes[row, 0].transAxes,
                          rotation=90, va='center', ha='right', fontsize=17,
                          fontweight='bold')
    batch_handles = [Line2D([], [], marker='o', linestyle='none',
                            color=context.BATCH_COLORS[batch], markersize=10, label=label)
                     for batch, label in [('ref', 'Reference (α=0)'), ('query', 'Query')]]
    fig.legend(handles=batch_handles, loc='upper center', bbox_to_anchor=(0.5, 0.925),
               ncol=2, frameon=False, fontsize=15)
    layer_handles = [Line2D([], [], marker='o', linestyle='none',
                            color=context.DOMAIN_COLORS[label], markersize=10,
                            label=label.replace('_', ' ')) for label in context.DOMAIN_ORDER]
    fig.legend(handles=layer_handles, loc='lower center', bbox_to_anchor=(0.5, 0.343),
               ncol=7, frameon=False, fontsize=14, columnspacing=1.4)
    fig.suptitle(f'Uncorrected embeddings and gene parameters · {context.MODE_LABELS[mode]}',
                 fontsize=23, fontweight='bold', y=0.995)
    fig.text(0.5, 0.955, 'Same slice 151673 · independent UMAP per α',
             fontsize=15, color='#555555', ha='center')
    fig.text(0.5, 0.014, f'Gene parameters: the same {parameter_data.gene.nunique():,} reference-selected genes · fixed axes and view',
             fontsize=14, color='#555555', ha='center')
    fig.subplots_adjust(left=0.035, right=0.98, bottom=0.105, top=0.84,
                        wspace=0.12, hspace=0.12)
    return fig


def main():
    _, _, candidate, sources = context.load_inputs()
    manifest_path = context.selected_source(candidate, 'simulation_manifest')
    panel_path = context.selected_source(candidate, 'panel')
    manifest = pd.read_csv(manifest_path)
    parameter_data = load_parameter_data(sorted(manifest.loc[manifest['alpha'].le(MAX_ALPHA), 'alpha'].unique()))
    genes = pd.read_csv(panel_path).sort_values('panel_order')['gene'].astype(str).tolist()
    context.configure_matplotlib()
    output_dir = context.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f'{STEM}.pdf'
    records, selected_sources, outputs = [], [], [pdf_path]
    with PdfPages(pdf_path, metadata={'Creator': Path(__file__).name,
                                    'CreationDate': None, 'ModDate': None}) as pdf:
        for mode in context.MODE_ORDER:
            rows = manifest.loc[manifest['mode'].eq(mode) & manifest['alpha'].le(MAX_ALPHA)].sort_values('alpha')
            ref_path = Path(rows.loc[rows['alpha'].eq(0), 'file'].item())
            reference, ids, layers = load_input(ref_path, genes)
            if len(ids) != context.N_SPOTS or set(layers) != set(context.DOMAIN_ORDER):
                raise ValueError(f'Unexpected reference spot or layer support: {mode}')
            frames = []
            for row in rows.itertuples(index=False):
                path = Path(row.file)
                counts, query_ids, query_layers = load_input(path, genes)
                if not np.array_equal(ids, query_ids) or not np.array_equal(layers, query_layers):
                    raise ValueError(f'Spot order or cortical labels changed: {mode}, {row.alpha}')
                print(f'Projecting uncorrected {mode}, alpha={row.alpha:g}', flush=True)
                values, zero_library_spots = context.uncorrected_pca(np.vstack([reference, counts]))
                projection = context.umap_projection(values)
                frame = pd.DataFrame({
                    'mode': mode, 'alpha': row.alpha, 'alpha_stratum': row.alpha_stratum,
                    'source_spot_id': np.tile(ids, 2),
                    'batch': np.repeat(['ref', 'query'], len(ids)),
                    'ground_truth': np.tile(layers, 2),
                    'umap_1': projection[:, 0], 'umap_2': projection[:, 1],
                })
                frames.append(frame)
                records.append(frame)
                selected_sources.append({'mode': mode, 'alpha': row.alpha,
                                         'query': context.repository_path(path),
                                         'reference': context.repository_path(ref_path),
                                         'zero_library_spots': zero_library_spots})
            fig = draw_mode(mode, frames, rows['alpha'].tolist(), parameter_data)
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            for suffix in ('png', 'svg'):
                path = output_dir / f'{STEM}_{mode}.{suffix}'
                fig.savefig(path, bbox_inches='tight', dpi=300)
                outputs.append(path)
            plt.close(fig)
    data_path = output_dir / f'{STEM}_plot_data.csv'
    pd.concat(records, ignore_index=True).to_csv(data_path, index=False)
    outputs.append(data_path)
    parameter_path = output_dir / f'{STEM}_parameter_cloud_plot_data.csv'
    parameter_data.to_csv(parameter_path, index=False)
    outputs.append(parameter_path)
    provenance = {
        'source': context.repository_path(Path(__file__)),
        'inputs': {key: context.repository_path(path) for key, path in
                   {**sources, 'simulation_manifest': manifest_path, 'panel': panel_path}.items()},
        'selected_inputs': selected_sources,
        'outputs': [context.repository_path(path) for path in outputs],
        'maximum_displayed_alpha': MAX_ALPHA,
        'representation': 'Same as plot_additional.py: fixed-panel counts, library-size normalization to 10000, log1p, PCA-20 with seed 42, column standardization.',
        'umap': context.UMAP_SETTINGS,
        'parameter_cloud': parameter_metadata(),
        'display_layout': 'Equal square panels; centered per-projection limits with 4% padding on each side of the longest extent; uniform x/y scale preserves shape.',
        'interpretation': 'Independent UMAP per mode/alpha; both color rows reuse identical coordinates. Compare mixing within panels, not absolute geometry between panels.',
        'alpha_zero': 'Reference and query contain identical copies of the alpha=0 input; no independent batch is implied.',
        'above_alpha_one': 'Extrapolation stress tests; the registered primary endpoint remains alpha=1.',
        'publication_status': 'supplementary_unpromoted_author_review_required',
        'composite_score_used': False, 'winner_ranking_authorized': False,
        'figure_promotion_authorized': False,
    }
    provenance_path = output_dir / f'{STEM}_provenance.json'
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    for path in [*outputs, provenance_path]:
        print(path, flush=True)


if __name__ == '__main__':
    main()
