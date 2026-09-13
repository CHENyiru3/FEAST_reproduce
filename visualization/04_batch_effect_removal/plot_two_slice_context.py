"""Visualize saved two-slice embeddings and the existing layer-transfer task."""
from __future__ import annotations

import os
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/feast-numba-cache')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import json
from pathlib import Path

import plot_real_slice_robustness as core
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

CONDITION = 'sim_1.00'
SEED = 42
K = 30
UMAP_SETTINGS = dict(n_neighbors=30, min_dist=0.3, metric='euclidean',
                     random_state=SEED, n_jobs=1)
METHODS = ['Uncorrected', *core.METHOD_ORDER]
LAYER_COLORS = {'Layer_1': '#E69F00', 'Layer_2': '#56B4E9',
                'Layer_3': '#009E73', 'Layer_4': '#F0E442',
                'Layer_5': '#0072B2', 'Layer_6': '#D55E00', 'WM': '#CC79A7'}
BATCH_COLORS = {'ref': '#0072B2', 'query': '#D55E00'}


def standardize(values):
    """Match real_slice_robustness/metrics.py preprocessing exactly."""
    values = np.asarray(values, dtype=np.float64)
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    return (values - values.mean(axis=0)) / scale


def load_context():
    atomic, _, _ = core.load_validated_inputs()
    baseline_path = core.STUDY_ROOT / 'baselines' / f'{CONDITION}.npz'
    with np.load(baseline_path, allow_pickle=True) as payload:
        baseline = np.asarray(payload['embedding'], dtype=np.float64)
        batch = payload['batch_labels'].astype(str)
        layers = payload['layer_labels'].astype(str)
        ids = payload['source_spot_ids'].astype(str)
        spatial = np.asarray(payload['spatial'], dtype=np.float64)
    if set(layers) != set(LAYER_COLORS):
        raise ValueError('Unexpected cortical-layer labels')
    if pd.MultiIndex.from_arrays([batch, ids]).has_duplicates:
        raise ValueError('Duplicate spot identifiers within a section')
    representations = {'Uncorrected': standardize(baseline)}
    sources = [baseline_path, core.METRICS_PATH, core.VALIDATION_PATH,
               core.QUALIFICATION_PATH]
    for method in core.METHOD_ORDER:
        row = atomic.loc[atomic['condition'].eq(CONDITION)
                         & atomic['method'].eq(method)
                         & atomic['method_seed'].eq(SEED)
                         & atomic['analysis_role'].eq('primary')].squeeze()
        candidate = Path(row['candidate_path'])
        with np.load(candidate / 'embeddings.npz', allow_pickle=True) as payload:
            values = np.asarray(payload['embedding'], dtype=np.float64)
            if not np.array_equal(payload['batch_labels'].astype(str), batch):
                raise ValueError(f'Embedding batch order differs for {method}')
        result = ad.read_h5ad(candidate / 'result.h5ad', backed='r')
        try:
            if not np.array_equal(result.obs['source_spot_id'].astype(str), ids):
                raise ValueError(f'Spot order differs for {method}')
            if not np.array_equal(result.obs['batch'].astype(str), batch):
                raise ValueError(f'Result batch order differs for {method}')
        finally:
            result.file.close()
        if method == 'GraphST':
            values = PCA(n_components=10, svd_solver='randomized',
                         random_state=SEED).fit_transform(values)
        representations[method] = standardize(values)
        sources.extend([candidate / 'embeddings.npz', candidate / 'result.h5ad'])
    for method, values in representations.items():
        if values.shape != (len(ids), 10) or not np.isfinite(values).all():
            raise ValueError(f'Invalid primary representation for {method}')
    context = pd.DataFrame(dict(batch=batch, source_spot_id=ids, layer=layers,
                                x=spatial[:, 0], y=spatial[:, 1]))
    return atomic, representations, context, sources


def predict_layers(atomic, representations, context):
    reference = context['batch'].eq('ref').to_numpy()
    layers = context['layer'].to_numpy()
    predictions, scores, records = {}, {}, []
    for method, values in representations.items():
        classifier = KNeighborsClassifier(n_neighbors=K)
        predicted = classifier.fit(values[reference], layers[reference]).predict(values[~reference])
        score = f1_score(layers[~reference], predicted,
                         labels=sorted(LAYER_COLORS), average='macro')
        selected = atomic.loc[atomic['condition'].eq(CONDITION)
                              & atomic['method'].eq(method)
                              & atomic['analysis_role'].eq(
                                  'uncorrected_baseline' if method == 'Uncorrected' else 'primary')]
        if method != 'Uncorrected':
            selected = selected.loc[selected['method_seed'].eq(SEED)]
        expected = float(selected['reference_to_query_layer_macro_f1'].item())
        if not np.isclose(score, expected, rtol=0, atol=1e-12):
            raise ValueError(f'Layer-transfer reconstruction differs for {method}: {score} vs {expected}')
        predictions[method], scores[method] = predicted, score
        frame = context.loc[~reference].copy()
        frame['method'] = method
        frame['predicted_layer'] = predicted
        frame['correct'] = predicted == layers[~reference]
        frame['macro_f1'] = score
        records.append(frame)
        print(f'{method}: reconstructed layer macro-F1 = {score:.6f}', flush=True)
    return predictions, scores, pd.concat(records, ignore_index=True)


def layer_legend(fig):
    handles = [Line2D([], [], marker='o', linestyle='none', color=color,
                      markersize=9, label=label.replace('_', ' '))
               for label, color in LAYER_COLORS.items()]
    fig.legend(handles=handles, loc='lower center', ncol=7, frameon=False,
               fontsize=13, bbox_to_anchor=(0.5, 0.005), handletextpad=0.3,
               columnspacing=1.1)


def save_figure(fig, stem):
    outputs = []
    for suffix in ('pdf', 'png', 'svg'):
        path = core.OUTPUT_DIR / f'{stem}.{suffix}'
        metadata = {'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else {}
        fig.savefig(path, bbox_inches='tight', dpi=300, metadata=metadata)
        outputs.append(path)
    plt.close(fig)
    return outputs


def draw_embedding(representations, context):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    order = np.random.default_rng(SEED).permutation(len(context))
    records = []
    for col, method in enumerate(METHODS):
        print(f'Projecting {method} with UMAP...', flush=True)
        projection = umap.UMAP(**UMAP_SETTINGS).fit_transform(representations[method])
        if projection.shape != (len(context), 2) or not np.isfinite(projection).all():
            raise ValueError(f'Invalid UMAP coordinates for {method}')
        frame = context.copy()
        frame['method'] = method
        frame['umap_1'], frame['umap_2'] = projection.T
        records.append(frame)
        for row, key, palette in [(0, 'batch', BATCH_COLORS), (1, 'layer', LAYER_COLORS)]:
            ax = axes[row, col]
            colors = context[key].map(palette).to_numpy()
            ax.scatter(projection[order, 0], projection[order, 1], c=colors[order],
                       s=3, alpha=0.85, linewidths=0, rasterized=True)
            ax.set_aspect('equal', adjustable='datalim')
            ax.axis('off')
        axes[0, col].set_title('Uncorrected input' if method == 'Uncorrected' else method,
                               fontsize=17, fontweight='bold', pad=9)
    for row, label in enumerate(['Slice', 'Cortical layer']):
        axes[row, 0].text(-0.08, 0.5, label, transform=axes[row, 0].transAxes,
                          rotation=90, va='center', ha='right', fontsize=16,
                          fontweight='bold')
    handles = [Line2D([], [], marker='o', linestyle='none', color=BATCH_COLORS[batch],
                      markersize=9, label=label)
               for batch, label in [('ref', 'Reference · 151675'), ('query', 'Query · 151676')]]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.905),
               ncol=2, fontsize=14, frameon=False)
    fig.suptitle('Two-slice embedding context', fontsize=22, fontweight='bold', y=0.995)
    fig.text(0.5, 0.944, 'α=1 endpoint · seed 42 · independent UMAP per method',
             ha='center', fontsize=14, color='#555555')
    layer_legend(fig)
    fig.tight_layout(rect=(0.035, 0.07, 0.995, 0.86), w_pad=0.8, h_pad=1.5)
    return save_figure(fig, 'two_slice_embedding_context'), pd.concat(records, ignore_index=True)


def draw_spatial(context, predictions, scores):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 10.5))
    reference = context.loc[context['batch'].eq('ref')]
    query = context.loc[context['batch'].eq('query')]
    panels = [(reference, reference['layer'], 'Reference · known layers\n151675'),
              (query, query['layer'], 'Query · known layers\n151676')]
    for method in METHODS:
        label = 'Uncorrected input' if method == 'Uncorrected' else method
        panels.append((query, predictions[method],
                       f'{label} · transferred layers\nMacro-F1 = {scores[method]:.3f}'))
    for index, (ax, (frame, labels, title)) in enumerate(zip(axes.flat, panels, strict=True)):
        ax.scatter(frame['x'], frame['y'], c=[LAYER_COLORS[label] for label in labels],
                   s=7, linewidths=0, rasterized=True)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=9, linespacing=1.4)
        ax.text(-0.06, 1.02, chr(65 + index), transform=ax.transAxes,
                fontsize=18, fontweight='bold')
    fig.suptitle('Two-slice cortical-layer transfer', fontsize=22, fontweight='bold', y=0.995)
    fig.text(0.5, 0.944, 'Reference → query · α=1 endpoint · seed 42',
             ha='center', fontsize=14, color='#555555')
    layer_legend(fig)
    fig.tight_layout(rect=(0.015, 0.06, 0.99, 0.915), h_pad=2.0, w_pad=1.7)
    return save_figure(fig, 'two_slice_layer_transfer_spatial')


def main():
    core.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    core.configure_matplotlib()
    atomic, representations, context, sources = load_context()
    predictions, scores, prediction_data = predict_layers(atomic, representations, context)
    outputs = draw_spatial(context, predictions, scores)
    embedding_outputs, embedding_data = draw_embedding(representations, context)
    outputs.extend(embedding_outputs)
    for stem, frame in [('two_slice_layer_transfer_spatial_plot_data', prediction_data),
                        ('two_slice_embedding_context_plot_data', embedding_data),
                        ('two_slice_spatial_annotations', context)]:
        path = core.OUTPUT_DIR / f'{stem}.csv'
        frame.to_csv(path, index=False)
        outputs.append(path)
    provenance = {
        'source': core.repository_path(Path(__file__)),
        'inputs': [core.repository_path(path) for path in sources],
        'outputs': [core.repository_path(path) for path in outputs],
        'condition': CONDITION, 'method_seed': SEED,
        'selection': 'Registered primary endpoint and first registered seed; no performance-based selection.',
        'representations': {'Uncorrected': 'fixed_panel_log1p_pca10_seed42', **core.METHOD_REPRESENTATIONS},
        'preprocessing': 'Scorer standardization across all reference and query spots.',
        'umap': UMAP_SETTINGS,
        'umap_interpretation': 'Independent fit per method; the two color views reuse identical coordinates. Absolute positions and distances cannot be compared across methods.',
        'layer_transfer': {'classifier': 'KNeighborsClassifier', 'n_neighbors': K,
                           'weights': 'uniform', 'metric': 'minkowski', 'p': 2,
                           'training': 'reference labels only', 'evaluation': 'query labels',
                           'reconstructed_macro_f1': scores, 'maximum_allowed_reconstruction_error': 1e-12},
        'spatial_interpretation': 'Known cortical labels and predicted labels, not newly fitted unsupervised clusters. Sections have distinct spots and are not spatially aligned.',
        'software': {'numpy': np.__version__, 'umap': umap.__version__},
        'publication_status': 'supplementary_unpromoted_author_review_required',
        'figure_promotion_authorized': False, 'winner_ranking_authorized': False,
        'composite_score_used': False,
    }
    path = core.OUTPUT_DIR / 'two_slice_context_provenance.json'
    path.write_text(json.dumps(provenance, indent=2) + '\n')
    for output in [*outputs, path]:
        print(output, flush=True)


if __name__ == '__main__':
    main()
