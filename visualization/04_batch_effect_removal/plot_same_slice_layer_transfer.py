"""Evaluate and display reference-to-query layer transfer in saved same-slice results."""
from __future__ import annotations

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/feast-numba-cache')

import json
from pathlib import Path

import plot_additional as core
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

ALPHA = 1.0
K = 30
METHODS = ['Uncorrected', *core.METHOD_ORDER]
STEM = 'same_slice_layer_transfer_spatial'


def load_input(path, genes):
    data = ad.read_h5ad(path, backed='r')
    try:
        return (
            core.dense(data[:, genes].X).astype(np.float32, copy=False),
            data.obs_names.astype(str).to_numpy(copy=True),
            data.obs['ground_truth'].astype(str).to_numpy(copy=True),
            np.asarray(data.obsm['spatial'], dtype=np.float64).copy(),
        )
    finally:
        data.file.close()


def load_mode(mode, manifest, genes, primary):
    paths = [Path(manifest.loc[manifest['mode'].eq(mode) & manifest['alpha'].eq(alpha),
                               'file'].item()) for alpha in (0.0, ALPHA)]
    ref_counts, ref_ids, ref_labels, ref_spatial = load_input(paths[0], genes)
    query_counts, query_ids, query_labels, query_spatial = load_input(paths[1], genes)
    if len(ref_ids) != core.N_SPOTS or len(set(query_ids)) != core.N_SPOTS:
        raise ValueError(f'Unexpected spot support for {mode}')
    if not np.array_equal(ref_ids, query_ids):
        raise ValueError(f'Reference/query spot order differs for {mode}')
    if not np.array_equal(ref_labels, query_labels):
        raise ValueError(f'Known cortical labels differ for {mode}')
    if not np.array_equal(ref_spatial, query_spatial):
        raise ValueError(f'Original tissue coordinates differ for {mode}')
    if set(ref_labels) != set(core.DOMAIN_ORDER):
        raise ValueError(f'Unexpected cortical layers for {mode}')
    uncorrected, zero_library_spots = core.uncorrected_pca(np.vstack([ref_counts, query_counts]))
    representations = {'Uncorrected': uncorrected}
    for method in core.METHOD_ORDER:
        values, result_path = core.load_primary_embedding(primary, method, mode, ALPHA)
        core.validate_result_order(result_path, ref_ids)
        result = ad.read_h5ad(result_path, backed='r')
        try:
            if not np.array_equal(result.obs['dlpfc_layer'].astype(str),
                                  np.concatenate([ref_labels, query_labels])):
                raise ValueError(f'Saved layer-label order differs for {method}/{mode}')
        finally:
            result.file.close()
        representations[method] = values
        paths.extend([result_path, result_path.parent / 'embeddings.npz'])
    return representations, ref_labels, query_labels, query_ids, query_spatial, paths, zero_library_spots


def predict_and_verify(values, reference_labels, query_labels):
    """Only embeddings and reference labels enter the prediction step."""
    n_ref = len(reference_labels)
    if values.shape[0] != n_ref + len(query_labels) or not np.isfinite(values).all():
        raise ValueError('Invalid standardized embedding support')
    classifier = KNeighborsClassifier(n_neighbors=K, weights='uniform',
                                      metric='minkowski', p=2, n_jobs=1)
    classifier.fit(values[:n_ref], reference_labels)
    predicted = classifier.predict(values[n_ref:])

    # Independently count votes; tied class counts follow sklearn's sorted-label rule.
    indices = classifier.kneighbors(values[n_ref:], return_distance=False)
    classes = np.unique(reference_labels)
    counts = np.column_stack([(reference_labels[indices] == label).sum(axis=1)
                              for label in classes])
    if not np.array_equal(predicted, classes[counts.argmax(axis=1)]):
        raise ValueError('Prediction disagrees with explicit 30-neighbor majority vote')

    # Evaluation starts here. Query labels never select neighbors or assign predictions.
    score = float(f1_score(query_labels, predicted, labels=core.DOMAIN_ORDER,
                           average='macro', zero_division=0))
    per_layer = []
    for label in core.DOMAIN_ORDER:
        tp = int(np.sum((query_labels == label) & (predicted == label)))
        fp = int(np.sum((query_labels != label) & (predicted == label)))
        fn = int(np.sum((query_labels == label) & (predicted != label)))
        denominator = 2 * tp + fp + fn
        per_layer.append({'layer': label, 'tp': tp, 'fp': fp, 'fn': fn,
                          'f1': 2 * tp / denominator if denominator else 0.0})
    reconstructed = float(np.mean([row['f1'] for row in per_layer]))
    if not np.isclose(score, reconstructed, atol=1e-12, rtol=0):
        raise ValueError('Macro-F1 differs from independent per-layer reconstruction')
    return predicted, score, per_layer


def draw_page(mode, spatial, ground_truth, predictions, scores, limits):
    fig, axes = plt.subplots(1, 5, figsize=(18.5, 6.3))
    panels = [('Ground truth', ground_truth, 'Slice 151673')]
    panels.extend(('Uncorrected input' if method == 'Uncorrected' else method,
                   predictions[method], f'Macro-F1 = {scores[method]:.3f}') for method in METHODS)
    for index, (ax, (title, labels, subtitle)) in enumerate(zip(axes, panels, strict=True)):
        ax.scatter(spatial[:, 0], spatial[:, 1], c=[core.DOMAIN_COLORS[label] for label in labels],
                   s=6.5, linewidths=0, rasterized=True)
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1][::-1])
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        ax.set_title(f'{title}\n{subtitle}', fontsize=17, fontweight='bold',
                     linespacing=1.45, pad=12)
        ax.text(-0.04, 1.03, chr(65 + index), transform=ax.transAxes,
                fontsize=18, fontweight='bold')
    fig.suptitle(f'Same-slice cortical-layer transfer · {core.MODE_LABELS[mode]}',
                 fontsize=23, fontweight='bold', y=0.99)
    fig.text(0.5, 0.907, 'Reference α=0 → query α=1', ha='center', fontsize=16, color='#555555')
    handles = [Line2D([], [], marker='o', linestyle='none', color=core.DOMAIN_COLORS[label],
                      markersize=10, label=label.replace('_', ' ')) for label in core.DOMAIN_ORDER]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.015),
               ncol=7, frameon=False, fontsize=15, columnspacing=1.25, handletextpad=0.4)
    fig.subplots_adjust(left=0.025, right=0.99, bottom=0.13, top=0.76, wspace=0.12)
    fig.canvas.draw()
    boxes = [ax.get_window_extent() for ax in axes]
    if np.ptp([box.width for box in boxes]) > 1e-6 or np.ptp([box.height for box in boxes]) > 1e-6:
        raise ValueError('Spatial panel sizes differ')
    return fig


def main():
    atomic, _, candidate, sources = core.load_inputs()
    primary = core.primary_rows(atomic)
    manifest_path = core.selected_source(candidate, 'simulation_manifest')
    panel_path = core.selected_source(candidate, 'panel')
    manifest = pd.read_csv(manifest_path)
    genes = pd.read_csv(panel_path).sort_values('panel_order')['gene'].astype(str).tolist()
    core.configure_matplotlib()
    contexts = {mode: load_mode(mode, manifest, genes, primary) for mode in core.MODE_ORDER}
    first = contexts[core.MODE_ORDER[0]]
    for mode, data in contexts.items():
        if not np.array_equal(data[3], first[3]) or not np.array_equal(data[4], first[4]):
            raise ValueError(f'Query tissue support differs across modes: {mode}')
    spatial = first[4]
    low, high = spatial.min(axis=0), spatial.max(axis=0)
    padding = (high - low) * 0.04
    limits = [(float(low[i] - padding[i]), float(high[i] + padding[i])) for i in range(2)]
    output_dir = core.OUTPUT_DIR
    pdf_path = output_dir / f'{STEM}.pdf'
    outputs, plot_records, score_records, layer_records = [pdf_path], [], [], []
    selected_sources, zero_counts = [], {}
    with PdfPages(pdf_path, metadata={'Creator': Path(__file__).name,
                                    'CreationDate': None, 'ModDate': None}) as pdf:
        for mode, data in contexts.items():
            representations, ref_labels, query_labels, ids, coordinates, paths, zero_count = data
            selected_sources.extend(paths)
            zero_counts[mode] = zero_count
            predictions, scores = {}, {}
            for method, values in representations.items():
                predicted, score, per_layer = predict_and_verify(values, ref_labels, query_labels)
                predictions[method], scores[method] = predicted, score
                plot_records.append(pd.DataFrame({'mode': mode, 'alpha': ALPHA, 'method': method,
                    'source_spot_id': ids, 'x': coordinates[:, 0], 'y': coordinates[:, 1],
                    'ground_truth': query_labels, 'predicted_layer': predicted,
                    'correct': predicted == query_labels}))
                score_records.append({'mode': mode, 'alpha': ALPHA, 'method': method,
                                      'n_reference': len(ref_labels), 'n_query': len(query_labels),
                                      'k': K, 'macro_f1': score})
                layer_records.extend({'mode': mode, 'alpha': ALPHA, 'method': method, **row}
                                     for row in per_layer)
                print(f'{mode}, {method}: macro-F1={score:.6f}; votes and score verified', flush=True)
            fig = draw_page(mode, coordinates, query_labels, predictions, scores, limits)
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            png_path = output_dir / f'{STEM}_{mode}.png'
            fig.savefig(png_path, bbox_inches='tight', dpi=300)
            outputs.append(png_path)
            plt.close(fig)
    for suffix, table in [('predictions', pd.concat(plot_records, ignore_index=True)),
                          ('scores', pd.DataFrame(score_records)),
                          ('per_layer_scores', pd.DataFrame(layer_records))]:
        path = output_dir / f'{STEM}_{suffix}.csv'
        table.to_csv(path, index=False)
        outputs.append(path)
    provenance = {
        'source': core.repository_path(Path(__file__)),
        'inputs': {key: core.repository_path(path) for key, path in
                   {**sources, 'simulation_manifest': manifest_path, 'panel': panel_path}.items()},
        'saved_artifacts': [core.repository_path(path) for path in selected_sources],
        'slice': '151673', 'modes': core.MODE_ORDER, 'reference_alpha': 0.0, 'query_alpha': ALPHA,
        'representations': {'Uncorrected': 'Existing fixed-panel library normalization to 10000, log1p, PCA-20 seed42, standardization',
                            'GraphST': 'Primary PCA-20 seed42, standardized',
                            'STAMP': 'Native 10D topics, standardized', 'scVI': 'Native 10D latent, standardized'},
        'classifier': {'name': 'KNeighborsClassifier', 'n_neighbors': K, 'weights': 'uniform',
                       'distance': 'Euclidean', 'class_vote_ties': 'First class in sorted label order'},
        'prediction_inputs': 'Reference/query embeddings and reference ground-truth labels only.',
        'query_labels': 'Used only for evaluation and the ground-truth display.',
        'spot_ids': 'Used only to verify ordering and identify exported rows; never used for prediction.',
        'spatial_coordinates': 'Used only for alignment checks and plotting; never used for prediction.',
        'verification': 'All spot/batch/label orders verified; majority votes explicitly counted; macro-F1 independently reconstructed from TP/FP/FN; identical panel sizes checked.',
        'interpretation': 'Biological-label preservation, complementary to exact-pair recovery of individual spot identity. This endpoint comparison alone does not quantify a change from alpha=0 performance.',
        'zero_library_spots': zero_counts, 'spatial_limits': limits, 'y_axis_inverted': True,
        'software': {'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'publication_status': 'supplementary_unpromoted_author_review_required',
        'figure_promotion_authorized': False, 'composite_score_used': False,
        'winner_ranking_authorized': False,
        'outputs': [core.repository_path(path) for path in outputs],
    }
    provenance_path = output_dir / f'{STEM}_provenance.json'
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    for path in [*outputs, provenance_path]:
        print(path, flush=True)


if __name__ == '__main__':
    main()
