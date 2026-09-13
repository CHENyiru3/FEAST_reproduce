"""Combine saved diagonal-affine UMAPs and spatial layer-transfer predictions.

Run with the reproduction Python environment. Coordinates and predictions are
read from the original plot-data CSVs; no UMAP or correction model is fitted.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/feast-reproduce-matplotlib')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D


OUTPUT = Path(__file__).resolve().parent / 'figures' / 'same_slice'
STEM = 'diagonal_affine_embedding_layer_transfer_combined'
METHODS = ['Uncorrected', 'GraphST', 'STAMP', 'scVI']
LAYERS = ['Layer_1', 'Layer_2', 'Layer_3', 'Layer_4', 'Layer_5', 'Layer_6', 'WM']
LAYER_COLORS = dict(zip(LAYERS, [
    '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
]))
BATCH_COLORS = {'ref': '#0072B2', 'query': '#D55E00'}
SOURCES = {
    'umap': 'representative_embedding_context_data.csv',
    'predictions': 'same_slice_layer_transfer_spatial_predictions.csv',
    'scores': 'same_slice_layer_transfer_spatial_scores.csv',
}


def read_inputs():
    tables = {}
    for key, filename in SOURCES.items():
        table = pd.read_csv(OUTPUT / filename)
        tables[key] = table.loc[table['mode'].eq('diagonal_affine') & table['alpha'].eq(1)].copy()
    embeddings, predictions, scores = (tables[key] for key in SOURCES)
    canonical = predictions.loc[predictions['method'].eq('Uncorrected')].set_index('source_spot_id')
    assert len(canonical) == 3611 and canonical.index.is_unique
    for method in METHODS:
        name = 'Uncorrected input' if method == 'Uncorrected' else method
        embedding = embeddings.loc[embeddings['representation'].eq(name)]
        prediction = predictions.loc[predictions['method'].eq(method)].set_index('source_spot_id')
        assert len(embedding) == 7222 and len(prediction) == 3611
        assert prediction.index.is_unique and set(prediction.index) == set(canonical.index)
        prediction = prediction.loc[canonical.index]
        pd.testing.assert_frame_equal(prediction[['x', 'y', 'ground_truth']],
                                      canonical[['x', 'y', 'ground_truth']])
        assert set(embedding['batch']) == {'ref', 'query'}
        for batch in BATCH_COLORS:
            labels = embedding.loc[embedding['batch'].eq(batch)].set_index('source_spot_id')
            assert labels.index.is_unique and set(labels.index) == set(canonical.index)
            assert labels.loc[canonical.index, 'ground_truth'].equals(canonical['ground_truth'])
        f1 = []
        for layer in LAYERS:
            actual, predicted = prediction['ground_truth'].eq(layer), prediction['predicted_layer'].eq(layer)
            tp = (actual & predicted).sum()
            f1.append(2 * tp / (actual.sum() + predicted.sum()))
        recorded = scores.loc[scores['method'].eq(method), 'macro_f1'].item()
        assert np.isclose(np.mean(f1), recorded, atol=1e-12, rtol=0)
    return embeddings, predictions, scores, canonical


def handles(colors):
    return [Line2D([], [], marker='o', linestyle='none', color=color,
                   markersize=7, label={'ref': 'Reference', 'query': 'Query'}.get(label, label.replace('_', ' ')))
            for label, color in colors.items()]


def draw_spatial(ax, frame, label_column, low, high):
    ax.scatter(frame['x'], frame['y'], c=frame[label_column].map(LAYER_COLORS),
               s=3.4, linewidths=0, rasterized=True)
    ax.set(xlim=(low[0], high[0]), ylim=(high[1], low[1]))
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


def main():
    for filename in ('arial.ttf', 'arialbd.ttf'):
        font_dir = Path(os.environ.get('FEAST_FONT_DIR', str(Path(sys.prefix) / 'fonts')))
        font_manager.fontManager.addfont(font_dir / filename)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    embeddings, predictions, scores, canonical = read_inputs()
    fig, axes = plt.subplots(3, 5, figsize=(15.4, 9.6))
    fig.subplots_adjust(left=0.02, right=0.985, top=0.855, bottom=0.13,
                        wspace=0.12, hspace=0.22)
    fig.suptitle('Same-slice batch mixing and cortical-layer preservation',
                 fontsize=22, fontweight='bold', y=0.98)
    fig.text(0.5, 0.93, 'DLPFC 151673 · diagonal affine · reference α=0 → query α=1',
             fontsize=15, color='#555555', ha='center')
    for ax in axes.flat:
        ax.axis('off')

    axes[0, 0].text(0.04, 0.94, 'A  Batch mixing', fontsize=17, fontweight='bold',
                    transform=axes[0, 0].transAxes)
    axes[0, 0].text(0.04, 0.77, 'UMAP · reference + query', fontsize=12,
                    color='#555555', transform=axes[0, 0].transAxes)
    axes[0, 0].legend(handles=handles(BATCH_COLORS), loc='upper left',
                      bbox_to_anchor=(0.0, 0.68), frameon=False, fontsize=13)
    axes[1, 0].text(0.04, 0.95, 'B  Layer structure', fontsize=17, fontweight='bold',
                    transform=axes[1, 0].transAxes)
    axes[1, 0].text(0.04, 0.78, 'UMAP · known layers', fontsize=12,
                    color='#555555', transform=axes[1, 0].transAxes)
    axes[1, 0].legend(handles=handles(LAYER_COLORS), loc='upper left',
                      bbox_to_anchor=(0.0, 0.69), frameon=False, ncol=2,
                      fontsize=11.5, columnspacing=0.5, handletextpad=0.2, labelspacing=0.6)

    xy = canonical[['x', 'y']].to_numpy()
    low, high = xy.min(axis=0), xy.max(axis=0)
    padding = (high - low) * 0.04
    low, high = low - padding, high + padding
    draw_spatial(axes[2, 0], canonical, 'ground_truth', low, high)
    axes[2, 0].set_title('C', loc='left', fontsize=17, fontweight='bold', pad=9)

    # Retain the original deterministic draw order for both UMAP colorings.
    order = np.random.default_rng(42).permutation(7222)
    for column, method in enumerate(METHODS, start=1):
        name = 'Uncorrected input' if method == 'Uncorrected' else method
        frame = embeddings.loc[embeddings['representation'].eq(name)].iloc[order]
        coords = frame[['umap_1', 'umap_2']].to_numpy()
        center = (coords.min(axis=0) + coords.max(axis=0)) / 2
        radius = np.ptp(coords, axis=0).max() * 0.54
        for row, color_key, palette in [(0, 'batch', BATCH_COLORS), (1, 'ground_truth', LAYER_COLORS)]:
            ax = axes[row, column]
            ax.scatter(coords[:, 0], coords[:, 1], c=frame[color_key].map(palette),
                       s=1.8, linewidths=0, alpha=0.88 if row == 0 else 0.9, rasterized=True)
            ax.set(xlim=(center[0] - radius, center[0] + radius),
                   ylim=(center[1] - radius, center[1] + radius))
            ax.set_aspect('equal', adjustable='box')
        axes[0, column].set_title(name, fontsize=16, fontweight='bold', pad=10)
        spatial = predictions.loc[predictions['method'].eq(method)]
        draw_spatial(axes[2, column], spatial, 'predicted_layer', low, high)
        score = scores.loc[scores['method'].eq(method), 'macro_f1'].item()
        axes[2, column].set_title(f'Predicted layers\nMacro-F1 = {score:.3f}',
                                 fontsize=13.5, fontweight='bold', pad=9)

    fig.text(0.5, 0.075, 'A–B: identical UMAP coordinates within each method; independent UMAPs across methods.',
             ha='center', fontsize=12, color='#555555')
    fig.text(0.5, 0.044, 'C: 30-neighbor reference-label transfer in the original standardized embeddings; query labels used for evaluation.',
             ha='center', fontsize=12, color='#555555')
    outputs = []
    for extension in ('pdf', 'svg', 'png'):
        path = OUTPUT / f'{STEM}.{extension}'
        fig.savefig(path, dpi=600 if extension == 'pdf' else 300)
        outputs.append(path.name)
        print(path)
    plt.close(fig)
    (OUTPUT / f'{STEM}_provenance.json').write_text(json.dumps({
        'source': Path(__file__).name, 'inputs': SOURCES, 'outputs': outputs,
        'slice': '151673', 'mode': 'diagonal_affine', 'reference_alpha': 0, 'query_alpha': 1,
        'verification': 'All 3,611 spot IDs, reference/query layer labels, and method spatial coordinates agree; all four saved macro-F1 scores match independent confusion counts.',
        'rendering': 'Saved UMAP coordinates with equal-aspect square viewports; original seed-42 draw order and palettes; identical spatial limits and inverted y across tissue maps.',
        'new_model_fits': False, 'new_predictions': False,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
