"""Render compact metrics from validated tables and saved-result layer transfer."""
from __future__ import annotations

import json
from pathlib import Path

import plot as same_slice
import plot_real_slice_robustness as two_slice
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


# Shared metric families, in the same panel positions in both figures. The
# two-slice batch metrics retain their existing layer-conditioned definitions.
SHARED_METRICS = [
    ('batch_asw', 'conditional_batch_asw_macro', 'Batch ASW ↓'),
    ('batch_mixing_entropy_k30', 'conditional_batch_entropy_macro', 'Batch entropy ↑'),
    ('mmd2_rbf_biased', 'conditional_mmd2_macro', 'Distribution MMD² ↓'),
    ('domain_asw', 'layer_asw', 'Layer separation ASW ↑'),
    ('domain_purity_k30', 'layer_purity_k', 'Layer purity (k=30) ↑'),
    ('reference_to_query_layer_macro_f1', 'reference_to_query_layer_macro_f1', 'Layer transfer macro-F1 ↑'),
]
PAIRED_METRICS = [
    ('paired_retrieval_top1', 'Paired retrieval top-1 ↑'),
    ('paired_retrieval_median_rank', 'Paired retrieval median rank ↓'),
]
TRANSFER_METRIC = 'reference_to_query_layer_macro_f1'


def build_same_slice_transfer(primary):
    """Extend the approved endpoint layer-transfer calculation along the displayed ladder."""
    import plot_same_slice_layer_transfer as transfer

    def labels_from_input(path):
        data = transfer.ad.read_h5ad(path, backed='r')
        try:
            return (data.obs_names.astype(str).to_numpy(copy=True),
                    data.obs['ground_truth'].astype(str).to_numpy(copy=True))
        finally:
            data.file.close()

    _, _, candidate, _ = same_slice.load_inputs()
    manifest_path = same_slice.selected_source(candidate, 'simulation_manifest')
    manifest = pd.read_csv(manifest_path)
    records, artifacts = [], []
    for mode in transfer.core.MODE_ORDER:
        ref_path = Path(manifest.loc[manifest['mode'].eq(mode) & manifest['alpha'].eq(0), 'file'].item())
        ref_ids, ref_labels = labels_from_input(ref_path)
        artifacts.append(ref_path)
        for alpha in sorted(primary['alpha'].unique()):
            query_path = Path(manifest.loc[manifest['mode'].eq(mode) & manifest['alpha'].eq(alpha), 'file'].item())
            query_ids, query_labels = labels_from_input(query_path)
            if not transfer.np.array_equal(ref_ids, query_ids):
                raise ValueError(f'Layer-transfer input order differs: {mode}/{alpha}')
            if not transfer.np.array_equal(ref_labels, query_labels):
                raise ValueError(f'Layer-transfer input labels differ: {mode}/{alpha}')
            artifacts.append(query_path)
            for method in same_slice.METHOD_ORDER:
                values, result_path = transfer.core.load_primary_embedding(primary, method, mode, alpha)
                transfer.core.validate_result_order(result_path, ref_ids)
                _, score, _ = transfer.predict_and_verify(values, ref_labels, query_labels)
                records.append({'method': method, 'mode': mode, 'alpha': alpha, TRANSFER_METRIC: score})
                artifacts.extend([result_path, result_path.parent / 'embeddings.npz'])
            print(f'Layer-transfer votes and macro-F1 verified: {mode}, α={alpha:g}', flush=True)
    scores = pd.DataFrame(records)
    endpoint_path = same_slice.OUTPUT_DIR / 'same_slice_layer_transfer_spatial_scores.csv'
    endpoint = pd.read_csv(endpoint_path).loc[lambda d: d['method'].isin(same_slice.METHOD_ORDER)]
    check = scores.loc[scores['alpha'].eq(1)].merge(endpoint, on=['method', 'mode', 'alpha'], validate='one_to_one')
    if len(check) != 6 or not transfer.np.allclose(check[TRANSFER_METRIC], check['macro_f1'], atol=1e-12, rtol=0):
        raise ValueError('Layer-transfer endpoint differs from the saved spatial figure')
    score_path = same_slice.OUTPUT_DIR / 'same_slice_layer_transfer_metric_ladder.csv'
    scores.to_csv(score_path, index=False)
    provenance_path = same_slice.OUTPUT_DIR / 'same_slice_layer_transfer_metric_ladder_provenance.json'
    provenance_path.write_text(json.dumps({
        'source': same_slice.repository_path(Path(__file__)),
        'calculation': same_slice.repository_path(Path(transfer.__file__)),
        'simulation_manifest': same_slice.repository_path(manifest_path),
        'inputs': [same_slice.repository_path(path) for path in artifacts],
        'output': same_slice.repository_path(score_path),
        'metric': TRANSFER_METRIC, 'n_neighbors': 30, 'weights': 'uniform', 'distance': 'Euclidean',
        'prediction': 'Standardized primary embeddings and reference labels only; query labels used for evaluation.',
        'verification': 'Spot/batch order checked; explicit majority votes and independent confusion-count macro-F1 checked for all 24 rows; six endpoint scores match the existing spatial figure.',
        'endpoint_comparison': same_slice.repository_path(endpoint_path),
        'primary_representations': {'GraphST': 'PCA-20 seed42', 'STAMP': 'native 10D topics', 'scVI': 'native 10D latent'},
        'scope': 'New diagnostic scores from saved results; original atomic tables preserved; no method retraining.',
    }, indent=2) + '\n')
    return scores, {'layer_transfer_scores': score_path, 'layer_transfer_provenance': provenance_path}


def configure_style() -> None:
    same_slice.configure_matplotlib()
    plt.rcParams.update({
        'font.size': 13, 'axes.titlesize': 15, 'axes.titleweight': 'bold',
        'axes.labelsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'legend.fontsize': 14,
    })


def style_panel(ax, index, title) -> None:
    title = title.replace('Paired retrieval median rank', 'Paired retrieval\nmedian rank')
    ax.set_title(title, loc='center', pad=14)
    ax.text(-0.14, 1.055, chr(ord('A') + index), transform=ax.transAxes,
            fontsize=17, fontweight='bold')
    ax.grid(axis='y', color='#DDDDDD', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    for spine in ax.spines.values():
        spine.set_color('#666666')
        spine.set_linewidth(0.8)


def draw_same_slice(primary):
    specs = [(a, title) for a, _, title in SHARED_METRICS] + PAIRED_METRICS
    fig, axes = plt.subplots(3, 3, figsize=(14, 11.7))
    for index, (metric, title) in enumerate(specs):
        ax = axes.flat[index]
        for method in same_slice.METHOD_ORDER:
            for mode, linestyle in same_slice.MODE_STYLES.items():
                rows = primary.loc[
                    primary['method'].eq(method) & primary['mode'].eq(mode)
                ].sort_values('alpha')
                ax.plot(rows['alpha'], rows[metric],
                        color=same_slice.METHOD_COLORS[method],
                        marker=same_slice.METHOD_MARKERS[method],
                        linestyle=linestyle, linewidth=1.7, markersize=4.8,
                        markeredgecolor='white', markeredgewidth=0.5)
        style_panel(ax, index, title)
        ax.set_xticks([0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel('Perturbation strength α')
        ax.margins(x=0.06, y=0.14)
        if metric == 'paired_retrieval_median_rank':
            ax.set_yscale('log')
            ax.set_ylabel('Rank (log scale)')
    legend_ax = axes.flat[-1]
    legend_ax.axis('off')
    handles = [Line2D([0], [0], color=same_slice.METHOD_COLORS[m],
                      marker=same_slice.METHOD_MARKERS[m], lw=1.7, label=m)
               for m in same_slice.METHOD_ORDER]
    handles += [Line2D([0], [0], color='#444444', linestyle=style,
                       label=same_slice.MODE_LABELS[mode])
                for mode, style in same_slice.MODE_STYLES.items()]
    legend_ax.legend(handles=handles, loc='upper left', frameon=False,
                     borderaxespad=0, labelspacing=0.5, markerscale=1.5,
                     handlelength=2.5)
    legend_ax.text(0, 0.02, 'A–F  Shared metric families\nG–H  Same-slice paired metrics',
                   transform=legend_ax.transAxes, fontsize=12,
                   color='#555555', linespacing=1.6)
    fig.suptitle('Same-slice batch removal and biological preservation',
                 fontsize=20, fontweight='bold', y=0.985)
    fig.text(0.5, 0.946, 'Batch metrics use all spots; paired metrics use known spot correspondence.',
             ha='center', color='#555555', fontsize=13)
    fig.tight_layout(rect=(0.015, 0.015, 0.995, 0.92), h_pad=2.0, w_pad=2.0)
    return fig, specs


def draw_two_slice(long_data, seed_summary):
    specs = [(b, title) for _, b, title in SHARED_METRICS]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.5))
    for index, (metric, title) in enumerate(specs):
        ax = axes.flat[index]
        two_slice.draw_method_metric(ax, metric, long_data, seed_summary)
        values = long_data.loc[long_data['metric'].eq(metric), 'value']
        two_slice.style_axis(ax, metric, values)
        # Include the complete observed range, including negative ASW values
        # should they occur, without altering the source scores.
        span = max(float(values.max() - values.min()), 0.02)
        ax.set_ylim(float(values.min()) - 0.12 * span,
                    float(values.max()) + 0.12 * span)
        style_panel(ax, index, title)
        ax.set_xticklabels(['Raw', '0', '0.5', '1.0'])
        ax.set_xlabel('Perturbation strength α')
    fig.legend(handles=two_slice.legend_handles(), loc='upper center',
               bbox_to_anchor=(0.5, 0.9), ncol=4, frameon=False,
               markerscale=1.5, handlelength=2.5)
    fig.text(0.5, 0.015, 'Raw: isolated natural-data anchor · α=0: resampling control',
             ha='center', fontsize=12, color='#555555')
    fig.suptitle('Two-slice batch removal and biological preservation',
                 fontsize=20, fontweight='bold', y=0.985)
    fig.text(0.5, 0.932, 'Batch metrics are layer-conditioned; A–F match the same-slice metric families.',
             ha='center', color='#555555', fontsize=13)
    fig.tight_layout(rect=(0.015, 0.055, 0.995, 0.83), h_pad=2.0, w_pad=2.0)
    return fig, specs


def save_outputs(fig, specs, data, output_dir, stem, sources, notes):
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f'{stem}.pdf'
    fig.savefig(pdf_path, bbox_inches='tight', metadata={
        'Creator': 'FEAST plot_combined_metrics.py',
        'CreationDate': None, 'ModDate': None,
    })
    plt.close(fig)
    data_path = output_dir / f'{stem}_plot_data.csv'
    data.to_csv(data_path, index=False, na_rep='NA')
    provenance = {
        'source': same_slice.repository_path(Path(__file__)),
        'inputs': {key: same_slice.repository_path(path) for key, path in sources.items()},
        'outputs': [same_slice.repository_path(pdf_path), same_slice.repository_path(data_path)],
        'displayed_metrics': [metric for metric, _ in specs],
        'shared_metric_families': [
            {'same_slice': a, 'two_slice': b, 'label': title}
            for a, b, title in SHARED_METRICS
        ],
        'displayed_alpha_maximum': 1.0,
        'notes': notes,
        'composite_score_used': False,
        'winner_ranking_authorized': False,
        'figure_promotion_authorized': False,
        'publication_status': 'supplementary_unpromoted_author_review_required',
    }
    provenance_path = output_dir / f'{stem}_provenance.json'
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    for path in (pdf_path, data_path, provenance_path):
        print(path)


def main() -> None:
    # Reuse the existing validation checks before rendering either design.
    same_metrics, _, _, same_sources = same_slice.load_inputs()
    real_metrics, _, _ = two_slice.load_validated_inputs()
    primary = same_metrics.loc[
        same_metrics['analysis_role'].eq('primary') & same_metrics['alpha'].le(1.0)
    ].copy()
    transfer_scores, transfer_sources = build_same_slice_transfer(primary)
    primary = primary.merge(transfer_scores, on=['method', 'mode', 'alpha'], validate='one_to_one')
    same_sources.update(transfer_sources)
    real_long, real_summary = two_slice.build_plot_data(real_metrics)
    configure_style()
    fig, specs = draw_same_slice(primary)
    same_long = same_slice.build_long_table(primary)
    transfer_long = same_long.loc[same_long['metric'].eq('batch_asw')].drop(columns='value').merge(
        transfer_scores, on=['method', 'mode', 'alpha'], validate='one_to_one').rename(columns={TRANSFER_METRIC: 'value'})
    transfer_long['metric'] = TRANSFER_METRIC
    transfer_long['metric_label'] = 'Layer transfer macro-F1 ↑'
    same_long = pd.concat([same_long, transfer_long], ignore_index=True)
    same_long = same_long.loc[same_long['metric'].isin([metric for metric, _ in specs])]
    save_outputs(fig, specs, same_long, same_slice.OUTPUT_DIR,
                 'same_slice_combined_metrics', same_sources, [
                     'Six shared metric families plus two same-slice paired metrics, all available for GraphST, STAMP, and scVI.',
                     'Layer-transfer macro-F1 is newly calculated from saved primary representations using the approved 30-reference-neighbor rule; alpha=1 scores match the spatial figure.',
                     'scVI-only expression correlation is excluded from this comparison and retained in the original atomic tables.',
                     'Same-slice batch metrics are unconditional; two-slice metrics are layer-conditioned.',
                     'Existing GraphST PCA sensitivity figures and external-environment limitations still apply.',
                 ])
    fig, specs = draw_two_slice(real_long, real_summary)
    selected = real_long.loc[real_long['metric'].isin([metric for metric, _ in specs])]
    save_outputs(fig, specs, selected, two_slice.OUTPUT_DIR,
                 'two_slice_combined_metrics', {
                     'atomic_metrics': two_slice.METRICS_PATH,
                     'validation': two_slice.VALIDATION_PATH,
                     'qualification': two_slice.QUALIFICATION_PATH,
                     'metrics_provenance': two_slice.METRICS_PROVENANCE_PATH,
                     'metrics_summary': two_slice.METRICS_SUMMARY_PATH,
                 }, [
                     'Six shared metric families; no cross-section spot pairing.',
                     'Batch metrics are layer-conditioned, unlike the same-slice versions.',
                     'Raw anchor is isolated; lines start at the alpha=0 resampling control.',
                     'Method lines show seed means; individual seeds and min-max bands are retained.',
                     'Uncorrected input is one deterministic baseline.',
                     'Existing GraphST PCA sensitivity figures remain applicable.',
                 ])


if __name__ == '__main__':
    main()
