"""Render additional validated Study 04 trade-off and qualitative diagnostics."""
from __future__ import annotations
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/feast-reproduce-matplotlib')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('SOURCE_DATE_EPOCH', '0')
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import umap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from plot import FONT_SCALE, MANIFEST_PATH, METHOD_COLORS, METHOD_MARKERS, METHOD_ORDER, MODE_LABELS, MODE_STYLES, OUTPUT_DIR, REPOSITORY_ROOT, configure_matplotlib, load_inputs, selected_source
STUDY_OUTPUT = REPOSITORY_ROOT / '04_batch_effect_removal/outputs/final_rerun_20260718_v2'
SELECTED_ALPHA = 1.0
EMBEDDING_ALPHA = 1.0
SELECTED_MODE = 'diagonal_affine'
MODE_ORDER = ['shift_only', 'diagonal_affine']
N_SPOTS = 3611
PUBLIC_SEED = 42
UMAP_SETTINGS = {'n_neighbors': 30, 'min_dist': 0.3, 'metric': 'euclidean', 'random_state': PUBLIC_SEED, 'n_jobs': 1}
DOMAIN_ORDER = ['Layer_1', 'Layer_2', 'Layer_3', 'Layer_4', 'Layer_5', 'Layer_6', 'WM']
DOMAIN_COLORS = {'Layer_1': '#E69F00', 'Layer_2': '#56B4E9', 'Layer_3': '#009E73', 'Layer_4': '#F0E442', 'Layer_5': '#0072B2', 'Layer_6': '#D55E00', 'WM': '#CC79A7'}
BATCH_COLORS = {'ref': '#0072B2', 'query': '#D55E00'}

def dense(matrix) -> np.ndarray:
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)

def standardize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    scale = array.std(axis=0)
    scale[scale == 0] = 1.0
    return (array - array.mean(axis=0)) / scale

def uncorrected_pca(counts: np.ndarray) -> tuple[np.ndarray, int]:
    totals = counts.sum(axis=1, keepdims=True)
    nonzero = totals[:, 0] > 0
    scale = np.zeros_like(totals, dtype=np.float32)
    scale[nonzero, 0] = 10000.0 / totals[nonzero, 0]
    log_normalized = np.log1p(counts * scale)
    values = PCA(n_components=20, svd_solver='randomized', random_state=PUBLIC_SEED).fit_transform(log_normalized)
    return (standardize(values), int((~nonzero).sum()))

def repository_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT))

def read_panel_counts(path: Path, genes: list[str]) -> np.ndarray:
    adata = ad.read_h5ad(path, backed='r')
    try:
        if any((gene not in adata.var_names for gene in genes)):
            raise ValueError(f'Fixed panel is incomplete in {path}')
        return dense(adata[:, genes].X).astype(np.float32, copy=False)
    finally:
        adata.file.close()

def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    outputs: list[Path] = []
    formats = {'.pdf': {'dpi': 600, 'metadata': {'Creator': 'FEAST Study 04 plot_additional.py', 'CreationDate': None, 'ModDate': None}}, '.svg': {'dpi': 600, 'metadata': {'Creator': 'FEAST Study 04 plot_additional.py', 'Date': '1970-01-01T00:00:00Z'}}, '.png': {'dpi': 600, 'metadata': {'Software': 'FEAST Study 04 plot_additional.py'}}}
    for suffix, kwargs in formats.items():
        path = OUTPUT_DIR / f'{stem}{suffix}'
        fig.savefig(path, bbox_inches='tight', **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs

def style_numeric_axis(ax: plt.Axes) -> None:
    ax.grid(axis='both', color='#E2E2E2', linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#555555')
        spine.set_linewidth(0.8)

def primary_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics.loc[metrics['analysis_role'].eq('primary')].copy()
    expected = {(method, mode, alpha) for method in METHOD_ORDER for mode in MODE_ORDER for alpha in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]}
    observed = set(zip(primary['method'], primary['mode'], primary['alpha'], strict=True))
    if observed != expected or len(primary) != 36:
        raise ValueError('Study 04 primary method/mode/alpha matrix changed')
    return primary

def metric_row(primary: pd.DataFrame, method: str, mode: str, alpha: float) -> pd.Series:
    rows = primary.loc[primary['method'].eq(method) & primary['mode'].eq(mode) & np.isclose(primary['alpha'], alpha)]
    if len(rows) != 1:
        raise ValueError(f'Expected one primary row for {method}, {mode}, {alpha}')
    return rows.iloc[0]

def candidate_dir(method: str, mode: str, alpha: float) -> Path:
    return STUDY_OUTPUT / 'methods' / method / mode / f'alpha_{alpha:.2f}'

def load_primary_embedding(primary: pd.DataFrame, method: str, mode: str, alpha: float) -> tuple[np.ndarray, Path]:
    row = metric_row(primary, method, mode, alpha)
    candidate = candidate_dir(method, mode, alpha)
    embedding_path = candidate / 'embeddings.npz'
    result_path = candidate / 'result.h5ad'
    with np.load(embedding_path, allow_pickle=False) as payload:
        embedding = np.asarray(payload['embedding'], dtype=np.float64)
    if embedding.shape[0] != 2 * N_SPOTS or not np.isfinite(embedding).all():
        raise ValueError(f'Invalid primary embedding for {method}, {mode}, {alpha}')
    if method == 'GraphST':
        embedding = PCA(n_components=20, svd_solver='randomized', random_state=PUBLIC_SEED).fit_transform(embedding)
    values = standardize(embedding)
    if values.shape[1] != (20 if method == 'GraphST' else 10):
        raise ValueError(f'Primary representation dimension changed for {method}')
    return (values, result_path)

def validate_result_order(result_path: Path, expected_ids: np.ndarray) -> None:
    result = ad.read_h5ad(result_path, backed='r')
    try:
        batches = result.obs['batch'].astype(str).to_numpy(copy=True)
        source_ids = result.obs['source_spot_id'].astype(str).to_numpy(copy=True)
    finally:
        result.file.close()
    expected_batches = np.array(['ref'] * N_SPOTS + ['query'] * N_SPOTS)
    expected_source_ids = np.concatenate([expected_ids, expected_ids])
    if not np.array_equal(batches, expected_batches):
        raise ValueError(f'Batch order changed in {result_path}')
    if not np.array_equal(source_ids, expected_source_ids):
        raise ValueError(f'Paired spot order changed in {result_path}')

def draw_tradeoff(primary: pd.DataFrame) -> tuple[list[Path], pd.DataFrame]:
    configure_matplotlib()
    plot_data = primary.loc[primary['alpha'].le(SELECTED_ALPHA), ['method', 'mode', 'alpha', 'alpha_stratum', 'batch_mixing_entropy_k30', 'paired_retrieval_top1', 'domain_purity_k30']].sort_values(['method', 'mode', 'alpha'])
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    panels = [('paired_retrieval_top1', 'Paired retrieval top-1', 'Paired identity preservation'), ('domain_purity_k30', 'Domain purity (k=30)', 'Local domain preservation')]
    for ax, (y_metric, y_label, title) in zip(axes, panels, strict=True):
        for method in METHOD_ORDER:
            for mode in MODE_ORDER:
                rows = plot_data.loc[plot_data['method'].eq(method) & plot_data['mode'].eq(mode)].sort_values('alpha')
                ax.plot(rows['batch_mixing_entropy_k30'], rows[y_metric], color=METHOD_COLORS[method], linestyle=MODE_STYLES[mode], linewidth=1.55, zorder=2)
                ax.scatter(rows['batch_mixing_entropy_k30'], rows[y_metric], s=28, marker=METHOD_MARKERS[method], facecolor=METHOD_COLORS[method], edgecolor='white', linewidth=0.5, zorder=3)
        ax.set_title(title, fontsize=12 * FONT_SCALE, fontweight='bold', loc='center', pad=8)
        ax.set_xlabel('Batch mixing entropy (k=30) ↑', fontsize=10.5 * FONT_SCALE)
        ax.set_ylabel(f'{y_label} ↑', fontsize=10.5 * FONT_SCALE)
        ax.text(0.98, 0.96, 'preferred joint direction (up-right)', transform=ax.transAxes, ha='right', va='top', fontsize=8 * FONT_SCALE, color='#666666')
        style_numeric_axis(ax)
    method_handles = [Line2D([0], [0], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], markeredgecolor='white', markeredgewidth=0.5, linewidth=1.8, label=method) for method in METHOD_ORDER]
    mode_handles = [Line2D([0], [0], color='#444444', linestyle=MODE_STYLES[mode], label=MODE_LABELS[mode]) for mode in MODE_ORDER]
    fig.legend(handles=method_handles + mode_handles, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 0.94), frameon=False, fontsize=8.5 * FONT_SCALE)
    fig.suptitle('Batch removal–preservation trade-offs', fontsize=14 * FONT_SCALE, fontweight='bold', y=0.995)
    fig.text(0.5, 0.018, 'Points are connected in increasing α order. No composite score or Pareto ranking is computed.', ha='center', fontsize=8 * FONT_SCALE, color='#777777')
    fig.tight_layout(rect=(0.02, 0.07, 0.995, 0.84), w_pad=2.0)
    return (save_figure(fig, 'batch_preservation_tradeoff'), plot_data)

def exact_pair_recovery(values: np.ndarray) -> np.ndarray:
    reference = values[:N_SPOTS]
    query = values[N_SPOTS:]
    nearest_distance = NearestNeighbors(n_neighbors=1).fit(reference).kneighbors(query, return_distance=True)[0][:, 0]
    paired_distance = np.linalg.norm(query - reference, axis=1)
    tolerance = 1e-12 + 1e-09 * np.maximum(1.0, np.abs(nearest_distance))
    return paired_distance <= nearest_distance + tolerance

def draw_spatial_recovery(primary: pd.DataFrame, spot_ids: np.ndarray, spatial: np.ndarray, domains: np.ndarray) -> tuple[list[Path], pd.DataFrame, list[dict[str, object]]]:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.0), sharex=True, sharey=True)
    records: list[pd.DataFrame] = []
    sources: list[dict[str, object]] = []
    for row_index, mode in enumerate(MODE_ORDER):
        for column_index, method in enumerate(METHOD_ORDER):
            ax = axes[row_index, column_index]
            values, result_path = load_primary_embedding(primary, method, mode, SELECTED_ALPHA)
            validate_result_order(result_path, spot_ids)
            correct = exact_pair_recovery(values)
            expected = float(metric_row(primary, method, mode, SELECTED_ALPHA)['paired_retrieval_top1'])
            if not np.isclose(correct.mean(), expected, rtol=0, atol=1e-12):
                raise ValueError(f'Reconstructed exact-pair recovery changed for {method}, {mode}')
            frame = pd.DataFrame({'method': method, 'mode': mode, 'alpha': SELECTED_ALPHA, 'source_spot_id': spot_ids, 'x': spatial[:, 0], 'y': spatial[:, 1], 'ground_truth': domains, 'exact_pair_recovered': correct})
            records.append(frame)
            sources.append({'method': method, 'mode': mode, 'alpha': SELECTED_ALPHA, 'result': repository_path(result_path), 'embedding': repository_path(result_path.parent / 'embeddings.npz')})
            ax.scatter(spatial[~correct, 0], spatial[~correct, 1], s=2.2, color='#D4D4D4', linewidths=0, rasterized=True)
            ax.scatter(spatial[correct, 0], spatial[correct, 1], s=2.8, color='#0072B2', linewidths=0, rasterized=True)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.axis('off')
            recovery_label = f'{100 * correct.mean():.1f}% recovered'
            if row_index == 0:
                ax.set_title(f'{method}\n{recovery_label}', fontsize=12 * FONT_SCALE, fontweight='bold', pad=5, linespacing=1.35, loc='center')
            else:
                ax.set_title(recovery_label, fontsize=10 * FONT_SCALE, fontweight='bold', color='#333333', pad=5, loc='center')
    for row_index, mode in enumerate(MODE_ORDER):
        axes[row_index, 0].text(-0.08, 0.5, MODE_LABELS[mode], transform=axes[row_index, 0].transAxes, ha='right', va='center', rotation=90, fontsize=10 * FONT_SCALE, fontweight='bold')
    legend = [Line2D([0], [0], marker='o', color='none', markerfacecolor='#0072B2', label='Exact pair recovered'), Line2D([0], [0], marker='o', color='none', markerfacecolor='#D4D4D4', label='Exact pair not recovered')]
    fig.legend(handles=legend, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 0.88), frameon=False, fontsize=9 * FONT_SCALE)
    fig.suptitle('Spatial distribution of exact-pair recovery', fontsize=14 * FONT_SCALE, fontweight='bold', y=0.99)
    fig.text(0.5, 0.925, 'DLPFC slice 151673 · α=1.0 · identical spot coordinates and recovery definition across methods', ha='center', fontsize=10 * FONT_SCALE, color='#666666')
    fig.tight_layout(rect=(0.045, 0.025, 0.995, 0.83), w_pad=0.8, h_pad=0.6)
    data = pd.concat(records, ignore_index=True)
    if len(data) != 2 * 3 * N_SPOTS:
        raise ValueError('Spatial recovery plotting table has unexpected size')
    return (save_figure(fig, 'paired_recovery_spatial'), data, sources)

def audit_embedding_conditions(simulation_manifest: Path, panel_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(simulation_manifest)
    panel = pd.read_csv(panel_path).sort_values('panel_order')
    genes = panel['gene'].astype(str).tolist()
    labels = np.concatenate([np.zeros(N_SPOTS, dtype=np.int8), np.ones(N_SPOTS, dtype=np.int8)])
    records: list[dict[str, object]] = []
    for mode in MODE_ORDER:
        reference_row = manifest.loc[manifest['mode'].eq(mode) & np.isclose(manifest['alpha'], 0.0)]
        if len(reference_row) != 1:
            raise ValueError(f'Expected one reference simulation for {mode}')
        reference_path = Path(reference_row.iloc[0]['file'])
        reference = read_panel_counts(reference_path, genes)
        candidates = manifest.loc[manifest['mode'].eq(mode) & manifest['alpha'].gt(0) & manifest['alpha'].le(EMBEDDING_ALPHA)].sort_values('alpha')
        for row in candidates.itertuples(index=False):
            query_path = Path(row.file)
            query = read_panel_counts(query_path, genes)
            values, zero_library_spots = uncorrected_pca(np.vstack([reference, query]))
            batch_asw = silhouette_score(values, labels, metric='euclidean', sample_size=3000, random_state=PUBLIC_SEED)
            centroid_distance = np.linalg.norm(values[:N_SPOTS].mean(axis=0) - values[N_SPOTS:].mean(axis=0))
            records.append({'mode': mode, 'alpha': float(row.alpha), 'alpha_stratum': row.alpha_stratum, 'uncorrected_pca20_batch_asw_sampled': float(batch_asw), 'uncorrected_pca20_batch_centroid_distance': float(centroid_distance), 'zero_library_spots_in_fixed_panel': zero_library_spots, 'query_file': repository_path(query_path)})
    audit = pd.DataFrame(records).sort_values(['mode', 'alpha']).reset_index(drop=True)
    selected = audit.loc[audit['uncorrected_pca20_batch_asw_sampled'].idxmax()]
    if selected['mode'] != SELECTED_MODE or not np.isclose(selected['alpha'], EMBEDDING_ALPHA):
        raise ValueError('The strongest displayed uncorrected batch-separation condition changed')
    audit['selected_for_embedding_context'] = False
    audit.loc[selected.name, 'selected_for_embedding_context'] = True
    return audit

def read_reference_context(simulation_manifest: Path, panel_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]]]:
    manifest = pd.read_csv(simulation_manifest)
    panel = pd.read_csv(panel_path).sort_values('panel_order')
    genes = panel['gene'].astype(str).tolist()
    matrices: list[np.ndarray] = []
    selected_sources: list[dict[str, str]] = []
    spot_ids: np.ndarray | None = None
    spatial: np.ndarray | None = None
    domains: np.ndarray | None = None
    for alpha in [0.0, EMBEDDING_ALPHA]:
        rows = manifest.loc[manifest['mode'].eq(SELECTED_MODE) & np.isclose(manifest['alpha'], alpha)]
        if len(rows) != 1:
            raise ValueError(f'Expected one simulation row for {SELECTED_MODE}, {alpha}')
        row = rows.iloc[0]
        path = Path(row['file'])
        adata = ad.read_h5ad(path, backed='r')
        try:
            if any((gene not in adata.var_names for gene in genes)):
                raise ValueError(f'Fixed panel is incomplete in {path}')
            matrix = dense(adata[:, genes].X).astype(np.float32, copy=False)
            ids = adata.obs_names.astype(str).to_numpy(copy=True)
            if alpha == 0.0:
                spot_ids = ids
                spatial = np.asarray(adata.obsm['spatial'], dtype=np.float64)
                domains = adata.obs['ground_truth'].astype(str).to_numpy(copy=True)
            elif not np.array_equal(ids, spot_ids):
                raise ValueError('Reference/query spot order changed')
        finally:
            adata.file.close()
        matrices.append(matrix)
        selected_sources.append({'path': repository_path(path)})
    if spot_ids is None or spatial is None or domains is None:
        raise RuntimeError('Reference context was not loaded')
    uncorrected, _zero_library_spots = uncorrected_pca(np.vstack(matrices))
    return (uncorrected, spot_ids, spatial, domains, selected_sources)

def umap_projection(values: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(**UMAP_SETTINGS)
    coordinates = reducer.fit_transform(values)
    if coordinates.shape != (2 * N_SPOTS, 2) or not np.isfinite(coordinates).all():
        raise ValueError('UMAP projection is invalid')
    return coordinates

def draw_embedding_context(primary: pd.DataFrame, input_representation: np.ndarray, spot_ids: np.ndarray, domains: np.ndarray) -> tuple[list[Path], pd.DataFrame, list[dict[str, object]]]:
    configure_matplotlib()
    representations: dict[str, np.ndarray] = {'Uncorrected input': input_representation}
    sources: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        values, result_path = load_primary_embedding(primary, method, SELECTED_MODE, EMBEDDING_ALPHA)
        validate_result_order(result_path, spot_ids)
        representations[method] = values
        sources.append({'method': method, 'mode': SELECTED_MODE, 'alpha': EMBEDDING_ALPHA, 'result': repository_path(result_path), 'embedding': repository_path(result_path.parent / 'embeddings.npz')})
    batches = np.array(['ref'] * N_SPOTS + ['query'] * N_SPOTS)
    source_ids = np.concatenate([spot_ids, spot_ids])
    domain_labels = np.concatenate([domains, domains])
    rng = np.random.default_rng(PUBLIC_SEED)
    plotting_order = rng.permutation(2 * N_SPOTS)
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 7.2))
    records: list[pd.DataFrame] = []
    for column_index, (name, values) in enumerate(representations.items()):
        projection = umap_projection(values)
        records.append(pd.DataFrame({'representation': name, 'mode': SELECTED_MODE, 'alpha': EMBEDDING_ALPHA, 'source_spot_id': source_ids, 'batch': batches, 'ground_truth': domain_labels, 'umap_1': projection[:, 0], 'umap_2': projection[:, 1]}))
        batch_ax = axes[0, column_index]
        domain_ax = axes[1, column_index]
        batch_ax.scatter(projection[plotting_order, 0], projection[plotting_order, 1], s=2.0, c=[BATCH_COLORS[value] for value in batches[plotting_order]], linewidths=0, alpha=0.88, rasterized=True)
        domain_ax.scatter(projection[plotting_order, 0], projection[plotting_order, 1], s=2.0, c=[DOMAIN_COLORS[value] for value in domain_labels[plotting_order]], linewidths=0, alpha=0.9, rasterized=True)
        batch_ax.set_title(name, fontsize=12 * FONT_SCALE, fontweight='bold', pad=6, loc='center')
        for ax in [batch_ax, domain_ax]:
            ax.axis('off')
    axes[0, 0].text(-0.08, 0.5, 'Colored by batch', transform=axes[0, 0].transAxes, ha='right', va='center', rotation=90, fontsize=10 * FONT_SCALE, fontweight='bold')
    axes[1, 0].text(-0.08, 0.5, 'Colored by cortical layer', transform=axes[1, 0].transAxes, ha='right', va='center', rotation=90, fontsize=10 * FONT_SCALE, fontweight='bold')
    batch_handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=color, label=label.title()) for label, color in BATCH_COLORS.items()]
    domain_handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=DOMAIN_COLORS[label], label=label.replace('_', ' ')) for label in DOMAIN_ORDER]
    first_legend = fig.legend(handles=batch_handles, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 0.86), frameon=False, fontsize=8.5 * FONT_SCALE, title='Batch', title_fontsize=8.5 * FONT_SCALE, columnspacing=1.5, handletextpad=0.45)
    fig.add_artist(first_legend)
    fig.legend(handles=domain_handles, ncol=7, loc='lower center', bbox_to_anchor=(0.5, 0.012), frameon=False, fontsize=8.2 * FONT_SCALE, columnspacing=1.25, handletextpad=0.4)
    fig.suptitle('Representative embedding context', fontsize=14 * FONT_SCALE, fontweight='bold', y=0.985)
    fig.text(0.5, 0.92, 'DLPFC slice 151673 · diagonal affine · α=1.0 endpoint · independent deterministic UMAP per column', ha='center', fontsize=10 * FONT_SCALE, color='#666666')
    fig.text(0.5, 0.062, 'Compare batch mixing and label co-localization within columns; absolute UMAP geometry is not comparable across columns.', ha='center', fontsize=8 * FONT_SCALE, color='#777777')
    fig.tight_layout(rect=(0.045, 0.105, 0.995, 0.835), w_pad=0.8, h_pad=0.8)
    data = pd.concat(records, ignore_index=True)
    if len(data) != 4 * 2 * N_SPOTS:
        raise ValueError('Embedding-context plotting table has unexpected size')
    return (save_figure(fig, 'representative_embedding_context'), data, sources)

def write_provenance(sources: dict[str, Path], panel_path: Path, simulation_sources: list[dict[str, str]], method_sources: list[dict[str, object]], embedding_selection: dict[str, object], outputs: list[Path]) -> Path:
    payload = {'schema_version': 1, 'figure_set': 'study04_additional_visual_diagnostics', 'publication_status': 'supplementary_unpromoted_author_review_required', 'scientific_disposition': {'active_manuscript_claim': None, 'supplementary_candidate_only': True, 'composite_score_present': False, 'winner_ranking_authorized': False, 'figure_promotion_authorized': False}, 'sources': {'publication_manifest': {'path': repository_path(MANIFEST_PATH)}, **{role: {'path': repository_path(path)} for role, path in sorted(sources.items())}, 'panel': {'path': repository_path(panel_path)}, 'selected_simulations': simulation_sources, 'selected_method_artifacts': method_sources, 'plot_script': {'path': repository_path(Path(__file__))}}, 'design': {'displayed_alpha_maximum': SELECTED_ALPHA, 'representative_conditions': {'spatial_exact_pair_recovery': {'slice': '151673', 'modes': MODE_ORDER, 'alpha': SELECTED_ALPHA, 'selection_rationale': 'registered primary endpoint'}, 'embedding_context': {'slice': '151673', 'mode': SELECTED_MODE, 'alpha': EMBEDDING_ALPHA, 'selection_rationale': 'maximum deterministic sampled batch ASW in the uncorrected standardized PCA-20 representation among displayed mode-by-alpha conditions', 'selection_evidence': embedding_selection}}, 'tradeoff_metrics': ['batch_mixing_entropy_k30', 'paired_retrieval_top1', 'domain_purity_k30'], 'paired_recovery_definition': 'true paired reference is tied for nearest Euclidean neighbor after scorer-standardized primary representation', 'umap': UMAP_SETTINGS, 'umap_scope': 'independent fit per representation; within-column interpretation only', 'uncorrected_representation': 'PCA-20 of log1p library-size-normalized fixed-panel counts', 'ranking_or_composite_used': False}, 'outputs': {path.name: None for path in sorted(outputs, key=lambda item: item.name)}}
    path = OUTPUT_DIR / 'additional_figure_provenance.json'
    path.write_text(json.dumps(payload, indent=2) + '\n')
    return path

def main() -> int:
    metrics, _validation, candidate, sources = load_inputs()
    primary = primary_rows(metrics)
    panel_path = selected_source(candidate, 'panel')
    simulation_manifest = selected_source(candidate, 'simulation_manifest')
    embedding_audit = audit_embedding_conditions(simulation_manifest, panel_path)
    input_representation, spot_ids, spatial, domains, simulation_sources = read_reference_context(simulation_manifest, panel_path)
    if set(domains) != set(DOMAIN_ORDER):
        raise ValueError('DLPFC domain set changed')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []
    table_paths: list[Path] = []
    method_sources: list[dict[str, object]] = []
    embedding_audit_path = OUTPUT_DIR / 'embedding_condition_selection_data.csv'
    embedding_audit.to_csv(embedding_audit_path, index=False, float_format='%.12g')
    table_paths.append(embedding_audit_path)
    outputs, tradeoff_data = draw_tradeoff(primary)
    figure_paths.extend(outputs)
    tradeoff_path = OUTPUT_DIR / 'batch_preservation_tradeoff_data.csv'
    tradeoff_data.to_csv(tradeoff_path, index=False, float_format='%.12g')
    table_paths.append(tradeoff_path)
    outputs, spatial_data, selected_sources = draw_spatial_recovery(primary, spot_ids, spatial, domains)
    figure_paths.extend(outputs)
    method_sources.extend(selected_sources)
    spatial_path = OUTPUT_DIR / 'paired_recovery_spatial_data.csv'
    spatial_data.to_csv(spatial_path, index=False, float_format='%.12g')
    table_paths.append(spatial_path)
    outputs, embedding_data, selected_sources = draw_embedding_context(primary, input_representation, spot_ids, domains)
    figure_paths.extend(outputs)
    method_sources.extend(selected_sources)
    embedding_path = OUTPUT_DIR / 'representative_embedding_context_data.csv'
    embedding_data.to_csv(embedding_path, index=False, float_format='%.12g')
    table_paths.append(embedding_path)
    unique_method_sources = {(item['method'], item['mode'], item['alpha']): item for item in method_sources}
    provenance_path = write_provenance(sources, panel_path, simulation_sources, list(unique_method_sources.values()), json.loads(embedding_audit.loc[embedding_audit['selected_for_embedding_context']].iloc[0].to_json()), table_paths + figure_paths)
    for path in table_paths + figure_paths + [provenance_path]:
        print(f'Saved: {path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
