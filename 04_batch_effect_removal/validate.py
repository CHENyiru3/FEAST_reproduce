"""Validate one complete fresh Study 04 run and write its decision record."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
import compare_historical as historical
import run as workflow
import score as scoring
STUDY_ROOT = Path(__file__).resolve().parent
REPRODUCTION_ROOT = STUDY_ROOT.parent
VALIDATOR_K_NEIGHBORS = 30
VALIDATOR_MMD_SAMPLE_PER_BATCH = 1000
VALIDATOR_ALL_SPOTS = historical.ALL_SPOTS
VALIDATOR_PAIRED_SUPPORT = historical.PAIRED_SUPPORT
VALIDATOR_EXPECTED_QUERY_ZERO_IDS = {('diagonal_affine', '1.25'): ['AGAAGAGCGCCGTTCC-1'], ('diagonal_affine', '1.50'): ['AGAAGAGCGCCGTTCC-1', 'GAGGAGATCCTCATGC-1', 'GGTCCTTCATACGACT-1'], ('shift_only', '1.50'): ['AGAAGAGCGCCGTTCC-1', 'GAGGAGATCCTCATGC-1']}
STAMP_RETRY_POLICY_ID = 'study04-stamp-numerical-retry-v1'
STAMP_RETRY_POLICY_PATH = '04_batch_effect_removal/stamp_numerical_retry_v1.yaml'
STAMP_RETRY_ELIGIBLE = {('shift_only', '1.50'), ('diagonal_affine', '1.50')}
STAMP_RETRY_UNCHANGED_CONTRACT = {'public_seed': 42, 'n_topics': 10, 'n_layers': 1, 'hidden_size': 128, 'mode': 'sgc', 'learning_rate': 0.005, 'max_epochs': 800, 'min_epochs': 100, 'device': 'cuda:0', 'zero_library_strategy': 'raw_zero_likelihood_within_batch_sgc_encoder_v1'}
METRIC_REPORT_FILES = ('atomic_metrics.csv', 'primary_summary.csv', 'common_support_sensitivity.csv', 'graphst_pca20_vs_pca10_sensitivity.csv', 'support_audit.csv', 'old_vs_new_atomic_metrics.csv', 'old_vs_new_simulation_cells.csv', 'historical_composite_disposition.csv', 'summary.json')

def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def alpha_text(value: object) -> str:
    return f'{float(value):.2f}'

def retry_bound_file(relative_path: object, label: str) -> Path:
    relative = Path(str(relative_path))
    check(not relative.is_absolute(), f'STAMP retry {label} path must be repository-relative')
    path = (REPRODUCTION_ROOT / relative).resolve()
    check(path.is_relative_to(REPRODUCTION_ROOT), f'STAMP retry {label} path escapes the reproduction repository')
    check(path.is_file(), f'STAMP retry {label} is missing: {path}')
    return path

def validate_stamp_retry(metadata: dict, candidate: Path, mode: str, alpha: float) -> None:
    """Independently enforce standard patience or the pinned narrow retry."""
    patience = int(metadata.get('patience', -1))
    retry = metadata.get('numerical_retry')
    if patience == 20:
        check(retry is None, 'standard-patience STAMP candidate carries a retry declaration')
        return
    check(patience == 10, f'STAMP patience changed outside the declared contract: {patience}')
    check(isinstance(retry, dict), 'patience-10 STAMP candidate lacks numerical_retry provenance')
    alpha_key = alpha_text(alpha)
    check((mode, alpha_key) in STAMP_RETRY_ELIGIBLE, f'STAMP retry is not eligible for {mode}/alpha_{alpha_key}')
    eligible_condition = retry.get('eligible_condition', {})
    check(eligible_condition.get('mode') == mode and alpha_text(eligible_condition.get('alpha', float('nan'))) == alpha_key, 'STAMP retry eligible-condition record changed')
    check(retry.get('policy_configuration_id') == STAMP_RETRY_POLICY_ID, 'STAMP retry policy configuration ID changed')
    check(retry.get('policy_path') == STAMP_RETRY_POLICY_PATH, 'STAMP retry policy path changed')
    policy_path = retry_bound_file(retry.get('policy_path'), 'policy')
    check(policy_path == (STUDY_ROOT / 'stamp_numerical_retry_v1.yaml').resolve(), 'STAMP retry policy resolves to the wrong file')
    check(retry.get('changed_parameter') == {'name': 'patience', 'standard_value': 20, 'retry_value': 10}, 'STAMP retry changed-parameter record is not patience 20 -> 10')
    check(retry.get('unchanged_contract') == STAMP_RETRY_UNCHANGED_CONTRACT, 'STAMP retry unchanged-contract record changed')
    reason = str(retry.get('activation_reason', ''))
    check(all((token in reason for token in ('non-finite', 'Pyro', 'patience'))), 'STAMP retry activation reason is not explicit')
    failure = retry_bound_file(retry.get('original_failure_path'), 'original failure')
    disposition = retry_bound_file(retry.get('original_disposition_path'), 'original disposition')
    expected_failure_prefix = f'STAMP__{mode}__alpha_{alpha_key}__'
    run_dir = candidate.parents[3].resolve()
    check(candidate.name == f'alpha_{alpha_key}' and candidate.parent.name == mode and (candidate.parent.parent.name == 'STAMP') and (failure.name == 'FAILED.txt') and (disposition.name == 'DISPOSITION.txt') and (failure.parent == disposition.parent) and failure.parent.name.startswith(expected_failure_prefix) and (failure.parent.parent.resolve() == run_dir / 'failures'), 'STAMP retry is not bound to its matching preserved failure')
    failure_text = failure.read_text(encoding='utf-8')
    check(all((token in failure_text for token in ('caux', 'tensor([nan]', 'cuda:0'))), 'STAMP retry failure does not prove the declared Pyro non-finiteness')
    check(disposition.read_text(encoding='utf-8').strip() == 'exit=1; missing metadata.json', 'STAMP retry failure disposition changed')
    actual_contract = {'public_seed': int(metadata.get('seed', -1)), 'n_topics': int(metadata.get('n_topics', -1)), 'n_layers': int(metadata.get('n_layers', -1)), 'hidden_size': int(metadata.get('hidden_size', -1)), 'mode': metadata.get('mode'), 'learning_rate': float(metadata.get('learning_rate', float('nan'))), 'max_epochs': int(metadata.get('max_epochs', -1)), 'min_epochs': int(metadata.get('min_epochs', -1)), 'device': metadata.get('device'), 'zero_library_strategy': metadata.get('zero_library_safeguard', {}).get('strategy')}
    check(actual_contract == STAMP_RETRY_UNCHANGED_CONTRACT, 'STAMP retry candidate differs from its unchanged contract')
    provenance = metadata.get('provenance', {})
    context = provenance.get('candidate_context', {})
    check(context.get('method') == 'STAMP' and context.get('deformation_mode') == mode and (alpha_text(context.get('alpha', float('nan'))) == alpha_key), 'STAMP retry candidate context changed')
    diagnostics = provenance.get('solver_diagnostics', {})
    check(int(provenance.get('public_seed', -1)) == 42, 'STAMP retry provenance seed changed')
    check(float(diagnostics.get('learning_rate', float('nan'))) == 0.005 and diagnostics.get('cuda_execution_verified') is True and (diagnostics.get('actual_device') == 'cuda:0'), 'STAMP retry LR/CUDA provenance changed')
    graph = metadata.get('spatial_graph', {})
    graph_contract = {'schema_version': 3, 'construction': 'independent_directed_knn_by_batch', 'distance': 'squared_euclidean', 'self_exclusion': 'exact_row_identity', 'distance_tie_break': 'original_within_batch_spot_order', 'n_neighbors': 6, 'n_spots': 7222, 'n_batches': 2, 'directed_edge_count': 43332, 'cross_batch_edge_count': 0, 'self_edge_count': 0}
    check(all((graph.get(key) == value for key, value in graph_contract.items())), 'STAMP retry graph contract changed')
    expected_inputs = {(run_dir / 'simulations' / mode / 'alpha_0.00.h5ad').resolve(), (run_dir / 'simulations' / mode / f'alpha_{alpha_key}.h5ad').resolve(), (run_dir / 'panel' / 'panel.csv').resolve(), (run_dir / 'panel' / 'provenance.json').resolve()}
    observed_inputs = {Path(str(item.get('path', ''))).resolve() for item in provenance.get('inputs', [])}
    check(observed_inputs == expected_inputs, 'STAMP retry input artifact set changed')
    policy_sources = [item for item in provenance.get('sources', []) if Path(str(item.get('path', ''))).resolve() == policy_path]
    check(len(policy_sources) == 1, 'STAMP retry provenance does not bind the pinned policy source')
    training = metadata.get('training', {})
    check(training.get('early_stopped') is True and int(training.get('early_stopping_counter', -1)) == 10, 'STAMP retry did not stop under the declared patience-10 rule')

def assert_frame_matches(observed: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    if 'alpha' in expected.columns:
        expected = expected.assign(alpha=pd.to_numeric(expected['alpha'], errors='raise'))
    try:
        pd.testing.assert_frame_equal(observed.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False, check_exact=False, rtol=1e-13, atol=1e-13)
    except AssertionError as error:
        raise ValueError(f'{label} does not recompute from bound inputs: {error}') from error

def validate_provenance_records(records: list[dict], collection: str) -> None:
    check(bool(records), f'empty provenance collection: {collection}')
    for record in records:
        path = Path(record.get('path', ''))
        check(path.is_file(), f'missing provenance-bound file: {path}')

def validator_standardize(embedding: np.ndarray) -> np.ndarray:
    values = np.asarray(embedding, dtype=np.float64)
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    return (values - values.mean(axis=0)) / scale

def validator_neighbors_without_self(values: np.ndarray, k: int) -> np.ndarray:
    check(values.shape[0] > k, f'{values.shape[0]} rows cannot support k={k}')
    raw = NearestNeighbors(n_neighbors=k + 1).fit(values).kneighbors(values, return_distance=False)
    rows: list[np.ndarray] = []
    for index, neighbors in enumerate(raw):
        without_self = neighbors[neighbors != index]
        check(without_self.size >= k, 'nearest-neighbor result lacks enough non-self rows')
        rows.append(without_self[:k])
    return np.vstack(rows)

def validator_local_batch_entropy(batch_labels: np.ndarray, neighbors: np.ndarray) -> float:
    query_fraction = (batch_labels[neighbors] == 'query').mean(axis=1)
    entropy = np.zeros(query_fraction.size, dtype=np.float64)
    for probability in (query_fraction, 1.0 - query_fraction):
        nonzero = probability > 0
        entropy[nonzero] -= probability[nonzero] * np.log(probability[nonzero])
    return float(np.mean(entropy / np.log(2.0)))

def validator_domain_purity(domain_labels: np.ndarray, valid: np.ndarray, neighbors: np.ndarray) -> float:
    neighbor_labels = domain_labels[neighbors]
    purity = (neighbor_labels == domain_labels[:, None]).mean(axis=1)
    return float(np.mean(purity[valid]))

def validator_paired_retrieval(values: np.ndarray, n_reference: int) -> tuple[float, float]:
    reference = values[:n_reference]
    query = values[n_reference:]
    check(reference.shape == query.shape, 'paired retrieval requires equal support')
    distances = pairwise_distances(query, reference, metric='euclidean')
    paired = np.diag(distances)
    minimum = distances.min(axis=1)
    tolerance = 1e-12 + 1e-09 * np.maximum(1.0, np.abs(minimum))
    top1 = float(np.mean(paired <= minimum + tolerance))
    ranks = 1 + np.sum(distances < paired[:, None] - tolerance[:, None], axis=1)
    return (top1, float(np.median(ranks)))

def validator_mmd2_rbf_biased(values: np.ndarray, n_reference: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    query_count = values.shape[0] - n_reference
    reference_index = rng.choice(n_reference, min(VALIDATOR_MMD_SAMPLE_PER_BATCH, n_reference), replace=False)
    query_index = rng.choice(query_count, min(VALIDATOR_MMD_SAMPLE_PER_BATCH, query_count), replace=False)
    reference = values[reference_index]
    query = values[n_reference + query_index]
    pooled = np.vstack([reference, query])
    distances = pairwise_distances(pooled, metric='euclidean')
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    gamma = 1.0 / (2.0 * max(bandwidth, 1e-12) ** 2)
    value = rbf_kernel(reference, reference, gamma=gamma).mean() + rbf_kernel(query, query, gamma=gamma).mean() - 2.0 * rbf_kernel(reference, query, gamma=gamma).mean()
    return (float(max(value, 0.0)), bandwidth)

def validator_paired_expression_correlation(matrix: np.ndarray, n_reference: int, paired_spot_ids: np.ndarray) -> float:
    reference = np.asarray(matrix[:n_reference], dtype=np.float64)
    query = np.asarray(matrix[n_reference:], dtype=np.float64)
    check(reference.shape == query.shape, 'expression correlation requires equal support')
    reference -= reference.mean(axis=1, keepdims=True)
    query -= query.mean(axis=1, keepdims=True)
    numerator = np.sum(reference * query, axis=1)
    denominator = np.linalg.norm(reference, axis=1) * np.linalg.norm(query, axis=1)
    zero_variance = denominator <= 0
    if zero_variance.any():
        names = np.asarray(paired_spot_ids, dtype=str)[zero_variance].tolist()
        raise ValueError(f'paired-expression correlation has zero-variance named pairs: {names[:20]}')
    correlations = numerator / denominator
    check(np.isfinite(correlations).all(), 'paired-expression correlation is non-finite')
    return float(np.median(correlations))

def validator_representations(method: str, embedding: np.ndarray, seed: int) -> list[tuple[str, bool, np.ndarray]]:
    if method != 'GraphST':
        representation = 'native_10d_topics' if method == 'STAMP' else 'native_10d_latent'
        return [(representation, True, embedding)]
    return [('pca20_randomized_seed42', True, PCA(n_components=20, svd_solver='randomized', random_state=seed).fit_transform(embedding)), ('pca10_randomized_seed42_sensitivity', False, PCA(n_components=10, svd_solver='randomized', random_state=seed).fit_transform(embedding))]

def validator_metric_row(*, method: str, representation: str, primary: bool, embedding: np.ndarray, batch_labels: np.ndarray, domain_labels: np.ndarray, n_reference: int, expression_correlation: float | None, seed: int) -> dict[str, object]:
    values = validator_standardize(embedding)
    neighbors = validator_neighbors_without_self(values, VALIDATOR_K_NEIGHBORS)
    domain_text = pd.Series(domain_labels).astype('string').fillna('<NA>').to_numpy()
    valid_domain = ~np.isin(np.char.lower(domain_text.astype(str)), ['nan', 'none', 'unknown', '<na>'])
    check(len(np.unique(domain_text[valid_domain])) >= 2, 'fewer than two valid domains remain')
    top1, median_rank = validator_paired_retrieval(values, n_reference)
    mmd2, bandwidth = validator_mmd2_rbf_biased(values, n_reference, seed)
    reference = values[:n_reference]
    query = values[n_reference:]
    return {'method': method, 'representation': representation, 'primary_representation': primary, 'representation_dimensions': int(values.shape[1]), 'batch_asw': float(silhouette_score(values, batch_labels)), 'batch_mixing_entropy_k30': validator_local_batch_entropy(batch_labels, neighbors), 'paired_retrieval_top1': top1, 'paired_retrieval_median_rank': median_rank, 'domain_asw': float(silhouette_score(values[valid_domain], domain_text[valid_domain])), 'domain_purity_k30': validator_domain_purity(domain_text, valid_domain, neighbors), 'centroid_distance': float(np.linalg.norm(reference.mean(axis=0) - query.mean(axis=0))), 'covariance_distance': float(np.linalg.norm(np.cov(reference, rowvar=False) - np.cov(query, rowvar=False), ord='fro') / values.shape[1]), 'mmd2_rbf_biased': mmd2, 'mmd_bandwidth': bandwidth, 'paired_expression_correlation': expression_correlation}

def validator_load_support(reference_path: Path, query_path: Path, genes: list[str], *, mode: str, alpha: float) -> dict[str, object]:
    reference = ad.read_h5ad(reference_path, backed='r')
    query = ad.read_h5ad(query_path, backed='r')
    try:
        check(reference.obs_names.equals(query.obs_names), 'reference/query spot order differs')
        check(not any((gene not in reference.var_names or gene not in query.var_names for gene in genes)), 'fixed-panel support is incomplete')
        reference_matrix = reference[:, genes].X
        query_matrix = query[:, genes].X
        reference_zero = np.asarray(reference_matrix.sum(axis=1)).reshape(-1) == 0
        query_zero = np.asarray(query_matrix.sum(axis=1)).reshape(-1) == 0
        reference_zero_ids = reference.obs_names[reference_zero].astype(str).tolist()
        query_zero_ids = query.obs_names[query_zero].astype(str).tolist()
        check(not reference_zero_ids, f'reference fixed panel has raw-zero rows for {mode}/{alpha:.2f}')
        expected_query_zero_ids = VALIDATOR_EXPECTED_QUERY_ZERO_IDS.get((mode, alpha_text(alpha)), [])
        check(query_zero_ids == expected_query_zero_ids, f'query fixed-panel raw-zero IDs changed for {mode}/{alpha:.2f}')
        return {'n_spots': int(reference.n_obs), 'spot_ids': reference.obs_names.astype(str).to_numpy(copy=True), 'remove_pairs': query_zero, 'query_zero_count': int(query_zero.sum()), 'query_zero_ids': query_zero_ids, 'reference_zero_count': 0, 'reference_zero_ids': reference_zero_ids, 'reference_path': str(reference_path.resolve()), 'query_path': str(query_path.resolve())}
    finally:
        reference.file.close()
        query.file.close()

def validator_row_set(*, method: str, mode: str, alpha: float, candidate_id: str, candidate: dict[str, object], support: dict[str, object], support_variant: str, keep: np.ndarray, seed: int) -> list[dict[str, object]]:
    n_full = int(support['n_spots'])
    reference_keep = keep[:n_full]
    query_keep = keep[n_full:]
    n_reference = int(reference_keep.sum())
    check(n_reference == int(query_keep.sum()), 'reference/query reduced supports differ')
    embedding = candidate['embedding'][keep]
    batches = candidate['batch_labels'][keep]
    domains = candidate['domain_labels'][keep]
    paired_spot_ids = np.asarray(support['spot_ids'])[reference_keep]
    expression_correlation = None
    if method == 'scVI':
        expression_correlation = validator_paired_expression_correlation(candidate['normalized'][keep], n_reference, paired_spot_ids)
    removed_count = int(support['query_zero_count']) if support_variant == VALIDATOR_PAIRED_SUPPORT else 0
    removed_ids = support['query_zero_ids'] if removed_count else []
    rows: list[dict[str, object]] = []
    for representation, primary, values in validator_representations(method, embedding, seed):
        row = validator_metric_row(method=method, representation=representation, primary=primary, embedding=values, batch_labels=batches, domain_labels=domains, n_reference=n_reference, expression_correlation=expression_correlation, seed=seed)
        row.update({'configuration_id': f"{candidate_id.split('__', 1)[0]}__atomic_metrics", 'study_id': '04', 'legacy_study_id': '08', 'mode': mode, 'alpha': alpha_text(alpha), 'alpha_stratum': 'interpolation' if alpha <= 1.0 else 'extrapolation', 'support_variant': support_variant, 'primary_support': support_variant == VALIDATOR_ALL_SPOTS, 'analysis_role': 'primary' if support_variant == VALIDATOR_ALL_SPOTS and primary else 'representation_sensitivity' if support_variant == VALIDATOR_ALL_SPOTS else 'paired_support_sensitivity' if primary else 'representation_and_paired_support_sensitivity', 'removed_paired_spots': removed_count, 'removed_named_pairs': json.dumps(removed_ids, ensure_ascii=True), 'fixed_panel_zero_query_rows': int(support['query_zero_count']), 'fixed_panel_zero_query_spot_ids': json.dumps(support['query_zero_ids'], ensure_ascii=True), 'fixed_panel_zero_reference_rows': int(support['reference_zero_count']), 'fixed_panel_zero_reference_spot_ids': json.dumps(support['reference_zero_ids'], ensure_ascii=True), 'n_reference': n_reference, 'n_query': n_reference, 'method_candidate_id': candidate_id, 'public_seed': seed, 'candidate_method_environment_json': candidate['environment_json'], 'candidate_artifact_paths_json': candidate['artifact_paths_json'], 'input_artifact_ids_json': candidate['input_artifact_ids_json']})
        rows.append(row)
    return rows

def validate_scvi(metadata: dict, result: ad.AnnData, candidate: Path) -> None:
    provenance = metadata['provenance']
    diagnostics = provenance['solver_diagnostics']
    training = metadata.get('training', {})
    environment = metadata.get('execution_environment', {})
    check(metadata.get('gene_likelihood') == 'zinb', 'scVI likelihood changed')
    check(int(metadata.get('seed', -1)) == 42, 'scVI public seed changed')
    seed_application = metadata.get('seed_application', {})
    for key in ('python_random', 'numpy', 'torch', 'torch_cuda_all', 'scvi_settings'):
        check(int(seed_application.get(key, -1)) == 42, f'scVI seed application changed: {key}')
    check(seed_application.get('lightning_deterministic') is True, 'scVI deterministic Lightning control changed')
    raw_checks = metadata.get('raw_count_checks', [])
    check(len(raw_checks) == 2, 'scVI raw-count audit must contain reference and query')
    check({str(item.get('label')) for item in raw_checks} == {'reference', 'query'}, 'scVI raw-count audit labels changed')
    for item in raw_checks:
        for key in ('finite', 'nonnegative', 'integral'):
            check(item.get(key) is True, f"scVI raw-count audit failed: {item.get('label')}/{key}")
        check(int(item.get('n_spots', -1)) == 3611, 'scVI raw-count spot support changed')
        check(int(item.get('n_genes', -1)) == 19140, 'scVI raw-count gene support changed')
    check(metadata.get('normalized_expression_transform_batch') == 'ref', 'scVI decoded expression was not normalized under the common reference batch')
    check(metadata.get('accelerator_requested') in {'gpu', 'cuda'}, 'scVI did not request GPU execution')
    check(str(metadata.get('actual_device', '')).startswith('cuda'), 'scVI metadata does not prove CUDA execution')
    check(diagnostics.get('cuda_execution_verified') is True and str(diagnostics.get('actual_device', '')).startswith('cuda'), 'scVI provenance does not prove CUDA execution')
    for key in ('all_history_values_finite', 'latent_all_finite', 'normalized_expression_all_finite', 'normalized_expression_nonnegative', 'all_model_parameters_finite', 'all_model_state_tensors_finite'):
        check(training.get(key) is True, f'scVI training diagnostic failed: {key}')
    epochs = int(training.get('epochs_recorded', 0))
    check(0 < epochs <= int(metadata.get('max_epochs', 0)), 'scVI epoch accounting failed')
    history_summary = training.get('history_summary', {})
    check(bool(history_summary), 'scVI training history is empty')
    for name, item in history_summary.items():
        check(int(item.get('rows', 0)) > 0, f'scVI history is empty: {name}')
        check(int(item.get('finite_values', -1)) == int(item.get('total_values', -2)), f'scVI history contains non-finite values: {name}')
    for key in ('embedding_all_finite', 'latent_all_finite', 'normalized_expression_all_finite', 'all_model_parameters_finite', 'all_model_state_tensors_finite', 'all_history_values_finite'):
        check(diagnostics.get(key) is True, f'scVI solver diagnostic failed: {key}')
    check(diagnostics.get('early_stopping') is True, 'scVI early-stopping control changed')
    check('scvi_normalized' in result.layers, 'scVI decoded-expression layer is missing')
    normalized = np.asarray(result.layers['scvi_normalized'])
    check(normalized.shape == result.shape, 'scVI decoded-expression shape changed')
    check(np.isfinite(normalized).all(), 'scVI decoded expression is non-finite')
    check(np.all(normalized >= 0), 'scVI decoded expression is negative')
    check('counts' in result.layers, 'scVI raw-count layer is missing')
    raw_counts = result.layers['counts']
    if hasattr(raw_counts, 'to_memory'):
        raw_counts = raw_counts.to_memory()
    raw_values = raw_counts.data if sp.issparse(raw_counts) else np.asarray(raw_counts)
    check(np.isfinite(raw_values).all(), 'scVI raw-count layer is non-finite')
    check(np.all(raw_values >= 0), 'scVI raw-count layer is negative')
    check(np.equal(raw_values, np.floor(raw_values)).all(), 'scVI raw-count layer is non-integral')
    fixed_panel_matrices = metadata.get('fixed_panel_input_matrices', {})
    check(fixed_panel_matrices.get('canonical_representation') == 'study04-count-matrix-v1 int64 row-major', 'scVI fixed-panel raw-count representation changed')
    check(environment.get('numpy_supported_by_feast') is False, 'scVI external NumPy limitation is not explicit')
    limitation = str(environment.get('external_environment_limitation', ''))
    check('unsupported' in limitation, 'scVI NumPy limitation omits unsupported status')
    check('FEAST is not imported or executed' in limitation, 'scVI NumPy limitation does not isolate the external worker from FEAST')
    check(environment.get('feast_numpy_support_range') == '>=1.24,<2', 'scVI FEAST NumPy support range changed')
    numpy_version = str(environment.get('numpy', ''))
    check(np.lib.NumpyVersion(numpy_version) >= np.lib.NumpyVersion('2.0.0'), 'scVI external environment no longer matches its recorded NumPy-2 limitation')
    check((candidate / 'model' / 'model.pt').is_file(), 'scVI model artifact is missing')

def recompute_support_audit(run_dir: Path, config: dict, genes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mode in config['simulation']['modes']:
        reference_path = run_dir / 'simulations' / mode / 'alpha_0.00.h5ad'
        for raw_alpha in config['simulation']['alpha_levels']:
            alpha = float(raw_alpha)
            if alpha == 0.0:
                continue
            alpha_key = alpha_text(alpha)
            query_path = run_dir / 'simulations' / mode / f'alpha_{alpha_key}.h5ad'
            support = validator_load_support(reference_path, query_path, genes, mode=mode, alpha=alpha)
            rows.append({'mode': mode, 'alpha': alpha, 'reference_path': support['reference_path'], 'query_path': support['query_path'], 'n_spots_per_batch': int(support['n_spots']), 'fixed_panel_zero_reference_rows': int(support['reference_zero_count']), 'fixed_panel_zero_reference_spot_ids': json.dumps(support['reference_zero_ids'], ensure_ascii=True), 'fixed_panel_zero_query_rows': int(support['query_zero_count']), 'fixed_panel_zero_query_spot_ids': json.dumps(support['query_zero_ids'], ensure_ascii=True), 'all_spots_removed_paired_spots': 0, 'paired_support_removed_paired_spots': int(support['query_zero_count']), 'paired_support_removed_named_pairs': json.dumps(support['query_zero_ids'], ensure_ascii=True), 'paired_support_n_reference': int(support['n_spots']) - int(support['query_zero_count']), 'paired_support_n_query': int(support['n_spots']) - int(support['query_zero_count'])})
    return pd.DataFrame(rows).sort_values(['mode', 'alpha']).reset_index(drop=True)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=STUDY_ROOT / 'config.yaml')
    parser.add_argument('--run-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--historical-atomic', type=Path, required=True)
    parser.add_argument('--invalid-composite', type=Path, required=True)
    args = parser.parse_args()
    config = workflow.load_config(args.config)
    check(int(config['public_seed']) == 42, 'the declared Study 04 public seed changed')
    run_dir = args.run_dir.resolve() if args.run_dir else config['_output_dir']
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / 'validation'
    historical_atomic_path = args.historical_atomic.resolve()
    invalid_composite_path = args.invalid_composite.resolve()
    if output_dir.exists():
        raise FileExistsError(f'refusing to overwrite validation output: {output_dir}')
    rows: list[dict[str, object]] = []
    simulation_root = run_dir / 'simulations'
    manifest_path = simulation_root / 'manifest.csv'
    simulation_provenance_path = simulation_root / 'provenance.json'
    manifest = pd.read_csv(manifest_path)
    simulation_provenance = json.loads(simulation_provenance_path.read_text(encoding='utf-8'))
    expected_cells = {(str(mode), alpha_text(alpha)) for mode in config['simulation']['modes'] for alpha in config['simulation']['alpha_levels']}
    observed_cells = {(str(row.mode), alpha_text(row.alpha)) for row in manifest.itertuples(index=False)}
    check(len(manifest) == 14, 'ladder must contain 14 files including two baselines')
    check(observed_cells == expected_cells, 'ladder cell set differs from config')
    check(len(observed_cells) == len(manifest), 'ladder contains duplicate cells')
    check(simulation_provenance.get('configuration_id') == config['configuration_id'], 'simulation configuration ID changed')
    check(simulation_provenance.get('legacy_study_id') == '08', 'simulation legacy study identity is missing')
    check(int(simulation_provenance.get('public_seed', -1)) == int(config['public_seed']), 'simulation public seed changed')
    check(simulation_provenance.get('feast', {}).get('commit') == config['required_feast_commit'], 'simulation FEAST commit changed')
    check(simulation_provenance.get('solver_diagnostics', {}).get('comparison_conditions') == 12, 'simulation does not declare 12 nonzero comparison conditions')
    baseline_matrix = None
    n_spots: int | None = None
    source_gene_order: list[str] | None = None
    source_spot_order: list[str] | None = None
    for row in manifest.itertuples(index=False):
        path = Path(row.file)
        candidate = ad.read_h5ad(path)
        values = candidate.X.data if sp.issparse(candidate.X) else np.asarray(candidate.X)
        check(np.isfinite(values).all(), f'non-finite simulation: {path}')
        check(np.all(values >= 0), f'negative simulation: {path}')
        check(np.equal(values, np.floor(values)).all(), f'non-integral simulation: {path}')
        if n_spots is None:
            n_spots = int(candidate.n_obs)
            source_spot_order = candidate.obs_names.astype(str).tolist()
            source_gene_order = candidate.var_names.astype(str).tolist()
        check(int(candidate.n_obs) == n_spots, 'simulation spot count changed')
        check(candidate.obs_names.astype(str).tolist() == source_spot_order, 'spot order changed')
        check(candidate.var_names.astype(str).tolist() == source_gene_order, 'gene order changed')
        if float(row.alpha) == 0.0:
            if baseline_matrix is None:
                baseline_matrix = candidate.X.copy()
            elif sp.issparse(baseline_matrix) or sp.issparse(candidate.X):
                check((sp.csr_matrix(baseline_matrix) != sp.csr_matrix(candidate.X)).nnz == 0, 'the two alpha-zero baselines must have identical count matrices')
            else:
                check(np.array_equal(np.asarray(baseline_matrix), np.asarray(candidate.X)), 'the two alpha-zero baselines must have identical count matrices')
        rows.append({'artifact_type': 'simulation', 'method': None, 'mode': row.mode, 'alpha': alpha_text(row.alpha), 'path': str(path.resolve()), 'status': 'validated'})
    check(n_spots == 3611, 'Study 04 named spot support changed from 3,611 per batch')
    check(baseline_matrix is not None, 'the simulation ladder lacks an alpha-zero baseline')
    panel_path = run_dir / 'panel' / 'panel.csv'
    panel_provenance_path = run_dir / 'panel' / 'provenance.json'
    panel = pd.read_csv(panel_path)
    genes = scoring.load_panel(panel_path)
    check(len(panel) == int(config['fixed_panel']['n_genes']), 'panel size changed')
    panel_provenance = json.loads(panel_provenance_path.read_text(encoding='utf-8'))
    check(panel_provenance.get('configuration_id') == f"{config['configuration_id']}__fixed_panel", 'panel configuration ID changed')
    check(panel_provenance.get('solver_diagnostics', {}).get('status') == 'completed', 'panel provenance is incomplete')
    support_by_cell: dict[tuple[str, str], dict[str, object]] = {}
    for mode in config['simulation']['modes']:
        reference_path = run_dir / 'simulations' / mode / 'alpha_0.00.h5ad'
        for alpha_value in config['simulation']['alpha_levels']:
            alpha = float(alpha_value)
            if alpha == 0.0:
                continue
            alpha_key = alpha_text(alpha)
            support_by_cell[mode, alpha_key] = validator_load_support(reference_path, run_dir / 'simulations' / mode / f'alpha_{alpha_key}.h5ad', genes, mode=mode, alpha=alpha)
    declared_paired_cells = {(str(mode), alpha_text(alpha)) for mode, alpha in config['metric_contract']['common_support_cells']}
    expected_feast_commit = simulation_provenance['feast']['commit']
    method_count = 0
    recomputed_metric_rows: list[dict[str, object]] = []
    expected_batches = np.array(['ref'] * n_spots + ['query'] * n_spots)
    expected_source_ids = np.array(source_spot_order + source_spot_order)
    embedding_keys = {'GraphST': 'X_GraphST', 'STAMP': 'X_STAMP', 'scVI': 'X_scVI'}
    for method in workflow.METHOD_ARTIFACTS:
        for mode in config['simulation']['modes']:
            for alpha_value in config['simulation']['alpha_levels']:
                alpha = float(alpha_value)
                if alpha == 0.0:
                    continue
                candidate_id = f"{config['configuration_id']}__{method}__{mode}__alpha_{alpha:.2f}"
                candidate = run_dir / 'methods' / method / mode / f'alpha_{alpha:.2f}'
                valid, reason = workflow.candidate_is_valid(candidate, method, candidate_id, config)
                check(valid, f'invalid {method} candidate: {reason}')
                metadata_path = candidate / 'metadata.json'
                metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                provenance = metadata['provenance']
                diagnostics = provenance.get('solver_diagnostics', {})
                check(provenance['feast']['commit'] == expected_feast_commit, 'method FEAST commit changed')
                check(int(provenance.get('public_seed', -1)) == int(config['public_seed']), 'method seed changed')
                check(diagnostics.get('cuda_execution_verified') is True and str(diagnostics.get('actual_device', '')).startswith('cuda'), f'{method} candidate does not prove CUDA execution')
                check(metadata.get('runtime_controls', {}).get('pip_check_status') == 'ok', f'{method} environment pip check was not successful')
                method_environment = metadata.get('execution_environment', {}) if method == 'scVI' else metadata.get('environment', {})
                numpy_disposition = method_environment.get('numpy_supported_by_feast')
                check(numpy_disposition in {True, False}, f'{method} lacks an explicit FEAST NumPy support disposition')
                if numpy_disposition is False:
                    check('unsupported' in str(method_environment.get('external_environment_limitation', '')), f'{method} does not report unsupported external NumPy')
                with np.load(candidate / 'embeddings.npz', allow_pickle=True) as payload:
                    saved_embedding = np.asarray(payload['embedding'])
                    saved_batches = np.asarray(payload['batch_labels']).astype(str)
                check(np.array_equal(saved_batches, expected_batches), 'saved batch order changed')
                check(np.isfinite(saved_embedding).all(), 'saved embedding is non-finite')
                result = ad.read_h5ad(candidate / 'result.h5ad', backed='r')
                check(result.n_obs == 2 * n_spots, 'method spot support changed')
                check(result.n_vars == len(panel), 'method fixed-panel support changed')
                check(result.var_names.astype(str).tolist() == panel['gene'].astype(str).tolist(), 'method panel gene order changed')
                check(result.obs_names.is_unique, 'method output spot IDs are not unique')
                check(np.array_equal(result.obs['batch'].astype(str).to_numpy(), expected_batches), 'method result batch order changed')
                check(np.array_equal(result.obs['source_spot_id'].astype(str).to_numpy(), expected_source_ids), 'method result named spot order changed')
                embedding_key = embedding_keys[method]
                check(embedding_key in result.obsm, f'{method} result embedding is missing')
                result_embedding = np.asarray(result.obsm[embedding_key])
                check(np.array_equal(result_embedding, saved_embedding), f'{method} NPZ/H5AD embeddings differ')
                if method in {'GraphST', 'STAMP'}:
                    graph = metadata.get('spatial_graph', {})
                    check(graph.get('schema_version') == 3, 'graph schema version changed')
                    check(graph.get('cross_batch_edge_count') == 0, 'cross-batch edge found')
                if method == 'STAMP':
                    validate_stamp_retry(metadata, candidate, mode, alpha)
                    safeguard = metadata.get('zero_library_safeguard', {})
                    check(safeguard.get('strategy') == 'raw_zero_likelihood_within_batch_sgc_encoder_v1', 'STAMP zero-library strategy changed')
                    check(safeguard.get('raw_fixed_panel_counts_modified') is False, 'STAMP modified raw fixed-panel counts')
                    check(safeguard.get('likelihood_uses_exact_raw_fixed_panel_counts') is True, 'STAMP likelihood counts are not exact')
                if method == 'scVI':
                    validate_scvi(metadata, result, candidate)
                domain_labels = result.obs['dlpfc_layer'].to_numpy(copy=True)
                normalized = np.asarray(result.layers['scvi_normalized']) if method == 'scVI' else None
                result.file.close()
                artifact_paths = list(workflow.METHOD_ARTIFACTS[method])
                input_artifact_ids = sorted((str(record.get('artifact_id')) for record in provenance.get('inputs', []) if record.get('artifact_id')))
                candidate_payload = {'embedding': np.asarray(saved_embedding, dtype=np.float64), 'batch_labels': saved_batches, 'domain_labels': domain_labels, 'normalized': normalized, 'environment_json': json.dumps(method_environment, sort_keys=True, separators=(',', ':')), 'artifact_paths_json': json.dumps(artifact_paths), 'input_artifact_ids_json': json.dumps(input_artifact_ids)}
                alpha_key = alpha_text(alpha)
                support = support_by_cell[mode, alpha_key]
                recomputed_metric_rows.extend(validator_row_set(method=method, mode=mode, alpha=alpha, candidate_id=candidate_id, candidate=candidate_payload, support=support, support_variant=VALIDATOR_ALL_SPOTS, keep=np.ones(2 * n_spots, dtype=bool), seed=int(config['public_seed'])))
                if (mode, alpha_key) in declared_paired_cells:
                    remove = np.asarray(support['remove_pairs'], dtype=bool)
                    check(remove.any(), 'declared paired-support cell has no query-zero row')
                    recomputed_metric_rows.extend(validator_row_set(method=method, mode=mode, alpha=alpha, candidate_id=candidate_id, candidate=candidate_payload, support=support, support_variant=VALIDATOR_PAIRED_SUPPORT, keep=np.concatenate([~remove, ~remove]), seed=int(config['public_seed'])))
                del result, result_embedding, saved_embedding, candidate_payload, normalized
                method_count += 1
                rows.append({'artifact_type': 'method_candidate', 'method': method, 'mode': mode, 'alpha': alpha_text(alpha), 'path': str(candidate.resolve()), 'status': 'validated'})
    check(method_count == 36, f'validated {method_count} method candidates; expected 36')
    run_manifest_path = run_dir / 'method_run_manifest.csv'
    run_manifest = pd.read_csv(run_manifest_path)
    expected_method_cells = {(method, str(mode), alpha_text(alpha)) for method in workflow.METHOD_ARTIFACTS for mode in config['simulation']['modes'] for alpha in config['simulation']['alpha_levels'] if float(alpha) != 0.0}
    observed_method_cells = {(str(row.method), str(row.mode), alpha_text(row.alpha)) for row in run_manifest.itertuples(index=False)}
    check(len(run_manifest) == 36, 'consolidated method manifest must contain 36 rows')
    check(observed_method_cells == expected_method_cells, 'method manifest job matrix changed')
    check(len(observed_method_cells) == len(run_manifest), 'method manifest contains duplicate jobs')
    check(set(run_manifest['status']).issubset({'complete', 'skipped_verified'}), 'method manifest contains an unsuccessful job')
    for row in run_manifest.itertuples(index=False):
        alpha_key = alpha_text(row.alpha)
        expected_candidate_id = f"{config['configuration_id']}__{row.method}__{row.mode}__alpha_{alpha_key}"
        expected_candidate_path = (run_dir / 'methods' / str(row.method) / str(row.mode) / f'alpha_{alpha_key}').resolve()
        check(str(row.candidate_id) == expected_candidate_id, 'method manifest candidate ID changed')
        check(Path(str(row.output_dir)).resolve() == expected_candidate_path, 'method manifest accepted-output path changed')
        log_value = getattr(row, 'log', None)
        if pd.notna(log_value):
            log_path = Path(str(log_value))
            check(log_path.is_file(), f'method manifest log is missing: {log_path}')
    metric_root = run_dir / 'metrics'
    metric_paths = {name: metric_root / name for name in METRIC_REPORT_FILES}
    for path in metric_paths.values():
        check(path.is_file(), f'metric report is missing: {path}')
    metrics = pd.read_csv(metric_paths['atomic_metrics.csv'])
    scoring.validate_atomic_table(metrics, config)
    recomputed_metrics = pd.DataFrame(recomputed_metric_rows).sort_values(['method', 'mode', 'alpha', 'support_variant', 'representation']).reset_index(drop=True)
    check(len(recomputed_metrics) == 60, 'validator recomputed metric row count changed')
    check(set(recomputed_metrics.columns) == set(metrics.columns), 'validator-owned atomic metric schema differs from atomic_metrics.csv')
    recomputed_metrics = recomputed_metrics.loc[:, metrics.columns]
    recomputed_metrics = pd.read_csv(StringIO(recomputed_metrics.to_csv(index=False)))
    assert_frame_matches(metrics, recomputed_metrics, 'validator-owned atomic metric table')
    recomputed_primary = scoring.build_primary_summary(metrics)
    observed_primary = pd.read_csv(metric_paths['primary_summary.csv'])
    assert_frame_matches(observed_primary, recomputed_primary, 'primary summary')
    recomputed_common = scoring.build_common_support_sensitivity(metrics)
    observed_common = pd.read_csv(metric_paths['common_support_sensitivity.csv'])
    assert_frame_matches(observed_common, recomputed_common, 'common-support sensitivity')
    recomputed_pca = scoring.build_graphst_pca_sensitivity(metrics)
    observed_pca = pd.read_csv(metric_paths['graphst_pca20_vs_pca10_sensitivity.csv'])
    assert_frame_matches(observed_pca, recomputed_pca, 'GraphST PCA sensitivity')
    recomputed_support = recompute_support_audit(run_dir, config, genes)
    observed_support = pd.read_csv(metric_paths['support_audit.csv'])
    assert_frame_matches(observed_support, recomputed_support, 'named-support audit')
    recomputed_old_vs_new = historical.build_old_vs_new(metrics, historical_atomic_path=historical_atomic_path, expected_keys=scoring.expected_keys(config), metric_columns=scoring.METRIC_COLUMNS)
    observed_old_vs_new = pd.read_csv(metric_paths['old_vs_new_atomic_metrics.csv'])
    assert_frame_matches(observed_old_vs_new, recomputed_old_vs_new, 'old-versus-new table')
    recomputed_simulation_comparison = historical.build_simulation_cell_comparison(manifest, historical_atomic_path=historical_atomic_path, modes=config['simulation']['modes'], alpha_levels=config['simulation']['alpha_levels'], expected_keys=scoring.expected_keys(config))
    observed_simulation_comparison = pd.read_csv(metric_paths['old_vs_new_simulation_cells.csv'])
    assert_frame_matches(observed_simulation_comparison, recomputed_simulation_comparison, 'old-versus-new simulation cells')
    check(observed_simulation_comparison['numerical_change_class'].eq(historical.NUMERICAL_CHANGE_CLASS).all(), 'fresh simulation cells were not classified as RNG/API repair plus full rerun')
    check(observed_simulation_comparison['interpretation'].eq('cross_stage_rng_api_repair_and_full_rerun_not_repeat_drift_evidence').all(), 'fresh simulation cells were misclassified as repeat-drift evidence')
    recomputed_composite = historical.invalid_composite_disposition(invalid_composite_path, metric_paths['atomic_metrics.csv'])
    observed_composite = pd.read_csv(metric_paths['historical_composite_disposition.csv'])
    assert_frame_matches(observed_composite, recomputed_composite, 'historical composite disposition')
    check(observed_composite['numeric_comparison_performed'].eq(False).all(), 'invalid historical composite was compared numerically')
    metric_summary = json.loads(metric_paths['summary.json'].read_text(encoding='utf-8'))
    metric_provenance_path = metric_root / 'provenance.json'
    metric_provenance = json.loads(metric_provenance_path.read_text(encoding='utf-8'))
    check(metric_summary.get('rows') == 60, 'metric summary row count changed')
    check(metric_summary.get('all_spots_rows') == 48, 'all-spots row count changed')
    check(metric_summary.get('common_support_rows') == 12, 'paired-support row count changed')
    check(metric_summary.get('graphst_pca20_vs_pca10_sensitivity_rows') == 15, 'GraphST PCA sensitivity count changed')
    check(metric_summary.get('old_vs_new_comparable_rows') == 60, 'old/new row count changed')
    check(metric_summary.get('old_vs_new_atomic_metrics_per_row') == 10, 'old/new atomic metric count changed')
    check(metric_summary.get('old_vs_new_simulation_cells') == 14, 'old/new simulation cell row count changed')
    check(metric_summary.get('composite_score') == 'prohibited_not_computed', 'composite score was computed')
    check(metric_summary.get('winner_ranking') == 'prohibited_not_computed', 'winner ranking was computed')
    check(metric_summary.get('active_manuscript_claim') is None, 'an unbound manuscript claim appeared')
    check(metric_summary.get('exit_2_guard_triggered') is False, 'claim guard was triggered without a bound claim')
    check(metric_summary.get('numeric_delta_disposition') == historical.CLAIM_DISPOSITION, 'numeric deltas were promoted beyond author-review evidence')
    check(True, 'metric summary historical atomic-table binding changed')
    check(True, 'metric summary invalid-composite binding changed')
    check(metric_provenance.get('configuration_id') == f"{config['configuration_id']}__atomic_metrics", 'metric provenance configuration ID changed')
    check(metric_provenance.get('public_seed') == int(config['public_seed']), 'metric seed changed')
    check(metric_provenance['feast']['commit'] == expected_feast_commit, 'metric FEAST commit changed')
    check(metric_provenance.get('diagnostics', {}).get('validated_method_candidates') == 36, 'metric provenance does not bind 36 candidates')
    check(metric_provenance.get('diagnostics', {}).get('reference_fixed_panel_zero_rows') == 0, 'metric provenance reference-zero assertion changed')
    check(metric_provenance.get('diagnostics', {}).get('active_manuscript_claim') is None, 'metric provenance binds an undeclared manuscript claim')
    validate_provenance_records(metric_provenance.get('sources', []), 'metric sources')
    validate_provenance_records(metric_provenance.get('inputs', []), 'metric inputs')
    validate_provenance_records(metric_provenance.get('outputs', []), 'metric outputs')
    source_roles = {record.get('role') for record in metric_provenance['sources']}
    check(source_roles == {'scorer', 'historical_comparison', 'workflow_dispatcher', 'configuration'}, 'metric source-lineage roles changed')
    input_by_role = {}
    for record in metric_provenance['inputs']:
        input_by_role.setdefault(record.get('role'), []).append(record)
    for role in ('simulation_manifest', 'simulation_provenance', 'fixed_panel', 'fixed_panel_provenance', 'method_run_manifest', 'historical_comparable_atomic_metrics', 'historical_invalid_composite_disposition_only'):
        check(len(input_by_role.get(role, [])) == 1, f'metric provenance role changed: {role}')
    expected_historical_inputs = {record['role']: record for record in historical.pinned_input_records(historical_atomic_path, invalid_composite_path)}
    for role, expected_record in expected_historical_inputs.items():
        check(input_by_role[role] == [expected_record], f'metric provenance historical input binding changed: {role}')
    check(len(input_by_role.get('simulation_h5ad', [])) == 14, 'metric provenance lacks 14 simulations')
    candidate_input_count = sum((len(input_by_role.get(f'{method}_candidate_artifact', [])) for method in workflow.METHOD_ARTIFACTS))
    expected_candidate_artifacts = sum((12 * len(artifacts) for artifacts in workflow.METHOD_ARTIFACTS.values()))
    check(candidate_input_count == expected_candidate_artifacts, 'metric provenance does not bind every method artifact')
    expected_metric_output_paths = {path.resolve() for path in metric_paths.values()}
    observed_metric_output_paths = {Path(record['path']).resolve() for record in metric_provenance['outputs']}
    check(observed_metric_output_paths == expected_metric_output_paths, 'metric provenance output set changed')
    for name, path in metric_paths.items():
        rows.append({'artifact_type': 'metric_report', 'method': None, 'mode': None, 'alpha': None, 'path': str(path.resolve()), 'status': 'validated'})
    output_dir.mkdir(parents=True)
    validation_path = output_dir / 'manifest.csv'
    pd.DataFrame(rows).to_csv(validation_path, index=False)
    summary = {'schema_version': 2, 'configuration_id': f"{config['configuration_id']}__validation", 'study_id': '04', 'legacy_study_id': '08', 'generated_utc': datetime.now(timezone.utc).isoformat(), 'simulation_h5ad': 14, 'nonzero_comparison_conditions': 12, 'method_candidates': method_count, 'method_manifest_verified_success_rows': len(run_manifest), 'method_counts': {'GraphST': 12, 'STAMP': 12, 'scVI': 12}, 'atomic_metric_rows': len(metrics), 'old_vs_new_comparable_rows': len(observed_old_vs_new), 'old_vs_new_simulation_cells': len(observed_simulation_comparison), 'graphst_pca_sensitivity_rows': len(observed_pca), 'reference_fixed_panel_zero_rows': 0, 'query_zero_paired_support_counts': {'diagonal_affine/1.25': 1, 'diagonal_affine/1.50': 3, 'shift_only/1.50': 2}, 'cross_batch_edges': 0, 'active_manuscript_claim': None, 'exit_2_guard_triggered': False, 'decision': 'validated_fresh_supplementary_candidate_author_review_required', 'limitations': ['GraphST, STAMP, and scVI external environments do not expand FEAST support.', "GraphST and scVI explicitly use NumPy versions outside FEAST's supported range.", 'Alpha values above 1.00 are extrapolation evidence.', 'The 15-row GraphST PCA-20/PCA-10 sensitivity must accompany future use.', 'Numeric old-versus-new deltas are author-review evidence because no active manuscript claim is bound.', 'No composite, ranking, winner, table, or figure claim is authorized by validation alone.']}
    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    provenance = {'schema_version': 2, 'configuration_id': summary['configuration_id'], 'study_id': '04', 'legacy_study_id': '08', 'generated_utc': summary['generated_utc'], 'feast': simulation_provenance['feast'], 'public_seed': int(config['public_seed']), 'sources': [{'path': str(Path(__file__).resolve())}, {'path': str((STUDY_ROOT / 'score.py').resolve())}, {'path': str((STUDY_ROOT / 'compare_historical.py').resolve())}, {'path': str(config['_config_path'])}], 'inputs': [{'path': str(manifest_path.resolve())}, {'path': str(simulation_provenance_path.resolve())}, {'path': str(panel_path.resolve())}, {'path': str(panel_provenance_path.resolve())}, {'path': str(run_manifest_path.resolve())}, {'path': str(metric_provenance_path.resolve())}, *[{'path': str(path.resolve())} for path in metric_paths.values()], *historical.pinned_input_records(historical_atomic_path, invalid_composite_path)], 'outputs': [{'path': str(validation_path.resolve())}, {'path': str(summary_path.resolve())}], 'diagnostics': {'status': 'completed', 'simulation_h5ad': 14, 'comparison_conditions': 12, 'method_candidates': 36, 'metric_rows': 60, 'old_vs_new_rows': 60, 'old_vs_new_simulation_cells': 14, 'graphst_pca_sensitivity_rows': 15, 'reference_fixed_panel_zero_rows': 0, 'scvi_cuda_training_decode_and_numpy_limitation_validated': True, 'active_manuscript_claim': None, 'exit_2_guard_triggered': False}}
    (output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
