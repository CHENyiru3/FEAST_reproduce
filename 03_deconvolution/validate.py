"""Validate the fresh Study 03 run and its matched-support score layer."""
from __future__ import annotations
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy.special import rel_entr
from scipy.stats import pearsonr
STUDY_ROOT = Path(__file__).resolve().parent
KEY_COLUMNS = ('method', 'slice_id', 'resolution')
METRICS = ('jsd_mean', 'pearson_mean', 'summed_rmse')
SCORE_METRICS = ('jsd_mean', 'jsd_median', 'pearson_mean', 'pearson_median', 'summed_rmse')
DIRECTION_TOLERANCE = 1e-12
CHANGE_TOLERANCE = 1e-12
EXPECTED_PREFERENCES = {'jsd_mean': 'lower', 'pearson_mean': 'higher', 'summed_rmse': 'lower'}
CHANGE_CLASSIFICATIONS = {'historical_to_prior': 'mixed_workflow_non_attributable_historical_to_prior_repair', 'prior_to_fresh': 'mixed_workflow_non_attributable_prior_repair_to_fresh_full_rerun', 'historical_to_fresh': 'mixed_workflow_non_attributable_historical_to_fresh_full_rerun'}

def artifact(path: Path, artifact_id: str) -> dict[str, str]:
    return {'artifact_id': artifact_id, 'path': str(path.resolve())}

def integer_libraries(matrix, label: str) -> np.ndarray:
    if hasattr(matrix, 'toarray'):
        data = np.asarray(matrix.data, dtype=np.float64)
        libraries = np.asarray(matrix.sum(axis=1)).ravel()
    else:
        values = np.asarray(matrix, dtype=np.float64)
        data, libraries = (values.ravel(), values.sum(axis=1))
    if not np.isfinite(data).all() or data.min(initial=0) < 0:
        raise RuntimeError(f'{label} has invalid counts')
    if np.max(np.abs(data - np.rint(data)), initial=0) > 1e-06:
        raise RuntimeError(f'{label} is not raw integer count data')
    return libraries

def with_other(frame: pd.DataFrame, common: pd.Index) -> np.ndarray:
    values = np.clip(frame.loc[:, common].to_numpy(np.float64), 0, None)
    residual = np.clip(1 - values.sum(axis=1), 0, None)
    result = np.column_stack([values, residual])
    totals = result.sum(axis=1, keepdims=True)
    if np.any(totals[:, 0] <= 0):
        raise RuntimeError('scored composition contains a zero-mass row')
    return result / totals

def recompute_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    midpoint = 0.5 * (truth + prediction)
    jsd = 0.5 * (rel_entr(truth, midpoint).sum(axis=1) + rel_entr(prediction, midpoint).sum(axis=1)) / np.log(2)
    correlations = []
    for index in range(truth.shape[1]):
        left, right = (truth[:, index], prediction[:, index])
        correlations.append(np.nan if np.std(left) == 0 or np.std(right) == 0 else float(pearsonr(left, right).statistic))
    result = {'jsd_mean': float(jsd.mean()), 'jsd_median': float(np.median(jsd)), 'pearson_mean': float(np.nanmean(correlations)), 'pearson_median': float(np.nanmedian(correlations)), 'summed_rmse': float(np.sqrt(np.mean((truth - prediction) ** 2)))}
    if not all((math.isfinite(value) for value in result.values())):
        raise RuntimeError('independent deconvolution scoring produced a non-finite metric')
    return result

def expected_pair_ids(config: dict) -> set[str]:
    return {f'{slice_id}__resolution_{float(resolution):g}' for slice_id in config['slices'] for resolution in config['resolutions']}

def score_keys(frame: pd.DataFrame) -> set[tuple[str, str, float]]:
    return set(zip(frame['method'].astype(str), frame['slice_id'].astype(str), frame['resolution'].astype(float)))

def read_score_layer(path: Path, label: str, expected_keys: set[tuple[str, str, float]]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={'slice_id': str})
    required = set(KEY_COLUMNS) | set(METRICS) | {'status'}
    if not required.issubset(frame.columns):
        raise RuntimeError(f'{label} lacks required columns')
    if len(frame) != 12 or frame.duplicated(list(KEY_COLUMNS)).any() or score_keys(frame) != expected_keys or (set(frame['method'].astype(str)) != {'rctd', 'cell2location'}) or (set(frame['status'].astype(str)) != {'ok'}) or (not np.isfinite(frame.loc[:, METRICS].apply(pd.to_numeric, errors='raise').to_numpy()).all()):
        raise RuntimeError(f'{label} is not the finite exact successful 12-key matrix')
    return frame

def read_historical_winners(path: Path) -> dict[str, str]:
    audit = pd.read_csv(path)
    historical = audit.loc[audit['layer'].astype(str) == 'historical'].copy()
    if len(historical) != len(METRICS) or historical['metric'].duplicated().any() or set(historical['metric'].astype(str)) != set(METRICS):
        raise RuntimeError('historical direction audit lacks the exact metric vector')
    historical = historical.set_index('metric')
    if historical['preference'].astype(str).to_dict() != EXPECTED_PREFERENCES:
        raise RuntimeError('historical direction audit has unexpected preferences')
    winners = historical['metric_winner'].astype(str).to_dict()
    if not set(winners.values()).issubset({'rctd', 'cell2location'}):
        raise RuntimeError('historical direction audit has invalid winners')
    return {metric: winners[metric] for metric in METRICS}

def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    return scores.groupby('method', as_index=False).agg(n_pairs=('pair_id', 'size'), jsd_mean=('jsd_mean', 'mean'), pearson_mean=('pearson_mean', 'mean'), summed_rmse=('summed_rmse', 'mean'))

def direction_details(summary: pd.DataFrame) -> tuple[dict[str, float], dict[str, str]]:
    indexed = summary.set_index('method')
    differences = {f'{metric}_rctd_minus_cell2location': float(indexed.loc['rctd', metric] - indexed.loc['cell2location', metric]) for metric in METRICS}
    winners: dict[str, str] = {}
    for metric in METRICS:
        difference = differences[f'{metric}_rctd_minus_cell2location']
        if abs(difference) <= DIRECTION_TOLERANCE:
            winners[metric] = 'tie_or_indeterminate'
        elif EXPECTED_PREFERENCES[metric] == 'lower':
            winners[metric] = 'rctd' if difference < 0 else 'cell2location'
        else:
            winners[metric] = 'rctd' if difference > 0 else 'cell2location'
    return (differences, winners)

def build_old_vs_new(historical: pd.DataFrame, prior: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    ordered_keys = list(KEY_COLUMNS)
    comparison: pd.DataFrame | None = None
    for layer, frame in (('historical', historical), ('prior_repaired', prior), ('fresh', fresh)):
        selected = frame.loc[:, ordered_keys + list(METRICS)].rename(columns={metric: f'{metric}_{layer}' for metric in METRICS})
        comparison = selected if comparison is None else comparison.merge(selected, on=ordered_keys, how='inner', validate='one_to_one')
    assert comparison is not None
    if len(comparison) != 12:
        raise RuntimeError('old-versus-new comparison is not the exact 12-key design')
    for metric in METRICS:
        historical_values = comparison[f'{metric}_historical']
        prior_values = comparison[f'{metric}_prior_repaired']
        fresh_values = comparison[f'{metric}_fresh']
        comparison[f'{metric}_historical_to_prior_delta'] = prior_values - historical_values
        comparison[f'{metric}_prior_to_fresh_delta'] = fresh_values - prior_values
        comparison[f'{metric}_historical_to_fresh_delta'] = fresh_values - historical_values
        comparison[f'{metric}_historical_to_prior_changed'] = comparison[f'{metric}_historical_to_prior_delta'].abs() > CHANGE_TOLERANCE
        comparison[f'{metric}_prior_to_fresh_changed'] = comparison[f'{metric}_prior_to_fresh_delta'].abs() > CHANGE_TOLERANCE
        comparison[f'{metric}_historical_to_fresh_changed'] = comparison[f'{metric}_historical_to_fresh_delta'].abs() > CHANGE_TOLERANCE
    for transition, classification in CHANGE_CLASSIFICATIONS.items():
        comparison[f'{transition}_classification'] = classification
    return comparison.sort_values(ordered_keys).reset_index(drop=True)

def assert_frame_matches(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    try:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
    except AssertionError as error:
        raise RuntimeError(f'{label} does not independently reproduce: {error}') from error

def valid_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith('Z'):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return False
    return parsed.utcoffset() == timezone.utc.utcoffset(parsed)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--scores-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=STUDY_ROOT / 'config.yaml')
    parser.add_argument('--historical-scores', type=Path, required=True)
    parser.add_argument('--prior-scores', type=Path, required=True)
    parser.add_argument('--direction-audit', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    simulation_manifest_path = args.run_dir / 'simulation_manifest.csv'
    method_manifest_path = args.run_dir / 'method_manifest.csv'
    simulations = pd.read_csv(simulation_manifest_path, dtype={'slice_id': str})
    methods = pd.read_csv(method_manifest_path, dtype={'slice_id': str})
    config = yaml.safe_load(args.config.read_text())
    expected_pairs = expected_pair_ids(config)
    expected_methods = {(pair_id, method) for pair_id in expected_pairs for method in ('rctd', 'cell2location')}
    if len(simulations) != 6 or set(simulations['pair_id'].astype(str)) != expected_pairs or simulations['pair_id'].duplicated().any() or (set(simulations['status'].astype(str)) != {'ok'}) or (len(methods) != 12) or (set(zip(methods['pair_id'].astype(str), methods['method'].astype(str))) != expected_methods) or methods[['pair_id', 'method']].duplicated().any() or (not all((str(row.pair_id) == f'{str(row.slice_id)}__resolution_{float(row.resolution):g}' for row in simulations.itertuples(index=False)))) or (not all((str(row.pair_id) == f'{str(row.slice_id)}__resolution_{float(row.resolution):g}' for row in methods.itertuples(index=False)))):
        raise RuntimeError('Study 03 manifests do not contain the exact declared job matrix')
    rows: list[dict] = []
    simulation_lookup = {str(row.pair_id): row for row in simulations.itertuples(index=False)}
    method_lookup = {(str(row.pair_id), str(row.method)): row for row in methods.itertuples(index=False)}
    adata_lookup: dict[str, ad.AnnData] = {}
    truth_lookup: dict[str, pd.DataFrame] = {}
    reference_lookup: dict[str, ad.AnnData] = {}
    library_lookup: dict[str, np.ndarray] = {}
    prediction_lookup: dict[tuple[str, str], pd.DataFrame] = {}
    for simulation in simulations.itertuples(index=False):
        pair_id = str(simulation.pair_id)
        path = Path(simulation.simulation_path)
        truth_path = Path(simulation.truth_path)
        reference_path = Path(simulation.reference_path)
        adata = ad.read_h5ad(path)
        reference = ad.read_h5ad(reference_path)
        truth = pd.read_csv(truth_path, index_col=0)
        truth.index = truth.index.astype(str)
        truth.columns = truth.columns.astype(str)
        truth_names = pd.Index(adata.uns['cell_type_names']).astype(str)
        embedded = adata.uns.get('feast_reproduce', {})
        libraries = integer_libraries(adata.X, f'simulation {pair_id}')
        integer_libraries(reference.X, f'reference {simulation.slice_id}')
        truth_values = truth.to_numpy(np.float64)
        if not np.isfinite(truth_values).all() or truth_values.min(initial=0) < -1e-12:
            raise RuntimeError(f'truth contains invalid proportions: {pair_id}')
        truth_sums = truth_values.sum(axis=1)
        valid = bool(adata.obs_names.equals(pd.Index([str(i) for i in range(adata.n_obs)])) and adata.var_names.equals(reference.var_names) and adata.var_names.is_unique and reference.var_names.is_unique and (simulation.feast_commit == config['required_feast_commit']) and (int(simulation.public_seed) == int(config['seed'])) and (embedded.get('feast_commit') == simulation.feast_commit) and (int(embedded.get('public_seed', -1)) == int(config['seed'])) and ('n_source_cells' in adata.obs) and (int(adata.obs['n_source_cells'].sum()) == int(embedded['solver_diagnostics']['aggregated_source_cell_count'])) and truth.index.equals(pd.Index([f'spot_{i}' for i in range(adata.n_obs)])) and truth.columns.equals(truth_names) and (truth.shape == np.asarray(adata.obsm['cell_type_proportions']).shape) and np.allclose(truth_values, adata.obsm['cell_type_proportions'], atol=1e-12, rtol=0) and np.allclose(truth_sums[libraries > 0], 1, atol=1e-10, rtol=0) and np.allclose(truth_sums[libraries <= 0], 0, atol=1e-12, rtol=0))
        rows.append({'kind': 'simulation', 'job_id': pair_id, 'valid': valid})
        adata_lookup[pair_id] = adata
        truth_lookup[pair_id] = truth
        reference_lookup[pair_id] = reference
        library_lookup[pair_id] = libraries
    for method in methods.itertuples(index=False):
        pair_id, method_name = (str(method.pair_id), str(method.method))
        output, metadata_path = (Path(method.output_path), Path(method.metadata_path))
        metadata = json.loads(metadata_path.read_text())
        prediction = pd.read_csv(output, index_col=0)
        prediction.index = prediction.index.astype(str)
        prediction.columns = prediction.columns.astype(str)
        simulation = adata_lookup[pair_id]
        simulation_row = simulation_lookup[pair_id]
        libraries = library_lookup[pair_id]
        expected_spots = pd.Index([f'spot_{i}' for i in range(simulation.n_obs)] if method_name == 'cell2location' else [f'spot_{i}' for i in np.flatnonzero(libraries > 0)])
        valid = bool(metadata.get('status') == 'validated_success' and metadata.get('method') == method_name and int(metadata.get('public_seed', -1)) == int(config['seed']))
        if method_name == 'cell2location':
            valid = bool(valid and metadata.get('actual_accelerator') == 'cuda' and (metadata.get('cuda_execution_verified') is True) and str(metadata.get('actual_device', '')).startswith('cuda:') and (int(metadata.get('peak_gpu_memory_allocated_bytes', 0)) > 0) and (metadata.get('abundance_named_columns_validated_and_ordered') is True) and (metadata.get('abundance_column_schema') in {'cell2location_prefixed_exact_order', 'factor_names_exact_order'}) and metadata.get('n_cells_per_location_policy', '').startswith('mean exact high-resolution source-cell assignments') and ('unsupported' in metadata.get('external_environment_limitation', '')))
        else:
            valid = bool(valid and metadata.get('rng_policy') == 'R set.seed(public_seed) before createRctd/runRctd' and set(('R', 'spacexr', 'SummarizedExperiment', 'SpatialExperiment')).issubset(metadata.get('environment', {})) and (metadata.get('solver_diagnostics', {}).get('weights_all_finite') is True))
        rows.append({'kind': 'method', 'job_id': f'{pair_id}__{method_name}', 'valid': valid})
        prediction_lookup[pair_id, method_name] = prediction
    expected_support_rows: list[dict] = []
    expected_score_rows: list[dict] = []
    expected_input_lineage: list[dict] = []
    for simulation in simulations.itertuples(index=False):
        pair_id = str(simulation.pair_id)
        adata = adata_lookup[pair_id]
        reference = reference_lookup[pair_id]
        truth = truth_lookup[pair_id]
        rctd = prediction_lookup[pair_id, 'rctd']
        cell2location = prediction_lookup[pair_id, 'cell2location']
        libraries = library_lookup[pair_id]
        full_spots = pd.Index([f'spot_{i}' for i in range(adata.n_obs)])
        positive_spots = full_spots[libraries > 0]
        zero_spots = full_spots[libraries <= 0]
        truth_types = set(truth.columns)
        reference_types = set(reference.obs['cell_type'].astype(str))
        cell2location_types = set(cell2location.columns)
        common = truth.columns[truth.columns.isin(rctd.columns)]
        support_valid = bool(rctd.index.equals(positive_spots) and cell2location.index.equals(full_spots) and (truth_types == reference_types) and (truth_types == cell2location_types) and (set(rctd.columns) == set(common)) and (len(common) > 0))
        if not support_valid:
            raise RuntimeError(f'independent support reconstruction failed: {pair_id}')
        if rctd.to_numpy(np.float64).min(initial=0) < -1e-06:
            raise RuntimeError(f'RCTD contains materially negative weights: {pair_id}')
        if cell2location.to_numpy(np.float64).min(initial=0) < 0:
            raise RuntimeError(f'Cell2location contains negative weights: {pair_id}')
        if np.max(np.abs(rctd.sum(axis=1).to_numpy(np.float64) - 1), initial=0) > 1e-05:
            raise RuntimeError(f'RCTD rows do not sum to one: {pair_id}')
        if np.max(np.abs(cell2location.loc[positive_spots].sum(axis=1).to_numpy(np.float64) - 1), initial=0) > 1e-06:
            raise RuntimeError(f'Cell2location scored rows do not sum to one: {pair_id}')
        zero_sums = cell2location.loc[zero_spots].sum(axis=1).to_numpy(np.float64)
        if not np.all(np.isclose(zero_sums, 0, atol=1e-06) | np.isclose(zero_sums, 1, atol=1e-06)):
            raise RuntimeError(f'Cell2location zero-library mass is invalid: {pair_id}')
        rows.append({'kind': 'support', 'job_id': pair_id, 'valid': support_valid})
        truth_scored = with_other(truth.loc[positive_spots], common)
        for method_name, prediction in (('rctd', rctd), ('cell2location', cell2location.loc[positive_spots])):
            expected_score_rows.append({'method': method_name, 'pair_id': pair_id, 'slice_id': str(simulation.slice_id), 'resolution': float(simulation.resolution), 'status': 'ok', 'comparison_scope': 'same_expression_exact_spots_common_named_types_plus_other', 'n_input_spots': len(full_spots), 'n_scored_spots': len(positive_spots), 'n_zero_library_spots_excluded': len(zero_spots), 'n_common_named_cell_types': len(common), 'n_scored_cell_types_including_other': len(common) + 1, **recompute_metrics(truth_scored, with_other(prediction, common))})
        expected_support_rows.append({'pair_id': pair_id, 'slice_id': str(simulation.slice_id), 'resolution': float(simulation.resolution), 'exact_positive_spot_support_between_methods': True, 'cell2location_full_spot_support_exact': True, 'n_full_spots': len(full_spots), 'n_scored_spots': len(positive_spots), 'n_zero_library_spots': len(zero_spots), 'zero_library_disposition': 'excluded_from_both_methods_scoring', 'n_truth_cell_types': len(truth.columns), 'n_reference_cell_types': len(reference_types), 'n_rctd_cell_types': len(rctd.columns), 'n_cell2location_cell_types': len(cell2location.columns), 'n_common_named_types': len(common), 'truth_type_set_equals_reference_type_set': truth_types == reference_types, 'truth_type_set_equals_cell2location_type_set': truth_types == cell2location_types, 'rctd_type_set_equals_truth_ordered_common_type_set': set(rctd.columns) == set(common), 'reference_common_type_set_equals_canonical_common_type_set': set(common).issubset(reference_types), 'cell2location_common_type_set_equals_canonical_common_type_set': set(common).issubset(cell2location_types), 'canonical_common_order_source': 'truth_columns', 'truth_cell_type_order_json': json.dumps(list(truth.columns)), 'canonical_common_cell_type_order_json': json.dumps(list(common)), 'rctd_raw_column_order_matches_canonical_common_order': rctd.columns.equals(common), 'rctd_raw_column_order_required': False, 'other_residual_included': True})
        method_lineage = {}
        for method_name in ('rctd', 'cell2location'):
            method = method_lookup[pair_id, method_name]
            method_lineage[method_name] = {'prediction': artifact(Path(method.output_path), f'{pair_id}:{method_name}:prediction'), 'metadata': artifact(Path(method.metadata_path), f'{pair_id}:{method_name}:metadata'), 'configuration_id': str(method.configuration_id)}
        expected_input_lineage.append({'pair_id': pair_id, 'slice_id': str(simulation.slice_id), 'resolution': float(simulation.resolution), 'simulation': artifact(Path(simulation.simulation_path), f'{pair_id}:simulation'), 'truth': artifact(Path(simulation.truth_path), f'{pair_id}:truth'), 'reference': artifact(Path(simulation.reference_path), f'{pair_id}:reference'), 'methods': method_lineage})
    scores_path = args.scores_dir / 'deconvolution_scores.csv'
    summary_path = args.scores_dir / 'deconvolution_summary.csv'
    support_path = args.scores_dir / 'support_audit.csv'
    old_vs_new_path = args.scores_dir / 'old_vs_new_deconvolution_scores.csv'
    decision_path = args.scores_dir / 'publication_decision.json'
    provenance_path = args.scores_dir / 'provenance.json'
    scores = pd.read_csv(scores_path, dtype={'slice_id': str})
    summary = pd.read_csv(summary_path)
    support = pd.read_csv(support_path, dtype={'slice_id': str})
    old_vs_new = pd.read_csv(old_vs_new_path, dtype={'slice_id': str})
    decision = json.loads(decision_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    expected_score_matrix = {(pair_id, method) for pair_id, method in expected_methods}
    observed_score_matrix = set(zip(scores['pair_id'].astype(str), scores['method'].astype(str)))
    score_matrix_valid = bool(len(scores) == 12 and (not scores.duplicated(['pair_id', 'method']).any()) and (observed_score_matrix == expected_score_matrix) and (set(scores['status'].astype(str)) == {'ok'}) and set(SCORE_METRICS).issubset(scores.columns) and np.isfinite(scores.loc[:, SCORE_METRICS].to_numpy(np.float64)).all())
    rows.append({'kind': 'score_matrix', 'job_id': 'deconvolution_scores', 'valid': score_matrix_valid})
    expected_scores = pd.DataFrame(expected_score_rows).sort_values(['method', 'pair_id']).reset_index(drop=True)
    if list(scores.columns) != list(expected_scores.columns):
        raise RuntimeError('deconvolution score schema differs from the scorer contract')
    actual_scores = scores.sort_values(['method', 'pair_id']).reset_index(drop=True)
    assert_frame_matches(actual_scores, expected_scores, 'deconvolution score table')
    expected_support = pd.DataFrame(expected_support_rows).sort_values('pair_id').reset_index(drop=True)
    actual_support = support.sort_values('pair_id').reset_index(drop=True)
    assert_frame_matches(actual_support, expected_support, 'support audit')
    expected_summary = summarize(scores)
    assert_frame_matches(summary, expected_summary, 'deconvolution summary')
    differences, metric_winners = direction_details(expected_summary)
    historical_metric_winners = read_historical_winners(args.direction_audit)
    changed_metric_directions = [metric for metric in METRICS if metric_winners[metric] != historical_metric_winners[metric]]
    direction_changed = bool(changed_metric_directions)
    expected_status = 'stop_headline_direction_changed' if direction_changed else 'candidate_direction_unchanged_pending_author_review'
    decision_valid = bool(decision.get('status') == expected_status and decision.get('historical_direction_audit_path') == str(args.direction_audit.resolve()) and (decision.get('historical_metric_winners') == historical_metric_winners) and (decision.get('metric_winners') == metric_winners) and (decision.get('changed_metric_directions') == changed_metric_directions) and (decision.get('direction_changed') is direction_changed) and (float(decision.get('direction_tolerance', -1)) == DIRECTION_TOLERANCE) and (decision.get('publication_claim_authorized') is False) and (decision.get('aggregate_winner_ranking_authorized') is False) and (set(decision.get('metric_differences', {})) == set(differences)) and all((np.isclose(float(decision['metric_differences'][key]), value, atol=1e-12, rtol=1e-12) for key, value in differences.items())))
    rows.append({'kind': 'direction', 'job_id': 'publication_decision', 'valid': decision_valid})
    expected_keys = score_keys(scores)
    historical = read_score_layer(args.historical_scores, 'article historical scores', expected_keys)
    prior = read_score_layer(args.prior_scores, 'prior repaired scores', expected_keys)
    expected_old_vs_new = build_old_vs_new(historical, prior, scores)
    assert_frame_matches(old_vs_new, expected_old_vs_new, 'old-versus-new table')
    output_paths = (scores_path, summary_path, support_path, old_vs_new_path, decision_path)
    expected_run_lineage = {'simulation_manifest': artifact(simulation_manifest_path, 'study03:simulation-manifest'), 'method_manifest': artifact(method_manifest_path, 'study03:method-manifest'), 'simulation_configuration_ids': sorted(set(simulations['configuration_id'].astype(str))), 'method_configuration_ids': sorted(set(methods['configuration_id'].astype(str)))}
    expected_historical_inputs = {'article_historical_scores': artifact(args.historical_scores, 'study03:article-historical-scores'), 'prior_repaired_scores': artifact(args.prior_scores, 'study03:prior-repaired-scores'), 'historical_direction_audit': artifact(args.direction_audit, 'study03:historical-direction-audit')}
    expected_config_artifact = artifact(args.config, 'study03:config')
    scorer_path = STUDY_ROOT / 'score.py'
    expected_scorer_artifact = artifact(scorer_path, 'study03:common-support-scorer')
    provenance_valid = bool(provenance.get('schema_version') == 3 and provenance.get('configuration_id') == 'study03-common-support-score-v3' and valid_utc_timestamp(provenance.get('generated_at_utc')) and (int(provenance.get('public_seed', -1)) == int(config['seed'])) and (provenance.get('config') == expected_config_artifact) and (provenance.get('scorer') == expected_scorer_artifact) and (provenance.get('feast_versions') == sorted(set(simulations['feast_version'].astype(str)))) and (provenance.get('feast_commits') == sorted(set(simulations['feast_commit'].astype(str)))) and (provenance.get('run_lineage') == expected_run_lineage) and (provenance.get('historical_inputs') == expected_historical_inputs) and (provenance.get('scoring_input_lineage') == expected_input_lineage) and (provenance.get('numerical_change_policy') == {'delta_definition': 'target_minus_source', 'change_tolerance': CHANGE_TOLERANCE, 'classifications': CHANGE_CLASSIFICATIONS, 'interpretation': 'All cross-layer deltas mix workflow changes and are not attributed to a single scientific or engineering cause.'}) and (set(provenance.get('outputs', {})) == {path.name for path in output_paths}) and all((path.is_file() for path in output_paths)))
    rows.append({'kind': 'provenance', 'job_id': 'score_lineage', 'valid': provenance_valid})
    result = pd.DataFrame(rows)
    if not result['valid'].all():
        failures = result.loc[~result['valid'], ['kind', 'job_id']].to_dict('records')
        raise RuntimeError(f'Study 03 validation failed: {failures}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f'Study 03 validated; publication decision: {expected_status}')
    return 2 if direction_changed else 0
if __name__ == '__main__':
    raise SystemExit(main())
