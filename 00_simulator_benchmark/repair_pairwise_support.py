"""Repair missing Study 00 pairwise metrics using verified coordinate bijections."""
from __future__ import annotations
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import yaml
from score import cosine_divergence, paired_spot_indices, zero_jaccard
ROOT = Path(__file__).resolve().parent
DEFAULT_OLD_ROOT = ROOT / 'outputs' / 'final_rerun_20260718_metrics_v2'
DEFAULT_OUTPUT_ROOT = ROOT / 'outputs' / 'final_rerun_20260718_metrics_final'
PRIOR_COMPARISON_ROOT = ROOT / 'outputs' / 'final_rerun_20260718_comparison'
PAIRWISE_METRICS = ('cosine_divergence', 'zero_mask_jaccard')
MUTABLE_COLUMNS = {'identity_flag', 'n_paired_spots', 'spot_pairing', *PAIRWISE_METRICS}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data' / 'local')
    parser.add_argument('--old-root', type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()

def validate_old_root(old_root: Path) -> tuple[Path, Path, Path]:
    metrics_path = old_root / 'simulator_quality_metrics.csv'
    panel_path = old_root / 'moran_gene_panel.csv'
    provenance_path = old_root / 'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    for path in (metrics_path, panel_path, provenance_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return (metrics_path, panel_path, provenance_path)

def aggregate_comparison(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    directions = {'cosine_divergence': 'lower', 'zero_mask_jaccard': 'higher'}
    for metric, direction in directions.items():
        old_wide = old.pivot(index='sample', columns='simulator', values=metric)
        new_wide = new.pivot(index='sample', columns='simulator', values=metric)
        old_common = old_wide.dropna(axis=0, how='any')
        new_common = new_wide.dropna(axis=0, how='any')
        old_means = old_common.mean(axis=0)
        new_means = new_common.mean(axis=0)
        old_ranks = old_means.rank(method='min', ascending=direction == 'lower')
        new_ranks = new_means.rank(method='min', ascending=direction == 'lower')
        for simulator in sorted(old_wide.columns):
            records.append({'metric': metric, 'preferred_direction': direction, 'simulator': simulator, 'old_n': int(len(old_common)), 'new_n': int(len(new_common)), 'old_common_samples': ';'.join(old_common.index.astype(str)), 'new_common_samples': ';'.join(new_common.index.astype(str)), 'old_mean': old_means.get(simulator, np.nan), 'new_mean': new_means.get(simulator, np.nan), 'mean_delta': new_means.get(simulator, np.nan) - old_means.get(simulator, np.nan), 'old_rank': old_ranks.loc[simulator], 'new_rank': new_ranks.loc[simulator]})
    result = pd.DataFrame(records)
    result['prior_rank_comparable'] = result['old_n'].gt(0) & result['new_n'].gt(0) & result['old_common_samples'].eq(result['new_common_samples'])
    result['rank_change_assessment'] = np.where(~result['prior_rank_comparable'], 'prior_indeterminate', np.where(result['old_rank'].eq(result['new_rank']), 'unchanged', 'changed'))
    return result

def main() -> int:
    args = parse_args()
    metrics_path, panel_path, old_provenance_path = validate_old_root(args.old_root)
    manifest_path = ROOT / 'data' / 'input_checksums.csv'
    config_path = ROOT / 'pairwise_support_config.yaml'
    config = yaml.safe_load(config_path.read_text())
    coordinate_decimals = int(config['coordinate_pairing_decimals'])
    manifest = pd.read_csv(manifest_path)
    old = pd.read_csv(metrics_path)
    if len(old) != 60 or old[['simulator', 'sample']].duplicated().any():
        raise RuntimeError('expected the frozen unique 60-row Study 00 table')
    repaired = old.copy()
    affected = old[old['spot_pairing'] == 'unavailable']
    if len(affected) != 21:
        raise RuntimeError(f'expected 21 unavailable rows, found {len(affected)}')
    audit_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for sample, sample_rows in affected.groupby('sample', sort=False):
        ref_record = manifest[(manifest['role'] == 'reference') & (manifest['sample'] == sample)].iloc[0]
        ref_path = args.data_root / ref_record.relative_path
        reference = ad.read_h5ad(ref_path)
        for old_row in sample_rows.itertuples(index=False):
            sim_record = manifest[(manifest['role'] == 'external') & (manifest['simulator'] == old_row.simulator) & (manifest['sample'] == sample)].iloc[0]
            sim_path = args.data_root / sim_record.relative_path
            simulated = ad.read_h5ad(sim_path)
            reference_indices, simulated_indices, diagnostics = paired_spot_indices(reference, simulated, coordinate_decimals)
            key = (repaired['simulator'] == old_row.simulator) & (repaired['sample'] == sample)
            common_genes = reference.var_names[reference.var_names.isin(simulated.var_names)]
            if common_genes.empty:
                raise RuntimeError(f'no common genes: {old_row.simulator}/{sample}')
            new_cosine = np.nan
            new_jaccard = np.nan
            if len(reference_indices):
                reference_gene_indices = reference.var_names.get_indexer(common_genes)
                simulated_gene_indices = simulated.var_names.get_indexer(common_genes)
                paired_reference = reference.X[reference_indices, :][:, reference_gene_indices]
                paired_simulated = simulated.X[simulated_indices, :][:, simulated_gene_indices]
                new_cosine = cosine_divergence(paired_reference, paired_simulated)
                new_jaccard = zero_jaccard(paired_reference, paired_simulated)
                repaired.loc[key, 'cosine_divergence'] = new_cosine
                repaired.loc[key, 'zero_mask_jaccard'] = new_jaccard
                repaired.loc[key, 'n_paired_spots'] = len(reference_indices)
                repaired.loc[key, 'spot_pairing'] = diagnostics['spot_pairing']
                repaired.loc[key, 'identity_flag'] = bool(old_row.input_mean_corr >= 0.995 and old_row.input_variance_corr >= 0.95 and (new_jaccard >= 0.95))
            audit_rows.append({'simulator': old_row.simulator, 'sample': sample, 'reference_artifact_id': f'study00:reference:{sample}', 'simulation_artifact_id': f'study00:external:{old_row.simulator}:{sample}', **diagnostics, 'n_paired_spots': int(len(reference_indices)), 'repair_status': 'repaired' if len(reference_indices) else 'unresolved'})
            comparison_rows.append({'simulator': old_row.simulator, 'sample': sample, 'change_class': 'coordinate_bijection_pairwise_metric_repair' if len(reference_indices) else 'unresolved_nonbijective_spatial_support', 'spot_pairing_old': old_row.spot_pairing, 'spot_pairing_new': diagnostics['spot_pairing'], 'n_paired_spots_old': int(old_row.n_paired_spots), 'n_paired_spots_new': int(len(reference_indices)), 'cosine_divergence_old': old_row.cosine_divergence, 'cosine_divergence_new': new_cosine, 'zero_mask_jaccard_old': old_row.zero_mask_jaccard, 'zero_mask_jaccard_new': new_jaccard, 'identity_flag_old': bool(old_row.identity_flag), 'identity_flag_new': bool(repaired.loc[key, 'identity_flag'].iloc[0])})
            del simulated
        del reference
    immutable = [column for column in old.columns if column not in MUTABLE_COLUMNS]
    pd.testing.assert_frame_equal(old[immutable], repaired[immutable], check_dtype=False)
    support_audit = pd.DataFrame(audit_rows)
    pairwise_changes = pd.DataFrame(comparison_rows)
    keys = ['simulator', 'sample']
    v2_vs_final = old.merge(repaired, on=keys, how='inner', validate='one_to_one', suffixes=('_v2', '_final'))
    affected_keys = set(map(tuple, support_audit[keys].to_numpy()))
    v2_vs_final.insert(2, 'change_class', ['coordinate_pairwise_support_repair' if tuple(row) in affected_keys else 'unchanged' for row in v2_vs_final[keys].to_numpy()])
    for metric in PAIRWISE_METRICS:
        v2_vs_final[f'{metric}_delta'] = v2_vs_final[f'{metric}_final'] - v2_vs_final[f'{metric}_v2']
        v2_vs_final[f'{metric}_availability_change'] = np.select([v2_vs_final[f'{metric}_v2'].isna() & v2_vs_final[f'{metric}_final'].notna(), v2_vs_final[f'{metric}_v2'].notna() & v2_vs_final[f'{metric}_final'].isna()], ['newly_available', 'became_unavailable'], default='unchanged')
    aggregates = aggregate_comparison(old, repaired)
    unresolved = support_audit[support_audit['repair_status'] == 'unresolved']
    prior_rank_comparable = bool(aggregates['prior_rank_comparable'].all())
    ranking_changed = bool((aggregates['rank_change_assessment'] == 'changed').any())
    complete_case_samples = {metric: samples.split(';') if samples else [] for metric, samples in aggregates.groupby('metric', sort=False)['new_common_samples'].first().items()}
    identity_changes = pairwise_changes[pairwise_changes['identity_flag_old'] != pairwise_changes['identity_flag_new']][['simulator', 'sample', 'identity_flag_old', 'identity_flag_new']].to_dict('records')
    decision = {'status': 'stop_pairwise_support_expansion_author_review_required' if not prior_rank_comparable or ranking_changed else 'pairwise_support_repaired_with_explicit_exception', 'configuration_id': 'study00-pairwise-coordinate-support-repair-v1', 'publication_claim_authorized': False, 'frozen_artifacts_modified': False, 'affected_rows': int(len(affected)), 'repaired_rows': int((support_audit['repair_status'] == 'repaired').sum()), 'unresolved_rows': int(len(unresolved)), 'unresolved': unresolved[['simulator', 'sample', 'pairing_issue', 'n_coordinate_overlap']].to_dict('records'), 'prior_pairwise_rank_comparable': prior_rank_comparable, 'aggregate_pairwise_ranking_changed': ranking_changed if prior_rank_comparable else None, 'pairwise_ranking_assessment': 'newly_assessable_on_complete_11_dataset_support; prior ranking indeterminate' if not prior_rank_comparable else 'changed' if ranking_changed else 'unchanged', 'pairwise_complete_case_samples': complete_case_samples, 'identity_flag_changes': identity_changes, 'disposition': 'Use the repaired table only after author review of the support and rank changes. scCube Slide-seq remains unscored for pairwise metrics.'}
    prior_decision_path = PRIOR_COMPARISON_ROOT / 'publication_decision.json'
    prior_decision = json.loads(prior_decision_path.read_text())
    publication_decision = {**prior_decision, 'status': 'stop_pairwise_support_expansion_author_review_required', 'headline_direction_unchanged': None, 'headline_direction_assessment': 'not asserted because the prior table had no all-five-method pairwise complete-case support', 'publication_claim_authorized': False, 'figure_promotion_authorized': False, 'aggregate_method_ranking_authorized': False, 'winner_or_composite_authorized': False, 'identity_flag_changes': prior_decision.get('identity_flag_changes', []) + identity_changes, 'pairwise_support_repair': {'configuration_id': decision['configuration_id'], 'affected_rows': decision['affected_rows'], 'repaired_rows': decision['repaired_rows'], 'unresolved_rows': decision['unresolved_rows'], 'unresolved': decision['unresolved'], 'complete_case_samples': complete_case_samples, 'prior_pairwise_rank_comparable': False, 'ranking_assessment': decision['pairwise_ranking_assessment']}, 'disposition': 'The original non-pairwise FEAST change classification is retained. Pairwise support expanded materially and requires author review before figure promotion or method-ranking claims. scCube Slide-seq remains NA.'}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    repaired_path = args.output_dir / 'simulator_quality_metrics.csv'
    panel_output = args.output_dir / 'moran_gene_panel.csv'
    audit_path = args.output_dir / 'pairing_support_audit.csv'
    changes_path = args.output_dir / 'old_vs_new_pairwise_metrics.csv'
    v2_comparison_path = args.output_dir / 'v2_vs_final_metrics.csv'
    aggregates_path = args.output_dir / 'aggregate_pairwise_comparison.csv'
    decision_path = args.output_dir / 'repair_decision.json'
    publication_decision_path = args.output_dir / 'publication_decision.json'
    repaired.to_csv(repaired_path, index=False)
    shutil.copy2(panel_path, panel_output)
    support_audit.to_csv(audit_path, index=False)
    pairwise_changes.to_csv(changes_path, index=False)
    v2_vs_final.to_csv(v2_comparison_path, index=False)
    aggregates.to_csv(aggregates_path, index=False)
    decision_path.write_text(json.dumps(decision, indent=2) + '\n')
    publication_decision_path.write_text(json.dumps(publication_decision, indent=2) + '\n')
    source = Path(__file__).resolve()
    score_source = ROOT / 'score.py'
    outputs = [repaired_path, panel_output, audit_path, changes_path, v2_comparison_path, aggregates_path, decision_path, publication_decision_path]
    provenance = {'schema_version': 1, 'configuration_id': 'study00-pairwise-coordinate-support-repair-v1', 'generated_utc': datetime.now(timezone.utc).isoformat(), 'calculation_randomness': 'none', 'frozen_artifacts_modified': False, 'pairing_contract': {'identifier_policy': 'prefer exact observation identifiers', 'fallback': 'complete unique spatial-coordinate bijection', 'coordinate_decimals': coordinate_decimals, 'partial_support': 'reject', 'duplicate_coordinates': 'reject'}, 'sources': {'old_metrics': {'path': str(metrics_path)}, 'old_moran_panel': {'path': str(panel_path)}, 'old_provenance': {'path': str(old_provenance_path)}, 'input_manifest': {'path': str(manifest_path)}, 'config': {'path': str(config_path)}, 'repair_script': {'path': str(source)}, 'score_source': {'path': str(score_source)}, 'prior_publication_decision': {'path': str(prior_decision_path)}}, 'input_artifacts': None, 'diagnostics': {'affected_rows': int(len(affected)), 'repaired_rows': int((support_audit['repair_status'] == 'repaired').sum()), 'unresolved_rows': int(len(unresolved)), 'unchanged_nonpairwise_columns_asserted': immutable}, 'outputs': {path.name: None for path in outputs}}
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(decision, indent=2))
    return 2 if not prior_rank_comparable or ranking_changed else 0
if __name__ == '__main__':
    raise SystemExit(main())
