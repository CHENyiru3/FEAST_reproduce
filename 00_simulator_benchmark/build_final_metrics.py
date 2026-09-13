"""Integrate the identity-preserving scCube Slide-seq rerun into a fresh table."""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import yaml
STUDY_ROOT = Path(__file__).resolve().parent
REPRO_ROOT = STUDY_ROOT.parent
BASE_ROOT = STUDY_ROOT / 'outputs' / 'final_rerun_20260718_metrics_final'
RERUN_ROOT = STUDY_ROOT / 'outputs' / 'sccube_slideseq_rerun_20260719'
OUTPUT_ROOT = STUDY_ROOT / 'outputs' / 'final_metrics'
REFERENCE = STUDY_ROOT / 'data' / 'local' / 'reference' / 'Slideseq_001.h5ad'
CANDIDATE = RERUN_ROOT / 'Slideseq_001_sccube_identity.h5ad'
sys.path.insert(0, str(STUDY_ROOT))
from score import pair_metrics

def relative(path: Path) -> str:
    return path.resolve().relative_to(REPRO_ROOT).as_posix()

def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')

def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value

def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f'fresh-only output already exists: {OUTPUT_ROOT}')
    rerun_provenance_path = RERUN_ROOT / 'provenance.json'
    rerun_provenance = json.loads(rerun_provenance_path.read_text(encoding='utf-8'))
    reference = ad.read_h5ad(REFERENCE)
    candidate = ad.read_h5ad(CANDIDATE)
    if not reference.obs_names.equals(candidate.obs_names):
        raise ValueError('scCube rerun does not preserve exact spot order')
    if not reference.var_names.equals(candidate.var_names):
        raise ValueError('scCube rerun does not preserve exact gene order')
    if not np.array_equal(reference.obsm['spatial'], candidate.obsm['spatial']):
        raise ValueError('scCube rerun does not preserve exact spatial coordinates')
    config = yaml.safe_load((STUDY_ROOT / 'config.yaml').read_text(encoding='utf-8'))
    metrics, panel = pair_metrics(reference, candidate, max_genes=int(config['metrics']['max_moran_genes']), neighbors=int(config['metrics']['spatial_neighbors']), coordinate_decimals=int(config['metrics'].get('coordinate_pairing_decimals', 8)))
    base_path = BASE_ROOT / 'simulator_quality_metrics.csv'
    table = pd.read_csv(base_path)
    mask = (table['simulator'] == 'scCube') & (table['sample'] == 'Slideseq_001')
    if int(mask.sum()) != 1:
        raise ValueError('expected exactly one scCube/Slideseq_001 row')
    old_row = table.loc[mask].iloc[0].copy()
    for column in table.columns:
        if column in metrics:
            table.loc[mask, column] = metrics[column]
    new_row = table.loc[mask].iloc[0].copy()
    if int(new_row['n_paired_spots']) != reference.n_obs:
        raise ValueError('scCube rerun lacks complete named spot support')
    if str(new_row['spot_pairing']) != 'exact_identifier':
        raise ValueError('scCube rerun did not use exact identifier pairing')
    if not np.isfinite(float(new_row['cosine_divergence'])):
        raise ValueError('scCube rerun cosine divergence is not finite')
    if not np.isfinite(float(new_row['zero_mask_jaccard'])):
        raise ValueError('scCube rerun zero-mask Jaccard is not finite')
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    table_path = OUTPUT_ROOT / 'simulator_quality_metrics.csv'
    table.to_csv(table_path, index=False)
    shutil.copyfile(BASE_ROOT / 'moran_gene_panel.csv', OUTPUT_ROOT / 'moran_gene_panel.csv')
    changes = []
    for column in table.columns:
        before, after = (old_row[column], new_row[column])
        before_missing, after_missing = (pd.isna(before), pd.isna(after))
        if before_missing and after_missing:
            continue
        if before_missing != after_missing or before != after:
            changes.append({'field': column, 'before': None if before_missing else json_scalar(before), 'after': None if after_missing else json_scalar(after)})
    comparison_path = OUTPUT_ROOT / 'sccube_slideseq_old_vs_new.json'
    write_json(comparison_path, {'simulator': 'scCube', 'sample': 'Slideseq_001', 'changes': changes})
    support_path = OUTPUT_ROOT / 'support_audit.json'
    write_json(support_path, {'simulator': 'scCube', 'sample': 'Slideseq_001', 'reference_spots': int(reference.n_obs), 'candidate_spots': int(candidate.n_obs), 'exact_spot_order': True, 'exact_gene_order': True, 'exact_spatial_order': True, 'paired_spots': int(metrics['n_paired_spots']), 'spot_pairing': metrics['spot_pairing'], 'identity_flag': bool(metrics['identity_flag'])})
    decision_path = OUTPUT_ROOT / 'publication_decision.json'
    write_json(decision_path, {'status': 'validated_external_rerun_author_review_required', 'publication_claim_authorized': False, 'figure_promotion_authorized': False, 'winner_or_composite_authorized': False, 'feast_aggregate_metrics_worsened': ['gene_variance_wasserstein'], 'scCube_slideseq_pairwise_metrics_available': True, 'source_lineage_complete': False, 'interpretation': 'the identity-preserving scCube Slide-seq rerun closes the missing named-support metrics; aggregate ranking remains unauthorized', 'external_environment_limitation': rerun_provenance['external_method_environment_note']})
    output_paths = {path.name: None for path in (table_path, OUTPUT_ROOT / 'moran_gene_panel.csv', comparison_path, support_path, decision_path)}
    provenance_path = OUTPUT_ROOT / 'provenance.json'
    write_json(provenance_path, {'schema_version': 1, 'configuration_id': 'study00-final-metrics-sccube-slideseq-identity-v1', 'calculation_randomness': 'none', 'base_table': {'path': relative(base_path)}, 'reference': {'artifact_id': rerun_provenance['input_artifact_id'], 'path': relative(REFERENCE)}, 'external_candidate': {'configuration_id': rerun_provenance['configuration_id'], 'method': 'scCube', 'version': rerun_provenance['method_version'], 'public_seed': rerun_provenance['seed'], 'path': relative(CANDIDATE), 'provenance_path': relative(rerun_provenance_path), 'solver_diagnostics': rerun_provenance['solver_diagnostics'], 'external_environment_supported_by_feast': False}, 'outputs': output_paths, 'composite_score': 'prohibited', 'method_ranking': 'prohibited_pending_author_review'})
    print(f'wrote final Study 00 metrics to {OUTPUT_ROOT}')
    print(f"cosine_divergence={float(new_row['cosine_divergence']):.12g}")
    print(f"zero_mask_jaccard={float(new_row['zero_mask_jaccard']):.12g}")
if __name__ == '__main__':
    main()
