"""Audit matrix-level behavior of Study 01 combined alterations."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
EXPECTED_DIRECTIONS = {'combined_mild_activation': ('mean_up', 'zeros_down'), 'combined_strong_activation': ('mean_up', 'zeros_down'), 'combined_mild_dropout': ('zeros_up',), 'combined_noisy_tissue': ('variance_up', 'zeros_up'), 'combined_suppressed': ('mean_down', 'zeros_up')}

def dense_matrix(adata: ad.AnnData) -> np.ndarray:
    return adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)

def matrix_stats(matrix: np.ndarray) -> dict[str, object]:
    finite = bool(np.isfinite(matrix).all())
    nonnegative = bool((matrix >= 0).all())
    integer_counts = bool(np.equal(matrix, np.floor(matrix)).all())
    gene_mean = np.asarray(matrix.mean(axis=0), dtype=float)
    gene_variance = np.asarray(matrix.var(axis=0), dtype=float)
    return {'finite': finite, 'nonnegative': nonnegative, 'integer_counts': integer_counts, 'global_mean': float(matrix.mean()), 'zero_fraction': float(np.count_nonzero(matrix == 0) / matrix.size), 'gene_mean': gene_mean, 'gene_variance': gene_variance}

def median_positive_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    keep = denominator > 0
    return float(np.median(numerator[keep] / denominator[keep]))

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulation-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    manifest_path = args.simulation_dir / 'simulation_manifest.csv'
    manifest = pd.read_csv(manifest_path, dtype={'slice_id': str, 'simulation_id': str})
    if len(manifest) != 81 or not manifest['status'].eq('ok').all():
        raise RuntimeError('expected 81 successful simulation rows')
    rows: list[dict[str, object]] = []
    for slice_id in sorted(manifest['slice_id'].unique()):
        slice_rows = manifest[manifest['slice_id'] == slice_id].set_index('simulation_id')
        baseline_row = slice_rows.loc['baseline']
        baseline_path = args.simulation_dir / baseline_row['file_path']
        baseline = ad.read_h5ad(baseline_path)
        baseline_obs = baseline.obs_names.copy()
        baseline_var = baseline.var_names.copy()
        baseline_stats = matrix_stats(dense_matrix(baseline))
        for simulation_id, directions in EXPECTED_DIRECTIONS.items():
            candidate_row = slice_rows.loc[simulation_id]
            candidate_path = args.simulation_dir / candidate_row['file_path']
            candidate = ad.read_h5ad(candidate_path)
            candidate_stats = matrix_stats(dense_matrix(candidate))
            mean_ratio = float(candidate_stats['global_mean'] / baseline_stats['global_mean'])
            zero_delta = float(candidate_stats['zero_fraction'] - baseline_stats['zero_fraction'])
            variance_ratio = median_positive_ratio(candidate_stats['gene_variance'], baseline_stats['gene_variance'])
            gene_mean_ratio = median_positive_ratio(candidate_stats['gene_mean'], baseline_stats['gene_mean'])
            checks = {'mean_up': mean_ratio > 1.0, 'mean_down': mean_ratio < 1.0, 'zeros_up': zero_delta > 0.0, 'zeros_down': zero_delta < 0.0, 'variance_up': variance_ratio > 1.0}
            structural_ok = all((candidate_stats['finite'], candidate_stats['nonnegative'], candidate_stats['integer_counts'], candidate.obs_names.equals(baseline_obs), candidate.var_names.equals(baseline_var)))
            direction_ok = all((checks[direction] for direction in directions))
            rows.append({'slice_id': slice_id, 'simulation_id': simulation_id, 'file_path': candidate_row['file_path'], 'n_spots': candidate.n_obs, 'n_genes': candidate.n_vars, 'finite': candidate_stats['finite'], 'nonnegative': candidate_stats['nonnegative'], 'integer_counts': candidate_stats['integer_counts'], 'exact_spot_order': candidate.obs_names.equals(baseline_obs), 'exact_gene_order': candidate.var_names.equals(baseline_var), 'global_mean_ratio_to_baseline': mean_ratio, 'zero_fraction_delta_to_baseline': zero_delta, 'median_gene_mean_ratio_to_baseline': gene_mean_ratio, 'median_gene_variance_ratio_to_baseline': variance_ratio, 'expected_directions': ';'.join(directions), 'direction_checks_passed': direction_ok, 'status': 'ok' if structural_ok and direction_ok else 'failed'})
    table = pd.DataFrame(rows)
    table_path = args.output_dir / 'combined_output_audit.csv'
    table.to_csv(table_path, index=False)
    script_path = Path(__file__).resolve()
    provenance = {'configuration_id': str(manifest['configuration_id'].iloc[0]), 'generated_utc': datetime.now(timezone.utc).isoformat(), 'public_seed': int(manifest['public_seed'].iloc[0]), 'feast_version': str(manifest['feast_version'].iloc[0]), 'feast_commit': str(manifest['feast_commit'].iloc[0]), 'audit_source': str(script_path), 'audited_rows': len(table), 'successful_rows': int(table['status'].eq('ok').sum()), 'decision': 'matrix_level_combined_directions_verified' if table['status'].eq('ok').all() else 'blocked_by_matrix_level_combined_direction_failure', 'diagnostic_disposition': 'Heavy-tailed fitted-distribution sample moments printed during fitting are not used as achieved count-matrix effects; publication interpretation uses the finite saved count matrices audited here.'}
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f"Combined-output audit: {provenance['successful_rows']}/{len(table)} passed")
    return 0 if table['status'].eq('ok').all() else 1
if __name__ == '__main__':
    raise SystemExit(main())
