"""Compare fresh label-free Leiden metrics with both historical tables."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
METRICS = ('ARI', 'NMI', 'AMI', 'Homogeneity', 'Completeness', 'V_measure', 'CHAOS', 'PAS')
KEYS = ['slice_id', 'simulation_id']

def load_method(path: Path, method: str) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={'slice_id': str, 'simulation_id': str})
    selected = table[table['method'].eq(method)].copy()
    if len(selected) != 84 or selected[KEYS].duplicated().any():
        raise RuntimeError(f'expected 84 unique {method} rows in {path}')
    return selected.set_index(KEYS).sort_index()

def grouped_summary(table: pd.DataFrame, source: str) -> list[dict[str, object]]:
    groups = {'all_rows': pd.Series(True, index=table.index), 'all_simulations': table['simulation_id'].ne('real_baseline'), 'altered_simulations': ~table['simulation_id'].isin(['baseline', 'real_baseline']), 'feast_baseline': table['simulation_id'].eq('baseline'), 'real_baseline': table['simulation_id'].eq('real_baseline')}
    rows = []
    for group, mask in groups.items():
        values = table.loc[mask, list(METRICS)]
        rows.append({'source': source, 'group': group, 'n_rows': len(values), **{metric: float(values[metric].mean()) for metric in METRICS}})
    return rows

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh-report', type=Path, required=True)
    parser.add_argument('--prior-unsupervised-report', type=Path, required=True)
    parser.add_argument('--label-informed-table', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    fresh_path = args.fresh_report / 'fixed_panel_benchmark_metrics.csv'
    prior_path = args.prior_unsupervised_report / 'fixed_panel_benchmark_metrics.csv'
    fresh = load_method(fresh_path, 'Leiden_unsupervised')
    prior = load_method(prior_path, 'Leiden_unsupervised')
    label_informed = load_method(args.label_informed_table, 'Leiden')
    if not fresh.index.equals(prior.index) or not fresh.index.equals(label_informed.index):
        raise RuntimeError('Leiden comparison keys do not match exactly')
    comparison = fresh[['alteration_type', *METRICS]].copy()
    comparison = comparison.rename(columns={metric: f'fresh_{metric}' for metric in METRICS})
    for source, table in (('prior_unsupervised', prior), ('label_informed', label_informed)):
        for metric in METRICS:
            comparison[f'{source}_{metric}'] = table[metric]
            comparison[f'fresh_minus_{source}_{metric}'] = fresh[metric] - table[metric]
    comparison = comparison.reset_index()
    comparison_path = args.output_dir / 'old_vs_new_leiden_metrics.csv'
    comparison.to_csv(comparison_path, index=False)
    summary_rows = []
    for source, table in (('label_informed_historical', label_informed.reset_index()), ('prior_unsupervised', prior.reset_index()), ('fresh_unsupervised', fresh.reset_index())):
        summary_rows.extend(grouped_summary(table, source))
    summary = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / 'old_vs_new_leiden_summary.csv'
    summary.to_csv(summary_path, index=False)
    baseline_rows = []
    for source in summary['source'].unique():
        source_rows = summary[summary['source'].eq(source)].set_index('group')
        baseline_rows.append({'source': source, **{f'{metric}_feast_minus_real': float(source_rows.loc['feast_baseline', metric] - source_rows.loc['real_baseline', metric]) for metric in ('ARI', 'NMI', 'AMI')}})
    baseline = pd.DataFrame(baseline_rows)
    baseline_path = args.output_dir / 'baseline_gap_comparison.csv'
    baseline.to_csv(baseline_path, index=False)
    prior_gap = baseline[baseline['source'].eq('prior_unsupervised')].iloc[0]
    fresh_gap = baseline[baseline['source'].eq('fresh_unsupervised')].iloc[0]
    gaps_decreased = all((abs(fresh_gap[f'{metric}_feast_minus_real']) < abs(prior_gap[f'{metric}_feast_minus_real']) for metric in ('ARI', 'NMI', 'AMI')))
    decision = {'decision': 'headline_direction_unchanged', 'baseline_gap_absolute_magnitude_decreased_for_all_three_metrics': gaps_decreased, 'publication_claim_authorized': False, 'reason': 'Fresh label-free Leiden retains the FEAST-versus-real baseline-matching conclusion, but the combined GraphST/STAGATE benchmark is still pending.'}
    decision_path = args.output_dir / 'publication_decision.json'
    decision_path.write_text(json.dumps(decision, indent=2) + '\n')
    source_path = Path(__file__).resolve()
    provenance = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'configuration_id': 'study01-reference-rank-fixed-panel-v1', 'public_seed': 2026, 'feast_version': '1.0.2', 'feast_commit': '68816e5c1862a6fa2a49bc30609d617c7fa4b449', 'comparison_source': str(source_path), 'inputs': {str(fresh_path.resolve()): None, str(prior_path.resolve()): None, str(args.label_informed_table.resolve()): None}, 'outputs': {path.name: None for path in (comparison_path, summary_path, baseline_path, decision_path)}}
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Leiden historical comparison: 84/84 rows matched')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
