"""Score a label-matched Cell2location corrective rerun against existing RCTD.

This diagnostic deliberately evaluates only a declared slice and resolutions.
It requires Cell2location's fitted factor set to equal RCTD's retained named
support exactly.  The ground truth retains all fine labels; its residual mass
outside that support is represented by ``__other__`` for both methods.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from score import integer_libraries, metrics, read_composition, with_other
STUDY_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = STUDY_ROOT / 'outputs' / 'final_rerun_20260718_v2'
DEFAULT_CANDIDATE_DIR = STUDY_ROOT / 'outputs' / 'cell2location_100_common61_20260806_v1'

def require_path(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f'missing {label}: {path}')

def artifact(path: Path) -> dict[str, str]:
    return {'path': str(path.resolve())}

def read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={'slice_id': str})
    if frame.empty:
        raise RuntimeError(f'empty manifest: {path}')
    return frame

def one_row(frame: pd.DataFrame, **criteria: object) -> pd.Series:
    selected = frame.copy()
    for key, value in criteria.items():
        if key == 'resolution':
            selected = selected[np.isclose(selected[key].astype(float), float(value), atol=0, rtol=0)]
        else:
            selected = selected[selected[key].astype(str) == str(value)]
    if len(selected) != 1:
        raise RuntimeError(f'expected one manifest record for {criteria}, found {len(selected)}')
    return selected.iloc[0]

def score_resolution(simulation_row: pd.Series, rctd_row: pd.Series, candidate_dir: Path) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    slice_id = str(simulation_row['slice_id'])
    resolution = float(simulation_row['resolution'])
    pair_id = str(simulation_row['pair_id'])
    simulation_path = Path(simulation_row['simulation_path'])
    truth_path = Path(simulation_row['truth_path'])
    reference_path = Path(simulation_row['reference_path'])
    for path, label in ((simulation_path, 'simulation'), (truth_path, 'truth'), (reference_path, 'reference'), (Path(rctd_row['output_path']), 'RCTD output')):
        require_path(path, f'{pair_id} {label}')
    candidate_path = candidate_dir / 'methods' / 'cell2location' / slice_id / f'resolution_{resolution:g}_proportions.csv'
    candidate_metadata_path = candidate_path.with_name(candidate_path.stem + '_metadata.json')
    if not candidate_path.is_file() or not candidate_metadata_path.is_file():
        raise FileNotFoundError(f'missing candidate Cell2location artifacts for {pair_id}')
    candidate_metadata = json.loads(candidate_metadata_path.read_text())
    if candidate_metadata.get('status') != 'validated_success':
        raise RuntimeError(f'candidate Cell2location did not validate: {pair_id}')
    simulation = ad.read_h5ad(simulation_path, backed='r')
    try:
        libraries = integer_libraries(simulation.X, f'simulation {pair_id}')
        full_spots = pd.Index([f'spot_{index}' for index in range(simulation.n_obs)])
        truth_names = pd.Index(simulation.uns['cell_type_names']).astype(str)
    finally:
        simulation.file.close()
    positive_spots = full_spots[libraries > 0]
    zero_spots = full_spots[libraries <= 0]
    truth = read_composition(truth_path, f'truth {pair_id}')
    rctd = read_composition(Path(rctd_row['output_path']), f'RCTD {pair_id}')
    candidate = read_composition(candidate_path, f'Cell2location candidate {pair_id}')
    if not truth.index.equals(full_spots) or not truth.columns.equals(truth_names):
        raise RuntimeError(f'truth index/column identity differs: {pair_id}')
    if not rctd.index.equals(positive_spots):
        raise RuntimeError(f'RCTD positive-spot support differs: {pair_id}')
    if not candidate.index.equals(full_spots):
        raise RuntimeError(f'candidate full-spot support differs: {pair_id}')
    common = truth.columns[truth.columns.isin(rctd.columns)]
    if common.empty or set(common) != set(rctd.columns) or set(candidate.columns) != set(rctd.columns):
        raise RuntimeError(f'methods do not share the exact retained named support: {pair_id}')
    candidate = candidate.loc[:, common]
    rctd = rctd.loc[:, common]
    if not np.allclose(rctd.sum(axis=1), 1.0, atol=1e-05):
        raise RuntimeError(f'RCTD rows are not normalized: {pair_id}')
    if not np.allclose(candidate.loc[positive_spots].sum(axis=1), 1.0, atol=1e-06):
        raise RuntimeError(f'candidate positive rows are not normalized: {pair_id}')
    if not np.allclose(candidate.loc[zero_spots].sum(axis=1), 0.0, atol=1e-12):
        raise RuntimeError(f'candidate zero-library rows were not reinserted as zeros: {pair_id}')
    if not np.allclose(truth.loc[positive_spots].sum(axis=1), 1.0, atol=1e-10):
        raise RuntimeError(f'truth positive rows are not normalized: {pair_id}')
    truth_scored = with_other(truth.loc[positive_spots], common)
    rows: list[dict[str, object]] = []
    for method, prediction in (('rctd', rctd), ('cell2location', candidate.loc[positive_spots])):
        rows.append({'method': method, 'pair_id': pair_id, 'slice_id': slice_id, 'resolution': resolution, 'status': 'ok', 'comparison_scope': 'same_expression_exact_positive_spots_exact_61_named_types_plus_other', 'n_input_spots': len(full_spots), 'n_scored_spots': len(positive_spots), 'n_zero_library_spots_excluded': len(zero_spots), 'n_common_named_cell_types': len(common), 'n_scored_cell_types_including_other': len(common) + 1, **metrics(truth_scored, with_other(prediction, common))})
    audit = {'pair_id': pair_id, 'slice_id': slice_id, 'resolution': resolution, 'n_full_spots': len(full_spots), 'n_positive_spots': len(positive_spots), 'n_zero_library_spots': len(zero_spots), 'n_truth_fine_types': len(truth.columns), 'n_exact_retained_named_types': len(common), 'rctd_and_candidate_column_sets_equal': True, 'candidate_positive_spot_order_exact': True, 'candidate_zero_rows_zero': True, 'canonical_common_cell_type_order_json': json.dumps(list(common))}
    lineage = [{'pair_id': pair_id, 'artifact': 'simulation', **artifact(simulation_path)}, {'pair_id': pair_id, 'artifact': 'truth', **artifact(truth_path)}, {'pair_id': pair_id, 'artifact': 'rctd', **artifact(Path(rctd_row['output_path']))}, {'pair_id': pair_id, 'artifact': 'cell2location_corrected', **artifact(candidate_path)}, {'pair_id': pair_id, 'artifact': 'cell2location_corrected_metadata', **artifact(candidate_metadata_path)}]
    return (rows, audit, lineage)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument('--candidate-dir', type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--slice-id', default='100')
    parser.add_argument('--resolution', type=float, action='append')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'refusing to overwrite {args.output_dir}')
    requested_resolutions = args.resolution if args.resolution is not None else [0.1, 0.25]
    resolutions = tuple(dict.fromkeys((float(value) for value in requested_resolutions)))
    if not resolutions:
        raise ValueError('at least one resolution is required')
    simulations = read_manifest(args.run_dir / 'simulation_manifest.csv')
    methods = read_manifest(args.run_dir / 'method_manifest.csv')
    rows, audits, lineage = ([], [], [])
    for resolution in resolutions:
        simulation_row = one_row(simulations, slice_id=args.slice_id, resolution=resolution, status='ok')
        rctd_row = one_row(methods, slice_id=args.slice_id, resolution=resolution, method='rctd', status='ok')
        result_rows, audit, result_lineage = score_resolution(simulation_row, rctd_row, args.candidate_dir)
        rows.extend(result_rows)
        audits.append(audit)
        lineage.extend(result_lineage)
    args.output_dir.mkdir(parents=True)
    scores = pd.DataFrame(rows).sort_values(['resolution', 'method'])
    scores_path = args.output_dir / 'deconvolution_scores.csv'
    scores.to_csv(scores_path, index=False)
    audit_path = args.output_dir / 'support_audit.csv'
    pd.DataFrame(audits).sort_values('resolution').to_csv(audit_path, index=False)
    lineage_path = args.output_dir / 'input_lineage.csv'
    pd.DataFrame(lineage).to_csv(lineage_path, index=False)
    payload = {'schema_version': 1, 'analysis': 'slice100_cell2location_label_matched_corrective_diagnostic', 'generated_utc': datetime.now(timezone.utc).isoformat(), 'publication_status': 'diagnostic_only_not_an_aggregate_method_ranking', 'scope': {'slice_id': str(args.slice_id), 'resolutions': list(resolutions), 'methods': ['rctd', 'cell2location'], 'comparison': 'same expression, exact positive spots, exact retained 61 named fine labels plus Other'}, 'source': {'script': str(Path(__file__).resolve()), 'run_dir': str(args.run_dir.resolve()), 'candidate_dir': str(args.candidate_dir.resolve())}, 'outputs': {path.name: None for path in (scores_path, audit_path, lineage_path)}}
    provenance_path = args.output_dir / 'scoring_provenance.json'
    provenance_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    for path in (scores_path, audit_path, lineage_path, provenance_path):
        print(f'Saved: {path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
