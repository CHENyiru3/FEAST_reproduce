"""Validate Study 02 geometry, artifacts, solver evidence, and row coverage."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import yaml
from FEAST.alignment import apply_spatial_transform

def expected_artifact_names(method: str) -> set[str]:
    common = {'aligned.h5ad', 'aligned_coordinates.csv', 'diagnostics.json', 'transform.json'}
    if method == 'spateo':
        return common | {'mapping_matrix.npy'}
    return common | {'coupling_matrix.npy', 'inner_emd_diagnostics.csv'}

def artifact_manifest_is_valid(output: Path, row: object) -> bool:
    try:
        artifacts = json.loads(row.artifacts)
    except (TypeError, json.JSONDecodeError):
        return False
    expected_names = expected_artifact_names(row.method)
    if not isinstance(artifacts, list) or len(artifacts) != len(expected_names):
        return False
    expected_paths = {name: (output / name).resolve() for name in expected_names}
    listed_names: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get('path'), str):
            return False
        path = Path(item['path'])
        name = path.name
        if name in listed_names or name not in expected_paths or path.resolve() != expected_paths[name] or (not path.is_file()):
            return False
        listed_names.add(name)
    try:
        actual_names = {path.name for path in output.iterdir() if path.is_file() and path.name != 'runner.log'}
    except OSError:
        return False
    runner_log = output / 'runner.log'
    return bool(listed_names == expected_names and actual_names == expected_names and runner_log.is_file())

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--rotation-dir', type=Path, required=True)
    parser.add_argument('--methods-dir', type=Path, required=True)
    parser.add_argument('--scores-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.config.read_text())
    tolerance = float(config['geometry_tolerance'])
    reference = ad.read_h5ad(args.reference)
    reference_coords = np.asarray(reference.obsm['spatial'], dtype=np.float64)
    if not reference.obs_names.is_unique or not np.isfinite(reference_coords).all():
        raise RuntimeError('reference spot IDs/coordinates are invalid')
    rotation_config = config['rotation']
    plate_bounds = np.stack([reference_coords.min(axis=0), reference_coords.max(axis=0)])
    plate_center = plate_bounds.mean(axis=0)
    plate_half_extent = (plate_bounds[1] - plate_bounds[0]) / 2.0
    rotation_center = plate_center + float(rotation_config['pivot_half_extent_fraction']) * plate_half_extent
    expected_retained = {float(angle): int(count) for angle, count in rotation_config['expected_retained_spots'].items()}
    rotations = pd.read_csv(args.rotation_dir / 'rotation_manifest.csv')
    expected_rotations = {(alteration, float(angle)) for alteration in config['alterations'] for angle in config['angles']}
    if len(rotations) != len(expected_rotations) or set(zip(rotations['alteration'], rotations['angle_degrees'].astype(float))) != expected_rotations or rotations[['alteration', 'angle_degrees']].duplicated().any():
        raise RuntimeError('rotation manifest does not match the configured inputs')
    rotation_rows = []
    for row in rotations.itertuples(index=False):
        source_path, moving_path = (Path(row.source_path), Path(row.output_path))
        source, moving = (ad.read_h5ad(source_path), ad.read_h5ad(moving_path))
        transform = moving.uns['feast_alignment_transform']
        source_coords = np.asarray(source.obsm['spatial'], dtype=np.float64)
        retained_positions = source.obs_names.get_indexer(moving.obs_names)
        forward_coords = apply_spatial_transform(source_coords, transform['forward_matrix'])
        expected_mask = np.all((forward_coords >= plate_bounds[0]) & (forward_coords <= plate_bounds[1]), axis=1)
        expected_positions = np.flatnonzero(expected_mask)
        restored = apply_spatial_transform(moving.obsm['spatial'], transform['inverse_matrix'])
        retained_n_spots = int(moving.n_obs)
        dropped_n_spots = int(source.n_obs - moving.n_obs)
        retained_fraction = float(moving.n_obs / source.n_obs)
        valid = bool(source.obs_names.equals(reference.obs_names) and np.array_equal(retained_positions, expected_positions) and moving.var_names.equals(source.var_names) and (np.max(np.abs(source_coords - reference_coords)) <= tolerance) and (np.max(np.abs(restored - source_coords[retained_positions])) <= tolerance) and np.all((moving.obsm['spatial'] >= plate_bounds[0]) & (moving.obsm['spatial'] <= plate_bounds[1])) and (retained_n_spots == expected_retained[float(row.angle_degrees)]) and (int(row.input_n_spots) == source.n_obs) and (int(row.retained_n_spots) == retained_n_spots) and (int(row.n_spots) == retained_n_spots) and (int(row.dropped_n_spots) == dropped_n_spots) and np.isclose(float(row.retained_fraction), retained_fraction) and bool(transform.get('plate_bounds_applied')) and (int(transform.get('input_n_spots', -1)) == source.n_obs) and (int(transform.get('retained_n_spots', -1)) == retained_n_spots) and (int(transform.get('dropped_n_spots', -1)) == dropped_n_spots) and np.allclose(np.asarray(transform.get('plate_bounds'), dtype=np.float64), plate_bounds, rtol=0.0, atol=tolerance) and np.allclose(np.asarray(transform.get('center'), dtype=np.float64), rotation_center, rtol=0.0, atol=tolerance))
        rotation_rows.append({'kind': 'rotation', 'job_id': f'{row.alteration}__{row.angle_degrees:g}', 'valid': valid})
    methods = pd.read_csv(args.methods_dir / 'method_manifest.csv')
    expected_methods = {(alteration, float(angle), method) for alteration, angle in expected_rotations for method in ('spateo', 'paste')}
    if len(methods) != len(expected_methods) or set(zip(methods['alteration'], methods['angle_degrees'].astype(float), methods['method'])) != expected_methods or methods['job_id'].duplicated().any():
        raise RuntimeError('method manifest does not match the configured jobs')
    method_rows = []
    for row in methods.itertuples(index=False):
        output = Path(row.output_dir)
        aligned = ad.read_h5ad(output / 'aligned.h5ad')
        moving = ad.read_h5ad(Path(row.moving))
        retained_positions = reference.obs_names.get_indexer(moving.obs_names)
        transform = json.loads((output / 'transform.json').read_text())
        diagnostics = json.loads((output / 'diagnostics.json').read_text())
        expected_configuration_id = f'study02-{row.method}-{row.alteration}-{float(row.angle_degrees):g}'
        matrix_path = output / ('mapping_matrix.npy' if row.method == 'spateo' else 'coupling_matrix.npy')
        matrix = np.load(matrix_path, mmap_mode='r', allow_pickle=False)
        matrix_finite = bool(matrix.size > 0 and np.isfinite(matrix).all())
        matrix_mass = float(np.sum(matrix, dtype=np.float64)) if matrix_finite else np.nan
        matrix_minimum = float(np.min(matrix)) if matrix_finite else np.nan
        if row.method == 'spateo':
            negative_tolerance = float(config['spateo']['mapping_negative_tolerance'])
            matrix_evidence_valid = bool(matrix_minimum >= -negative_tolerance and diagnostics.get('mapping_shape') == list(matrix.shape) and (diagnostics.get('mapping_all_finite') is True) and np.isclose(float(diagnostics.get('mapping_minimum', np.nan)), matrix_minimum, rtol=1e-12, atol=1e-12) and np.isclose(float(diagnostics.get('mapping_mass', np.nan)), matrix_mass, rtol=1e-12, atol=1e-12) and (float(diagnostics.get('mapping_negative_tolerance', np.nan)) == negative_tolerance))
        else:
            coupling = diagnostics.get('coupling', {})
            matrix_evidence_valid = bool(coupling.get('shape') == list(matrix.shape) and np.isclose(float(coupling.get('mass', np.nan)), matrix_mass, rtol=1e-12, atol=1e-12))
        solver_valid = diagnostics.get('status') == 'completed' and diagnostics.get('cuda_available') is True and (diagnostics.get('mapping_all_finite') is True) and (diagnostics.get('spot_support_preserved') is True) and (diagnostics.get('gene_order_preserved') is True) and (diagnostics.get('convergence_claim') is False) and (diagnostics.get('iteration_schedule', {}).get('kind') == 'fixed') and (diagnostics.get('iteration_schedule', {}).get('completed_return') is True) if row.method == 'spateo' else diagnostics.get('status') == 'converged' and diagnostics.get('positive_convergence_evidence') is True and (diagnostics.get('all_inner_emd_optimal') is True) and (diagnostics.get('outer_conditional_gradient', {}).get('positive_convergence_evidence') is True)
        required = [output / 'aligned.h5ad', matrix_path, output / 'aligned_coordinates.csv']
        artifacts_present = all(path.is_file() for path in required)
        manifest_artifacts_valid = artifact_manifest_is_valid(output, row)
        valid = bool(row.status == 'ok' and row.configuration_id == expected_configuration_id and (int(row.public_seed) == int(config['seed'])) and (transform.get('status') == 'ok') and (transform.get('configuration_id') == expected_configuration_id) and (int(transform.get('public_seed', -1)) == int(config['seed'])) and (transform.get('feast', {}).get('commit') == config['required_feast_commit']) and solver_valid and artifacts_present and manifest_artifacts_valid and np.all(retained_positions >= 0) and np.all(np.diff(retained_positions) > 0) and aligned.obs_names.equals(moving.obs_names) and aligned.var_names.equals(moving.var_names) and (int(row.reference_n_spots) == reference.n_obs) and (int(row.moving_n_spots) == moving.n_obs) and (int(row.dropped_n_spots) == reference.n_obs - moving.n_obs) and np.isclose(float(row.retained_fraction), moving.n_obs / reference.n_obs) and (matrix.shape == (reference.n_obs, moving.n_obs)) and matrix_finite and np.isfinite(matrix_mass) and (matrix_mass > 0) and matrix_evidence_valid)
        method_rows.append({'kind': 'method', 'job_id': row.job_id, 'valid': valid})
    metrics = pd.read_csv(args.scores_dir / 'alignment_metrics.csv')
    score_provenance = json.loads((args.scores_dir / 'provenance.json').read_text())
    score_valid = bool(len(metrics) == len(expected_methods) and metrics['job_id'].is_unique and (metrics['status'] == 'ok').all() and metrics['canonical_candidate'].all() and np.isfinite(metrics['transport_total_mass']).all() and (metrics['transport_total_mass'] > 0).all() and metrics['n_reference_spots'].eq(reference.n_obs).all() and np.allclose(metrics['retained_fraction'], metrics['n_moving_spots'] / reference.n_obs) and (metrics.loc[metrics['method'] == 'spateo', 'transport_minimum'] >= -float(config['spateo']['mapping_negative_tolerance'])).all())
    rows = rotation_rows + method_rows + [{'kind': 'score_table', 'job_id': 'alignment_metrics', 'valid': score_valid}]
    result = pd.DataFrame(rows)
    if len(rotation_rows) != len(expected_rotations) or len(method_rows) != len(expected_methods) or (not result['valid'].all()):
        raise RuntimeError('Study 02 validation failed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f'Study 02 validated: {len(expected_rotations)} rotations, {len(expected_methods)} method rows')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
