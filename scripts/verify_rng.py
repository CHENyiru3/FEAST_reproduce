"""Verify that a real FEAST run is independent of ambient NumPy RNG state."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
AMBIENT_SEEDS = (1, 99991, 1)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _portable_path(path: Path) -> str:
    """Record a path relative to this repository, never as a site absolute."""
    return Path(os.path.relpath(path.resolve(), REPOSITORY_ROOT)).as_posix()

def _read_build_record(path: Path) -> dict[str, str]:
    fields = {}
    for line in path.read_text().splitlines():
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        fields[key.strip()] = value.strip()
    required = {'Package version', 'Required source commit'}
    missing = required.difference(fields)
    if missing:
        raise ValueError(f'build record is missing fields: {sorted(missing)}')
    return fields

def _worker(args: argparse.Namespace) -> int:
    import numpy as np
    np.random.seed(args.ambient_seed)
    import anndata as ad
    import FEAST
    reference = ad.read_h5ad(args.input)
    simulated = FEAST.simulate(reference, seed=args.seed, parameter_mode='hungarian', spatial_mode='reference_rank', assignment_blocks=False, assignment_solver='scipy', verbose=False)
    diagnostics = simulated.uns.get('simulation_diagnostics', {})
    assignment = diagnostics.get('copula_rank_diagnostics', {})
    payload = {'ambient_seed': args.ambient_seed, 'public_seed': args.seed, 'shape': list(simulated.shape), 'spatial_mode': diagnostics.get('spatial_mode'), 'assignment_blocks': assignment.get('assignment_blocks'), 'requested_assignment_solver': 'scipy', 'assignment_diagnostics': {'assignment_method': assignment.get('assignment_method'), 'assignment_blocks': assignment.get('assignment_blocks'), 'assignment_block_size': assignment.get('assignment_block_size'), 'assignment_block_multiplier': assignment.get('assignment_block_multiplier')}, 'environment': {'python': platform.python_version(), 'executable_relative_to_repository': _portable_path(Path(sys.executable)), 'numpy': np.__version__, 'anndata': metadata.version('anndata'), 'scipy': metadata.version('scipy'), 'feast': metadata.version('FEAST-py'), 'feast_module_relative_to_repository': _portable_path(Path(FEAST.__file__))}}
    Path(args.result).write_text(json.dumps(payload, sort_keys=True) + '\n')
    simulated.write_h5ad(args.simulation_output)
    return 0

def _simulations_equal(left_path: Path, right_path: Path) -> bool:
    import anndata as ad
    import numpy as np
    from scipy import sparse
    left = ad.read_h5ad(left_path)
    right = ad.read_h5ad(right_path)
    if left.shape != right.shape or not left.obs_names.equals(right.obs_names) or not left.var_names.equals(right.var_names):
        return False
    if sparse.issparse(left.X) or sparse.issparse(right.X):
        if (sparse.csr_matrix(left.X) != sparse.csr_matrix(right.X)).nnz:
            return False
    elif not np.array_equal(np.asarray(left.X), np.asarray(right.X)):
        return False
    return all((key in right.obsm and np.array_equal(np.asarray(value), np.asarray(right.obsm[key])) for key, value in left.obsm.items()))

def _parent(args: argparse.Namespace) -> int:
    started_at = _utc_now()
    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    build_record_path = Path(args.feast_build_record).resolve()
    if not build_record_path.is_file():
        raise FileNotFoundError(build_record_path)
    build_record = _read_build_record(build_record_path)
    records = []
    simulation_paths: list[Path] = []
    with tempfile.TemporaryDirectory(prefix='feast-rng-check-') as temp_dir:
        for index, ambient_seed in enumerate(AMBIENT_SEEDS):
            result_path = Path(temp_dir) / f'result_{index}.json'
            simulation_path = Path(temp_dir) / f'simulation_{index}.h5ad'
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--input', str(input_path), '--seed', str(args.seed), '--ambient-seed', str(ambient_seed), '--result', str(result_path), '--simulation-output', str(simulation_path)]
            worker_env = os.environ.copy()
            worker_env['MPLCONFIGDIR'] = str(Path(temp_dir) / 'matplotlib')
            worker_env['XDG_CACHE_HOME'] = str(Path(temp_dir) / 'cache')
            worker_env['PYTHONPYCACHEPREFIX'] = str(Path(temp_dir) / 'pycache')
            subprocess.run(command, check=True, env=worker_env)
            records.append(json.loads(result_path.read_text()))
            simulation_paths.append(simulation_path)
        simulations_identical = all((_simulations_equal(simulation_paths[0], path) for path in simulation_paths[1:]))
    contract_ok = all((record['spatial_mode'] == 'reference_rank' and record['assignment_blocks'] in (False, 'False') and (record['requested_assignment_solver'] == 'scipy') for record in records))
    environments = {json.dumps(record['environment'], sort_keys=True) for record in records}
    installed_version = records[0]['environment']['feast']
    expected_version = build_record['Package version'].split()[0]
    build_ok = installed_version == expected_version and len(environments) == 1
    passed = simulations_identical and contract_ok and build_ok
    result = {'schema_version': 1, 'configuration_id': args.configuration_id, 'status': 'validated_success' if passed else 'failed_noncanonical', 'started_at_utc': started_at, 'completed_at_utc': _utc_now(), 'input_artifact': {'artifact_id': args.input_artifact_id or input_path.name, 'path_relative_to_repository': _portable_path(input_path)}, 'feast_build': {'version': installed_version, 'required_source_commit': build_record['Required source commit'], 'build_record_path_relative_to_repository': _portable_path(build_record_path)}, 'public_configuration': {'seed': args.seed, 'parameter_mode': 'hungarian', 'spatial_mode': 'reference_rank', 'assignment_blocks': False, 'assignment_solver': 'scipy'}, 'ambient_seeds': list(AMBIENT_SEEDS), 'records': records, 'comparisons': {'seed_1_vs_99991_equal': simulations_identical, 'seed_1_repeat_equal': simulations_identical}, 'checks': {'simulation_outputs_identical': simulations_identical, 'simulation_contract_valid': contract_ok, 'worker_environments_identical': len(environments) == 1, 'installed_feast_version_matches_build': installed_version == expected_version}, 'runner': {'path_relative_to_repository': _portable_path(Path(__file__)), 'python': platform.python_version(), 'executable_relative_to_repository': _portable_path(Path(sys.executable))}, 'decision': 'ambient_rng_independence_verified_for_this_article_input_and_configuration' if passed else 'rng_gate_failed_output_is_noncanonical', 'scope_limitation': 'This gate covers the declared input and configuration; it is not a blanket claim for other FEAST modes or configurations.'}
    rendered = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output:
        output_path = Path(args.output).resolve()
        if output_path.exists():
            raise FileExistsError(f'refusing to overwrite existing RNG evidence: {output_path}')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered)
    print(rendered, end='')
    if not passed:
        print('ambient-RNG independence: FAILED', file=sys.stderr)
        return 1
    for record in records:
        if record['spatial_mode'] != 'reference_rank':
            print('unexpected spatial mode in RNG canary', file=sys.stderr)
            return 1
        if record['assignment_blocks'] not in (False, 'False'):
            print('blocked assignment was used in RNG canary', file=sys.stderr)
            return 1
    print('ambient-RNG independence: OK')
    return 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--configuration-id', default='article-rng-reference-rank-global-scipy-v1')
    parser.add_argument('--input-artifact-id')
    parser.add_argument('--output')
    parser.add_argument('--feast-build-record', default=str(Path(__file__).resolve().parents[1] / 'FEAST_BUILD.txt'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--ambient-seed', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--result', help=argparse.SUPPRESS)
    parser.add_argument('--simulation-output', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker and (args.ambient_seed is None or not args.result or not args.simulation_output):
        parser.error('worker mode requires --ambient-seed, --result, and --simulation-output')
    return args

def main() -> int:
    args = parse_args()
    return _worker(args) if args.worker else _parent(args)
if __name__ == '__main__':
    raise SystemExit(main())
