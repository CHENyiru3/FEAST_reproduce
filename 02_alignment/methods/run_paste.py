"""Run one PASTE job, failing closed without positive solver evidence."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import platform
import time
import traceback
import warnings
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import yaml

def preprocess(adata: ad.AnnData, min_cells: int=1) -> ad.AnnData:
    result = adata.copy()
    result.var_names_make_unique()
    if min_cells:
        sc.pp.filter_genes(result, min_cells=min_cells)
    sc.pp.normalize_total(result, target_sum=10000.0)
    sc.pp.log1p(result)
    return result

def add_joint_pca(reference: ad.AnnData, moving: ad.AnnData, n_pcs: int):
    common = reference.var_names.intersection(moving.var_names)
    if common.empty:
        raise RuntimeError('reference and moving inputs have no shared genes')
    reference = reference[:, common].copy()
    moving = moving[:, common].copy()
    combined = ad.concat([reference, moving], label='slice', keys=['reference', 'moving'], index_unique='-', join='inner')
    used_pcs = min(n_pcs, combined.n_obs - 1, combined.n_vars - 1)
    if used_pcs < 2:
        raise RuntimeError('insufficient dimensions for joint PCA')
    sc.pp.pca(combined, n_comps=used_pcs, svd_solver='arpack')
    values = np.asarray(combined.obsm['X_pca'], dtype=np.float64)
    reference.obsm['X_pca'] = values[:reference.n_obs]
    moving.obsm['X_pca'] = values[reference.n_obs:]
    return (reference, moving, used_pcs, len(common))

def weighted_procrustes(reference_coords, moving_coords, coupling):
    reference_mass = coupling.sum(axis=1)
    moving_mass = coupling.sum(axis=0)
    reference_center = reference_mass @ reference_coords
    moving_center = moving_mass @ moving_coords
    reference_centered = reference_coords - reference_center
    moving_centered = moving_coords - moving_center
    covariance = moving_centered.T @ (coupling.T @ reference_centered)
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    aligned = moving_centered @ rotation.T + reference_center
    angle = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    return (aligned, rotation, angle, reference_center, moving_center)

def install_solver_gate(config: dict):
    import ot
    state = {'inner_emd_calls': [], 'outer_conditional_gradient': None}
    original_emd = ot.optim.emd
    original_cg = ot.optim.cg
    inner_limit = int(config['inner_emd_max_iterations'])

    def strict_emd(a, b, M, numItermax=100000, log=False, center_dual=True, numThreads=1, check_marginals=True, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            transport, diagnostics = original_emd(a, b, M, numItermax=numItermax, log=True, center_dual=center_dual, numThreads=numThreads, check_marginals=check_marginals, **kwargs)
        row = {'call_index': len(state['inner_emd_calls']) + 1, 'num_iter_max': int(numItermax), 'result_code': int(diagnostics.get('result_code', -1)), 'solver_warning': diagnostics.get('warning'), 'caught_warnings': [str(item.message) for item in caught], 'cost': float(diagnostics.get('cost', np.nan))}
        row['positive_optimality_evidence'] = bool(row['result_code'] == 1 and row['solver_warning'] is None and (not row['caught_warnings']) and np.isfinite(row['cost']))
        state['inner_emd_calls'].append(row)
        if not row['positive_optimality_evidence']:
            raise RuntimeError(f'inner POT EMD did not report optimality: {row}')
        return (transport, diagnostics) if log else transport

    def strict_cg(a, b, M, reg, f, df, G0=None, line_search=None, **kwargs):
        if line_search is not None:
            original_line_search = line_search

            def compatible_line_search(cost, G, deltaG, Mi, cost_G, df_G=None, **line_kwargs):
                return original_line_search(cost, G, deltaG, Mi, cost_G, **line_kwargs)
            line_search = compatible_line_search
        kwargs['numItermaxEmd'] = inner_limit
        result = original_cg(a, b, M, reg, f, df, G0, line_search, **kwargs)
        if not kwargs.get('log'):
            raise RuntimeError('outer PASTE solve did not return a diagnostic log')
        losses = np.asarray(result[1].get('loss', []), dtype=np.float64)
        if len(losses) < 2 or not np.isfinite(losses).all():
            raise RuntimeError('outer PASTE loss history is absent or non-finite')
        iterations = len(losses) - 1
        absolute_delta = float(abs(losses[-1] - losses[-2]))
        relative_delta = float(absolute_delta / abs(losses[-1])) if losses[-1] != 0 else np.inf
        max_iterations = int(kwargs.get('numItermax', config['outer_max_iterations']))
        relative_tolerance = float(kwargs.get('stopThr', config['outer_relative_tolerance']))
        absolute_tolerance = float(kwargs.get('stopThr2', config['outer_absolute_tolerance']))
        positive = bool(iterations < max_iterations and (relative_delta < relative_tolerance or absolute_delta < absolute_tolerance))
        state['outer_conditional_gradient'] = {'iterations': iterations, 'num_iter_max': max_iterations, 'final_loss': float(losses[-1]), 'final_absolute_loss_delta': absolute_delta, 'final_relative_loss_delta': relative_delta, 'relative_tolerance': relative_tolerance, 'absolute_tolerance': absolute_tolerance, 'positive_convergence_evidence': positive}
        if not positive:
            raise RuntimeError('outer PASTE solve lacks positive convergence evidence')
        return result
    ot.optim.emd = strict_emd
    ot.optim.cg = strict_cg

    def restore():
        ot.optim.emd = original_emd
        ot.optim.cg = original_cg
    return (state, restore)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--moving', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--configuration-id', required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = {'inner_emd_calls': [], 'outer_conditional_gradient': None}
    try:
        import paste as pst
        config = yaml.safe_load(args.config.read_text())['paste']
        np.random.seed(args.seed)
        reference_raw = sc.read_h5ad(args.reference)
        moving_raw = sc.read_h5ad(args.moving)
        if not reference_raw.obs_names.is_unique or not moving_raw.obs_names.is_unique:
            raise RuntimeError('spot identifiers must be unique')
        moving_names = moving_raw.obs_names.copy()
        moving_vars = moving_raw.var_names.copy()
        reference = preprocess(reference_raw)
        moving = preprocess(moving_raw)
        reference, moving, used_pcs, common_genes = add_joint_pca(reference, moving, int(config['n_pcs']))
        state, restore = install_solver_gate(config)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                coupling, objective = pst.pairwise_align(reference, moving, alpha=float(config['alpha']), dissimilarity=str(config['dissimilarity']), use_rep='X_pca', norm=bool(config['normalize_spatial_distances']), numItermax=int(config['outer_max_iterations']), return_obj=True, verbose=False, gpu_verbose=False)
        finally:
            restore()
        escaped = [str(item.message) for item in caught]
        dangerous = [message for message in escaped if 'numItermax' in message or 'optimal' in message.lower() or 'numerical' in message.lower()]
        if dangerous:
            raise RuntimeError('solver warning escaped strict gate: ' + ' | '.join(dangerous))
        coupling = np.asarray(coupling, dtype=np.float64)
        expected_shape = (reference.n_obs, moving.n_obs)
        if coupling.shape != expected_shape:
            raise RuntimeError(f'unexpected coupling shape: {coupling.shape}')
        if not np.isfinite(coupling).all() or float(coupling.min()) < -1e-12:
            raise RuntimeError('coupling is non-finite or materially negative')
        mass = float(coupling.sum())
        row_error = float(np.max(np.abs(coupling.sum(axis=1) - 1 / reference.n_obs)))
        column_error = float(np.max(np.abs(coupling.sum(axis=0) - 1 / moving.n_obs)))
        tolerance = float(config['marginal_tolerance'])
        if abs(mass - 1) > tolerance or max(row_error, column_error) > tolerance:
            raise RuntimeError('coupling mass/marginal validation failed')
        if not state['inner_emd_calls'] or not all((row['positive_optimality_evidence'] for row in state['inner_emd_calls'])):
            raise RuntimeError('inner EMD positive evidence is incomplete')
        outer = state['outer_conditional_gradient']
        if not outer or not outer['positive_convergence_evidence']:
            raise RuntimeError('outer solver positive evidence is incomplete')
        reference_coords = np.asarray(reference.obsm['spatial'], dtype=np.float64)
        moving_coords = np.asarray(moving.obsm['spatial'], dtype=np.float64)
        aligned_coords, rotation, rotation_angle, ref_center, mov_center = weighted_procrustes(reference_coords, moving_coords, coupling)
        if not moving_raw.obs_names.equals(moving_names) or not moving_raw.var_names.equals(moving_vars):
            raise RuntimeError('PASTE preprocessing changed source identity/order')
        if not np.isfinite(aligned_coords).all():
            raise RuntimeError('aligned coordinates are non-finite')
        output_adata = moving_raw.copy()
        output_adata.obsm['spatial_aligned'] = aligned_coords
        aligned_path = args.output_dir / 'aligned.h5ad'
        coupling_path = args.output_dir / 'coupling_matrix.npy'
        coordinates_path = args.output_dir / 'aligned_coordinates.csv'
        output_adata.write_h5ad(aligned_path, compression='gzip')
        np.save(coupling_path, coupling)
        pd.DataFrame({'spot_barcode': moving_names.astype(str), 'x_moving': moving_coords[:, 0], 'y_moving': moving_coords[:, 1], 'x_aligned': aligned_coords[:, 0], 'y_aligned': aligned_coords[:, 1]}).to_csv(coordinates_path, index=False)
        pd.DataFrame(state['inner_emd_calls']).to_csv(args.output_dir / 'inner_emd_diagnostics.csv', index=False)
        diagnostics = {'status': 'converged', 'positive_convergence_evidence': True, 'inner_emd_calls': len(state['inner_emd_calls']), 'all_inner_emd_optimal': True, 'outer_conditional_gradient': outer, 'uncaptured_warnings': escaped, 'coupling': {'shape': list(coupling.shape), 'mass': mass, 'max_row_marginal_error': row_error, 'max_column_marginal_error': column_error, 'objective': float(np.asarray(objective))}}
        (args.output_dir / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
        transform = {'configuration_id': args.configuration_id, 'method': 'paste', 'status': 'ok', 'canonical_candidate': True, 'public_seed': args.seed, 'runtime_seconds': round(time.monotonic() - started, 3), 'feast': {'version': moving_raw.uns['feast_reproduce']['feast_version'], 'commit': moving_raw.uns['feast_reproduce']['feast_commit']}, 'parameters': config, 'n_pcs_used': used_pcs, 'common_gene_count': common_genes, 'rotation_angle': rotation_angle, 'rotation_matrix': rotation.tolist(), 'reference_center': ref_center.tolist(), 'moving_center': mov_center.tolist(), 'input_artifacts': {}, 'source_artifacts': {}, 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scanpy': sc.__version__, 'paste_bio': importlib.metadata.version('paste-bio'), 'pot': importlib.metadata.version('POT'), 'feast_imported_in_method_worker': False, 'feast_numpy_support_range': '>=1.24,<2', 'numpy_supported_by_feast': bool(np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion('1.24.0') and np.lib.NumpyVersion(np.__version__) < np.lib.NumpyVersion('2.0.0'))}, 'output_artifacts': {path.name: None for path in (aligned_path, coupling_path, coordinates_path)}, 'solver_diagnostics': diagnostics}
        (args.output_dir / 'transform.json').write_text(json.dumps(transform, indent=2) + '\n')
        return 0
    except Exception as exc:
        (args.output_dir / 'FAILED.json').write_text(json.dumps({'status': 'failed_noncanonical', 'configuration_id': args.configuration_id, 'error': repr(exc), 'traceback': traceback.format_exc(), 'solver_state': state}, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value)) + '\n')
        return 1
if __name__ == '__main__':
    raise SystemExit(main())
