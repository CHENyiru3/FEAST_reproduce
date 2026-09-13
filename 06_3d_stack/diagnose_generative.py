"""Bounded empirical/core diagnostic on one shared class, retaining all genes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import anndata as ad
from FEAST.de_novo import simulate_local_references, select_z_references
from FEAST.de_novo.conditional import fit_reference, simulate_from_reference, ReferenceFitConfig
from FEAST.de_novo.core import SliceBlueprint
from workflow import load_config
from run import read_reference, target_metadata, simulation_config


def main():
    config = load_config()
    root = Path(__file__).resolve().parent
    out = root / 'outputs/generative_five_reference_v1_diagnostics'
    out.mkdir(exist_ok=True, parents=True)
    data_dir = root.parent.parent / 'Datasets/Processed/Allen_Zhuang_ABCA_1/h5ad'
    # Gap-3 target 5 has retained references 4, 7, 10, 13, 16 nearest in actual z.
    refs = [read_reference(data_dir / f'Zhuang-ABCA-1.{i:03d}.h5ad', f'Zhuang-ABCA-1.{i:03d}', 'class') for i in [4, 7, 10, 13, 16]]
    meta = target_metadata(data_dir / 'Zhuang-ABCA-1.005.h5ad', config)
    shared = set(meta['labels'])
    for ref in refs:
        shared &= set(ref.obs['class'].value_counts().loc[lambda x: x >= 150].index)
    label = sorted(shared)[0]
    refs = [ref[np.flatnonzero(ref.obs['class'].astype(str).to_numpy() == label)[:150]].copy() for ref in refs]
    target_indices = np.flatnonzero(meta['labels'] == label)[:150]
    blueprint = SliceBlueprint(coordinates=meta['spatial_3d'][target_indices],
        domain_map=meta['labels'][target_indices], obs=pd.DataFrame({'class': meta['labels'][target_indices]},
        index=np.array(meta['obs_names'])[target_indices]))
    z = {ref.uns['reference_name']: float(ref.obs['z'].iloc[0]) for ref in refs}
    target_z = float(meta['z'][0])
    sim = simulation_config(config, 0., False)
    rows = []
    for count, mode in [(2, 'empirical'), (2, 'core'), (5, 'core')]:
        if mode == 'empirical':
            names, weights, bandwidth = select_z_references(z, target_z, n_references=count)
            model = fit_reference([ref for ref in refs if ref.uns['reference_name'] in names], 'class',
                                  ReferenceFitConfig(min_gene_spots=0, min_gene_mean=0, max_gene_zero_prop=1))
            result = simulate_from_reference(model, blueprint, config=sim, random_seed=2026, reference_weights=weights)
        else:
            result = simulate_local_references(refs, blueprint, label_key='class', n_references=count,
                reference_z=z, target_z=target_z, config=sim, random_seed=2026, parameter_seed=2026,
                cache_path=out / 'reference_parameters.sqlite')
        result = result[:, meta['var_names']].copy()
        path = out / f'{count}_reference_{mode}.h5ad'
        result.write_h5ad(path, compression='gzip')
        restored = ad.read_h5ad(path)
        assert np.array_equal(result.X, restored.X)
        assert np.array_equal(result.obsm['spatial_3d'], blueprint.coordinates)
        rows.append({'mode': mode, 'n_references': count, 'label': label,
                     'n_positions': result.n_obs, 'n_genes': result.n_vars,
                     'mean': float(result.X.mean()), 'zero_prop': float((result.X == 0).mean()),
                     'bytes': path.stat().st_size})
        (out / 'summary.json').write_text(json.dumps({'diagnostics': rows,
            'scope': 'first 150 positions in first alphabetic shared class with at least 150 reference positions; all 1122 genes',
            'weighting': 'actual-z exponential; diagnostic pool contains five nearest retained references',
            'preprocessing': 'one class; no local merging needed; core includes random batch deformation',
            'attribution': 'differences include decoder, parameter fitting, weighting and batch effects; not solely reference count'}, indent=2))
        print(rows[-1], flush=True)

if __name__ == '__main__':
    main()
