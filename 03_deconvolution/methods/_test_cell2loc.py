#!/usr/bin/env python3
"""Full cell2location test with traceback capture."""
import sys, traceback
import numpy as np
import scanpy as sc
import cell2location
import scvi
from cell2location.models import RegressionModel, Cell2location

scvi.settings.seed = 2026
scvi.settings.device = "cpu"

# Load data
ref = sc.read_h5ad('/maiziezhou_lab2/yiru/Datasets/Processed/Allen_Zhuang_ABCA_1/h5ad/Zhuang-ABCA-1.007.h5ad')
ref.var_names_make_unique()
spat = sc.read_h5ad('/maiziezhou_lab2/yiru/FEAST_experiments/outputs/05_deconvolution/007/resolution_0.1.h5ad')
spat.var_names_make_unique()

print(f'Reference: {ref.shape}')
print(f'Spatial: {spat.shape}')

# Shared genes
shared_genes = ref.var_names.intersection(spat.var_names)
print(f'Shared genes: {len(shared_genes)}')

ref = ref[:, shared_genes].copy()
spat = spat[:, shared_genes].copy()

# Prepare reference
ref.obs['_cell_type'] = ref.obs['cell_type'].astype(str)
print(f'Cell types: {ref.obs["_cell_type"].nunique()}')

if hasattr(ref.X, 'toarray'):
    ref.X = ref.X.toarray()
ref.X = np.maximum(np.round(ref.X).astype(np.int64), 0)
ref.obs['batch'] = 'all'

# Train RegressionModel
print('\n=== RegressionModel ===')
RegressionModel.setup_anndata(ref, batch_key='batch', labels_key='_cell_type')
mod_ref = RegressionModel(ref)
mod_ref.train(max_epochs=250, batch_size=2500, train_size=1, lr=0.002)
print('RegressionModel training done')

# Export posterior
ref = mod_ref.export_posterior(ref, sample_kwargs={'num_samples': 1000, 'batch_size': 2500})
print('Export posterior done')

# Extract reference signatures
factor_names = list(ref.uns['mod']['factor_names'])
mdf = ref.varm['means_per_cluster_mu_fg']
n_ct = len(factor_names)
inf_aver = mdf.iloc[:, :n_ct].T.copy()
inf_aver.index = factor_names
print(f'inf_aver shape: {inf_aver.shape}')
print(f'inf_aver index[:3]: {list(inf_aver.index[:3])}')
print(f'inf_aver columns[:3]: {list(inf_aver.columns[:3])}')
# inf_aver: (cell_types x genes)

# Prepare spatial
if hasattr(spat.X, 'toarray'):
    spat.X = spat.X.toarray()
spat.X = np.round(spat.X).astype(np.float32)

n_genes_before = spat.n_vars
sc.pp.filter_genes(spat, min_cells=1)
print(f'After filter_genes: {spat.n_vars} genes (removed {n_genes_before - spat.n_vars})')

if spat.n_vars < n_genes_before:
    shared_genes_after = spat.var_names.intersection(inf_aver.columns)
    inf_aver = inf_aver[shared_genes_after]
    spat = spat[:, shared_genes_after].copy()
    print(f'After re-alignment: spat={spat.n_vars}, inf_aver cols={inf_aver.shape[1]}')

spat.obs['batch'] = 'all'
spat.uns['spatial'] = {'batch': {'scalefactors': {'tissue_hires_scalef': 1.0, 'spot_diameter_fullres': 50.0}}}

print(f'\nSpatial ready: {spat.shape}')
print(f'inf_aver ready: {inf_aver.shape}')
print(f'Genes match: {set(spat.var_names) == set(inf_aver.columns)}')

# Cell2location
print('\n=== Cell2location ===')
N_cells = float(spat.X.sum(axis=1).mean())
print(f'Mean counts per spot: {N_cells:.1f}')

try:
    Cell2location.setup_anndata(spat, batch_key='batch')
    print('setup_anndata done')

    mod_c2l = Cell2location(
        spat,
        cell_state_df=inf_aver,
        N_cells_per_location=N_cells,
        detection_alpha=20,
    )
    print(f'Cell2location model created')
    print(f'  n_factors: {mod_c2l.n_factors}')
    print(f'  n_genes: {mod_c2l.n_genes}')

    print('Training (max_epochs=5000 for test)...')
    mod_c2l.train(max_epochs=5000, batch_size=None, train_size=1)
    print('Cell2location training done!')

    # Export posterior
    spat = mod_c2l.export_posterior(spat, sample_kwargs={'num_samples': 1000, 'batch_size': None})
    print(f'obsm keys: {list(spat.obsm.keys())}')

    for key in ['means_cell_abundance_w_sf', 'q05_cell_abundance_w_sf', 'q50_cell_abundance_w_sf']:
        if key in spat.obsm:
            print(f'  {key}: {spat.obsm[key].shape}')

    print('SUCCESS!')

except Exception as e:
    print(f'Cell2location FAILED: {type(e).__name__}: {e}')
    traceback.print_exc()
    sys.exit(1)
