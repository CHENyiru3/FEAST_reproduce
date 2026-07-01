#!/usr/bin/env python3
"""Debug means_per_cluster_mu_fg column names vs factor_names."""
import sys, traceback
import numpy as np
import scanpy as sc
import cell2location
import scvi
from cell2location.models import RegressionModel

scvi.settings.seed = 2026
scvi.settings.device = "cpu"

ref = sc.read_h5ad('/maiziezhou_lab2/yiru/Datasets/Processed/Allen_Zhuang_ABCA_1/h5ad/Zhuang-ABCA-1.007.h5ad')
ref.var_names_make_unique()

ref.obs['_cell_type'] = ref.obs['cell_type'].astype(str)
cell_types_orig = sorted(ref.obs['_cell_type'].unique())
print(f'Original cell types: {len(cell_types_orig)}')
print(f'Original cell types[:3]: {cell_types_orig[:3]}')

if hasattr(ref.X, 'toarray'):
    ref.X = ref.X.toarray()
ref.X = np.maximum(np.round(ref.X).astype(np.int64), 0)
ref.obs['batch'] = 'all'

print('Training RegressionModel...')
RegressionModel.setup_anndata(ref, batch_key='batch', labels_key='_cell_type')
mod_ref = RegressionModel(ref)
mod_ref.train(max_epochs=250, batch_size=2500, train_size=1, lr=0.002)
print('Training completed')

print('Exporting posterior...')
ref = mod_ref.export_posterior(ref, sample_kwargs={'num_samples': 1000, 'batch_size': 2500})

# Check factor_names
fn = list(ref.uns['mod']['factor_names'])
print(f'\nfactor_names count: {len(fn)}')
print(f'factor_names[:3]: {fn[:3]}')

# Check varm columns
mdf = ref.varm['means_per_cluster_mu_fg']
print(f'\nmeans_per_cluster_mu_fg shape: {mdf.shape}')
print(f'means_per_cluster_mu_fg index (genes) [:3]: {list(mdf.index[:3])}')
print(f'means_per_cluster_mu_fg columns (cell types) [:5]: {list(mdf.columns[:5])}')
print(f'means_per_cluster_mu_fg columns [-3:]: {list(mdf.columns[-3:])}')

# Compare
fn_set = set(fn)
col_set = set(mdf.columns)
common = fn_set & col_set
only_fn = fn_set - col_set
only_col = col_set - fn_set
print(f'\nCommon: {len(common)}')
print(f'Only in factor_names: {len(only_fn)}')
if only_fn:
    print(f'  samples: {list(only_fn)[:3]}')
print(f'Only in columns: {len(only_col)}')
if only_col:
    print(f'  samples: {list(only_col)[:3]}')

# Check if columns are factor_names but with different format (e.g. numeric)
print(f'\nColumn dtype: {mdf.columns.dtype}')
print(f'factor_names dtype: {type(fn[0]) if fn else "N/A"}')

# Check if columns are numeric (categorical codes)
if mdf.columns.dtype.kind in ('i', 'f'):
    print('Columns are NUMERIC — mapping through categorical')
    cat = ref.obs['_cell_type'].astype('category')
    for i in range(min(5, len(mdf.columns))):
        col_name = mdf.columns[i]
        print(f'  col[{i}] = {col_name}')
else:
    print('Columns are strings but differ from factor_names')

# Try to extract via index position
n_ct = len(fn)
inf_aver = mdf.iloc[:, :n_ct]  # Take first n_ct columns
inf_aver = inf_aver.T  # Now (cell_types x genes)
inf_aver.index = fn  # Set the index to factor_names
print(f'\nWorkaround: inf_aver shape = {inf_aver.shape}')
print(f'  index[:3]: {list(inf_aver.index[:3])}')
print('OK')
