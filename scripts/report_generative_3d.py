"""Report local modeling groups, original labels, batch effects and stage timings."""
from pathlib import Path
import gzip
import json
import os
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def correlation(a, b):
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 2 or np.std(a[keep]) == 0 or np.std(b[keep]) == 0:
        return float('nan')
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def moran(values, xy):
    if len(xy) < 2:
        return np.full(values.shape[1], np.nan)
    k = min(6, len(xy) - 1)
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    weights = sparse.csr_matrix((np.ones(len(xy) * k) / k,
                               (np.repeat(np.arange(len(xy)), k), neighbors.ravel())), shape=(len(xy), len(xy)))
    out = []
    for start in range(0, values.shape[1], 64):
        centered = values[:, start:start+64].astype(float)
        centered -= centered.mean(axis=0)
        denom = np.square(centered).sum(axis=0)
        numerator = (centered * (weights @ centered)).sum(axis=0)
        out.extend(np.divide(numerator, denom, out=np.full(len(denom), np.nan), where=denom > 0))
    return np.asarray(out)


def storage_bytes(root):
    seen = set(); total = 0
    for path in root.rglob('*'):
        if path.is_file():
            stat = path.stat(); identity = (stat.st_dev, stat.st_ino)
            if identity not in seen:
                total += stat.st_size; seen.add(identity)
    return total


def report(study):
    is06 = study == '06_3d_stack'
    output = ROOT / study / 'outputs' / ('generative_five_reference_v1' if is06 else 'generative_transfer_v1')
    report_dir = output / 'local_diagnostics'
    report_dir.mkdir(exist_ok=True)
    files = sorted(output.glob('targets/*/*/generated.h5ad') if is06 else output.glob('E*/*.h5ad'))
    expected = 336 if is06 else 360
    if len(files) != expected:
        raise ValueError(f'{study}: expected {expected} completed targets, got {len(files)}')
    target_data_dir = Path(json.loads((output / 'plan.json').read_text())['data_dir']) if is06 else None
    slice_rows = []; group_rows = []
    with gzip.open(report_dir / 'gene_statistics.csv.gz', 'wt') as gene_out:
        header = True
        for path in files:
            data = ad.read_h5ad(path)
            local = json.loads(data.uns['local_generation_json'])
            info = data.uns['study06'] if is06 else data.uns['publication_reproduction']
            cohort = str(info['density_name'] if is06 else info['age'])
            target_id = int(info['target_slice'] if is06 else info['z_index'])
            z = float(data.obs['z'].iloc[0])
            truth = None
            if is06:
                real = ad.read_h5ad(target_data_dir / f'{path.parent.name}.h5ad')
                truth = dense(real[:, data.var_names].layers['counts'])
            values = dense(data.layers['counts'])
            common = {'study': study, 'cohort': cohort, 'target_id': target_id, 'z': z}
            batch = local['batch']
            slice_rows.append({**common, 'n_positions': data.n_obs,
                'merged_positions': local['merging']['target_affected_positions'],
                'merged_fraction': local['merging']['target_affected_positions'] / data.n_obs,
                'donor_groups': len(local['donors']), 'n_merges': len(local['merging']['merges']),
                'D_mean': batch['D'][0], 'D_fano': batch['D'][1], 'D_zero': batch['D'][2],
                'b_mean': batch['b'][0], 'b_fano': batch['b'][1], 'b_zero': batch['b'][2],
                **local['timings'], **dict(data.uns['de_novo']['timings']), 'h5ad_bytes': path.stat().st_size})
            for label_kind, key in [('original', 'class' if is06 else 'region'), ('modeling', '_feast_model_group')]:
                labels = data.obs[key].astype(str).to_numpy()
                for label in sorted(set(labels)):
                    mask = labels == label
                    x = values[mask]
                    mean, variance, zero = x.mean(axis=0), x.var(axis=0), (x == 0).mean(axis=0)
                    gene_frame = pd.DataFrame({**common, 'label_kind': label_kind, 'label': label,
                        'gene': list(data.var_names), 'n_positions': int(mask.sum()),
                        'generated_mean': mean, 'generated_variance': variance, 'generated_zero_prop': zero})
                    group_row = {**common, 'label_kind': label_kind, 'label': label,
                                 'n_positions': int(mask.sum()), 'mean': float(mean.mean()), 'zero_prop': float(zero.mean())}
                    if label_kind == 'modeling':
                        target = local['target_statistics'][label]
                        for stage in ['fused', 'deformed']:
                            stats = pd.DataFrame(target[stage], index=local['genes']).loc[data.var_names]
                            for column in ['mean', 'variance', 'zero_prop']:
                                gene_frame[f'{stage}_{column}'] = stats[column].to_numpy()
                        diag = data.uns['de_novo']['count_diagnostics'][label]
                        for column in ['intensity_variance_ratio', 'clipped_positions']:
                            gene_frame[column] = pd.Series(diag[column], index=local['genes']).loc[data.var_names].to_numpy()
                        group_row['members'] = json.dumps(local['merging']['members'][label])
                        group_row['donor'] = local['donors'].get(label, '')
                    if truth is not None:
                        real_x = truth[mask]
                        real_mean, real_var, real_zero = real_x.mean(axis=0), real_x.var(axis=0), (real_x == 0).mean(axis=0)
                        for key, val in [('mean', real_mean), ('variance', real_var), ('zero_prop', real_zero)]:
                            gene_frame[f'real_{key}'] = val
                        xy = np.asarray(data.obsm['spatial'])[mask, :2]
                        gene_frame['generated_moran'] = moran(x, xy)
                        gene_frame['real_moran'] = moran(real_x, xy)
                        group_row.update(mean_correlation=correlation(mean, real_mean),
                            variance_correlation=correlation(variance, real_var), zero_correlation=correlation(zero, real_zero),
                            moran_correlation=correlation(gene_frame['generated_moran'].to_numpy(), gene_frame['real_moran'].to_numpy()))
                    # Fixed union of columns permits streaming both label types.
                    columns = ['study','cohort','target_id','z','label_kind','label','gene','n_positions',
                        'generated_mean','generated_variance','generated_zero_prop',
                        'fused_mean','fused_variance','fused_zero_prop','deformed_mean','deformed_variance','deformed_zero_prop',
                        'intensity_variance_ratio','clipped_positions','real_mean','real_variance','real_zero_prop',
                        'generated_moran','real_moran']
                    gene_frame.reindex(columns=columns).to_csv(gene_out, index=False, header=header)
                    header = False
                    group_rows.append(group_row)
            print(study, cohort, target_id, flush=True)
    pd.DataFrame(slice_rows).to_csv(report_dir / 'slice_diagnostics.csv', index=False)
    pd.DataFrame(group_rows).to_csv(report_dir / 'group_diagnostics.csv', index=False)
    (report_dir / 'provenance.json').write_text(json.dumps({'study': study, 'targets': len(files),
        'storage_bytes_deduplicating_hardlinks': storage_bytes(output),
        'target_expression_accuracy_claim': is06, 'spatial_metric': 'within-label 6-NN per-gene Moran correlation',
        'trajectory_identity': 'original labels; modeling memberships recorded separately',
        'timing_units': 'seconds', 'statistics': 'requested fused/deformed and realized counts; unfavorable results retained'}, indent=2))
    plot(report_dir, pd.DataFrame(slice_rows))


def plot(output, frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.family': 'DejaVu Sans',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained')
    for cohort, group in frame.groupby('cohort', sort=False):
        group = group.sort_values('z')
        axes[0].plot(group['z'], group['merged_fraction'], label=cohort)
        axes[1].plot(group['z'], group['b_mean'], label=cohort)
        axes[2].plot(group['z'], group['ot'], label=cohort)
    for ax, ylabel in zip(axes, ['Fraction of positions with merged labels', 'Batch shift in log mean', 'OT time (seconds)']):
        ax.set(xlabel='Actual z', ylabel=ylabel)
    axes[0].legend(frameon=False, fontsize=8)
    for suffix in ['pdf','svg','png']:
        fig.savefig(output / f'local_generation_diagnostics.{suffix}', dpi=300)
    plt.close(fig)

if __name__ == '__main__':
    for study in ['06_3d_stack', '07_3d_transfer']:
        report(study)
