"""Export the bounded empirical/core diagnostic, without production claims."""
from pathlib import Path
import json
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent / 'outputs/generative_five_reference_v1_diagnostics'
records = json.loads((root / 'summary.json').read_text())
rows = []
for record in records['diagnostics']:
    mode, count = record['mode'], record['n_references']
    data = ad.read_h5ad(root / f'{count}_reference_{mode}.h5ad')
    values = np.asarray(data.X)
    for i, gene in enumerate(data.var_names):
        rows.append({'method': f'{count} references / {mode}', 'gene': gene,
                     'mean': float(values[:,i].mean()), 'variance': float(values[:,i].var()),
                     'zero_prop': float((values[:,i] == 0).mean())})
frame = pd.DataFrame(rows)
frame.to_csv(root / 'diagnostic_plot_data.csv', index=False)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout='constrained')
methods = list(frame['method'].unique())
colors = ['#B64342', '#42949E', '#0F4D92']
for ax, column in zip(axes, ['mean', 'variance', 'zero_prop']):
    for method, color in zip(methods, colors):
        values = np.sort(frame.loc[frame['method'] == method, column].to_numpy())
        ax.plot(values, np.arange(1, len(values)+1)/len(values), label=method, color=color)
    if column != 'zero_prop':
        ax.set_xscale('symlog', linthresh=.1)
    ax.set(xlabel={'mean': 'Generated gene mean', 'variance': 'Generated gene variance', 'zero_prop': 'Generated gene zero fraction'}[column], ylabel='Cumulative gene fraction')
axes[0].legend(frameon=False, fontsize=8)
for extension in ['pdf','svg','png']:
    fig.savefig(root / f'bounded_diagnostic.{extension}', dpi=300)
plt.close(fig)
(root / 'figure_provenance.json').write_text(json.dumps({'inputs': records,
    'interpretation': 'bounded implementation diagnostic on 150 positions and 1122 genes; not production performance',
    'formats': ['pdf','svg','png'], 'editable_text': True}, indent=2))
