"""Plot the existing per-slice expression summary metrics without smoothing."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from plot import configure_matplotlib, style_axis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    source = args.input_root / 'evaluation/slice_metrics.csv'
    frame = pd.read_csv(source)
    stem = args.output_dir / 'full_axis_expression_summaries'
    paths = [stem.with_suffix(ext) for ext in ['.pdf', '.svg', '.png']]
    record_path = args.output_dir / 'expression_summaries_provenance.json'
    if any(p.exists() for p in [*paths, record_path]):
        raise FileExistsError('expression summary outputs already exist')
    configure_matplotlib()
    fig, axes = plt.subplots(3, 2, figsize=(7.5, 6.8), sharex='col', sharey='row')
    metrics = [('mean_library_size', 'Mean counts per position', 1),
               ('mean_detected_genes', 'Mean detected genes (of 550)', 1),
               ('overall_zero_fraction', 'Zero count entries (%)', 100)]
    for column, (age, color) in enumerate([('E15.5', '#0072B2'), ('E18.5', '#D55E00')]):
        data = frame[frame.age == age].sort_values('z_index')
        for row, (metric, label, scale) in enumerate(metrics):
            axis = axes[row, column]
            axis.plot(data.z_world, data[metric] * scale, color=color, linewidth=1.15)
            axis.set_xlim(data.z_world.min(), data.z_world.max())
            style_axis(axis)
            if column == 0: axis.set_ylabel(label)
            if row == 0: axis.set_title(age)
            if row == 2: axis.set_xlabel('DevCCF z coordinate')
    fig.subplots_adjust(left=.115, right=.98, bottom=.12, top=.94, wspace=.16, hspace=.24)
    fig.text(.5, .035, 'All positions and 550 genes; unsmoothed per-slice summaries. No target-expression ground truth.', ha='center', fontsize=7)
    for path in paths:
        fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    record_path.write_text(json.dumps({'input': str(source.resolve()), 'outputs': [str(p.resolve()) for p in paths],
        'metrics': [item[0] for item in metrics], 'smoothing': 'none', 'all_positions': True,
        'n_genes': 550, 'dpi': args.dpi, 'accuracy_claim_authorized': False}, indent=2) + '\n')
    print(stem)


if __name__ == '__main__': main()
