"""Render a compact, one-row comparison of the two batch-removal tasks.

Settings transcribed from 04_batch_effect_removal/config.yaml, its README,
real_slice_robustness/config.yaml, and this visualization directory's README.
Run with the reproduction Python environment; no experimental data are loaded.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/feast-reproduce-matplotlib')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle


OUTPUT_DIR = Path(__file__).resolve().parent / 'figures' / 'overview'
FONT_DIR = Path(os.environ.get('FEAST_FONT_DIR', str(Path(sys.prefix) / 'fonts')))
ROWS = [
    ('Reference', '151673 · simulated at α=0', '151675 · raw section'),
    ('Query', '151673 · FEAST-perturbed copy', '151676 · raw or FEAST-perturbed'),
    ('Perturbation', 'Shift only / diagonal affine', 'Diagonal affine'),
    ('Conditions shown', 'α = 0.25, 0.5, 0.75, 1.0', 'Raw anchor; α = 0, 0.5, 1.0'),
    ('Spot pairing', 'Known one-to-one pairs', 'No cross-section pairs'),
]


def main():
    for filename in ('arial.ttf', 'arialbd.ttf'):
        font_manager.fontManager.addfont(FONT_DIR / filename)
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 13,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'text.color': '#24313F', 'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
    })
    # Preserve the page dimensions when exporting, for easy figure assembly.
    fig = plt.figure(figsize=(9.5, 2.65))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    left, split, right = 0.012, 0.585, 0.988
    columns = (0.024, 0.205, 0.603)
    top, header_bottom, bottom = 0.975, 0.813, 0.235

    ax.add_patch(Rectangle((left, header_bottom), split - left, top - header_bottom,
                           facecolor='#F0F5FA', edgecolor='none'))
    ax.add_patch(Rectangle((split, header_bottom), right - split, top - header_bottom,
                           facecolor='#FBF3EC', edgecolor='none'))
    ax.text(columns[0], 0.894, 'Setting', va='center', fontsize=12, fontweight='bold')
    ax.text(columns[1], 0.894, 'Same-slice recovery', va='center',
            fontsize=16, fontweight='bold', color='#35679C')
    ax.text(columns[2], 0.894, 'Two-slice robustness', va='center',
            fontsize=16, fontweight='bold', color='#A46737')
    ax.plot((left, right), (header_bottom, header_bottom), color='#ACB7C2', lw=0.8)
    row_height = (header_bottom - bottom) / len(ROWS)
    for index, row in enumerate(ROWS):
        y_top = header_bottom - index * row_height
        y = y_top - row_height / 2
        for column, content in enumerate(row):
            ax.text(columns[column], y, content, va='center',
                    fontsize=12 if column == 0 else 13,
                    fontweight='bold' if column == 0 else 'normal')
        ax.plot((left, right), (y_top - row_height, y_top - row_height),
                color='#D9DFE5', lw=0.55)
    ax.plot((split, split), (bottom, top), color='#D0D8DF', lw=0.65)

    ax.text(columns[0], 0.153,
            'α=0: resampling control    ·    α=1: primary endpoint    ·    '
            'α>1: stress tests (not shown)',
            fontsize=11.7, va='center', color='#596571')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'svg', 'png'):
        path = OUTPUT_DIR / f'experimental_setting_table.{extension}'
        fig.savefig(path, dpi=600)
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
