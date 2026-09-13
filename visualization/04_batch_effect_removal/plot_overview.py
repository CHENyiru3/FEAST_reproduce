"""Render a presentation-style overview of the two Study 04 tasks."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/feast-reproduce-matplotlib')
os.environ.setdefault('SOURCE_DATE_EPOCH', '0')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import anndata as ad
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from scipy.spatial import ConvexHull

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'figures' / 'overview'

INK = '#24313F'
MUTED = '#66717D'
LIGHT_LINE = '#D8DEE6'
CARD_FILL = '#FAFBFC'
REFERENCE = '#2878B5'
QUERY = '#E76F51'
ACCENT_A = '#4676B7'
ACCENT_B = '#B76E3F'
FIXED_FILL = '#EAF5F1'
FIXED_EDGE = '#8AC2B3'
SLICE_PATHS = {
    '151673': REPOSITORY_ROOT / '04_batch_effect_removal' / 'data' / 'local' / '151673.h5ad',
    '151675': REPOSITORY_ROOT / '04_batch_effect_removal' / 'real_slice_robustness' / 'data' / 'local' / '151675.h5ad',
    '151676': REPOSITORY_ROOT / '04_batch_effect_removal' / 'real_slice_robustness' / 'data' / 'local' / '151676.h5ad',
}
LAYER_COLORS = {
    'Layer_1': '#51448F',
    'Layer_2': '#7567B2',
    'Layer_3': '#258F83',
    'Layer_4': '#66B9AD',
    'Layer_5': '#E6BF54',
    'Layer_6': '#E79445',
    'WM': '#8A929B',
}


def configure_matplotlib() -> None:
    arial_paths = (Path(sys.prefix) / 'fonts' / 'arial.ttf', Path(sys.prefix) / 'fonts' / 'arialbd.ttf')
    if not all(path.is_file() for path in arial_paths):
        raise FileNotFoundError(f'Arial Regular and Bold are required: {arial_paths}')
    for path in arial_paths:
        font_manager.fontManager.addfont(path)
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'text.color': INK,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })


def load_slice_geometries() -> dict[str, dict[str, object]]:
    """Load only real spot coordinates and layer labels from the source slices."""
    geometries: dict[str, dict[str, object]] = {}
    for section, path in SLICE_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f'Real slice input is missing: {path}')
        adata = ad.read_h5ad(path, backed='r')
        try:
            if 'spatial' not in adata.obsm or 'dlpfc_layer' not in adata.obs:
                raise ValueError(f'{section} lacks spatial coordinates or DLPFC layer labels')
            coordinates = np.asarray(adata.obsm['spatial'][:], dtype=float)
            layers = adata.obs['dlpfc_layer'].astype(str).to_numpy()
        finally:
            adata.file.close()
        if coordinates.shape != (len(layers), 2) or not np.isfinite(coordinates).all():
            raise ValueError(f'{section} has invalid spatial geometry')
        unknown_layers = sorted(set(layers) - set(LAYER_COLORS))
        if unknown_layers:
            raise ValueError(f'{section} has unexpected DLPFC layers: {unknown_layers}')
        centered = coordinates - 0.5 * (coordinates.min(axis=0) + coordinates.max(axis=0))
        scale = float(np.ptp(coordinates, axis=0).max())
        normalized = centered / scale
        normalized[:, 1] *= -1.0
        hull = normalized[ConvexHull(normalized).vertices]
        geometries[section] = {
            'coordinates': normalized,
            'layers': layers,
            'hull': hull,
            'n_spots': int(len(layers)),
            'path': path,
        }
    return geometries


def rounded_box(axis: plt.Axes, x: float, y: float, width: float, height: float, *,
                facecolor: str, edgecolor: str, linewidth: float = 1.0,
                radius: float = 0.015, zorder: int = 1) -> FancyBboxPatch:
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f'round,pad=0.008,rounding_size={radius}',
        transform=axis.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    axis.add_patch(box)
    return box


def arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float],
          *, color: str = '#8B96A3', linewidth: float = 1.5) -> None:
    axis.add_patch(FancyArrowPatch(
        start, end,
        transform=axis.transAxes,
        arrowstyle='-|>',
        mutation_scale=12,
        linewidth=linewidth,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=6,
    ))


def draw_tissue(axis: plt.Axes, center: tuple[float, float], *, width: float,
                height: float, outline: str, geometry: dict[str, object],
                alpha: float = 1.0, label: str | None = None) -> None:
    cx, cy = center
    points = np.asarray(geometry['coordinates'])
    hull = np.asarray(geometry['hull'])
    hull_xy = np.column_stack((cx + hull[:, 0] * width, cy + hull[:, 1] * height))
    axis.add_patch(Polygon(
        hull_xy, closed=True, transform=axis.transAxes,
        facecolor='#F7F5F0', edgecolor=outline,
        linewidth=1.3, alpha=alpha, zorder=3,
    ))
    xs = cx + points[:, 0] * width
    ys = cy + points[:, 1] * height
    colors = [LAYER_COLORS[layer] for layer in geometry['layers']]
    axis.scatter(xs, ys, s=1.25, c=colors, edgecolors='none', alpha=alpha,
                 transform=axis.transAxes, rasterized=False, zorder=4)
    if label:
        axis.text(cx, cy - height * 0.58, label, ha='center', va='top',
                  fontsize=7.3, color=MUTED, transform=axis.transAxes)


def expression_intensity(geometry: dict[str, object], *, mode: str,
                         alpha: float) -> np.ndarray:
    """Create a deterministic conceptual expression overlay for a real slice."""
    points = np.asarray(geometry['coordinates'])
    base = 0.45 + 0.18 * np.sin(11.0 * points[:, 0]) + 0.15 * np.cos(9.0 * points[:, 1])
    base = (base - base.min()) / (base.max() - base.min())
    if mode == 'before':
        values = 0.12 + 0.66 * base
    elif mode == 'shift_only':
        values = 0.12 + 0.66 * base + 0.22 * alpha
    elif mode == 'diagonal_affine':
        modulation = 0.5 + 0.5 * np.sin(17.0 * points[:, 0] - 7.0 * points[:, 1])
        values = (0.12 + 0.66 * base) * (1.0 - 0.18 * alpha) + 0.34 * alpha * modulation
    else:
        raise ValueError(f'Unknown conceptual perturbation mode: {mode}')
    return np.clip(values, 0.0, 1.0)


def draw_expression_slice(axis: plt.Axes, center: tuple[float, float], *,
                          geometry: dict[str, object], label: str, mode: str,
                          alpha: float, width: float, height: float) -> None:
    """Overlay schematic expression intensity on the unchanged real geometry."""
    cx, cy = center
    points = np.asarray(geometry['coordinates'])
    hull = np.asarray(geometry['hull'])
    hull_xy = np.column_stack((cx + hull[:, 0] * width, cy + hull[:, 1] * height))
    axis.add_patch(Polygon(
        hull_xy, closed=True, transform=axis.transAxes,
        facecolor='#FFF8F3', edgecolor=QUERY,
        linewidth=1.1, zorder=3,
    ))
    values = expression_intensity(geometry, mode=mode, alpha=alpha)
    colors = plt.get_cmap('Oranges')(0.16 + 0.78 * values)
    axis.scatter(cx + points[:, 0] * width, cy + points[:, 1] * height,
                 s=1.1, c=colors, edgecolors='none', transform=axis.transAxes,
                 rasterized=False, zorder=4)
    axis.text(cx, cy - height * 0.59, label, ha='center', va='top', fontsize=7.0,
              color=INK, fontweight='bold', transform=axis.transAxes)


def draw_feast_box(axis: plt.Axes, center: tuple[float, float], *, detail: str) -> None:
    cx, cy = center
    rounded_box(axis, cx - 0.061, cy - 0.064, 0.122, 0.128,
                facecolor='#FFF2ED', edgecolor=QUERY, linewidth=1.2,
                radius=0.012, zorder=3)
    axis.text(cx, cy + 0.025, 'FEAST', ha='center', va='center', fontsize=9.0,
              color=QUERY, fontweight='bold', transform=axis.transAxes, zorder=5)
    axis.text(cx, cy - 0.016, detail, ha='center', va='center', fontsize=6.8,
              color=MUTED, linespacing=1.25, transform=axis.transAxes, zorder=5)


def draw_fixed_callout(axis: plt.Axes, *, x: float, y: float, width: float) -> None:
    rounded_box(axis, x, y, width, 0.032, facecolor=FIXED_FILL,
                edgecolor=FIXED_EDGE, linewidth=0.8, radius=0.009, zorder=2)
    axis.text(x + width / 2, y + 0.016,
              'FIXED  •  spot coordinates  •  spot IDs  •  layer labels',
              ha='center', va='center', fontsize=6.9, color='#317567',
              fontweight='bold', transform=axis.transAxes, zorder=4)


def task_card(axis: plt.Axes, *, y: float, panel: str, accent: str, title: str,
              subtitle: str, real_slices: bool,
              geometries: dict[str, dict[str, object]]) -> None:
    height = 0.315
    rounded_box(axis, 0.035, y, 0.93, height, facecolor=CARD_FILL,
                edgecolor=LIGHT_LINE, linewidth=1.1, radius=0.018, zorder=0)
    rounded_box(axis, 0.047, y + height - 0.058, 0.034, 0.038,
                facecolor=accent, edgecolor=accent, radius=0.010, zorder=2)
    axis.text(0.064, y + height - 0.039, panel, ha='center', va='center', color='white',
              fontsize=10.5, fontweight='bold', transform=axis.transAxes, zorder=4)
    axis.text(0.092, y + height - 0.032, title, ha='left', va='center', fontsize=11.5,
              fontweight='bold', transform=axis.transAxes, zorder=4)
    axis.text(0.092, y + height - 0.058, subtitle, ha='left', va='center', fontsize=7.5,
              color=MUTED, transform=axis.transAxes, zorder=4)

    content_y = y + 0.135
    if real_slices:
        draw_tissue(axis, (0.125, content_y + 0.005), width=0.115, height=0.190,
                    outline=REFERENCE, geometry=geometries['151675'], label='151675 reference')
        draw_tissue(axis, (0.265, content_y + 0.005), width=0.115, height=0.190,
                    outline=QUERY, geometry=geometries['151676'], label='151676 query')
        axis.text(0.195, y + 0.010, 'Different sections • no paired spots', ha='center', va='center',
                  fontsize=7.0, color=ACCENT_B, fontweight='bold', transform=axis.transAxes)
    else:
        draw_tissue(axis, (0.125, content_y + 0.005), width=0.115, height=0.190,
                    outline=REFERENCE, geometry=geometries['151673'], label='151673 reference')
        draw_tissue(axis, (0.265, content_y + 0.005), width=0.115, height=0.190,
                    outline=QUERY, geometry=geometries['151673'], label='Query copy')
        axis.plot([0.125, 0.265], [content_y + 0.005, content_y + 0.005], color='#A9B1BA',
                  linewidth=0.7, linestyle=':', transform=axis.transAxes, zorder=2)
        axis.text(0.195, y + 0.010, 'Same section • known 1:1 spots', ha='center', va='center',
                  fontsize=7.0, color=ACCENT_A, fontweight='bold', transform=axis.transAxes)

    axis.plot([0.385, 0.385], [y + 0.030, y + height - 0.083], color=LIGHT_LINE,
              linewidth=1.0, transform=axis.transAxes, zorder=2)
    arrow(axis, (0.330, content_y + 0.005), (0.397, content_y + 0.005), color=QUERY)
    query_geometry = geometries['151676'] if real_slices else geometries['151673']
    draw_expression_slice(axis, (0.445, content_y + 0.005), geometry=query_geometry,
                          label='Before', mode='before', alpha=0.0,
                          width=0.105, height=0.165)
    arrow(axis, (0.500, content_y + 0.005), (0.548, content_y + 0.005), color=QUERY)
    if real_slices:
        draw_feast_box(axis, (0.615, content_y + 0.005), detail='diagonal-affine\nα controls strength')
        arrow(axis, (0.680, content_y + 0.005), (0.724, content_y + 0.005), color=QUERY)
        axis.text(0.842, y + height - 0.095, 'PERTURBATION LADDER', ha='center', va='center',
                  fontsize=6.8, color=QUERY, fontweight='bold', transform=axis.transAxes)
        for x, level, label in zip((0.750, 0.842, 0.934), (0.0, 0.5, 1.0),
                                   ('α = 0', 'α = 0.5', 'α = 1.0'), strict=True):
            draw_expression_slice(axis, (x, content_y + 0.005), geometry=query_geometry,
                                  label=label, mode='diagonal_affine', alpha=level,
                                  width=0.076, height=0.132)
    else:
        draw_feast_box(axis, (0.615, content_y + 0.005), detail='α controls\nperturbation strength')
        arrow(axis, (0.680, content_y + 0.005), (0.744, content_y + 0.005), color=QUERY)
        axis.text(0.855, y + height - 0.095, 'TWO MODES  •  α = 0.25 → 1.0', ha='center', va='center',
                  fontsize=6.8, color=QUERY, fontweight='bold', transform=axis.transAxes)
        draw_expression_slice(axis, (0.805, content_y + 0.005), geometry=query_geometry,
                              label='Shift only', mode='shift_only', alpha=1.0,
                              width=0.090, height=0.145)
        draw_expression_slice(axis, (0.910, content_y + 0.005), geometry=query_geometry,
                              label='Diagonal affine', mode='diagonal_affine', alpha=1.0,
                              width=0.090, height=0.145)
    draw_fixed_callout(axis, x=0.425, y=y + 0.006, width=0.505)


def save_figure(figure: plt.Figure, stem: Path, dpi: int) -> dict[str, Path]:
    outputs = {'pdf': stem.with_suffix('.pdf'), 'svg': stem.with_suffix('.svg'), 'png': stem.with_suffix('.png')}
    figure.savefig(outputs['pdf'], bbox_inches='tight', metadata={
        'Title': 'How Study 04 evaluates batch-effect removal',
        'Creator': 'FEAST Study 04 plot_overview.py',
        'CreationDate': None,
        'ModDate': None,
    })
    figure.savefig(outputs['svg'], bbox_inches='tight', metadata={
        'Title': 'How Study 04 evaluates batch-effect removal',
        'Creator': 'FEAST Study 04 plot_overview.py',
        'Date': None,
    })
    figure.savefig(outputs['png'], bbox_inches='tight', dpi=dpi, metadata={
        'Software': 'FEAST Study 04 plot_overview.py',
    })
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    geometries = load_slice_geometries()
    figure, axis = plt.subplots(figsize=(13.333, 7.5))
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    axis.text(0.04, 0.945, 'How the two batch-effect tasks are constructed', ha='left', va='center',
              fontsize=24, fontweight='bold', transform=axis.transAxes)
    axis.text(0.04, 0.905, 'Real tissue geometry is retained; FEAST perturbs query expression while spatial structure stays fixed.',
              ha='left', va='center', fontsize=11.5, color=MUTED, transform=axis.transAxes)
    rounded_box(axis, 0.835, 0.916, 0.125, 0.038, facecolor='#F0F3F6', edgecolor='#D5DBE2',
                linewidth=0.8, radius=0.012, zorder=2)
    axis.text(0.8975, 0.935, 'Real Visium geometry', ha='center', va='center', fontsize=8.0,
              color=MUTED, fontweight='bold', transform=axis.transAxes, zorder=4)

    for x, label in zip((0.195, 0.675),
                        ('1  Real slice setup', '2  Query-expression perturbation'), strict=True):
        axis.text(x, 0.855, label, ha='center', va='center', fontsize=8.3, fontweight='bold',
                  color=MUTED, transform=axis.transAxes)

    task_card(
        axis, y=0.485, panel='A', accent=ACCENT_A,
        title='Controlled same-slice recovery',
        subtitle='Duplicate 151673; keep the reference fixed and perturb only the paired query copy.',
        real_slices=False,
        geometries=geometries,
    )
    task_card(
        axis, y=0.125, panel='B', accent=ACCENT_B,
        title='Real two-slice robustness',
        subtitle='Keep 151675 raw; add a controlled expression perturbation only to 151676.',
        real_slices=True,
        geometries=geometries,
    )

    axis.text(0.50, 0.045, 'All slice shapes and spot positions are real • orange intensity overlays illustrate expression perturbation',
              ha='center', va='center', fontsize=7.2, color=MUTED, transform=axis.transAxes)

    stem = output / 'batch_effect_removal_design'
    figure_outputs = save_figure(figure, stem, int(args.dpi))
    provenance = {
        'schema_version': 1,
        'study': '04_batch_effect_removal',
        'figure_configuration_id': 'study04-core-slice-based-perturbation-v3',
        'figure_type': 'conceptual_schematic',
        'new_experiments_used': False,
        'quantitative_values_rendered': False,
        'real_spatial_geometry_rendered': True,
        'tasks': {
            'A': 'controlled same-slice recovery with known one-to-one spot correspondence',
            'B': 'real two-slice robustness with no cross-section spot pairing',
        },
        'rendered_components': ['real slice setup', 'query-expression perturbation'],
        'omitted_components': ['batch-correction methods', 'quantitative evaluation readouts'],
        'conceptual_expression_slice_overlays': True,
        'slice_inputs': {section: {'path': geometry['path'].resolve().relative_to(REPOSITORY_ROOT).as_posix(), 'n_spots': geometry['n_spots'], 'coordinate_key': 'obsm/spatial', 'layer_key': 'obs/dlpfc_layer'} for section, geometry in geometries.items()},
        'outputs': {name: {'path': path.resolve().relative_to(REPOSITORY_ROOT).as_posix()} for name, path in figure_outputs.items()},
        'editable_text': {'pdf_fonttype': 42, 'svg_fonttype': 'none'},
        'png_dpi': int(args.dpi),
    }
    provenance_path = output / 'batch_effect_removal_design_provenance.json'
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    print(f'wrote Study 04 core illustration to {output}')


if __name__ == '__main__':
    main()
