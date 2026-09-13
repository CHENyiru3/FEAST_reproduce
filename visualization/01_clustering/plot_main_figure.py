"""Render the compact cause-to-response figure for Study 01 clustering."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import plot_clustering_stability as stability
import plot_spatial_expression as spatial


OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_STEM = "clustering_intervention_response"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_expression_panels() -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray], float]:
    xy = spatial._read_xy(spatial.REFERENCE_DIR / f"{spatial.SLICE}.h5ad")
    panels: dict[tuple[str, str], np.ndarray] = {}
    for factor, _detail, lower_id, upper_id in spatial.ROW_SPECS:
        panels[factor, "baseline"] = np.log1p(spatial._read_gene(spatial._path_for("baseline", lower_id, upper_id), spatial.GENE))
        panels[factor, "lower"] = np.log1p(spatial._read_gene(spatial._path_for("decrease", lower_id, upper_id), spatial.GENE))
        panels[factor, "upper"] = np.log1p(spatial._read_gene(spatial._path_for("increase", lower_id, upper_id), spatial.GENE))
    pooled = np.concatenate(list(panels.values()))
    vmax = float(np.nanpercentile(pooled, 99.5))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = max(float(np.nanmax(pooled)), 1.0)
    return (xy, panels, vmax)


def draw_spatial_panel(ax: plt.Axes, xy: np.ndarray, values: np.ndarray, vmax: float) -> None:
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=spatial.POINT_SIZE,
        c=values,
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        marker=".",
        linewidths=0,
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")


def draw_response_panel(ax: plt.Axes, data, factor: str, show_ylabel: bool) -> None:
    subset = data.loc[data["factor"] == factor]
    for method, _label, color, marker in stability.METHOD_SPECS:
        method_data = subset.loc[subset["method"] == method]
        for slice_id in stability.SLICES:
            slice_data = method_data.loc[method_data["slice_id"] == slice_id].sort_values("nominal_level")
            ax.plot(slice_data["nominal_level"], slice_data[stability.PLOT_METRIC], color=color, linewidth=0.8, alpha=0.28, zorder=1)
        summary = method_data.groupby("nominal_level", as_index=False)[stability.PLOT_METRIC].mean()
        ax.plot(summary["nominal_level"], summary[stability.PLOT_METRIC], color=color, marker=marker, markerfacecolor="white", markeredgewidth=0.85, markersize=5.1, linewidth=2.1, zorder=3)
    ax.axhline(0.0, color="#4D4D4D", linewidth=0.8, linestyle="--", alpha=0.8, zorder=0)
    ax.axvline(stability.FACTOR_SPECS[factor]["neutral"], color="#767676", linewidth=0.8, linestyle="--", alpha=0.8, zorder=0)
    ax.set_ylim(0.0, 1.02)
    stability._format_xticks(ax, factor)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.set_xlabel(stability.FACTOR_SPECS[factor]["x_label"], fontsize=8.7)
    ax.set_title(stability.FACTOR_SPECS[factor]["title"], fontsize=10.5, fontweight="bold", pad=7)
    if show_ylabel:
        ax.set_ylabel("Partition change\n(1 − ARI) ↑", fontsize=9.4, fontweight="bold", labelpad=5)


def draw_figure(output_dir: Path, dpi: int) -> list[Path]:
    configure_matplotlib()
    xy, expression_panels, vmax = load_expression_panels()
    stability_data, verified_paths, verified_cluster_count = stability.load_stability_data()

    fig = plt.figure(figsize=(17.0, 8.4))
    grid = fig.add_gridspec(
        3,
        6,
        left=0.11,
        right=0.99,
        top=0.75,
        bottom=0.14,
        width_ratios=(1, 1, 1, 1.32, 1.32, 1.32),
        wspace=0.22,
        hspace=0.06,
    )
    spatial_axes = np.empty((3, 3), dtype=object)
    response_axes = []
    column_specs = (
        ("baseline", "FEAST\nbaseline"),
        ("lower", "Lower\nendpoint"),
        ("upper", "Higher\nendpoint"),
    )

    for row_idx, (factor, detail, _lower_id, _upper_id) in enumerate(spatial.ROW_SPECS):
        for col_idx, (condition, label) in enumerate(column_specs):
            ax = fig.add_subplot(grid[row_idx, col_idx])
            spatial_axes[row_idx, col_idx] = ax
            draw_spatial_panel(ax, xy, expression_panels[factor, condition], vmax)
            if row_idx == 0:
                ax.set_title(label, fontsize=10.3, fontweight="bold", pad=6)
        bounds = spatial_axes[row_idx, 0].get_position()
        fig.text(0.102, bounds.y0 + 0.58 * bounds.height, factor, ha="right", va="center", fontsize=10.2, fontweight="bold")
        fig.text(0.102, bounds.y0 + 0.39 * bounds.height, detail, ha="right", va="center", fontsize=7.7, color="#666666")

    response_grid = grid[:, 3:].subgridspec(1, 3, wspace=0.34)
    for index, factor in enumerate(stability.FACTOR_SPECS):
        ax = fig.add_subplot(response_grid[0, index])
        response_axes.append(ax)
        draw_response_panel(ax, stability_data, factor, show_ylabel=index == 0)

    first_spatial = spatial_axes[0, 0].get_position()
    first_response = response_axes[0].get_position()
    fig.text(first_spatial.x0, 0.885, "A  Controlled spatial intervention", ha="left", va="center", fontsize=13, fontweight="bold")
    fig.text(first_spatial.x0, 0.858, f"{spatial.GENE} expression in the representative DLPFC slice {spatial.SLICE}; shared scale across all panels.", ha="left", va="center", fontsize=8.6, color="#666666")
    fig.text(first_response.x0, 0.885, "B  Partition response across all registered slices", ha="left", va="center", fontsize=13, fontweight="bold")
    fig.text(first_response.x0, 0.858, "Same-method FEAST baseline; thin lines = slices, solid lines = three-slice mean.", ha="left", va="center", fontsize=8.6, color="#666666")

    handles = [
        Line2D([], [], color=color, marker=marker, markerfacecolor="white", markeredgewidth=0.85, linewidth=2.0, markersize=5.5, label=label)
        for _method, label, color, marker in stability.METHOD_SPECS
    ]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.77, 0.805), ncol=3, fontsize=8.8, handlelength=1.7, columnspacing=1.2)

    colorbar_axis = fig.add_axes([0.19, 0.075, 0.17, 0.014])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=vmax)), cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(f"log1p({spatial.GENE})", fontsize=8.3, labelpad=2)
    colorbar.ax.tick_params(labelsize=7.5)

    fig.suptitle("Controlled simulation interventions and clustering response", y=0.98, fontsize=17, fontweight="bold")
    fig.text(0.5, 0.945, "Pre-specified low and high settings make the spatial input and ensuing partition response directly comparable.", ha="center", va="center", fontsize=9.5, color="#4F4F4F")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / OUTPUT_STEM
    outputs: list[Path] = []
    for suffix, kwargs in {
        "pdf": {"metadata": {"Creator": "FEAST Study 01 intervention-response figure"}},
        "png": {"dpi": dpi, "metadata": {"Software": "FEAST Study 01 intervention-response figure"}},
        "svg": {"metadata": {"Creator": "FEAST Study 01 intervention-response figure"}},
    }.items():
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, **kwargs)
        print(f"Saved: {path}")
        outputs.append(path)
    plt.close(fig)

    provenance = {
        "schema_version": 1,
        "figure_set": "study01_controlled_intervention_and_partition_response",
        "source": {
            "plot_script": str(Path(__file__).resolve().relative_to(spatial.REPOSITORY_ROOT)),
            "spatial_expression_script": str(Path(spatial.__file__).resolve().relative_to(spatial.REPOSITORY_ROOT)),
            "partition_response_script": str(Path(stability.__file__).resolve().relative_to(stability.REPOSITORY_ROOT)),
            "verified_cluster_paths": sorted(verified_paths),
            "verified_cluster_count": verified_cluster_count,
        },
        "design": {
            "representative_slice": spatial.SLICE,
            "marker_gene": spatial.GENE,
            "spatial_conditions": "FEAST baseline plus pre-specified lower and higher endpoints for each control",
            "spatial_endpoints": {factor: {"lower": lower, "higher": upper} for factor, _detail, lower, upper in spatial.ROW_SPECS},
            "spatial_color_scale": "shared log1p pooled 99.5th percentile scale",
            "partition_response": "1 - ARI versus the same-method FEAST baseline across all three registered slices",
            "method_ranking_claimed": False,
        },
        "outputs": {path.name: None for path in outputs},
    }
    provenance_path = output_dir / f"{OUTPUT_STEM}_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved: {provenance_path}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    draw_figure(args.output_dir, args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
