# %%
"""Reference versus two FEAST simulated slices — one curated sample per platform.

Hardcoded sample and gene selections; no auto-selection machinery.
"""

# %%
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


# %%
EXPERIMENT_ROOT = Path("/maiziezhou_lab2/yiru/FEAST_experiments")
BENCHMARK_ROOT = EXPERIMENT_ROOT / "00_simulator_benchmark"
REFERENCE_DIR = BENCHMARK_ROOT / "exper_data"
SIMULATION_DIR = BENCHMARK_ROOT / "outputs" / "simulation_saved"
ARCHIVED_SIMULATION_DIR = BENCHMARK_ROOT / "outputs" / "archived_20260623_110615"
_ARCHIVED_FEAST_SAMPLES = {"DLPFC_151670", "Slideseq_001"}
OUTPUT_DIR = Path("/maiziezhou_lab2/yiru/Reproduce/Visualization/figures/slice_panel_compare")
OUTPUT_STEM = "reference_vs_two_feast_best_platform_samples"
POINT_SIZE_SCALE = 2.2

PLATFORM_POINT_MULT = {
    "10X Visium": 2.2,
    "MERFISH": 1.5,
    "Slide-seq V2": 1.8,
    "Stereo-seq": 1.0,
    "OpenST": 1.0,
    "Xenium": 0.7,
}

PANEL_SPECS = [
    {"sample": "DLPFC_151670",        "platform": "10X Visium",    "dataset": "Human DLPFC",       "gene": "SCGB2A2"},
    {"sample": "MERFISH_120",         "platform": "MERFISH",       "dataset": "Mouse brain",        "gene": "Pvalb"},
    {"sample": "OpenST_006",          "platform": "OpenST",        "dataset": "Mouse embryo",       "gene": "IGKC"},
    {"sample": "Slideseq_001",        "platform": "Slide-seq V2",  "dataset": "Mouse cerebellum",   "gene": "IGLC2"},
    {"sample": "Stereoseq_E12_5_E2S1","platform": "Stereo-seq",    "dataset": "Mouse embryo",       "gene": "Fabp7"},
    {"sample": "Xenium_LymphNode",    "platform": "Xenium",        "dataset": "Human lymph node",   "gene": "CD3E"},
]

FEAST_METHODS = ["FEAST_Rank", "FEAST_OT_Spatial"]


# %%
def h5ad_path_for(kind: str, sample: str) -> Path:
    if kind == "Reference":
        return REFERENCE_DIR / f"{sample}.h5ad"
    if kind in FEAST_METHODS:
        if sample in _ARCHIVED_FEAST_SAMPLES:
            return ARCHIVED_SIMULATION_DIR / kind / f"{sample}.h5ad"
        return SIMULATION_DIR / kind / f"{sample}.h5ad"
    raise ValueError(f"Unknown panel kind: {kind}")


def read_gene_vector(path: Path, gene: str, layer: str | None = None) -> np.ndarray:
    data = ad.read_h5ad(path)
    if gene not in data.var_names:
        raise ValueError(f"{gene!r} is not in {path}")
    gene_view = data[:, gene]
    matrix = gene_view.layers[layer] if layer else gene_view.X
    if sparse.issparse(matrix):
        values = matrix.toarray()
    else:
        values = np.asarray(matrix)
    return np.asarray(values, dtype=float).ravel()


def read_spatial_coordinates(path: Path) -> np.ndarray:
    data = ad.read_h5ad(path, backed="r")
    try:
        if "spatial" in data.obsm:
            coords = np.asarray(data.obsm["spatial"])
        elif {"x", "y"}.issubset(data.obs.columns):
            coords = data.obs[["x", "y"]].to_numpy()
        else:
            raise ValueError(f"Cannot find spatial coordinates in {path}")
        if coords.shape[1] < 2:
            raise ValueError(f"Spatial coordinate matrix has shape {coords.shape}: {path}")
        return coords[:, :2].astype(float)
    finally:
        data.file.close()


# %%
def minimal_spatial_adata(path: Path, gene: str, vmax: float, point_cap: int) -> ad.AnnData:
    coords = read_spatial_coordinates(path)
    expr = np.log1p(read_gene_vector(path, gene))

    n_obs = len(expr)
    if point_cap > 0 and n_obs > point_cap:
        rng = np.random.default_rng(2026)
        keep = np.sort(rng.choice(n_obs, size=point_cap, replace=False))
        coords = coords[keep]
        expr = expr[keep]

    obs = pd.DataFrame(
        {"_plot_expr": expr},
        index=pd.Index([f"spot_{idx}" for idx in range(len(expr))], name="spot_id"),
    )
    plot_data = ad.AnnData(obs=obs)
    plot_data.obsm["X_spatial"] = coords
    plot_data.uns["_plot_vmax"] = float(vmax)
    return plot_data


def clean_spatial_axis(ax: plt.Axes) -> None:
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")


def point_size_for_n_obs(n_obs: int, platform: str, scale: float = POINT_SIZE_SCALE) -> float:
    if n_obs > 250_000:
        base_size = 0.08
    elif n_obs > 80_000:
        base_size = 0.12
    elif n_obs > 35_000:
        base_size = 0.22
    elif n_obs > 15_000:
        base_size = 0.35
    elif n_obs > 7_000:
        base_size = 0.55
    else:
        base_size = 1.6
    platform_mult = PLATFORM_POINT_MULT.get(platform, 1.0)
    return base_size * scale * platform_mult


# %%
def build_panel_config() -> pd.DataFrame:
    """Compute per-panel vmax from the three h5ads (ref + 2 FEAST) for each spec."""
    rows = []
    for spec in PANEL_SPECS:
        sample = spec["sample"]
        gene = spec["gene"]
        paths = [h5ad_path_for("Reference", sample)] + [h5ad_path_for(m, sample) for m in FEAST_METHODS]
        expr_values = [np.log1p(read_gene_vector(p, gene)) for p in paths]
        combined = np.concatenate(expr_values)
        vmax = float(np.nanpercentile(combined, 99.5))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(np.nanmax(combined))

        rows.append({
            "platform": spec["platform"],
            "dataset": spec["dataset"],
            "sample": sample,
            "gene": gene,
            "reference_h5ad": str(h5ad_path_for("Reference", sample)),
            "feast_rank_h5ad": str(h5ad_path_for("FEAST_Rank", sample)),
            "feast_ot_spatial_h5ad": str(h5ad_path_for("FEAST_OT_Spatial", sample)),
            "vmax_log1p_p99_5": vmax,
        })
    return pd.DataFrame(rows)


def draw_figure(
    panel_config: pd.DataFrame,
    output_dir: Path,
    output_stem: str,
    dpi: int,
    point_cap: int,
    point_size_scale: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    sc.settings.set_figure_params(
        dpi=150, dpi_save=dpi, frameon=False, vector_friendly=True, facecolor="white",
    )
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 9,
    })

    rows = ["Reference", "FEAST_Rank", "FEAST_OT_Spatial"]
    row_labels = ["Real\nReference", "FEAST\nRank", "FEAST OT\nSpatial"]
    n_cols = len(panel_config)
    fig, axes = plt.subplots(
        len(rows), n_cols,
        figsize=(2.55 * n_cols, 6.65),
        constrained_layout=False,
    )

    for col_idx, panel in enumerate(panel_config.itertuples(index=False)):
        col_vmax = float(panel.vmax_log1p_p99_5)
        for row_idx, panel_kind in enumerate(rows):
            path = h5ad_path_for(panel_kind, panel.sample)
            plot_data = minimal_spatial_adata(path, panel.gene, vmax=col_vmax, point_cap=point_cap)
            ax = axes[row_idx, col_idx]
            sc.pl.embedding(
                plot_data,
                basis="spatial",
                color="_plot_expr",
                ax=ax,
                show=False,
                title="",
                size=point_size_for_n_obs(plot_data.n_obs, platform=panel.platform, scale=point_size_scale),
                cmap="viridis",
                vmin=0,
                vmax=col_vmax,
                colorbar_loc=None,
                frameon=False,
                sort_order=True,
            )
            clean_spatial_axis(ax)

        top_ax = axes[0, col_idx]
        top_ax.text(
            0.5, 1.33,
            f"{panel.dataset}\n({panel.platform})",
            transform=top_ax.transAxes,
            ha="center", va="bottom",
            fontsize=10, linespacing=1.12,
        )
        top_ax.plot(
            [0.16, 0.84], [1.26, 1.26],
            transform=top_ax.transAxes,
            color="black", linewidth=0.8, clip_on=False,
        )
        top_ax.text(
            0.5, 1.10, panel.gene,
            transform=top_ax.transAxes,
            ha="center", va="bottom",
            fontsize=9, fontstyle="italic",
        )

    plt.subplots_adjust(
        left=0.075, right=0.98, top=0.79, bottom=0.075,
        wspace=0.04, hspace=0.06,
    )

    for row_idx, label in enumerate(row_labels):
        bbox = axes[row_idx, 0].get_position()
        y_center = bbox.y0 + bbox.height / 2
        fig.text(
            0.012, y_center, label,
            rotation=90, ha="center", va="center",
            fontsize=11, fontweight="bold",
        )

    for suffix, kwargs in {"pdf": {}, "svg": {}, "png": {"dpi": dpi}}.items():
        path = output_dir / f"{output_stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"Saved: {path}")

    plt.close(fig)


# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-stem", default=OUTPUT_STEM)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--point-cap", type=int, default=0,
        help="Maximum spots plotted per panel; 0 disables downsampling.",
    )
    parser.add_argument(
        "--point-size-scale", type=float, default=POINT_SIZE_SCALE,
        help="Global multiplier for Scanpy spatial marker area.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel_config = build_panel_config()
    config_path = args.output_dir / f"{args.output_stem}_panel_config.csv"
    panel_config.to_csv(config_path, index=False)
    print(f"Saved: {config_path}")

    draw_figure(
        panel_config=panel_config,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        dpi=args.dpi,
        point_cap=args.point_cap,
        point_size_scale=args.point_size_scale,
    )
    return 0


# %%
if __name__ == "__main__":
    raise SystemExit(main())
