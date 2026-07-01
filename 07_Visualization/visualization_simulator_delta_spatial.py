# %%
"""Spatial delta maps: simulated − reference expression per spot.

5 platforms × 4 simulators grid.  Each panel shows Δ = log1p(sim) − log1p(ref)
at every spot, using a diverging RdBu colormap (blue = under, white = match,
red = over).
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
from matplotlib.colors import TwoSlopeNorm
from scipy import sparse

# %%
EXPERIMENT_ROOT = Path("/maiziezhou_lab2/yiru/FEAST_experiments")
BENCHMARK_ROOT = EXPERIMENT_ROOT / "00_simulator_benchmark"
REFERENCE_DIR = BENCHMARK_ROOT / "exper_data"
SIM_DIR = BENCHMARK_ROOT / "outputs" / "simulation_saved"

OUTPUT_DIR = Path("/maiziezhou_lab2/yiru/Reproduce/Visualization/figures/simulator_benchmark")
OUTPUT_STEM = "simulator_delta_spatial"

SIMULATORS = ["scCube", "SRTsim", "FEAST_Rank", "FEAST_OT_Spatial"]
SIM_LABELS = ["scCube", "SRTsim", "FEAST Rank", "FEAST OT Spatial"]
SIM_DIRS = {
    "scCube": SIM_DIR / "sccube",
    "SRTsim": SIM_DIR / "srtsim_updated",
    "FEAST_Rank": SIM_DIR / "FEAST_Rank",
    "FEAST_OT_Spatial": SIM_DIR / "FEAST_OT_Spatial",
}

ROWS = [
    {"sample": "DLPFC_151670",        "platform": "10X Visium",   "gene": "SCGB2A2"},
    {"sample": "MERFISH_120",         "platform": "MERFISH",      "gene": "Pvalb"},
    {"sample": "OpenST_005",          "platform": "OpenST",       "gene": "IGKC"},
    {"sample": "Stereoseq_E12_5_E2S1","platform": "Stereo-seq",   "gene": "Fabp7"},
    {"sample": "Xenium_LymphNode",    "platform": "Xenium",       "gene": "CD3E"},
]

PLATFORM_POINT_SIZE = {
    "10X Visium": 2.0, "MERFISH": 1.4, "OpenST": 0.9,
    "Slide-seq V2": 1.6, "Stereo-seq": 0.9, "Xenium": 0.6,
}


# %%
def _sim_path(sim: str, sample: str) -> Path:
    if sim == "scCube":
        return SIM_DIRS[sim] / f"{sample}_sccube_sim.h5ad"
    return SIM_DIRS[sim] / f"{sample}.h5ad"


def _read_gene(path: Path, gene: str) -> np.ndarray:
    data = ad.read_h5ad(path)
    vals = data[:, gene].X
    if sparse.issparse(vals):
        vals = vals.toarray()
    return np.asarray(vals, dtype=float).ravel()


def _read_xy(path: Path) -> np.ndarray:
    data = ad.read_h5ad(path, backed="r")
    try:
        if "spatial" in data.obsm:
            coords = np.asarray(data.obsm["spatial"])
        elif {"x", "y"}.issubset(data.obs.columns):
            coords = data.obs[["x", "y"]].to_numpy()
        else:
            raise ValueError(f"No spatial coords in {path}")
        return coords[:, :2].astype(float)
    finally:
        data.file.close()


# %%
def draw_figure(output_dir: Path, output_stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.size": 9,
    })

    n_rows = len(ROWS)
    n_cols = len(SIMULATORS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.2 * n_cols, 2.9 * n_rows),
        constrained_layout=False,
    )

    for row_idx, row_spec in enumerate(ROWS):
        sample = row_spec["sample"]
        gene = row_spec["gene"]
        platform = row_spec["platform"]

        ref_path = REFERENCE_DIR / f"{sample}.h5ad"
        ref_expr = np.log1p(_read_gene(ref_path, gene))
        xy = _read_xy(ref_path)

        # Compute delta for all 4 simulators
        deltas = []
        for sim in SIMULATORS:
            sim_path = _sim_path(sim, sample)
            sim_expr = np.log1p(_read_gene(sim_path, gene))
            delta = sim_expr - ref_expr
            deltas.append(delta)

        # Shared symmetric vrange per row
        all_delta = np.concatenate(deltas)
        vmax = float(np.nanpercentile(np.abs(all_delta), 99))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 0.5
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        # Subsample large datasets for plotting speed
        n_obs = len(ref_expr)
        point_cap = 80000
        if n_obs > point_cap:
            rng = np.random.default_rng(2026)
            keep = np.sort(rng.choice(n_obs, size=point_cap, replace=False))
            xy_plot = xy[keep]
        else:
            keep = slice(None)
            xy_plot = xy

        pt_size = PLATFORM_POINT_SIZE.get(platform, 1.0)

        for col_idx, (sim, sim_label) in enumerate(zip(SIMULATORS, SIM_LABELS)):
            ax = axes[row_idx, col_idx]
            delta = deltas[col_idx]
            if isinstance(keep, np.ndarray):
                delta = delta[keep]

            # Sort by |delta| so largest deviations plot on top
            order = np.argsort(np.abs(delta))
            ax.scatter(
                xy_plot[order, 0], xy_plot[order, 1],
                c=delta[order], cmap="RdBu_r", norm=norm,
                s=pt_size, linewidths=0, alpha=0.85,
            )

            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.axis("off")

        # Row label
        row_label = f"{platform}\n{gene}"
        axes[row_idx, 0].text(
            -0.18, 0.5, row_label,
            transform=axes[row_idx, 0].transAxes,
            ha="right", va="center", fontsize=11, fontweight="bold",
            rotation=0,
        )

    # Column titles
    for col_idx, sim_label in enumerate(SIM_LABELS):
        axes[0, col_idx].set_title(sim_label, fontsize=13, fontweight="bold", pad=10)

    plt.subplots_adjust(
        left=0.08, right=0.98, top=0.93, bottom=0.03,
        wspace=0.06, hspace=0.08,
    )

    # Single colorbar
    cbar_ax = fig.add_axes([0.15, 0.01, 0.70, 0.012])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Δ = log1p(simulated) − log1p(reference)", fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    draw_figure(args.output_dir, args.output_stem, args.dpi)
    return 0


# %%
if __name__ == "__main__":
    raise SystemExit(main())
