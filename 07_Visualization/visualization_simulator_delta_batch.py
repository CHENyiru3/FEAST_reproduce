# %%
"""Batch spatial delta maps — multiple genes per sample, one figure per sample.

Each figure: rows = genes, columns = simulators
Output dir: simulator_delta/
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

OUTPUT_DIR = Path("/maiziezhou_lab2/yiru/Reproduce/Visualization/figures/simulator_benchmark/simulator_delta")

SIM_ORDER = ["FEAST_Rank", "FEAST_OT_Spatial", "SRTsim", "scCube"]
SIM_LABELS = ["FEAST Rank", "FEAST OT Spatial", "SRTsim", "scCube"]
SIM_DIRS = {
    "scCube": SIM_DIR / "sccube",
    "SRTsim": SIM_DIR / "srtsim_updated",
    "FEAST_Rank": SIM_DIR / "FEAST_Rank",
    "FEAST_OT_Spatial": SIM_DIR / "FEAST_OT_Spatial",
}

SAMPLE_CONFIGS = [
    {
        "sample": "DLPFC_151670",
        "platform": "10X Visium",
        "genes": ["SCGB2A2", "MB", "HPCAL1", "PCP4"],
    },
    {
        "sample": "MERFISH_120",
        "platform": "MERFISH",
        "genes": ["Pvalb", "Cldn11", "Nefh", "Slc32a1"],
    },
    {
        "sample": "OpenST_005",
        "platform": "OpenST",
        "genes": ["IGKC", "MS4A1", "CD3E", "CD79A"],
    },
    {
        "sample": "Stereoseq_E12_5_E2S1",
        "platform": "Stereo-seq",
        "genes": ["Fabp7", "Sox2", "Vim", "Tubb3"],
    },
    {
        "sample": "Xenium_LymphNode",
        "platform": "Xenium",
        "genes": ["CD3E", "MS4A1", "CXCR4", "CD79A"],
    },
]

PLATFORM_POINT_SIZE = {
    "10X Visium": 2.0, "MERFISH": 1.4, "OpenST": 0.9,
    "Stereo-seq": 0.9, "Xenium": 0.6,
}
POINT_CAP = 80000


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
def draw_one_sample(
    sample_cfg: dict,
    output_dir: Path,
    dpi: int,
) -> None:
    sample = sample_cfg["sample"]
    platform = sample_cfg["platform"]
    genes = sample_cfg["genes"]
    n_genes = len(genes)
    n_sims = len(SIM_ORDER)

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.size": 9,
    })

    fig, axes = plt.subplots(
        n_genes, n_sims,
        figsize=(3.0 * n_sims, 2.7 * n_genes),
        constrained_layout=False,
    )
    if n_genes == 1:
        axes = axes.reshape(1, -1)
    if n_sims == 1:
        axes = axes.reshape(-1, 1)

    ref_path = REFERENCE_DIR / f"{sample}.h5ad"
    xy = _read_xy(ref_path)

    # Subsample for large datasets
    n_obs = xy.shape[0]
    if n_obs > POINT_CAP:
        rng = np.random.default_rng(2026)
        keep = np.sort(rng.choice(n_obs, size=POINT_CAP, replace=False))
        xy_plot = xy[keep]
    else:
        keep = slice(None)
        xy_plot = xy

    pt_size = PLATFORM_POINT_SIZE.get(platform, 1.0)

    for row_idx, gene in enumerate(genes):
        # Read reference and all simulator expressions for this gene
        ref_expr = np.log1p(_read_gene(ref_path, gene))

        deltas = []
        for sim in SIM_ORDER:
            sim_path = _sim_path(sim, sample)
            sim_expr = np.log1p(_read_gene(sim_path, gene))
            delta = sim_expr - ref_expr
            deltas.append(delta)

        # Shared symmetric vrange per row
        all_delta = np.concatenate(deltas)
        vmax = float(np.nanpercentile(np.abs(all_delta), 99))
        if not np.isfinite(vmax) or vmax <= 0.1:
            vmax = 0.5
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        for col_idx, (sim, sim_label) in enumerate(zip(SIM_ORDER, SIM_LABELS)):
            ax = axes[row_idx, col_idx]
            delta = deltas[col_idx]
            if isinstance(keep, np.ndarray):
                delta = delta[keep]

            order = np.argsort(np.abs(delta))
            ax.scatter(
                xy_plot[order, 0], xy_plot[order, 1],
                c=delta[order], cmap="RdBu_r", norm=norm,
                s=pt_size, linewidths=0, alpha=0.85,
            )
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.axis("off")

        # Gene label
        axes[row_idx, 0].text(
            -0.22, 0.5, gene,
            transform=axes[row_idx, 0].transAxes,
            ha="right", va="center", fontsize=11, fontweight="bold",
            fontstyle="italic",
        )

    # Column titles
    for col_idx, sim_label in enumerate(SIM_LABELS):
        axes[0, col_idx].set_title(sim_label, fontsize=13, fontweight="bold", pad=10)

    # Suptitle
    fig.suptitle(f"{platform}  —  {sample}", fontsize=14, fontweight="bold", y=0.98)

    plt.subplots_adjust(
        left=0.09, right=0.98, top=0.91, bottom=0.06,
        wspace=0.06, hspace=0.10,
    )

    # Colorbar
    cbar_ax = fig.add_axes([0.18, 0.015, 0.64, 0.015])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Δ = log1p(simulated) − log1p(reference)", fontsize=9, fontweight="bold")
    cbar.ax.tick_params(labelsize=7)

    stem = f"delta_{sample}"
    for suffix, kwargs in {"pdf": {}, "svg": {}, "png": {"dpi": dpi}}.items():
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"Saved: {path}")

    plt.close(fig)


# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cfg in SAMPLE_CONFIGS:
        draw_one_sample(cfg, args.output_dir, args.dpi)

    return 0


# %%
if __name__ == "__main__":
    raise SystemExit(main())
