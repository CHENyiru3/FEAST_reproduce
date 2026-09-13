#!/usr/bin/env python3
"""Spatial delta maps: simulated − reference expression per spot.

6 platforms × 3 simulators (FEAST, SRTsim, scCube) grid.
Δ = log1p(sim) − log1p(ref) at every spot. A shared linear scale is used per
row: blue = under, light gray = match, red = over.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import sparse


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_STEM = "simulator_delta_spatial"

REFERENCE_DIR = REPOSITORY_ROOT / "00_simulator_benchmark" / "data" / "local" / "reference"
FEAST_SIM_DIR = (
    REPOSITORY_ROOT
    / "00_simulator_benchmark"
    / "outputs"
    / "final_rerun_20260718"
    / "simulations"
)
SRTSIM_DIR = REPOSITORY_ROOT / "00_simulator_benchmark" / "data" / "local" / "external" / "SRTsim"
SCCUBE_DIR = REPOSITORY_ROOT / "00_simulator_benchmark" / "data" / "local" / "external" / "scCube"

SIMULATORS = ["FEAST", "SRTsim", "scCube"]
SIM_LABELS = ["FEAST", "SRTsim", "scCube"]

PANEL_SPECS = [
    {"sample": "DLPFC_151670",         "platform": "10X Visium",    "gene": "SCGB2A2"},
    {"sample": "MERFISH_120",          "platform": "MERFISH",       "gene": "Pvalb"},
    {"sample": "OpenST_006",           "platform": "OpenST",        "gene": "IGKC"},
    {"sample": "Slideseq_001",         "platform": "Slide-seq V2",  "gene": "IGLC2"},
    {"sample": "Stereoseq_E12_5_E2S1", "platform": "Stereo-seq",    "gene": "Fabp7"},
    {"sample": "Xenium_LymphNode",     "platform": "Xenium",        "gene": "CD3E"},
]

PLATFORM_POINT_SIZE = {
    "10X Visium": 1.8, "MERFISH": 1.2, "Slide-seq V2": 1.5,
    "Stereo-seq": 0.8, "OpenST": 0.8, "Xenium": 0.5,
}
POINT_CAP = 20_000
DELTA_CMAP = LinearSegmentedColormap.from_list(
    "feast_delta_linear",
    [
        (0.00, "#2166AC"),
        (0.25, "#67A9CF"),
        (0.50, "#E0E0E0"),
        (0.75, "#EF8A62"),
        (1.00, "#B2182B"),
    ],
    N=256,
)


def _sim_path(sim: str, sample: str) -> Path:
    if sim == "FEAST":
        return FEAST_SIM_DIR / f"{sample}.h5ad"
    if sim == "SRTsim":
        return SRTSIM_DIR / f"{sample}.h5ad"
    if sim == "scCube":
        return SCCUBE_DIR / f"{sample}.h5ad"
    raise ValueError(f"Unknown simulator: {sim}")


def _read_gene(path: Path, gene: str) -> np.ndarray:
    data = ad.read_h5ad(path, backed="r")
    try:
        if gene not in data.var_names:
            raise ValueError(f"{gene!r} not in {path}")
        vals = data[:, gene].X
        if sparse.issparse(vals):
            vals = vals.toarray()
        return np.asarray(vals, dtype=float).ravel()
    finally:
        data.file.close()


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


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def draw_figure(output_dir: Path, output_stem: str, dpi: int) -> list[Path]:
    configure_matplotlib()
    n_rows = len(PANEL_SPECS)
    n_cols = len(SIMULATORS)
    normalized_delta_norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.0 * n_cols, 3.4 * n_rows),
    )

    for row_idx, spec in enumerate(PANEL_SPECS):
        sample = spec["sample"]
        gene = spec["gene"]
        platform = spec["platform"]

        ref_path = REFERENCE_DIR / f"{sample}.h5ad"
        ref_expr = np.log1p(_read_gene(ref_path, gene))
        xy = _read_xy(ref_path)

        deltas = []
        for sim in SIMULATORS:
            sim_path = _sim_path(sim, sample)
            sim_expr = np.log1p(_read_gene(sim_path, gene))
            deltas.append(sim_expr - ref_expr)

        # shared symmetric vrange per row
        all_delta = np.concatenate(deltas)
        vmax = float(np.nanpercentile(np.abs(all_delta), 99))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 0.5
        # downsample large datasets
        n_obs = len(ref_expr)
        if POINT_CAP > 0 and n_obs > POINT_CAP:
            rng = np.random.default_rng(2026)
            keep = np.sort(rng.choice(n_obs, size=POINT_CAP, replace=False))
            xy_plot = xy[keep]
        else:
            keep = slice(None)
            xy_plot = xy

        pt_size = PLATFORM_POINT_SIZE.get(platform, 1.0)

        for col_idx, sim_label in enumerate(SIM_LABELS):
            ax = axes[row_idx, col_idx]
            delta = deltas[col_idx]
            if isinstance(keep, np.ndarray):
                delta = delta[keep]

            order = np.argsort(np.abs(delta))
            ax.scatter(
                xy_plot[order, 0], xy_plot[order, 1],
                c=(delta[order] / vmax), cmap=DELTA_CMAP, norm=normalized_delta_norm,
                s=pt_size, linewidths=0, alpha=0.85,
                rasterized=True,
            )
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.axis("off")

        # row label
        axes[row_idx, 0].text(
            -0.24, 0.5,
            f"{platform}\n{gene}\nq99 ±{vmax:.2g}",
            transform=axes[row_idx, 0].transAxes,
            ha="right", va="center",
            fontsize=12, fontweight="bold",
        )

    for col_idx, sim_label in enumerate(SIM_LABELS):
        axes[0, col_idx].set_title(sim_label, fontsize=15, fontweight="bold", pad=12)

    plt.subplots_adjust(
        left=0.10, right=0.98, top=0.94, bottom=0.06,
        wspace=0.05, hspace=0.06,
    )

    cbar_ax = fig.add_axes([0.22, 0.015, 0.56, 0.012])
    sm = plt.cm.ScalarMappable(cmap=DELTA_CMAP, norm=normalized_delta_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(
        "Row-normalized Δ = [log1p(simulated) − log1p(reference)] / q99(|Δ|)",
        fontsize=12, fontweight="bold",
    )
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(["−1", "−0.5", "0", "0.5", "1"])
    cbar.ax.tick_params(labelsize=10.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / output_stem
    paths: list[Path] = []
    for suffix, kwargs in {
        "pdf": {"metadata": {"Creator": "FEAST Study 00 delta spatial"}},
        "svg": {"metadata": {"Date": None, "Creator": "FEAST Study 00 delta spatial"}},
        "png": {"dpi": dpi, "metadata": {"Software": "FEAST Study 00 delta spatial"}},
    }.items():
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"Saved: {path}")
        paths.append(path)
    plt.close(fig)
    return paths


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


if __name__ == "__main__":
    raise SystemExit(main())
