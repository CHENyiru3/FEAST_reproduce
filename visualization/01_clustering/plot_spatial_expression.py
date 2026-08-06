#!/usr/bin/env python3
"""Render a shared-scale spatial-expression comparison for Study 01.

The representative DLPFC slice is 151676.  Every panel shows the same marker
gene and uses one log1p-expression scale, so colour differences can be read
directly across the reference, FEAST baseline, and altered simulations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_STEM = "alteration_spatial_expression"

REFERENCE_DIR = REPOSITORY_ROOT / "01_clustering" / "data" / "local"
SIM_DIR = (
    REPOSITORY_ROOT
    / "01_clustering"
    / "outputs"
    / "final_rerun_20260718"
    / "simulations"
    / "simulations"
    / "151676"
)

SLICE = "151676"
GENE = "MBP"

COLUMN_SPECS = (
    ("reference", "Observed\nreference"),
    ("baseline", "FEAST\nbaseline"),
    ("decrease", "Decreased\ncondition"),
    ("increase", "Increased\ncondition"),
)

# (row label, setting detail, decreased-simulation ID, increased-simulation ID)
ROW_SPECS = (
    ("Mean", "fold change 0.50 / 2.00", "mean_0_50", "mean_2"),
    ("Variance", "fold change 0.50 / 2.00", "variance_0_50", "variance_2"),
    ("Zero fraction", "logit shift −1.0 / +1.0", "sparsity_neg_1", "sparsity_pos_1"),
)

# Matplotlib scatter area; this is a further 15% diameter increase from the
# prior spatial-panel setting while preserving visible spot boundaries.
POINT_SIZE = 9.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            raise ValueError(f"No spatial coordinates in {path}")
        return coords[:, :2].astype(float)
    finally:
        data.file.close()


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "feast-study01-spatial-expression-v2",
    })


def _path_for(column_key: str, decrease_id: str, increase_id: str) -> Path:
    if column_key == "reference":
        return REFERENCE_DIR / f"{SLICE}.h5ad"
    if column_key == "baseline":
        return SIM_DIR / "baseline.h5ad"
    if column_key == "decrease":
        return SIM_DIR / f"{decrease_id}.h5ad"
    if column_key == "increase":
        return SIM_DIR / f"{increase_id}.h5ad"
    raise ValueError(f"Unknown column key: {column_key}")


def draw_figure(output_dir: Path, output_stem: str, dpi: int) -> list[Path]:
    configure_matplotlib()
    ref_path = REFERENCE_DIR / f"{SLICE}.h5ad"
    xy = _read_xy(ref_path)

    expr_by_panel: dict[tuple[str, str], np.ndarray] = {}
    for row_label, _detail, decrease_id, increase_id in ROW_SPECS:
        for column_key, _column_label in COLUMN_SPECS:
            path = _path_for(column_key, decrease_id, increase_id)
            expr_by_panel[(row_label, column_key)] = np.log1p(_read_gene(path, GENE))

    pooled = np.concatenate(list(expr_by_panel.values()))
    vmax = float(np.nanpercentile(pooled, 99.5))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(pooled))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    n_rows = len(ROW_SPECS)
    n_cols = len(COLUMN_SPECS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.8, 8.9))
    scatter = None
    for row_idx, (row_label, detail, _decrease_id, _increase_id) in enumerate(ROW_SPECS):
        for col_idx, (column_key, _column_label) in enumerate(COLUMN_SPECS):
            ax = axes[row_idx, col_idx]
            scatter = ax.scatter(
                xy[:, 0], xy[:, 1],
                s=POINT_SIZE,
                c=expr_by_panel[(row_label, column_key)],
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

        axes[row_idx, 0].text(
            -0.24,
            0.56,
            row_label,
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        axes[row_idx, 0].text(
            -0.24,
            0.37,
            detail,
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8.5,
            color="#666666",
        )

    for col_idx, (_column_key, column_label) in enumerate(COLUMN_SPECS):
        axes[0, col_idx].set_title(column_label, fontsize=12, fontweight="bold", pad=9)

    fig.suptitle(
        f"{GENE} expression — DLPFC slice {SLICE}",
        fontsize=14,
        fontweight="bold",
        x=0.47,
        y=0.985,
    )
    cbar = fig.colorbar(scatter, ax=axes, fraction=0.023, pad=0.018)
    cbar.set_label(
        f"log1p({GENE} expression)\nshared 0–99.5th percentile scale",
        fontsize=10,
        labelpad=10,
    )
    cbar.ax.tick_params(labelsize=9)

    fig.subplots_adjust(
        left=0.17,
        right=0.91,
        top=0.90,
        bottom=0.055,
        wspace=0.025,
        hspace=0.055,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / output_stem
    paths: list[Path] = []
    for suffix, kwargs in {
        "pdf": {"metadata": {"Creator": "FEAST Study 01 spatial expression"}},
        "png": {"dpi": dpi, "metadata": {"Software": "FEAST Study 01 spatial expression"}},
        "svg": {"metadata": {"Creator": "FEAST Study 01 spatial expression"}},
    }.items():
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, **kwargs)
        print(f"Saved: {path}")
        paths.append(path)
    plt.close(fig)
    return paths


def write_provenance(output_dir: Path, outputs: list[Path]) -> Path:
    input_paths = [REFERENCE_DIR / f"{SLICE}.h5ad", SIM_DIR / "baseline.h5ad"]
    for _row_label, _detail, decrease_id, increase_id in ROW_SPECS:
        input_paths.extend((SIM_DIR / f"{decrease_id}.h5ad", SIM_DIR / f"{increase_id}.h5ad"))
    payload = {
        "schema_version": 1,
        "figure_set": "study01_spatial_expression",
        "display_slice": SLICE,
        "marker_gene": GENE,
        "normalization": "Shared log1p scale, vmin=0, pooled 99.5th percentile vmax.",
        "source": {
            "plot_script": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
            "plot_script_sha256": sha256(Path(__file__)),
            "inputs": {
                str(path.resolve().relative_to(REPOSITORY_ROOT)): sha256(path)
                for path in input_paths
            },
        },
        "outputs": {path.name: sha256(path) for path in sorted(outputs)},
        "scientific_disposition": {
            "figure_promotion_authorized": False,
            "supported_interpretation": "Descriptive spatial comparison of a registered representative slice.",
        },
    }
    output_path = output_dir / "spatial_expression_provenance.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-stem", default=OUTPUT_STEM)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = draw_figure(args.output_dir, args.output_stem, args.dpi)
    print(f"Saved: {write_provenance(args.output_dir, outputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
