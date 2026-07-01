# %%
"""Spatial domain panel across alteration factors — one figure per alteration type.

Each figure: 2 rows (GraphST, STAGATE) × 4 fold changes (0.5, 1.0, 1.5, 2.0)
on DLPFC slice 151676.
"""

# %%
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc

# %%
EXPERIMENT_ROOT = Path("/maiziezhou_lab2/yiru/FEAST_experiments")
CLUSTERING_METHODS_DIR = EXPERIMENT_ROOT / "01_clustering" / "outputs" / "methods"
OUTPUT_DIR = Path("/maiziezhou_lab2/yiru/Reproduce/Visualization/figures/clustering")

SLICE_ID = "151676"
# Reverse map from directory suffix → fold_change for non-baseline dirs
_DIR_TO_FOLD = {
    "0_50": 0.5, "0_67": 0.67, "0_80": 0.8,
    "1_25": 1.25, "1_50": 1.5, "2": 2.0,
    "neg_0_25": -0.25, "neg_0_50": -0.5, "neg_1": -1.0,
    "pos_0_25": 0.25, "pos_0_50": 0.5, "pos_1": 1.0,
}

def available_folds(alteration_type: str) -> list[float]:
    """Return sorted fold_changes available on disk for this alteration type."""
    method_dir = CLUSTERING_METHODS_DIR / "GraphST" / SLICE_ID
    prefix = f"{alteration_type}_"
    folds = []
    for entry in method_dir.iterdir():
        if entry.name == "baseline":
            folds.append(1.0)
        elif entry.name.startswith(prefix):
            suffix = entry.name[len(prefix):]
            if suffix in _DIR_TO_FOLD:
                folds.append(_DIR_TO_FOLD[suffix])
    # Also check that the STAGATE equivalent exists
    folds = [f for f in folds
             if (CLUSTERING_METHODS_DIR / "STAGATE_mclust" / SLICE_ID / sim_id_for(alteration_type, f)).exists()]
    return sorted(folds)

ALTERATION_TYPES = ["mean", "variance", "sparsity"]

METHOD_SPECS = [
    ("GraphST", "domain", "GraphST"),
    ("STAGATE_mclust", "mclust", "STAGATE"),
]

DOMAIN_PALETTE = [
    "#fb8072", "#80b1d3", "#8dd3c7", "#bebada",
    "#fdb462", "#b3b3b3", "#fccde5", "#ccebc5",
]

_FOLD_TO_SUFFIX = {
    0.5: "0_50", 0.67: "0_67", 0.8: "0_80",
    1.25: "1_25", 1.5: "1_50", 2.0: "2",
}

_SPARSITY_SUFFIX = {
    -0.25: "neg_0_25", -0.5: "neg_0_50", -1.0: "neg_1",
    0.25: "pos_0_25", 0.5: "pos_0_50", 1.0: "pos_1",
}


def sim_id_for(alteration_type: str, fold: float) -> str:
    if fold == 1.0:
        return "baseline"
    if alteration_type == "sparsity":
        suffix = _SPARSITY_SUFFIX[fold]
    else:
        suffix = _FOLD_TO_SUFFIX[fold]
    return f"{alteration_type}_{suffix}"


def result_h5ad_path(method: str, alter_type: str, fold: float) -> Path:
    return CLUSTERING_METHODS_DIR / method / SLICE_ID / sim_id_for(alter_type, fold) / "result.h5ad"


def ensure_spatial_basis(adata):
    if "X_spatial" not in adata.obsm:
        if "spatial" in adata.obsm:
            adata.obsm["X_spatial"] = adata.obsm["spatial"]
        else:
            raise ValueError("Cannot find spatial coordinates in obsm")
    return adata


def clean_spatial_axis(ax):
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.axis("off")


# %%
def draw_one_figure(alteration_type: str, output_dir: Path, dpi: int, point_size: float) -> None:
    fold_changes = available_folds(alteration_type)
    n_cols = len(fold_changes)

    for method_dir, label_key, method_label in METHOD_SPECS:
        n_cols_per_row = (n_cols + 1) // 2
        n_rows = 2 if n_cols > 2 else 1
        fig, axes = plt.subplots(n_rows, n_cols_per_row, figsize=(2.8 * n_cols_per_row, 3.1 * n_rows))
        axes = axes.flatten()

        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        })

        for col_idx, fold in enumerate(fold_changes):
            ax = axes[col_idx]
            path = result_h5ad_path(method_dir, alteration_type, fold)

            if not path.exists():
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="#999")
                clean_spatial_axis(ax)
                continue

            adata = sc.read_h5ad(path)
            adata = ensure_spatial_basis(adata)

            if label_key not in adata.obs.columns:
                ax.text(0.5, 0.5, "no labels", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="#999")
                clean_spatial_axis(ax)
                continue

            adata.obs[label_key] = adata.obs[label_key].astype(str).astype("category")
            n_cats = adata.obs[label_key].nunique()

            sc.pl.embedding(
                adata,
                basis="spatial",
                color=label_key,
                ax=ax,
                show=False,
                title="",
                size=point_size,
                palette=DOMAIN_PALETTE[:n_cats],
                legend_loc=None,
                frameon=False,
                sort_order=False,
            )
            clean_spatial_axis(ax)

            # Fold change label at top
            fc_label = f"FC = {fold:.2f}" if fold != 1.0 else "FC = 1.00 (baseline)"
            ax.set_title(
                fc_label, fontsize=16, fontweight="bold", pad=6,
            )

        # Hide unused axes
        for extra_ax in axes[len(fold_changes):]:
            extra_ax.axis("off")

        # Count domains from baseline
        baseline_path = result_h5ad_path(method_dir, alteration_type, 1.0)
        n_domains = "?"
        if baseline_path.exists():
            b = sc.read_h5ad(baseline_path)
            if label_key in b.obs.columns:
                n_domains = str(b.obs[label_key].nunique())

        axes[0].set_ylabel(
            f"{method_label}\n{n_domains} domains",
            fontsize=18, rotation=0, ha="right", va="center",
            labelpad=35,
        )

        plt.subplots_adjust(
            left=0.14, right=0.98, top=0.90, bottom=0.04,
            wspace=0.04,
        )

        stem = f"spatial_{alteration_type}_alteration_panel_{method_label.lower()}"
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
    parser.add_argument("--point-size", type=float, default=12.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sc.settings.set_figure_params(
        dpi=150, dpi_save=args.dpi, frameon=False, vector_friendly=True,
    )

    for atype in ALTERATION_TYPES:
        draw_one_figure(atype, args.output_dir, args.dpi, args.point_size)

    return 0


# %%
if __name__ == "__main__":
    raise SystemExit(main())
