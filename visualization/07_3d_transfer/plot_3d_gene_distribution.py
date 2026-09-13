#!/usr/bin/env python3
"""Render Study 07 full-volume geometry and selected-gene 3D distributions."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import Normalize


REPRO_ROOT = Path(__file__).resolve().parents[2]
FINAL_ROOT = REPRO_ROOT / "07_3d_transfer" / "outputs" / "final"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "figures"
GENES = ("Slc17a7", "Gad2", "Gfap", "Reln")
AGES = ("E15.5", "E18.5")
SAMPLE_CAP_PER_SLICE = 700
SAMPLE_SEED = 2026
VIEWS = (("Oblique view", 23.0, -55.0), ("Lateral view", 12.0, 0.0))


def configure_matplotlib() -> None:
    arial_paths = (
        Path(sys.prefix) / "fonts" / "arial.ttf",
        Path(sys.prefix) / "fonts" / "arialbd.ttf",
    )
    if not all(path.is_file() for path in arial_paths):
        raise FileNotFoundError(f"Arial Regular and Bold are required: {arial_paths}")
    for path in arial_paths:
        font_manager.fontManager.addfont(path)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=FINAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def sampled_indices(size: int, seed: int) -> np.ndarray:
    if size <= SAMPLE_CAP_PER_SLICE:
        return np.arange(size)
    return np.sort(
        np.random.default_rng(seed).choice(
            size, size=SAMPLE_CAP_PER_SLICE, replace=False
        )
    )


def require_inputs() -> tuple[dict, dict[str, pd.DataFrame]]:
    validation = json.loads(
        (FINAL_ROOT / "validation.json").read_text(encoding="utf-8")
    )
    if validation.get("status") != "validated":
        raise ValueError("complete Study 07 validation is required")
    manifests: dict[str, pd.DataFrame] = {}
    expected = {record["age"]: record for record in validation["ages"]}
    for age in AGES:
        manifest = pd.read_csv(FINAL_ROOT / age / "manifest.csv").sort_values("z_index")
        record = expected[age]
        if (
            len(manifest) != int(record["n_z_levels"])
            or int(manifest["n_spots"].sum()) != int(record["n_spots"])
            or not manifest["all_converged"].astype(bool).all()
        ):
            raise ValueError(f"{age} manifest does not match validated output")
        manifests[age] = manifest
    return validation, manifests


def collect_plot_data(manifests: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    expected_gene_order: tuple[str, ...] | None = None
    for age_index, age in enumerate(AGES):
        for row in manifests[age].itertuples(index=False):
            path = FINAL_ROOT / age / str(row.filename)
            data = ad.read_h5ad(path, backed="r")
            try:
                if data.n_obs != int(row.n_spots):
                    raise ValueError(f"spot count differs from manifest in {path}")
                gene_order = tuple(map(str, data.var_names))
                if expected_gene_order is None:
                    expected_gene_order = gene_order
                    missing = sorted(set(GENES) - set(gene_order))
                    if missing:
                        raise ValueError(f"selected genes are absent: {missing}")
                elif gene_order != expected_gene_order:
                    raise ValueError(f"gene order changed in {path}")
                coordinates = np.asarray(data.obsm["spatial_3d"], dtype=float)
                if coordinates.shape != (data.n_obs, 3):
                    raise ValueError(f"invalid spatial_3d coordinates in {path}")
                if not np.allclose(coordinates[:, 2], float(row.z_world)):
                    raise ValueError(f"z coordinates differ from manifest in {path}")
                z_index = int(row.z_index)
                selected = sampled_indices(
                    data.n_obs, SAMPLE_SEED + age_index * 100_000 + z_index
                )
                gene_indices = [gene_order.index(gene) for gene in GENES]
                matrix = data.layers["counts"] if "counts" in data.layers else data.X
                expression = np.asarray(matrix[:, gene_indices], dtype=float)[selected]
                frame = pd.DataFrame(
                    {
                        "age": age,
                        "z_index": z_index,
                        "spot_index": selected,
                        "x": coordinates[selected, 0],
                        "y": coordinates[selected, 1],
                        "z": coordinates[selected, 2],
                    }
                )
                for column, gene in enumerate(GENES):
                    frame[gene] = expression[:, column]
                records.append(frame)
            finally:
                data.file.close()
    return pd.concat(records, ignore_index=True).sort_values(
        ["age", "z_index", "spot_index"]
    )


def spatial_limits(frame: pd.DataFrame) -> tuple[tuple[float, float], ...]:
    limits = []
    for column in ("x", "y", "z"):
        low, high = float(frame[column].min()), float(frame[column].max())
        margin = max(0.03 * (high - low), 0.01)
        limits.append((low - margin, high + margin))
    return tuple(limits)


def style_3d(
    axis: plt.Axes,
    limits: tuple[tuple[float, float], ...],
    elev: float,
    azim: float,
    *,
    labels: bool,
) -> None:
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_zlim(*limits[2])
    axis.set_box_aspect(tuple(high - low for low, high in limits))
    axis.view_init(elev=elev, azim=azim)
    axis.grid(False)
    if labels:
        axis.set_xlabel("x", labelpad=3)
        axis.set_ylabel("y", labelpad=3)
        axis.set_zlabel("z", labelpad=3)
        for dimension in (axis.xaxis, axis.yaxis, axis.zaxis):
            dimension.set_major_locator(mpl.ticker.MaxNLocator(3))
        for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
            pane.fill = False
    else:
        axis.set_axis_off()


def export(figure: plt.Figure, stem: Path, title: str, dpi: int) -> None:
    figure.savefig(
        stem.with_suffix(".pdf"),
        metadata={
            "Title": title,
            "Creator": "FEAST publication visualization",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(
        stem.with_suffix(".svg"),
        metadata={
            "Title": title,
            "Creator": "FEAST publication visualization",
            "Date": None,
        },
    )
    figure.savefig(
        stem.with_suffix(".png"),
        dpi=dpi,
        metadata={"Software": "FEAST Study 07 3D gene plotting script"},
    )
    plt.close(figure)


def render_volume(frame: pd.DataFrame, stem: Path, dpi: int) -> None:
    limits = spatial_limits(frame)
    norm = Normalize(vmin=float(frame["z"].min()), vmax=float(frame["z"].max()))
    figure = plt.figure(figsize=(8.4, 6.8))
    grid = figure.add_gridspec(
        2, 2, left=0.11, right=0.84, bottom=0.08, top=0.92, hspace=0.10, wspace=0.12
    )
    for row, (view_name, elev, azim) in enumerate(VIEWS):
        for column, age in enumerate(AGES):
            axis = figure.add_subplot(grid[row, column], projection="3d")
            subset = frame.loc[frame["age"].eq(age)]
            axis.scatter(
                subset["x"],
                subset["y"],
                subset["z"],
                c=subset["z"],
                cmap="viridis",
                norm=norm,
                s=0.34,
                alpha=0.78,
                linewidths=0,
                depthshade=False,
                rasterized=True,
            )
            style_3d(axis, limits, elev, azim, labels=True)
            if row == 0:
                axis.set_title(age, pad=5)
            if column == 0:
                axis.text2D(
                    -0.12,
                    0.5,
                    view_name,
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontweight="bold",
                )
    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="viridis"),
        cax=figure.add_axes([0.90, 0.19, 0.025, 0.61]),
    )
    colorbar.set_label("DevCCF z coordinate")
    figure.suptitle("Study 07 generated DevCCF volumes", y=0.985, fontsize=12, fontweight="bold")
    export(figure, stem, "FEAST Study 07 generated DevCCF volumes", dpi)


def positive_norm(values: np.ndarray) -> tuple[np.ndarray, Normalize, float]:
    transformed = np.log1p(values)
    positive = transformed[transformed > 0]
    upper = float(np.quantile(positive, 0.99)) if len(positive) else 1.0
    upper = max(upper, 1e-12)
    return transformed, Normalize(vmin=0.0, vmax=upper, clip=True), upper


def render_genes(frame: pd.DataFrame, stem: Path, dpi: int) -> dict[str, float]:
    limits = spatial_limits(frame)
    figure = plt.figure(figsize=(7.3, 10.8))
    grid = figure.add_gridspec(
        len(GENES),
        len(AGES),
        left=0.065,
        right=0.88,
        bottom=0.045,
        top=0.94,
        hspace=0.03,
        wspace=0.01,
    )
    upper_limits: dict[str, float] = {}
    for row, gene in enumerate(GENES):
        all_values, norm, upper = positive_norm(frame[gene].to_numpy(dtype=float))
        upper_limits[gene] = upper
        working = frame.assign(_display=all_values)
        row_axes = []
        for column, age in enumerate(AGES):
            axis = figure.add_subplot(grid[row, column], projection="3d")
            row_axes.append(axis)
            subset = working.loc[working["age"].eq(age)].sort_values("_display")
            axis.scatter(
                subset["x"], subset["y"], subset["z"],
                color="#D6DCE0", s=0.28, alpha=0.1, linewidths=0,
                depthshade=False, rasterized=True,
            )
            positive = subset.loc[subset["_display"].gt(0)]
            axis.scatter(
                positive["x"], positive["y"], positive["z"],
                c=positive["_display"], cmap="viridis", norm=norm,
                s=0.42, alpha=0.88, linewidths=0,
                depthshade=False, rasterized=True,
            )
            style_3d(axis, limits, 23.0, -55.0, labels=False)
            if row == 0:
                axis.set_title(age, pad=4)
            if column == 0:
                axis.text2D(
                    -0.08, 0.5, gene, transform=axis.transAxes,
                    rotation=90, ha="center", va="center",
                    fontsize=10, fontweight="bold",
                )
        colorbar = figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap="viridis"),
            ax=row_axes,
            fraction=0.022,
            pad=0.018,
        )
        colorbar.set_label("log1p counts", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    figure.suptitle(
        "Study 07 gene-level 3D expression distributions",
        y=0.985,
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "Uniform within-slice display sample; gray points have zero generated counts. Each gene uses one shared scale across ages.",
        ha="center",
        fontsize=7.2,
        color="#555555",
    )
    export(figure, stem, "FEAST Study 07 gene-level 3D expression", dpi)
    return upper_limits


def main() -> None:
    global FINAL_ROOT
    args = parse_args()
    FINAL_ROOT = args.input_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    volume_stem = output / "full_axis_3d_volume"
    gene_stem = output / "full_axis_3d_gene_distribution"
    figure_paths = [
        stem.with_suffix(suffix)
        for stem in (volume_stem, gene_stem)
        for suffix in (".pdf", ".svg", ".png")
    ]
    data_path = output / "gene_3d_plot_data.csv"
    provenance_path = output / "gene_3d_provenance.json"
    targets = figure_paths + [data_path, provenance_path]
    if any(path.exists() for path in targets):
        raise FileExistsError("fresh-only Study 07 3D gene outputs already exist")

    validation, manifests = require_inputs()
    frame = collect_plot_data(manifests)
    frame.to_csv(data_path, index=False)
    configure_matplotlib()
    render_volume(frame, volume_stem, int(args.dpi))
    upper_limits = render_genes(frame, gene_stem, int(args.dpi))

    record = {
        "schema_version": 1,
        "study": "07_3d_transfer",
        "configuration_id": validation["configuration_id"],
        "figure_promotion_authorized": False,
        "publication_claim_authorized": False,
        "target_expression_accuracy_claim_authorized": False,
        "interpretation": "descriptive generated DevCCF geometry and selected-gene expression distributions; no target-expression ground truth",
        "genes": list(GENES),
        "expression_display": {
            "transform": "log1p raw generated counts",
            "scale": "shared across ages within each gene",
            "upper_limit": "99th percentile among positive uniformly sampled values",
            "upper_limits": upper_limits,
            "zero_expression": "muted gray anatomical context",
        },
        "sampling": {
            "method": "uniform without replacement within each generated z slice",
            "seed": SAMPLE_SEED,
            "maximum_spots_per_slice": SAMPLE_CAP_PER_SLICE,
            "sampled_spots": int(len(frame)),
            "selection_by_expression": False,
        },
        "inputs": {
            "validation": str(FINAL_ROOT / "validation.json"),
            "manifests": [
                str(FINAL_ROOT / age / "manifest.csv") for age in AGES
            ],
            "validated_generated_outputs": int(sum(len(frame) for frame in manifests.values())),
        },
        "outputs": [path.name for path in figure_paths] + [data_path.name],
        "editable_text": {"pdf_fonttype": 42, "svg_fonttype": "none"},
        "png_dpi": int(args.dpi),
    }
    provenance_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote Study 07 3D volume and gene distributions to {output}")


if __name__ == "__main__":
    main()
