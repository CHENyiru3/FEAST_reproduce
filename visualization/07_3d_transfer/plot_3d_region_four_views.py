#!/usr/bin/env python3
"""Render complete Study 07 DevCCF region labels from four 3D views."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


REPRO_ROOT = Path(__file__).resolve().parents[2]
FINAL_ROOT = REPRO_ROOT / "07_3d_transfer" / "outputs" / "final"
SCHEMA_PATH = (
    REPRO_ROOT.parent
    / "Datasets"
    / "Processed"
    / "DevCCFv1_figshare_26377171"
    / "coordinate_system"
    / "GSE269617_region_merged"
    / "gse269617_region_schema.tsv"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "figures"
AGES = ("E15.5", "E18.5")
SAMPLE_CAP_PER_SLICE = 700
SAMPLE_SEED = 2026
VIEWS = (
    ("Oblique", 25.0, -55.0, 0.0),
    ("Coronal", 90.0, -90.0, 0.0),
    ("Sagittal", 0.0, 0.0, 0.0),
    ("Dorsal", 0.0, -90.0, 0.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=FINAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "legend.fontsize": 7.2,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def sampled_indices(size: int, seed: int) -> np.ndarray:
    if size <= SAMPLE_CAP_PER_SLICE:
        return np.arange(size)
    return np.sort(
        np.random.default_rng(seed).choice(
            size, size=SAMPLE_CAP_PER_SLICE, replace=False
        )
    )


def require_inputs() -> tuple[dict, dict[str, pd.DataFrame], dict[str, str]]:
    validation = json.loads(
        (FINAL_ROOT / "validation.json").read_text(encoding="utf-8")
    )
    if validation.get("status") != "validated":
        raise ValueError("complete Study 07 validation is required")

    validated = {record["age"]: record for record in validation["ages"]}
    manifests: dict[str, pd.DataFrame] = {}
    for age in AGES:
        manifest = pd.read_csv(FINAL_ROOT / age / "manifest.csv").sort_values(
            "z_index"
        )
        expected = validated[age]
        if (
            len(manifest) != int(expected["n_z_levels"])
            or int(manifest["n_spots"].sum()) != int(expected["n_spots"])
            or not manifest["all_converged"].astype(bool).all()
        ):
            raise ValueError(f"{age} manifest does not match validated output")
        manifests[age] = manifest

    schema = pd.read_csv(SCHEMA_PATH, sep="\t")
    included = schema.loc[schema["include_in_final_generation"].astype(bool)]
    region_colors = dict(zip(included["region_label"], included["hex_color"], strict=True))
    return validation, manifests, region_colors


def collect_plot_data(
    manifests: dict[str, pd.DataFrame], region_colors: dict[str, str]
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for age_index, age in enumerate(AGES):
        for row in manifests[age].itertuples(index=False):
            path = FINAL_ROOT / age / str(row.filename)
            data = ad.read_h5ad(path, backed="r")
            try:
                if int(data.n_obs) != int(row.n_spots):
                    raise ValueError(f"spot count differs from manifest in {path}")
                coordinates = np.asarray(data.obsm["spatial_3d"], dtype=float)
                if coordinates.shape != (data.n_obs, 3):
                    raise ValueError(f"invalid spatial_3d coordinates in {path}")
                if not np.allclose(coordinates[:, 2], float(row.z_world)):
                    raise ValueError(f"z coordinates differ from manifest in {path}")
                regions = data.obs["region"].astype(str).to_numpy()
                unknown = sorted(set(regions) - set(region_colors))
                if unknown:
                    raise ValueError(f"unmapped regions in {path}: {unknown}")
                selected = sampled_indices(
                    data.n_obs, SAMPLE_SEED + age_index * 100_000 + int(row.z_index)
                )
                records.append(
                    pd.DataFrame(
                        {
                            "age": age,
                            "z_index": int(row.z_index),
                            "spot_index": selected,
                            "x": coordinates[selected, 0],
                            "y": coordinates[selected, 1],
                            "z": coordinates[selected, 2],
                            "region": regions[selected],
                        }
                    )
                )
            finally:
                data.file.close()
    return pd.concat(records, ignore_index=True).sort_values(
        ["age", "z_index", "spot_index"]
    )


def spatial_limits(frame: pd.DataFrame) -> tuple[tuple[float, float], ...]:
    limits = []
    for column in ("x", "y", "z"):
        low, high = float(frame[column].min()), float(frame[column].max())
        margin = max(0.035 * (high - low), 0.01)
        limits.append((low - margin, high + margin))
    return tuple(limits)


def style_3d(
    axis: plt.Axes,
    limits: tuple[tuple[float, float], ...],
    elev: float,
    azim: float,
    roll: float,
) -> None:
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_zlim(*limits[2])
    axis.set_box_aspect(tuple(high - low for low, high in limits))
    axis.view_init(elev=elev, azim=azim, roll=roll)
    axis.set_proj_type("ortho")
    axis.grid(False)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((0.96, 0.96, 0.96, 0.56))
        pane.set_edgecolor((0.72, 0.72, 0.72, 0.65))


def render_age(
    frame: pd.DataFrame,
    age: str,
    limits: tuple[tuple[float, float], ...],
    region_colors: dict[str, str],
    output_stem: Path,
    dpi: int,
) -> None:
    subset = frame.loc[frame["age"].eq(age)]
    point_colors = subset["region"].map(region_colors).to_numpy()
    figure = plt.figure(figsize=(7.6, 7.2))
    grid = figure.add_gridspec(
        2,
        2,
        left=0.035,
        right=0.965,
        bottom=0.035,
        top=0.85,
        hspace=0.10,
        wspace=0.07,
    )
    for index, (view, elev, azim, roll) in enumerate(VIEWS):
        axis = figure.add_subplot(grid[index // 2, index % 2], projection="3d")
        axis.scatter(
            subset["x"],
            subset["y"],
            subset["z"],
            c=point_colors,
            s=0.34,
            alpha=0.72,
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )
        style_3d(axis, limits, elev, azim, roll)
        axis.set_title(view, pad=2)

    region_order = list(region_colors)
    handles = [
        Patch(facecolor=region_colors[region], edgecolor="none", label=region)
        for region in region_order
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.925),
        columnspacing=1.25,
        handlelength=0.9,
        handletextpad=0.4,
    )
    figure.suptitle(
        f"{age} complete FEAST target volume · DevCCF broad-region labels",
        y=0.985,
        fontsize=11.5,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "Atlas labels are exact target geometry; displayed points are a fixed uniform within-slice sample.",
        ha="center",
        fontsize=7.0,
        color="#555555",
    )
    title = f"FEAST Study 07 {age} DevCCF broad-region four-view volume"
    figure.savefig(
        output_stem.with_suffix(".pdf"),
        metadata={
            "Title": title,
            "Creator": "FEAST publication visualization",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(
        output_stem.with_suffix(".svg"),
        metadata={
            "Title": title,
            "Creator": "FEAST publication visualization",
            "Date": None,
        },
    )
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=dpi,
        metadata={"Software": "FEAST Study 07 four-view region plotting script"},
    )
    plt.close(figure)


def main() -> None:
    global FINAL_ROOT
    args = parse_args()
    FINAL_ROOT = args.input_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stems = {
        age: output / f"{age.replace('.', '_')}_full_axis_region_four_views"
        for age in AGES
    }
    figure_paths = [
        stem.with_suffix(suffix)
        for stem in stems.values()
        for suffix in (".pdf", ".svg", ".png")
    ]
    provenance_path = output / "full_axis_region_four_views_provenance.json"
    if any(path.exists() for path in [*figure_paths, provenance_path]):
        raise FileExistsError("fresh-only four-view region output already exists")

    validation, manifests, region_colors = require_inputs()
    frame = collect_plot_data(manifests, region_colors)
    limits = spatial_limits(frame)
    configure_matplotlib()
    for age in AGES:
        render_age(frame, age, limits, region_colors, stems[age], int(args.dpi))

    provenance = {
        "schema_version": 1,
        "study": "07_3d_transfer",
        "configuration_id": validation["configuration_id"],
        "status": "candidate_author_review_required",
        "interpretation": "complete generated target geometry shown from four matched views using exact DevCCF broad-region labels",
        "target_expression_accuracy_claim_authorized": False,
        "ages": list(AGES),
        "views": [
            {"name": name, "elev": elev, "azim": azim, "roll": roll}
            for name, elev, azim, roll in VIEWS
        ],
        "regions": region_colors,
        "spatial_limits": {
            axis: list(limit)
            for axis, limit in zip(("x", "y", "z"), limits, strict=True)
        },
        "sampling": {
            "method": "uniform without replacement within each generated z slice",
            "seed": SAMPLE_SEED,
            "maximum_spots_per_slice": SAMPLE_CAP_PER_SLICE,
            "sampled_spots": int(len(frame)),
            "selection_by_region": False,
            "shared_coordinates_across_views": True,
        },
        "inputs": {
            "validation": str(FINAL_ROOT / "validation.json"),
            "manifests": [
                str(FINAL_ROOT / age / "manifest.csv") for age in AGES
            ],
            "region_schema": "Datasets/Processed/DevCCFv1_figshare_26377171/coordinate_system/GSE269617_region_merged/gse269617_region_schema.tsv",
        },
        "outputs": [path.name for path in figure_paths],
        "editable_text": {"pdf_fonttype": 42, "svg_fonttype": "none"},
        "png_dpi": int(args.dpi),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote Study 07 four-view region volumes to {output}")


if __name__ == "__main__":
    main()
