#!/usr/bin/env python3
"""Show oblique DevCCF targets and their extracted FEAST virtual sections."""
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
OBLIQUE_VIEW = (25.0, -55.0, 0.0)
SECTION_QUANTILES = (0.2, 0.5, 0.8)
SECTION_COLORS = ("#5E3C99", "#1B9E77", "#E66101")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=FINAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def configure_matplotlib() -> None:
    arial_paths = (
        Path(sys.prefix) / "fonts" / "arial.ttf",
        Path(sys.prefix) / "fonts" / "arialbd.ttf",
    )
    if not all(path.is_file() for path in arial_paths):
        raise FileNotFoundError(
            f"Arial Regular and Bold are required for this figure: {arial_paths}"
        )
    for path in arial_paths:
        font_manager.fontManager.addfont(path)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
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
                selected = np.arange(data.n_obs)
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


def select_sections(subset: pd.DataFrame) -> list[dict[str, float | int]]:
    levels = subset[["z_index", "z"]].drop_duplicates().sort_values("z_index")
    positions = np.rint((len(levels) - 1) * np.asarray(SECTION_QUANTILES)).astype(int)
    return [
        {
            "z_index": int(levels.iloc[position]["z_index"]),
            "z_world": float(levels.iloc[position]["z"]),
            "color": SECTION_COLORS[index],
        }
        for index, position in enumerate(positions)
    ]


def draw_section_planes(
    axis: plt.Axes,
    limits: tuple[tuple[float, float], ...],
    sections: list[dict[str, float | int]],
) -> None:
    x_values, y_values = np.meshgrid(limits[0], limits[1])
    for section in sections:
        z_values = np.full_like(x_values, float(section["z_world"]))
        axis.plot_surface(
            x_values,
            y_values,
            z_values,
            color=str(section["color"]),
            alpha=0.18,
            shade=False,
            linewidth=0,
            rcount=2,
            ccount=2,
        )


def style_section(axis: plt.Axes, limits: tuple[tuple[float, float], ...]) -> None:
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#AAAAAA")
        spine.set_linewidth(0.5)


def render_age(
    frame: pd.DataFrame,
    age: str,
    limits: tuple[tuple[float, float], ...],
    region_colors: dict[str, str],
    output_stem: Path,
    dpi: int,
) -> list[dict[str, float | int]]:
    subset = frame.loc[frame["age"].eq(age)]
    point_colors = subset["region"].map(region_colors).to_numpy()
    sections = select_sections(subset)
    figure = plt.figure(figsize=(10.4, 5.8))
    grid = figure.add_gridspec(
        3,
        2,
        width_ratios=[2.1, 1.0],
        left=0.035,
        right=0.975,
        bottom=0.12,
        top=0.84,
        hspace=0.22,
        wspace=0.10,
    )
    volume_axis = figure.add_subplot(grid[:, 0], projection="3d")
    volume_axis.scatter(
        subset["x"],
        subset["y"],
        subset["z"],
        c=point_colors,
        s=0.20,
        alpha=0.22,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )
    draw_section_planes(volume_axis, limits, sections)
    style_3d(volume_axis, limits, *OBLIQUE_VIEW)
    volume_axis.set_title("Oblique 3D DevCCF target", pad=4)

    for index, section in enumerate(sections):
        axis = figure.add_subplot(grid[index, 1])
        section_data = subset.loc[
            subset["z_index"].eq(int(section["z_index"]))
        ].sort_values("spot_index")
        axis.scatter(
            section_data["x"],
            section_data["y"],
            c=section_data["region"].map(region_colors),
            marker="s",
            s=0.75,
            linewidths=0,
            rasterized=True,
        )
        style_section(axis, limits)
        axis.set_title(
            f"Virtual section {index + 1}: z = {float(section['z_world']):.2f}",
            color=str(section["color"]),
            fontsize=7.2,
            loc="left",
            pad=2,
        )

    region_order = list(region_colors)
    handles = [
        Patch(facecolor=region_colors[region], edgecolor="none", label=region)
        for region in region_order
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=8,
        bbox_to_anchor=(0.5, 0.035),
        columnspacing=1.0,
        handlelength=0.9,
        handletextpad=0.4,
    )
    figure.suptitle(
        f"{age} FEAST virtual-section extraction overview",
        y=0.985,
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.67,
        0.50,
        "extract\n→",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#444444",
    )
    figure.text(
        0.5,
        0.085,
        "All generated target positions are shown. Transparent planes mark three z levels; panels at right show the corresponding FEAST virtual sections.",
        ha="center",
        fontsize=7.0,
        color="#555555",
    )
    title = f"FEAST Study 07 {age} oblique DevCCF virtual-section overview"
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
        metadata={"Software": "FEAST Study 07 virtual-section overview plotting script"},
    )
    plt.close(figure)
    return sections


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
    sections = {
        age: render_age(frame, age, limits, region_colors, stems[age], int(args.dpi))
        for age in AGES
    }

    provenance = {
        "schema_version": 1,
        "study": "07_3d_transfer",
        "configuration_id": validation["configuration_id"],
        "status": "candidate_author_review_required",
        "interpretation": "oblique generated target geometry with transparent z planes linked to exact FEAST virtual spatial-transcriptomics sections",
        "target_expression_accuracy_claim_authorized": False,
        "ages": list(AGES),
        "view": {"name": "Oblique", "elev": OBLIQUE_VIEW[0], "azim": OBLIQUE_VIEW[1], "roll": OBLIQUE_VIEW[2]},
        "virtual_sections": sections,
        "regions": region_colors,
        "spatial_limits": {
            axis: list(limit)
            for axis, limit in zip(("x", "y", "z"), limits, strict=True)
        },
        "sampling": {
            "method": "all generated target positions",
            "displayed_spots": int(len(frame)),
            "selection_by_region": False,
            "section_selection": "three z levels at 20%, 50%, and 80% of each age's ordered generated slices",
        },
        "inputs": {
            "validation": str(FINAL_ROOT / "validation.json"),
            "manifests": [
                str(FINAL_ROOT / age / "manifest.csv") for age in AGES
            ],
            "region_schema": "Datasets/Processed/DevCCFv1_figshare_26377171/coordinate_system/GSE269617_region_merged/gse269617_region_schema.tsv",
        },
        "outputs": [path.name for path in figure_paths],
        "font_family": "Arial",
        "editable_text": {"pdf_fonttype": 42, "svg_fonttype": "none"},
        "png_dpi": int(args.dpi),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote Study 07 four-view region volumes to {output}")


if __name__ == "__main__":
    main()
