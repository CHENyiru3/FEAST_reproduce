#!/usr/bin/env python3
"""Link full-axis regional support to atlas and marker-gene cross-sections."""
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
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot import REGION_COLORS, coverage_for_plot, style_axis


REPRO_ROOT = Path(__file__).resolve().parents[2]
FINAL_ROOT = REPRO_ROOT / "07_3d_transfer" / "outputs" / "final"
EVALUATION_ROOT = FINAL_ROOT / "evaluation"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "figures"
AGES = ("E15.5", "E18.5")
GENES = ("Slc17a7", "Gad2", "Gfap", "Reln")
EXPRESSION_CMAP = "magma"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=FINAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def require_inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    validation_path = FINAL_ROOT / "validation.json"
    coverage_path = EVALUATION_ROOT / "region_coverage.csv"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "validated":
        raise ValueError("complete Study 07 validation is required")

    coverage = pd.read_csv(coverage_path)
    manifests: dict[str, pd.DataFrame] = {}
    validated = {record["age"]: record for record in validation["ages"]}
    for age in AGES:
        manifest = pd.read_csv(FINAL_ROOT / age / "manifest.csv").sort_values("z_index")
        expected = validated[age]
        if (
            len(manifest) != int(expected["n_z_levels"])
            or int(manifest["n_spots"].sum()) != int(expected["n_spots"])
            or not manifest["all_converged"].astype(bool).all()
        ):
            raise ValueError(f"{age} manifest does not match validated output")
        manifests[age] = manifest
    return coverage, manifests


def select_peak_support_slices(
    coverage: pd.DataFrame, manifests: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    totals = (
        coverage.groupby(["age", "z_index", "z_world"], as_index=False)["n_spots"]
        .sum()
        .sort_values(["age", "n_spots", "z_index"], ascending=[True, False, True])
    )
    selected = totals.groupby("age", sort=False).head(1).copy()
    rows = []
    for age in AGES:
        support = selected.loc[selected["age"].eq(age)].iloc[0]
        manifest = manifests[age]
        matched = manifest.loc[manifest["z_index"].eq(int(support["z_index"]))]
        if len(matched) != 1:
            raise ValueError(f"{age} peak-support slice is not unique in manifest")
        record = matched.iloc[0]
        if (
            not np.isclose(float(record["z_world"]), float(support["z_world"]))
            or int(record["n_spots"]) != int(support["n_spots"])
        ):
            raise ValueError(f"{age} support and manifest disagree at selected z")
        rows.append(
            {
                "age": age,
                "z_index": int(record["z_index"]),
                "z_world": float(record["z_world"]),
                "n_spots": int(record["n_spots"]),
                "filename": str(record["filename"]),
                "selection_rule": "maximum blueprint spots per z level",
            }
        )
    return pd.DataFrame(rows)


def load_cross_sections(selection: pd.DataFrame) -> dict[str, dict[str, object]]:
    sections: dict[str, dict[str, object]] = {}
    expected_gene_order: tuple[str, ...] | None = None
    for row in selection.itertuples(index=False):
        path = FINAL_ROOT / str(row.age) / str(row.filename)
        data = ad.read_h5ad(path, backed="r")
        try:
            gene_order = tuple(map(str, data.var_names))
            if expected_gene_order is None:
                expected_gene_order = gene_order
                missing = sorted(set(GENES) - set(gene_order))
                if missing:
                    raise ValueError(f"selected genes are absent: {missing}")
            elif gene_order != expected_gene_order:
                raise ValueError(f"gene order changed in {path}")

            coordinates = np.asarray(data.obsm["spatial"], dtype=float)
            if coordinates.shape != (int(row.n_spots), 2):
                raise ValueError(f"invalid spatial coordinates in {path}")
            if int(data.n_obs) != int(row.n_spots):
                raise ValueError(f"spot count differs from manifest in {path}")
            observed_z = data.obs["z"].to_numpy(dtype=float)
            if not np.allclose(observed_z, float(row.z_world)):
                raise ValueError(f"z coordinates differ from manifest in {path}")

            gene_indices = [gene_order.index(gene) for gene in GENES]
            matrix = data.layers["counts"] if "counts" in data.layers else data.X
            expression = np.asarray(matrix[:, gene_indices], dtype=float)
            sections[str(row.age)] = {
                "xy": coordinates,
                "regions": data.obs["region"].astype(str).to_numpy(),
                "expression": expression,
            }
        finally:
            data.file.close()
    return sections


def shared_limits(sections: dict[str, dict[str, object]]) -> tuple[float, float, float, float]:
    coordinates = np.vstack([np.asarray(sections[age]["xy"]) for age in AGES])
    x_min, y_min = coordinates.min(axis=0)
    x_max, y_max = coordinates.max(axis=0)
    x_margin = max(0.025 * (x_max - x_min), 0.01)
    y_margin = max(0.025 * (y_max - y_min), 0.01)
    return x_min - x_margin, x_max + x_margin, y_min - y_margin, y_max + y_margin


def expression_norms(
    sections: dict[str, dict[str, object]],
) -> tuple[dict[str, Normalize], dict[str, float]]:
    norms: dict[str, Normalize] = {}
    limits: dict[str, float] = {}
    for gene_index, gene in enumerate(GENES):
        values = np.concatenate(
            [np.asarray(sections[age]["expression"])[:, gene_index] for age in AGES]
        )
        transformed = np.log1p(values)
        positive = transformed[transformed > 0]
        upper = float(np.quantile(positive, 0.99)) if len(positive) else 1.0
        upper = max(upper, 1e-12)
        norms[gene] = Normalize(vmin=0.0, vmax=upper, clip=True)
        limits[gene] = upper
    return norms, limits


def plot_support(
    axis: plt.Axes,
    age: str,
    coverage: pd.DataFrame,
    region_order: list[str],
    region_colors: dict[str, str],
    selected: pd.Series,
) -> None:
    age_frame = coverage.loc[coverage["age"].eq(age)]
    pivot = age_frame.pivot_table(
        index="z_world", columns="display_region", values="n_spots", fill_value=0
    ).sort_index()
    series = [
        pivot[region].to_numpy(dtype=float)
        if region in pivot
        else np.zeros(len(pivot))
        for region in region_order
    ]
    x = pivot.index.to_numpy(dtype=float)
    total = np.sum(np.vstack(series), axis=0)
    axis.stackplot(
        x,
        series,
        colors=[region_colors[region] for region in region_order],
        linewidth=0,
        alpha=0.97,
    )
    axis.plot(x, total, color="#333333", linewidth=0.8)
    selected_z = float(selected["z_world"])
    selected_total = float(total[np.argmin(np.abs(x - selected_z))])
    axis.axvline(selected_z, color="#111111", linewidth=1.0, linestyle=(0, (2.5, 2)))
    axis.scatter(
        [selected_z], [selected_total], marker="D", s=25, color="#111111", zorder=5
    )
    axis.annotate(
        f"displayed section\nz = {selected_z:.2f}",
        xy=(selected_z, selected_total),
        xytext=(0.96, 0.95),
        textcoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7.0,
        color="#333333",
        arrowprops={"arrowstyle": "-", "color": "#333333", "linewidth": 0.6},
    )
    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(bottom=0)
    axis.set_xlabel("DevCCF z coordinate")
    axis.set_ylabel("Blueprint spots per z level")
    axis.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda value, _position: "0" if value == 0 else f"{value / 1000:g}k")
    )
    style_axis(axis)


def style_section(axis: plt.Axes, limits: tuple[float, float, float, float]) -> None:
    axis.set_xlim(limits[0], limits[1])
    axis.set_ylim(limits[2], limits[3])
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def plot_region_section(
    axis: plt.Axes,
    section: dict[str, object],
    region_order: list[str],
    region_colors: dict[str, str],
    limits: tuple[float, float, float, float],
) -> None:
    xy = np.asarray(section["xy"])
    regions = np.asarray(section["regions"])
    colors = [region_colors[str(region)] for region in regions]
    axis.scatter(
        xy[:, 0], xy[:, 1], c=colors, marker="s", s=1.2, linewidths=0,
        rasterized=True,
    )
    style_section(axis, limits)


def plot_gene_section(
    axis: plt.Axes,
    section: dict[str, object],
    gene_index: int,
    norm: Normalize,
    limits: tuple[float, float, float, float],
) -> None:
    xy = np.asarray(section["xy"])
    values = np.log1p(np.asarray(section["expression"])[:, gene_index])
    order = np.argsort(values, kind="stable")
    axis.scatter(
        xy[:, 0], xy[:, 1], color="#E2E2E2", marker="s", s=1.2,
        linewidths=0, rasterized=True,
    )
    positive = order[values[order] > 0]
    axis.scatter(
        xy[positive, 0], xy[positive, 1], c=values[positive], cmap=EXPRESSION_CMAP,
        norm=norm, marker="s", s=1.2, linewidths=0, rasterized=True,
    )
    style_section(axis, limits)


def render(
    coverage: pd.DataFrame,
    selection: pd.DataFrame,
    sections: dict[str, dict[str, object]],
    output_stem: Path,
    dpi: int,
) -> dict[str, float]:
    coverage_plot, region_order = coverage_for_plot(coverage)
    region_colors = {
        region: REGION_COLORS[index] for index, region in enumerate(region_order)
    }
    limits = shared_limits(sections)
    norms, upper_limits = expression_norms(sections)

    configure_matplotlib()
    figure = plt.figure(figsize=(14.2, 6.6))
    grid = figure.add_gridspec(
        3,
        6,
        height_ratios=[1.0, 1.0, 0.12],
        width_ratios=[2.3, 1.0, 1.0, 1.0, 1.0, 1.0],
        left=0.065,
        right=0.99,
        bottom=0.15,
        top=0.90,
        hspace=0.34,
        wspace=0.16,
    )
    map_titles = ("Atlas regions", *GENES)
    for row, age in enumerate(AGES):
        selected = selection.loc[selection["age"].eq(age)].iloc[0]
        support_axis = figure.add_subplot(grid[row, 0])
        plot_support(
            support_axis,
            age,
            coverage_plot,
            region_order,
            region_colors,
            selected,
        )
        support_axis.set_title(f"{age} regional support", loc="left", pad=5)
        if row == 0:
            support_axis.set_xlabel("")
        for column, title in enumerate(map_titles, start=1):
            axis = figure.add_subplot(grid[row, column])
            if column == 1:
                plot_region_section(axis, sections[age], region_order, region_colors, limits)
            else:
                gene = GENES[column - 2]
                plot_gene_section(
                    axis,
                    sections[age],
                    column - 2,
                    norms[gene],
                    limits,
                )
            if row == 0:
                axis.set_title(title, pad=5)

    legend_axis = figure.add_subplot(grid[2, :2])
    legend_axis.set_axis_off()
    region_handles = [
        Patch(facecolor=region_colors[region], edgecolor="none", label=region)
        for region in region_order
    ]
    selection_handle = Line2D(
        [], [], color="#111111", marker="D", linestyle=(0, (2.5, 2)),
        linewidth=1.0, markersize=4, label="Displayed z",
    )
    legend_axis.legend(
        handles=[*region_handles, selection_handle],
        loc="center left",
        ncol=5,
        columnspacing=0.9,
        handlelength=0.9,
        handletextpad=0.35,
    )
    for column, gene in enumerate(GENES, start=2):
        colorbar_axis = figure.add_subplot(grid[2, column])
        colorbar = figure.colorbar(
            mpl.cm.ScalarMappable(norm=norms[gene], cmap=EXPRESSION_CMAP),
            cax=colorbar_axis,
            orientation="horizontal",
        )
        colorbar.set_label("log1p generated counts", fontsize=6.5, labelpad=1)
        colorbar.ax.tick_params(labelsize=6, length=2, pad=1)

    figure.suptitle(
        "Full-axis anatomical support linked to generated marker-gene sections",
        y=0.985,
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.025,
        "Peak-support sections are selected within age and are not homologous cross-age positions; atlas labels are exact, but target expression was not observed.",
        ha="center",
        fontsize=7.2,
        color="#555555",
    )
    figure.savefig(
        output_stem.with_suffix(".pdf"),
        metadata={
            "Title": "FEAST full-axis support and marker-gene cross-sections",
            "Creator": "FEAST publication visualization",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(
        output_stem.with_suffix(".svg"),
        metadata={
            "Title": "FEAST full-axis support and marker-gene cross-sections",
            "Creator": "FEAST publication visualization",
            "Date": None,
        },
    )
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=dpi,
        metadata={"Software": "FEAST Study 07 cross-section plotting script"},
    )
    plt.close(figure)
    return upper_limits


def main() -> None:
    global FINAL_ROOT, EVALUATION_ROOT
    args = parse_args()
    FINAL_ROOT = args.input_root.resolve()
    EVALUATION_ROOT = FINAL_ROOT / "evaluation"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "full_axis_marker_cross_sections"
    selection_path = output / "full_axis_marker_cross_sections_selection.csv"
    provenance_path = output / "full_axis_marker_cross_sections_provenance.json"
    targets = [stem.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png")]
    targets.extend([selection_path, provenance_path])
    if any(path.exists() for path in targets):
        raise FileExistsError("fresh-only cross-section figure output already exists")

    coverage, manifests = require_inputs()
    selection = select_peak_support_slices(coverage, manifests)
    sections = load_cross_sections(selection)
    upper_limits = render(coverage, selection, sections, stem, int(args.dpi))
    selection.to_csv(selection_path, index=False)
    provenance = {
        "schema_version": 1,
        "study": "07_3d_transfer",
        "status": "candidate_author_review_required",
        "interpretation": "anatomical support linked to atlas-label and generated marker-gene cross-sections",
        "target_expression_exists": False,
        "target_expression_accuracy_claim_authorized": False,
        "selection": selection.to_dict("records"),
        "genes": list(GENES),
        "expression_display": {
            "transform": "log1p raw generated counts",
            "scale": "shared across ages within each gene",
            "upper_limit": "99th percentile among positive values in the two displayed sections",
            "upper_limits": upper_limits,
            "zero_expression": "light gray anatomical context",
        },
        "region_display": "exact DevCCF blueprint labels stored in each validated generated H5AD",
        "cross_age_section_alignment": "not homologous; each age independently uses its maximum-support z level",
        "inputs": {
            "coverage": str(EVALUATION_ROOT / "region_coverage.csv"),
            "validation": str(FINAL_ROOT / "validation.json"),
            "manifests": [
                str(FINAL_ROOT / age / "manifest.csv") for age in AGES
            ],
        },
        "outputs": [path.name for path in targets if path != provenance_path],
        "editable_text": {"pdf_fonttype": 42, "svg_fonttype": "none"},
        "png_dpi": int(args.dpi),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote Study 07 axis-linked cross-sections to {output}")


if __name__ == "__main__":
    main()
