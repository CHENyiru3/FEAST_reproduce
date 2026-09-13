#!/usr/bin/env python3
"""Render Study 06 whole-volume geometry and selected-gene 3D distributions."""
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
DEFAULT_INPUT_ROOT = REPRO_ROOT / "06_3d_stack" / "outputs" / "reference_density_rank_v1"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "figures"
GENES = ("Slc17a7", "Gad2", "Gfap", "Reln")
GAPS = (0, 3, 5, 10)
GAP_LABELS = {
    0: "Ground truth (144 real)",
}
SAMPLE_CAP_PER_SLICE = 1_200
SAMPLE_SEED = 2026
VIEWS = (("Oblique view", 25.0, -55.0), ("Lateral view", 12.0, 0.0))


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
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
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


def require_plan(input_root: Path) -> tuple[dict, dict]:
    validation = json.loads(
        (input_root / "validation_summary.json").read_text(encoding="utf-8")
    )
    if validation.get("status") != "passed" or validation.get(
        "validated_outputs"
    ) != 336 or validation.get("positions_per_stack") != 144 or validation.get("stack_geometries_identical") is not True:
        raise ValueError("complete Study 06 validation is required")
    plan = json.loads((input_root / "plan.json").read_text(encoding="utf-8"))
    return plan, validation


def sample_file(
    path: Path,
    metadata: dict[str, object],
    expected_gene_order: tuple[str, ...] | None,
    seed: int,
) -> tuple[pd.DataFrame, tuple[str, ...], int]:
    data = ad.read_h5ad(path, backed="r")
    try:
        gene_order = tuple(map(str, data.var_names))
        if expected_gene_order is None:
            missing = sorted(set(GENES) - set(gene_order))
            if missing:
                raise ValueError(f"selected genes are absent: {missing}")
        elif gene_order != expected_gene_order:
            raise ValueError(f"gene order changed in {path}")
        coordinates = np.asarray(data.obsm["spatial_3d"], dtype=float)
        if coordinates.shape != (data.n_obs, 3):
            raise ValueError(f"invalid spatial_3d coordinates in {path}")
        selected = sampled_indices(data.n_obs, seed)
        gene_indices = [gene_order.index(gene) for gene in GENES]
        matrix = data.layers["counts"] if "counts" in data.layers else data.X
        expression = matrix[:, gene_indices]
        if hasattr(expression, "toarray"):
            expression = expression.toarray()
        expression = np.asarray(expression, dtype=float)[selected]
        frame = pd.DataFrame(
            {
                **metadata,
                "spot_index": selected,
                "x": coordinates[selected, 0],
                "y": coordinates[selected, 1],
                "z": coordinates[selected, 2],
            }
        )
        for column, gene in enumerate(GENES):
            frame[gene] = expression[:, column]
        return frame, gene_order, int(data.n_obs)
    finally:
        data.file.close()


def collect_plot_data(plan: dict, input_root: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    records: list[pd.DataFrame] = []
    expected_gene_order: tuple[str, ...] | None = None
    real_paths = [Path(plan["data_dir"]) / f"Zhuang-ABCA-1.{int(slice_id):03d}.h5ad" for slice_id in plan["available_slices"]]
    if not real_paths:
        raise FileNotFoundError("Study 06 real slices are required")
    real_spots = 0
    for real_index, path in enumerate(real_paths):
        target_slice = int(path.stem.rsplit(".", 1)[1])
        frame, expected_gene_order, n_obs = sample_file(
            path,
            {
                "density_name": "ground_truth",
                "gap": 0,
                "target_index": real_index,
                "target_slice": target_slice,
                "source_kind": "real",
            },
            expected_gene_order,
            SAMPLE_SEED + real_index,
        )
        records.append(frame)
        real_spots += n_obs

    for density_name, density in plan["densities"].items():
        gap = int(density["spec"]["gap"])
        references = {int(item) for item in density["reference_pool"]}
        for target_index, target_slice in enumerate(map(int, plan["available_slices"])):
            source_kind = "real" if target_slice in references else "simulated"
            path = Path(plan["data_dir"]) / f"Zhuang-ABCA-1.{target_slice:03d}.h5ad" if source_kind == "real" else input_root / "targets" / density_name / f"Zhuang-ABCA-1.{target_slice:03d}" / "generated.h5ad"
            frame, expected_gene_order, _n_obs = sample_file(
                path,
                {
                    "density_name": density_name,
                    "gap": gap,
                    "target_index": target_index,
                    "target_slice": target_slice,
                    "source_kind": source_kind,
                },
                expected_gene_order,
                SAMPLE_SEED + gap * 10_000 + target_index,
            )
            records.append(frame)
    result = pd.concat(records, ignore_index=True).sort_values(
        ["gap", "target_index", "spot_index"]
    )
    return result, {"slices": len(real_paths), "spots": real_spots}


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
        axis.set_xlabel("x", labelpad=-7)
        axis.set_ylabel("y", labelpad=-7)
        axis.set_zlabel("z", labelpad=-5)
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
        metadata={"Software": "FEAST Study 06 3D gene plotting script"},
    )
    plt.close(figure)


def render_volume(frame: pd.DataFrame, stem: Path, dpi: int) -> None:
    limits = spatial_limits(frame)
    norm = Normalize(vmin=float(frame["z"].min()), vmax=float(frame["z"].max()))
    figure = plt.figure(figsize=(13.2, 6.4))
    grid = figure.add_gridspec(
        2, 4, left=0.035, right=0.91, bottom=0.05, top=0.92, hspace=0.08, wspace=0.02
    )
    for row, (view_name, elev, azim) in enumerate(VIEWS):
        for column, gap in enumerate(GAPS):
            axis = figure.add_subplot(grid[row, column], projection="3d")
            subset = frame.loc[frame["gap"].eq(gap)]
            for source_kind, marker, size, alpha in (("simulated", "o", 0.42, 0.72), ("real", "^", 0.72, 0.95)):
                points = subset.loc[subset["source_kind"].eq(source_kind)]
                if not points.empty:
                    axis.scatter(points["x"], points["y"], points["z"], c=points["z"], cmap="viridis", norm=norm, marker=marker, s=size, alpha=alpha, linewidths=0, depthshade=False, rasterized=True)
            style_3d(axis, limits, elev, azim, labels=True)
            if row == 0:
                axis.set_title(GAP_LABELS[gap], pad=5)
            if column == 0:
                axis.text2D(
                    -0.15,
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
        ax=figure.axes,
        shrink=0.7,
        pad=0.025,
    )
    colorbar.set_label("Native z coordinate")
    figure.legend(handles=[mpl.lines.Line2D([], [], color="#555555", marker="^", linestyle="None", label="Retained real slice"), mpl.lines.Line2D([], [], color="#555555", marker="o", linestyle="None", label="FEAST-simulated slice")], loc="lower center", ncol=2, frameon=False)
    figure.suptitle("Study 06 real and generated 3D volumes", y=0.985, fontsize=12, fontweight="bold")
    export(figure, stem, "FEAST Study 06 real and generated 3D volumes", dpi)


def positive_norm(values: np.ndarray) -> tuple[np.ndarray, Normalize, float]:
    transformed = np.log1p(values)
    positive = transformed[transformed > 0]
    upper = float(np.quantile(positive, 0.99)) if len(positive) else 1.0
    upper = max(upper, 1e-12)
    return transformed, Normalize(vmin=0.0, vmax=upper, clip=True), upper


def render_genes(frame: pd.DataFrame, stem: Path, dpi: int) -> dict[str, float]:
    limits = spatial_limits(frame)
    figure = plt.figure(figsize=(13.2, 10.8))
    grid = figure.add_gridspec(
        len(GENES),
        len(GAPS),
        left=0.055,
        right=0.92,
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
        for column, gap in enumerate(GAPS):
            axis = figure.add_subplot(grid[row, column], projection="3d")
            row_axes.append(axis)
            subset = working.loc[working["gap"].eq(gap)].sort_values("_display")
            for source_kind, marker, size in (("simulated", "o", 0.48), ("real", "^", 0.72)):
                source = subset.loc[subset["source_kind"].eq(source_kind)]
                axis.scatter(source["x"], source["y"], source["z"], color="#D6DCE0", marker=marker, s=size, alpha=0.12, linewidths=0, depthshade=False, rasterized=True)
                positive = source.loc[source["_display"].gt(0)]
                axis.scatter(positive["x"], positive["y"], positive["z"], c=positive["_display"], cmap="viridis", norm=norm, marker=marker, s=size, alpha=0.88, linewidths=0, depthshade=False, rasterized=True)
            style_3d(axis, limits, 25.0, -55.0, labels=False)
            if row == 0:
                axis.set_title(GAP_LABELS[gap], pad=4)
            if column == 0:
                axis.text2D(
                    -0.08, 0.5, gene, transform=axis.transAxes,
                    rotation=90, ha="center", va="center",
                    fontsize=10, fontweight="bold",
                )
        colorbar = figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap="viridis"),
            ax=row_axes,
            fraction=0.018,
            pad=0.015,
        )
        colorbar.set_label("log1p counts", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    figure.suptitle(
        "Study 06 ground-truth and generated gene-level 3D distributions",
        y=0.985,
        fontsize=12,
        fontweight="bold",
    )
    figure.legend(handles=[mpl.lines.Line2D([], [], color="#555555", marker="^", linestyle="None", label="Retained real slice"), mpl.lines.Line2D([], [], color="#555555", marker="o", linestyle="None", label="FEAST-simulated slice")], loc="lower center", ncol=2, frameon=False)
    figure.text(
        0.5,
        0.012,
        "Ground truth contains the 144 modeled real slices. Triangles are retained real slices; circles are FEAST-simulated slices.",
        ha="center",
        fontsize=7.2,
        color="#555555",
    )
    export(figure, stem, "FEAST Study 06 ground-truth and generated gene-level 3D expression", dpi)
    return upper_limits


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    volume_stem = output / "conditional_stack_3d_volume"
    gene_stem = output / "conditional_stack_3d_gene_distribution"
    figure_paths = [
        stem.with_suffix(suffix)
        for stem in (volume_stem, gene_stem)
        for suffix in (".pdf", ".svg", ".png")
    ]
    data_path = output / "gene_3d_plot_data.csv"
    provenance_path = output / "gene_3d_provenance.json"
    targets = figure_paths + [data_path, provenance_path]
    if any(path.exists() for path in targets):
        raise FileExistsError("fresh-only Study 06 3D gene outputs already exist")

    input_root = args.input_root.resolve()
    plan, validation = require_plan(input_root)
    GAP_LABELS.update({int(density['spec']['gap']): f"Gap {int(density['spec']['gap'])} ({len(density['reference_pool'])} real / {len(density['target_pool'])} simulated)" for density in plan['densities'].values()})
    frame, ground_truth = collect_plot_data(plan, input_root)
    frame.to_csv(data_path, index=False)
    configure_matplotlib()
    render_volume(frame, volume_stem, int(args.dpi))
    upper_limits = render_genes(frame, gene_stem, int(args.dpi))

    record = {
        "schema_version": 1,
        "study": "06_3d_stack",
        "configuration_id": validation["configuration_id"],
        "figure_promotion_authorized": False,
        "publication_claim_authorized": False,
        "interpretation": "descriptive comparison of all real-slice ground truth with generated-volume geometry and selected-gene expression distributions",
        "genes": list(GENES),
        "expression_display": {
            "transform": "log1p raw generated counts",
            "scale": "shared across real ground truth and all density arms within each gene",
            "upper_limit": "99th percentile among positive uniformly sampled values",
            "upper_limits": upper_limits,
            "zero_expression": "muted gray anatomical context",
        },
        "sampling": {
            "method": "uniform without replacement within each real or generated z slice",
            "seed": SAMPLE_SEED,
            "maximum_spots_per_slice": SAMPLE_CAP_PER_SLICE,
            "sampled_spots": int(len(frame)),
            "selection_by_expression": False,
        },
        "inputs": {
            "plan": str(input_root / "plan.json"),
            "validation": str(input_root / "validation_summary.json"),
            "ground_truth": {
                "data_dir": plan["data_dir"],
                "slices": ground_truth["slices"],
                "spots": ground_truth["spots"],
            },
            "validated_generated_outputs": int(validation["validated_outputs"]),
        },
        "outputs": [path.name for path in figure_paths] + [data_path.name],
        "editable_text": {"pdf_fonttype": 42, "svg_fonttype": "none"},
        "png_dpi": int(args.dpi),
    }
    provenance_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote Study 06 3D volume and gene distributions to {output}")


if __name__ == "__main__":
    main()
