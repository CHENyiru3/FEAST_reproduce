"""Render the retained Study 04A parameter-cloud visualizations.

Axes are fixed to log(mean), log(variance), and raw zero proportion.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/feast-reproduce-matplotlib")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SIMULATION_ROOT = REPOSITORY_ROOT / "04_batch_effect_removal" / "outputs" / "final_rerun_20260718_v2" / "simulations"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "same_slice"
MODES = ("shift_only", "diagonal_affine")
ALPHAS = (0.0, 0.5, 1.0)
ALPHA_COLORS = {0.0: "#6E6E6E", 0.5: "#0072B2", 1.0: "#D55E00"}
REFERENCE_DISPLAY_REGION = {
    "log_mean_reference": (-3.0, -1.5),
    "log_variance_reference": (-2.25, -1.35),
    "zero_proportion_reference": (0.80, 0.91),
}
DISPLAY_LIMITS = {
    "log_mean": (-4.3, -1.2),
    "log_variance": (-3.5, -0.5),
    "zero_proportion": (0.72, 1.02),
}


def source_path(mode: str, alpha: float) -> Path:
    path = SIMULATION_ROOT / mode / f"alpha_{alpha:.2f}.h5ad"
    if not path.is_file():
        raise FileNotFoundError(f"Study 04A simulation is missing: {path}")
    return path


def load_cloud(mode: str, alpha: float) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """Load stored gene parameters and express them on the retained axes."""
    path = source_path(mode, alpha)
    dataset = ad.read_h5ad(path, backed="r")
    try:
        columns = [f"{coordinate}_{suffix}" for coordinate in ("theta_mu", "theta_omega", "theta_pi0") for suffix in ("ref", "batch")]
        missing = sorted(set(columns) - set(dataset.var.columns))
        if missing:
            raise ValueError(f"Parameter-cloud coordinates are missing from {path}: {missing}")
        deformation = dataset.uns.get("batch_deformation")
        if not isinstance(deformation, dict) or not np.isclose(float(deformation["alpha"]), alpha):
            raise ValueError(f"Batch-deformation metadata is invalid in {path}")
        cloud = dataset.var.loc[:, columns].copy()
    finally:
        dataset.file.close()
    cloud.columns = [column.replace("_batch", "").replace("_ref", "_reference") for column in cloud.columns]
    for suffix in ("_reference", ""):
        log_mean = cloud[f"theta_mu{suffix}"].to_numpy(dtype=float)
        cloud[f"log_mean{suffix}"] = log_mean
        cloud[f"log_variance{suffix}"] = log_mean + cloud[f"theta_omega{suffix}"].to_numpy(dtype=float)
        cloud[f"zero_proportion{suffix}"] = 1.0 / (1.0 + np.exp(-cloud[f"theta_pi0{suffix}"].to_numpy(dtype=float)))
    numeric = cloud.loc[:, ["log_mean", "log_variance", "zero_proportion"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite parameter-cloud coordinate in {path}")
    return cloud, {"D": [float(value) for value in deformation["D"]], "b": [float(value) for value in deformation["b"]]}


def configure_matplotlib() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12.5, "axes.labelsize": 12.5, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


def save_figure(figure: plt.Figure, stem: str) -> list[Path]:
    outputs = []
    for suffix, kwargs in {
        ".pdf": {"metadata": {"Creator": "FEAST Study 04 plot_parameter_cloud.py", "CreationDate": None, "ModDate": None}},
        ".svg": {"metadata": {"Creator": "FEAST Study 04 plot_parameter_cloud.py", "Date": "1970-01-01T00:00:00Z"}},
        ".png": {"dpi": 600, "metadata": {"Software": "FEAST Study 04 plot_parameter_cloud.py"}},
    }.items():
        output = OUTPUT_DIR / f"{stem}{suffix}"
        figure.savefig(output, bbox_inches="tight", pad_inches=0.08, **kwargs)
        outputs.append(output)
    plt.close(figure)
    return outputs


def draw_batch_simulation_3d(clouds: dict[tuple[str, float], pd.DataFrame]) -> list[Path]:
    configure_matplotlib()
    reference = clouds[("shift_only", 0.0)]
    selected = np.logical_and.reduce([
        reference[coordinate].between(lower, upper).to_numpy()
        for coordinate, (lower, upper) in REFERENCE_DISPLAY_REGION.items()
    ])
    selected_index = reference.index[selected]
    if len(selected_index) == 0:
        raise ValueError("The illustrative reference-cloud region contains no genes")
    figure = plt.figure(figsize=(15.0, 9.2))
    for row, mode in enumerate(MODES):
        descriptor = "translation only" if mode == "shift_only" else "translation + axis-specific scaling"
        for column, alpha in enumerate(ALPHAS):
            axis = figure.add_subplot(2, 3, row * len(ALPHAS) + column + 1, projection="3d")
            cloud = clouds[(mode, alpha)].loc[selected_index]
            axis.scatter(cloud["log_mean"], cloud["log_variance"], cloud["zero_proportion"], s=2.2, color=ALPHA_COLORS[alpha], alpha=0.18, depthshade=False, linewidths=0, rasterized=True)
            axis.set(xlim=DISPLAY_LIMITS["log_mean"], ylim=DISPLAY_LIMITS["log_variance"], zlim=DISPLAY_LIMITS["zero_proportion"])
            axis.set_xlabel("log mean", labelpad=5)
            axis.set_ylabel("log variance", labelpad=5)
            axis.set_zlabel("zero proportion", labelpad=5)
            axis.view_init(elev=12, azim=-45)
            axis.set_title(f"{mode.replace('_', ' ')}\nα = {alpha:g}", fontsize=12.5, fontweight="bold", pad=7, loc="center")
            axis.grid(True, color="#E4E4E4", linewidth=0.5)
        figure.text(0.012, 0.69 - row * 0.39, descriptor, rotation=90, va="center", ha="center", fontsize=10, color="#4D4D4D")
    figure.suptitle("FEAST batch-effect simulation: illustrative parameter-cloud region", fontsize=15.5, fontweight="bold", y=0.985)
    figure.text(0.5, 0.935, f"Same {len(selected_index):,} genes at each α; selected in the reference cloud. Every panel has identical axes and view.", ha="center", fontsize=11.5, color="#4D4D4D")
    figure.subplots_adjust(left=0.035, right=0.99, bottom=0.03, top=0.84, hspace=0.18, wspace=0.02)
    return save_figure(figure, "parameter_cloud_batch_simulation_3d")


def draw_original_reference_cloud(cloud: pd.DataFrame) -> list[Path]:
    configure_matplotlib()
    figure = plt.figure(figsize=(8.1, 7.0))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(cloud["log_mean_reference"], cloud["log_variance_reference"], cloud["zero_proportion_reference"], s=2.6, color="#0072B2", alpha=0.15, depthshade=False, linewidths=0, rasterized=True)
    axis.set_xlabel("log mean expression", labelpad=10)
    axis.set_ylabel("log variance", labelpad=10)
    axis.set_zlabel("zero proportion", labelpad=10)
    axis.view_init(elev=21, azim=-60)
    figure.suptitle("Original real parameter cloud", fontsize=16, fontweight="bold", y=0.98)
    figure.text(0.5, 0.925, "Reference slice 151673; one point per gene (19,140 genes). No filtering or simulated deformation.", ha="center", fontsize=11.5, color="#4D4D4D")
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.84)
    return save_figure(figure, "original_reference_parameter_cloud_hybrid_scale_3d")


def write_provenance(deformations: dict[str, dict[str, list[float]]], outputs: list[Path]) -> Path:
    payload = {
        "schema_version": 1,
        "figure_set": "study04a_parameter_cloud_batch_simulation",
        "scientific_scope": "Illustrates the already generated Study 04A batch simulation; no correction-method result or ranking is plotted.",
        "source_simulations": [str(source_path(mode, alpha).relative_to(REPOSITORY_ROOT)) for mode in MODES for alpha in ALPHAS],
        "coordinates": ["log(mean)", "log(variance)", "zero proportion (raw scale)"],
        "modes": deformations,
        "alphas": list(ALPHAS),
        "rendering": {"original_reference_all_genes_plotted": True, "batch_simulation_illustrative_reference_region": REFERENCE_DISPLAY_REGION, "batch_simulation_display_limits": DISPLAY_LIMITS, "point_cloud_rasterized_in_vector_exports": True, "shared_axis_limits_in_batch_simulation": True},
        "outputs": [str(path.relative_to(REPOSITORY_ROOT)) for path in outputs],
        "plot_script": str(Path(__file__).relative_to(REPOSITORY_ROOT)),
    }
    output = OUTPUT_DIR / "parameter_cloud_batch_simulation_provenance.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clouds: dict[tuple[str, float], pd.DataFrame] = {}
    deformations: dict[str, dict[str, list[float]]] = {}
    for mode in MODES:
        for alpha in ALPHAS:
            cloud, deformation = load_cloud(mode, alpha)
            clouds[(mode, alpha)] = cloud
            deformations[mode] = deformation
    reference = clouds[("shift_only", 0.0)][["log_mean_reference", "log_variance_reference", "zero_proportion_reference"]]
    for cloud in clouds.values():
        if not reference.equals(cloud[["log_mean_reference", "log_variance_reference", "zero_proportion_reference"]]):
            raise ValueError("Reference parameter cloud changes across registered Study 04A simulations")
    outputs = draw_batch_simulation_3d(clouds)
    outputs.extend(draw_original_reference_cloud(clouds[("shift_only", 0.0)]))
    provenance = write_provenance(deformations, outputs)
    for path in outputs + [provenance]:
        print(path)


if __name__ == "__main__":
    main()
