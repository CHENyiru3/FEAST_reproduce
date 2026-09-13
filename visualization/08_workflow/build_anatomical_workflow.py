"""Build affine-aware anatomical components and the FEAST workflow abstract."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
from matplotlib import colors as mcolors
from matplotlib import patches
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

from build_components import (
    DEVCCF_ROOT,
    GENERATED,
    INK,
    SOURCE_ROOT,
    STUDY06,
    dense_counts,
    find_study06_case,
    generated_path,
    load_h5ad,
    load_schema,
    nifti_data,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "figures" / "anatomical_workflow_review_v6"
DEFAULT_COMPONENT_OUTPUT = HERE / "figures" / "anatomical_workflow_components_v6"
CONTEXT = "#B8C5CD"
PANEL_FACE = "#F5F8FA"
FEAST_FACE = "#E8F4FA"
VIRIDIS = mpl.colormaps["viridis"]


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8.5,
        "figure.facecolor": "white",
        "axes.facecolor": "none",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.color": INK,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--study06-target", type=int, default=83)
    parser.add_argument("--study07-age", choices=("E15.5", "E18.5"), default="E15.5")
    parser.add_argument("--study07-plane", type=int, default=78)
    parser.add_argument("--render-mode", choices=("glass-shell",), default="glass-shell")
    parser.add_argument("--components-only", action="store_true", help="write only standalone component pages")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def export(figure: plt.Figure, output: Path, stem: str, dpi: int) -> list[Path]:
    paths: list[Path] = []
    metadata = {"Title": stem, "Creator": "FEAST anatomical workflow builder"}
    for suffix in (".pdf", ".svg", ".png"):
        path = output / f"{stem}{suffix}"
        kwargs: dict[str, Any] = {"dpi": dpi, "bbox_inches": "tight", "pad_inches": 0.03}
        if suffix == ".pdf":
            kwargs["metadata"] = {**metadata, "CreationDate": None, "ModDate": None}
        elif suffix == ".svg":
            kwargs["metadata"] = {**metadata, "Date": None}
        else:
            kwargs["metadata"] = {"Software": metadata["Creator"]}
        figure.savefig(path, **kwargs)
        paths.append(path)
    plt.close(figure)
    return paths


def apply_affine(indices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    values = np.asarray(indices, dtype=float)
    homogeneous = np.column_stack([values, np.ones(len(values), dtype=float)])
    return (homogeneous @ np.asarray(affine, dtype=float).T)[:, :3]


def label_mask(volume: np.ndarray, labels: Iterable[int]) -> np.ndarray:
    selected = tuple(int(label) for label in labels)
    if len(selected) == 1:
        return volume == selected[0]
    return np.isin(volume, selected)


def mask_bounds(mask: np.ndarray, pad: int = 2) -> tuple[slice, slice, slice]:
    occupied = []
    for axis in range(3):
        reduced = mask.any(axis=tuple(other for other in range(3) if other != axis))
        locations = np.flatnonzero(reduced)
        if not len(locations):
            raise ValueError("cannot extract a surface from an empty mask")
        occupied.append(slice(max(int(locations[0]) - pad, 0), min(int(locations[-1]) + pad + 1, mask.shape[axis])))
    return tuple(occupied)  # type: ignore[return-value]


def extract_mesh(
    volume: np.ndarray,
    affine: np.ndarray,
    labels: Iterable[int],
    *,
    step_size: int = 2,
    smoothing_iterations: int = 5,
    minimum_component_faces: int = 200,
) -> trimesh.Trimesh:
    mask = label_mask(volume, labels)
    bounds = mask_bounds(mask)
    cropped = mask[bounds]
    vertices, faces, _normals, _values = marching_cubes(
        cropped.astype(np.uint8),
        level=0.5,
        step_size=step_size,
        allow_degenerate=False,
    )
    offset = np.array([part.start for part in bounds], dtype=float)
    world = apply_affine(vertices + offset, affine)
    mesh = trimesh.Trimesh(vertices=world, faces=faces, process=True)
    components = [part for part in mesh.split(only_watertight=False) if len(part.faces) >= minimum_component_faces]
    if components:
        mesh = trimesh.util.concatenate(components)
    if smoothing_iterations:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.35, nu=0.36, iterations=smoothing_iterations)
    del mask, cropped
    return mesh


def mesh_collection(mesh: trimesh.Trimesh, color: str, alpha: float, *, rasterized: bool = True) -> Poly3DCollection:
    base = np.asarray(mcolors.to_rgba(color), dtype=float)
    light = np.asarray([0.45, -0.65, 0.62], dtype=float)
    light /= np.linalg.norm(light)
    illumination = 0.58 + 0.42 * np.clip(np.asarray(mesh.face_normals) @ light, 0, 1)
    facecolors = np.tile(base, (len(mesh.faces), 1))
    facecolors[:, :3] *= illumination[:, None]
    facecolors[:, 3] = alpha
    collection = Poly3DCollection(
        mesh.vertices[mesh.faces],
        facecolors=facecolors,
        edgecolors="none",
        linewidth=0,
        rasterized=rasterized,
    )
    return collection


def set_physical_3d_view(axis: plt.Axes, bounds: np.ndarray, *, elev: float = 27, azim: float = -60) -> None:
    lower, upper = np.asarray(bounds, dtype=float)
    span = np.maximum(upper - lower, 1e-9)
    margin = span * 0.035
    axis.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
    axis.set_ylim(lower[1] - margin[1], upper[1] + margin[1])
    axis.set_zlim(lower[2] - margin[2], upper[2] + margin[2])
    axis.set_box_aspect(span)
    axis.view_init(elev=elev, azim=azim)
    axis.set_proj_type("persp", focal_length=0.9)
    axis.set_axis_off()


def library_values(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    item = load_h5ad(path, counts=True)
    return item, np.asarray(item["library"], dtype=float)


def atlas_plane_texture(path: Path, affine: np.ndarray) -> dict[str, Any]:
    data = ad.read_h5ad(path)
    try:
        required = {"voxel_i", "voxel_j", "voxel_k", "region"}
        if required - set(data.obs.columns):
            raise ValueError(f"generated atlas plane lacks columns: {sorted(required - set(data.obs.columns))}")
        counts = dense_counts(data)
        values = np.log1p(counts.sum(axis=1)).astype(float)
        indices = data.obs[["voxel_i", "voxel_j", "voxel_k"]].to_numpy(int)
        world = apply_affine(indices, affine)
        if not np.allclose(world, np.asarray(data.obsm["spatial_3d"], dtype=float), atol=2.1e-2):
            raise ValueError("generated atlas voxel indices do not match the NIfTI affine")
        fixed_j = int(np.unique(indices[:, 1]).item())
        finite = np.isfinite(values)
        norm = Normalize(vmin=float(np.quantile(values[finite], 0.02)), vmax=float(np.quantile(values[finite], 0.98)), clip=True)
        return {
            "centers": world,
            "values": values,
            "affine": np.asarray(affine, dtype=float),
            "norm": norm,
            "n_spots": int(data.n_obs),
            "z_world": float(world[0, 2]),
            "fixed_j": fixed_j,
        }
    finally:
        del data


def add_texture_plane(axis: plt.Axes, plane: dict[str, Any], *, z_lift: float = 0.01) -> None:
    centers = np.asarray(plane["centers"], dtype=float)
    affine = np.asarray(plane["affine"], dtype=float)
    half_i = affine[:3, 0] * 0.49
    half_k = affine[:3, 2] * 0.49
    normal = affine[:3, 1] / np.linalg.norm(affine[:3, 1])
    centers = centers + normal * z_lift
    polygons = np.stack([
        centers - half_i - half_k,
        centers + half_i - half_k,
        centers + half_i + half_k,
        centers - half_i + half_k,
    ], axis=1)
    rgba = VIRIDIS(plane["norm"](plane["values"]))
    rgba[:, 3] = 0.98
    axis.add_collection3d(Poly3DCollection(
        polygons,
        facecolors=rgba,
        edgecolors="none",
        linewidth=0,
        rasterized=True,
    ))


def render_atlas(
    axis: plt.Axes,
    shell: trimesh.Trimesh,
    region_meshes: dict[str, trimesh.Trimesh],
    region_colors: dict[str, str],
    *,
    plane: dict[str, Any] | None,
    expression_free: bool,
    show_context_shell: bool = True,
) -> None:
    if not expression_free:
        for region, mesh in region_meshes.items():
            axis.add_collection3d(mesh_collection(mesh, region_colors[region], 0.45))
    if show_context_shell:
        axis.add_collection3d(mesh_collection(shell, CONTEXT, 0.48))
    if plane is not None:
        add_texture_plane(axis, plane)
    bounds = shell.bounds if show_context_shell else np.array([
        np.min([mesh.bounds[0] for mesh in region_meshes.values()], axis=0),
        np.max([mesh.bounds[1] for mesh in region_meshes.values()], axis=0),
    ])
    set_physical_3d_view(axis, bounds)


def expression_grid(
    xy: np.ndarray,
    values: np.ndarray,
    limits: tuple[tuple[float, float], tuple[float, float]],
    *,
    bins: int = 180,
) -> dict[str, Any]:
    x_edges = np.linspace(*limits[0], bins + 1)
    y_edges = np.linspace(*limits[1], bins + 1)
    weighted, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(x_edges, y_edges), weights=values)
    count, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(x_edges, y_edges))
    smooth_weighted = gaussian_filter(weighted, sigma=1.0)
    smooth_count = gaussian_filter(count, sigma=1.0)
    texture = np.divide(smooth_weighted, smooth_count, out=np.zeros_like(smooth_weighted), where=smooth_count > 1e-3)
    alpha = np.clip(smooth_count / np.quantile(smooth_count[smooth_count > 0], 0.55), 0, 1)
    alpha[smooth_count < 0.08] = 0
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    return {"x": x_centers, "y": y_centers, "texture": texture, "alpha": alpha}


def geometry_grid(
    xy: np.ndarray,
    limits: tuple[tuple[float, float], tuple[float, float]],
    *,
    bins: int = 180,
) -> dict[str, Any]:
    return expression_grid(xy, np.ones(len(xy), dtype=float), limits, bins=bins)


def add_grid_plane(
    axis: plt.Axes,
    grid: dict[str, Any],
    display_z: float,
    norm: Normalize,
    *,
    color: str | None = None,
    alpha_scale: float = 1.0,
) -> None:
    x, y = np.meshgrid(grid["x"], grid["y"], indexing="ij")
    z = np.full_like(x, display_z, dtype=float)
    if color is None:
        rgba = VIRIDIS(norm(grid["texture"]))
    else:
        rgba = np.broadcast_to(mcolors.to_rgba(color), (*grid["texture"].shape, 4)).copy()
    rgba[..., 3] = grid["alpha"] * alpha_scale
    axis.plot_surface(x, y, z, facecolors=rgba, rstride=1, cstride=1, linewidth=0, shade=False, antialiased=False, rasterized=True)


def add_grid_outline(axis: plt.Axes, grid: dict[str, Any], display_z: float, *, color: str, linewidth: float) -> None:
    axis.contour(
        grid["x"],
        grid["y"],
        grid["alpha"].T,
        levels=(0.15,),
        zdir="z",
        offset=display_z + 0.012,
        colors=(color,),
        linewidths=linewidth,
    )


def study06_data(target_slice: int) -> dict[str, Any]:
    target, lower_path, upper_path, generated_path_value = find_study06_case(target_slice)
    lower, lower_values = library_values(lower_path)
    upper, upper_values = library_values(upper_path)
    generated, generated_values = library_values(generated_path_value)
    plan = json.loads((STUDY06 / "outputs" / "final" / "plan.json").read_text(encoding="utf-8"))
    data_dir = Path(plan["data_dir"])
    blueprint_path = data_dir / f"Zhuang-ABCA-1.{target_slice:03d}.h5ad"
    blueprint = load_h5ad(blueprint_path, counts=False)
    context_slices = [slice_id for slice_id in range(target_slice - 3, target_slice + 4) if (data_dir / f"Zhuang-ABCA-1.{slice_id:03d}.h5ad").is_file()]
    context_items: dict[int, dict[str, Any]] = {}
    context_values: dict[int, np.ndarray] = {}
    for slice_id in context_slices:
        if slice_id == target_slice:
            continue
        if slice_id == int(target["lower_ref_slice"]):
            context_items[slice_id], context_values[slice_id] = lower, lower_values
        elif slice_id == int(target["upper_ref_slice"]):
            context_items[slice_id], context_values[slice_id] = upper, upper_values
        else:
            context_items[slice_id], context_values[slice_id] = library_values(data_dir / f"Zhuang-ABCA-1.{slice_id:03d}.h5ad")
    all_xy = np.vstack([blueprint["xy"], *(item["xy"] for item in context_items.values())])
    pad = (all_xy.max(axis=0) - all_xy.min(axis=0)) * 0.025
    limits = (
        (float(all_xy[:, 0].min() - pad[0]), float(all_xy[:, 0].max() + pad[0])),
        (float(all_xy[:, 1].min() - pad[1]), float(all_xy[:, 1].max() + pad[1])),
    )
    all_values = np.concatenate([*context_values.values(), generated_values])
    norm = Normalize(vmin=float(np.quantile(all_values, 0.02)), vmax=float(np.quantile(all_values, 0.98)), clip=True)
    return {
        "target": target,
        "lower": lower,
        "upper": upper,
        "blueprint": blueprint,
        "generated": generated,
        "lower_grid": expression_grid(lower["xy"], lower_values, limits),
        "upper_grid": expression_grid(upper["xy"], upper_values, limits),
        "blueprint_grid": geometry_grid(blueprint["xy"], limits),
        "generated_grid": expression_grid(generated["xy"], generated_values, limits),
        "context_slices": context_slices,
        "context_grids": {slice_id: expression_grid(item["xy"], context_values[slice_id], limits) for slice_id, item in context_items.items()},
        "limits": limits,
        "norm": norm,
    }


def study06_display_positions(context_slices: list[int], target_slice: int) -> dict[int, float]:
    steps = np.full(max(len(context_slices) - 1, 0), 0.65, dtype=float)
    target_index = context_slices.index(target_slice)
    if 0 < target_index < len(context_slices) - 1:
        steps[target_index - 1:target_index + 1] = 1.25
    positions = np.concatenate([[0.0], np.cumsum(steps)])
    return dict(zip(context_slices, positions - positions[target_index]))


def render_study06_stack(axis: plt.Axes, data: dict[str, Any], *, include_blueprint: bool, include_generated: bool) -> None:
    context_slices = data["context_slices"]
    positions = study06_display_positions(context_slices, int(data["target"]["target_slice"]))
    lower_slice = int(data["target"]["lower_ref_slice"])
    upper_slice = int(data["target"]["upper_ref_slice"])
    target_slice = int(data["target"]["target_slice"])
    for slice_id, grid in data["context_grids"].items():
        if slice_id not in (lower_slice, upper_slice):
            add_grid_plane(axis, grid, positions[slice_id], data["norm"], alpha_scale=0.58)
            add_grid_outline(axis, grid, positions[slice_id], color="#AAB7C0", linewidth=0.65)
    add_grid_plane(axis, data["lower_grid"], positions[lower_slice], data["norm"], alpha_scale=0.96)
    if include_generated:
        add_grid_plane(axis, data["generated_grid"], positions[target_slice], data["norm"], alpha_scale=0.98)
        add_grid_outline(axis, data["generated_grid"], positions[target_slice], color=GENERATED, linewidth=1.25)
    elif include_blueprint:
        add_grid_outline(axis, data["blueprint_grid"], positions[target_slice], color="#AAB7C0", linewidth=1.15)
    add_grid_plane(axis, data["upper_grid"], positions[upper_slice], data["norm"], alpha_scale=0.96)
    xlim, ylim = data["limits"]
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_zlim(min(positions.values()) - 0.3, max(positions.values()) + 0.3)
    axis.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], 5.7))
    axis.view_init(elev=24, azim=-57)
    axis.set_proj_type("persp", focal_length=0.9)
    axis.set_axis_off()


def categorical_source_image(path: Path, colors: dict[str, str], *, bins: int = 220) -> np.ndarray:
    data = ad.read_h5ad(path, backed="r")
    try:
        xy = np.asarray(data.obsm["spatial"], dtype=float)
        labels = data.obs["region"].astype(str).to_numpy()
    finally:
        data.file.close()
    x_edges = np.linspace(xy[:, 0].min(), xy[:, 0].max(), bins + 1)
    y_edges = np.linspace(xy[:, 1].min(), xy[:, 1].max(), bins + 1)
    ordered_labels = sorted(set(labels))
    scores = []
    for label in ordered_labels:
        selected = labels == label
        counts, _, _ = np.histogram2d(xy[selected, 0], xy[selected, 1], bins=(x_edges, y_edges))
        scores.append(gaussian_filter(counts, sigma=1.15))
    score_stack = np.stack(scores)
    winners = np.argmax(score_stack, axis=0)
    support = score_stack.sum(axis=0)
    image = np.zeros((bins, bins, 4), dtype=float)
    for index, label in enumerate(ordered_labels):
        image[winners == index] = mcolors.to_rgba(colors.get(label, "#D9D9D9"))
    positive = support[support > 0]
    image[..., 3] = np.clip(support / np.quantile(positive, 0.30), 0, 1)
    image[support < 0.025, 3] = 0
    return np.rot90(image)


def add_panel_box(figure: plt.Figure, xywh: tuple[float, float, float, float]) -> None:
    x, y, width, height = xywh
    figure.add_artist(patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        transform=figure.transFigure,
        facecolor=PANEL_FACE,
        edgecolor="#D7E0E5",
        linewidth=1.0,
        zorder=-10,
    ))


def add_arrow(figure: plt.Figure, start: tuple[float, float], end: tuple[float, float]) -> None:
    figure.add_artist(patches.FancyArrowPatch(
        start,
        end,
        transform=figure.transFigure,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.5,
        color="#758792",
        connectionstyle="arc3,rad=0",
    ))


def add_feast_node(figure: plt.Figure, xywh: tuple[float, float, float, float]) -> None:
    x, y, width, height = xywh
    figure.add_artist(patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.01,rounding_size=0.028",
        transform=figure.transFigure,
        facecolor=FEAST_FACE,
        edgecolor=GENERATED,
        linewidth=1.5,
    ))
    figure.text(x + width / 2, y + height * 0.60, "FEAST", ha="center", va="center", fontsize=13, weight="bold", color=GENERATED)
    figure.text(x + width / 2, y + height * 0.27, "conditional transfer", ha="center", va="center", fontsize=7.2, color="#567383")


def atlas_spec(age: str) -> dict[str, Any]:
    if age == "E15.5":
        return {
            "volume": "E15.5_broad_region_annotations.nii.gz",
            "source_glob": "*E14*.h5ad",
            "source_label": "10 E14 sections",
            "levels": 158,
        }
    return {
        "volume": "E18.5_broad_region_annotations.nii.gz",
        "source_glob": "*E18M*.h5ad",
        "source_label": "5 E18 sections",
        "levels": 202,
    }


def build_atlas_geometry(age: str, plane_index: int) -> dict[str, Any]:
    spec = atlas_spec(age)
    volume, affine = nifti_data(DEVCCF_ROOT / spec["volume"])
    colors = load_schema()
    shell = extract_mesh(volume, affine, range(1, 10))
    region_meshes = {
        region: extract_mesh(volume, affine, (region_id,))
        for region_id, region in enumerate(
            ("Background", "DorsalVZ", "IZ", "CP", "VentralVZ", "GE", "SeptalVZ", "Septum", "BT")
        )
        if region_id > 0 and np.any(volume == region_id)
    }
    plane_path = generated_path(age, plane_index)
    plane = atlas_plane_texture(plane_path, affine)
    return {
        "spec": spec,
        "volume": volume,
        "affine": affine,
        "colors": colors,
        "shell": shell,
        "region_meshes": region_meshes,
        "plane": plane,
        "plane_path": plane_path,
        "plane_index": int(plane_index),
    }


def standalone_assets(output: Path, study06: dict[str, Any], atlas: dict[str, Any], age: str, dpi: int) -> list[Path]:
    outputs: list[Path] = []
    figure = plt.figure(figsize=(4.6, 3.8))
    axis = figure.add_subplot(111, projection="3d")
    render_study06_stack(axis, study06, include_blueprint=True, include_generated=False)
    axis.set_title("Direct references embedded in a native stack", weight="bold", pad=0)
    outputs += export(figure, output, "study06_anatomical_input", dpi)

    figure = plt.figure(figsize=(4.6, 3.8))
    axis = figure.add_subplot(111, projection="3d")
    render_study06_stack(axis, study06, include_blueprint=False, include_generated=True)
    axis.set_title("Generated target within the native stack", weight="bold", pad=0)
    outputs += export(figure, output, "study06_anatomical_output", dpi)

    age_stem = age.replace(".", "_")
    spec = atlas["spec"]
    source_paths = sorted(SOURCE_ROOT.glob(spec["source_glob"]))
    representative = source_paths[len(source_paths) // 2]
    figure = plt.figure(figsize=(4.8, 4.2))
    axis = figure.add_axes((0.12, 0.15, 0.76, 0.70))
    axis.imshow(categorical_source_image(representative, atlas["colors"]))
    axis.set_axis_off()
    figure.text(0.5, 0.94, f"Measured source cohort: {spec['source_label']}", ha="center", weight="bold", fontsize=11)
    figure.text(0.5, 0.055, "Representative region-labeled section; cohort count shown above", ha="center", fontsize=8, color="#667781")
    outputs += export(figure, output, f"study07_{age_stem}_source_cohort", dpi)

    figure = plt.figure(figsize=(3.5, 2.1))
    add_feast_node(figure, (0.12, 0.27, 0.76, 0.46))
    outputs += export(figure, output, "feast_conditional_transfer_module", dpi)

    figure = plt.figure(figsize=(4.8, 4.2))
    axis = figure.add_subplot(111, projection="3d")
    render_atlas(axis, atlas["shell"], atlas["region_meshes"], atlas["colors"], plane=None, expression_free=True)
    axis.set_title(f"Expression-free DevCCF {age} geometry", weight="bold", pad=0)
    outputs += export(figure, output, f"study07_{age_stem}_atlas_shell", dpi)

    figure = plt.figure(figsize=(4.8, 4.2))
    axis = figure.add_subplot(111, projection="3d")
    render_atlas(axis, atlas["shell"], atlas["region_meshes"], atlas["colors"], plane=atlas["plane"], expression_free=False, show_context_shell=False)
    axis.set_title(f"Final generated {age} atlas support", weight="bold", pad=0)
    scalar = mpl.cm.ScalarMappable(norm=atlas["plane"]["norm"], cmap=VIRIDIS)
    colorbar = figure.colorbar(scalar, ax=axis, fraction=0.035, pad=0.005, shrink=0.58)
    colorbar.set_label("log1p total counts", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    outputs += export(figure, output, f"study07_{age_stem}_generated_atlas", dpi)
    return outputs


def graphical_abstract(output: Path, study06: dict[str, Any], atlas: dict[str, Any], age: str, dpi: int) -> list[Path]:
    figure = plt.figure(figsize=(16.0, 8.3), facecolor="white")
    figure.text(0.04, 0.965, "FEAST reconstructs spatial expression across missing tissue and atlas coordinates", fontsize=18, weight="bold", va="top")
    figure.text(0.04, 0.925, "Two conditional 3D settings, distinguished by whether target expression is observed", fontsize=10, color="#667781", va="top")
    add_panel_box(figure, (0.035, 0.515, 0.93, 0.37))
    add_panel_box(figure, (0.035, 0.075, 0.93, 0.37))

    figure.text(0.055, 0.855, "Native-stack reconstruction", fontsize=13, weight="bold")
    figure.text(0.055, 0.826, "Held-out expression exists for evaluation", fontsize=8.5, color="#667781")
    input06 = figure.add_axes((0.055, 0.555, 0.25, 0.245), projection="3d")
    render_study06_stack(input06, study06, include_blueprint=True, include_generated=False)
    figure.text(0.18, 0.545, "measured 082 / 084 + outline-only target 083", ha="center", fontsize=8)
    add_feast_node(figure, (0.38, 0.635, 0.115, 0.095))
    output06 = figure.add_axes((0.56, 0.555, 0.25, 0.245), projection="3d")
    render_study06_stack(output06, study06, include_blueprint=False, include_generated=True)
    figure.text(0.685, 0.545, "generated 083 (blue boundary)", ha="center", fontsize=8)
    figure.text(0.885, 0.70, "93", ha="center", fontsize=28, weight="bold", color=GENERATED)
    figure.text(0.885, 0.655, "validated targets", ha="center", fontsize=9)
    figure.text(0.885, 0.616, "49 dense  ·  29 medium  ·  15 sparse", ha="center", fontsize=7.5, color="#667781")
    figure.text(0.885, 0.575, "expanded 082–083–084 display gap", ha="center", fontsize=7.2, color="#87949B")
    add_arrow(figure, (0.31, 0.68), (0.375, 0.68))
    add_arrow(figure, (0.50, 0.68), (0.555, 0.68))

    spec = atlas["spec"]
    figure.text(0.055, 0.415, "DevCCF atlas transfer", fontsize=13, weight="bold")
    figure.text(0.055, 0.386, "Expression-free anatomical coordinates receive generated expression", fontsize=8.5, color="#667781")
    source_paths = sorted(SOURCE_ROOT.glob(spec["source_glob"]))
    representative = source_paths[len(source_paths) // 2]
    source_image = categorical_source_image(representative, atlas["colors"])
    source_axis = figure.add_axes((0.055, 0.125, 0.16, 0.22))
    source_axis.imshow(source_image)
    source_axis.set_axis_off()
    for offset in (0.010, 0.020):
        figure.add_artist(patches.FancyBboxPatch(
            (0.065 + offset, 0.135 + offset), 0.13, 0.18,
            boxstyle="round,pad=0.002,rounding_size=0.006",
            transform=figure.transFigure,
            facecolor="none",
            edgecolor="#9AAAB4",
            linewidth=0.8,
            zorder=-1,
        ))
    figure.text(0.135, 0.105, spec["source_label"], ha="center", fontsize=8)
    figure.text(0.235, 0.235, "+", ha="center", va="center", fontsize=19, color="#758792")
    atlas_input = figure.add_axes((0.255, 0.095, 0.22, 0.28), projection="3d")
    render_atlas(atlas_input, atlas["shell"], atlas["region_meshes"], atlas["colors"], plane=None, expression_free=True)
    figure.text(0.365, 0.105, f"expression-free DevCCF {age}", ha="center", fontsize=8)
    add_feast_node(figure, (0.505, 0.195, 0.115, 0.095))
    atlas_output = figure.add_axes((0.655, 0.095, 0.22, 0.28), projection="3d")
    render_atlas(atlas_output, atlas["shell"], atlas["region_meshes"], atlas["colors"], plane=atlas["plane"], expression_free=False, show_context_shell=False)
    figure.text(0.765, 0.105, "final generated atlas support", ha="center", fontsize=8)
    figure.text(0.915, 0.257, str(spec["levels"]), ha="center", fontsize=28, weight="bold", color=GENERATED)
    figure.text(0.915, 0.213, "atlas planes", ha="center", fontsize=9)
    figure.text(0.915, 0.176, "550 genes", ha="center", fontsize=8, color="#667781")
    figure.text(0.90, 0.143, "descriptive transfer", ha="center", fontsize=7.1, color="#87949B")
    figure.text(0.90, 0.121, "no target-expression truth", ha="center", fontsize=7.1, color="#87949B")
    add_arrow(figure, (0.48, 0.24), (0.50, 0.24))
    add_arrow(figure, (0.625, 0.24), (0.65, 0.24))

    return export(figure, output, "feast_workflow_graphical_abstract", dpi)


def write_records(output: Path, study06: dict[str, Any], atlas: dict[str, Any], age: str, outputs: list[Path]) -> list[Path]:
    mesh_rows = [{
        "object": "atlas_shell_labels_1_9",
        "vertices": int(len(atlas["shell"].vertices)),
        "faces": int(len(atlas["shell"].faces)),
    }]
    mesh_rows.extend({
        "object": f"atlas_region_{region}",
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
    } for region, mesh in atlas["region_meshes"].items())
    mesh_path = output / "mesh_geometry_summary.csv"
    pd.DataFrame(mesh_rows).to_csv(mesh_path, index=False)
    summary_path = output / "workflow_data_summary.csv"
    pd.DataFrame([
        {"workflow": "Study 06", "resource": "Allen Zhuang ABCA-1", "inputs": "147 sections", "display_case": "082 -> 083 -> 084", "outputs": "93 validated targets"},
        {"workflow": "Study 07", "resource": "GSE269617 + DevCCFv1", "inputs": atlas["spec"]["source_label"], "display_case": age, "outputs": f"{atlas['spec']['levels']} atlas planes"},
    ]).to_csv(summary_path, index=False)
    manifest_path = output / "figure_provenance.json"
    manifest = {
        "schema_version": 1,
        "figure_set": "study08_affine_aware_anatomical_workflow",
        "purpose": "graphical abstract and reusable data-derived anatomical components",
        "sources": {
            "study06_plan": str(STUDY06 / "outputs" / "final" / "plan.json"),
            "study07_nifti": str(DEVCCF_ROOT / atlas["spec"]["volume"]),
            "study07_region_schema": str(DEVCCF_ROOT / "gse269617_region_schema.tsv"),
            "study07_generated_plane": str(atlas["plane_path"]),
        },
        "study06": {
            "target_slice": int(study06["target"]["target_slice"]),
            "reference_slices": [int(study06["target"]["lower_ref_slice"]), int(study06["target"]["upper_ref_slice"])],
            "local_stack_context_slices": list(map(int, study06["context_slices"])),
            "context_expression_inputs": "only the two declared reference slices; other displayed observed slices provide local visual context only",
            "display_z": "exploded and explicitly labeled; display spacing is 0.65 for contextual neighbors and 1.25 around the 082–083–084 bracket; XY uses one shared physical coordinate system",
            "expression": "log1p total counts; 2nd-98th percentile scale pooled across the displayed measured and generated slices; pre-FEAST target is outline-only and generated target has a blue boundary",
        },
        "study07": {
            "age": age,
            "plane_index": int(atlas["plane_index"]),
            "plane_voxel_j": int(atlas["plane"]["fixed_j"]),
            "plane_z_world": float(atlas["plane"]["z_world"]),
            "shell_labels": list(range(1, 10)),
            "generated_region_labels": list(range(1, 9)),
            "generated_output_context_shell_rendered": False,
            "expression": "log1p total counts; central-plane 2nd-98th percentile scale; no interpolation into missing voxels",
            "geometry": "marching cubes in voxel space followed by NIfTI affine; step_size=2; mesh components below 200 faces omitted; Taubin smoothing iterations=5 for display only; input renders the full gray atlas shell while output frames only retained final-generation labels 1-8",
            "target_expression_accuracy_claim": False,
        },
        "scientific_disposition": {
            "figure_promotion_authorized": False,
            "target_expression_accuracy_claim_authorized": False,
            "winner_or_composite_claim_authorized": False,
        },
        "rendering": {
            "pdf_fonttype": 42,
            "svg_fonttype": "none",
            "png_dpi": 600,
            "dense_layers": "rasterized; text remains vector",
            "matplotlib": mpl.__version__,
            "numpy": np.__version__,
            "trimesh": trimesh.__version__,
        },
        "outputs": sorted(path.name for path in [*outputs, mesh_path, summary_path]),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return [mesh_path, summary_path, manifest_path]


def main() -> None:
    args = parse_args()
    output = (DEFAULT_COMPONENT_OUTPUT if args.components_only and args.output_dir == DEFAULT_OUTPUT else args.output_dir).resolve()
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(f"use a fresh output directory: {output}")
    else:
        output.mkdir(parents=True)
    configure_matplotlib()
    study06 = study06_data(int(args.study06_target))
    atlas = build_atlas_geometry(str(args.study07_age), int(args.study07_plane))
    outputs = standalone_assets(output, study06, atlas, str(args.study07_age), int(args.dpi))
    if not args.components_only:
        outputs += graphical_abstract(output, study06, atlas, str(args.study07_age), int(args.dpi))
    records = write_records(output, study06, atlas, str(args.study07_age), outputs)
    print(f"wrote {len(outputs)} figure files and {len(records)} records to {output}")


if __name__ == "__main__":
    main()
