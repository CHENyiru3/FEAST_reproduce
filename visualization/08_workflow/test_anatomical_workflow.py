"""Geometry checks for the affine-aware workflow renderer."""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_anatomical_workflow import (  # noqa: E402
    DEVCCF_ROOT,
    apply_affine,
    extract_mesh,
    generated_path,
    nifti_data,
    render_atlas,
    study06_display_positions,
)


def test_devccf_affine_matches_coordinate_table() -> None:
    volume_path = DEVCCF_ROOT / "E15.5_broad_region_annotations.nii.gz"
    coordinate_path = DEVCCF_ROOT / "E15.5_nonbackground_voxels.tsv.gz"
    volume, affine = nifti_data(volume_path)
    with gzip.open(coordinate_path, "rt", encoding="utf-8") as handle:
        frame = pd.read_csv(handle, sep="\t", nrows=64)
    indices = frame[["voxel_i", "voxel_j", "voxel_k"]].to_numpy(int)
    expected = frame[["x", "y", "z"]].to_numpy(float)
    assert np.allclose(apply_affine(indices, affine), expected, atol=1e-7)
    observed_labels = volume[indices[:, 0], indices[:, 1], indices[:, 2]]
    assert np.array_equal(observed_labels, frame["broad_region_id"].to_numpy(int))


def test_generated_plane_uses_same_physical_coordinates() -> None:
    _volume, affine = nifti_data(DEVCCF_ROOT / "E15.5_broad_region_annotations.nii.gz")
    data = ad.read_h5ad(generated_path("E15.5", 78), backed="r")
    try:
        positions = np.linspace(0, data.n_obs - 1, 128, dtype=int)
        indices = data.obs.iloc[positions][["voxel_i", "voxel_j", "voxel_k"]].to_numpy(int)
        expected = np.asarray(data.obsm["spatial_3d"])[positions]
    finally:
        data.file.close()
    assert np.allclose(apply_affine(indices, affine), expected, atol=1e-7)


def test_mesh_vertices_are_transformed_to_world_space() -> None:
    volume = np.zeros((9, 10, 11), dtype=np.uint8)
    volume[2:7, 3:8, 4:9] = 1
    affine = np.array([
        [0.02, 0.00, 0.00, 1.0],
        [0.00, 0.00, 0.02, 2.0],
        [0.00, -0.02, 0.00, 3.0],
        [0.00, 0.00, 0.00, 1.0],
    ])
    mesh = extract_mesh(volume, affine, (1,), step_size=1, smoothing_iterations=0)
    assert np.allclose(mesh.bounds[0], [1.03, 2.07, 2.85])
    assert np.allclose(mesh.bounds[1], [1.13, 2.17, 2.95])


def test_atlas_before_after_use_identical_view_geometry() -> None:
    shell = trimesh.creation.box(extents=(2.0, 3.0, 4.0))
    figure = plt.figure()
    before = figure.add_subplot(121, projection="3d")
    after = figure.add_subplot(122, projection="3d")
    try:
        render_atlas(before, shell, {}, {}, plane=None, expression_free=True)
        render_atlas(after, shell, {}, {}, plane=None, expression_free=False)
        assert np.allclose(before.get_w_lims(), after.get_w_lims())
        assert before.elev == after.elev
        assert before.azim == after.azim
    finally:
        plt.close(figure)


def test_native_stack_expands_only_the_target_bracket() -> None:
    positions = study06_display_positions([80, 81, 82, 83, 84, 85, 86], 83)
    assert np.isclose(positions[81] - positions[80], 0.65)
    assert np.isclose(positions[83] - positions[82], 1.25)
    assert np.isclose(positions[84] - positions[83], 1.25)
    assert np.isclose(positions[86] - positions[85], 0.65)
