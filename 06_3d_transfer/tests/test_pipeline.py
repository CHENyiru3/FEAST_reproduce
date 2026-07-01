#!/usr/bin/env python
"""Tests for the GSE269617 -> DevCCF 3D transfer pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------------------
# Paths and helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path("/maiziezhou_lab2/yiru")
FEAST_SRC = REPO_ROOT / "FEAST" / "src"
if FEAST_SRC.exists():
    sys.path.insert(0, str(FEAST_SRC))

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"

DEVCCF_SCHEMA = (
    Path("/maiziezhou_lab2/yiru/Datasets/Processed/DevCCFv1_figshare_26377171")
    / "coordinate_system/GSE269617_region_merged/gse269617_region_schema.tsv"
)

DEVCCF_VOLUME_E15 = (
    Path("/maiziezhou_lab2/yiru/Datasets/Processed/DevCCFv1_figshare_26377171")
    / "coordinate_system/GSE269617_region_merged/E15.5_broad_region_annotations.nii.gz"
)

BLUEPRINT_JSON = (
    Path(__file__).resolve().parents[1] / "outputs" / "E15.5_blueprints.json"
)

GSE269617_H5AD_DIR = (
    Path("/maiziezhou_lab2/yiru/Datasets/Processed/GSE269617/h5ad_region_annotated")
)

EXTRACT_SCRIPT = SCRIPTS_DIR / "extract_devccf_blueprints.py"
Z_REGULARIZE_SCRIPT = SCRIPTS_DIR / "z_regularize.py"


# ---------------------------------------------------------------------------
# Test 1: Blueprint extraction
# ---------------------------------------------------------------------------

def test_blueprint_json_exists_and_has_expected_keys():
    """Test that a blueprint JSON file exists and has the expected top-level structure."""
    if not BLUEPRINT_JSON.exists():
        pytest.skip(f"Blueprint JSON not found: {BLUEPRINT_JSON}")

    with open(BLUEPRINT_JSON, "r") as fh:
        payload = json.load(fh)

    assert "metadata" in payload, "Blueprint JSON missing 'metadata' key"
    assert "blueprints" in payload, "Blueprint JSON missing 'blueprints' key"

    metadata = payload["metadata"]
    for key in ["age", "affine", "n_z_levels", "voxel_size_mm", "region_labels", "masked_labels"]:
        assert key in metadata, f"Metadata missing key: {key}"

    blueprints = payload["blueprints"]
    assert isinstance(blueprints, dict), "blueprints must be a dict"
    assert len(blueprints) > 0, "No blueprints extracted"

    # Check first blueprint
    first_key = sorted(blueprints.keys())[0]
    first_bp = blueprints[first_key]
    for key in ["z_world", "voxel_j", "x", "y", "region", "n_spots"]:
        assert key in first_bp, f"Blueprint missing key: {key}"

    assert len(first_bp["x"]) == first_bp["n_spots"]
    assert len(first_bp["y"]) == first_bp["n_spots"]
    assert len(first_bp["region"]) == first_bp["n_spots"]


def test_blueprint_extraction_script_runnable():
    """Test that extract_devccf_blueprints.py can be invoked and produces valid output."""
    if not EXTRACT_SCRIPT.exists():
        pytest.skip(f"Extract script not found: {EXTRACT_SCRIPT}")
    if not DEVCCF_VOLUME_E15.exists():
        pytest.skip(f"DevCCF volume not found: {DEVCCF_VOLUME_E15}")
    if not DEVCCF_SCHEMA.exists():
        pytest.skip(f"DevCCF schema not found: {DEVCCF_SCHEMA}")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_blueprints.json"
        result = subprocess.run(
            [
                sys.executable,
                str(EXTRACT_SCRIPT),
                "--volume", str(DEVCCF_VOLUME_E15),
                "--region-schema", str(DEVCCF_SCHEMA),
                "--age", "E15.5",
                "--output", str(output_path),
                "--z-step", "20",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"Script failed:\nSTDERR:\n{result.stderr}"

        assert output_path.exists(), "Output JSON not created"
        with open(output_path, "r") as fh:
            payload = json.load(fh)
        assert "metadata" in payload
        assert "blueprints" in payload
        assert len(payload["blueprints"]) > 0


# ---------------------------------------------------------------------------
# Test 2: Blueprint structure -> SliceBlueprint construction
# ---------------------------------------------------------------------------

def test_slice_blueprint_from_json():
    """Test that a SliceBlueprint can be constructed from the blueprint JSON."""
    if not BLUEPRINT_JSON.exists():
        pytest.skip(f"Blueprint JSON not found: {BLUEPRINT_JSON}")

    from FEAST import SliceBlueprint

    with open(BLUEPRINT_JSON, "r") as fh:
        payload = json.load(fh)

    blueprints = payload["blueprints"]
    first_key = sorted(blueprints.keys())[0]
    raw = blueprints[first_key]

    # Construct blueprint from the flat data
    coords = np.column_stack([
        np.asarray(raw["x"], dtype=float),
        np.asarray(raw["y"], dtype=float),
    ])

    bp = SliceBlueprint(
        coordinates=coords,
        domain_map=np.asarray(raw["region"], dtype=str),
        metadata={"z_world": raw["z_world"], "voxel_j": raw["voxel_j"]},
    )

    assert bp.n_spots == raw["n_spots"]
    assert bp.coordinate_dim == 2
    assert "domain" in bp.obs.columns
    assert list(bp.obs["domain"]) == raw["region"]

    # Test round-trip via to_dict / from_dict
    bp2 = SliceBlueprint.from_dict(bp.to_dict())
    assert bp2.n_spots == bp.n_spots
    np.testing.assert_allclose(bp2.coordinates, bp.coordinates)

    # Test active_subset (no mask -> all active)
    active = bp.active_subset()
    assert active.n_spots == bp.n_spots
    assert active.mask is None


# ---------------------------------------------------------------------------
# Test 3: Region label mapping
# ---------------------------------------------------------------------------

def test_region_labels_are_consistent():
    """Test that GSE269617 region labels match the DevCCF blueprint region labels."""
    if not BLUEPRINT_JSON.exists():
        pytest.skip(f"Blueprint JSON not found: {BLUEPRINT_JSON}")

    # Load blueprint region labels
    with open(BLUEPRINT_JSON, "r") as fh:
        payload = json.load(fh)

    blueprint_labels = set(payload["metadata"]["region_labels"])
    masked_labels = set(payload["metadata"]["masked_labels"])

    # Load region schema
    schema_df = pd.read_csv(DEVCCF_SCHEMA, sep="\t")
    schema_labels = set(
        schema_df[schema_df["include_in_final_generation"] == True]["region_label"]
        .tolist()
    )
    schema_masked = set(
        schema_df[schema_df["include_in_final_generation"] == False]["region_label"]
        .tolist()
    )

    # Blueprint included labels should match schema
    assert blueprint_labels == schema_labels - {"Background"}, (
        f"Blueprint labels {sorted(blueprint_labels)} "
        f"do not match schema labels {sorted(schema_labels - {'Background'})}"
    )

    # Load a GSE269617 h5ad and check region labels exist
    gse_files = sorted(GSE269617_H5AD_DIR.glob("*.h5ad"))
    if not gse_files:
        pytest.skip("No GSE269617 h5ad files found")

    adata = ad.read_h5ad(gse_files[0])
    gse_labels = set(adata.obs["region"].astype(str).unique().tolist())

    # All blueprint labels should appear in GSE269617
    for label in blueprint_labels:
        assert label in gse_labels, (
            f"Blueprint label '{label}' not found in GSE269617 region column"
        )

    # "Other" should be in GSE269617 but masked from blueprints
    assert "Other" in gse_labels, "GSE269617 should contain 'Other' label"

    print(f"Region labels consistent: "
          f"blueprint={sorted(blueprint_labels)}, "
          f"GSE={sorted(gse_labels)}, "
          f"masked={sorted(masked_labels)}")


# ---------------------------------------------------------------------------
# Test 4: Reference loading
# ---------------------------------------------------------------------------

def test_gse269617_h5ad_loads_correctly():
    """Test that GSE269617 h5ad files load with expected shape, obs columns, and region labels."""
    gse_files = sorted(GSE269617_H5AD_DIR.glob("*.h5ad"))
    if not gse_files:
        pytest.skip("No GSE269617 h5ad files found")

    expected_regions = {
        "DorsalVZ", "IZ", "CP", "VentralVZ", "GE",
        "SeptalVZ", "Septum", "BT", "Other",
    }

    for path in gse_files[:2]:  # Test first 2 files to keep fast
        adata = ad.read_h5ad(path)

        # Shape
        assert adata.n_obs > 0, f"{path.name} has no observations"
        assert adata.n_vars > 0, f"{path.name} has no variables"

        # Required obs columns
        for col in ["region", "developmental_stage", "x", "y"]:
            assert col in adata.obs.columns, f"{path.name} missing obs column: {col}"

        # Region labels
        regions_found = set(adata.obs["region"].astype(str).unique())
        assert regions_found.issubset(expected_regions), (
            f"{path.name} has unexpected region labels: {regions_found - expected_regions}"
        )

        # Must have spatial coordinates
        assert "spatial" in adata.obsm, f"{path.name} missing obsm['spatial']"
        coords = adata.obsm["spatial"]
        assert coords.shape == (adata.n_obs, 2), (
            f"{path.name} spatial should be (n_obs, 2), got {coords.shape}"
        )

        # All stage info consistent
        stage = adata.obs["developmental_stage"].unique()
        assert len(stage) == 1, f"{path.name} has multiple stages: {stage}"

        print(f"{path.name}: n_obs={adata.n_obs}, n_vars={adata.n_vars}, "
              f"stage={stage[0]}, regions={sorted(regions_found)}")


def test_gse269617_counts_layer_exists():
    """Test that GSE269617 h5ad files have a 'counts' layer."""
    gse_files = sorted(GSE269617_H5AD_DIR.glob("*.h5ad"))
    if not gse_files:
        pytest.skip("No GSE269617 h5ad files found")

    adata = ad.read_h5ad(gse_files[0])
    assert "counts" in adata.layers, (
        f"{gse_files[0].name} is missing 'counts' layer"
    )
    assert adata.layers["counts"].shape == (adata.n_obs, adata.n_vars)


# ---------------------------------------------------------------------------
# Test 5: Coordinate assignment
# ---------------------------------------------------------------------------

def test_assign_generated_coordinates():
    """Test that assign_generated_coordinates correctly sets spatial, spatial_3d, and obs['z']."""
    from FEAST.de_novo.core import assign_generated_coordinates

    n_spots = 50
    rng = np.random.default_rng(42)
    adata = ad.AnnData(
        X=rng.integers(0, 20, size=(n_spots, 5)).astype(np.int32),
        obs=pd.DataFrame(index=[f"spot_{i}" for i in range(n_spots)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(5)]),
    )

    coords_2d = rng.uniform(0, 100, size=(n_spots, 2))
    z_value = 1.5

    assign_generated_coordinates(adata, coords_2d, z_value=z_value)

    # Check spatial (2D)
    assert "spatial" in adata.obsm
    np.testing.assert_allclose(adata.obsm["spatial"], coords_2d, atol=1e-10)

    # Check spatial_3d
    assert "spatial_3d" in adata.obsm
    xyz = adata.obsm["spatial_3d"]
    assert xyz.shape == (n_spots, 3)
    np.testing.assert_allclose(xyz[:, :2], coords_2d, atol=1e-10)
    np.testing.assert_allclose(xyz[:, 2], z_value, atol=1e-10)

    # Check obs['z']
    assert "z" in adata.obs
    np.testing.assert_allclose(
        np.asarray(adata.obs["z"], dtype=float), z_value, atol=1e-10
    )

    # Check uns metadata
    de_novo = adata.uns.get("de_novo", {})
    assert de_novo.get("coordinate_dim") == 3
    assert de_novo.get("coordinate_keys", {}).get("xyz") == "spatial_3d"

    print(f"Coordinate assignment OK: spatial={adata.obsm['spatial'].shape}, "
          f"spatial_3d={adata.obsm['spatial_3d'].shape}, z={adata.obs['z'].iloc[0]}")


def test_assign_generated_coordinates_3d_input():
    """Test coordinate assignment with 3D input coordinates."""
    from FEAST.de_novo.core import assign_generated_coordinates

    n_spots = 30
    rng = np.random.default_rng(99)
    coords_3d = rng.uniform(0, 50, size=(n_spots, 3))

    adata = ad.AnnData(
        X=np.zeros((n_spots, 3), dtype=np.int32),
        obs=pd.DataFrame(index=[f"spot_{i}" for i in range(n_spots)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(3)]),
    )

    assign_generated_coordinates(adata, coords_3d)

    assert "spatial" in adata.obsm
    np.testing.assert_allclose(adata.obsm["spatial"], coords_3d[:, :2], atol=1e-10)

    assert "spatial_3d" in adata.obsm
    np.testing.assert_allclose(adata.obsm["spatial_3d"], coords_3d, atol=1e-10)

    # No z_value provided, so obs['z'] should NOT be set
    assert "z" not in adata.obs


def test_assign_generated_coordinates_rejects_mismatched_rows():
    """Test that mismatched coordinate counts raise ValueError."""
    from FEAST.de_novo.core import assign_generated_coordinates

    adata = ad.AnnData(
        X=np.zeros((10, 2), dtype=np.int32),
    )
    bad_coords = np.zeros((9, 2))
    with pytest.raises(ValueError):
        assign_generated_coordinates(adata, bad_coords)


# ---------------------------------------------------------------------------
# Test 6: Z-regularization math
# ---------------------------------------------------------------------------

def test_gaussian_filter1d_on_synthetic_z_profile():
    """Test that Gaussian 1D smoothing on a synthetic z-profile works as expected."""
    # Create a synthetic z-series: 10 z-levels, 3 genes, 1 region
    nz = 10
    ng = 3
    rng = np.random.default_rng(2026)

    # Ground truth: smooth signal with noise
    z = np.arange(nz, dtype=float)
    true_signal = np.column_stack([
        10.0 + 2.0 * np.sin(z * 0.5),       # gene 0: sinusoidal
        5.0 + 0.5 * z,                        # gene 1: linear upward
        20.0 - 1.0 * z,                       # gene 2: linear downward
    ])

    noisy = true_signal + rng.normal(0, 0.5, size=(nz, ng))
    noisy = np.maximum(noisy, 0.1)

    # Apply Gaussian filter
    sigma = 1.0
    smoothed = gaussian_filter1d(noisy, sigma=sigma, axis=0, mode="nearest")

    # The smoothed result should be closer to true_signal than noisy is
    noisy_mse = np.mean((noisy - true_signal) ** 2)
    smooth_mse = np.mean((smoothed - true_signal) ** 2)

    assert smooth_mse < noisy_mse, (
        f"Gaussian filter should reduce MSE: "
        f"noisy_mse={noisy_mse:.4f}, smooth_mse={smooth_mse:.4f}"
    )

    # Adjacent z-levels should be more correlated after smoothing
    orig_adj_corr = np.mean([
        np.corrcoef(noisy[i], noisy[i + 1])[0, 1] for i in range(nz - 1)
    ])
    smooth_adj_corr = np.mean([
        np.corrcoef(smoothed[i], smoothed[i + 1])[0, 1] for i in range(nz - 1)
    ])

    assert smooth_adj_corr > orig_adj_corr, (
        f"Smoothed adjacent-z correlation ({smooth_adj_corr:.4f}) "
        f"should exceed original ({orig_adj_corr:.4f})"
    )

    print(f"Noisy MSE: {noisy_mse:.4f}, Smooth MSE: {smooth_mse:.4f}")
    print(f"Original adj-cor: {orig_adj_corr:.4f}, Smoothed adj-cor: {smooth_adj_corr:.4f}")


def test_gaussian_filter1d_mode_nearest_boundary():
    """Test that mode='nearest' handles boundaries by not extrapolating."""
    x = np.array([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0])
    smoothed = gaussian_filter1d(x.reshape(-1, 1), sigma=1.0, axis=0, mode="nearest")
    # With mode='nearest', edge values stay close to original
    assert np.all(np.isfinite(smoothed))
    # The central plateau should be preserved
    assert np.allclose(smoothed[2:-2], 2.0, atol=0.3)


def test_scale_rescaling_math():
    """Test that the scale=smoothed/original rescaling logic is correct."""
    means = np.array([10.0, 12.0, 8.0, 9.0, 11.0], dtype=float)
    sigma = 1.0
    smoothed = gaussian_filter1d(means, sigma=sigma, mode="nearest")

    # Each spot's count should be multiplied by smoothed[i]/means[i]
    for i in range(len(means)):
        scale = smoothed[i] / means[i]
        assert np.isfinite(scale), f"Scale at {i} should be finite"
        assert scale > 0, f"Scale at {i} should be positive"

    # At minimum, scales should stay near 1.0
    scales = smoothed / np.maximum(means, 1e-8)
    assert np.all(np.abs(scales - 1.0) < 0.5), "Scales should not deviate drastically"


def test_skip_low_spot_counts_logic():
    """Test that z-levels with fewer than min_spots are skipped."""
    # Build a simple scenario: 3 z-levels, 1 gene, 2 labels
    # Label "A" has > min_spots at z0, z2 but < min_spots at z1
    min_spots = 10
    n_spots_per_z = [20, 5, 15]  # z1 has only 5 spots for label "A"

    z_pass = np.array([n >= min_spots for n in n_spots_per_z], dtype=bool)
    assert z_pass.tolist() == [True, False, True], "z1 should be skipped"

    means = np.array([10.0, np.nan, 12.0])
    valid = means[np.asarray(z_pass)]
    assert list(valid) == [10.0, 12.0]

    smoothed = gaussian_filter1d(valid, sigma=1.0, mode="nearest")
    assert len(smoothed) == 2
    assert np.all(np.isfinite(smoothed))


# ---------------------------------------------------------------------------
# Test 7: End-to-end smoke test
# ---------------------------------------------------------------------------

def test_pipeline_can_discover_and_load_slices():
    """Smoke test: the z_regularize.py script can at least discover and load slices."""
    if not Z_REGULARIZE_SCRIPT.exists():
        pytest.skip(f"z_regularize.py not found: {Z_REGULARIZE_SCRIPT}")

    # We need at least 2 generated h5ad files with z metadata to test
    # Since no generated data exists yet for 07_3d_transfer, create synthetic ones
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()

        # Create synthetic generated slices
        rng = np.random.default_rng(42)
        n_genes = 20
        gene_names = [f"Gene_{i}" for i in range(n_genes)]
        z_values = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        labels = ["DorsalVZ", "IZ", "CP", "GE", "BT"]

        for zi, z in enumerate(z_values):
            n_spots = 200
            counts = rng.integers(0, 30, size=(n_spots, n_genes)).astype(np.int32)
            # Assign labels with different proportions
            region = rng.choice(labels, size=n_spots)

            xy = rng.uniform(0, 50, size=(n_spots, 2))
            xyz = np.column_stack([xy, np.full(n_spots, z)])

            adata = ad.AnnData(
                X=counts,
                obs=pd.DataFrame({"region": region, "z": z}, index=[f"s{i}" for i in range(n_spots)]),
                var=pd.DataFrame(index=gene_names),
            )
            adata.obsm["spatial"] = xy
            adata.obsm["spatial_3d"] = xyz
            adata.layers["counts"] = counts.astype(np.int32)

            adata.write_h5ad(input_dir / f"slice_{zi:03d}.h5ad")

        # Run z_regularize
        result = subprocess.run(
            [
                sys.executable,
                str(Z_REGULARIZE_SCRIPT),
                "--input-dir", str(input_dir),
                "--output-dir", str(output_dir),
                "--label-key", "region",
                "--sigma", "1.0",
                "--min-spots", "10",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, (
            f"z_regularize.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

        # Check outputs
        metrics_csv = output_dir / "z_coherence_metrics.csv"
        summary_csv = output_dir / "z_coherence_summary.csv"
        assert metrics_csv.exists(), f"Missing {metrics_csv}"
        assert summary_csv.exists(), f"Missing {summary_csv}"

        # Check metrics CSV content
        metrics_df = pd.read_csv(metrics_csv)
        assert len(metrics_df) > 0, "Metrics CSV is empty"
        assert set(metrics_df.columns) >= {"metric", "class", "gene", "value"}
        assert "adjacent_z_pearson" in metrics_df["metric"].values, (
            "Missing adjacent_z_pearson rows"
        )
        assert "z_autocorr_lag1" in metrics_df["metric"].values, (
            "Missing z_autocorr_lag1 rows"
        )

        # Check summary CSV
        summary_df = pd.read_csv(summary_csv)
        assert len(summary_df) > 0, "Summary CSV is empty"

        # Check that regularized h5ads were written
        h5ad_files = sorted(output_dir.glob("regularized_*.h5ad"))
        assert len(h5ad_files) == len(z_values), (
            f"Expected {len(z_values)} regularized files, got {len(h5ad_files)}"
        )

        # Check one regularized h5ad has expected structure
        reg = ad.read_h5ad(h5ad_files[0])
        assert reg.n_obs == n_spots
        assert reg.n_vars == n_genes
        assert "z" in reg.obs
        assert "region" in reg.obs

        print(f"End-to-end smoke test passed: {len(h5ad_files)} regularized slices, "
              f"{len(metrics_df)} metric rows")


def test_pipeline_full_extract_and_regularize():
    """Integration test: extract blueprints and run z-regularization end-to-end.

    Requires the full DevCCF volume + extract_devccf_blueprints.py. Skips if
    the volume or extract script is unavailable.
    """
    if not EXTRACT_SCRIPT.exists():
        pytest.skip(f"Extract script not found: {EXTRACT_SCRIPT}")
    if not DEVCCF_VOLUME_E15.exists():
        pytest.skip(f"DevCCF volume not found: {DEVCCF_VOLUME_E15}")
    if not DEVCCF_SCHEMA.exists():
        pytest.skip(f"DevCCF schema not found: {DEVCCF_SCHEMA}")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Step 1: Extract blueprints
        blueprint_path = tmp / "blueprints.json"
        result = subprocess.run(
            [
                sys.executable,
                str(EXTRACT_SCRIPT),
                "--volume", str(DEVCCF_VOLUME_E15),
                "--region-schema", str(DEVCCF_SCHEMA),
                "--age", "E15.5",
                "--output", str(blueprint_path),
                "--z-step", "50",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Blueprint extraction failed:\n{result.stderr}"
        )

        with open(blueprint_path, "r") as fh:
            payload = json.load(fh)
        assert len(payload["blueprints"]) > 0

        # Step 2: Create synthetic generated slices from the blueprint metadata
        generated_dir = tmp / "generated"
        generated_dir.mkdir()

        blueprints = payload["blueprints"]
        z_keys = sorted(blueprints.keys())
        n_z = min(len(z_keys), 5)  # Use up to 5 z-levels for speed

        rng = np.random.default_rng(2026)
        n_genes = 30
        gene_names = [f"Gene_{i}" for i in range(n_genes)]
        region_labels = payload["metadata"]["region_labels"]

        for idx, zk in enumerate(z_keys[:n_z]):
            raw = blueprints[zk]
            n_spots = raw["n_spots"]

            counts = rng.integers(0, 20, size=(n_spots, n_genes)).astype(np.int32)
            regions = rng.choice(region_labels, size=n_spots)

            xy = np.column_stack([
                np.asarray(raw["x"], dtype=float),
                np.asarray(raw["y"], dtype=float),
            ])
            xyz = np.column_stack([xy, np.full(n_spots, raw["z_world"])])

            adata = ad.AnnData(
                X=counts,
                obs=pd.DataFrame({
                    "region": regions,
                    "z": raw["z_world"],
                }, index=[f"s{i}" for i in range(n_spots)]),
                var=pd.DataFrame(index=gene_names),
            )
            adata.obsm["spatial"] = xy
            adata.obsm["spatial_3d"] = xyz
            adata.layers["counts"] = counts.astype(np.int32)

            adata.write_h5ad(generated_dir / f"slice_{idx:03d}.h5ad")

        # Step 3: Run z-regularization
        regularized_dir = tmp / "regularized"
        result = subprocess.run(
            [
                sys.executable,
                str(Z_REGULARIZE_SCRIPT),
                "--input-dir", str(generated_dir),
                "--output-dir", str(regularized_dir),
                "--label-key", "region",
                "--sigma", "1.0",
                "--min-spots", "5",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"Regularization failed:\n{result.stdout}\n{result.stderr}"
        )

        # Verify outputs
        assert (regularized_dir / "z_coherence_metrics.csv").exists()
        assert (regularized_dir / "z_coherence_summary.csv").exists()

        reg_files = sorted(regularized_dir.glob("regularized_*.h5ad"))
        assert len(reg_files) == n_z, f"Expected {n_z} files, got {len(reg_files)}"

        metrics = pd.read_csv(regularized_dir / "z_coherence_metrics.csv")
        print(f"Integration test passed: {len(reg_files)} slices, "
              f"{len(metrics)} metric rows")
