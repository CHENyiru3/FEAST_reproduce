"""Tests for theta coordinate transforms and cloud extraction."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

FEAST_ROOT = Path("/maiziezhou_lab2/yiru/FEAST")
sys.path.insert(0, str(FEAST_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_assessment.cloud_extraction import (
    stats_to_theta,
    theta_to_stats,
    extract_cloud,
    extract_clouds,
    filter_common_genes,
)


class TestThetaTransform:
    def test_roundtrip(self):
        """stats -> theta -> stats recovers original within tolerance."""
        df = pd.DataFrame({
            "mean": [1.0, 5.0, 0.5],
            "variance": [2.0, 15.0, 0.8],
            "zero_prop": [0.1, 0.5, 0.8],
        })
        theta = stats_to_theta(df)
        recovered = theta_to_stats(theta)
        np.testing.assert_allclose(recovered["mean"].values, df["mean"].values, rtol=1e-10)
        np.testing.assert_allclose(recovered["variance"].values, df["variance"].values, rtol=1e-10)
        np.testing.assert_allclose(recovered["zero_prop"].values, df["zero_prop"].values, rtol=1e-10)

    def test_omega_formula(self):
        """Fano factor: omega = variance / mean."""
        df = pd.DataFrame({
            "mean": [2.0, 4.0],
            "variance": [6.0, 8.0],
            "zero_prop": [0.3, 0.3],
        })
        theta = stats_to_theta(df)
        omega = np.exp(theta[:, 1])
        np.testing.assert_allclose(omega, [3.0, 2.0], rtol=1e-10)

    def test_logit_boundary(self):
        """zero_prop near 0 or 1 is clipped before logit."""
        df = pd.DataFrame({
            "mean": [1.0, 1.0, 1.0],
            "variance": [1.0, 1.0, 1.0],
            "zero_prop": [0.0, 1.0, 0.99],
        })
        theta = stats_to_theta(df)
        assert np.all(np.isfinite(theta[:, 2]))

    def test_nan_handling(self):
        """Gene with all-zero counts produces finite theta after clipping."""
        df = pd.DataFrame({
            "mean": [0.0],
            "variance": [0.0],
            "zero_prop": [1.0],
        })
        theta = stats_to_theta(df)
        assert np.all(np.isfinite(theta))

    def test_inverse_reconstruction(self):
        """theta_to_stats reconstructs interpretable parameter ranges."""
        df = pd.DataFrame({
            "mean": [0.01, 0.1, 10.0],
            "variance": [0.02, 0.3, 50.0],
            "zero_prop": [0.05, 0.5, 0.9],
        })
        theta = stats_to_theta(df)
        recovered = theta_to_stats(theta)
        for col in ["mean", "variance", "zero_prop"]:
            np.testing.assert_allclose(
                recovered[col].values, df[col].values, rtol=1e-10
            )


class TestExtractCloud:
    def test_output_shape(self, synthetic_adata_dict):
        """extract_cloud returns theta with shape (G, 3) and stats_df with (G, 3)."""
        adata = synthetic_adata_dict["S1"]
        cloud = extract_cloud(adata)
        assert cloud["theta"].shape == (adata.n_vars, 3)
        assert cloud["stats_df"].shape == (adata.n_vars, 3)

    def test_gene_ids_preserved(self, synthetic_adata_dict):
        """Gene names match var_names."""
        adata = synthetic_adata_dict["S1"]
        cloud = extract_cloud(adata)
        assert list(cloud["gene_ids"]) == list(adata.var_names)

    def test_extract_clouds_multiple(self, synthetic_adata_dict):
        """Three adata objects produce three entries."""
        clouds = extract_clouds(synthetic_adata_dict)
        assert len(clouds) == 3
        for sid in ["S1", "S2", "S3"]:
            assert sid in clouds
            assert clouds[sid]["theta"].shape[1] == 3

    def test_filter_common_genes(self, synthetic_adata_dict):
        """After filtering, all clouds have identical gene_ids in same order."""
        clouds = extract_clouds(synthetic_adata_dict)
        filtered = filter_common_genes(clouds)
        gene_sets = [tuple(c["gene_ids"]) for c in filtered.values()]
        assert len(set(gene_sets)) == 1  # all identical

    def test_cloud_slice_id(self, synthetic_adata_dict):
        """Each cloud has slice_id matching the dict key."""
        clouds = extract_clouds(synthetic_adata_dict)
        for sid, cloud in clouds.items():
            assert cloud["slice_id"] == sid
