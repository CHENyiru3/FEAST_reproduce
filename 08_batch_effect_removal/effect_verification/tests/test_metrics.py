"""Tests for centroid, covariance, distance, and QC metrics."""

import numpy as np
import pytest
from pathlib import Path
import sys
from numpy.testing import assert_allclose

FEAST_ROOT = Path("/maiziezhou_lab2/yiru/FEAST")
sys.path.insert(0, str(FEAST_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_assessment.metrics import (
    compute_centroids,
    compute_covariances,
    pairwise_distances,
    slice_qc_metrics,
)


class TestCentroids:
    def test_centroid_shape(self, synthetic_theta_cloud):
        result = compute_centroids(synthetic_theta_cloud)
        centroids = result["centroids"]
        assert centroids.shape == (3, 3)

    def test_known_shift(self, synthetic_theta_cloud):
        """Cloud B has centroid ~ [0.5, 0, 0], A has ~ [0, 0, 0]."""
        result = compute_centroids(synthetic_theta_cloud, ref_slice="A")
        centroids = result["centroids"]
        assert centroids[0, 0] == pytest.approx(0.0, abs=0.15)
        assert centroids[1, 0] == pytest.approx(0.5, abs=0.15)

    def test_identity_shift(self, synthetic_theta_cloud):
        """Same cloud compared to itself yields near-zero shift."""
        result = compute_centroids(synthetic_theta_cloud, ref_slice="A")
        shifts = result["shifts"]
        assert_allclose(shifts[0], [0, 0, 0], atol=1e-10)

    def test_shift_relative_to_ref(self, synthetic_theta_cloud):
        """With ref='A', shift_A = 0, shift_B ≈ [0.5, 0, 0]."""
        result = compute_centroids(synthetic_theta_cloud, ref_slice="A")
        shifts = result["shifts"]
        assert shifts[1, 0] == pytest.approx(0.5, abs=0.15)

    def test_ref_not_in_clouds_uses_first(self, synthetic_theta_cloud):
        result = compute_centroids(synthetic_theta_cloud, ref_slice="nonexistent")
        assert result["ref_slice"] == "A"
        assert_allclose(result["shifts"][0], [0, 0, 0], atol=1e-10)


class TestCovariances:
    def test_identity_covariance(self, synthetic_theta_cloud):
        """Cloud A covariance is approximately identity."""
        result = compute_covariances(synthetic_theta_cloud)
        cov_A = result["covariances"][0]
        for i in range(3):
            assert cov_A[i, i] == pytest.approx(1.0, abs=0.3)
            for j in range(3):
                if i != j:
                    assert cov_A[i, j] == pytest.approx(0.0, abs=0.3)

    def test_elongated_covariance(self, synthetic_theta_cloud):
        """Cloud C has inflated variance along dim 0."""
        result = compute_covariances(synthetic_theta_cloud)
        cov_C = result["covariances"][2]
        assert cov_C[0, 0] > 1.5  # inflated along dim 0

    def test_log_det_ordering(self, synthetic_theta_cloud):
        """Cloud C (inflated var) has larger log_det than Cloud A (identity)."""
        result = compute_covariances(synthetic_theta_cloud)
        assert result["log_det"][2] > result["log_det"][0]

    def test_frobenius_norm_positive(self, synthetic_theta_cloud):
        result = compute_covariances(synthetic_theta_cloud)
        for fn in result["frobenius_norm"]:
            assert fn > 0


class TestPairwiseDistances:
    def test_self_distance_zero(self, synthetic_theta_cloud):
        """All metrics return zero (or near-zero) for self-comparison."""
        metrics = ["centroid_euclidean", "covariance_frobenius", "mmd_rbf", "sliced_wasserstein"]
        dists = pairwise_distances(synthetic_theta_cloud, metrics)
        for metric in metrics:
            d = dists[metric]
            assert np.all(np.diag(d) < 1e-8), f"{metric} self-distance not near-zero: {np.diag(d)}"

    def test_symmetry(self, synthetic_theta_cloud):
        """All distance matrices are symmetric."""
        metrics = ["centroid_euclidean", "covariance_frobenius", "mmd_rbf", "sliced_wasserstein"]
        dists = pairwise_distances(synthetic_theta_cloud, metrics)
        for metric in metrics:
            d = dists[metric]
            assert_allclose(d, d.T, atol=1e-10)

    def test_centroid_distance_ordering(self, synthetic_theta_cloud):
        """D(A,B) > 0 but B's centroid is shifted, C's is same as A's."""
        dists = pairwise_distances(synthetic_theta_cloud, ["centroid_euclidean"])
        d = dists["centroid_euclidean"]
        assert d[0, 1] > 0.2
        # A and C have same centroid, so distance should be small
        assert d[0, 2] < 0.2

    def test_mmd_ordering(self, synthetic_theta_cloud):
        """Both B and C differ distributionally from A."""
        dists = pairwise_distances(synthetic_theta_cloud, ["mmd_rbf"])
        d = dists["mmd_rbf"]
        assert d[0, 1] > 0
        assert d[0, 2] > 0

    def test_sliced_wasserstein_nonnegative(self, synthetic_theta_cloud):
        dists = pairwise_distances(synthetic_theta_cloud, ["sliced_wasserstein"])
        d = dists["sliced_wasserstein"]
        assert np.all(d >= 0)

    def test_covariance_riemannian_invariance(self, synthetic_theta_cloud):
        """Applying the same rotation to both clouds preserves Riemannian distance."""
        dists_before = pairwise_distances(synthetic_theta_cloud, ["covariance_riemannian"])
        d_before = dists_before["covariance_riemannian"]

        clouds = {}
        rng = np.random.default_rng(123)
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        for k in synthetic_theta_cloud:
            clouds[k] = dict(synthetic_theta_cloud[k])
            clouds[k]["theta"] = synthetic_theta_cloud[k]["theta"].copy() @ R.T

        dists_after = pairwise_distances(clouds, ["covariance_riemannian"])
        d_after = dists_after["covariance_riemannian"]

        assert_allclose(d_before, d_after, atol=1e-8)

    def test_distance_matrix_shape(self, synthetic_theta_cloud):
        dists = pairwise_distances(synthetic_theta_cloud, ["centroid_euclidean", "mmd_rbf"])
        n = len(synthetic_theta_cloud)
        assert dists["centroid_euclidean"].shape == (n, n)
        assert dists["mmd_rbf"].shape == (n, n)


class TestQC:
    def test_qc_columns(self, synthetic_adata_dict):
        qc = slice_qc_metrics(synthetic_adata_dict)
        expected_cols = [
            "total_counts_mean", "total_counts_std", "detected_genes",
            "zero_fraction", "mean_of_means", "mean_of_variances",
            "mean_of_zero_prop", "n_spots",
        ]
        for col in expected_cols:
            assert col in qc.columns

    def test_total_counts_positive(self, synthetic_adata_dict):
        qc = slice_qc_metrics(synthetic_adata_dict)
        assert (qc["total_counts_mean"] > 0).all()

    def test_zero_fraction_range(self, synthetic_adata_dict):
        qc = slice_qc_metrics(synthetic_adata_dict)
        assert (qc["zero_fraction"] >= 0).all()
        assert (qc["zero_fraction"] <= 1).all()

    def test_n_spots_correct(self, synthetic_adata_dict):
        qc = slice_qc_metrics(synthetic_adata_dict)
        for sid in synthetic_adata_dict:
            assert qc.loc[sid, "n_spots"] == synthetic_adata_dict[sid].n_obs
