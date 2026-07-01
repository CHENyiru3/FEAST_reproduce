"""Tests for diagonal affine deformation fitting."""

import numpy as np
import pytest
from pathlib import Path
import sys
from numpy.testing import assert_allclose

FEAST_ROOT = Path("/maiziezhou_lab2/yiru/FEAST")
sys.path.insert(0, str(FEAST_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_assessment.affine_model import (
    fit_diagonal_affine,
    residual_analysis,
)


class TestDiagonalAffine:
    def test_exact_recovery(self, synthetic_theta_cloud):
        """Create cloud B from A via known D=diag([1.2, 0.8, 1.0]), b=[0.3, -0.1, 0.0].
        fit_diagonal_affine should recover D and b exactly.
        """
        true_D = np.array([1.2, 0.8, 1.0])
        true_b = np.array([0.3, -0.1, 0.0])

        theta_A = synthetic_theta_cloud["A"]["theta"]
        theta_X = theta_A * true_D[None, :] + true_b[None, :]

        clouds = {
            "A": synthetic_theta_cloud["A"],
            "X": {
                "slice_id": "X",
                "gene_ids": synthetic_theta_cloud["A"]["gene_ids"],
                "theta": theta_X,
                "stats_df": None,
            },
        }

        result = fit_diagonal_affine(clouds, ref_slice="A")
        D_recovered = result["D_matrices"]
        b_recovered = result["b_vectors"]

        # A's own D should be [1,1,1], b should be [0,0,0]
        assert_allclose(D_recovered[1], true_D, atol=1e-8)
        assert_allclose(b_recovered[1], true_b, atol=1e-8)

    def test_r2_perfect(self, synthetic_theta_cloud):
        """With exact linear transform, r^2 = 1.0 for each dimension."""
        true_D = np.array([1.5, 0.9, 1.1])
        true_b = np.array([0.2, 0.0, -0.1])

        theta_A = synthetic_theta_cloud["A"]["theta"]
        theta_X = theta_A * true_D[None, :] + true_b[None, :]

        clouds = {
            "A": synthetic_theta_cloud["A"],
            "X": {
                "slice_id": "X",
                "gene_ids": synthetic_theta_cloud["A"]["gene_ids"],
                "theta": theta_X,
                "stats_df": None,
            },
        }

        result = fit_diagonal_affine(clouds, ref_slice="A")
        r2 = result["r2_scores"]
        assert_allclose(r2[1], [1.0, 1.0, 1.0], atol=1e-8)

    def test_r2_degraded_with_noise(self, synthetic_theta_cloud):
        """Add Gaussian noise; r^2 should drop but stay positive."""
        rng = np.random.default_rng(999)
        true_D = np.array([1.0, 1.0, 1.0])
        true_b = np.array([0.3, 0.0, 0.0])

        theta_A = synthetic_theta_cloud["A"]["theta"]
        noise = rng.normal(0, 0.5, size=theta_A.shape)
        theta_X = theta_A * true_D[None, :] + true_b[None, :] + noise

        clouds = {
            "A": synthetic_theta_cloud["A"],
            "X": {
                "slice_id": "X",
                "gene_ids": synthetic_theta_cloud["A"]["gene_ids"],
                "theta": theta_X,
                "stats_df": None,
            },
        }

        result = fit_diagonal_affine(clouds, ref_slice="A")
        r2 = result["r2_scores"]
        # r^2 should be positive but less than 1
        assert np.all(r2[1] > 0.3)
        assert np.all(r2[1] < 1.0)

    def test_output_format(self, synthetic_theta_cloud):
        """Result has all expected keys."""
        result = fit_diagonal_affine(synthetic_theta_cloud, ref_slice="A")
        expected_keys = [
            "slice_order", "ref_slice", "dimension_names",
            "D_matrices", "b_vectors", "r2_scores", "residual_variance", "params_df",
        ]
        for k in expected_keys:
            assert k in result

        S = len(synthetic_theta_cloud)
        assert result["D_matrices"].shape == (S, 3)
        assert result["b_vectors"].shape == (S, 3)
        assert result["r2_scores"].shape == (S, 3)
        assert result["residual_variance"].shape == (S, 3)

    def test_ref_slice_identity(self, synthetic_theta_cloud):
        """Reference slice should have D=[1,1,1], b=[0,0,0], r2=1."""
        result = fit_diagonal_affine(synthetic_theta_cloud, ref_slice="A")
        ref_idx = result["slice_order"].index("A")
        assert_allclose(result["D_matrices"][ref_idx], [1.0, 1.0, 1.0], atol=1e-8)
        assert_allclose(result["b_vectors"][ref_idx], [0.0, 0.0, 0.0], atol=1e-8)
        assert_allclose(result["r2_scores"][ref_idx], [1.0, 1.0, 1.0], atol=1e-8)

    def test_residual_variance_positive(self, synthetic_theta_cloud):
        result = fit_diagonal_affine(synthetic_theta_cloud, ref_slice="A")
        assert (result["residual_variance"] >= 0).all()


class TestResidualAnalysis:
    def test_residual_cloud_same_shape(self, synthetic_theta_cloud):
        """Residual clouds have same shape as originals."""
        affine = fit_diagonal_affine(synthetic_theta_cloud, ref_slice="A")
        residuals = residual_analysis(synthetic_theta_cloud, affine)
        for sid, resid in residuals["residual_clouds"].items():
            assert resid.shape == synthetic_theta_cloud[sid]["theta"].shape

    def test_ref_residual_near_zero(self, synthetic_theta_cloud):
        """Reference slice's residual cloud should be near zero."""
        affine = fit_diagonal_affine(synthetic_theta_cloud, ref_slice="A")
        residuals = residual_analysis(synthetic_theta_cloud, affine)
        assert_allclose(residuals["residual_clouds"]["A"], 0.0, atol=1e-8)

    def test_residual_covariance_smaller(self, synthetic_theta_cloud):
        """Residual covariance trace < original covariance trace (affine removes structure)."""
        import numpy as np
        affine = fit_diagonal_affine(synthetic_theta_cloud, ref_slice="A")
        residuals = residual_analysis(synthetic_theta_cloud, affine)
        for sid in synthetic_theta_cloud:
            if sid == "A":
                continue
            orig_trace = np.trace(np.cov(synthetic_theta_cloud[sid]["theta"].T))
            resid_trace = np.trace(np.cov(residuals["residual_clouds"][sid].T))
            assert resid_trace < orig_trace

    def test_per_gene_residual_norms_shape(self, synthetic_theta_cloud):
        """per_gene_residual_norms has one entry per gene per slice."""
        affine = fit_diagonal_affine(synthetic_theta_cloud, ref_slice="A")
        residuals = residual_analysis(synthetic_theta_cloud, affine)
        n_genes = synthetic_theta_cloud["A"]["theta"].shape[0]
        for sid in synthetic_theta_cloud:
            assert len(residuals["per_gene_residual_norms"][sid]) == n_genes
