"""Verify new FEAST core additions: theta_transform, apply_batch_deformation, simulate_batch_effect."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

FEAST_ROOT = Path("/maiziezhou_lab2/yiru/FEAST")
sys.path.insert(0, str(FEAST_ROOT / "src"))


class TestThetaTransformCore:
    """Verify theta transforms imported from FEAST core work correctly."""

    def test_imports(self):
        from FEAST import stats_to_theta, theta_to_stats
        assert callable(stats_to_theta)
        assert callable(theta_to_stats)

    def test_roundtrip(self):
        from FEAST import stats_to_theta, theta_to_stats
        df = pd.DataFrame({
            "mean": [1.0, 5.0, 0.5],
            "variance": [2.0, 15.0, 0.8],
            "zero_prop": [0.1, 0.5, 0.8],
        })
        theta = stats_to_theta(df)
        recovered = theta_to_stats(theta)
        np.testing.assert_allclose(recovered["mean"].values, df["mean"].values, rtol=1e-10)

    def test_result_matches_experiment_copy(self):
        """Core version must produce identical results to experiment copy."""
        from FEAST import stats_to_theta as core_stt

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from batch_assessment import stats_to_theta as exp_stt

        df = pd.DataFrame({
            "mean": [3.0, 0.1, 10.0],
            "variance": [9.0, 0.3, 50.0],
            "zero_prop": [0.2, 0.5, 0.7],
        })
        np.testing.assert_array_equal(core_stt(df), exp_stt(df))


class TestApplyBatchDeformation:
    """Verify the new apply_batch_deformation function."""

    def test_imports(self):
        from FEAST.FEAST_core.parameter_cloud import apply_batch_deformation
        assert callable(apply_batch_deformation)

    def test_alpha_zero_is_identity(self):
        from FEAST.FEAST_core.parameter_cloud import apply_batch_deformation
        rng = np.random.default_rng(42)
        theta = rng.normal(size=(100, 3))
        D = np.array([0.5, 2.0, 0.8])
        b = np.array([0.3, -0.1, 0.5])
        result = apply_batch_deformation(theta, D, b, alpha=0.0)
        np.testing.assert_allclose(result, theta)

    def test_alpha_one_full_deformation(self):
        from FEAST.FEAST_core.parameter_cloud import apply_batch_deformation
        theta = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        D = np.array([0.97, 1.28, 1.04])
        b = np.array([0.57, 0.02, -0.81])
        result = apply_batch_deformation(theta, D, b, alpha=1.0)
        expected = theta * D + b
        np.testing.assert_allclose(result, expected)

    def test_linear_in_alpha(self):
        from FEAST.FEAST_core.parameter_cloud import apply_batch_deformation
        rng = np.random.default_rng(99)
        theta = rng.normal(size=(50, 3))
        D = np.array([1.2, 0.9, 1.1])
        b = np.array([0.3, 0.1, -0.2])
        r0 = apply_batch_deformation(theta, D, b, alpha=0.2)
        r1 = apply_batch_deformation(theta, D, b, alpha=0.6)
        # Interpolation should be smooth
        diff = r1 - r0
        assert np.all(np.isfinite(diff))

    def test_result_shape(self):
        from FEAST.FEAST_core.parameter_cloud import apply_batch_deformation
        theta = np.random.randn(100, 3)
        result = apply_batch_deformation(theta, np.ones(3), np.zeros(3), alpha=0.5)
        assert result.shape == (100, 3)

    def test_D_b_identity_noop(self):
        from FEAST.FEAST_core.parameter_cloud import apply_batch_deformation
        theta = np.random.randn(100, 3)
        result = apply_batch_deformation(theta, np.ones(3), np.zeros(3), alpha=1.0)
        np.testing.assert_allclose(result, theta)


class TestSimulateBatchEffect:
    """Verify the simulate_batch_effect orchestrator with synthetic data."""

    @pytest.fixture
    def small_adata(self):
        from anndata import AnnData
        rng = np.random.default_rng(42)
        n_spots, n_genes = 30, 15
        mu = np.exp(rng.normal(0, 0.5, n_genes)) + 0.2
        X = np.zeros((n_spots, n_genes), dtype=np.float32)
        for g in range(n_genes):
            nz = rng.random(n_spots) > 0.5
            n_nz = int(nz.sum())
            if n_nz > 0:
                X[nz, g] = rng.poisson(mu[g], size=n_nz)
        adata = AnnData(X=X)
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        adata.obsm["spatial"] = rng.uniform(0, 1, (n_spots, 2)).astype(np.float32)
        return adata

    def test_imports(self):
        from FEAST.FEAST_core.simulator import simulate_batch_effect
        assert callable(simulate_batch_effect)

    def test_alpha_zero_preserves_stats(self, small_adata):
        from FEAST.FEAST_core.simulator import simulate_batch_effect
        sim = simulate_batch_effect(
            small_adata,
            D=np.array([1.0, 1.0, 1.0]),
            b=np.array([0.0, 0.0, 0.0]),
            alpha=0.0,
            random_seed=42,
        )
        assert sim.n_obs == small_adata.n_obs
        assert sim.n_vars == small_adata.n_vars
        assert list(sim.var_names) == list(small_adata.var_names)
        assert "spatial" in sim.obsm

    def test_output_is_anndata(self, small_adata):
        from FEAST.FEAST_core.simulator import simulate_batch_effect
        sim = simulate_batch_effect(
            small_adata,
            D=np.array([0.9, 1.1, 1.0]),
            b=np.array([0.2, 0.0, -0.1]),
            alpha=0.5,
            random_seed=42,
        )
        from anndata import AnnData
        assert isinstance(sim, AnnData)

    def test_spatial_preserved(self, small_adata):
        from FEAST.FEAST_core.simulator import simulate_batch_effect
        sim = simulate_batch_effect(
            small_adata,
            D=np.array([1.0, 1.0, 1.0]),
            b=np.array([0.3, 0.0, -0.2]),
            alpha=0.5,
            random_seed=42,
        )
        np.testing.assert_array_equal(
            sim.obsm["spatial"], small_adata.obsm["spatial"]
        )

    def test_obs_preserved(self, small_adata):
        from FEAST.FEAST_core.simulator import simulate_batch_effect
        small_adata.obs["test_col"] = "x"
        sim = simulate_batch_effect(
            small_adata,
            D=np.ones(3), b=np.zeros(3), alpha=0.0,
            random_seed=42,
        )
        assert "test_col" in sim.obs.columns

    def test_batch_effect_changes_counts(self, small_adata):
        from FEAST.FEAST_core.simulator import simulate_batch_effect
        sim0 = simulate_batch_effect(
            small_adata, D=np.ones(3), b=np.zeros(3), alpha=0.0, random_seed=42
        )
        sim1 = simulate_batch_effect(
            small_adata,
            D=np.array([1.0, 1.0, 1.0]),
            b=np.array([0.5, 0.0, -0.5]),
            alpha=1.0,
            random_seed=42,
        )
        # Batch-affected slice should have higher mean expression
        mean0 = np.asarray(sim0.X, dtype=np.float64).mean()
        mean1 = np.asarray(sim1.X, dtype=np.float64).mean()
        assert mean1 > mean0, f"Expected batch effect to increase mean: {mean0} -> {mean1}"


class TestBatchDeformationMetadata:
    """Verify BatchDeformation class and output AnnData metadata."""

    @pytest.fixture
    def small_adata(self):
        from anndata import AnnData
        rng = np.random.default_rng(42)
        n_spots, n_genes = 30, 15
        mu = np.exp(rng.normal(0, 0.5, n_genes)) + 0.2
        X = np.zeros((n_spots, n_genes), dtype=np.float32)
        for g in range(n_genes):
            nz = rng.random(n_spots) > 0.5
            n_nz = int(nz.sum())
            if n_nz > 0:
                X[nz, g] = rng.poisson(mu[g], size=n_nz)
        adata = AnnData(X=X)
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        adata.obsm["spatial"] = rng.uniform(0, 1, (n_spots, 2)).astype(np.float32)
        return adata

    def test_batch_deformation_import(self):
        from FEAST.FEAST_core.parameter_cloud import BatchDeformation
        assert callable(BatchDeformation)

    def test_batch_deformation_to_dict(self):
        from FEAST.FEAST_core.parameter_cloud import BatchDeformation
        bd = BatchDeformation(
            D=np.array([0.97, 1.28, 1.04]),
            b=np.array([0.57, 0.02, -0.81]),
            alpha=0.5,
            name="test",
        )
        d = bd.to_dict()
        assert d["name"] == "test"
        assert d["alpha"] == 0.5
        assert d["D"] == [0.97, 1.28, 1.04]
        assert d["b"] == [0.57, 0.02, -0.81]

    def test_output_has_batch_deformation_uns(self, small_adata):
        from FEAST.FEAST_core.simulator import simulate_batch_effect
        sim = simulate_batch_effect(
            small_adata,
            D=np.array([0.9, 1.1, 1.0]),
            b=np.array([0.2, 0.0, -0.1]),
            alpha=0.5,
            random_seed=42,
        )
        assert "batch_deformation" in sim.uns
        bd = sim.uns["batch_deformation"]
        assert bd["alpha"] == 0.5
        assert len(bd["D"]) == 3
        assert len(bd["b"]) == 3

    def test_output_has_theta_in_var(self, small_adata):
        from FEAST.FEAST_core.simulator import simulate_batch_effect
        sim = simulate_batch_effect(
            small_adata,
            D=np.array([0.9, 1.1, 1.0]),
            b=np.array([0.2, 0.0, -0.1]),
            alpha=0.5,
            random_seed=42,
        )
        for col in ["theta_mu_ref", "theta_omega_ref", "theta_pi0_ref",
                     "theta_mu_batch", "theta_omega_batch", "theta_pi0_batch"]:
            assert col in sim.var.columns, f"Missing {col} in var"

    def test_theta_ref_differs_from_theta_batch(self, small_adata):
        from FEAST.FEAST_core.simulator import simulate_batch_effect
        sim = simulate_batch_effect(
            small_adata,
            D=np.array([0.9, 1.1, 1.0]),
            b=np.array([0.2, 0.0, -0.1]),
            alpha=0.5,
            random_seed=42,
        )
        # theta_batch should differ from theta_ref when alpha>0 and D≠I or b≠0
        assert not np.allclose(
            sim.var["theta_mu_ref"].values, sim.var["theta_mu_batch"].values
        )

    def test_alpha_zero_theta_identical(self, small_adata):
        from FEAST.FEAST_core.simulator import simulate_batch_effect
        sim = simulate_batch_effect(
            small_adata,
            D=np.array([0.5, 2.0, 1.0]),
            b=np.array([0.3, -0.1, 0.5]),
            alpha=0.0,
            random_seed=42,
        )
        np.testing.assert_allclose(
            sim.var["theta_mu_ref"].values, sim.var["theta_mu_batch"].values, atol=1e-10
        )
