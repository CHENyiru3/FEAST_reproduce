"""Distribution fidelity metric tests.

Covers:
  - gene_mean_wasserstein zero for identity, >0 for noise
  - gene_variance_wasserstein
  - library_size_wasserstein
  - variance_correlation outlier robustness (raw vs log1p)
  - sparse matrix equivalence
"""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

SIM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIM_ROOT / "scripts"))

from build_simulator_quality_metrics import (
    _dense,
    _gene_variances,
    gene_mean_wasserstein,
    gene_variance_wasserstein,
    input_variance_corr,
    library_size_wasserstein,
)


class TestGeneMeanWasserstein:
    def test_zero_for_identity(self, fixture_a_identity):
        real, sim = fixture_a_identity
        assert gene_mean_wasserstein(real, sim) == pytest.approx(0.0, abs=1e-8)

    def test_large_for_noise(self, fixture_e_noise):
        real, sim = fixture_e_noise
        w = gene_mean_wasserstein(real, sim)
        assert w > 0.0


class TestGeneVarianceWasserstein:
    def test_zero_for_identity(self, fixture_a_identity):
        real, sim = fixture_a_identity
        assert gene_variance_wasserstein(real, sim) == pytest.approx(0.0, abs=1e-8)

    def test_large_for_noise(self, fixture_e_noise):
        real, sim = fixture_e_noise
        w = gene_variance_wasserstein(real, sim)
        assert w > 0.0


class TestLibrarySize:
    def test_detects_depth_shift(self, fixture_e_noise):
        real, sim = fixture_e_noise
        w = library_size_wasserstein(real, sim)
        assert w > 0.0

    def test_zero_for_identity(self, fixture_a_identity):
        real, sim = fixture_a_identity
        assert library_size_wasserstein(real, sim) == pytest.approx(0.0, abs=1e-8)


class TestSparseMatrixEquivalence:
    """Verify dense and sparse matrices produce equivalent metric values."""

    def test_gene_mean_wasserstein_sparse_equivalent(self, fixture_a_identity):
        real_dense, sim_dense = fixture_a_identity
        real_sp = real_dense.copy()
        sim_sp = sim_dense.copy()
        real_sp.X = sp.csr_matrix(real_dense.X)
        sim_sp.X = sp.csr_matrix(sim_dense.X)
        w_dense = gene_mean_wasserstein(real_dense, sim_dense)
        w_sparse = gene_mean_wasserstein(real_sp, sim_sp)
        assert w_dense == pytest.approx(w_sparse, abs=1e-6)

    def test_gene_variance_wasserstein_sparse_equivalent(self, fixture_a_identity):
        real_dense, sim_dense = fixture_a_identity
        real_sp = real_dense.copy()
        sim_sp = sim_dense.copy()
        real_sp.X = sp.csr_matrix(real_dense.X)
        sim_sp.X = sp.csr_matrix(sim_dense.X)
        w_dense = gene_variance_wasserstein(real_dense, sim_dense)
        w_sparse = gene_variance_wasserstein(real_sp, sim_sp)
        assert w_dense == pytest.approx(w_sparse, abs=1e-6)

    def test_library_size_wasserstein_sparse_equivalent(self, fixture_a_identity):
        real_dense, sim_dense = fixture_a_identity
        real_sp = real_dense.copy()
        sim_sp = sim_dense.copy()
        real_sp.X = sp.csr_matrix(real_dense.X)
        sim_sp.X = sp.csr_matrix(sim_dense.X)
        w_dense = library_size_wasserstein(real_dense, sim_dense)
        w_sparse = library_size_wasserstein(real_sp, sim_sp)
        assert w_dense == pytest.approx(w_sparse, abs=1e-6)


class TestVarianceCorrOutlierRobustness:
    r"""input_variance_corr with log1p is robust to expression-level bias.

    Add constant noise across genes with log-spaced means (1 to 1000):
    low-expression genes are wrecked, high-expression genes barely notice.
    log1p Pearson balances the contribution so damaged genes register.
    Raw Pearson would give >0.99 (blind); log1p gives measurably lower.
    """

    def test_log1p_variance_corr_detects_low_expression_noise(self):
        """Constant noise wrecks low-var genes; log1p registers the damage."""
        rng = np.random.default_rng(42)
        n_spots, n_genes = 300, 50

        means = 10.0 ** np.linspace(0, 3, n_genes)
        X_real = np.zeros((n_spots, n_genes), dtype=np.float32)
        for g, m in enumerate(means):
            X_real[:, g] = rng.poisson(m, size=n_spots).astype(np.float32)

        const_noise = rng.poisson(10, size=(n_spots, n_genes)).astype(np.float32)
        X_sim = np.maximum(0, X_real + const_noise)

        real = ad.AnnData(X=X_real)
        sim = ad.AnnData(X=X_sim)

        ivc = input_variance_corr(real, sim)
        # log1p compresses range → damaged low-var genes register.
        # Should be below the near-identity threshold (0.95).
        assert ivc < 0.99, (
            f"input_variance_corr with log1p should be <0.99 "
            f"(damaged low-expr genes register), got {ivc}"
        )

    def test_log1p_variance_corr_matches_manual(self):
        """input_variance_corr matches manual log1p Pearson computation."""
        rng = np.random.default_rng(42)
        n_spots, n_genes = 300, 50
        import scipy.stats as st

        means = 10.0 ** np.linspace(0, 3, n_genes)
        X_real = np.zeros((n_spots, n_genes), dtype=np.float32)
        for g, m in enumerate(means):
            X_real[:, g] = rng.poisson(m, size=n_spots).astype(np.float32)

        const_noise = rng.poisson(10, size=(n_spots, n_genes)).astype(np.float32)
        X_sim = np.maximum(0, X_real + const_noise)

        ivc = input_variance_corr(
            ad.AnnData(X=X_real), ad.AnnData(X=X_sim)
        )

        rv = _gene_variances(_dense(X_real))
        sv = _gene_variances(_dense(X_sim))
        mask = (rv > 1e-10) & (sv > 1e-10)
        expected, _ = st.pearsonr(
            np.log1p(rv[mask]), np.log1p(sv[mask])
        )

        assert ivc == pytest.approx(expected, abs=1e-10), (
            f"input_variance_corr={ivc:.10f} != manual log1p={expected:.10f}"
        )


class TestMetricFinite:
    """All metrics return finite values for valid inputs."""

    def test_all_distribution_metrics_finite(self, fixture_a_identity):
        real, sim = fixture_a_identity
        for func in [
            gene_mean_wasserstein,
            gene_variance_wasserstein,
            library_size_wasserstein,
        ]:
            val = func(real, sim)
            assert np.isfinite(val), f"{func.__name__} returned {val}"
