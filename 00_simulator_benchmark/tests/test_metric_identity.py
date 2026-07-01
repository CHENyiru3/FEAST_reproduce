"""Identity & copying metric tests.

Covers:
  - Identity copy sets identity flag (3 conditions)
  - Near-identity sets identity flag
  - Permuted genes do not set identity flag
  - Zero mask jaccard bounds
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SIM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIM_ROOT / "scripts"))

from build_simulator_quality_metrics import (
    input_mean_corr,
    input_variance_corr,
    zero_mask_jaccard,
    IDENTITY_MEAN_CORR_THRESHOLD,
    IDENTITY_VAR_CORR_THRESHOLD,
    IDENTITY_ZERO_JACCARD_THRESHOLD,
)


def _is_identity(real, sim):
    """Check all 3 identity threshold conditions."""
    return bool(
        input_mean_corr(real, sim) >= IDENTITY_MEAN_CORR_THRESHOLD
        and input_variance_corr(real, sim) >= IDENTITY_VAR_CORR_THRESHOLD
        and zero_mask_jaccard(real, sim) >= IDENTITY_ZERO_JACCARD_THRESHOLD
    )


class TestIdentityCopy:
    """Fixture A: exact identity."""

    def test_identity_copy_sets_identity_flag(self, fixture_a_identity):
        real, sim = fixture_a_identity
        assert input_mean_corr(real, sim) == pytest.approx(1.0, abs=1e-8)
        assert input_variance_corr(real, sim) == pytest.approx(1.0, abs=1e-8)
        assert zero_mask_jaccard(real, sim) == pytest.approx(1.0, abs=1e-8)
        assert _is_identity(real, sim) is True

    def test_identity_copy_zero_mask_perfect(self, fixture_a_identity):
        real, sim = fixture_a_identity
        assert zero_mask_jaccard(real, sim) == pytest.approx(1.0, abs=1e-8)


class TestNearIdentity:
    """Fixture B: near-identity with small noise."""

    def test_near_identity_sets_identity_flag(self, fixture_b_near_identity):
        real, sim = fixture_b_near_identity
        assert input_mean_corr(real, sim) >= 0.995
        assert input_variance_corr(real, sim) >= 0.95
        assert zero_mask_jaccard(real, sim) >= 0.95
        assert _is_identity(real, sim) is True


class TestPermutedGeneIdentity:
    """Fixture C: permuted genes."""

    def test_permuted_gene_identity_does_not_set_identity_flag(
        self, fixture_c_permuted_genes
    ):
        real, sim = fixture_c_permuted_genes
        imc = input_mean_corr(real, sim)
        assert imc < 0.99
        assert _is_identity(real, sim) is False

    def test_permuted_genes_preserve_gene_zero_distribution(
        self, fixture_c_permuted_genes
    ):
        """Distributional zero fractions unchanged despite permuted columns."""
        real, sim = fixture_c_permuted_genes
        from build_simulator_quality_metrics import gene_zero_fraction_wasserstein
        gzw = gene_zero_fraction_wasserstein(real, sim)
        assert gzw == pytest.approx(0.0, abs=1e-8)

    def test_permuted_genes_preserve_library_sizes(self, fixture_c_permuted_genes):
        real, sim = fixture_c_permuted_genes
        from build_simulator_quality_metrics import library_size_wasserstein
        lw = library_size_wasserstein(real, sim)
        assert lw == pytest.approx(0.0, abs=1e-8)


class TestDistributionPreservedNonIdentity:
    """Fixture F: per-gene distribution preserved, spot assignments shuffled."""

    def test_distribution_preserved_not_identity(self, fixture_f_dist_preserved):
        real, sim = fixture_f_dist_preserved
        assert _is_identity(real, sim) is False

    def test_distribution_preserved_low_gene_mean_wasserstein(
        self, fixture_f_dist_preserved
    ):
        real, sim = fixture_f_dist_preserved
        from build_simulator_quality_metrics import gene_mean_wasserstein
        gmw = gene_mean_wasserstein(real, sim)
        assert gmw == pytest.approx(0.0, abs=1e-8)
