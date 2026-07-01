"""Zero-structure metric tests.

Covers:
  - gene_zero_fraction_wasserstein detects distribution match
  - zero_mask_jaccard detects exact zero copying
  - same zero fraction, different mask → not identity
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SIM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIM_ROOT / "scripts"))

from build_simulator_quality_metrics import (
    IDENTITY_MEAN_CORR_THRESHOLD,
    IDENTITY_VAR_CORR_THRESHOLD,
    IDENTITY_ZERO_JACCARD_THRESHOLD,
    gene_zero_fraction_wasserstein,
    input_mean_corr,
    input_variance_corr,
    zero_mask_jaccard,
)


def _is_identity(real, sim):
    return bool(
        input_mean_corr(real, sim) >= IDENTITY_MEAN_CORR_THRESHOLD
        and input_variance_corr(real, sim) >= IDENTITY_VAR_CORR_THRESHOLD
        and zero_mask_jaccard(real, sim) >= IDENTITY_ZERO_JACCARD_THRESHOLD
    )


class TestGeneZeroFractionWasserstein:
    def test_zero_for_identity(self, fixture_a_identity):
        real, sim = fixture_a_identity
        w = gene_zero_fraction_wasserstein(real, sim)
        assert w == pytest.approx(0.0, abs=1e-8)

    def test_bounded(self, fixture_e_noise):
        real, sim = fixture_e_noise
        w = gene_zero_fraction_wasserstein(real, sim)
        assert w >= 0.0


class TestZeroMaskJaccard:
    def test_detects_exact_zero_copying(self, fixture_a_identity):
        real, sim = fixture_a_identity
        zmj = zero_mask_jaccard(real, sim)
        assert zmj == pytest.approx(1.0, abs=1e-8)

    def test_bounded_0_1(self, fixture_e_noise):
        real, sim = fixture_e_noise
        zmj = zero_mask_jaccard(real, sim)
        assert 0.0 <= zmj <= 1.0


class TestSameZeroFractionDifferentMask:
    """Fixture D: same per-gene zero count, different zero positions."""

    def test_not_identity(self, fixture_d_same_zero_diff_mask):
        real, sim = fixture_d_same_zero_diff_mask
        assert _is_identity(real, sim) is False

    def test_gene_zero_fraction_wasserstein_zero_for_same_fraction(
        self, fixture_d_same_zero_diff_mask
    ):
        real, sim = fixture_d_same_zero_diff_mask
        w = gene_zero_fraction_wasserstein(real, sim)
        assert w == pytest.approx(0.0, abs=1e-8)

    def test_zero_mask_jaccard_below_identity_threshold(
        self, fixture_d_same_zero_diff_mask
    ):
        real, sim = fixture_d_same_zero_diff_mask
        zmj = zero_mask_jaccard(real, sim)
        assert zmj < 0.95
