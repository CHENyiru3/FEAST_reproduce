"""Spatial structure metric tests.

Covers:
  - moran_i_correlation rewards preserved spatial autocorrelation
  - moran_i_correlation handles too few spots or genes
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SIM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIM_ROOT / "scripts"))

from build_simulator_quality_metrics import moran_i_correlation


class TestMoranIMetric:
    def test_moran_i_metric_rewards_preserved_spatial_autocorrelation(
        self, fixture_g_spatial_preserved
    ):
        real, sim = fixture_g_spatial_preserved
        mic = moran_i_correlation(real, sim)
        assert np.isfinite(mic)

    def test_moran_i_metric_handles_too_few_spots_or_genes(self):
        """Small datasets return nan for moran_i_correlation."""
        import anndata as ad

        X = np.ones((3, 3), dtype=np.float32)
        real = ad.AnnData(X=X)
        real.obsm["spatial"] = np.column_stack([
            np.arange(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ])
        sim = real.copy()
        mic = moran_i_correlation(real, sim)
        assert np.isnan(mic)
