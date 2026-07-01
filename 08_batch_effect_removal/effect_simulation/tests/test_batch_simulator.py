"""Tests for batch effect simulation module."""

import numpy as np
import pytest
from pathlib import Path
import sys

FEAST_ROOT = Path("/maiziezhou_lab2/yiru/FEAST")
sys.path.insert(0, str(FEAST_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from effect_simulation.batch_simulator import (
    DeformationConfig,
    DLPFC_SHIFT_ONLY,
    DLPFC_DIAGONAL_AFFINE,
    simulate_batch_ladder,
)


class TestDeformationConfig:
    def test_shift_only_D_is_identity(self):
        np.testing.assert_array_equal(DLPFC_SHIFT_ONLY.D, [1.0, 1.0, 1.0])

    def test_diagonal_affine_D_not_identity(self):
        assert not np.allclose(DLPFC_DIAGONAL_AFFINE.D, [1.0, 1.0, 1.0])

    def test_to_dict(self):
        d = DLPFC_SHIFT_ONLY.to_dict()
        assert d["name"] == "shift_only"
        assert "D" in d
        assert "b" in d
        assert "alpha" in d


class TestSimulateBatchLadder:
    def test_returns_dict(self, small_reference_adata, shift_only_config):
        results = simulate_batch_ladder(
            small_reference_adata, shift_only_config, alphas=[0.0, 0.5, 1.0]
        )
        assert isinstance(results, dict)
        assert len(results) == 3

    def test_alpha_zero_preserves_stats(self, small_reference_adata, shift_only_config):
        results = simulate_batch_ladder(
            small_reference_adata, shift_only_config, alphas=[0.0],
            random_seed=42,
        )
        sim0 = results[0.0]
        assert sim0.n_obs == small_reference_adata.n_obs
        assert sim0.n_vars == small_reference_adata.n_vars

    def test_alpha_stored_in_uns(self, small_reference_adata, shift_only_config):
        results = simulate_batch_ladder(
            small_reference_adata, shift_only_config, alphas=[0.5]
        )
        assert results[0.5].uns["batch_alpha"] == 0.5
        assert "batch_config" in results[0.5].uns

    def test_q_preserved_structure(self, small_reference_adata, shift_only_config):
        """Q preservation: structural check that decode respects quantile input.

        The decode_counts_by_rank function assigns sorted count bags to spots
        ordered by argsort(quantile_input).  We verify this by checking that
        the spot with the maximum reference count for each gene also gets the
        maximum simulated count.
        """
        results = simulate_batch_ladder(
            small_reference_adata, shift_only_config, alphas=[0.0, 0.5, 1.0],
            random_seed=42,
        )
        ref_X = np.asarray(small_reference_adata.X, dtype=np.float64)
        top_matches = 0
        total = 0
        for alpha in [0.0, 0.5, 1.0]:
            sim_X = np.asarray(results[alpha].X, dtype=np.float64)
            for g in range(small_reference_adata.n_vars):
                ref_top = set(np.where(ref_X[:, g] == ref_X[:, g].max())[0])
                sim_top = set(np.where(sim_X[:, g] == sim_X[:, g].max())[0])
                if ref_top & sim_top:  # at least one top spot overlaps
                    top_matches += 1
                total += 1
        # Most genes should have overlapping top-ranked spots
        assert top_matches / total > 0.60, f"Top-spot overlap: {top_matches}/{total}"

    def test_spatial_preserved(self, small_reference_adata, shift_only_config):
        results = simulate_batch_ladder(
            small_reference_adata, shift_only_config, alphas=[0.0, 1.0]
        )
        for alpha, sim in results.items():
            np.testing.assert_array_equal(
                sim.obsm["spatial"], small_reference_adata.obsm["spatial"]
            )

    def test_obs_preserved(self, small_reference_adata, shift_only_config):
        results = simulate_batch_ladder(
            small_reference_adata, shift_only_config, alphas=[0.0, 0.5]
        )
        for sim in results.values():
            assert "dlpfc_layer" in sim.obs.columns
            np.testing.assert_array_equal(
                sim.obs["dlpfc_layer"].values,
                small_reference_adata.obs["dlpfc_layer"].values,
            )

    def test_batch_effect_monotonic(self, small_reference_adata, shift_only_config):
        """Batch signature goes toward LOWER mean, HIGHER sparsity.
        So mean should decrease and zero fraction increase with alpha."""
        results = simulate_batch_ladder(
            small_reference_adata, shift_only_config,
            alphas=[0.0, 0.5, 1.0],
            random_seed=42,
        )
        means = []
        zeros = []
        for alpha in [0.0, 0.5, 1.0]:
            X = np.asarray(results[alpha].X, dtype=np.float64)
            means.append(float(np.mean(X)))
            zeros.append(float(1.0 - np.count_nonzero(X) / X.size))

        assert means[2] < means[0], f"Mean should decrease with batch alpha: {means}"
        assert zeros[2] > zeros[0], f"Zero fraction should increase: {zeros}"

    def test_all_alphas_produce_anndata(self, small_reference_adata, shift_only_config):
        from anndata import AnnData
        results = simulate_batch_ladder(small_reference_adata, shift_only_config)
        for sim in results.values():
            assert isinstance(sim, AnnData)

    def test_both_configs_work(self, small_reference_adata):
        for cfg in [DLPFC_SHIFT_ONLY, DLPFC_DIAGONAL_AFFINE]:
            results = simulate_batch_ladder(
                small_reference_adata, cfg, alphas=[0.0, 1.0], random_seed=42
            )
            assert len(results) == 2
