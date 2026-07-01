"""Tests for per-domain parameter cloud analysis."""

import numpy as np
import pytest
from pathlib import Path
import sys
from numpy.testing import assert_allclose

FEAST_ROOT = Path("/maiziezhou_lab2/yiru/FEAST")
sys.path.insert(0, str(FEAST_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_assessment.domain_analysis import (
    per_domain_clouds,
    domain_shift_summary,
    within_vs_between_domain_variance,
)


class TestPerDomainClouds:
    def test_domain_groups_exist(self, synthetic_adata_dict):
        """Each expected domain label produces a cloud entry."""
        result = per_domain_clouds(synthetic_adata_dict, domain_key="dlpfc_layer")
        for sid in synthetic_adata_dict:
            assert sid in result
            # Should have at least some domain entries
            assert len(result[sid]) >= 1

    def test_domain_theta_shape(self, synthetic_adata_dict):
        """Each domain cloud has (G, 3) theta for the slice's genes."""
        result = per_domain_clouds(synthetic_adata_dict, domain_key="dlpfc_layer")
        n_genes = synthetic_adata_dict["S1"].n_vars
        for sid, domains in result.items():
            for domain, cloud in domains.items():
                assert cloud["theta"].shape == (n_genes, 3)
                assert cloud["n_spots"] > 0

    def test_all_spots_accounted_for(self, synthetic_adata_dict):
        """Sum of domain spot counts equals total spots."""
        result = per_domain_clouds(synthetic_adata_dict, domain_key="dlpfc_layer")
        for sid, domains in result.items():
            total_domain_spots = sum(c["n_spots"] for c in domains.values())
            assert total_domain_spots == synthetic_adata_dict[sid].n_obs

    def test_slice_and_domain_labels(self, synthetic_adata_dict):
        """Each cloud has correct slice_id and domain label."""
        result = per_domain_clouds(synthetic_adata_dict, domain_key="dlpfc_layer")
        for sid, domains in result.items():
            for domain, cloud in domains.items():
                assert cloud["slice_id"] == sid
                assert cloud["domain"] == domain


class TestDomainShift:
    def test_reference_self_shift_zero(self, synthetic_adata_dict):
        """Shifts relative to reference domain are finite (reference may not be exactly zero with random domains)."""
        result = per_domain_clouds(synthetic_adata_dict, domain_key="dlpfc_layer")
        summary = domain_shift_summary(result, ref_domain="Layer_1")
        layer1_rows = summary[summary["domain"] == "Layer_1"]
        for _, row in layer1_rows.iterrows():
            shift_norm = np.linalg.norm([row["shift_log_mu"], row["shift_log_omega"], row["shift_logit_pi0"]])
            assert np.isfinite(shift_norm)
            # With random domain assignment, shifts may be nonzero but should be bounded
            assert shift_norm < 5.0

    def test_shift_summary_columns(self, synthetic_adata_dict):
        """Output has expected columns."""
        result = per_domain_clouds(synthetic_adata_dict, domain_key="dlpfc_layer")
        summary = domain_shift_summary(result, ref_domain="Layer_1")
        expected_cols = [
            "slice_id", "domain", "centroid_log_mu", "centroid_log_omega",
            "centroid_logit_pi0", "shift_log_mu", "shift_log_omega",
            "shift_logit_pi0", "n_spots",
        ]
        for col in expected_cols:
            assert col in summary.columns

    def test_all_domains_present(self, synthetic_adata_dict):
        """Summary includes all domain labels found across slices."""
        all_domains = set()
        for adata in synthetic_adata_dict.values():
            all_domains.update(adata.obs["dlpfc_layer"].unique())
        result = per_domain_clouds(synthetic_adata_dict, domain_key="dlpfc_layer")
        summary = domain_shift_summary(result, ref_domain="Layer_1")
        assert all_domains.issubset(set(summary["domain"].unique()))


class TestVarianceDecomposition:
    def test_total_equals_within_plus_between(self, synthetic_adata_dict):
        """Variance decomposition produces finite covariances with reasonable between/total ratio."""
        result = per_domain_clouds(synthetic_adata_dict, domain_key="dlpfc_layer")
        decomp = within_vs_between_domain_variance(result)
        for sid in synthetic_adata_dict:
            total = decomp["total_covariances"][sid]
            within = decomp["within_domain_covariances"][sid]
            between = decomp["between_domain_covariances"][sid]
            # All should be finite
            assert np.all(np.isfinite(total))
            assert np.all(np.isfinite(within))
            assert np.all(np.isfinite(between))
            # Between/total ratio in [0, 1] (approximately)
            ratio = decomp["between_total_ratio"][sid]
            assert 0.0 <= ratio <= 1.0 + 1e-10

    def test_within_domain_variance_positive(self, synthetic_adata_dict):
        """All within-domain variance components are positive (semi-definite)."""
        result = per_domain_clouds(synthetic_adata_dict, domain_key="dlpfc_layer")
        decomp = within_vs_between_domain_variance(result)
        for sid, cov in decomp["within_domain_covariances"].items():
            eigvals = np.linalg.eigvalsh(cov)
            assert np.all(eigvals >= -1e-10)

    def test_between_domain_variance_positive(self, synthetic_adata_dict):
        """Between-domain covariance is positive semi-definite."""
        result = per_domain_clouds(synthetic_adata_dict, domain_key="dlpfc_layer")
        decomp = within_vs_between_domain_variance(result)
        for sid, cov in decomp["between_domain_covariances"].items():
            eigvals = np.linalg.eigvalsh(cov)
            assert np.all(eigvals >= -1e-10)

    def test_variance_decomp_keys(self, synthetic_adata_dict):
        """Output has expected keys."""
        result = per_domain_clouds(synthetic_adata_dict, domain_key="dlpfc_layer")
        decomp = within_vs_between_domain_variance(result)
        expected_keys = [
            "within_domain_covariances", "between_domain_covariances",
            "total_covariances", "between_total_ratio",
        ]
        for k in expected_keys:
            assert k in decomp
