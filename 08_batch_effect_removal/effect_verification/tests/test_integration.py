"""End-to-end integration tests for batch assessment pipeline."""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

FEAST_ROOT = Path("/maiziezhou_lab2/yiru/FEAST")
sys.path.insert(0, str(FEAST_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_assessment.cloud_extraction import extract_clouds, filter_common_genes, stats_to_theta
from batch_assessment.metrics import (
    compute_centroids, compute_covariances, pairwise_distances, slice_qc_metrics,
)
from batch_assessment.affine_model import fit_diagonal_affine, residual_analysis
from batch_assessment.domain_analysis import (
    per_domain_clouds, domain_shift_summary, within_vs_between_domain_variance,
)


class TestEndToEnd:
    def test_full_pipeline_synthetic(self, synthetic_adata_dict):
        """Full pipeline on synthetic data runs without errors."""
        # QC
        qc = slice_qc_metrics(synthetic_adata_dict)
        assert qc.shape[0] == len(synthetic_adata_dict)

        # Cloud extraction
        clouds = extract_clouds(synthetic_adata_dict)
        assert len(clouds) == len(synthetic_adata_dict)

        # Centroids
        centroids = compute_centroids(clouds, ref_slice="S1")
        assert centroids["centroids"].shape == (len(clouds), 3)

        # Covariances
        cov_results = compute_covariances(clouds)
        assert cov_results["covariances"].shape == (len(clouds), 3, 3)

        # Distances
        dists = pairwise_distances(
            clouds, ["centroid_euclidean", "mmd_rbf", "sliced_wasserstein"]
        )
        n = len(clouds)
        for metric in dists:
            if metric == "slice_order":
                continue
            assert dists[metric].shape == (n, n)
            assert np.all(dists[metric] >= -1e-10)
            assert np.allclose(dists[metric], dists[metric].T)

        # Affine fit
        affine = fit_diagonal_affine(clouds, ref_slice="S1")
        assert affine["D_matrices"].shape == (n, 3)

        # Residuals
        resid = residual_analysis(clouds, affine)
        for sid in clouds:
            assert resid["residual_clouds"][sid].shape == clouds[sid]["theta"].shape

        # Domain analysis
        domain_clouds = per_domain_clouds(synthetic_adata_dict, domain_key="dlpfc_layer")
        assert len(domain_clouds) == len(synthetic_adata_dict)

        domain_summary = domain_shift_summary(domain_clouds, ref_domain="Layer_1")
        assert domain_summary.shape[0] > 0

        decomp = within_vs_between_domain_variance(domain_clouds)
        assert decomp["between_total_ratio"] is not None

    def test_full_pipeline_real_slices(self, real_adata_dict):
        """Full pipeline on real DLPFC slices validates key invariants."""
        pytest.importorskip("scanpy")
        if len(real_adata_dict) < 2:
            pytest.skip("Fewer than two real slices available")

        clouds = extract_clouds(real_adata_dict)
        clouds = filter_common_genes(clouds)
        assert len(clouds) >= 2

        # Basic invariants
        centroids = compute_centroids(clouds)
        assert centroids["centroids"].shape[0] == len(clouds)

        covs = compute_covariances(clouds)
        for fn in covs["frobenius_norm"]:
            assert fn > 0

        dists = pairwise_distances(clouds, ["centroid_euclidean", "mmd_rbf"])
        assert dists["centroid_euclidean"].shape == (len(clouds), len(clouds))
        assert np.allclose(dists["centroid_euclidean"], dists["centroid_euclidean"].T)
        assert np.all(np.diag(dists["centroid_euclidean"]) < 1e-8)

        # Affine should find some linear structure
        affine = fit_diagonal_affine(clouds, ref_slice=list(clouds.keys())[0])
        r2 = affine["r2_scores"]
        assert np.all(np.isfinite(r2))


class TestOutputArtifacts:
    def test_json_serializable(self, synthetic_theta_cloud):
        """Centroids and affine results are JSON-serializable."""
        centroids = compute_centroids(synthetic_theta_cloud)
        payload = {
            "slice_order": centroids["slice_order"],
            "centroids": centroids["centroids"].tolist(),
            "shifts": centroids["shifts"].tolist(),
        }
        json_str = json.dumps(payload)
        assert len(json_str) > 0

        affine = fit_diagonal_affine(synthetic_theta_cloud, ref_slice="A")
        payload2 = {
            "slice_order": affine["slice_order"],
            "D_matrices": affine["D_matrices"].tolist(),
            "b_vectors": affine["b_vectors"].tolist(),
        }
        json_str2 = json.dumps(payload2)
        assert len(json_str2) > 0

    def test_csv_reproducible(self, synthetic_theta_cloud):
        """Same input produces identical centroid output."""
        centroids1 = compute_centroids(synthetic_theta_cloud, ref_slice="A")
        centroids2 = compute_centroids(synthetic_theta_cloud, ref_slice="A")
        pd.testing.assert_frame_equal(
            pd.DataFrame(centroids1["centroids"]),
            pd.DataFrame(centroids2["centroids"]),
        )

    def test_qc_csv_columns_match(self, synthetic_adata_dict):
        """QC output for synthetic data has expected types."""
        qc = slice_qc_metrics(synthetic_adata_dict)
        for sid in synthetic_adata_dict:
            assert isinstance(qc.loc[sid, "total_counts_mean"], float)
            assert isinstance(qc.loc[sid, "n_spots"], (int, np.integer))
