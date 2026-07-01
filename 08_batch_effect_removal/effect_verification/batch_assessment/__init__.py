"""Batch effect assessment for FEAST parameter clouds.

Characterises slice-to-slice distributional differences in
(log_mu, log_omega, logit_pi0) space.
"""

from batch_assessment.cloud_extraction import (
    stats_to_theta,
    theta_to_stats,
    extract_cloud,
    extract_clouds,
    filter_common_genes,
)
from batch_assessment.metrics import (
    compute_centroids,
    compute_covariances,
    pairwise_distances,
    slice_qc_metrics,
)
from batch_assessment.affine_model import (
    fit_diagonal_affine,
    residual_analysis,
)
from batch_assessment.domain_analysis import (
    per_domain_clouds,
    domain_shift_summary,
    within_vs_between_domain_variance,
)

__all__ = [
    # cloud_extraction
    "stats_to_theta",
    "theta_to_stats",
    "extract_cloud",
    "extract_clouds",
    "filter_common_genes",
    # metrics
    "compute_centroids",
    "compute_covariances",
    "pairwise_distances",
    "slice_qc_metrics",
    # affine_model
    "fit_diagonal_affine",
    "residual_analysis",
    # domain_analysis
    "per_domain_clouds",
    "domain_shift_summary",
    "within_vs_between_domain_variance",
]
