"""Per-domain parameter cloud analysis.

Splits slices by spatial domain (e.g. cortical layers) to separate
biological from technical variation in the parameter cloud.
"""

import numpy as np
import pandas as pd
from collections import OrderedDict


def per_domain_clouds(adata_dict: dict, domain_key: str = "dlpfc_layer") -> dict:
    """Extract theta clouds for each domain within each slice.

    Parameters
    ----------
    adata_dict : dict  slice_id -> AnnData
    domain_key : str   key in adata.obs for domain labels

    Returns
    -------
    dict  slice_id -> {domain_label -> cloud_dict}
        Each cloud_dict has keys: slice_id, domain, gene_ids, theta, n_spots
    """
    from batch_assessment.cloud_extraction import stats_to_theta
    from FEAST.FEAST_core.simulator import _gene_stats_from_matrix

    result = OrderedDict()

    for sid, adata in adata_dict.items():
        from scipy.sparse import issparse
        X = np.asarray(adata.X.toarray() if issparse(adata.X) else adata.X,
                       dtype=np.float64)
        labels = adata.obs[domain_key].values
        genes = list(adata.var_names)
        unique_labels = sorted(set(labels))

        domains = OrderedDict()
        for label in unique_labels:
            mask = labels == label
            n_spots = int(np.sum(mask))
            if n_spots < 3:
                continue
            X_sub = X[mask, :]
            stats_df = _gene_stats_from_matrix(X_sub, genes)
            theta = stats_to_theta(stats_df)
            domains[label] = {
                "slice_id": sid,
                "domain": label,
                "gene_ids": genes,
                "theta": theta,
                "n_spots": n_spots,
            }
        result[sid] = domains

    return result


def domain_shift_summary(
    domain_clouds: dict, ref_domain: str = "Layer_1"
) -> pd.DataFrame:
    """Compute centroid shifts per (slice, domain) relative to a reference domain.

    Returns DataFrame with columns:
        slice_id, domain, centroid_log_mu, centroid_log_omega, centroid_logit_pi0,
        shift_log_mu, shift_log_omega, shift_logit_pi0, n_spots
    """
    rows = []

    # Compute reference centroid: average of the reference domain across all slices
    ref_centroids = []
    for sid, domains in domain_clouds.items():
        if ref_domain in domains:
            ref_centroids.append(domains[ref_domain]["theta"].mean(axis=0))
    if ref_centroids:
        ref_centroid = np.mean(ref_centroids, axis=0)
    else:
        ref_centroid = np.zeros(3)

    for sid, domains in domain_clouds.items():
        for domain, cloud in domains.items():
            centroid = cloud["theta"].mean(axis=0)
            shift = centroid - ref_centroid
            rows.append({
                "slice_id": sid,
                "domain": domain,
                "centroid_log_mu": centroid[0],
                "centroid_log_omega": centroid[1],
                "centroid_logit_pi0": centroid[2],
                "shift_log_mu": shift[0],
                "shift_log_omega": shift[1],
                "shift_logit_pi0": shift[2],
                "n_spots": cloud["n_spots"],
            })

    df = pd.DataFrame(rows)
    return df.sort_values(["slice_id", "domain"]).reset_index(drop=True)


def within_vs_between_domain_variance(domain_clouds: dict) -> dict:
    """Decompose total cloud variance into within-domain and between-domain.

    Law of Total Variance:
        Cov_total = Cov_within + Cov_between

    Returns
    -------
    dict with keys:
        within_domain_covariances (dict: slice_id -> 3x3 ndarray)
        between_domain_covariances (dict: slice_id -> 3x3 ndarray)
        total_covariances (dict: slice_id -> 3x3 ndarray)
        between_total_ratio (dict: slice_id -> float)
    """
    within = OrderedDict()
    between = OrderedDict()
    total = OrderedDict()
    ratios = OrderedDict()

    for sid, domains in domain_clouds.items():
        # Gather domain centroids and sizes
        domain_names = list(domains.keys())
        n_domains = len(domain_names)
        if n_domains < 2:
            continue

        centroids = np.zeros((n_domains, 3))
        weights = np.zeros(n_domains)
        covs = np.zeros((n_domains, 3, 3))

        for i, dname in enumerate(domain_names):
            theta = domains[dname]["theta"]
            centroids[i] = theta.mean(axis=0)
            weights[i] = domains[dname]["n_spots"]
            covs[i] = np.cov(theta.T)

        total_weight = weights.sum()
        weights_norm = weights / total_weight

        # Within-domain covariance: weighted average of per-domain covariances
        within_cov = np.sum(
            covs * weights_norm[:, None, None], axis=0
        )

        # Between-domain covariance: weighted covariance of centroids
        grand_centroid = np.sum(centroids * weights_norm[:, None], axis=0)
        centered = centroids - grand_centroid[None, :]
        between_cov = np.sum(
            centered[:, :, None] * centered[:, None, :] * weights_norm[:, None, None],
            axis=0
        )

        # Total covariance: compute from all spots in the slice
        all_theta = np.vstack([domains[d]["theta"] for d in domain_names])
        total_cov = np.cov(all_theta.T)

        within[sid] = within_cov
        between[sid] = between_cov
        total[sid] = total_cov

        total_trace = np.trace(total_cov)
        between_trace = np.trace(between_cov)
        ratios[sid] = float(between_trace / max(total_trace, 1e-15))

    return OrderedDict([
        ("within_domain_covariances", within),
        ("between_domain_covariances", between),
        ("total_covariances", total),
        ("between_total_ratio", ratios),
    ])
