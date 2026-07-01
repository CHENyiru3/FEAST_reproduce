"""Theta coordinate transforms and parameter-cloud extraction.

theta = [log(mu), log(omega), logit(pi0)]
where omega = variance / mean (Fano factor).
"""

import numpy as np
import pandas as pd
from typing import Optional

from FEAST import stats_to_theta, theta_to_stats


def extract_cloud(adata, gene_names: Optional[list] = None) -> dict:
    """Extract the theta parameter cloud from an AnnData slice.

    Returns dict with keys: slice_id, gene_ids, theta, stats_df.
    """
    from FEAST.FEAST_core.simulator import _gene_stats_from_matrix
    from scipy.sparse import issparse

    var_names = gene_names if gene_names is not None else list(adata.var_names)
    X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X, dtype=np.float64)
    stats_df = _gene_stats_from_matrix(X, var_names)
    theta = stats_to_theta(stats_df)
    return {
        "slice_id": getattr(adata, "uns", {}).get("sample_id", None),
        "gene_ids": var_names,
        "theta": theta,
        "stats_df": stats_df,
    }


def extract_clouds(
    adata_dict: dict, gene_filter: Optional[dict] = None
) -> dict:
    """Extract theta clouds from multiple AnnData slices.

    Parameters
    ----------
    adata_dict : dict[str, AnnData]
    gene_filter : dict, optional
        Keys: min_mean, min_spots_detected, common_genes_only.

    Returns
    -------
    dict[str, dict]  -- slice_id -> cloud dict
    """
    clouds = {}
    for sid, adata in adata_dict.items():
        cloud = extract_cloud(adata)
        if cloud["slice_id"] is None:
            cloud["slice_id"] = sid
        clouds[sid] = cloud

    if gene_filter:
        clouds = _apply_gene_filter(clouds, adata_dict, gene_filter)

    if gene_filter and gene_filter.get("common_genes_only", False):
        clouds = filter_common_genes(clouds)

    return clouds


def filter_common_genes(clouds: dict) -> dict:
    """Intersect gene IDs across clouds and align ordering.

    Returns new clouds dict with identical gene_ids in identical order.
    """
    gene_sets = [set(c["gene_ids"]) for c in clouds.values()]
    common = gene_sets[0]
    for gs in gene_sets[1:]:
        common = common & gs
    common = sorted(common)

    filtered = {}
    for sid, cloud in clouds.items():
        idx_map = {g: i for i, g in enumerate(cloud["gene_ids"])}
        keep_idx = [idx_map[g] for g in common]
        filtered[sid] = {
            "slice_id": cloud["slice_id"] or sid,
            "gene_ids": common,
            "theta": cloud["theta"][keep_idx, :],
            "stats_df": cloud["stats_df"].iloc[keep_idx].reset_index(drop=True),
        }
    return filtered


def _apply_gene_filter(clouds: dict, adata_dict: dict, gene_filter: dict) -> dict:
    """Filter clouds and AnnData objects by gene-level criteria."""
    min_mean = gene_filter.get("min_mean", None)
    min_spots = gene_filter.get("min_spots_detected", None)

    if min_mean is None and min_spots is None:
        return clouds

    for sid, cloud in clouds.items():
        adata = adata_dict[sid]
        mask = np.ones(adata.n_vars, dtype=bool)

        if min_mean is not None:
            mask &= cloud["stats_df"]["mean"].values >= min_mean

        if min_spots is not None:
            n_detected = (adata.X.toarray() if hasattr(adata.X, "toarray")
                          else adata.X.todense() if hasattr(adata.X, "todense")
                          else adata.X) > 0
            n_detected = np.asarray((n_detected).sum(axis=0)).ravel()
            mask &= n_detected >= min_spots

        idx = np.where(mask)[0]
        clouds[sid] = {
            "slice_id": cloud["slice_id"] or sid,
            "gene_ids": [cloud["gene_ids"][i] for i in idx],
            "theta": cloud["theta"][idx, :],
            "stats_df": cloud["stats_df"].iloc[idx].reset_index(drop=True),
        }

    return clouds
