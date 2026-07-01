"""Diagonal affine deformation model for batch effect characterisation.

Fits  theta_{s,g} = D_s * theta_{ref,g} + b_s + epsilon_{s,g}

where D_s = diag(d_mu, d_omega, d_pi0) captures per-dimension scaling
and b_s captures global shift.  This is ordinary least squares per coordinate.
"""

import numpy as np
import pandas as pd
from collections import OrderedDict

DIMENSION_NAMES = ["log_mu", "log_omega", "logit_pi0"]


def fit_diagonal_affine(clouds: dict, ref_slice: str) -> dict:
    """Fit diagonal affine transforms from reference to each slice.

    For each dimension k in {0, 1, 2}:
        theta_s[:, k] = d_k * theta_ref[:, k] + b_k  (+ epsilon)

    Parameters
    ----------
    clouds : dict  slice_id -> cloud with "theta" (G, 3)
    ref_slice : str

    Returns
    -------
    dict with keys:
        slice_order (list), ref_slice (str), dimension_names (list),
        D_matrices (S x 3), b_vectors (S x 3), r2_scores (S x 3),
        residual_variance (S x 3), params_df (DataFrame)
    """
    slice_order = list(clouds.keys())
    if ref_slice not in slice_order:
        ref_slice = slice_order[0]

    ref_idx = slice_order.index(ref_slice)
    S = len(slice_order)
    theta_ref = clouds[ref_slice]["theta"]

    D_matrices = np.ones((S, 3), dtype=np.float64)
    b_vectors = np.zeros((S, 3), dtype=np.float64)
    r2_scores = np.ones((S, 3), dtype=np.float64)
    residual_variance = np.zeros((S, 3), dtype=np.float64)

    for i, sid in enumerate(slice_order):
        if sid == ref_slice:
            continue
        theta_s = clouds[sid]["theta"]

        for k in range(3):
            x = theta_ref[:, k]
            y = theta_s[:, k]

            ss_xx = np.sum((x - x.mean()) ** 2)
            ss_yy = np.sum((y - y.mean()) ** 2)

            if ss_xx < 1e-15:
                d_k, b_k = 1.0, y.mean() - x.mean()
                r2 = 0.0
                resid_var = ss_yy / len(y)
            else:
                d_k = np.sum((x - x.mean()) * (y - y.mean())) / ss_xx
                b_k = y.mean() - d_k * x.mean()
                y_pred = d_k * x + b_k
                ss_res = np.sum((y - y_pred) ** 2)
                r2 = 1.0 - ss_res / max(ss_yy, 1e-15)
                r2 = max(0.0, min(1.0, r2))
                resid_var = ss_res / len(y)

            D_matrices[i, k] = d_k
            b_vectors[i, k] = b_k
            r2_scores[i, k] = r2
            residual_variance[i, k] = resid_var

    params_df = pd.DataFrame({
        "slice_id": slice_order,
        "d_mu": D_matrices[:, 0],
        "d_omega": D_matrices[:, 1],
        "d_pi0": D_matrices[:, 2],
        "b_mu": b_vectors[:, 0],
        "b_omega": b_vectors[:, 1],
        "b_pi0": b_vectors[:, 2],
        "r2_mu": r2_scores[:, 0],
        "r2_omega": r2_scores[:, 1],
        "r2_pi0": r2_scores[:, 2],
    }).set_index("slice_id")

    return OrderedDict([
        ("slice_order", slice_order),
        ("ref_slice", ref_slice),
        ("dimension_names", DIMENSION_NAMES),
        ("D_matrices", D_matrices),
        ("b_vectors", b_vectors),
        ("r2_scores", r2_scores),
        ("residual_variance", residual_variance),
        ("params_df", params_df),
    ])


def residual_analysis(clouds: dict, affine_result: dict) -> dict:
    """Compute residual clouds after removing the affine deformation.

    epsilon_{s,g} = theta_{s,g} - (D_s * theta_{ref,g} + b_s)

    Returns
    -------
    dict with keys:
        residual_clouds (dict: slice_id -> (G, 3) ndarray),
        residual_covariances (dict: slice_id -> (3, 3) ndarray),
        per_gene_residual_norms (dict: slice_id -> (G,) ndarray),
        residual_covariances (S x 3 x 3 ndarray)
    """
    slice_order = affine_result["slice_order"]
    ref_slice = affine_result["ref_slice"]
    ref_idx = slice_order.index(ref_slice)
    theta_ref = clouds[ref_slice]["theta"]

    S = len(slice_order)
    residual_clouds = {}
    per_gene_residual_norms = {}
    residual_covs = np.zeros((S, 3, 3), dtype=np.float64)

    for i, sid in enumerate(slice_order):
        if sid == ref_slice:
            resid = np.zeros_like(theta_ref)
        else:
            theta_s = clouds[sid]["theta"]
            D = affine_result["D_matrices"][i]
            b = affine_result["b_vectors"][i]
            resid = theta_s - (theta_ref * D[None, :] + b[None, :])

        residual_clouds[sid] = resid
        per_gene_residual_norms[sid] = np.linalg.norm(resid, axis=1)

        if sid != ref_slice or S == 1:
            residual_covs[i] = np.cov(resid.T)

    return OrderedDict([
        ("slice_order", slice_order),
        ("ref_slice", ref_slice),
        ("residual_clouds", residual_clouds),
        ("per_gene_residual_norms", per_gene_residual_norms),
        ("residual_covariances", residual_covs),
    ])
