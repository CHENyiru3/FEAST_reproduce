"""Parameter-cloud metrics: centroids, covariances, distances, QC.

Computes slice-level summary statistics and pairwise dissimilarities
between FEAST parameter clouds in theta space.
"""

import numpy as np
import pandas as pd
from typing import Optional
from collections import OrderedDict


def compute_centroids(clouds: dict, ref_slice: Optional[str] = None) -> dict:
    """Compute per-slice centroids and batch shifts relative to reference.

    Parameters
    ----------
    clouds : dict  slice_id -> cloud with key "theta" (G, 3)
    ref_slice : str, optional  reference for shift computation

    Returns
    -------
    dict with keys: centroids (S x 3 ndarray), shifts (S x 3 ndarray),
    slice_order (list), ref_slice (str), dimension_names (list)
    """
    slice_order = list(clouds.keys())
    S = len(slice_order)
    centroids = np.zeros((S, 3), dtype=np.float64)
    for i, sid in enumerate(slice_order):
        centroids[i] = clouds[sid]["theta"].mean(axis=0)

    if ref_slice is not None and ref_slice not in slice_order:
        ref_slice = slice_order[0]
    if ref_slice is None:
        ref_slice = slice_order[0]

    ref_idx = slice_order.index(ref_slice)
    shifts = centroids - centroids[ref_idx]

    return OrderedDict([
        ("slice_order", slice_order),
        ("ref_slice", ref_slice),
        ("dimension_names", ["log_mu", "log_omega", "logit_pi0"]),
        ("centroids", centroids),
        ("shifts", shifts),
    ])


def compute_covariances(clouds: dict) -> dict:
    """Compute per-slice 3x3 covariance matrices and scalar summaries.

    Returns
    -------
    dict with keys: slice_order, covariances (S x 3 x 3), log_det (S,),
    frobenius_norm (S,), eigenvalues (S x 3)
    """
    slice_order = list(clouds.keys())
    S = len(slice_order)
    covariances = np.zeros((S, 3, 3), dtype=np.float64)
    log_det = np.zeros(S, dtype=np.float64)
    frob_norm = np.zeros(S, dtype=np.float64)
    eigenvalues = np.zeros((S, 3), dtype=np.float64)

    for i, sid in enumerate(slice_order):
        cov = np.cov(clouds[sid]["theta"].T)
        covariances[i] = cov
        sign, logdet = np.linalg.slogdet(cov)
        log_det[i] = logdet if sign > 0 else np.nan
        frob_norm[i] = np.linalg.norm(cov, ord="fro")
        eigenvalues[i] = np.sort(np.linalg.eigvalsh(cov))[::-1]

    # principal angles between top-2 subspace of each pair
    principal_angles = None
    if S >= 2:
        principal_angles = np.zeros((S, S), dtype=np.float64)
        for i in range(S):
            for j in range(S):
                if i == j:
                    principal_angles[i, j] = 0.0
                elif j > i:
                    Ui = _top_eigenvectors(covariances[i], k=2)
                    Uj = _top_eigenvectors(covariances[j], k=2)
                    _, s, _ = np.linalg.svd(Ui.T @ Uj)
                    theta = np.arccos(np.clip(s, 0, 1))
                    principal_angles[i, j] = np.max(theta) if len(theta) > 0 else 0.0
                    principal_angles[j, i] = principal_angles[i, j]

    return OrderedDict([
        ("slice_order", slice_order),
        ("covariances", covariances),
        ("log_det", log_det),
        ("frobenius_norm", frob_norm),
        ("eigenvalues", eigenvalues),
        ("principal_angles", principal_angles),
    ])


def _top_eigenvectors(cov: np.ndarray, k: int = 2) -> np.ndarray:
    """Return top-k eigenvectors of a symmetric matrix."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvecs[:, ::-1][:, :k]


def pairwise_distances(clouds: dict, metrics: list) -> dict:
    """Compute S x S pairwise distance matrices for each requested metric.

    Available metrics:
    - centroid_euclidean:  ||m_i - m_j||_2
    - covariance_frobenius: ||Sigma_i - Sigma_j||_F
    - covariance_riemannian: affine-invariant Riemannian metric on SPD(3)
    - mmd_rbf: Maximum Mean Discrepancy with RBF kernel
    - sliced_wasserstein: 1D projection average (approximate 3D Wasserstein)
    """
    slice_order = list(clouds.keys())
    S = len(slice_order)
    thetas = [clouds[s]["theta"] for s in slice_order]

    result = OrderedDict()
    for m in metrics:
        result[m] = np.zeros((S, S), dtype=np.float64)

    for i in range(S):
        for j in range(i, S):
            if "centroid_euclidean" in metrics:
                mi = thetas[i].mean(axis=0)
                mj = thetas[j].mean(axis=0)
                d = np.linalg.norm(mi - mj)
                result["centroid_euclidean"][i, j] = d
                result["centroid_euclidean"][j, i] = d

            if "covariance_frobenius" in metrics:
                cov_i = np.cov(thetas[i].T)
                cov_j = np.cov(thetas[j].T)
                d = np.linalg.norm(cov_i - cov_j, ord="fro")
                result["covariance_frobenius"][i, j] = d
                result["covariance_frobenius"][j, i] = d

            if "covariance_riemannian" in metrics:
                cov_i = np.cov(thetas[i].T)
                cov_j = np.cov(thetas[j].T)
                d = _riemannian_distance(cov_i, cov_j)
                result["covariance_riemannian"][i, j] = d
                result["covariance_riemannian"][j, i] = d

            if "mmd_rbf" in metrics:
                d = _mmd_rbf(thetas[i], thetas[j])
                result["mmd_rbf"][i, j] = d
                result["mmd_rbf"][j, i] = d

            if "sliced_wasserstein" in metrics:
                d = _sliced_wasserstein(thetas[i], thetas[j], n_projections=100)
                result["sliced_wasserstein"][i, j] = d
                result["sliced_wasserstein"][j, i] = d

    result["slice_order"] = slice_order
    return result


def _riemannian_distance(S1: np.ndarray, S2: np.ndarray) -> float:
    """Affine-invariant Riemannian distance between two SPD matrices.

    d(S1, S2) = ||log(S1^{-1/2} S2 S1^{-1/2})||_F
    """
    from scipy.linalg import sqrtm
    eps = 1e-8 * np.eye(S1.shape[0])
    S1_reg = S1 + eps
    S2_reg = S2 + eps
    S1_inv_sqrt = np.linalg.inv(sqrtm(S1_reg))
    M = S1_inv_sqrt @ S2_reg @ S1_inv_sqrt
    logM = sqrtm(M)
    logM = np.real(np.linalg.eigvals(logM))
    logM = np.where(logM > 0, np.log(logM), 0)
    # Actually compute: log(S1^{-1/2} S2 S1^{-1/2}) via eigendecomposition
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.clip(eigvals, 1e-10, None)
    log_eigvals = np.log(eigvals)
    return float(np.sqrt(np.sum(log_eigvals ** 2)))


def _mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> float:
    """Maximum Mean Discrepancy with RBF kernel.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)]
    """
    from sklearn.metrics import pairwise_kernels

    if gamma is None:
        all_data = np.vstack([X, Y])
        dists = np.sum((all_data[:, None, :] - all_data[None, :, :]) ** 2, axis=-1)
        mask = ~np.eye(len(all_data), dtype=bool)
        median_dist = np.median(np.sqrt(dists[mask]))
        gamma = 1.0 / (2.0 * (max(median_dist, 1e-6) ** 2))

    K_XX = pairwise_kernels(X, metric="rbf", gamma=gamma)
    K_YY = pairwise_kernels(Y, metric="rbf", gamma=gamma)
    K_XY = pairwise_kernels(X, Y, metric="rbf", gamma=gamma)

    mmd2 = float(np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY))
    return np.sqrt(max(mmd2, 0.0))


def _sliced_wasserstein(
    X: np.ndarray, Y: np.ndarray, n_projections: int = 100, seed: int = 42
) -> float:
    """Approximate 3D Wasserstein-1 distance via random 1D projections."""
    from scipy.stats import wasserstein_distance
    rng = np.random.default_rng(seed)
    dim = X.shape[1]
    distances = np.zeros(n_projections)

    for p in range(n_projections):
        direction = rng.normal(size=dim)
        direction /= np.linalg.norm(direction)
        proj_X = X @ direction
        proj_Y = Y @ direction
        distances[p] = wasserstein_distance(proj_X, proj_Y)

    return float(np.mean(distances))


def slice_qc_metrics(adata_dict: dict) -> pd.DataFrame:
    """Basic QC metrics per slice: total counts, genes detected, zero fraction, etc.

    Returns DataFrame indexed by slice_id.
    """
    rows = []
    for sid, adata in adata_dict.items():
        from scipy.sparse import issparse
        X = np.asarray(adata.X.toarray() if issparse(adata.X) else adata.X,
                       dtype=np.float64)
        total_counts = X.sum(axis=1)
        rows.append({
            "slice_id": sid,
            "total_counts_mean": float(np.mean(total_counts)),
            "total_counts_std": float(np.std(total_counts)),
            "detected_genes": int(np.sum(X.sum(axis=0) > 0)),
            "zero_fraction": float(1.0 - np.count_nonzero(X) / X.size),
            "mean_of_means": float(np.mean(np.mean(X, axis=0))),
            "mean_of_variances": float(np.mean(np.var(X, axis=0))),
            "mean_of_zero_prop": float(np.mean(
                1.0 - np.count_nonzero(X, axis=0) / X.shape[0]
            )),
            "n_spots": X.shape[0],
        })

    qc = pd.DataFrame(rows).set_index("slice_id")
    qc.index.name = None
    return qc
