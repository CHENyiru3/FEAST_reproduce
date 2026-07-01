"""Shared fixtures for batch effect assessment tests."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from anndata import AnnData


@pytest.fixture
def synthetic_theta_cloud():
    """Three synthetic theta clouds with known properties.

    A: baseline, standard normal N(0, I)
    B: mean-shifted along dim 0 (log_mu), same covariance as A
    C: inflated variance along dim 0, same centroid as A
    """
    rng = np.random.default_rng(42)
    G = 100

    cov_I = np.eye(3)
    theta_A = rng.multivariate_normal([0.0, 0.0, 0.0], cov_I, size=G)
    theta_B = rng.multivariate_normal([0.5, 0.0, 0.0], cov_I, size=G)
    cov_C = np.diag([2.0, 1.0, 1.0])
    theta_C = rng.multivariate_normal([0.0, 0.0, 0.0], cov_C, size=G)

    genes = [f"gene_{i}" for i in range(G)]
    return {
        "A": {"slice_id": "A", "theta": theta_A, "gene_ids": genes},
        "B": {"slice_id": "B", "theta": theta_B, "gene_ids": genes},
        "C": {"slice_id": "C", "theta": theta_C, "gene_ids": genes},
    }


@pytest.fixture
def synthetic_adata_dict():
    """Three small AnnData objects with controlled count distributions.

    S1: reference slice
    S2: higher mean expression (batch shift in mu)
    S3: same mean as S1 but higher zero proportion
    Each has spatial coordinates and fake domain labels.
    """
    rng = np.random.default_rng(42)
    n_spots, n_genes = 50, 20
    genes = [f"gene_{i}" for i in range(n_genes)]

    configs = {
        "S1": {"mu_shift": 0.0, "pi0_offset": 0.0},
        "S2": {"mu_shift": 0.3, "pi0_offset": 0.0},
        "S3": {"mu_shift": 0.0, "pi0_offset": 0.5},
    }
    result = {}

    for slice_id, cfg in configs.items():
        log_mu = rng.normal(0, 1, n_genes) + cfg["mu_shift"]
        mu = np.exp(log_mu) + 0.1
        var = mu * np.exp(rng.normal(0, 0.3, n_genes))
        pi0_logit = -1.0 + cfg["pi0_offset"] + rng.normal(0, 0.3, n_genes)
        pi0 = 1.0 / (1.0 + np.exp(-pi0_logit))
        pi0 = np.clip(pi0, 0.0, 0.9)

        X = np.zeros((n_spots, n_genes), dtype=np.float32)
        for g in range(n_genes):
            nonzero = rng.random(n_spots) > pi0[g]
            n_nz = int(nonzero.sum())
            if n_nz > 0:
                X[nonzero, g] = rng.poisson(mu[g], size=n_nz)

        adata = AnnData(X=X)
        adata.var_names = genes
        adata.obsm["spatial"] = rng.uniform(0, 1, (n_spots, 2)).astype(np.float32)
        adata.obs["dlpfc_layer"] = rng.choice(
            ["Layer_1", "Layer_2", "Layer_3", "WM"], size=n_spots
        )
        result[slice_id] = adata

    return result


@pytest.fixture
def real_adata_dict():
    """Load real DLPFC slices for integration tests.  Marked @pytest.mark.slow."""
    import sys
    import scanpy as sc

    feast_root = Path("/maiziezhou_lab2/yiru/FEAST")
    sys.path.insert(0, str(feast_root / "src"))

    data_dir = Path(
        "/maiziezhou_lab2/yiru/Datasets/Processed/spatialLIBD_DLPFC_Visium/h5ad"
    )
    slices = ["151508", "151670", "151676"]
    result = {}
    for s in slices:
        p = data_dir / f"{s}.h5ad"
        if p.exists():
            result[s] = sc.read_h5ad(p)
    return result
