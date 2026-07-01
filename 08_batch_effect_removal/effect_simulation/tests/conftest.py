"""Fixtures for batch simulation tests."""

import numpy as np
import pytest
from pathlib import Path
import sys

FEAST_ROOT = Path("/maiziezhou_lab2/yiru/FEAST")
sys.path.insert(0, str(FEAST_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def small_reference_adata():
    """Small synthetic AnnData serving as clean reference."""
    from anndata import AnnData
    rng = np.random.default_rng(42)
    n_spots, n_genes = 40, 20
    mu = np.exp(rng.normal(0, 0.5, n_genes)) + 0.3
    X = np.zeros((n_spots, n_genes), dtype=np.float32)
    for g in range(n_genes):
        nz = rng.random(n_spots) > 0.4
        n_nz = int(nz.sum())
        if n_nz > 0:
            X[nz, g] = rng.poisson(mu[g], size=n_nz)
    adata = AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["spatial"] = rng.uniform(0, 1, (n_spots, 2)).astype(np.float32)
    adata.obs["dlpfc_layer"] = rng.choice(
        ["Layer_1", "Layer_2", "Layer_3", "WM"], size=n_spots
    )
    return adata


@pytest.fixture
def shift_only_config():
    from effect_simulation.batch_simulator import DLPFC_SHIFT_ONLY
    return DLPFC_SHIFT_ONLY


@pytest.fixture
def diagonal_affine_config():
    from effect_simulation.batch_simulator import DLPFC_DIAGONAL_AFFINE
    return DLPFC_DIAGONAL_AFFINE
