"""Shared fixtures for Figure 2 single-metric quality tests.

Fixtures A-H from SPEC: single_metrics_test_plan.md
All fixtures are deterministic (fixed seeds) and use small synthetic matrices.
"""

import numpy as np
import anndata as ad
import pytest

SEED = 42


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_adata(X, *, with_spatial=True):
    """Create a minimal AnnData with optional spatial coordinates."""
    n_obs = X.shape[0]
    adata = ad.AnnData(X=X.astype(np.float32))
    if with_spatial:
        adata.obsm["spatial"] = np.column_stack([
            np.arange(n_obs, dtype=np.float32),
            np.zeros(n_obs, dtype=np.float32),
        ])
    adata.var_names = [f"gene_{i}" for i in range(X.shape[1])]
    adata.obs_names = [f"spot_{i}" for i in range(n_obs)]
    return adata


# ---------------------------------------------------------------------------
# Fixture A: Identity copy
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_a_identity():
    """real == sim. Should trigger identity_flag, all zero distances == 0."""
    rng = np.random.default_rng(SEED)
    X = rng.poisson(5, size=(50, 20)).astype(np.float32)
    real = _make_adata(X)
    sim = _make_adata(X.copy())
    return real, sim


# ---------------------------------------------------------------------------
# Fixture B: Near-identity with small noise
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_b_near_identity():
    """Perturb a few positive counts. Should still trigger identity_flag."""
    rng = np.random.default_rng(SEED)
    X = rng.poisson(8, size=(50, 20)).astype(np.float32)
    X_sim = X.copy()
    # Perturb ~1% of positive entries
    pos_mask = X_sim > 0
    n_pos = int(pos_mask.sum())
    n_perturb = max(1, n_pos // 100)
    pos_idx = np.where(pos_mask)
    perturb = rng.choice(n_pos, size=n_perturb, replace=False)
    for idx in range(n_perturb):
        r, c = pos_idx[0][perturb[idx]], pos_idx[1][perturb[idx]]
        X_sim[r, c] += int(rng.integers(-2, 3))
        X_sim[r, c] = max(0, X_sim[r, c])
    real = _make_adata(X)
    sim = _make_adata(X_sim)
    return real, sim


# ---------------------------------------------------------------------------
# Fixture C: Permuted gene identity
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_c_permuted_genes():
    """Gene columns permuted. Spot and gene marginals preserved exactly."""
    rng = np.random.default_rng(SEED)
    X = rng.poisson(5, size=(50, 20)).astype(np.float32)
    perm = rng.permutation(X.shape[1])
    real = _make_adata(X)
    sim = _make_adata(X[:, perm])
    return real, sim


# ---------------------------------------------------------------------------
# Fixture D: Same zero fractions, different zero mask
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_d_same_zero_diff_mask():
    """For each gene, preserve zero count but randomize zero positions."""
    rng = np.random.default_rng(SEED)
    X = rng.poisson(3, size=(50, 20)).astype(np.float32)
    X_sim = X.copy()
    for g in range(X.shape[1]):
        n_zeros = int(np.sum(X[:, g] == 0))
        if n_zeros == 0:
            continue
        # Keep positive values, shuffle into new positions
        positives = X_sim[X_sim[:, g] > 0, g].copy()
        if len(positives) == 0:
            continue
        n_pos = len(positives)
        new_vals = np.zeros(50, dtype=np.float32)
        pos_slots = rng.choice(50, size=n_pos, replace=False)
        new_vals[pos_slots] = positives
        X_sim[:, g] = new_vals
    real = _make_adata(X)
    sim = _make_adata(X_sim)
    return real, sim


# ---------------------------------------------------------------------------
# Fixture E: Novel but unrealistic noise
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_e_noise():
    """Random Poisson(1) counts. High novelty, poor fidelity."""
    rng = np.random.default_rng(SEED)
    X_real = rng.poisson(5, size=(50, 20)).astype(np.float32)
    X_sim = rng.poisson(1, size=(50, 20)).astype(np.float32)
    real = _make_adata(X_real)
    sim = _make_adata(X_sim)
    return real, sim


# ---------------------------------------------------------------------------
# Fixture F: Distribution-preserved non-identity
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_f_dist_preserved():
    """Per-gene empirical distribution preserved but spot assignments changed."""
    rng = np.random.default_rng(SEED)
    X = rng.poisson(5, size=(50, 20)).astype(np.float32)
    X_sim = np.zeros_like(X)
    for g in range(X.shape[1]):
        col = X[:, g].copy()
        rng.shuffle(col)
        X_sim[:, g] = col
    real = _make_adata(X)
    sim = _make_adata(X_sim)
    return real, sim


# ---------------------------------------------------------------------------
# Fixture G: Spatial pattern preserved
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_g_spatial_preserved():
    """Smooth spatial gradient on a 1D line with added count noise."""
    rng = np.random.default_rng(SEED)
    n_spots, n_genes = 30, 10
    coords = np.linspace(0, 1, n_spots).reshape(-1, 1)
    # Smooth gradient for first 5 genes
    X_real = np.zeros((n_spots, n_genes), dtype=np.float32)
    for g in range(5):
        base = 5 + 10 * np.sin(np.pi * coords.flatten() + g * 0.5)
        X_real[:, g] = rng.poisson(np.clip(base, 0.5, None)).astype(np.float32)
    for g in range(5, n_genes):
        X_real[:, g] = rng.poisson(3, size=n_spots).astype(np.float32)

    X_sim = X_real.copy()
    # Add moderate independent noise
    noise = rng.poisson(1, size=(n_spots, n_genes)).astype(np.float32)
    X_sim = np.maximum(0, X_sim + noise - 1)

    real = _make_adata(X_real, with_spatial=False)
    sim = _make_adata(X_sim, with_spatial=False)
    real.obsm["spatial"] = np.column_stack([
        coords.flatten(), np.zeros(n_spots),
    ]).astype(np.float32)
    sim.obsm["spatial"] = np.column_stack([
        coords.flatten(), np.zeros(n_spots),
    ]).astype(np.float32)
    return real, sim


# ---------------------------------------------------------------------------
# Fixture H: Spatial pattern destroyed
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_h_spatial_destroyed():
    """Same counts as Fixture G but spatial coordinates permuted."""
    rng = np.random.default_rng(SEED)
    n_spots, n_genes = 30, 10
    coords = np.linspace(0, 1, n_spots).reshape(-1, 1)
    X_real = np.zeros((n_spots, n_genes), dtype=np.float32)
    for g in range(5):
        base = 5 + 10 * np.sin(np.pi * coords.flatten() + g * 0.5)
        X_real[:, g] = rng.poisson(np.clip(base, 0.5, None)).astype(np.float32)
    for g in range(5, n_genes):
        X_real[:, g] = rng.poisson(3, size=n_spots).astype(np.float32)

    X_sim = X_real.copy()
    noise = rng.poisson(1, size=(n_spots, n_genes)).astype(np.float32)
    X_sim = np.maximum(0, X_sim + noise - 1)

    real = _make_adata(X_real, with_spatial=False)
    sim = _make_adata(X_sim, with_spatial=False)
    real.obsm["spatial"] = np.column_stack([
        coords.flatten(), np.zeros(n_spots),
    ]).astype(np.float32)
    # Permute spatial coordinates for sim
    perm_coords = coords.copy()
    rng.shuffle(perm_coords)
    sim.obsm["spatial"] = np.column_stack([
        perm_coords.flatten(), np.zeros(n_spots),
    ]).astype(np.float32)
    return real, sim
