"""Integration tests for block OT transport on actual benchmark data.

Validates that _block_ot_transport (from FEAST.FEAST_core.simulator) produces
results consistent with dense optimal transport on real spatial transcriptomics
datasets from the simulator benchmark.

Tests:
  1. DLPFC_151670 (~3,480 spots) -- small-scale self-transport and split-transport
  2. MERFISH_006 (~10k spots)   -- medium-scale self-transport

For each dataset, the dense OT transport (ground truth) is computed via an exact
replica of the transport block in simulator.py (lines 245-289) and compared
against _block_ot_transport on the same source/target.

NOTE: This test depends on _block_ot_transport being implemented in simulator.py.
If the import fails, all tests in this module are skipped.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist

sys.path.insert(0, "/maiziezhou_lab2/yiru/FEAST/src")

# ---------------------------------------------------------------------------
# Import function under test (may not be available yet)
# ---------------------------------------------------------------------------

try:
    from FEAST.FEAST_core.simulator import _block_ot_transport  # noqa: F401

    BLOCK_OT_AVAILABLE = True
    IMPORT_ERROR_MSG = None
except ImportError as exc:
    BLOCK_OT_AVAILABLE = False
    IMPORT_ERROR_MSG = str(exc)

# ---------------------------------------------------------------------------
# Always-available imports (used for ground-truth dense OT)
# ---------------------------------------------------------------------------

from FEAST.de_novo._ot_transport import sinkhorn_transport
from FEAST.de_novo.quantile_field import midpoint_rank_normalize

# ---------------------------------------------------------------------------
# Module-level skip marker -- applied to every test in this file
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not BLOCK_OT_AVAILABLE,
    reason=(
        "_block_ot_transport not available in FEAST.FEAST_core.simulator"
        + (f" ({IMPORT_ERROR_MSG})" if IMPORT_ERROR_MSG else "")
    ),
)

# lazily imported -- only when tests are actually selected
anndata = pytest.importorskip("anndata", reason="anndata not installed")


# ===================================================================
# Dense OT transport (ground truth -- exact replica of sim.py L245-289)
# ===================================================================


def dense_ot_transport(reference_matrix, source_coords, target_coords, reg=0.05):
    """Exact replica of the OT transport block in simulator.py.

    Parameters
    ----------
    reference_matrix : ndarray (n_source, n_genes)
        Source expression matrix.
    source_coords : ndarray (n_source, 2)
        Spatial coordinates of source spots.
    target_coords : ndarray (n_target, 2)
        Spatial coordinates of target spots.
    reg : float
        Sinkhorn regularisation parameter.

    Returns
    -------
    transported : ndarray (n_target, n_genes)
        Rank-normalised transported quantiles in [0, 1].
    """
    n_source, n_target = source_coords.shape[0], target_coords.shape[0]
    cost = cdist(source_coords, target_coords, metric="euclidean")
    a = np.ones(n_source) / n_source
    b = np.ones(n_target) / n_target
    plan = sinkhorn_transport(M=cost, a=a, b=b, reg=reg)
    ref_quantiles = midpoint_rank_normalize(
        reference_matrix, tie_policy="stable_ordinal", clip_eps=1e-6
    )
    col_mass = plan.sum(axis=0, keepdims=True)
    safe_mass = np.where(col_mass > 1e-12, col_mass, 1.0)
    transported = (plan / safe_mass).T @ ref_quantiles
    transported = midpoint_rank_normalize(
        transported, tie_policy="stable_ordinal", clip_eps=1e-6
    )
    return transported


# ===================================================================
# Evaluation helpers
# ===================================================================


def _pearson_per_gene(a, b):
    """Pearson correlation per gene (column).

    Returns
    -------
    corrs : ndarray of shape (n_genes,)  -- one correlation per gene.
    """
    n_genes = a.shape[1]
    corrs = np.zeros(n_genes, dtype=np.float64)
    for g in range(n_genes):
        col_a = a[:, g]
        col_b = b[:, g]
        # Skip constant columns (zero variance)
        if np.std(col_a) < 1e-15 or np.std(col_b) < 1e-15:
            corrs[g] = 0.0
            continue
        with np.errstate(invalid="ignore"):
            c = np.corrcoef(col_a, col_b)[0, 1]
        corrs[g] = c if np.isfinite(c) else 0.0
    return corrs


def _mean_abs_diff(a, b):
    """Mean absolute element-wise difference."""
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float64)
                               - np.asarray(b, dtype=np.float64))))


def _validate_output(result, expected_shape, name="output"):
    """Assert output contract: shape, finite, range [0, 1]."""
    assert result.shape == expected_shape, (
        f"{name} shape {result.shape} != expected {expected_shape}"
    )
    assert np.all(np.isfinite(result)), (
        f"{name} contains non-finite values"
    )
    assert float(result.min()) >= -1e-10, (
        f"{name} min={result.min():.6f} violates lower bound 0"
    )
    assert float(result.max()) <= 1.0 + 1e-10, (
        f"{name} max={result.max():.6f} violates upper bound 1"
    )


# ===================================================================
# Data paths & loading helpers
# ===================================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "exper_data"

# Number of genes to keep for performance (subsets top-expressed genes)
MAX_GENES_DLPFC = 300
MAX_GENES_MERFISH = 200

# Block sizes chosen to produce ~3 tiles per dimension (9 tiles total),
# ensuring the tiling and overlap assembly paths are exercised.
#   DLPFC  (3,480 spots): sqrt(3480/500)  ~ 2.6 -> 3 tiles/dim -> 9 tiles
#   MERFISH (~10k spots): sqrt(10000/2000) ~ 2.2 -> 3 tiles/dim -> 9 tiles
BLOCK_SIZE_DLPFC = 500
BLOCK_SIZE_MERFISH = 2000


def _load_and_extract(h5ad_path, max_genes):
    """Load an .h5ad and return (dense_matrix, spatial_coords).

    Subsets to *max_genes* genes with highest mean expression for speed.
    Skips the test if the file is missing.
    """
    if not h5ad_path.exists():
        pytest.skip(f"Data file not found: {h5ad_path}")

    adata = anndata.read_h5ad(h5ad_path)

    # Dense matrix
    if hasattr(adata.X, "toarray"):
        X = adata.X.toarray()
    else:
        X = np.asarray(adata.X)
    X = X.astype(np.float64, copy=False)

    # Spatial coordinates
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)

    # Subset to top-expressed genes
    if X.shape[1] > max_genes:
        gene_means = np.mean(X, axis=0)
        top_idx = np.argsort(gene_means)[-max_genes:]
        X = X[:, top_idx]

    return X, coords


def _count_tiles_used(n_spots, block_size):
    """Compute how many tiles the block OT logic would create."""
    n_per_dim = max(1, int(np.ceil(np.sqrt(n_spots / max(1, block_size)))))
    return n_per_dim * n_per_dim


# ===================================================================
# Module-level fixtures
# ===================================================================


@pytest.fixture(scope="module")
def dlpfc_data():
    """Return (matrix, coords) for DLPFC_151670 (~3,480 spots)."""
    return _load_and_extract(DATA_DIR / "DLPFC_151670.h5ad", MAX_GENES_DLPFC)


@pytest.fixture(scope="module")
def merfish_data():
    """Return (matrix, coords) for MERFISH_006 (~10k spots)."""
    return _load_and_extract(DATA_DIR / "MERFISH_006.h5ad", MAX_GENES_MERFISH)


# ===================================================================
# DLPFC_151670  (~3,480 spots)
# ===================================================================


class TestDLPFC151670:
    """Integration tests on DLPFC_151670 human DLPFC data (small-scale).

    Uses block_size=500 to force ~9 tiles (3 per dimension), exercising the
    full tiling, local OT, overlap assembly, and uncovered-spot fallback paths.
    """

    # -- block OT kwargs ---------------------------------------------------
    _BOT_KW = dict(reg=0.05, block_size=BLOCK_SIZE_DLPFC, overlap_frac=0.25)

    def test_tile_count_expected(self, dlpfc_data):
        """Sanity-check: the chosen block_size actually creates multiple tiles."""
        _, coords = dlpfc_data
        n_tiles = _count_tiles_used(coords.shape[0], BLOCK_SIZE_DLPFC)
        assert n_tiles > 1, (
            f"block_size={BLOCK_SIZE_DLPFC} produces only {n_tiles} tile(s); "
            f"increase block_size fragmentation"
        )

    def test_self_transport_output_shape(self, dlpfc_data):
        """Block OT output shape matches dense OT and the reference dimensions."""
        ref, coords = dlpfc_data
        n_spots, n_genes = ref.shape

        dense = dense_ot_transport(ref, coords, coords)
        _validate_output(dense, (n_spots, n_genes), "dense_ot")

        block = _block_ot_transport(ref, coords, coords, **self._BOT_KW)
        _validate_output(block, (n_spots, n_genes), "block_ot")

    def test_self_transport_correlation(self, dlpfc_data):
        """Per-gene Pearson correlation median > 0.90 vs dense OT."""
        ref, coords = dlpfc_data
        dense = dense_ot_transport(ref, coords, coords)
        block = _block_ot_transport(ref, coords, coords, **self._BOT_KW)

        corrs = _pearson_per_gene(dense, block)
        median_corr = np.median(corrs)

        assert median_corr > 0.90, (
            f"Median per-gene Pearson r = {median_corr:.4f} (threshold: 0.90)\n"
            f"  percentiles: [p10={np.percentile(corrs, 10):.4f}, "
            f"p25={np.percentile(corrs, 25):.4f}, "
            f"p75={np.percentile(corrs, 75):.4f}, "
            f"p90={np.percentile(corrs, 90):.4f}]"
        )

    def test_self_transport_mean_abs_diff(self, dlpfc_data):
        """Mean absolute difference < 0.05 between block and dense OT."""
        ref, coords = dlpfc_data
        dense = dense_ot_transport(ref, coords, coords)
        block = _block_ot_transport(ref, coords, coords, **self._BOT_KW)

        mae = _mean_abs_diff(dense, block)
        assert mae < 0.05, f"MAE = {mae:.5f} (threshold: 0.05)"

    def test_self_transport_values_in_range(self, dlpfc_data):
        """Block OT output values lie within [0, 1]."""
        ref, coords = dlpfc_data
        block = _block_ot_transport(ref, coords, coords, **self._BOT_KW)

        assert np.all(block >= -1e-10), f"min value = {block.min():.8f}"
        assert np.all(block <= 1.0 + 1e-10), f"max value = {block.max():.8f}"

    def test_no_overlap_still_correlated(self, dlpfc_data):
        """Even with zero overlap, block OT maintains per-gene correlation > 0.70."""
        ref, coords = dlpfc_data
        dense = dense_ot_transport(ref, coords, coords)
        block = _block_ot_transport(
            ref, coords, coords, reg=0.05,
            block_size=BLOCK_SIZE_DLPFC, overlap_frac=0.0,
        )
        corrs = _pearson_per_gene(dense, block)
        assert np.median(corrs) > 0.70, (
            f"Median per-gene Pearson r = {np.median(corrs):.4f} "
            f"(overlap_frac=0.0, threshold: 0.70)"
        )

    def test_high_overlap_converges_to_dense(self, dlpfc_data):
        """With high overlap (0.5), block OT approaches dense OT quality."""
        ref, coords = dlpfc_data
        dense = dense_ot_transport(ref, coords, coords)
        block = _block_ot_transport(
            ref, coords, coords, reg=0.05,
            block_size=BLOCK_SIZE_DLPFC, overlap_frac=0.5,
        )
        corrs = _pearson_per_gene(dense, block)
        mae = _mean_abs_diff(dense, block)
        assert np.median(corrs) > 0.95, (
            f"Median Pearson r = {np.median(corrs):.4f} (overlap=0.5)"
        )
        assert mae < 0.03, f"MAE = {mae:.5f} (overlap=0.5)"

    def test_split_transport_shape(self, dlpfc_data):
        """Split half of spots as source, half as target: output shape correct."""
        ref, coords = dlpfc_data
        n = ref.shape[0]
        mid = n // 2

        src_coords = coords[:mid]
        tgt_coords = coords[mid:]
        src_matrix = ref[:mid]

        n_target = tgt_coords.shape[0]
        n_genes = ref.shape[1]

        dense = dense_ot_transport(src_matrix, src_coords, tgt_coords)
        _validate_output(dense, (n_target, n_genes), "dense_ot_split")

        block = _block_ot_transport(
            src_matrix, src_coords, tgt_coords,
            reg=0.05, block_size=BLOCK_SIZE_DLPFC, overlap_frac=0.25,
        )
        _validate_output(block, (n_target, n_genes), "block_ot_split")

    def test_split_transport_correlation(self, dlpfc_data):
        """Split transport: per-gene correlation median > 0.75."""
        ref, coords = dlpfc_data
        n = ref.shape[0]
        mid = n // 2

        src_coords = coords[:mid]
        tgt_coords = coords[mid:]
        src_matrix = ref[:mid]

        dense = dense_ot_transport(src_matrix, src_coords, tgt_coords)
        block = _block_ot_transport(
            src_matrix, src_coords, tgt_coords,
            reg=0.05, block_size=BLOCK_SIZE_DLPFC, overlap_frac=0.25,
        )

        corrs = _pearson_per_gene(dense, block)
        median_corr = np.median(corrs)

        # Split-transport is harder -- relaxed threshold
        assert median_corr > 0.75, (
            f"Median per-gene Pearson r = {median_corr:.4f} (threshold: 0.75)\n"
            f"  percentiles: [p10={np.percentile(corrs, 10):.4f}, "
            f"p25={np.percentile(corrs, 25):.4f}, "
            f"p75={np.percentile(corrs, 75):.4f}]"
        )

    def test_deterministic(self, dlpfc_data):
        """Block OT is deterministic (same inputs -> same outputs)."""
        ref, coords = dlpfc_data
        r1 = _block_ot_transport(ref, coords, coords, **self._BOT_KW)
        r2 = _block_ot_transport(ref, coords, coords, **self._BOT_KW)
        assert np.allclose(r1, r2, atol=1e-10), "block OT is not deterministic"


# ===================================================================
# MERFISH_006  (~10k spots)
# ===================================================================


class TestMERFISH006:
    """Integration tests on MERFISH_006 mouse cortex data (medium-scale).

    Uses block_size=2000 to force ~9 tiles, testing tiling on ~10k spots.
    """

    _BOT_KW = dict(reg=0.05, block_size=BLOCK_SIZE_MERFISH, overlap_frac=0.25)

    def test_tile_count_expected(self, merfish_data):
        """Sanity-check: the chosen block_size creates multiple tiles."""
        _, coords = merfish_data
        n_tiles = _count_tiles_used(coords.shape[0], BLOCK_SIZE_MERFISH)
        assert n_tiles > 1, (
            f"block_size={BLOCK_SIZE_MERFISH} produces only {n_tiles} tile(s)"
        )

    def test_self_transport_shape(self, merfish_data):
        """Block OT output shape matches dense OT for 10k-spot data."""
        ref, coords = merfish_data
        n_spots, n_genes = ref.shape

        dense = dense_ot_transport(ref, coords, coords)
        assert dense.shape == (n_spots, n_genes)

        block = _block_ot_transport(ref, coords, coords, **self._BOT_KW)
        assert block.shape == (n_spots, n_genes)

    def test_self_transport_correlation(self, merfish_data):
        """Per-gene Pearson correlation median > 0.90 on 10k spots."""
        ref, coords = merfish_data
        dense = dense_ot_transport(ref, coords, coords)
        block = _block_ot_transport(ref, coords, coords, **self._BOT_KW)

        corrs = _pearson_per_gene(dense, block)
        median_corr = np.median(corrs)

        assert median_corr > 0.90, (
            f"Median per-gene Pearson r = {median_corr:.4f} (threshold: 0.90)\n"
            f"  percentiles: [p10={np.percentile(corrs, 10):.4f}, "
            f"p25={np.percentile(corrs, 25):.4f}, "
            f"p75={np.percentile(corrs, 75):.4f}, "
            f"p90={np.percentile(corrs, 90):.4f}]"
        )

    def test_self_transport_mean_abs_diff(self, merfish_data):
        """Mean absolute difference < 0.05 on 10k spots."""
        ref, coords = merfish_data
        dense = dense_ot_transport(ref, coords, coords)
        block = _block_ot_transport(ref, coords, coords, **self._BOT_KW)

        mae = _mean_abs_diff(dense, block)
        assert mae < 0.05, f"MAE = {mae:.5f} (threshold: 0.05)"

    def test_values_in_range(self, merfish_data):
        """Block OT output values are in [0, 1] range."""
        ref, coords = merfish_data
        block = _block_ot_transport(ref, coords, coords, **self._BOT_KW)

        assert np.all(block >= -1e-10), f"min value = {block.min():.8f}"
        assert np.all(block <= 1.0 + 1e-10), f"max value = {block.max():.8f}"

    def test_no_overlap_still_correlated(self, merfish_data):
        """Even with zero overlap, block OT maintains per-gene correlation > 0.70."""
        ref, coords = merfish_data
        dense = dense_ot_transport(ref, coords, coords)
        block = _block_ot_transport(
            ref, coords, coords, reg=0.05,
            block_size=BLOCK_SIZE_MERFISH, overlap_frac=0.0,
        )
        corrs = _pearson_per_gene(dense, block)
        assert np.median(corrs) > 0.70, (
            f"Median per-gene Pearson r = {np.median(corrs):.4f} "
            f"(overlap_frac=0.0, threshold: 0.70)"
        )

    def test_deterministic(self, merfish_data):
        """Block OT is deterministic on 10k data."""
        ref, coords = merfish_data
        r1 = _block_ot_transport(ref, coords, coords, **self._BOT_KW)
        r2 = _block_ot_transport(ref, coords, coords, **self._BOT_KW)
        assert np.allclose(r1, r2, atol=1e-10), "block OT is not deterministic"
