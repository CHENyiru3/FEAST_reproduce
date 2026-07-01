"""Block-based OT transport validation against dense OT ground truth.

Validates that partitioned spatial optimal transport produces results consistent
with full-dense transport, tests accuracy-vs-speed across block sizes, verifies
the dense-fallback threshold, and checks convergence with increasing overlap.

The dense transport is copied exactly from simulator.py lines 245-289.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "FEAST" / "src"))

from FEAST.de_novo._ot_transport import sinkhorn_transport
from FEAST.de_novo.quantile_field import midpoint_rank_normalize

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

SEED = 20260620


def generate_synthetic_spatial_data(n_spots=200, n_genes=50, seed=SEED):
    """Generate a small synthetic spatial transcriptomics dataset.

    Source and target are 2D grids with small independent offsets so the
    transport is nontrivial.  Gene expression is built from overlapping
    Gaussian spatial patterns evaluated at the *source* coordinates (target
    does not carry its own expression -- it receives transported values).

    Returns
    -------
    source_coords : ndarray (n_spots, 2)
    target_coords : ndarray (n_spots, 2)
    reference_matrix : ndarray (n_spots, n_genes)
    """
    rng = np.random.default_rng(int(seed))

    side = int(np.ceil(np.sqrt(n_spots)))
    x = np.linspace(0, 1, side)
    y = np.linspace(0, 1, side)
    xx, yy = np.meshgrid(x, y)

    base_x = xx.ravel()[:n_spots]
    base_y = yy.ravel()[:n_spots]

    source_coords = np.column_stack([
        base_x + rng.normal(0, 0.008, n_spots),
        base_y + rng.normal(0, 0.008, n_spots),
    ])

    target_coords = np.column_stack([
        base_x + rng.normal(0, 0.008, n_spots) + 0.02,
        base_y + rng.normal(0, 0.008, n_spots) - 0.01,
    ])

    reference_matrix = np.zeros((n_spots, n_genes), dtype=np.float64)
    for g in range(n_genes):
        cx = rng.uniform(0.2, 0.8)
        cy = rng.uniform(0.2, 0.8)
        scale = 0.12 + rng.uniform(0, 0.25)
        dist_sq = (
            (source_coords[:, 0] - cx) ** 2
            + (source_coords[:, 1] - cy) ** 2
        )
        expr = np.exp(-dist_sq / (2.0 * scale * scale))
        expr += rng.normal(0, 0.04, n_spots)
        expr = np.clip(expr, 0.0, None)
        reference_matrix[:, g] = expr * rng.uniform(4, 18)

    return (
        source_coords.astype(np.float64),
        target_coords.astype(np.float64),
        reference_matrix,
    )


# ---------------------------------------------------------------------------
# Dense OT transport (ground truth -- replica of simulator.py L245-289)
# ---------------------------------------------------------------------------

def dense_ot_transport(reference_matrix, source_coords, target_coords, reg=0.05):
    """Exact replica of the OT transport block in simulator.py."""
    n_source = source_coords.shape[0]
    n_target = target_coords.shape[0]

    cost = cdist(source_coords, target_coords, metric="euclidean")
    a = np.ones(n_source) / n_source
    b = np.ones(n_target) / n_target

    plan = sinkhorn_transport(M=cost, a=a, b=b, reg=reg)

    ref_quantiles = midpoint_rank_normalize(
        reference_matrix,
        tie_policy="stable_ordinal",
        clip_eps=1e-6,
    )

    col_mass = plan.sum(axis=0, keepdims=True)
    safe_mass = np.where(col_mass > 1e-12, col_mass, 1.0)
    transported = (plan / safe_mass).T @ ref_quantiles

    transported = midpoint_rank_normalize(
        transported,
        tie_policy="stable_ordinal",
        clip_eps=1e-6,
    )

    return transported


# ---------------------------------------------------------------------------
# Block-based OT transport
# ---------------------------------------------------------------------------

def block_ot_transport(
    reference_matrix,
    source_coords,
    target_coords,
    reg=0.05,
    block_size=50,
    overlap_frac=0.3,
    min_spots_per_tile=5,
):
    """Partition spatial domain into tiles, run OT locally, and assemble.

    Parameters
    ----------
    reference_matrix : ndarray (n_source, n_genes)
    source_coords : ndarray (n_source, 2)
    target_coords : ndarray (n_target, 2)
    reg : float
        Sinkhorn regularisation (same as simulator default).
    block_size : int
        Approximate target spots per tile (guides the tiling grid density).
    overlap_frac : float in [0, 1]
        Fraction of tile width to extend on each side for continuity.
    min_spots_per_tile : int
        Skip tiles that would have fewer source or target spots.

    Returns
    -------
    transported : ndarray (n_target, n_genes)
        Rank-normalised transported quantiles, same shape as the dense output.
    """
    n_source = source_coords.shape[0]
    n_target = target_coords.shape[0]
    n_genes = reference_matrix.shape[1]

    ref_quantiles = midpoint_rank_normalize(
        reference_matrix,
        tie_policy="stable_ordinal",
        clip_eps=1e-6,
    )

    tgt_x, tgt_y = target_coords[:, 0], target_coords[:, 1]
    src_x, src_y = source_coords[:, 0], source_coords[:, 1]

    x_min, x_max = tgt_x.min(), tgt_x.max()
    y_min, y_max = tgt_y.min(), tgt_y.max()
    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0

    n_tiles_per_dim = max(1, int(np.ceil(np.sqrt(n_target / max(1, block_size)))))
    x_edges = np.linspace(x_min, x_max, n_tiles_per_dim + 1)
    y_edges = np.linspace(y_min, y_max, n_tiles_per_dim + 1)

    x_overlap = overlap_frac * (x_range / n_tiles_per_dim)
    y_overlap = overlap_frac * (y_range / n_tiles_per_dim)

    transported_accum = np.zeros((n_target, n_genes), dtype=np.float64)
    count_accum = np.zeros(n_target, dtype=np.float64)

    for ix in range(n_tiles_per_dim):
        for iy in range(n_tiles_per_dim):
            xl = max(x_min, x_edges[ix] - x_overlap)
            xr = min(x_max, x_edges[ix + 1] + x_overlap)
            yl = max(y_min, y_edges[iy] - y_overlap)
            yr = min(y_max, y_edges[iy + 1] + y_overlap)

            tgt_mask = (tgt_x >= xl) & (tgt_x <= xr) & (tgt_y >= yl) & (tgt_y <= yr)
            tgt_idx = np.where(tgt_mask)[0]
            if tgt_idx.size < min_spots_per_tile:
                continue

            src_mask = (src_x >= xl) & (src_x <= xr) & (src_y >= yl) & (src_y <= yr)
            src_idx = np.where(src_mask)[0]
            if src_idx.size < min_spots_per_tile:
                continue

            cost = cdist(
                source_coords[src_idx], target_coords[tgt_idx], metric="euclidean"
            )
            a = np.ones(src_idx.size) / src_idx.size
            b = np.ones(tgt_idx.size) / tgt_idx.size

            plan = sinkhorn_transport(M=cost, a=a, b=b, reg=reg)

            col_mass = plan.sum(axis=0, keepdims=True)
            safe_mass = np.where(col_mass > 1e-12, col_mass, 1.0)
            transported_local = (plan / safe_mass).T @ ref_quantiles[src_idx, :]

            transported_accum[tgt_idx, :] += transported_local
            count_accum[tgt_idx] += 1

    uncovered = count_accum == 0
    if np.any(uncovered):
        uncovered_idx = np.where(uncovered)[0]
        dist_all = cdist(target_coords[uncovered_idx], source_coords, metric="euclidean")
        nearest_src = np.argmin(dist_all, axis=1)
        transported_accum[uncovered_idx, :] = ref_quantiles[nearest_src, :]
        count_accum[uncovered_idx] = 1.0

    transported = transported_accum / count_accum[:, None]

    transported = midpoint_rank_normalize(
        transported,
        tie_policy="stable_ordinal",
        clip_eps=1e-6,
    )

    return transported


def smart_ot_transport(
    reference_matrix,
    source_coords,
    target_coords,
    reg=0.05,
    block_size=50,
    overlap_frac=0.3,
    fallback_threshold=50,
):
    """Dense OT when below threshold, block OT otherwise."""
    n_target = target_coords.shape[0]
    if n_target < fallback_threshold:
        return dense_ot_transport(
            reference_matrix, source_coords, target_coords, reg=reg
        )
    return block_ot_transport(
        reference_matrix,
        source_coords,
        target_coords,
        reg=reg,
        block_size=block_size,
        overlap_frac=overlap_frac,
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def pearson_per_gene(a, b):
    """Pearson correlation per gene (column)."""
    n_genes = a.shape[1]
    corrs = np.zeros(n_genes, dtype=np.float64)
    for g in range(n_genes):
        with np.errstate(invalid="ignore"):
            c = np.corrcoef(a[:, g], b[:, g])[0, 1]
        corrs[g] = c if np.isfinite(c) else 0.0
    return corrs


def mean_abs_diff(a, b):
    """Mean absolute element-wise difference."""
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float64) -
                               np.asarray(b, dtype=np.float64))))


def neighbor_smoothness(coords, values):
    """Mean absolute neighbour difference -- proxy for spatial smoothness."""
    dist = cdist(coords, coords)
    np.fill_diagonal(dist, np.inf)
    nn_idx = np.argmin(dist, axis=1)
    diffs = np.mean(np.abs(values - values[nn_idx, :]), axis=0)
    return float(np.mean(diffs))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerators:
    """Sanity-checks on the synthetic data."""

    def test_generated_shapes(self):
        src, tgt, ref = generate_synthetic_spatial_data(200, 50)
        assert src.shape == (200, 2)
        assert tgt.shape == (200, 2)
        assert ref.shape == (200, 50)

    def test_spatial_extent(self):
        src, tgt, ref = generate_synthetic_spatial_data(200, 50)
        assert -0.1 <= src.min() <= src.max() <= 1.3
        assert -0.1 <= tgt.min() <= tgt.max() <= 1.3


class TestDenseOT:
    """Dense transport produces well-formed output."""

    @pytest.fixture(scope="class")
    def data(self):
        return generate_synthetic_spatial_data(200, 50)

    def test_output_shape_and_range(self, data):
        src, tgt, ref = data
        result = dense_ot_transport(ref, src, tgt)
        assert result.shape == (200, 50)
        assert np.all(np.isfinite(result))
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_deterministic(self, data):
        src, tgt, ref = data
        r1 = dense_ot_transport(ref, src, tgt)
        r2 = dense_ot_transport(ref, src, tgt)
        assert np.allclose(r1, r2, atol=1e-6)


class TestBlockOTvsDense:
    """Core accuracy comparisons between block and dense OT."""

    @pytest.fixture(scope="class")
    def data(self):
        src, tgt, ref = generate_synthetic_spatial_data(200, 50)
        dense = dense_ot_transport(ref, src, tgt)
        return src, tgt, ref, dense

    def test_block_size_approx_50(self, data):
        src, tgt, ref, dense = data
        block = block_ot_transport(ref, src, tgt, block_size=50, overlap_frac=0.3)
        corrs = pearson_per_gene(dense, block)
        mae = mean_abs_diff(dense, block)
        assert np.median(corrs) > 0.65, f"median corr={np.median(corrs):.3f}"
        assert mae < 0.35, f"MAE={mae:.3f}"

    def test_block_size_approx_100(self, data):
        src, tgt, ref, dense = data
        block = block_ot_transport(ref, src, tgt, block_size=100, overlap_frac=0.3)
        corrs = pearson_per_gene(dense, block)
        mae = mean_abs_diff(dense, block)
        assert np.median(corrs) > 0.75, f"median corr={np.median(corrs):.3f}"
        assert mae < 0.25, f"MAE={mae:.3f}"

    def test_block_size_equals_or_exceeds_total_spots_matches_dense(self, data):
        src, tgt, ref, dense = data
        block = block_ot_transport(ref, src, tgt, block_size=500, overlap_frac=0.0)
        corrs = pearson_per_gene(dense, block)
        assert np.median(corrs) > 0.99, f"median corr={np.median(corrs):.3f}"
        # MAE looser because per-tile marginal weights differ from global weights
        assert mean_abs_diff(dense, block) < 0.04

    def test_large_overlap_converges_to_dense(self, data):
        src, tgt, ref, dense = data
        block = block_ot_transport(ref, src, tgt, block_size=50, overlap_frac=0.9)
        corrs = pearson_per_gene(dense, block)
        mae = mean_abs_diff(dense, block)
        assert np.median(corrs) > 0.85, f"median corr={np.median(corrs):.3f}"
        assert mae < 0.20, f"MAE={mae:.3f}"

    def test_zero_overlap_degradation(self, data):
        src, tgt, ref, dense = data
        block = block_ot_transport(ref, src, tgt, block_size=50, overlap_frac=0.0)
        corrs = pearson_per_gene(dense, block)
        mae = mean_abs_diff(dense, block)
        assert np.median(corrs) > 0.40, f"median corr={np.median(corrs):.3f}"
        assert mae < 0.45, f"MAE={mae:.3f}"

    def test_spatial_smoothness_preserved(self, data):
        src, tgt, ref, dense = data
        block = block_ot_transport(ref, src, tgt, block_size=50, overlap_frac=0.3)
        dense_smooth = neighbor_smoothness(tgt, dense)
        block_smooth = neighbor_smoothness(tgt, block)
        ratio = block_smooth / max(dense_smooth, 1e-8)
        assert 0.5 < ratio < 2.0, (
            f"dense smoothness={dense_smooth:.4f}, "
            f"block smoothness={block_smooth:.4f}"
        )


class TestFallbackLogic:
    """When the dataset is small, smart_ot_transport falls back to dense OT."""

    def test_small_falls_back_to_dense(self):
        src, tgt, ref = generate_synthetic_spatial_data(n_spots=30, n_genes=20, seed=77)
        result = smart_ot_transport(
            ref, src, tgt, block_size=50, fallback_threshold=50
        )
        expected = dense_ot_transport(ref, src, tgt)
        assert np.allclose(result, expected, atol=1e-6)

    def test_above_threshold_uses_block(self):
        src, tgt, ref = generate_synthetic_spatial_data(n_spots=100, n_genes=20, seed=88)
        result = smart_ot_transport(
            ref, src, tgt, block_size=50, fallback_threshold=50, overlap_frac=0.0
        )
        expected = dense_ot_transport(ref, src, tgt)
        assert not np.allclose(result, expected, atol=1e-12)
        corrs = pearson_per_gene(result, expected)
        assert np.median(corrs) > 0.6

    def test_at_threshold_uses_block(self):
        """At the boundary (n == threshold) the block path is taken."""
        n = 50
        src, tgt, ref = generate_synthetic_spatial_data(n_spots=n, n_genes=20, seed=99)
        result = smart_ot_transport(
            ref, src, tgt, block_size=30, fallback_threshold=50, overlap_frac=0.0
        )
        expected = dense_ot_transport(ref, src, tgt)
        assert not np.allclose(result, expected, atol=1e-12)


class TestEdgeCases:
    """Corner-case robustness."""

    def test_single_spot(self):
        n = 1
        ref = np.array([[10.0, 20.0, 5.0]])
        src = np.array([[0.5, 0.5]])
        tgt = np.array([[0.51, 0.49]])
        result = block_ot_transport(ref, src, tgt, block_size=50)
        assert result.shape == (1, 3)
        assert np.all(np.isfinite(result))

    def test_two_spots_symmetric(self):
        n = 2
        ref = np.array([[1.0, 100.0], [100.0, 1.0]])
        src = np.array([[0.0, 0.0], [1.0, 1.0]])
        tgt = np.array([[0.1, 0.1], [0.9, 0.9]])
        dense = dense_ot_transport(ref, src, tgt)
        block = block_ot_transport(ref, src, tgt, block_size=50)
        corrs = pearson_per_gene(dense, block)
        assert np.median(corrs) > 0.95

    def test_identical_source_and_target(self):
        """Identical coordinates -> identity-like plan."""
        src, tgt, ref = generate_synthetic_spatial_data(100, 30, seed=42)
        tgt = src.copy()
        dense = dense_ot_transport(ref, src, tgt)
        block = block_ot_transport(ref, src, tgt, block_size=50, overlap_frac=0.3)
        corrs = pearson_per_gene(dense, block)
        assert np.median(corrs) > 0.70

    def test_all_zero_expression_produces_finite_output(self):
        src, tgt, _ = generate_synthetic_spatial_data(100, 10, seed=11)
        ref = np.zeros((100, 10), dtype=np.float64)
        dense = dense_ot_transport(ref, src, tgt)
        block = block_ot_transport(ref, src, tgt, block_size=50)
        assert np.all(np.isfinite(dense))
        assert np.all(np.isfinite(block))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
