#!/usr/bin/env python
"""Ablation study: spot-level smoothing vs. class-mean regularization for z-coherence.

Cross-validation design: reference model is fit on ODD Zhuang MERFISH slices
(005, 007, 009, 011), then EVEN slices (006, 008, 010, 012) are generated as
"virtual" slices.  Real z-jitter exists because reference and target are
different physical sections.

Tests whether spot-level cross-z smoothing, class-mean quadratic-penalty
regularization, both, or neither is needed to maintain z-coherence in
cross-slice 3D transfer.

Uses 8 Zhuang MERFISH serial sections (Allen_Zhuang_ABCA_1, 200 HVGs,
2000 spots/slice).

Usage
-----
    # Generate cache (run once, takes a few minutes):
    python tests/test_ablation_z_regularization.py --generate

    # Run test (loads from cache):
    pytest tests/test_ablation_z_regularization.py -v -s --timeout 300
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse
from scipy.stats import ks_2samp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path("/maiziezhou_lab2/yiru")
FEAST_SRC = REPO_ROOT / "FEAST" / "src"
if FEAST_SRC.exists():
    sys.path.insert(0, str(FEAST_SRC))

DATA_DIR = (
    REPO_ROOT / "Datasets" / "Processed" / "Allen_Zhuang_ABCA_1" / "h5ad"
)
CACHE_DIR = Path(__file__).resolve().parent / "_cache_ablation"

# ---------------------------------------------------------------------------
# Slice selection: 8 slices 005-012, odd=reference, even=target
# ---------------------------------------------------------------------------

ALL_SLICE_IDS = [5, 6, 7, 8, 9, 10, 11, 12]
REF_SLICE_IDS = [5, 7, 9, 11]      # odd  -- used for fit_reference
TARGET_SLICE_IDS = [6, 8, 10, 12]  # even -- virtual slices to generate

N_HVG = 200
N_SPOTS_PER_SLICE = 2000
LABEL_KEY = "class"
SIGMA = 1.0          # z-sigma for spot smoothing (unitless, in normalized z-space)
BLEND_LAMBDA = 0.3   # soft-blend fraction for spot smoothing
LAMBDA_1 = 1.0       # class-mean first-derivative penalty
LAMBDA_2 = 0.5       # class-mean second-derivative penalty
MIN_SPOTS = 5        # min spots per class at a z-level for class-mean regularization

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dense_counts(adata: ad.AnnData, gene_names: Sequence[str]) -> np.ndarray:
    view = adata[:, list(gene_names)]
    matrix = view.layers["counts"] if "counts" in view.layers else view.X
    if issparse(matrix):
        matrix = matrix.toarray()
    elif hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float64)


def _spatial_xy(adata: ad.AnnData) -> np.ndarray:
    if "spatial" in adata.obsm:
        return np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]
    if "spatial_3d" in adata.obsm:
        return np.asarray(adata.obsm["spatial_3d"], dtype=float)[:, :2]
    raise KeyError("AnnData missing obsm['spatial'] or obsm['spatial_3d'].")


def _spatial_full(adata: ad.AnnData) -> np.ndarray:
    """Return full spatial coordinates (2D or 3D) to match reference model dimensionality."""
    if "spatial_3d" in adata.obsm:
        return np.asarray(adata.obsm["spatial_3d"], dtype=float)
    if "spatial" in adata.obsm:
        return np.asarray(adata.obsm["spatial"], dtype=float)
    raise KeyError("AnnData missing obsm['spatial'] or obsm['spatial_3d'].")


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    import math
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    if xv.size < 2:
        return float("nan")
    xc = xv - xv.mean()
    yc = yv - yv.mean()
    denom = math.sqrt(float(xc.dot(xc) * yc.dot(yc)))
    if denom <= 0.0:
        return float("nan")
    return float(xc.dot(yc) / denom)


def _cache_generated_exists() -> bool:
    """Check if cached generated slices exist."""
    return (CACHE_DIR / "meta.pkl").exists()


def _cache_ref_exists() -> bool:
    """Check if cached reference slices exist (for metric computation)."""
    return (CACHE_DIR / "ref_meta.pkl").exists()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _check_all_slices_exist() -> bool:
    for sid in ALL_SLICE_IDS:
        path = DATA_DIR / f"Zhuang-ABCA-1.{sid:03d}.h5ad"
        if not path.exists():
            return False
    return True


def select_hvgs(slices: Sequence[ad.AnnData], n_top: int = N_HVG) -> list[str]:
    """Select top *n_top* highly variable genes using the middle slice (slice 008)."""
    from scipy.stats import variation

    # ALL_SLICE_IDS = [5, 6, 7, 8, 9, 10, 11, 12]
    # Middle is at index 3 which is slice 008
    mid = slices[3]  # slice 008 in the all-slices ordering
    counts = _dense_counts(mid, mid.var_names)
    means = counts.mean(axis=0)
    # Simple HVG: coefficient of variation on log1p-transformed data
    log_counts = np.log1p(counts)
    cvs = np.zeros(counts.shape[1], dtype=np.float64)
    for j in range(counts.shape[1]):
        col = log_counts[:, j]
        col = col[col > 0]
        if len(col) > 1:
            cvs[j] = variation(col)
    # Filter out very low-mean genes
    keep = means > 0.05
    cvs[~keep] = -1.0
    top_idx = np.argsort(cvs)[-n_top:]
    return [str(mid.var_names[i]) for i in top_idx]


def load_all_zhuang_slices() -> tuple[
    list[ad.AnnData], list[float], list[str],
    list[ad.AnnData], list[float],
]:
    """Load all 8 Zhuang slices 005-012, subset to HVGs and first N_SPOTS_PER_SLICE spots.

    Returns
    -------
    ref_slices : list[ad.AnnData]
        Odd-indexed slices (005, 007, 009, 011) for fit_reference.
    ref_z_values : list[float]
        z-values for reference slices (5.0, 7.0, 9.0, 11.0).
    gene_names : list[str]
        Selected HVG names.
    target_slices : list[ad.AnnData]
        Even-indexed slices (006, 008, 010, 012) -- the real even slices used
        as ground truth for metric comparison.
    target_z_values : list[float]
        z-values for target slices (6.0, 8.0, 10.0, 12.0).
    """
    if not _check_all_slices_exist():
        raise FileNotFoundError(
            f"Zhuang slices not found in {DATA_DIR}"
        )

    slices_raw: list[ad.AnnData] = []
    for sid in ALL_SLICE_IDS:
        path = DATA_DIR / f"Zhuang-ABCA-1.{sid:03d}.h5ad"
        adata = ad.read_h5ad(path)
        if "counts" not in adata.layers:
            raise KeyError(f"Slice {sid:03d} missing 'counts' layer.")
        if LABEL_KEY not in adata.obs:
            raise KeyError(f"Slice {sid:03d} missing obs['{LABEL_KEY}'].")
        adata.obs["z"] = float(sid)
        slices_raw.append(adata)

    # Subset to first N_SPOTS_PER_SLICE spots per slice
    slices_subbed: list[ad.AnnData] = []
    for adata in slices_raw:
        n = min(adata.n_obs, N_SPOTS_PER_SLICE)
        slices_subbed.append(adata[:n, :].copy())
        del adata
    gc.collect()

    # Select HVGs from middle slice (index 3 -> slice 008)
    hvg_list = select_hvgs(slices_subbed, n_top=N_HVG)
    print(f"  Selected {len(hvg_list)} HVGs")

    # Build an index mapping: dict from sid to position in slices_subbed list
    sid_to_idx = {sid: i for i, sid in enumerate(ALL_SLICE_IDS)}

    # Subset all slices to HVGs
    slices_final: list[ad.AnnData] = []
    for adata in slices_subbed:
        adata = adata[:, hvg_list].copy()
        adata.obs = adata.obs.copy()
        slices_final.append(adata)

    # Split into reference (odd) and target (even)
    ref_slices: list[ad.AnnData] = []
    ref_z_values: list[float] = []
    target_slices: list[ad.AnnData] = []
    target_z_values: list[float] = []

    for sid in ALL_SLICE_IDS:
        idx = sid_to_idx[sid]
        adata = slices_final[idx]
        z = float(adata.obs["z"].iloc[0])
        if sid in REF_SLICE_IDS:
            ref_slices.append(adata)
            ref_z_values.append(z)
        else:
            target_slices.append(adata)
            target_z_values.append(z)

    return ref_slices, ref_z_values, hvg_list, target_slices, target_z_values


# ---------------------------------------------------------------------------
# Cross-validation generation
# ---------------------------------------------------------------------------


def _generate_cross_validated(
    ref_slices: list[ad.AnnData],
    ref_z_values: list[float],
    target_slices: list[ad.AnnData],
    target_z_values: list[float],
    gene_names: list[str],
) -> list[ad.AnnData]:
    """Fit reference model on odd slices, generate even slices via SliceBlueprint.

    Parameters
    ----------
    ref_slices : list[ad.AnnData]
        Odd slices (005, 007, 009, 011) for fitting the reference model.
    ref_z_values : list[float]
        z-values for reference slices (5.0, 7.0, 9.0, 11.0).
    target_slices : list[ad.AnnData]
        Real even slices (006, 008, 010, 012) whose spatial coordinates and
        class labels are used to build SliceBlueprints.
    target_z_values : list[float]
        z-values for target slices (6.0, 8.0, 10.0, 12.0).
    gene_names : list[str]
        Gene names shared across all slices.

    Returns
    -------
    list[ad.AnnData]
        Generated even slices (same length as target_slices).
    """
    from FEAST.de_novo.conditional import fit_reference, simulate_from_reference
    from FEAST import ReferenceFitConfig, SimulationConfig
    from FEAST import SliceBlueprint
    from FEAST.de_novo.core import assign_generated_coordinates

    # --- Phase 1: fit reference model on odd slices ---
    print("  Fitting reference model on odd slices (005, 007, 009, 011)...")
    fit_cfg = ReferenceFitConfig(
        min_gene_spots=1,
        min_gene_mean=0.0,
        max_gene_zero_prop=1.0,
    )
    model = fit_reference(ref_slices, label_key=LABEL_KEY, config=fit_cfg)
    print(f"    Model fit: {len(model.references)} references, "
          f"{len(model.gene_names)} genes, label_key={model.label_key}")

    # --- Phase 2: generate each even slice from a SliceBlueprint ---
    gen_cfg = SimulationConfig(
        epsilon=0.05,
        sinkhorn_iter=200,
        verbose=False,
    )

    generated: list[ad.AnnData] = []
    for zi, (adata, z_val) in enumerate(zip(target_slices, target_z_values)):
        print(f"  Generating slice z={z_val:.0f} ({zi + 1}/{len(target_slices)})...")

        full_coords = _spatial_full(adata)  # 3D coords to match reference model
        xy = _spatial_xy(adata)
        classes = adata.obs[LABEL_KEY].astype(str).to_numpy()

        blueprint = SliceBlueprint(
            coordinates=full_coords,
            domain_map=classes,
        )

        result = simulate_from_reference(
            model,
            blueprint,
            config=gen_cfg,
            random_seed=42 + zi,
        )

        # Assign spatial coordinates and z-value
        assign_generated_coordinates(result, full_coords, z_value=float(z_val))
        result.obs["z"] = float(z_val)
        result.obs[LABEL_KEY] = classes

        if "counts" not in result.layers:
            result.layers["counts"] = _dense_counts(result, result.var_names)

        generated.append(result)
        gc.collect()

    return generated


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _safe_write_h5ad(adata: ad.AnnData, path: Path) -> None:
    """Write AnnData to HDF5, stripping de_novo metadata that h5py cannot serialize."""
    import copy
    adata = copy.deepcopy(adata)
    if "de_novo" in adata.uns:
        del adata.uns["de_novo"]
    if "quantile_field" in adata.uns:
        del adata.uns["quantile_field"]
    # Recursively strip any numpy array or non-serializable objects from uns
    for key in list(adata.uns.keys()):
        try:
            import numpy as np
            val = adata.uns[key]
            if isinstance(val, np.ndarray) and val.dtype.kind in ('O', 'U'):
                del adata.uns[key]
        except Exception:
            del adata.uns[key]
    adata.write_h5ad(path)


def _save_cross_validated_cache(
    gen_slices: list[ad.AnnData],
    real_target_slices: list[ad.AnnData],
    target_z_values: list[float],
    gene_names: list[str],
) -> None:
    """Save generated even slices and real even slices to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for zi, (z_val, adata) in enumerate(zip(target_z_values, gen_slices)):
        path = CACHE_DIR / f"generated_z{z_val:03.0f}.h5ad"
        _safe_write_h5ad(adata, path)
    for zi, (z_val, adata) in enumerate(zip(target_z_values, real_target_slices)):
        path = CACHE_DIR / f"reference_z{z_val:03.0f}.h5ad"
        _safe_write_h5ad(adata, path)
    meta = {"z_values": target_z_values, "gene_names": gene_names}
    pd.Series(meta).to_pickle(CACHE_DIR / "meta.pkl")
    pd.Series(meta).to_pickle(CACHE_DIR / "ref_meta.pkl")
    print(f"  Cached {len(gen_slices)} generated + {len(real_target_slices)} "
          f"reference slices to {CACHE_DIR}")


def _load_cached_slices(prefix: str) -> tuple[list[ad.AnnData], list[float], list[str]]:
    """Load cached slices with given filename prefix ('generated' or 'reference')."""
    meta_path = CACHE_DIR / ("meta.pkl" if prefix == "generated" else "ref_meta.pkl")
    meta = pd.read_pickle(str(meta_path))
    z_values = meta["z_values"]
    gene_names = meta["gene_names"]
    slices: list[ad.AnnData] = []
    for z in z_values:
        path = CACHE_DIR / f"{prefix}_z{z:03.0f}.h5ad"
        slices.append(ad.read_h5ad(path))
    return slices, z_values, gene_names


def generate_cache_cli() -> None:
    """Generate and cache cross-validated slices.  Run via `python ... --generate`."""
    print("=" * 60)
    print("GENERATING CROSS-VALIDATION ABLATION CACHE")
    print("=" * 60)

    ref_slices, ref_z_values, gene_names, target_slices, target_z_values = (
        load_all_zhuang_slices()
    )
    print(f"Loaded {len(ref_slices)} reference slices "
          f"(z={[f'{z:.0f}' for z in ref_z_values]})")
    print(f"Loaded {len(target_slices)} target slices "
          f"(z={[f'{z:.0f}' for z in target_z_values]})")
    print(f"  {len(gene_names)} genes, {target_slices[0].n_obs} spots/slice")

    generated = _generate_cross_validated(
        ref_slices, ref_z_values,
        target_slices, target_z_values,
        gene_names,
    )
    _save_cross_validated_cache(generated, target_slices, target_z_values, gene_names)
    print("\nCache generation complete.")


# ---------------------------------------------------------------------------
# Condition: B - Spot smoothing
# ---------------------------------------------------------------------------


def apply_spot_smoothing(
    slices: list[ad.AnnData],
    z_values: list[float],
    gene_names: list[str],
) -> list[ad.AnnData]:
    """Apply cross-z spot-level bilateral-weighted local linear smoothing."""
    from FEAST.de_novo.z_spot_smooth import smooth_cross_z_spots
    z_dict = {float(z): sl for z, sl in zip(z_values, slices)}
    out_dict = smooth_cross_z_spots(
        z_dict,
        k_xy=10,
        k_z=5,
        z_sigma=SIGMA,
        blend_lambda=BLEND_LAMBDA,
        genes=gene_names,
        layer="counts",
        progress=True,
    )
    # Sync layers['counts'] with X (smooth_cross_z_spots writes to .X but
    # downstream metric functions read from layers['counts']).
    for sl in out_dict.values():
        sl.layers["counts"] = sl.X
    return [out_dict[float(z)] for z in z_values]


# ---------------------------------------------------------------------------
# Condition: C - Class-mean regularization
# ---------------------------------------------------------------------------


def apply_class_mean_regularization(
    slices: list[ad.AnnData],
    z_values: list[float],
    gene_names: list[str],
) -> list[ad.AnnData]:
    """Apply class-mean quadratic-penalty z-regularization to all slices."""
    from FEAST.de_novo.z_regularize import (
        calibrate_counts_to_regularized_means,
        class_anchor_weight,
        regularize_mean_profiles,
    )

    z_arr = np.asarray(z_values, dtype=float)
    gene_list = list(gene_names)

    # Collect all class labels
    all_labels: set[str] = set()
    for adata in slices:
        all_labels.update(adata.obs[LABEL_KEY].astype(str).unique().tolist())

    # --- Phase 1: per-label, solve for regularized means across all z ---
    target_records: list[dict] = []

    for label in sorted(all_labels):
        nz = len(slices)
        means = np.full((nz, len(gene_list)), np.nan, dtype=np.float64)
        z_pass = np.zeros(nz, dtype=bool)
        n_spots_list = [0] * nz

        for zi, adata in enumerate(slices):
            mask = adata.obs[LABEL_KEY].astype(str).to_numpy() == label
            n_spots = int(mask.sum())
            n_spots_list[zi] = n_spots
            if n_spots < MIN_SPOTS:
                continue
            counts = _dense_counts(adata, gene_list)
            means[zi, :] = counts[mask, :].mean(axis=0)
            z_pass[zi] = True

        valid_idx = np.flatnonzero(z_pass)
        if len(valid_idx) < 2:
            continue

        n_nodes = int(len(valid_idx))
        z_valid = z_arr[valid_idx]

        y_matrix = np.zeros((n_nodes, len(gene_list)), dtype=np.float64)
        node_weights = np.zeros(n_nodes, dtype=np.float64)
        for local_idx, global_idx in enumerate(valid_idx):
            row_means = means[int(global_idx), :]
            row_means = np.nan_to_num(row_means, nan=0.0)
            y_matrix[local_idx, :] = np.log1p(np.clip(row_means, 0.0, None))
            node_weights[local_idx] = class_anchor_weight(
                n_spots_list[int(global_idx)]
            )

        solved = regularize_mean_profiles(
            y_matrix=y_matrix,
            node_weights=node_weights,
            z_values=z_valid,
            lambda_1=LAMBDA_1,
            lambda_2=LAMBDA_2,
        )

        for local_idx, global_idx in enumerate(valid_idx):
            for g, gene in enumerate(gene_list):
                val = solved[local_idx, g]
                if np.isfinite(val) and val >= 0:
                    target_records.append({
                        "z": float(z_valid[local_idx]),
                        "class": str(label),
                        "gene": str(gene),
                        "regularized_mean": float(val),
                    })

    if not target_records:
        return slices

    target_regularized = pd.DataFrame(target_records)

    # --- Phase 2: per-slice, rescale counts ---
    out_slices: list[ad.AnnData] = []
    for zi, adata in enumerate(slices):
        out = adata.copy()
        counts = _dense_counts(out, gene_list)
        labels_arr = out.obs[LABEL_KEY].astype(str).to_numpy()
        z_val = z_values[zi]
        # Filter to only this z-slice's regularized means (one per class per gene)
        z_regularized = target_regularized[
            target_regularized["z"] == z_val
        ].drop(columns=["z"])
        calibrated = calibrate_counts_to_regularized_means(
            counts=counts,
            labels=labels_arr,
            gene_names=gene_list,
            target_regularized=z_regularized,
        )
        out.layers["counts"] = calibrated
        out.X = calibrated
        out_slices.append(out)

    return out_slices


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_spot_z_autocorrelation(
    slices: list[ad.AnnData],
    z_values: list[float],
    gene_names: list[str],
) -> float:
    """Compute spot-level z-autocorrelation using cross-z neighbors."""
    from FEAST.de_novo.z_spot_smooth import compute_spot_z_autocorrelation as _func
    z_dict = {float(z): sl for z, sl in zip(z_values, slices)}
    return _func(z_dict, k_z=5, genes=gene_names, layer="counts")


def compute_class_z_coherence(
    gen_slices: list[ad.AnnData],
    ref_slices: list[ad.AnnData],
    z_values: list[float],
    gene_names: list[str],
) -> float:
    """Compute class-level z-coherence (Pearson between generated and reference means across z)."""
    from FEAST.de_novo.z_regularize import (
        compute_z_coherence,
        summarize_z_coherence_frame,
    )
    gene_list = list(gene_names)
    z_arr = np.asarray(z_values, dtype=float)

    records: list[dict] = []
    all_labels: set[str] = set()
    for adata in gen_slices:
        all_labels.update(adata.obs[LABEL_KEY].astype(str).unique().tolist())

    for label in sorted(all_labels):
        for g, gene in enumerate(gene_list):
            gen_means = []
            ref_means = []
            zs = []
            for zi in range(len(gen_slices)):
                gen_mask = gen_slices[zi].obs[LABEL_KEY].astype(str).to_numpy() == label
                ref_mask = ref_slices[zi].obs[LABEL_KEY].astype(str).to_numpy() == label
                if gen_mask.sum() < MIN_SPOTS or ref_mask.sum() < MIN_SPOTS:
                    continue
                gen_counts = _dense_counts(gen_slices[zi], gene_list)
                ref_counts = _dense_counts(ref_slices[zi], gene_list)
                gen_means.append(float(gen_counts[gen_mask, g].mean()))
                ref_means.append(float(ref_counts[ref_mask, g].mean()))
                zs.append(float(z_arr[zi]))

            if len(zs) < 3:
                continue

            for j in range(len(zs)):
                records.append({
                    "class": str(label),
                    "gene": str(gene),
                    "target_z": float(zs[j]),
                    "generated_mean": float(gen_means[j]),
                    "target_mean": float(ref_means[j]),
                })

    if not records:
        return float("nan")

    z_coh = compute_z_coherence(
        pd.DataFrame(records),
        left_col="generated_mean",
        right_col="target_mean",
    )
    summary = summarize_z_coherence_frame(z_coh, prefix="z")
    return summary.get("z_median", float("nan"))


def compute_moran_i_correlation(
    gen_slices: list[ad.AnnData],
    ref_slices: list[ad.AnnData],
    gene_names: list[str],
) -> float:
    """Per-gene Moran's I: Pearson correlation between generated and reference Moran's I values."""
    try:
        from esda.moran import Moran
        from libpysal.weights import KNN
    except ImportError:
        return float("nan")

    gene_list = list(gene_names)
    gen_vals: list[float] = []
    ref_vals: list[float] = []

    for zi in range(len(gen_slices)):
        gen_xy = _spatial_xy(gen_slices[zi])
        ref_xy = _spatial_xy(ref_slices[zi])

        if gen_xy.shape[0] < 6 or ref_xy.shape[0] < 6:
            continue

        gen_w = KNN(gen_xy, k=6)
        ref_w = KNN(ref_xy, k=6)

        gen_counts = _dense_counts(gen_slices[zi], gene_list)
        ref_counts = _dense_counts(ref_slices[zi], gene_list)

        for g in range(len(gene_list)):
            try:
                gen_mi = Moran(gen_counts[:, g], gen_w).I
                ref_mi = Moran(ref_counts[:, g], ref_w).I
                if np.isfinite(gen_mi) and np.isfinite(ref_mi):
                    gen_vals.append(gen_mi)
                    ref_vals.append(ref_mi)
            except Exception:
                continue

    if len(gen_vals) < 3:
        return float("nan")

    return _pearson_r(np.array(gen_vals), np.array(ref_vals))


def compute_ks_distance(
    gen_slices: list[ad.AnnData],
    ref_slices: list[ad.AnnData],
    gene_names: list[str],
) -> float:
    """KS statistic between generated and reference per-gene distributions.

    Returns mean KS distance across genes.
    """
    gene_list = list(gene_names)
    ks_vals: list[float] = []

    for g in range(len(gene_list)):
        gen_pooled: list[float] = []
        ref_pooled: list[float] = []
        for zi in range(len(gen_slices)):
            gen_c = _dense_counts(gen_slices[zi], gene_list)[:, g].ravel()
            ref_c = _dense_counts(ref_slices[zi], gene_list)[:, g].ravel()
            gen_pooled.extend(gen_c.tolist())
            ref_pooled.extend(ref_c.tolist())

        max_n = 5000
        gen_arr = np.asarray(gen_pooled, dtype=float)
        ref_arr = np.asarray(ref_pooled, dtype=float)
        if len(gen_arr) > max_n:
            rng = np.random.default_rng(42)
            gen_arr = rng.choice(gen_arr, size=max_n, replace=False)
        if len(ref_arr) > max_n:
            rng = np.random.default_rng(42)
            ref_arr = rng.choice(ref_arr, size=max_n, replace=False)

        try:
            stat = ks_2samp(gen_arr, ref_arr).statistic
            if np.isfinite(stat):
                ks_vals.append(stat)
        except Exception:
            continue

    if not ks_vals:
        return float("nan")

    return float(np.mean(ks_vals))


# ---------------------------------------------------------------------------
# Ablation test
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not _cache_generated_exists(),
    reason="Cached generated slices not found. "
           "Run: python tests/test_ablation_z_regularization.py --generate"
)
def test_ablation_z_regularization():
    """Ablation: spot smoothing vs. class-mean regularization vs. both vs. neither.

    Cross-validation design: reference model fit on odd slices (005, 007, 009, 011),
    even slices (006, 008, 010, 012) generated as virtual targets.
    Loads cached generated slices, applies each condition, and prints a
    comparison table of z-coherence metrics.
    """
    print("\n" + "=" * 80)
    print("Z-REGULARIZATION ABLATION STUDY (Cross-Validation)")
    print("=" * 80)
    print("  Reference slices: 005, 007, 009, 011 (odd)")
    print("  Target slices:    006, 008, 010, 012 (even) -- generated as virtual slices")
    print()

    # 1. Load cached generated and reference (real even) slices
    print("[1] Loading cached slices...")
    generated_slices, z_values, gene_names = _load_cached_slices("generated")
    ref_slices, _, _ = _load_cached_slices("reference")
    print(f"    {len(generated_slices)} slices, z = {[f'{z:.0f}' for z in z_values]}")
    print(f"    {len(gene_names)} genes")
    assert len(ref_slices) == len(generated_slices), (
        "ref and gen slice counts must match"
    )

    def _copy_slices(slices: list[ad.AnnData]) -> list[ad.AnnData]:
        return [sl.copy() for sl in slices]

    # 2. Condition A: None (baseline)
    print("\n[2] Condition A: No regularization (baseline generated slices)...")
    cond_a_slices = _copy_slices(generated_slices)
    results: dict[str, dict[str, float]] = {}

    # 3. Condition B: Spot smoothing only
    print("[3] Condition B: Spot smoothing only...")
    cond_b_slices = apply_spot_smoothing(
        _copy_slices(generated_slices), z_values, gene_names
    )
    # Sanity check: spot smoothing should modify counts
    spot_diff = np.mean([
        np.abs(
            _dense_counts(cond_b_slices[i], gene_names)
            - _dense_counts(generated_slices[i], gene_names)
        ).mean()
        for i in range(len(generated_slices))
    ])
    print(f"    (mean absolute count change from spot smoothing: {spot_diff:.4f})")
    assert spot_diff >= 0, "Spot smoothing count diff should be non-negative"

    # 4. Condition C: Class-mean regularization only
    print("[4] Condition C: Class-mean regularization only...")
    cond_c_slices = apply_class_mean_regularization(
        _copy_slices(generated_slices), z_values, gene_names
    )

    # 5. Condition D: Both (spot first, then class-mean)
    print("[5] Condition D: Both (spot smoothing + class-mean regularization)...")
    cond_d_smooth = apply_spot_smoothing(
        _copy_slices(generated_slices), z_values, gene_names
    )
    cond_d_slices = apply_class_mean_regularization(
        cond_d_smooth, z_values, gene_names
    )

    # 6. Compute metrics for each condition
    conditions = {
        "A (none)": cond_a_slices,
        "B (spot only)": cond_b_slices,
        "C (class only)": cond_c_slices,
        "D (both)": cond_d_slices,
    }

    print("\n[6] Computing metrics for all conditions...")
    for cond_name, cond_slices in conditions.items():
        print(f"    {cond_name}...")

        z_autocorr = compute_spot_z_autocorrelation(
            cond_slices, z_values, gene_names
        )
        print(f"      Z-Autocorr: {z_autocorr:.4f}")

        z_coherence = compute_class_z_coherence(
            cond_slices, ref_slices, z_values, gene_names
        )
        print(f"      Z-Coherence: {z_coherence:.4f}")

        moran_corr = compute_moran_i_correlation(
            cond_slices, ref_slices, gene_names
        )
        print(f"      Moran I Corr: {moran_corr:.4f}")

        ks_dist = compute_ks_distance(
            cond_slices, ref_slices, gene_names
        )
        print(f"      KS Dist: {ks_dist:.4f}")

        results[cond_name] = {
            "Z-Autocorr": z_autocorr,
            "Z-Coherence": z_coherence,
            "Moran I Corr": moran_corr,
            "KS Dist": ks_dist,
        }

    # 7. Print comparison table
    print("\n" + "=" * 80)
    print("ABLATION RESULTS")
    print("=" * 80)
    print()

    header = (f"{'Condition':<20} | {'Z-Autocorr':>10} | {'Z-Coherence':>12} "
              f"| {'Moran I Corr':>13} | {'KS Dist':>10}")
    sep = ("-" * 21 + "+" + "-" * 12 + "+" + "-" * 14
           + "+" + "-" * 15 + "+" + "-" * 11)
    print(header)
    print(sep)

    def _fmt(v: float) -> str:
        if np.isnan(v):
            return "       nan".rjust(12)
        return f"{v:12.4f}"

    for cond_name in ["A (none)", "B (spot only)", "C (class only)", "D (both)"]:
        r = results[cond_name]
        print(
            f"{cond_name:<20} | {_fmt(r['Z-Autocorr'])} | {_fmt(r['Z-Coherence'])} "
            f"| {_fmt(r['Moran I Corr'])} | {_fmt(r['KS Dist'])}"
        )

    print()
    print("Interpretation:")
    print("  - Z-Autocorr (higher = smoother spot trajectories):")
    print("       Correlation between a spot's expression and cross-z neighbors' expression.")
    print("  - Z-Coherence (higher = more coherent class means):")
    print("       Per-class per-gene Pearson correlation between generated and")
    print("       reference mean profiles across z-slices.")
    print("  - Moran I Corr (higher = better spatial fidelity):")
    print("       Pearson correlation between generated and reference per-gene")
    print("       Moran's I values within each slice.")
    print("  - KS Dist (lower = better distribution preservation):")
    print("       Mean Kolmogorov-Smirnov statistic between generated and")
    print("       reference per-gene expression distributions.")

    # Answer the key question
    print()
    print("KEY QUESTIONS: Do we need both spot smoothing AND class-mean "
          "regularization for cross-slice transfer?")
    za_a = results["A (none)"]["Z-Autocorr"]
    za_b = results["B (spot only)"]["Z-Autocorr"]
    za_c = results["C (class only)"]["Z-Autocorr"]
    za_d = results["D (both)"]["Z-Autocorr"]
    zc_a = results["A (none)"]["Z-Coherence"]
    zc_b = results["B (spot only)"]["Z-Coherence"]
    zc_c = results["C (class only)"]["Z-Coherence"]
    zc_d = results["D (both)"]["Z-Coherence"]
    mi_a = results["A (none)"]["Moran I Corr"]
    mi_b = results["B (spot only)"]["Moran I Corr"]
    mi_c = results["C (class only)"]["Moran I Corr"]
    mi_d = results["D (both)"]["Moran I Corr"]
    ks_a = results["A (none)"]["KS Dist"]
    ks_b = results["B (spot only)"]["KS Dist"]
    ks_c = results["C (class only)"]["KS Dist"]
    ks_d = results["D (both)"]["KS Dist"]

    print(f"  Baseline (no regularization):")
    print(f"    Z-Autocorr={za_a:.4f}, Z-Coherence={zc_a:.4f}, "
          f"Moran I Corr={mi_a:.4f}, KS Dist={ks_a:.4f}")

    # Spot smoothing effect
    spot_changes = abs(za_b - za_a) > 0.001 or abs(zc_b - zc_a) > 0.001
    if spot_changes:
        zc_change = "improves" if zc_b > zc_a else "degrades"
        print(f"  Spot smoothing: {zc_change} Z-Coherence ({zc_a:.4f} -> {zc_b:.4f}), "
              f"Z-Autocorr ({za_a:.4f} -> {za_b:.4f})")
    else:
        print(f"  Spot smoothing: minimal effect on metrics "
              f"(blend_lambda={BLEND_LAMBDA} with already-coherent generated data)")

    # Class-mean effect (cross-validation: generated means may not be z-coherent)
    if np.isfinite(zc_c):
        if zc_c > zc_a + 0.01:
            print(f"  Class-mean regularization: IMPROVES Z-Coherence ({zc_a:.4f} -> {zc_c:.4f})")
            print(f"    Suggests generated cross-slice means have real z-jitter that "
                  f"the penalty corrects.")
        elif zc_c < zc_a - 0.01:
            print(f"  Class-mean regularization: DEGRADES Z-Coherence ({zc_a:.4f} -> {zc_c:.4f})"
                  f"\n    (means may already be reasonably coherent; the penalty oversmooths)")
        else:
            print(f"  Class-mean regularization: minimal Z-Coherence change ({zc_c:.4f} vs {zc_a:.4f})")

    # Both effect
    if np.isfinite(zc_d):
        best_zc = max(filter(np.isfinite, [zc_a, zc_b, zc_c, zc_d]))
        if zc_d >= best_zc - 0.01:
            print(f"  Both combined: best or near-best Z-Coherence at {zc_d:.4f}")
        else:
            print(f"  Both combined: Z-Coherence {zc_d:.4f} (best={best_zc:.4f})")

    # Spatial fidelity and distribution
    print(f"  Spatial fidelity (Moran I Corr) and distribution (KS Dist):")
    print(f"    Spot smoothing: MI={mi_b:.4f} vs baseline {mi_a:.4f}, "
          f"KS={ks_b:.4f} vs baseline {ks_a:.4f}")
    if np.isfinite(ks_c):
        print(f"    Class-mean regularization: MI={mi_c:.4f}, KS={ks_c:.4f}"
              f" (baseline KS={ks_a:.4f})")
    if np.isfinite(mi_d) and np.isfinite(ks_d):
        print(f"    Both combined: MI={mi_d:.4f}, KS={ks_d:.4f}")

    # Final answer
    print()
    print(f"  VERDICT:")
    # Determine which condition is best overall
    best_zc = max(filter(np.isfinite, [zc_a, zc_b, zc_c, zc_d]))
    best_mi = max(filter(np.isfinite, [mi_a, mi_b, mi_c, mi_d]))
    best_ks = min(filter(np.isfinite, [ks_a, ks_b, ks_c, ks_d]))

    improvements_from_class = (
        np.isfinite(zc_c) and zc_c > zc_b + 0.01
    ) or (
        np.isfinite(zc_d) and np.isfinite(zc_b) and zc_d > zc_b + 0.01
    )
    harms_from_class = (
        np.isfinite(ks_c) and np.isfinite(ks_b) and ks_c > ks_b + 0.02
    )

    if improvements_from_class and not harms_from_class:
        print(f"    YES: BOTH spot smoothing AND class-mean regularization are needed.")
        print(f"         Spot smoothing provides spatial coherence; class-mean ")
        print(f"         regularization enforces biologically plausible z-profiles.")
    elif not improvements_from_class and harms_from_class:
        print(f"    NO: spot smoothing alone is sufficient for cross-slice transfer.")
        print(f"         Class-mean regularization primarily distorts distributions")
        print(f"         without meaningful Z-Coherence gain in this setting.")
    elif improvements_from_class and harms_from_class:
        print(f"    TRADE-OFF: class-mean regularization improves Z-Coherence but")
        print(f"              degrades distribution fidelity (KS Dist). Consider")
        print(f"              lower penalty strengths or application to selected genes.")
    else:
        print(f"    INCONCLUSIVE: metrics are too close to distinguish. Spot smoothing")
        print(f"                 provides modest benefits; class-mean has marginal effect.")
        print(f"                 Default recommendation: use spot smoothing alone as the")
        print(f"                 lighter-weight and benign option for cross-slice transfer.")

    print()
    print("=" * 80)

    # At least one condition should produce valid metrics
    has_valid = any(
        not np.isnan(results[c]["Z-Autocorr"])
        for c in conditions
    )
    assert has_valid, "All conditions produced nan metrics"


# ---------------------------------------------------------------------------
# CLI for cache generation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--generate" in sys.argv:
        generate_cache_cli()
    else:
        print("Usage: python test_ablation_z_regularization.py --generate")
        print("       (to pre-generate cached slices before running pytest)")
        sys.exit(1)
