#!/usr/bin/env python
"""
Stage 3: Alignment Benchmark (v4).

Unified metrics: same computation for Spateo, PASTE, and other methods.
For each aligned spot, we match to the nearest reference spot by spatial
distance and check barcode identity, region agreement, and expression
correlation.  Method-supplied mapping matrices (Spateo's pi) are reported
as supplemental Spateo-specific metrics.

Usage:
    conda run -p ${FEAST_ENV} python scripts/alignment_benchmark.py \
      --simulation-manifest outputs/simulation_manifest.csv \
      --methods-dir outputs/methods \
      --reference outputs/simulations/reference.h5ad \
      --output outputs/benchmarks/alignment_benchmark_results.csv
"""

import sys, os, csv, json, argparse, traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors

REGION_KEY = "ground_truth"


# ---------------------------------------------------------------------------
#  Coordinate-level metrics (same for both methods)
# ---------------------------------------------------------------------------

def recovered_rotation(aligned_coords, orig_coords):
    """Procrustes rotation (degrees) from aligned -> original.  0 = perfect."""
    c_a = aligned_coords - aligned_coords.mean(axis=0)
    c_o = orig_coords - orig_coords.mean(axis=0)
    H = c_a.T @ c_o
    try:
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    except np.linalg.LinAlgError:
        return np.nan


def coordinate_metrics(orig_coords, aligned_coords):
    dists = np.linalg.norm(aligned_coords - orig_coords, axis=1)
    px, _ = pearsonr(orig_coords[:, 0], aligned_coords[:, 0])
    py, _ = pearsonr(orig_coords[:, 1], aligned_coords[:, 1])
    return {
        "recovered_rotation": round(recovered_rotation(aligned_coords, orig_coords), 4),
        "mean_spatial_error": round(float(np.mean(dists)), 4),
        "median_spatial_error": round(float(np.median(dists)), 4),
        "rmse_spatial_error": round(float(np.sqrt(np.mean(dists ** 2))), 4),
        "pearson_x": round(float(px), 6),
        "pearson_y": round(float(py), 6),
    }


# ---------------------------------------------------------------------------
#  Unified nearest-neighbour matching (same for both methods)
# ---------------------------------------------------------------------------

def _to_dense(a):
    return a.X.toarray() if issparse(a.X) else a.X


def unified_nn_metrics(ref_adata, mov_adata, aligned_coords):
    """
    For each aligned spot, find nearest reference spot by spatial distance.
    Compute: spot-match accuracy, region accuracy, expression correlation.
    Same computation for all alignment methods.
    """
    ref_spatial = ref_adata.obsm["spatial"]
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(ref_spatial)
    distances, indices = nn.kneighbors(aligned_coords)

    nearest_ref_idx = indices[:, 0]

    # --- spot-match accuracy: does nearest ref spot share the barcode? ---
    ref_barcodes = np.array(ref_adata.obs_names)
    mov_barcodes = np.array(mov_adata.obs_names)
    nearest_barcodes = ref_barcodes[nearest_ref_idx]
    correct = (nearest_barcodes == mov_barcodes).mean()

    # --- region accuracy: do matched spots share the same layer? ---
    if REGION_KEY in ref_adata.obs and REGION_KEY in mov_adata.obs:
        ref_regions = ref_adata.obs[REGION_KEY].values.astype(str)
        mov_regions = mov_adata.obs[REGION_KEY].values.astype(str)
        region_acc = (ref_regions[nearest_ref_idx] == mov_regions).mean()
    else:
        region_acc = np.nan

    # --- expression correlation: per-spot Pearson between paired spots ---
    # For each matched spot pair, correlate the expression vectors.
    ref_X = _to_dense(ref_adata)
    mov_X = _to_dense(mov_adata)
    paired_ref = ref_X[nearest_ref_idx]
    paired_mov = mov_X

    # per-spot Pearson: corr(ref[i,:], mov[i,:]) for each i
    mr = paired_ref - paired_ref.mean(axis=1, keepdims=True)
    mm = paired_mov - paired_mov.mean(axis=1, keepdims=True)
    num = (mr * mm).sum(axis=1)
    den = np.sqrt((mr ** 2).sum(axis=1)) * np.sqrt((mm ** 2).sum(axis=1))
    with np.errstate(divide='ignore', invalid='ignore'):
        spot_corrs = np.divide(num, den)
        spot_corrs[np.isnan(spot_corrs)] = 0.0
    nz = (paired_ref.sum(axis=1) > 0) & (paired_mov.sum(axis=1) > 0)
    ge_corr = float(spot_corrs[nz].mean()) if nz.sum() > 1 else np.nan

    return {
        "nn_accuracy": round(float(correct), 6),
        "nn_region_accuracy": round(float(region_acc), 6) if not np.isnan(region_acc) else np.nan,
        "nn_mean_distance": round(float(distances.mean()), 2),
        "ge_correlation": round(float(ge_corr), 6) if not np.isnan(ge_corr) else np.nan,
        "n_aligned": len(aligned_coords),
    }


# ---------------------------------------------------------------------------
#  Spateo-specific: mapping-matrix metrics
# ---------------------------------------------------------------------------

def mapping_matrix_metrics(pi, ref_adata, mov_adata):
    """Metrics from morpho_align's probability matrix (Spateo only)."""
    pi = np.nan_to_num(pi, nan=0)
    pi = np.clip(pi, 0, None)

    # build ground truth
    common = np.intersect1d(ref_adata.obs_names, mov_adata.obs_names)
    id_to_idx = {id_: i for i, id_ in enumerate(mov_adata.obs_names)}
    gt = np.zeros((ref_adata.n_obs, mov_adata.n_obs))
    for i, bid in enumerate(ref_adata.obs_names):
        if bid in id_to_idx:
            gt[i, id_to_idx[bid]] = 1

    # argmax prediction
    pred = np.zeros_like(pi)
    rows = np.arange(pi.shape[0])
    cols = np.argmax(pi, axis=1)
    pred[rows, cols] = 1

    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    total_gt = gt.sum()

    # bidirectional consistency
    forward = np.argmax(pi, axis=1)
    backward = np.argmax(pi, axis=0)
    consistent = sum(1 for i, j in enumerate(forward)
                     if j < len(backward) and backward[j] == i)
    consistency = consistent / len(forward) if len(forward) > 0 else 0.0

    return {
        "morpho_accuracy": round(tp / total_gt, 6) if total_gt > 0 else 0.0,
        "morpho_precision": round(tp / (tp + fp), 6) if (tp + fp) > 0 else 0.0,
        "morpho_recall": round(tp / (tp + fn), 6) if (tp + fn) > 0 else 0.0,
        "morpho_f1": round(2 * tp / (2 * tp + fp + fn), 6) if (2 * tp + fp + fn) > 0 else 0.0,
        "morpho_consistency": round(float(consistency), 6),
    }


# ---------------------------------------------------------------------------
#  Per-job benchmark
# ---------------------------------------------------------------------------

def benchmark_one(methods_dir, alteration, method, angle, ref_adata,
                  manifest_lookup):
    angle_tag = str(int(angle)) if angle == int(angle) else str(angle)
    angle_dir = methods_dir / alteration / method / f"angle_{angle_tag}"

    info = manifest_lookup.get((alteration, angle), {})
    row = {
        "alteration_id": alteration,
        "alteration_type": info.get("alteration_type", "unknown"),
        "fold_change": info.get("fold_change", np.nan),
        "method": method, "angle": angle, "status": "ok",
        "recovered_rotation": np.nan,
        "mean_spatial_error": np.nan,
        "median_spatial_error": np.nan,
        "rmse_spatial_error": np.nan,
        "pearson_x": np.nan, "pearson_y": np.nan,
        "nn_accuracy": np.nan,
        "nn_region_accuracy": np.nan,
        "nn_mean_distance": np.nan,
        "ge_correlation": np.nan,
        "n_aligned": 0,
        "morpho_accuracy": np.nan,
        "morpho_precision": np.nan,
        "morpho_recall": np.nan,
        "morpho_f1": np.nan,
        "morpho_consistency": np.nan,
        "runtime_seconds": np.nan,
        "error_message": "",
    }

    fail_file = angle_dir / "FAILED.txt"
    if fail_file.exists():
        row["status"] = "failed"
        row["error_message"] = fail_file.read_text()[:200]
        return row

    coord_file = angle_dir / "aligned_coordinates.csv"
    if not coord_file.exists():
        row["status"] = "missing"
        row["error_message"] = "aligned_coordinates.csv not found"
        return row

    try:
        # --- load aligned coordinates ---
        coords_df = pd.read_csv(coord_file)
        aligned_c = coords_df[["x_aligned", "y_aligned"]].values.astype(float)
        orig_c = coords_df[["x_original", "y_original"]].values.astype(float)

        # --- coordinate metrics ---
        row.update(coordinate_metrics(orig_c, aligned_c))

        # --- unified NN metrics ---
        rot_path = manifest_lookup.get((alteration, angle), {}).get("rotated_path")
        if rot_path is None:
            rot_path = str(
                methods_dir.parent.parent / "simulations" /
                f"{alteration}_rotated_{angle_tag}.h5ad")
        mov_adata = sc.read_h5ad(rot_path)
        mov_adata.obs_names_make_unique()

        row.update(unified_nn_metrics(ref_adata, mov_adata, aligned_c))

        # --- Spateo-specific: mapping matrix ---
        pi_file = angle_dir / "mapping_matrix.npy"
        if pi_file.exists():
            pi = np.load(pi_file)
            row.update(mapping_matrix_metrics(pi, ref_adata, mov_adata))

        # --- runtime ---
        tf_file = angle_dir / "transform.json"
        if tf_file.exists():
            row["runtime_seconds"] = json.loads(
                tf_file.read_text()).get("runtime_seconds", np.nan)

    except Exception:
        row["status"] = "failed"
        row["error_message"] = traceback.format_exc()[:400]

    return row


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Alignment benchmark (v4)")
    parser.add_argument("--simulation-manifest", type=str, required=True)
    parser.add_argument("--methods-dir", type=str, required=True)
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.simulation_manifest)
    rotated = manifest[manifest["type"] == "rotated"]

    manifest_lookup = {}
    for _, r in rotated.iterrows():
        key = (r["alteration_id"], float(r["angle"]))
        manifest_lookup[key] = {
            "alteration_type": r["alteration_type"],
            "fold_change": r["fold_change"],
            "rotated_path": r["file_path"],
        }

    ref_adata = sc.read_h5ad(args.reference)
    ref_adata.obs_names_make_unique()
    print(f"Reference: {ref_adata.n_obs} spots, {ref_adata.n_vars} genes")

    alterations = sorted(set(rotated["alteration_id"]))
    angles = sorted(set(rotated["angle"]))
    methods_dir = Path(args.methods_dir)

    total = 0
    for alt in alterations:
        d = methods_dir / alt
        if d.is_dir():
            for md in sorted(d.iterdir()):
                if md.is_dir() and not md.name.startswith("."):
                    total += len(angles)

    print(f"Alterations: {alterations}")
    print(f"Angles: {angles}")
    print(f"Total jobs: {total}")

    results = []
    count = 0
    for alt in alterations:
        d = methods_dir / alt
        if not d.is_dir():
            continue
        for md in sorted(d.iterdir()):
            if not md.is_dir() or md.name.startswith("."):
                continue
            method = md.name
            for angle in angles:
                count += 1
                row = benchmark_one(methods_dir, alt, method, angle,
                                    ref_adata, manifest_lookup)
                results.append(row)
                tag = f"{alt}/{method}/angle_{angle}"
                if row["status"] == "ok":
                    print(f"  [{count}/{total}] {tag}: "
                          f"nn_acc={row['nn_accuracy']:.4f} "
                          f"nn_reg={row['nn_region_accuracy']:.4f} "
                          f"spat={row['mean_spatial_error']:.1f} "
                          f"ge_corr={row['ge_correlation']:.4f}")
                else:
                    print(f"  [{count}/{total}] {tag}: {row['status']}")

    # --- write ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys()) if results else []
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nResults: {out_path} ({len(results)} rows)")

    # --- summary ---
    df = pd.DataFrame(results)
    ok = df[df["status"] == "ok"]
    summary_path = out_path.parent / "alignment_benchmark_summary.csv"

    sum_rows = []
    for method in sorted(ok["method"].unique()):
        for alt in sorted(ok["alteration_id"].unique()):
            m = ok[(ok["method"] == method) & (ok["alteration_id"] == alt)]
            if len(m) == 0:
                continue
            sum_rows.append({
                "alteration_id": alt,
                "alteration_type": m["alteration_type"].iloc[0],
                "fold_change": m["fold_change"].iloc[0],
                "method": method,
                "mean_nn_accuracy": round(m["nn_accuracy"].mean(), 6),
                "mean_nn_region_accuracy": round(m["nn_region_accuracy"].mean(), 6),
                "mean_spatial_error": round(m["mean_spatial_error"].mean(), 2),
                "mean_ge_correlation": round(m["ge_correlation"].mean(), 6),
                "success_rate": round(len(m) / 5, 4),
                "n_angles": len(m),
            })

    with open(summary_path, "w", newline="") as f:
        if sum_rows:
            w = csv.DictWriter(f, fieldnames=list(sum_rows[0].keys()))
            w.writeheader()
            w.writerows(sum_rows)
    print(f"Summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
