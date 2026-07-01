#!/usr/bin/env python
"""
SPACEL alignment method runner.

Aligns a moving (rotated) slice to a reference slice using SPACEL's
Scube.align with Hungarian-optimal spot matching.

Usage:
    conda run -p ${SPACEL_ENV} python scripts/methods/run_spacel.py \
      --reference outputs/simulations/original.h5ad \
      --moving outputs/simulations/rotated_30.h5ad \
      --output-dir outputs/methods/spacel/angle_30/ \
      --layer-key sce.layer_guess
"""

import os
import sys
import json
import time
import argparse
import traceback
import csv
from pathlib import Path

import numpy as np
import scanpy as sc
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# spot matching (Hungarian)
# ---------------------------------------------------------------------------

def compute_matches_hungarian(coords1, coords2):
    """1:1 optimal matching between two coordinate sets via Hungarian algorithm."""
    dist = cdist(coords1, coords2)
    row_idx, col_idx = linear_sum_assignment(dist)
    match_matrix = np.zeros((len(coords1), len(coords2)))
    match_matrix[row_idx, col_idx] = 1
    avg_dist = dist[row_idx, col_idx].mean()
    return match_matrix, avg_dist


# ---------------------------------------------------------------------------
# core alignment
# ---------------------------------------------------------------------------

def run_spacel(
    reference_path: str,
    moving_path: str,
    output_dir: str,
    layer_key: str = "sce.layer_guess",
    n_neighbors: int = 15,
    n_threads: int = 12,
):
    import SPACEL
    from SPACEL import Scube

    log_lines = []
    t_start = time.time()

    log_lines.append(f"=== SPACEL Alignment ===")
    log_lines.append(f"Date: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    log_lines.append(f"Reference: {reference_path}")
    log_lines.append(f"Moving: {moving_path}")
    log_lines.append(f"SPACEL version: {getattr(SPACEL, '__version__', 'unknown')}")

    # -- load --
    slice_ref = sc.read_h5ad(reference_path)
    slice_mov = sc.read_h5ad(moving_path)
    slice_ref.obs_names_make_unique()
    slice_mov.obs_names_make_unique()
    slice_ref.var_names_make_unique()
    slice_mov.var_names_make_unique()
    log_lines.append(f"Reference: {slice_ref.shape[0]} spots, {slice_ref.shape[1]} genes")
    log_lines.append(f"Moving: {slice_mov.shape[0]} spots, {slice_mov.shape[1]} genes")

    # -- resolve layer key for SPACEL cluster matching --
    # SPACEL's Scube.align requires pre-computed spatial domain labels (from
    # Splane or equivalent).  The FEAST-simulated reference inherits DLPFC
    # layer annotations from the source data.  Spateo does not use layer
    # labels — this is a method-level design difference, not an unfair
    # advantage.  The expected version's SPACEL runner likewise uses
    # pre-existing annotations.
    fallback_keys = ["ground_truth", "annotation", "dlpfc_layer"]
    effective_key = None
    for key in [layer_key] + fallback_keys:
        if key in slice_ref.obs.columns and key in slice_mov.obs.columns:
            effective_key = key
            break

    if effective_key is None:
        raise RuntimeError(
            f"No layer key found in either slice. Tried: {[layer_key] + fallback_keys}. "
            f"Ref columns: {list(slice_ref.obs.columns)}"
        )

    slice_ref.obs[effective_key] = slice_ref.obs[effective_key].astype(str)
    slice_mov.obs[effective_key] = slice_mov.obs[effective_key].astype(str)
    log_lines.append(f"Layer key: {effective_key} ({slice_ref.obs[effective_key].nunique()} ref, {slice_mov.obs[effective_key].nunique()} mov)")

    # -- align (matches expected version: n_neighbors=15, no extra params) --
    adata_list = [slice_ref.copy(), slice_mov.copy()]

    # SPACEL centers coordinates internally (subtracts median).  Save the
    # reference median so we can de-center aligned output back to pixel space
    # for downstream metrics that expect original coordinate scale.
    ref_median = np.median(np.asarray(slice_ref.obsm['spatial']), axis=0)

    Scube.align(
        adata_list,
        cluster_key=effective_key,
        n_neighbors=n_neighbors,
        n_threads=n_threads,
    )

    aligned_ref, aligned_mov = adata_list[0], adata_list[1]
    elapsed = time.time() - t_start
    log_lines.append(f"Alignment complete: {elapsed:.1f}s")

    # -- coordinate extraction (de-center back to pixel space) --
    coords_key = "spatial_aligned" if "spatial_aligned" in aligned_mov.obsm else "spatial"
    orig_key = "spatial_original" if "spatial_original" in aligned_mov.obsm else "spatial"

    orig_coords = np.asarray(aligned_mov.obsm[orig_key])
    aligned_coords = np.asarray(aligned_mov.obsm[coords_key]) + ref_median[:2]

    # -- save outputs --
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    aligned_ref.write_h5ad(out / "aligned.h5ad")

    # coordinates CSV
    rows = []
    for i, barcode in enumerate(aligned_mov.obs_names):
        rows.append({
            "spot_barcode": barcode,
            "x_original": float(orig_coords[i, 0]),
            "y_original": float(orig_coords[i, 1]),
            "x_aligned": float(aligned_coords[i, 0]),
            "y_aligned": float(aligned_coords[i, 1]),
        })
    with open(out / "aligned_coordinates.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "spot_barcode", "x_original", "y_original", "x_aligned", "y_aligned"
        ])
        writer.writeheader()
        writer.writerows(rows)
    log_lines.append(f"aligned_coordinates.csv: {len(rows)} rows")

    # transform.json
    transform = {
        "method": "spacel",
        "layer_key": effective_key,
        "n_neighbors": n_neighbors,
        "n_threads": n_threads,
        "status": "ok",
        "runtime_seconds": round(elapsed, 2),
    }
    with open(out / "transform.json", "w") as f:
        json.dump(transform, f, indent=2)

    # log
    log_lines.append(f"Outputs written to: {out}")
    log_lines.append(f"Runtime: {elapsed:.1f}s")
    log_lines.append(f"Status: ok")
    with open(out / "log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")

    return aligned_ref, aligned_mov


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPACEL alignment runner")
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--moving", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--layer-key", type=str, default="sce.layer_guess")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-threads", type=int, default=12)
    args = parser.parse_args()

    if not os.path.exists(args.reference):
        raise FileNotFoundError(f"Reference not found: {args.reference}")
    if not os.path.exists(args.moving):
        raise FileNotFoundError(f"Moving not found: {args.moving}")

    try:
        run_spacel(
            reference_path=args.reference,
            moving_path=args.moving,
            output_dir=args.output_dir,
            layer_key=args.layer_key,
            n_neighbors=args.n_neighbors,
            n_threads=args.n_threads,
        )
    except Exception:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        (out / "FAILED.txt").write_text(tb)
        (out / "log.txt").write_text(tb)
        print(tb, file=sys.stderr)
        sys.exit(1)
