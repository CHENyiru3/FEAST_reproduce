#!/usr/bin/env python
"""
PASTE alignment method runner.

Aligns a moving rotated slice to a reference slice using PASTE's
pairwise FGW-OT coupling. The coupling is converted to a rigid transform
that maps moving coordinates back into the reference coordinate system.
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc


def _preprocess_for_pca(adata, min_genes, min_cells):
    adata = adata.copy()
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    # Keep all spots: benchmark.py assumes aligned_coordinates.csv remains in
    # the same observation order as the original moving .h5ad.
    if min_cells > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def _add_joint_pca(slice_ref, slice_mov, n_pcs):
    common = slice_ref.var_names.intersection(slice_mov.var_names)
    if len(common) == 0:
        raise ValueError("No common genes between reference and moving slices")

    slice_ref = slice_ref[:, common].copy()
    slice_mov = slice_mov[:, common].copy()

    combined = ad.concat(
        [slice_ref, slice_mov],
        axis=0,
        join="inner",
        label="slice",
        keys=["reference", "moving"],
        index_unique="-",
    )
    max_pcs = min(n_pcs, combined.n_obs - 1, combined.n_vars - 1)
    if max_pcs < 2:
        raise ValueError(f"Not enough observations/genes for PCA: {combined.shape}")

    sc.pp.pca(combined, n_comps=max_pcs, svd_solver="arpack")
    pca = np.asarray(combined.obsm["X_pca"], dtype=np.float64)
    slice_ref.obsm["X_pca"] = pca[: slice_ref.n_obs]
    slice_mov.obsm["X_pca"] = pca[slice_ref.n_obs :]
    return slice_ref, slice_mov, max_pcs, len(common)


def _weighted_procrustes_to_reference(ref_coords, mov_coords, pi):
    """Return moving coordinates rigidly aligned into reference coordinates."""
    ref_mass = pi.sum(axis=1)
    mov_mass = pi.sum(axis=0)
    ref_center = ref_mass.dot(ref_coords)
    mov_center = mov_mass.dot(mov_coords)

    ref_centered = ref_coords - ref_center
    mov_centered = mov_coords - mov_center
    h = mov_centered.T.dot(pi.T.dot(ref_centered))
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T.dot(u.T)
    aligned = mov_centered.dot(rotation.T) + ref_center

    angle = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    return aligned, rotation, angle, ref_center, mov_center


def _patch_paste_pot_line_search():
    """Adapt paste-bio 1.4.0's line-search callback for current POT."""
    import ot

    original_cg = ot.optim.cg
    if getattr(original_cg, "_paste_compat", False):
        return

    def cg_compat(a, b, M, reg, f, df, G0=None, line_search=None, **kwargs):
        if line_search is not None:
            original_line_search = line_search

            def line_search_compat(cost, G, deltaG, Mi, cost_G, df_G=None, **ls_kwargs):
                return original_line_search(cost, G, deltaG, Mi, cost_G, **ls_kwargs)

            line_search = line_search_compat
        return original_cg(a, b, M, reg, f, df, G0, line_search, **kwargs)

    cg_compat._paste_compat = True
    ot.optim.cg = cg_compat


def run_paste(
    reference_path,
    moving_path,
    output_dir,
    alpha=0.1,
    dissimilarity="euclidean",
    n_pcs=50,
    num_iter_max=200,
    norm=True,
    min_genes=1,
    min_cells=1,
):
    import paste as pst

    log_lines = []
    t_start = time.time()

    log_lines.append("=== PASTE Alignment ===")
    log_lines.append(f"Date: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    log_lines.append(f"Reference: {reference_path}")
    log_lines.append(f"Moving: {moving_path}")
    log_lines.append(f"PASTE version: {getattr(pst, '__version__', 'unknown')}")
    log_lines.append(f"alpha={alpha} dissimilarity={dissimilarity} n_pcs={n_pcs} norm={norm}")

    slice_ref_raw = sc.read_h5ad(reference_path)
    slice_mov_raw = sc.read_h5ad(moving_path)
    slice_ref_raw.obs_names_make_unique()
    slice_mov_raw.obs_names_make_unique()
    slice_ref_raw.var_names_make_unique()
    slice_mov_raw.var_names_make_unique()

    log_lines.append(
        f"Reference raw: {slice_ref_raw.shape[0]} spots, {slice_ref_raw.shape[1]} genes"
    )
    log_lines.append(
        f"Moving raw: {slice_mov_raw.shape[0]} spots, {slice_mov_raw.shape[1]} genes"
    )

    slice_ref = _preprocess_for_pca(slice_ref_raw, min_genes, min_cells)
    slice_mov = _preprocess_for_pca(slice_mov_raw, min_genes, min_cells)
    slice_ref, slice_mov, used_pcs, n_common = _add_joint_pca(slice_ref, slice_mov, n_pcs)
    log_lines.append(f"Common genes: {n_common}; PCA dimensions: {used_pcs}")

    _patch_paste_pot_line_search()
    pi = pst.pairwise_align(
        slice_ref,
        slice_mov,
        alpha=alpha,
        dissimilarity=dissimilarity,
        use_rep="X_pca",
        norm=norm,
        numItermax=num_iter_max,
        verbose=False,
        gpu_verbose=False,
    )
    pi = np.asarray(pi, dtype=np.float64)

    ref_coords = np.asarray(slice_ref.obsm["spatial"], dtype=np.float64)
    mov_coords = np.asarray(slice_mov.obsm["spatial"], dtype=np.float64)
    aligned_coords, rotation, rotation_angle, ref_center, mov_center = (
        _weighted_procrustes_to_reference(ref_coords, mov_coords, pi)
    )

    elapsed = time.time() - t_start
    log_lines.append(f"Alignment complete: {elapsed:.1f}s")
    log_lines.append(f"Coupling matrix shape: {pi.shape}")
    log_lines.append(f"Estimated rotation angle: {rotation_angle:.4f}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    failed_marker = out / "FAILED.txt"
    if failed_marker.exists():
        failed_marker.unlink()

    aligned_mov = slice_mov_raw.copy()
    aligned_mov.obsm["spatial_original"] = mov_coords
    aligned_mov.obsm["spatial_aligned"] = aligned_coords
    aligned_mov.write_h5ad(out / "aligned.h5ad")

    np.save(out / "coupling_matrix.npy", pi)

    rows = []
    for i, barcode in enumerate(aligned_mov.obs_names):
        rows.append(
            {
                "spot_barcode": barcode,
                "x_original": float(mov_coords[i, 0]),
                "y_original": float(mov_coords[i, 1]),
                "x_aligned": float(aligned_coords[i, 0]),
                "y_aligned": float(aligned_coords[i, 1]),
            }
        )
    with open(out / "aligned_coordinates.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "spot_barcode",
                "x_original",
                "y_original",
                "x_aligned",
                "y_aligned",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    log_lines.append(f"aligned_coordinates.csv: {len(rows)} rows")

    transform = {
        "method": "paste",
        "alpha": alpha,
        "dissimilarity": dissimilarity,
        "use_rep": "X_pca",
        "n_pcs": used_pcs,
        "norm": norm,
        "num_iter_max": num_iter_max,
        "rotation_angle": rotation_angle,
        "rotation_matrix": rotation.tolist(),
        "reference_center": ref_center.tolist(),
        "moving_center": mov_center.tolist(),
        "status": "ok",
        "runtime_seconds": round(elapsed, 2),
    }
    with open(out / "transform.json", "w") as f:
        json.dump(transform, f, indent=2)

    log_lines.append(f"Outputs written to: {out}")
    log_lines.append(f"Runtime: {elapsed:.1f}s")
    log_lines.append("Status: ok")
    with open(out / "log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")

    return aligned_mov, pi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PASTE alignment runner")
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--moving", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--dissimilarity", type=str, default="euclidean")
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--num-iter-max", type=int, default=200)
    parser.add_argument("--no-norm", action="store_true")
    parser.add_argument("--min-genes", type=int, default=1)
    parser.add_argument("--min-cells", type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists(args.reference):
        raise FileNotFoundError(f"Reference not found: {args.reference}")
    if not os.path.exists(args.moving):
        raise FileNotFoundError(f"Moving not found: {args.moving}")

    try:
        run_paste(
            reference_path=args.reference,
            moving_path=args.moving,
            output_dir=args.output_dir,
            alpha=args.alpha,
            dissimilarity=args.dissimilarity,
            n_pcs=args.n_pcs,
            num_iter_max=args.num_iter_max,
            norm=not args.no_norm,
            min_genes=args.min_genes,
            min_cells=args.min_cells,
        )
    except Exception:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        (out / "FAILED.txt").write_text(tb)
        (out / "log.txt").write_text(tb)
        print(tb, file=sys.stderr)
        sys.exit(1)
