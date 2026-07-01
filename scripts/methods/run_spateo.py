#!/usr/bin/env python
"""
Spateo alignment method runner.

Aligns a moving (rotated) slice to a reference slice using Spateo's
morpho_align with group PCA + cosine dissimilarity.

Usage:
    conda run -p ${FEAST_ENV} python scripts/methods/run_spateo.py \
      --reference outputs/simulations/original.h5ad \
      --moving outputs/simulations/rotated_30.h5ad \
      --output-dir outputs/methods/spateo/angle_30/
"""

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path

import numpy as np
import scanpy as sc

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fix_nested_dict_keys(d):
    """Recursively convert integer keys to strings in nested dicts."""
    for key in list(d.keys()):
        if isinstance(key, int):
            d[str(key)] = d.pop(key)
        val = d[str(key)] if isinstance(key, int) else d[key]
        if isinstance(val, dict):
            _fix_nested_dict_keys(val)
    return d


def convert_int_keys_to_str(adata):
    """Fix integer keys in adata.uns so h5py serialization succeeds."""
    for key in list(adata.uns.keys()):
        if isinstance(adata.uns[key], dict):
            _fix_nested_dict_keys(adata.uns[key])
    return adata


# ---------------------------------------------------------------------------
# core alignment
# ---------------------------------------------------------------------------

def run_spateo(
    reference_path: str,
    moving_path: str,
    output_dir: str,
    device: str = "cpu",
    dissimilarity: str = "cos",
    rep_layer: str = "X_pca",
    min_genes: int = 1,
    min_cells: int = 1,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if device == "cuda" else ""

    import spateo as st

    log_lines = []
    t_start = time.time()

    log_lines.append(f"=== Spateo Alignment ===")
    log_lines.append(f"Date: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    log_lines.append(f"Reference: {reference_path}")
    log_lines.append(f"Moving: {moving_path}")
    log_lines.append(f"Device: {device}")
    log_lines.append(f"Spateo version: {st.__version__}")

    # -- load --
    slice_ref = sc.read_h5ad(reference_path)
    slice_mov = sc.read_h5ad(moving_path)
    log_lines.append(f"Reference: {slice_ref.shape[0]} spots, {slice_ref.shape[1]} genes")
    log_lines.append(f"Moving: {slice_mov.shape[0]} spots, {slice_mov.shape[1]} genes")

    # -- preprocess --
    def preprocess(s):
        s.obs_names_make_unique()
        s.var_names_make_unique()
        sc.pp.filter_cells(s, min_genes=min_genes)
        sc.pp.filter_genes(s, min_cells=min_cells)
        s.layers["counts"] = s.X.copy()
        sc.pp.normalize_total(s)
        sc.pp.log1p(s)

    preprocess(slice_ref)
    preprocess(slice_mov)

    # -- joint PCA --
    st.align.group_pca([slice_ref, slice_mov], pca_key=rep_layer)

    # -- align --
    aligned_slices, pis = st.align.morpho_align(
        models=[slice_ref, slice_mov],
        rep_layer=rep_layer,
        rep_field="obsm",
        dissimilarity=dissimilarity,
        verbose=False,
        spatial_key="spatial",
        key_added="spatial_aligned",
        device=device,
        return_mapping=True,
    )

    elapsed = time.time() - t_start
    log_lines.append(f"Alignment complete: {elapsed:.1f}s")
    log_lines.append(f"Mapping matrix shape: {pis[0].shape}")

    # -- fix & save --
    aligned_ref = convert_int_keys_to_str(aligned_slices[0])
    aligned_mov = convert_int_keys_to_str(aligned_slices[1])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    aligned_ref.write_h5ad(out / "aligned.h5ad")
    np.save(out / "mapping_matrix.npy", pis[0])

    # -- coordinates CSV --
    orig_coords = aligned_mov.obsm.get("spatial_original", aligned_mov.obsm["spatial"])
    aligned_coords = aligned_mov.obsm.get("spatial_aligned", aligned_mov.obsm["spatial"])

    rows = []
    for i, barcode in enumerate(aligned_mov.obs_names):
        rows.append({
            "spot_barcode": barcode,
            "x_original": float(orig_coords[i, 0]),
            "y_original": float(orig_coords[i, 1]),
            "x_aligned": float(aligned_coords[i, 0]),
            "y_aligned": float(aligned_coords[i, 1]),
        })

    import csv
    with open(out / "aligned_coordinates.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "spot_barcode", "x_original", "y_original", "x_aligned", "y_aligned"
        ])
        writer.writeheader()
        writer.writerows(rows)
    log_lines.append(f"aligned_coordinates.csv: {len(rows)} rows")

    # -- transform.json --
    transform = {
        "method": "spateo",
        "device": device,
        "dissimilarity": dissimilarity,
        "rep_layer": rep_layer,
        "status": "ok",
        "runtime_seconds": round(elapsed, 2),
    }
    with open(out / "transform.json", "w") as f:
        json.dump(transform, f, indent=2)

    # -- log --
    log_lines.append(f"Outputs written to: {out}")
    log_lines.append(f"Runtime: {elapsed:.1f}s")
    log_lines.append(f"Status: ok")
    with open(out / "log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")

    return aligned_slices, pis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spateo alignment runner")
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--moving", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dissimilarity", type=str, default="cos")
    parser.add_argument("--rep-layer", type=str, default="X_pca")
    parser.add_argument("--min-genes", type=int, default=1)
    parser.add_argument("--min-cells", type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists(args.reference):
        raise FileNotFoundError(f"Reference not found: {args.reference}")
    if not os.path.exists(args.moving):
        raise FileNotFoundError(f"Moving not found: {args.moving}")

    try:
        run_spateo(
            reference_path=args.reference,
            moving_path=args.moving,
            output_dir=args.output_dir,
            device=args.device,
            dissimilarity=args.dissimilarity,
            rep_layer=args.rep_layer,
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
