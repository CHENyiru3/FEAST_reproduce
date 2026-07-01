#!/usr/bin/env python
"""FEAST Batch Effect Assessment — Main Entry Point.

Loads multiple spatial transcriptomics slices, fits FEAST parameter clouds,
and produces diagnostic reports describing slice-to-slice distributional
differences in (log_mu, log_omega, logit_pi0) space.

Usage:
    python run_assessment.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# -- path setup ----------------------------------------------------------
FEAST_ROOT = Path("/maiziezhou_lab2/yiru/FEAST")
sys.path.insert(0, str(FEAST_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from batch_assessment.cloud_extraction import extract_clouds, filter_common_genes
from batch_assessment.metrics import (
    compute_centroids, compute_covariances, pairwise_distances, slice_qc_metrics,
)
from batch_assessment.affine_model import fit_diagonal_affine, residual_analysis
from batch_assessment.domain_analysis import (
    per_domain_clouds, domain_shift_summary, within_vs_between_domain_variance,
)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_safe(obj):
    """Recursively convert numpy types for JSON serialisation."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def run_assessment(config: dict) -> Path:
    """Execute the full batch assessment pipeline.

    Returns path to the timestamped output directory.
    """
    import scanpy as sc

    # ---- setup ---------------------------------------------------------
    out_root = Path(config.get("output_dir", "outputs"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _ensure_dir(out_root / timestamp)
    seed = config.get("random_seed", 42)
    np.random.seed(seed)

    data_dir = Path(config["data_dir"])
    slices = config["slices"]
    ref_slice = config.get("reference_slice", slices[0])
    gene_filter = config.get("gene_filter", {})
    dist_metrics = config.get("distance_metrics",
        ["centroid_euclidean", "covariance_frobenius", "mmd_rbf"])
    domain_key = config.get("domain_key", "dlpfc_layer")

    print(f"Output directory: {out_dir}")
    print(f"Slices: {slices}")
    print(f"Reference slice: {ref_slice}")

    # ---- 1.  Load slices ------------------------------------------------
    print("\n[1/8] Loading slices ...")
    adata_dict = {}
    for s in slices:
        p = data_dir / f"{s}.h5ad"
        if p.exists():
            adata = sc.read_h5ad(p)
            adata_dict[s] = adata
            print(f"  {s}: {adata.n_obs} spots x {adata.n_vars} genes")
        else:
            print(f"  {s}: SKIPPED (not found at {p})")
    print(f"  Loaded {len(adata_dict)} slices.")

    # ---- 2.  Slice QC ---------------------------------------------------
    print("\n[2/8] Slice-level QC ...")
    qc = slice_qc_metrics(adata_dict)
    qc.to_csv(out_dir / "qc_summary.csv")
    print(qc.to_string())

    # ---- 3.  Extract parameter clouds ------------------------------------
    print("\n[3/8] Extracting parameter clouds ...")
    clouds = extract_clouds(adata_dict, gene_filter=gene_filter)
    if gene_filter.get("common_genes_only", False):
        clouds = filter_common_genes(clouds)
    n_genes = clouds[list(clouds.keys())[0]]["theta"].shape[0]
    print(f"  {len(clouds)} clouds, {n_genes} common genes")

    # ---- 4.  Centroids --------------------------------------------------
    print("\n[4/8] Computing centroids and batch shifts ...")
    centroids = compute_centroids(clouds, ref_slice=ref_slice)
    with open(out_dir / "centroids.json", "w") as f:
        json.dump(_json_safe(centroids), f, indent=2)
    pd.DataFrame(
        centroids["shifts"],
        index=centroids["slice_order"],
        columns=centroids["dimension_names"],
    ).to_csv(out_dir / "centroid_shifts.csv")
    for i, sid in enumerate(centroids["slice_order"]):
        shift = centroids["shifts"][i]
        print(f"  {sid}: Δ(log_μ)={shift[0]:+.4f}  Δ(log_ω)={shift[1]:+.4f}  Δ(logit_π₀)={shift[2]:+.4f}")

    # ---- 5.  Covariances ------------------------------------------------
    print("\n[5/8] Computing covariance matrices ...")
    cov_results = compute_covariances(clouds)
    np.savez(out_dir / "covariances.npz",
             covariances=cov_results["covariances"],
             log_det=cov_results["log_det"],
             frobenius_norm=cov_results["frobenius_norm"],
             eigenvalues=cov_results["eigenvalues"])
    cov_summary = pd.DataFrame({
        "slice_id": cov_results["slice_order"],
        "log_det": cov_results["log_det"],
        "frobenius_norm": cov_results["frobenius_norm"],
        "eigval_1": cov_results["eigenvalues"][:, 0],
        "eigval_2": cov_results["eigenvalues"][:, 1],
        "eigval_3": cov_results["eigenvalues"][:, 2],
    }).set_index("slice_id")
    cov_summary.to_csv(out_dir / "covariance_summary.csv")
    print(cov_summary.to_string())

    # ---- 6.  Pairwise distances -----------------------------------------
    print("\n[6/8] Computing pairwise distances ...")
    dists = pairwise_distances(clouds, dist_metrics)
    for metric, d in dists.items():
        if metric == "slice_order":
            continue
        np.savez(out_dir / f"distance_{metric}.npz",
                 matrix=d, slice_order=np.array(dists["slice_order"]))
        print(f"  {metric}: range [{d.min():.4f}, {d.max():.4f}]")

    # ---- 7.  Affine fit -------------------------------------------------
    print("\n[7/8] Fitting diagonal affine deformations ...")
    affine = fit_diagonal_affine(clouds, ref_slice=ref_slice)
    affine["params_df"].to_csv(out_dir / "affine_params.csv")
    residuals = residual_analysis(clouds, affine)
    resid_arrays = {
        f"residual_{sid}": residuals["residual_clouds"][sid]
        for sid in residuals["slice_order"]
    }
    np.savez(out_dir / "affine_residuals.npz", **resid_arrays)
    print(affine["params_df"].to_string())

    # ---- 8.  Domain analysis --------------------------------------------
    print("\n[8/8] Per-domain analysis ...")
    has_domains = all(domain_key in adata.obs.columns for adata in adata_dict.values())
    if has_domains:
        dom_clouds = per_domain_clouds(adata_dict, domain_key=domain_key)
        dom_summary = domain_shift_summary(dom_clouds, ref_domain="Layer_1")
        dom_summary.to_csv(out_dir / "domain_shifts.csv")

        decomp = within_vs_between_domain_variance(dom_clouds)
        ratio_df = pd.DataFrame({
            "slice_id": list(decomp["between_total_ratio"].keys()),
            "between_total_ratio": list(decomp["between_total_ratio"].values()),
        }).set_index("slice_id")
        ratio_df.to_csv(out_dir / "domain_variance_decomposition.csv")
        print(ratio_df.to_string())
    else:
        print("  No domain labels found, skipping domain analysis.")

    # ---- done -----------------------------------------------------------
    print(f"\nAssessment complete. Results in: {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="FEAST Batch Effect Assessment"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path

    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_assessment(config)


if __name__ == "__main__":
    main()
