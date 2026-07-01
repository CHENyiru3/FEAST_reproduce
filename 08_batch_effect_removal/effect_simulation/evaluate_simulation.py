#!/usr/bin/env python
"""Validate batch-effect ladder: P-deformation fidelity + Q preservation.

Usage:
    python evaluate_simulation.py outputs/20250101_120000/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEAST_ROOT = Path("/maiziezhou_lab2/yiru/FEAST")
sys.path.insert(0, str(FEAST_ROOT / "src"))
_effect_root = Path(__file__).resolve().parent.parent / "effect_verification"
sys.path.insert(0, str(_effect_root))

from batch_assessment.cloud_extraction import extract_clouds
from batch_assessment.metrics import compute_centroids


def evaluate(output_dir: Path) -> dict:
    import scanpy as sc
    from scipy.sparse import issparse

    def _dense(x):
        return x.toarray() if issparse(x) else np.asarray(x, dtype=np.float64)

    manifest = pd.read_csv(output_dir / "manifest.csv")
    report = {"output_dir": str(output_dir), "checks": []}

    def check(name: str, passed: bool, detail: str = ""):
        report["checks"].append({"name": name, "passed": bool(passed), "detail": detail})
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if detail and not passed:
            print(f"         {detail}")

    for mode_name in manifest["mode"].unique():
        mode_df = manifest[manifest["mode"] == mode_name].sort_values("alpha")
        alphas = mode_df["alpha"].values

        print(f"\n=== {mode_name} ===")

        # ---- load all slices ----
        adata_dict = {}
        for _, row in mode_df.iterrows():
            p = Path(row["file"])
            # manifest paths are relative to this script's directory
            if not p.exists():
                p = Path(__file__).resolve().parent / p
            if p.exists():
                adata_dict[f"alpha={row['alpha']:.2f}"] = sc.read_h5ad(p)

        # ---- 1. QC monotonic ----
        check("total_counts monotonic",
              np.all(np.diff(mode_df["total_counts_mean"].values) < 0),
              str(mode_df["total_counts_mean"].values))
        check("zero_fraction monotonic",
              np.all(np.diff(mode_df["zero_fraction"].values) > 0),
              str(mode_df["zero_fraction"].values))

        # ---- 2. PARAMETER-CLOUD VALIDATION ----
        # Verify theta_batch stored in .var matches the deformation formula
        D = np.array(adata_dict["alpha=0.00"].uns["batch_deformation"]["D"])
        b = np.array(adata_dict["alpha=0.00"].uns["batch_deformation"]["b"])
        theta_ref = adata_dict["alpha=0.00"].var[[
            "theta_mu_ref", "theta_omega_ref", "theta_pi0_ref"
        ]].values

        max_theta_err = 0.0
        for alpha_label, adata in adata_dict.items():
            alpha = adata.uns["batch_deformation"]["alpha"]
            theta_batch_actual = adata.var[[
                "theta_mu_batch", "theta_omega_batch", "theta_pi0_batch"
            ]].values
            expected = theta_ref * ((1 - alpha) + alpha * D)[None, :] + alpha * b[None, :]
            err = np.max(np.abs(theta_batch_actual - expected))
            max_theta_err = max(max_theta_err, err)
        check(
            "theta_batch matches formula (max err < 1e-10)",
            max_theta_err < 1e-10,
            f"max error = {max_theta_err:.2e}",
        )

        # ---- 3. Re-extract theta from simulated counts and verify ----
        # This validates that the count matrix actually realizes the intended
        # parameter-cloud deformation (not just the stored metadata).
        clouds = extract_clouds(adata_dict)
        centroids = compute_centroids(clouds, ref_slice="alpha=0.00")
        shifts = centroids["shifts"]
        mags = np.linalg.norm(shifts, axis=1)

        check("centroid shift monotonic", np.all(np.diff(mags[1:]) >= -1e-10))
        check("centroid shift linear in alpha",
              _check_linearity(alphas[1:], mags[1:]),
              f"r={np.corrcoef(alphas[1:], mags[1:])[0,1]:.4f}")

        # Expected shift per alpha from formula
        theta_ref_centroid = theta_ref.mean(axis=0)
        shift_per_unit = (D - 1.0) * theta_ref_centroid + b
        expected_mags_nonzero = np.array([
            np.linalg.norm(alpha * shift_per_unit) for alpha in alphas[1:]
        ])
        # Observed centroid from re-extracted stats includes decode noise,
        # so magnitudes differ from pure-theta formula.  Verify they are
        # strongly correlated (proportional) rather than numerically equal.
        corr_obs_exp = np.corrcoef(mags[1:], expected_mags_nonzero)[0, 1]
        check(
            "extracted centroid shift proportional to expected (r > 0.99)",
            corr_obs_exp > 0.99,
            f"r={corr_obs_exp:.4f}, "
            f"observed={[f'{m:.4f}' for m in mags[1:]]}, "
            f"expected={[f'{m:.4f}' for m in expected_mags_nonzero]}",
        )

        # ---- 4. SPATIAL PRESERVATION ----
        # 4a. Domain labels identical
        ref_keys = list(adata_dict.keys())
        ref_adata = adata_dict[ref_keys[0]]
        if "dlpfc_layer" in ref_adata.obs.columns:
            ref_domains = ref_adata.obs["dlpfc_layer"].values
            all_match = all(
                np.array_equal(ref_domains, adata_dict[k].obs["dlpfc_layer"].values)
                for k in ref_keys
            )
            check("domain labels identical across all alpha", all_match)

        # 4b. Spatial coords preserved
        all_spatial = all(
            np.allclose(ref_adata.obsm["spatial"], adata_dict[k].obsm["spatial"])
            for k in ref_keys[1:]
        )
        check("spatial coordinates identical across all alpha", all_spatial)

        # 4c. Moran's I stability: spatial autocorrelation preserved
        if len(ref_keys) >= 3:
            moran_check = _check_moran_stability(
                adata_dict, ref_keys, n_genes=100, seed=42
            )
            check(
                "Moran's I stable across alpha (r > 0.75)",
                moran_check["pass"],
                f"mean corr alpha=0 vs alpha=1.5: {moran_check.get('mean_corr', 0):.4f}",
            )

    passed = sum(1 for c in report["checks"] if c["passed"])
    total = len(report["checks"])
    print(f"\n=== {passed}/{total} checks passed ===")

    with open(output_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def _check_linearity(x: np.ndarray, y: np.ndarray) -> bool:
    if len(x) < 3:
        return True
    return abs(np.corrcoef(x, y)[0, 1]) > 0.95


def _check_moran_stability(adata_dict: dict, keys: list, n_genes: int = 100,
                           seed: int = 42) -> dict:
    """Check that Moran's I is preserved across alpha levels."""
    from scipy.sparse import issparse

    def _dense(x):
        return x.toarray() if issparse(x) else np.asarray(x, dtype=np.float64)

    # Compute spatial weight matrix from coordinates (k-NN)
    from sklearn.neighbors import NearestNeighbors
    coords = adata_dict[keys[0]].obsm["spatial"]
    nn = NearestNeighbors(n_neighbors=6, metric="euclidean").fit(coords)
    adj = nn.kneighbors_graph(coords, mode="connectivity").toarray()
    # Symmetrize
    adj = (adj + adj.T) > 0
    W = adj.astype(np.float64)
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1
    W = W / row_sums[:, None]

    rng = np.random.default_rng(seed)
    n_total = adata_dict[keys[0]].n_vars
    gene_idx = rng.choice(n_total, min(n_genes, n_total), replace=False)

    def moran_i(x):
        x = x - x.mean()
        n = len(x)
        num = n * np.sum(W * np.outer(x, x))
        denom = np.sum(W) * np.sum(x ** 2)
        return num / denom if denom > 1e-15 else 0.0

    moran_by_alpha = {}
    for k in [keys[0], keys[-1]]:
        X = _dense(adata_dict[k].X)
        moran_by_alpha[k] = np.array([moran_i(X[:, g]) for g in gene_idx])

    corr = np.corrcoef(moran_by_alpha[keys[0]], moran_by_alpha[keys[-1]])[0, 1]
    return {"pass": corr > 0.75, "mean_corr": float(corr)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate batch-effect ladder")
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    evaluate(Path(args.output_dir))


if __name__ == "__main__":
    main()
