#!/usr/bin/env python3
"""Ablation test: distributional vs legacy alteration with new parameterization.

Tests mean (with fano coupling), variance (level + heterogeneity), and
sparsity (logit shift δ_z) on a 2000-gene subset for speed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

FEAST_ROOT = Path("/maiziezhou_lab2/yiru/FEAST")
sys.path.insert(0, str(FEAST_ROOT / "src"))

from FEAST import GeneParameterSimulator
from FEAST.FEAST_core.parameter_cloud import (
    convert_params_for_new_simulator,
    calculate_fold_change,
)
from FEAST import Alteration

DATA = Path("/maiziezhou_lab2/yiru/Datasets/Processed/spatialLIBD_DLPFC_Visium/h5ad/151670.h5ad")


def run_single(adata, alt_config, use_dist, seed=42):
    sim = GeneParameterSimulator()
    sim.hybrid_alpha = 0.2
    sim.fit(adata, visualize_fits=False)

    t0 = time.time()
    table, diag = sim.build_gene_parameter_table(
        alteration_config=alt_config,
        simulation_mode="generative",
        assignment_weights={"mean": 3, "variance": 1, "zero_prop": 1.0},
        random_seed=seed,
        assignment_method="hybrid",
        verbose=False,
        use_distributional_alteration=use_dist,
    )
    elapsed = time.time() - t0

    model_params = convert_params_for_new_simulator(table, n_spots=adata.n_obs)
    model_counts = pd.Series(model_params["model_selected"]).value_counts().to_dict()
    var_inflated = diag.get("count_feasibility", {}).get("variance_inflated_gene_count", 0)
    target_fc = diag.get("target_stage_achieved_change", {})

    # Compute per-gene-level realized stats from original
    orig_matrix = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    orig_stats = pd.DataFrame({
        "mean": np.mean(orig_matrix, axis=0),
        "variance": np.var(orig_matrix, axis=0),
        "zero_prop": 1 - (np.count_nonzero(orig_matrix, axis=0) / adata.n_obs),
    }, index=adata.var_names).clip(lower=1e-10)
    realized_fc = calculate_fold_change(orig_stats, table)

    return {
        "elapsed_s": round(elapsed, 1),
        "n_genes": len(table),
        **{f"fc_{k}": realized_fc.get(k) for k in ["mean", "variance", "zero_prop"]},
        "target_mean_fc": target_fc.get("mean"),
        "target_variance_fc": target_fc.get("variance"),
        "target_zero_prop_fc": target_fc.get("zero_prop"),
        "model_counts": model_counts,
        "var_inflated": var_inflated,
        "alteration_path": diag.get("alteration_path", "unknown"),
    }


def main():
    adata_full = ad.read_h5ad(str(DATA))
    rng = np.random.default_rng(42)
    gene_idx = rng.choice(adata_full.n_vars, 2000, replace=False)
    adata = adata_full[:, gene_idx].copy()
    print(f"Loaded subset: {adata.n_obs} spots × {adata.n_vars} genes\n")

    conditions = [
        ("baseline", Alteration()),
        ("mean_2x", Alteration.mean_only(2.0, variance_coupling="fano")),
        ("mean_0.5x", Alteration.mean_only(0.5, variance_coupling="fano")),
        ("variance_2x", Alteration.variance_only(2.0)),
        ("var_hetero_1.5", Alteration.variance_heterogeneity(1.5)),
        ("sparsity_dz+1", Alteration.sparsity_logit(1.0)),
        ("sparsity_dz-1", Alteration.sparsity_logit(-1.0)),
    ]

    print(f"{'Condition':<22s} {'Path':<15s} {'Time':>6s}  "
          f"{'mean_fc':>8s} {'var_fc':>8s} {'zp_fc':>8s}  "
          f"{'var_infl':>8s} {'models':>35s}")
    print("-" * 120)

    results = []
    for cond_name, alt_config in conditions:
        for path, use_dist in [("distributional", True), ("legacy", False)]:
            try:
                r = run_single(adata, alt_config, use_dist)
                results.append({"condition": cond_name, "path": path, **r})
                mc = r["model_counts"]
                mc_str = ", ".join(f"{k}:{v}" for k, v in sorted(mc.items()))
                print(f"{cond_name:<22s} {path:<15s} {r['elapsed_s']:>5.1f}s  "
                      f"{r['fc_mean']:>8.4f} {r['fc_variance']:>8.4f} {r['fc_zero_prop']:>8.4f}  "
                      f"{r['var_inflated']:>8d} {mc_str:<35s}")
            except Exception as e:
                print(f"{cond_name:<22s} {path:<15s} FAILED: {e}")

    # --- Normalized analysis ---
    print("\n" + "=" * 100)
    print("NORMALIZED RESULTS (baseline-corrected)")
    print("=" * 100)
    df = pd.DataFrame(results)
    baseline = df[df["condition"] == "baseline"]
    if len(baseline) > 0:
        b_dist = baseline[baseline["path"] == "distributional"].iloc[0]
        b_leg = baseline[baseline["path"] == "legacy"].iloc[0]
        for cond in df["condition"].unique():
            if cond == "baseline":
                continue
            sub = df[df["condition"] == cond]
            for _, row in sub.iterrows():
                b = b_dist if row["path"] == "distributional" else b_leg
                norm_mean = row["fc_mean"] / b["fc_mean"] if b["fc_mean"] and b["fc_mean"] != 0 else float("nan")
                norm_var = row["fc_variance"] / b["fc_variance"] if b["fc_variance"] and b["fc_variance"] != 0 else float("nan")
                norm_zp = row["fc_zero_prop"] / b["fc_zero_prop"] if b["fc_zero_prop"] and b["fc_zero_prop"] != 0 else float("nan")
                print(f"  {cond:<20s} {row['path']:<15s}  "
                      f"mean_norm={norm_mean:>7.4f}  var_norm={norm_var:>7.4f}  zp_norm={norm_zp:>7.4f}  "
                      f"var_infl={row['var_inflated']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
