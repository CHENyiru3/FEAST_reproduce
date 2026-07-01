#!/usr/bin/env python3
"""Quick test: FEAST_Rank on DLPFC_151670 with theoretical-best (exact) settings.

Compares input_mean_corr and input_variance_corr against the target values:
  target_mean_corr  = 0.999318825
  target_var_corr   = 0.9874277194
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).resolve().parent
FEAST_ENV = "/maiziezhou_lab2/yiru/envs/feast-py311-conda"
PYTHON = f"{FEAST_ENV}/bin/python"

INPUT_PATH = SCRIPT_DIR / "exper_data" / "DLPFC_151670.h5ad"
OUTPUT_PATH = SCRIPT_DIR / "outputs" / "test_best_settings" / "FEAST_Rank" / "DLPFC_151670.h5ad"

TARGET_MEAN_CORR = 0.999318825
TARGET_VAR_CORR = 0.9874277194


def compute_quick_metrics(ref_path: Path, sim_path: Path) -> dict:
    """Compute input_mean_corr and input_variance_corr."""
    ref = ad.read_h5ad(str(ref_path))
    sim = ad.read_h5ad(str(sim_path))

    # Align genes
    common_genes = ref.var_names.intersection(sim.var_names)
    if len(common_genes) < len(ref.var_names):
        print(f"  WARNING: {len(ref.var_names) - len(common_genes)} genes missing in simulated")
    ref = ref[:, common_genes]
    sim = sim[:, common_genes]

    ref_X = ref.X.toarray() if hasattr(ref.X, "toarray") else np.asarray(ref.X)
    sim_X = sim.X.toarray() if hasattr(sim.X, "toarray") else np.asarray(sim.X)

    ref_log = np.log1p(ref_X)
    sim_log = np.log1p(sim_X)

    ref_means = np.mean(ref_log, axis=0)
    sim_means = np.mean(sim_log, axis=0)
    mean_corr, _ = pearsonr(ref_means, sim_means)

    ref_vars = np.var(ref_log, axis=0)
    sim_vars = np.var(sim_log, axis=0)
    var_corr, _ = pearsonr(ref_vars, sim_vars)

    return {
        "input_mean_corr": mean_corr,
        "input_variance_corr": var_corr,
    }


def main():
    print("=" * 70)
    print("FEAST_Rank on DLPFC_151670 — Theoretical Best Settings Test")
    print("=" * 70)
    print(f"  Input:  {INPUT_PATH}")
    print(f"  Output: {OUTPUT_PATH}")
    print()

    # --- Stage 1: Run FEAST simulation ---
    print("--- Stage 1: Running FEAST simulation (exact-mode) ---")
    print()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON,
        str(SCRIPT_DIR / "scripts" / "ParameterCloud_sim.py"),
        "--input", str(INPUT_PATH),
        "--output", str(OUTPUT_PATH),
        "--parameter-mode", "hungarian",
        "--spatial-mode", "reference_rank",
        "--assignment-method", "hybrid",
        "--annotation-key", "auto",
        "--seed", "2026",
        # --- Theoretical best (exact) settings ---
        "--ppf-method", "exact",
        "--beta-n-jobs", "1",
        "--beta-early-stopping-patience", "0",
        "--assignment-solver", "scipy",
        "--no-assignment-blocks",
        "--convert-n-jobs", "1",
    ]

    print(f"  Command: {' '.join(cmd)}")
    print()

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"FAILED with return code {result.returncode}")
        return 1

    print(f"  Simulation completed in {elapsed:.1f}s")
    print()

    # --- Stage 2: Read metadata from output ---
    print("--- Stage 2: Output metadata ---")
    sim_adata = ad.read_h5ad(str(OUTPUT_PATH))
    if "figure2_simulator_benchmark" in sim_adata.uns:
        meta = dict(sim_adata.uns["figure2_simulator_benchmark"])
        for k in ["ppf_method", "beta_early_stopping_patience", "assignment_blocks",
                   "beta_n_jobs", "convert_n_jobs", "elapsed_seconds", "feast_version"]:
            print(f"  {k}: {meta.get(k, 'N/A')}")
    print()

    # --- Stage 3: Compute quick metrics ---
    print("--- Stage 3: Quick quality metrics ---")
    metrics = compute_quick_metrics(INPUT_PATH, OUTPUT_PATH)

    print(f"  input_mean_corr:    {metrics['input_mean_corr']:.10f}")
    print(f"  input_variance_corr: {metrics['input_variance_corr']:.10f}")
    print()
    print(f"  Target input_mean_corr:    {TARGET_MEAN_CORR:.10f}")
    print(f"  Target input_variance_corr: {TARGET_VAR_CORR:.10f}")
    print()
    print(f"  Delta mean_corr:  {metrics['input_mean_corr'] - TARGET_MEAN_CORR:+.6e}")
    print(f"  Delta var_corr:   {metrics['input_variance_corr'] - TARGET_VAR_CORR:+.6e}")
    print()

    # Identity check
    identity = (
        metrics["input_mean_corr"] >= 0.995
        and metrics["input_variance_corr"] >= 0.95
    )
    print(f"  Meets identity thresholds: {identity}")
    print(f"  (mean_corr >= 0.995: {metrics['input_mean_corr'] >= 0.995})")
    print(f"  (var_corr >= 0.95: {metrics['input_variance_corr'] >= 0.95})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
