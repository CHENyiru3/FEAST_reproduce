#!/usr/bin/env python3
"""Smoke test: prefit copula once, run baseline + mean_1.25 alterations."""
import sys, time
from pathlib import Path

FEAST_ROOT = Path("/maiziezhou_lab2/yiru/FEAST")
sys.path.insert(0, str(FEAST_ROOT / "src"))

import anndata as ad
import numpy as np
import yaml

from run_simulation import build_alterations, PrefitSimulator, run_simulation

with open("../configs/clustering.yaml") as f:
    config = yaml.safe_load(f)

slice_id = "151670"
adata = ad.read_h5ad(f"data/{slice_id}.h5ad")
print(f"Loaded: {adata.n_obs} spots x {adata.n_vars} genes", flush=True)

# Subset for speed
rng = np.random.default_rng(42)
gene_idx = rng.choice(adata.n_vars, 5000, replace=False)
adata = adata[:, gene_idx].copy()
print(f"Subset to {adata.n_vars} genes", flush=True)

alterations = build_alterations(config)
test_alts = [a for a in alterations if a["simulation_id"] in ("baseline", "mean_1_25")]
print(f"Testing {len(test_alts)} alterations with prefit caching...", flush=True)

# Pre-fit once
prefit = PrefitSimulator(adata, seed=2026)

for alt in test_alts:
    out = Path("outputs") / "smoke_test" / slice_id / f"{alt['simulation_id']}.h5ad"
    result = run_simulation(
        adata=adata, alteration=alt, annotation_key="ground_truth",
        seed=2026, output_path=out,
        parameter_mode="hungarian", use_distributional_alteration=True,
        prefit_sim=prefit,
    )
    print(f"  {alt['simulation_id']}: {result['n_spots']}x{result['n_genes']} in {result['elapsed_seconds']:.1f}s", flush=True)

print("Smoke test passed!", flush=True)
