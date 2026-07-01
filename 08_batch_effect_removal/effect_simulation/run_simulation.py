#!/usr/bin/env python
"""Generate batch-effect ladder from a clean reference slice.

Estimates deformation (D, b) from real paired slices via
FEAST.characterize_batch(), then simulates at multiple alpha severities.

Usage:
    python run_simulation.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

FEAST_ROOT = Path("/maiziezhou_lab2/yiru/FEAST")
sys.path.insert(0, str(FEAST_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from effect_simulation.batch_simulator import (
    DeformationConfig,
    build_deformation_modes,
    simulate_batch_ladder,
)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def run(config: dict) -> Path:
    import scanpy as sc

    out_root = Path(config.get("output_dir", "outputs"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _ensure_dir(out_root / timestamp)

    data_dir = Path(config["data_dir"])
    ref_slice = config["reference_slice"]
    query_slice = config["query_slice"]
    alphas = config.get("alpha_levels", [0.0, 0.25, 0.5, 0.75, 1.0])
    seed = config.get("random_seed", 42)
    modes = config.get("modes", ["shift_only", "diagonal_affine"])

    print(f"Reference: {ref_slice}")
    print(f"Query:     {query_slice}")
    print(f"Modes:     {modes}")
    print(f"Output:    {out_dir}")
    print(f"Alphas:    {alphas}")

    adata_ref = sc.read_h5ad(data_dir / f"{ref_slice}.h5ad")
    adata_qry = sc.read_h5ad(data_dir / f"{query_slice}.h5ad")
    print(f"Loaded ref {ref_slice}: {adata_ref.n_obs} spots x {adata_ref.n_vars} genes")
    print(f"Loaded qry {query_slice}: {adata_qry.n_obs} spots x {adata_qry.n_vars} genes")

    # Estimate deformations from real data
    deformations = build_deformation_modes(adata_ref, adata_qry, modes=modes)
    for name, d in deformations.items():
        print(f"  {name}: D={d.D.round(4).tolist()}, b={d.b.round(4).tolist()}")

    manifest_rows = []

    for mode_name, deform in deformations.items():
        mode_dir = _ensure_dir(out_dir / mode_name)
        print(f"\n--- {mode_name}: D={deform.D}, b={deform.b} ---")

        results = simulate_batch_ladder(
            adata_ref, deform, alphas=alphas, random_seed=seed
        )

        for alpha, sim in results.items():
            fname = f"alpha_{alpha:.2f}.h5ad"
            path = mode_dir / fname
            sim.write_h5ad(path)

            X = sim.X.toarray() if hasattr(sim.X, "toarray") else np.asarray(sim.X, dtype=np.float64)
            manifest_rows.append({
                "mode": mode_name,
                "alpha": alpha,
                "file": str(path),
                "n_spots": sim.n_obs,
                "n_genes": sim.n_vars,
                "total_counts_mean": float(X.sum(1).mean()),
                "zero_fraction": float(1.0 - np.count_nonzero(X) / X.size),
                "mean_of_means": float(X.mean()),
            })
            print(f"  alpha={alpha:.2f}: counts_mean={X.sum(1).mean():.0f}, zero_frac={1-np.count_nonzero(X)/X.size:.4f}")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)

    print(f"\nDone. {len(manifest_rows)} slices in {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="FEAST Batch Effect Ladder Simulation")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path

    with open(config_path) as f:
        config = yaml.safe_load(f)

    run(config)


if __name__ == "__main__":
    main()
