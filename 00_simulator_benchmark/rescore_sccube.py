#!/usr/bin/env python3
"""Re-score scCube samples after spatial coordinate recovery."""
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import pair_metrics

DATA_ROOT = Path("data/local")
REF_ROOT = DATA_ROOT / "reference"
SCUBE_ROOT = DATA_ROOT / "external/scCube"

config = yaml.safe_load(Path("config.yaml").read_text())
metrics_csv = Path("outputs/final_rerun_20260718_metrics_v2/simulator_quality_metrics.csv")

df = pd.read_csv(metrics_csv)
for sample in config["samples"]:
    ref_path = REF_ROOT / f"{sample}.h5ad"
    scube_path = SCUBE_ROOT / f"{sample}.h5ad"
    if not scube_path.exists():
        print(f"SKIP {sample}: no scCube file")
        continue

    ref = ad.read_h5ad(ref_path)
    sim = ad.read_h5ad(scube_path)
    metrics, _ = pair_metrics(
        ref, sim,
        max_genes=int(config["metrics"]["max_moran_genes"]),
        neighbors=int(config["metrics"]["spatial_neighbors"]),
        coordinate_decimals=int(config["metrics"].get("coordinate_pairing_decimals", 8)),
    )
    mask = (df["simulator"] == "scCube") & (df["sample"] == sample)
    if mask.sum() != 1:
        print(f"SKIP {sample}: found {mask.sum()} scCube rows (expected 1)")
        continue
    for key, val in metrics.items():
        df.loc[mask, key] = val
    print(f"{sample}: paired={metrics['n_paired_spots']}, jaccard={metrics['zero_mask_jaccard']}, pairing={metrics['spot_pairing']}")

df.to_csv(metrics_csv, index=False)
print(f"\nUpdated {metrics_csv}")
