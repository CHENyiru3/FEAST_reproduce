#!/usr/bin/env python3
"""Generate FEAST hungarian x ot_spatial simulations for clustering benchmark.

Usage:
    NUMBA_DISABLE_JIT=1 MPLCONFIGDIR=/tmp/matplotlib \
    conda run -p /maiziezhou_lab2/yiru/envs/feast-py311-conda \
    python 03_clustering/run_simulation_ot.py \
      --input-dir data \
      --slices 151508,151670,151676 \
      --output-dir outputs/03_clustering_ot/simulations \
      --parameter-mode hungarian \
      --spatial-mode ot_spatial \
      --seed 2026 \
      --config configs/clustering_ot.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

FEAST_ROOT = Path("/maiziezhou_lab2/yiru/FEAST")
sys.path.insert(0, str(FEAST_ROOT / "src"))


def _get_git_info() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=FEAST_ROOT, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=FEAST_ROOT, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=FEAST_ROOT, text=True
        ).strip()
        return {"commit": commit, "branch": branch, "dirty": bool(dirty)}
    except Exception:
        return {"commit": "unknown", "branch": "unknown", "dirty": None}


def build_alterations(config: dict) -> list[dict]:
    """Build the list of (alteration_id, Alteration kwargs) from config."""
    from FEAST import Alteration

    alterations = []
    strength = config.get("dispersion_strength", 0.2)
    alt_cfg = config["alterations"]

    for level in alt_cfg["mean"]["levels"]:
        alterations.append({
            "simulation_id": f"mean_{level:.2f}".replace(".", "_").replace("_00", ""),
            "alteration_type": "mean",
            "fold_change": level,
            "config": Alteration.mean_only(fold_change=level),
        })

    for level in alt_cfg["variance"]["levels"]:
        alterations.append({
            "simulation_id": f"variance_{level:.2f}".replace(".", "_").replace("_00", ""),
            "alteration_type": "variance",
            "fold_change": level,
            "config": Alteration.variance_only(fold_change=level, dispersion=strength),
        })

    for level in alt_cfg["sparsity"]["levels"]:
        alterations.append({
            "simulation_id": f"sparsity_{level:.2f}".replace(".", "_").replace("_00", ""),
            "alteration_type": "sparsity_logit_shift",
            "fold_change": level,
            "config": Alteration.sparsity_logit(delta=float(np.log(level))),
        })

    # baseline
    alterations.append({
        "simulation_id": "baseline",
        "alteration_type": "baseline",
        "fold_change": 1.0,
        "config": Alteration(),
    })

    return alterations


def run_simulation(
    adata: ad.AnnData,
    alteration: dict,
    annotation_key: str,
    seed: int,
    output_path: Path,
) -> dict:
    from FEAST import __version__ as feast_version
    from FEAST import simulate

    start_time = time.time()

    simulated = simulate(
        adata=adata,
        annotation_key=annotation_key,
        parameter_mode="hungarian",
        spatial_mode="ot_spatial",
        target_adata=adata,
        alteration=alteration["config"],
        seed=seed,
        verbose=True,
        clip_overshoot_factor=0.0,
        boundary_multiplier=1.1,
        use_heuristic_search=False,
    )

    elapsed = time.time() - start_time
    git_info = _get_git_info()

    simulated.uns["figure2_clustering"] = {
        "feast_version": feast_version,
        "git_commit": git_info["commit"],
        "git_branch": git_info["branch"],
        "git_dirty": git_info["dirty"],
        "parameter_mode": "hungarian",
        "spatial_mode": "ot_spatial",
        "alteration_type": alteration["alteration_type"],
        "fold_change": alteration["fold_change"],
        "random_seed": seed,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    simulated.write_h5ad(str(output_path), compression="gzip")

    return {
        "n_spots": simulated.n_obs,
        "n_genes": simulated.n_vars,
        "elapsed_seconds": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data"),
                        help="Directory with reference .h5ad files (default: data)")
    parser.add_argument("--slices", type=str, default="151508,151670,151676",
                        help="Comma-separated slice IDs (default: 151508,151670,151676)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/03_clustering_ot/simulations"),
                        help="Output directory (default: outputs/03_clustering_ot/simulations)")
    parser.add_argument("--parameter-mode", type=str, default="hungarian",
                        choices=["hungarian", "reference_stats"])
    parser.add_argument("--spatial-mode", type=str, default="ot_spatial",
                        choices=["ot_spatial", "reference_rank", "bin_dens", "categorical_adjacency", "identity"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--config", type=Path, default=Path("configs/clustering_ot.yaml"),
                        help="YAML config file")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    slice_ids = [s.strip() for s in args.slices.split(",")]
    alterations = build_alterations(config)
    seed = config.get("seed", args.seed)
    annotation_key = config.get("annotation_key", "ground_truth")

    total = len(slice_ids) * len(alterations)
    manifest_rows = []
    done_count = 0

    print(f"Clustering simulation (hungarian x ot_spatial): {len(slice_ids)} slices x {len(alterations)} alterations = {total} jobs")
    print(f"  parameter_mode: {args.parameter_mode}")
    print(f"  spatial_mode: {args.spatial_mode}")
    print(f"  seed: {seed}")
    print(f"  annotation_key: {annotation_key}")
    print()

    for slice_id in slice_ids:
        input_path = args.input_dir / f"{slice_id}.h5ad"
        if not input_path.exists():
            print(f"SKIP {slice_id}: input not found at {input_path}")
            continue

        adata = ad.read_h5ad(str(input_path))
        print(f"[{slice_id}] loaded: {adata.n_obs} spots x {adata.n_vars} genes")

        for alt in alterations:
            sim_id = alt["simulation_id"]
            output_path = args.output_dir / slice_id / f"{sim_id}.h5ad"

            if output_path.exists() and not args.overwrite:
                print(f"  [{done_count + 1}/{total}] SKIP {slice_id}/{sim_id} — exists")
                manifest_rows.append({
                    "slice_id": slice_id,
                    "simulation_id": sim_id,
                    "alteration_type": alt["alteration_type"],
                    "fold_change": alt["fold_change"],
                    "file_path": str(output_path),
                    "n_spots": None,
                    "n_genes": None,
                    "elapsed_seconds": None,
                    "status": "skipped",
                })
                done_count += 1
                continue

            print(f"  [{done_count + 1}/{total}] RUN  {slice_id}/{sim_id} "
                  f"({alt['alteration_type']}={alt['fold_change']})")

            try:
                result = run_simulation(
                    adata=adata,
                    alteration=alt,
                    annotation_key=annotation_key,
                    seed=seed,
                    output_path=output_path,
                )
                manifest_rows.append({
                    "slice_id": slice_id,
                    "simulation_id": sim_id,
                    "alteration_type": alt["alteration_type"],
                    "fold_change": alt["fold_change"],
                    "file_path": str(output_path),
                    "n_spots": result["n_spots"],
                    "n_genes": result["n_genes"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "status": "ok",
                })
                print(f"      -> {result['n_spots']}x{result['n_genes']} in {result['elapsed_seconds']:.1f}s")
            except Exception as e:
                print(f"      -> FAILED: {e}", file=sys.stderr)
                manifest_rows.append({
                    "slice_id": slice_id,
                    "simulation_id": sim_id,
                    "alteration_type": alt["alteration_type"],
                    "fold_change": alt["fold_change"],
                    "file_path": str(output_path),
                    "n_spots": None,
                    "n_genes": None,
                    "elapsed_seconds": None,
                    "status": f"failed: {e}",
                })

            done_count += 1

    manifest_path = args.output_dir / "simulation_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    ok_count = sum(1 for r in manifest_rows if r["status"] == "ok")
    skip_count = sum(1 for r in manifest_rows if r["status"] == "skipped")
    fail_count = sum(1 for r in manifest_rows if r["status"].startswith("failed"))
    print(f"\nDone: {ok_count} ok, {skip_count} skipped, {fail_count} failed")
    print(f"Manifest: {manifest_path}")

    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
