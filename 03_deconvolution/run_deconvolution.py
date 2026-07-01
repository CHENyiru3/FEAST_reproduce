#!/usr/bin/env python3
"""Generate FEAST deconvolution simulation data for Figure 2 benchmark.

Usage:
    conda run -p /maiziezhou_lab2/yiru/envs/feast-py311-conda \\
    python scripts/run_deconvolution.py \\
      --input-dir data \\
      --slices 007,050,100 \\
      --output-dir outputs/simulations \\
      --resolutions 0.1 0.25 \\
      --cell-type-key cell_type \\
      --seed 2026 \\
      --config configs/deconvolution.yaml
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

CELL_TYPE_FALLBACKS = ["CellType", "cell_type", "annotation", "ground_truth"]


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


def _resolve_cell_type_key(adata_obj: ad.AnnData, requested: str) -> str | None:
    if requested in adata_obj.obs.columns:
        return requested
    for key in CELL_TYPE_FALLBACKS:
        if key in adata_obj.obs.columns:
            print(f"Note: '{requested}' not found, using '{key}' as cell-type key")
            return key
    return None


def run_one_resolution(
    adata_obj: ad.AnnData,
    resolution: float,
    cell_type_key: str,
    seed: int,
    output_path: Path,
    truth_dir: Path,
    simulation_kwargs: dict | None = None,
) -> dict:
    from FEAST.deconvolution import simulate_deconvolution_from_single_cells
    from FEAST import __version__ as feast_version

    t0 = time.time()

    simulated = simulate_deconvolution_from_single_cells(
        reference_adata=adata_obj,
        cell_type_key=cell_type_key,
        downsampling_factor=resolution,
        grid_type="hexagonal",
        alpha=0.01,
        random_seed=seed,
        verbose=True,
        **(simulation_kwargs or {}),
    )

    elapsed = time.time() - t0
    git_info = _get_git_info()

    simulated.uns["figure2_deconvolution"] = {
        "feast_version": feast_version,
        "git_commit": git_info["commit"],
        "git_branch": git_info["branch"],
        "git_dirty": git_info["dirty"],
        "resolution": resolution,
        "downsampling_factor": resolution,
        "grid_type": "hexagonal",
        "alpha": 0.01,
        "cell_type_key": cell_type_key,
        "random_seed": seed,
        "parameter_mode": "hungarian",
        "spatial_mode": "reference_rank",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = simulated.X
    if hasattr(counts, "toarray"):
        counts = counts.toarray()
    rounded = np.round(counts).astype(np.int64)
    if not np.allclose(counts, rounded):
        print(f"WARNING: counts at resolution {resolution} are not integer-compatible")
    simulated.X = rounded
    simulated.write_h5ad(str(output_path), compression="gzip")

    truth_path = truth_dir / f"resolution_{resolution}_proportions.csv"
    if "cell_type_proportions" in simulated.obsm:
        proportions = simulated.obsm["cell_type_proportions"]
        cell_type_names = simulated.uns.get(
            "cell_type_names",
            [f"CT_{i}" for i in range(proportions.shape[1])],
        )
        prop_df = pd.DataFrame(
            proportions,
            index=[f"spot_{i}" for i in range(proportions.shape[0])],
            columns=cell_type_names,
        )
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        prop_df.to_csv(truth_path)
    else:
        raise RuntimeError(
            "simulate_deconvolution_from_single_cells did not produce "
            "cell_type_proportions in obsm"
        )

    return {
        "resolution": resolution,
        "n_spots": simulated.n_obs,
        "n_genes": simulated.n_vars,
        "n_cell_types": len(cell_type_names),
        "elapsed_seconds": round(elapsed, 2),
        "feast_version": feast_version,
        "git_commit": git_info["commit"],
        "status": "ok",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data"),
                        help="Directory with reference .h5ad files (default: data)")
    parser.add_argument("--slices", type=str, default="007,050,100",
                        help="Comma-separated slice IDs (default: 007,050,100)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/simulations"),
                        help="Output directory for simulated .h5ad files")
    parser.add_argument("--resolutions", type=float, nargs="+",
                        default=[0.1, 0.25])
    parser.add_argument("--cell-type-key", type=str, default="cell_type")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/deconvolution.yaml"),
                        help="YAML config file")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing outputs")
    parser.add_argument("--ppf-method", choices=["interp", "exact"], default=None)
    parser.add_argument("--beta-n-jobs", type=int, default=4)
    parser.add_argument("--beta-early-stopping-patience", type=int, default=2)
    parser.add_argument("--assignment-solver", choices=["scipy", "lapjv"], default="scipy")
    parser.add_argument("--assignment-blocks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--assignment-block-size", type=int, default=None)
    parser.add_argument("--assignment-block-multiplier", type=int, default=8)
    parser.add_argument("--convert-n-jobs", type=int, default=4)
    args = parser.parse_args()

    config = {}
    if args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    slice_ids = [s.strip() for s in args.slices.split(",")]
    resolutions = args.resolutions or config.get("resolutions", [0.1, 0.25])
    seed = config.get("seed", args.seed)
    cell_type_key_cfg = config.get("cell_type_key", args.cell_type_key)
    simulation_kwargs = {
        "parameter_mode": config.get("parameter_mode", "hungarian"),
        "spatial_mode": config.get("spatial_mode", "reference_rank"),
        "ppf_method": config.get("ppf_method", args.ppf_method or "interp"),
        "beta_n_jobs": int(config.get("beta_n_jobs", args.beta_n_jobs)),
        "beta_early_stopping_patience": int(config.get(
            "beta_early_stopping_patience",
            args.beta_early_stopping_patience,
        )),
        "assignment_solver": config.get("assignment_solver", args.assignment_solver),
        "assignment_blocks": config.get(
            "assignment_blocks",
            True if args.assignment_blocks is None else args.assignment_blocks,
        ),
        "assignment_block_size": config.get("assignment_block_size", args.assignment_block_size),
        "assignment_block_multiplier": int(config.get(
            "assignment_block_multiplier",
            args.assignment_block_multiplier,
        )),
        "convert_n_jobs": int(config.get("convert_n_jobs", args.convert_n_jobs)),
    }

    truth_root = args.output_dir.parent / "ground_truth"
    total = len(slice_ids) * len(resolutions)
    manifest_rows = []
    done_count = 0

    print(f"Deconvolution simulation: {len(slice_ids)} slices x {len(resolutions)} resolutions = {total} jobs")
    print(f"  seed: {seed}")
    print(f"  cell_type_key: {cell_type_key_cfg}")
    print("  FEAST knobs:")
    for key, value in simulation_kwargs.items():
        print(f"    {key}: {value}")
    print()

    for slice_id in slice_ids:
        input_path = args.input_dir / f"Zhuang-ABCA-1.{slice_id}.h5ad"
        if not input_path.exists():
            print(f"SKIP {slice_id}: input not found at {input_path}")
            for res in resolutions:
                manifest_rows.append({
                    "slice_id": slice_id, "resolution": res,
                    "status": f"failed: input not found",
                })
            done_count += len(resolutions)
            continue

        adata_obj = ad.read_h5ad(str(input_path))
        adata_obj.var_names_make_unique()
        print(f"[{slice_id}] loaded: {adata_obj.n_obs} spots x {adata_obj.n_vars} genes")

        ct_key = _resolve_cell_type_key(adata_obj, cell_type_key_cfg)
        if ct_key is None:
            print(f"ERROR: no cell-type column found in {slice_id}", file=sys.stderr)
            for res in resolutions:
                manifest_rows.append({
                    "slice_id": slice_id, "resolution": res,
                    "status": "failed: no cell-type column",
                })
            done_count += len(resolutions)
            continue
        print(f"  cell_type_key resolved to: '{ct_key}' "
              f"({adata_obj.obs[ct_key].nunique()} unique types)")

        for resolution in resolutions:
            out_path = args.output_dir / slice_id / f"resolution_{resolution}.h5ad"
            truth_dir = truth_root / slice_id

            done_count += 1
            if out_path.exists() and not args.overwrite:
                truth_path = truth_dir / f"resolution_{resolution}_proportions.csv"
                if truth_path.exists():
                    print(f"  [{done_count}/{total}] SKIP {slice_id}/resolution_{resolution} — exists")
                    manifest_rows.append({
                        "slice_id": slice_id, "resolution": resolution,
                        "file_path": str(out_path),
                        "truth_path": str(truth_path),
                        "status": "skipped",
                    })
                    continue

            print(f"  [{done_count}/{total}] RUN  {slice_id}/resolution_{resolution}")

            try:
                result = run_one_resolution(
                    adata_obj, resolution, ct_key, seed,
                    out_path, truth_dir,
                    simulation_kwargs=simulation_kwargs,
                )
                truth_path = truth_dir / f"resolution_{resolution}_proportions.csv"
                manifest_rows.append({
                    "slice_id": slice_id,
                    "resolution": resolution,
                    "file_path": str(out_path),
                    "truth_path": str(truth_path),
                    "n_spots": result["n_spots"],
                    "n_genes": result["n_genes"],
                    "n_cell_types": result["n_cell_types"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "feast_version": result["feast_version"],
                    "git_commit": result["git_commit"],
                    "status": "ok",
                })
                print(f"      -> {result['n_spots']} spots x {result['n_genes']} genes "
                      f"({result['n_cell_types']} cell types) in {result['elapsed_seconds']:.1f}s")
            except Exception as e:
                print(f"      -> FAILED: {e}", file=sys.stderr)
                manifest_rows.append({
                    "slice_id": slice_id,
                    "resolution": resolution,
                    "file_path": str(out_path),
                    "status": f"failed: {e}",
                })

    manifest_path = args.output_dir.parent / "simulation_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    ok_count = sum(1 for r in manifest_rows if r["status"] == "ok")
    skip_count = sum(1 for r in manifest_rows if r["status"] == "skipped")
    fail_count = sum(1 for r in manifest_rows if r["status"].startswith("failed"))
    print(f"\nDone: {ok_count} ok, {skip_count} skipped, {fail_count} failed")
    print(f"Manifest: {manifest_path}")

    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
