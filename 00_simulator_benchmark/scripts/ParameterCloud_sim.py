#!/usr/bin/env python3
"""Run FEAST single-slice simulation for the Figure 2 simulator benchmark.

Usage:
    NUMBA_DISABLE_JIT=1 MPLCONFIGDIR=/tmp/matplotlib \\
    python scripts/ParameterCloud_sim.py \\
      --input exper_data/DLPFC_151670.h5ad \\
      --output outputs/simulation_saved/FEAST_Rank/DLPFC_151670.h5ad \\
      --parameter-mode hungarian \\
      --spatial-mode reference_rank \\
      --assignment-method hybrid \\
      --seed 2026
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np

FEAST_ROOT = Path("/maiziezhou_lab2/yiru/FEAST")

LABEL_COLUMNS = ["ground_truth", "cell_type", "annotation", "region", "cluster"]

ASSIGNMENT_METHODS = ("hybrid", "copula_rank")


def _resolve_annotation_key(adata: ad.AnnData) -> str | None:
    for col in LABEL_COLUMNS:
        if col in adata.obs.columns:
            return col
    return None


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


def run_simulation(
    input_path: Path,
    output_path: Path,
    parameter_mode: str,
    spatial_mode: str,
    assignment_method: str,
    annotation_key: str | None,
    seed: int,
    target_path: str | None = None,
    ppf_method: str = "interp",
    beta_n_jobs: int = 4,
    beta_early_stopping_patience: int = 2,
    assignment_solver: str = "scipy",
    assignment_blocks: bool = True,
    assignment_block_size: int | None = None,
    assignment_block_multiplier: int = 8,
    convert_n_jobs: int = 4,
) -> int:
    sys.path.insert(0, str(FEAST_ROOT / "src"))
    from FEAST import __version__ as feast_version
    from FEAST import simulate

    adata = ad.read_h5ad(str(input_path))

    if annotation_key == "auto":
        annotation_key = _resolve_annotation_key(adata)

    # Resolve target_adata for ot_spatial mode
    target_adata = None
    if spatial_mode == "ot_spatial":
        if target_path:
            target_adata = ad.read_h5ad(str(target_path))
        else:
            target_adata = adata  # same-slice: identity transport test

    start_time = time.time()
    start_dt = datetime.now(timezone.utc)

    print(f"FEAST simulation: {input_path.name}")
    print(f"  parameter_mode:       {parameter_mode}")
    print(f"  spatial_mode:         {spatial_mode}")
    print(f"  assignment_method:    {assignment_method}")
    print(f"  ppf_method:           {ppf_method}")
    print(f"  beta_n_jobs:          {beta_n_jobs}")
    print(f"  beta_patience:        {beta_early_stopping_patience}")
    print(f"  assignment_solver:    {assignment_solver}")
    print(f"  assignment_blocks:    {assignment_blocks}")
    print(f"  assignment_block_mul: {assignment_block_multiplier}")
    print(f"  convert_n_jobs:       {convert_n_jobs}")
    print(f"  annotation_key:       {annotation_key}")
    print(f"  seed:                 {seed}")
    print(f"  target_adata:         {'same-slice' if target_adata is adata else target_path or 'none'}")
    print(f"  shape:                {adata.n_obs} spots x {adata.n_vars} genes")

    simulated = simulate(
        adata=adata,
        annotation_key=annotation_key,
        parameter_mode=parameter_mode,
        spatial_mode=spatial_mode,
        target_adata=target_adata,
        assignment_method=assignment_method,
        seed=seed,
        verbose=True,
        clip_overshoot_factor=0.0,
        boundary_multiplier=1.1,
        use_heuristic_search=False,
        ppf_method=ppf_method,
        beta_n_jobs=beta_n_jobs,
        beta_early_stopping_patience=beta_early_stopping_patience,
        assignment_solver=assignment_solver,
        assignment_blocks=assignment_blocks,
        assignment_block_size=assignment_block_size,
        assignment_block_multiplier=assignment_block_multiplier,
        convert_n_jobs=convert_n_jobs,
    )

    elapsed = time.time() - start_time
    git_info = _get_git_info()

    simulated.uns["figure2_simulator_benchmark"] = {
        "feast_version": feast_version,
        "git_commit": git_info["commit"],
        "git_branch": git_info["branch"],
        "git_dirty": git_info["dirty"],
        "parameter_mode": parameter_mode,
        "spatial_mode": spatial_mode,
        "assignment_method": assignment_method,
        "ppf_method": ppf_method,
        "beta_n_jobs": beta_n_jobs,
        "beta_early_stopping_patience": beta_early_stopping_patience,
        "assignment_solver": assignment_solver,
        "assignment_blocks": assignment_blocks,
        "assignment_block_size": assignment_block_size,
        "assignment_block_multiplier": assignment_block_multiplier,
        "convert_n_jobs": convert_n_jobs,
        "annotation_key": annotation_key,
        "random_seed": seed,
        "start_time": start_dt.isoformat(),
        "elapsed_seconds": round(elapsed, 2),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    simulated.write_h5ad(str(output_path), compression="gzip")

    print(f"  output: {output_path} ({simulated.n_obs} x {simulated.n_vars})")
    print(f"  elapsed: {elapsed:.1f}s")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to input .h5ad file")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path to output simulated .h5ad file")
    parser.add_argument("--parameter-mode", type=str, default="hungarian",
                        choices=("hungarian", "reference_stats"),
                        help="Parameter estimation mode: 'hungarian' or 'reference_stats' (default: hungarian)")
    parser.add_argument("--spatial-mode", type=str, default="reference_rank",
                        choices=("reference_rank", "ot_spatial"),
                        help="Spatial assignment mode: 'reference_rank' or 'ot_spatial' (default: reference_rank)")
    parser.add_argument("--assignment-method", type=str, default="hybrid",
                        choices=ASSIGNMENT_METHODS,
                        help="Gene-profile assignment: 'hybrid' or 'copula_rank' (default: hybrid)")
    parser.add_argument("--annotation-key", type=str, default="auto",
                        help="Annotation column in adata.obs (default: auto-detect)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Random seed (default: 2026)")
    parser.add_argument("--target", type=Path, default=None,
                        help="Path to target .h5ad for ot_spatial mode (default: same as input)")
    parser.add_argument("--ppf-method", type=str, default="interp",
                        choices=("interp", "exact"),
                        help="StudentT PPF method (default: interp optimized path)")
    parser.add_argument("--beta-n-jobs", type=int, default=4,
                        help="Parallel workers for Beta component search (default: 4)")
    parser.add_argument("--beta-early-stopping-patience", type=int, default=2,
                        help="Beta component early-stopping patience; 0 disables early stopping (default: 2)")
    parser.add_argument("--assignment-solver", type=str, default="scipy",
                        choices=("scipy", "lapjv"),
                        help="Hungarian solver backend (default: scipy)")
    parser.add_argument("--assignment-blocks", action=argparse.BooleanOptionalAction, default=True,
                        help="Use batched Hungarian assignment (default: true; use --no-assignment-blocks for full global assignment)")
    parser.add_argument("--assignment-block-size", type=int, default=None,
                        help="Optional number of genes per batched Hungarian block")
    parser.add_argument("--assignment-block-multiplier", type=int, default=8,
                        help="Candidate-pool multiplier per block (default: 8)")
    parser.add_argument("--convert-n-jobs", type=int, default=4,
                        help="Parallel workers for count-model parameter conversion (default: 4)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 1

    return run_simulation(
        args.input, args.output,
        args.parameter_mode,
        args.spatial_mode,
        args.assignment_method,
        args.annotation_key, args.seed,
        target_path=args.target,
        ppf_method=args.ppf_method,
        beta_n_jobs=args.beta_n_jobs,
        beta_early_stopping_patience=args.beta_early_stopping_patience,
        assignment_solver=args.assignment_solver,
        assignment_blocks=args.assignment_blocks,
        assignment_block_size=args.assignment_block_size,
        assignment_block_multiplier=args.assignment_block_multiplier,
        convert_n_jobs=args.convert_n_jobs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
