#!/usr/bin/env python
"""
Stage 1: Alignment Simulation (v2).

Generates reference (raw data) + expression-altered simulated slices with
known rotations for the Figure 2 alignment benchmark.

Design: Reference = raw Visium data (no FEAST simulation). Moving slices are
FEAST-simulated with mild expression alterations, then rotated. Methods must
match altered simulated expression back to real biological data.

Usage:
    conda run -p ${FEAST_ENV} python scripts/run_alignment_simulation.py \
      --input data/151675.h5ad \
      --output-dir outputs/simulations \
      --angles 1 5 10 30 45 \
      --seed 2026 \
      --config configs/alignment.yaml
"""

import sys
import os
import argparse
import time
import csv
import yaml
from pathlib import Path

import scanpy as sc
import numpy as np

FEAST_ROOT = Path("/maiziezhou_lab2/yiru/FEAST")
sys.path.insert(0, str(FEAST_ROOT / "src"))


def make_manifest_entry(alteration_id, alteration_type, fold_change, sim_type,
                        angle, file_path, adata, seed):
    return {
        "alteration_id": alteration_id,
        "alteration_type": alteration_type,
        "fold_change": fold_change,
        "type": sim_type,
        "angle": angle,
        "file_path": file_path,
        "n_spots": adata.shape[0],
        "n_genes": adata.shape[1],
        "seed": seed,
    }


def build_alterations(config: dict) -> list[dict]:
    """Build alteration list from config."""
    from FEAST import Alteration

    alterations = []
    for alt_cfg in config.get("alterations", []):
        alt_type = alt_cfg["type"]
        fold_change = alt_cfg.get("fold_change")
        alt_id = alt_cfg["id"]

        if alt_type == "baseline":
            alt_config = Alteration()
        elif alt_type == "mean":
            alt_config = Alteration.mean_only(fold_change=fold_change)
        elif alt_type == "variance":
            alt_config = Alteration.variance_only(fold_change=fold_change)
        elif alt_type in {"sparsity_logit_shift", "sparsity_logit"}:
            delta_z = alt_cfg.get("logit_shift", alt_cfg.get("delta_z"))
            if delta_z is None:
                raise ValueError(
                    f"sparsity alteration '{alt_id}' requires logit_shift or delta_z"
                )
            alt_config = Alteration.sparsity_logit(delta=float(delta_z))
        elif alt_type == "sparsity":
            if "logit_shift" in alt_cfg or "delta_z" in alt_cfg:
                delta_z = alt_cfg.get("logit_shift", alt_cfg.get("delta_z"))
                alt_config = Alteration.sparsity_logit(delta=float(delta_z))
            else:
                alt_config = Alteration.sparsity_logit(delta=float(np.log(fold_change)))
        else:
            print(f"WARNING: unknown alteration type '{alt_type}', skipping")
            continue

        alterations.append({
            "id": alt_id,
            "type": alt_type,
            "fold_change": fold_change,
            "config": alt_config,
        })

    return alterations


def main():
    parser = argparse.ArgumentParser(
        description="Generate alignment rotation simulation datasets (v2)"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to reference .h5ad file"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory for simulation outputs"
    )
    parser.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=[1, 5, 10, 30, 45],
        help="Rotation angles in degrees (default: 1 5 10 30 45)",
    )
    parser.add_argument(
        "--seed", type=int, default=2026, help="Random seed for expression simulation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/alignment.yaml",
        help="YAML config with simulation + alteration parameters",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing outputs"
    )
    parser.add_argument("--ppf-method", choices=["interp", "exact"], default=None)
    parser.add_argument("--beta-n-jobs", type=int, default=4)
    parser.add_argument("--beta-early-stopping-patience", type=int, default=2)
    parser.add_argument("--assignment-solver", choices=["scipy", "lapjv"], default="scipy")
    parser.add_argument("--assignment-blocks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--assignment-block-size", type=int, default=None)
    parser.add_argument("--assignment-block-multiplier", type=int, default=8)
    parser.add_argument("--convert-n-jobs", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- load config ---
    with open(args.config) as f:
        config = yaml.safe_load(f) or {}

    seed = config.get("seed", args.seed)
    sim_cfg = config.get("simulation", {})
    data_type = config.get("data_type", "imaging")
    sim_kwargs = {
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

    from FEAST import simulate
    from FEAST.alignment.spatial_align_alter import RotationTransformer

    # ------------------------------------------------------------------
    # 1. Load and preprocess reference (raw data)
    # ------------------------------------------------------------------
    print(f"Loading reference: {args.input}")
    adata = sc.read_h5ad(args.input)
    print(f"  Raw: {adata.shape[0]} spots, {adata.shape[1]} genes")
    print("FEAST knobs:")
    for key, value in sim_kwargs.items():
        print(f"  {key}: {value}")

    sc.pp.filter_genes(adata, min_cells=sim_cfg.get("min_cells", 30))
    print(f"  After min_cells={sim_cfg.get('min_cells', 30)} filter: "
          f"{adata.shape[0]} spots, {adata.shape[1]} genes")

    # ------------------------------------------------------------------
    # 2. Generate and save reference (FEAST-simulated, no rotation)
    # ------------------------------------------------------------------
    from FEAST import Alteration

    ref_path = output_dir / "reference.h5ad"
    if not ref_path.exists() or args.overwrite:
        print(f"Generating FEAST-simulated reference (seed={seed})")
        ref_simulated = simulate(
            adata=adata,
            parameter_mode=sim_cfg.get("parameter_mode", "hungarian"),
            spatial_mode=sim_cfg.get("spatial_mode", "reference_rank"),
            alteration=Alteration(),  # baseline, no alteration
            seed=seed,
            verbose=sim_cfg.get("verbose", True),
            **sim_kwargs,
        )
        ref_simulated.write_h5ad(ref_path)
        print(f"Reference saved: {ref_path} ({ref_simulated.shape[0]} spots)")
    else:
        print(f"Reference exists: {ref_path}")
        ref_simulated = sc.read_h5ad(str(ref_path))

    manifest_rows = []
    manifest_rows.append(
        make_manifest_entry("reference", "none", 0.0, "reference",
                            0.0, str(ref_path), ref_simulated, seed)
    )

    # ------------------------------------------------------------------
    # 3. Build alterations
    # ------------------------------------------------------------------
    alterations = build_alterations(config)
    if not alterations:
        print("ERROR: no alterations defined in config")
        return 1

    print(f"\nAlterations: {len(alterations)}")
    for alt in alterations:
        print(f"  {alt['id']} ({alt['type']}, fold_change={alt['fold_change']})")

    # ------------------------------------------------------------------
    # 4. Per alteration: simulate expression + rotate
    # ------------------------------------------------------------------

    total_jobs = len(alterations) * (1 + len(args.angles))
    job_count = 0

    for alt in alterations:
        alt_id = alt["id"]

        # --- expression simulation (once per alteration) ---
        sim_path = output_dir / f"{alt_id}.h5ad"
        if sim_path.exists() and not args.overwrite:
            print(f"\n  [{job_count + 1}/{total_jobs}] SKIP {alt_id} — exists")
            simulated = sc.read_h5ad(str(sim_path))
        else:
            print(f"\n  [{job_count + 1}/{total_jobs}] SIM  {alt_id} "
                  f"({alt['type']}, fold_change={alt['fold_change']})")
            t0 = time.time()

            simulated = simulate(
                adata=adata,
                parameter_mode=sim_cfg.get("parameter_mode", "hungarian"),
                spatial_mode=sim_cfg.get("spatial_mode", "reference_rank"),
                alteration=alt["config"],
                seed=seed,
                verbose=sim_cfg.get("verbose", True),
                **sim_kwargs,
            )

            elapsed = time.time() - t0
            print(f"    → {simulated.shape[0]} spots, {simulated.shape[1]} genes "
                  f"({elapsed:.1f}s)")

            simulated.uns["figure2_alignment"] = {
                "stage": "simulation",
                "alteration_id": alt_id,
                "alteration_type": alt["type"],
                "fold_change": alt["fold_change"],
                "seed": seed,
                "parameter_mode": sim_cfg.get("parameter_mode", "hungarian"),
                "spatial_mode": sim_cfg.get("spatial_mode", "reference_rank"),
                "source_file": str(args.input),
            }
            simulated.write_h5ad(str(sim_path))
        job_count += 1

        manifest_rows.append(
            make_manifest_entry(alt_id, alt["type"], alt["fold_change"],
                                "simulated", 0.0, str(sim_path),
                                simulated, seed)
        )

        # --- rotation (per angle) ---
        for angle in args.angles:
            angle_tag = str(int(angle)) if angle == int(angle) else str(angle)
            rot_path = output_dir / f"{alt_id}_rotated_{angle_tag}.h5ad"

            if rot_path.exists() and not args.overwrite:
                print(f"  [{job_count + 1}/{total_jobs}] SKIP {alt_id}/rotated_{angle_tag} — exists")
                rotated = sc.read_h5ad(str(rot_path))
            else:
                print(f"  [{job_count + 1}/{total_jobs}] ROT  {alt_id}/rotated_{angle_tag} "
                      f"({angle} deg)")
                t0 = time.time()

                transformer = RotationTransformer(simulated)
                if data_type == "sequencing":
                    rotated = transformer.transform_sequencing(
                        rotation_angle=float(angle),
                        center_correction=config.get("transformation", {}).get("center_correction", 0),
                        min_space=config.get("transformation", {}).get("min_space"),
                        max_grid_size=config.get("transformation", {}).get("max_grid_size", 10000),
                    )
                else:
                    rotated = transformer.transform_imaging(
                        rotation_angle=float(angle),
                        center_correction=config.get("transformation", {}).get("center_correction", 0),
                        keep_bounds=config.get("transformation", {}).get("keep_bounds", True),
                    )

                elapsed = time.time() - t0
                print(f"    → {rotated.shape[0]} spots ({elapsed:.1f}s)")

                rotated.uns["figure2_alignment"] = {
                    "stage": "simulation",
                    "type": "rotated",
                    "alteration_id": alt_id,
                    "alteration_type": alt["type"],
                    "fold_change": alt["fold_change"],
                    "rotation_angle": float(angle),
                    "seed": seed,
                    "data_type": data_type,
                    "source_file": str(args.input),
                }
                rotated.write_h5ad(str(rot_path))
            job_count += 1

            manifest_rows.append(
                make_manifest_entry(alt_id, alt["type"], alt["fold_change"],
                                    "rotated", float(angle), str(rot_path),
                                    rotated, seed)
            )

    # ------------------------------------------------------------------
    # 5. Write simulation manifest
    # ------------------------------------------------------------------
    manifest_path = output_dir / "simulation_manifest.csv"
    fieldnames = [
        "alteration_id", "alteration_type", "fold_change",
        "type", "angle", "file_path",
        "n_spots", "n_genes", "seed",
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nManifest: {manifest_path} ({len(manifest_rows)} rows)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
