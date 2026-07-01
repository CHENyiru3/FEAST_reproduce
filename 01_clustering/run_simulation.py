#!/usr/bin/env python3
"""Generate FEAST empirical-mode simulations for the Figure 2 clustering benchmark.

Usage:
    NUMBA_DISABLE_JIT=1 MPLCONFIGDIR=/tmp/matplotlib \\
    conda run -p /maiziezhou_lab2/yiru/envs/feast-py311-conda \\
    python scripts/run_clustering_simulation.py \\
      --input-dir data \\
      --slices 151508,151670,151676 \\
      --output-dir outputs/simulations \\
      --simulation-mode empirical \\
      --seed 2026 \\
      --config configs/clustering.yaml
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
    """Build the list of (alteration_id, Alteration kwargs) from config.

    Supports the new distribution-level parameterization:
      - mean: α_μ with optional fano variance coupling
      - variance: α_v with optional ρ_v heterogeneity
      - variance_heterogeneity: ρ_v sweep (level fixed)
      - sparsity_logit_shift: δ_z logit-space shifts
    """
    from FEAST import Alteration

    alterations = []
    alt_cfg = config["alterations"]

    # --- Mean alterations ---
    mean_coupling = alt_cfg.get("mean", {}).get("variance_coupling", None)
    for level in alt_cfg.get("mean", {}).get("levels", []):
        sim_id = f"mean_{level:.2f}".replace(".", "_").replace("_00", "")
        value_str = f"{level:.3g}x"
        if level == 1.0:
            continue  # baseline covers neutral
        alterations.append({
            "simulation_id": sim_id,
            "alteration_type": "mean",
            "fold_change": level,
            "config": Alteration.mean_only(
                fold_change=level,
                variance_coupling=mean_coupling,
            ),
        })

    # --- Variance level alterations ---
    rho_v = alt_cfg.get("variance", {}).get("heterogeneity_scale", 1.0)
    for level in alt_cfg.get("variance", {}).get("levels", []):
        sim_id = f"variance_{level:.2f}".replace(".", "_").replace("_00", "")
        if level == 1.0:
            continue
        alterations.append({
            "simulation_id": sim_id,
            "alteration_type": "variance",
            "fold_change": level,
            "config": Alteration.variance_only(
                fold_change=level,
                dispersion=rho_v,
            ),
        })

    # --- Variance heterogeneity alterations (level fixed at 1.0) ---
    for rho in alt_cfg.get("variance_heterogeneity", {}).get("levels", []):
        sim_id = f"var_hetero_{rho:.2f}".replace(".", "_").replace("_00", "")
        if rho == 1.0:
            continue  # neutral = baseline shape
        alterations.append({
            "simulation_id": sim_id,
            "alteration_type": "variance_heterogeneity",
            "fold_change": rho,
            "config": Alteration.variance_heterogeneity(rho=rho),
        })

    # --- Sparsity logit-shift alterations ---
    for delta_z in alt_cfg.get("sparsity_logit_shift", {}).get("levels", []):
        sign = "pos" if delta_z > 0 else "neg" if delta_z < 0 else "zero"
        sim_id = f"sparsity_{sign}_{abs(delta_z):.2f}".replace(".", "_").replace("_00", "")
        if delta_z == 0.0:
            continue
        alterations.append({
            "simulation_id": sim_id,
            "alteration_type": "sparsity_logit_shift",
            "fold_change": delta_z,  # stores δ_z, not a fold-change
            "config": Alteration.sparsity_logit(delta=delta_z),
        })

    # --- Combined perturbations ---
    for name, params in alt_cfg.get("combined", {}).items():
        alterations.append({
            "simulation_id": f"combined_{name}",
            "alteration_type": "combined",
            "fold_change": None,
            "config": Alteration.comprehensive(
                mean_fc=params.get("mean", 1.0),
                var_fc=params.get("variance", 1.0),
                delta_z=params.get("sparsity", 0.0),
                mean_var_coupling=mean_coupling,
            ),
        })

    # --- Baseline ---
    alterations.append({
        "simulation_id": "baseline",
        "alteration_type": "baseline",
        "fold_change": 1.0,
        "config": Alteration(),
    })

    return alterations


class PrefitSimulator:
    """Fit copula + marginals once per slice, clone for each alteration.

    The vine copula C is invariant under marginal alteration (monotonic
    transformations preserve copulas).  By fitting once and deep-copying
    for each alteration condition, we avoid re-fitting the copula 27 times
    per slice.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        seed: int = 2026,
        ppf_method: str = "interp",
        beta_n_jobs: int = 4,
        beta_early_stopping_patience: int = 2,
        assignment_solver: str = "scipy",
        assignment_blocks: bool = True,
        assignment_block_size: int | None = None,
        assignment_block_multiplier: int = 8,
        convert_n_jobs: int = 4,
    ):
        from copy import deepcopy
        from FEAST import GeneParameterSimulator

        self.adata = adata
        self.seed = seed
        self.assignment_solver = assignment_solver
        self.assignment_blocks = assignment_blocks
        self.assignment_block_size = assignment_block_size
        self.assignment_block_multiplier = assignment_block_multiplier
        self.convert_n_jobs = convert_n_jobs
        self._sim = GeneParameterSimulator(
            ppf_method=ppf_method,
            beta_n_jobs=beta_n_jobs,
            beta_early_stopping_patience=beta_early_stopping_patience,
        )
        self._sim.hybrid_alpha = 0.2
        t0 = time.time()
        print("    Fitting GeneParameterSimulator (marginals + vine copula)...", end=" ", flush=True)
        self._sim.fit(adata, visualize_fits=False)
        self._fit_elapsed = time.time() - t0
        print(f"done in {self._fit_elapsed:.0f}s", flush=True)

    def run_alteration(self, alteration_config, output_path: Path) -> dict:
        from copy import deepcopy
        from FEAST.FEAST_core.parameter_cloud import convert_params_for_new_simulator

        # Clone fitted simulator — copula is preserved in the copy
        sim = deepcopy(self._sim)

        t0 = time.time()

        table, diag = sim.build_gene_parameter_table(
            alteration_config=alteration_config,
            simulation_mode="generative",
            assignment_weights={"mean": 3, "variance": 1, "zero_prop": 1.0},
            random_seed=self.seed,
            assignment_method="hybrid",
            verbose=False,
            use_distributional_alteration=True,
            assignment_solver=self.assignment_solver,
            assignment_blocks=self.assignment_blocks,
            assignment_block_size=self.assignment_block_size,
            assignment_block_multiplier=self.assignment_block_multiplier,
        )

        model_params = convert_params_for_new_simulator(
            table,
            n_spots=self.adata.n_obs,
            n_jobs=self.convert_n_jobs,
        )

        param_elapsed = time.time() - t0

        # Count decoding
        from FEAST.FEAST_core.count_decoding import decode_counts_by_rank
        reference_matrix = self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else np.asarray(self.adata.X)

        n_spots, n_genes = reference_matrix.shape

        simulated_matrix = decode_counts_by_rank(
            reference_matrix,
            model_params,
            boundary_multiplier=1.1,
            reference_X=reference_matrix,
            random_seed=self.seed,
        ).astype(np.float32)

        simulated = ad.AnnData(
            X=simulated_matrix,
            obs=self.adata.obs.copy(),
            var=self.adata.var.copy(),
            obsm={"spatial": self.adata.obsm["spatial"].copy()},
        )

        total_elapsed = time.time() - t0
        print(f"      param={param_elapsed:.0f}s  decode={total_elapsed - param_elapsed:.0f}s  total={total_elapsed:.0f}s", flush=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        simulated.write_h5ad(str(output_path), compression="gzip")

        return {
            "n_spots": simulated.n_obs,
            "n_genes": simulated.n_vars,
            "elapsed_seconds": round(total_elapsed, 2),
        }


def run_simulation(
    adata: ad.AnnData,
    alteration: dict,
    annotation_key: str,
    seed: int,
    output_path: Path,
    parameter_mode: str = "hungarian",
    use_distributional_alteration: bool = True,
    prefit_sim: PrefitSimulator | None = None,
    ppf_method: str = "interp",
    beta_n_jobs: int = 4,
    beta_early_stopping_patience: int = 2,
    assignment_solver: str = "scipy",
    assignment_blocks: bool = True,
    assignment_block_size: int | None = None,
    assignment_block_multiplier: int = 8,
    convert_n_jobs: int = 4,
) -> dict:
    from FEAST import __version__ as feast_version
    from FEAST import simulate

    start_time = time.time()

    if prefit_sim is not None and use_distributional_alteration:
        # Fast path: reuse pre-fitted copula, only re-alter marginals
        result = prefit_sim.run_alteration(alteration["config"], output_path)
        result["n_spots"] = result.get("n_spots", adata.n_obs)
        result["n_genes"] = result.get("n_genes", adata.n_vars)
        return result

    simulated = simulate(
        adata=adata,
        annotation_key=annotation_key,
        parameter_mode=parameter_mode,
        spatial_mode="reference_rank",
        alteration=alteration["config"],
        seed=seed,
        verbose=True,
        clip_overshoot_factor=0.0,
        boundary_multiplier=1.1,
        use_heuristic_search=False,
        use_distributional_alteration=use_distributional_alteration,
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

    simulated.uns["figure2_clustering"] = {
        "feast_version": feast_version,
        "git_commit": git_info["commit"],
        "git_branch": git_info["branch"],
        "git_dirty": git_info["dirty"],
        "parameter_mode": parameter_mode,
        "spatial_mode": "reference_rank",
        "use_distributional_alteration": use_distributional_alteration,
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/simulations"),
                        help="Output directory (default: outputs/simulations)")
    parser.add_argument("--parameter-mode", type=str, default="hungarian",
                        choices=["hungarian", "reference_stats"])
    parser.add_argument("--use-distributional-alteration", action="store_true",
                        default=True,
                        help="Use distribution-level intervention (θ → θ') "
                             "instead of post-sampling scalar multiplication")
    parser.add_argument("--legacy-alteration", action="store_true",
                        help="Use legacy post-sampling scalar multiplication (ablation)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--config", type=Path, default=Path("configs/clustering.yaml"),
                        help="YAML config file")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing outputs")
    parser.add_argument("--ppf-method", choices=["interp", "exact"], default=None,
                        help="StudentT PPF method for parameter-cloud simulation")
    parser.add_argument("--beta-n-jobs", type=int, default=4,
                        help="Parallel workers for Beta component search")
    parser.add_argument("--beta-early-stopping-patience", type=int, default=2,
                        help="Beta component-search early stopping patience; 0 disables")
    parser.add_argument("--assignment-solver", choices=["scipy", "lapjv"], default="scipy",
                        help="Hungarian/JV assignment backend")
    parser.add_argument("--assignment-blocks", action=argparse.BooleanOptionalAction,
                        default=None, help="Use batched Hungarian assignment")
    parser.add_argument("--assignment-block-size", type=int, default=None,
                        help="Genes per assignment batch; default chooses by memory budget")
    parser.add_argument("--assignment-block-multiplier", type=int, default=8,
                        help="Candidate multiplier per assignment batch")
    parser.add_argument("--convert-n-jobs", type=int, default=4,
                        help="Parallel workers for count-model parameter conversion")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    slice_ids = [s.strip() for s in args.slices.split(",")]
    alterations = build_alterations(config)
    seed = config.get("seed", args.seed)
    annotation_key = config.get("annotation_key", "ground_truth")
    parameter_mode = config.get("parameter_mode", args.parameter_mode)
    use_distributional = (
        False if args.legacy_alteration
        else config.get("use_distributional_alteration", args.use_distributional_alteration)
    )
    ppf_method = config.get("ppf_method", args.ppf_method or "interp")
    beta_n_jobs = int(config.get("beta_n_jobs", args.beta_n_jobs))
    beta_early_stopping_patience = int(config.get(
        "beta_early_stopping_patience",
        args.beta_early_stopping_patience,
    ))
    assignment_solver = config.get("assignment_solver", args.assignment_solver)
    assignment_blocks = config.get(
        "assignment_blocks",
        True if args.assignment_blocks is None else args.assignment_blocks,
    )
    assignment_block_size = config.get("assignment_block_size", args.assignment_block_size)
    assignment_block_multiplier = int(config.get(
        "assignment_block_multiplier",
        args.assignment_block_multiplier,
    ))
    convert_n_jobs = int(config.get("convert_n_jobs", args.convert_n_jobs))

    total = len(slice_ids) * len(alterations)
    manifest_rows = []
    done_count = 0
    previous_status: dict[tuple[str, str], str] = {}
    previous_manifest_path = args.output_dir / "simulation_manifest.csv"
    if previous_manifest_path.exists() and not args.overwrite:
        try:
            previous_manifest = pd.read_csv(previous_manifest_path)
            previous_status = {
                (str(row["slice_id"]), str(row["simulation_id"])): str(row["status"])
                for _, row in previous_manifest.iterrows()
                if "slice_id" in row and "simulation_id" in row and "status" in row
            }
        except Exception:
            previous_status = {}

    print(f"Clustering simulation: {len(slice_ids)} slices x {len(alterations)} alterations = {total} jobs")
    print(f"  parameter_mode: {parameter_mode}")
    print(f"  use_distributional_alteration: {use_distributional}")
    print(f"  seed: {seed}")
    print(f"  annotation_key: {annotation_key}")
    print(f"  ppf_method: {ppf_method}")
    print(f"  beta_n_jobs: {beta_n_jobs}")
    print(f"  beta_early_stopping_patience: {beta_early_stopping_patience}")
    print(f"  assignment_solver: {assignment_solver}")
    print(f"  assignment_blocks: {assignment_blocks}")
    print(f"  assignment_block_size: {assignment_block_size}")
    print(f"  assignment_block_multiplier: {assignment_block_multiplier}")
    print(f"  convert_n_jobs: {convert_n_jobs}")
    print()

    for slice_id in slice_ids:
        input_path = args.input_dir / f"{slice_id}.h5ad"
        if not input_path.exists():
            print(f"SKIP {slice_id}: input not found at {input_path}")
            continue

        adata = ad.read_h5ad(str(input_path))
        print(f"[{slice_id}] loaded: {adata.n_obs} spots x {adata.n_vars} genes", flush=True)

        # Pre-fit copula once per slice (copula is invariant under marginal alteration)
        prefit = None
        if use_distributional and parameter_mode == "hungarian":
            prefit = PrefitSimulator(
                adata,
                seed=seed,
                ppf_method=ppf_method,
                beta_n_jobs=beta_n_jobs,
                beta_early_stopping_patience=beta_early_stopping_patience,
                assignment_solver=assignment_solver,
                assignment_blocks=assignment_blocks,
                assignment_block_size=assignment_block_size,
                assignment_block_multiplier=assignment_block_multiplier,
                convert_n_jobs=convert_n_jobs,
            )
            print(f"  Pre-fit done in {prefit._fit_elapsed:.0f}s — reusing for all alterations", flush=True)

        for alt in alterations:
            sim_id = alt["simulation_id"]
            output_path = args.output_dir / slice_id / f"{sim_id}.h5ad"
            prior_status = previous_status.get((str(slice_id), str(sim_id)))
            can_skip_existing = (
                output_path.exists()
                and not args.overwrite
                and (prior_status is None or prior_status in {"ok", "skipped"})
            )

            if can_skip_existing:
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
                    parameter_mode=parameter_mode,
                    use_distributional_alteration=use_distributional,
                    prefit_sim=prefit,
                    ppf_method=ppf_method,
                    beta_n_jobs=beta_n_jobs,
                    beta_early_stopping_patience=beta_early_stopping_patience,
                    assignment_solver=assignment_solver,
                    assignment_blocks=assignment_blocks,
                    assignment_block_size=assignment_block_size,
                    assignment_block_multiplier=assignment_block_multiplier,
                    convert_n_jobs=convert_n_jobs,
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
