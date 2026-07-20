#!/usr/bin/env python3
"""Generate fresh Study 06 targets from one frozen preflight plan."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np
import scipy.sparse as sp

from workflow import (
    load_config,
    read_json,
    require_sha,
    sha256_file,
    verify_clean_wheel,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Root created by preflight.py")
    parser.add_argument("--gap", type=int, choices=(3, 5, 10), required=True)
    parser.add_argument("--target-id", type=int, nargs="+", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_frozen_run(
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    preflight_path = output_dir / "preflight.json"
    plan_path = output_dir / "plan.json"
    config_path = output_dir / "frozen_config.yaml"
    manifest_path = output_dir / "input_manifest.csv"
    for path in (preflight_path, plan_path, config_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"frozen preflight artifact is missing: {path}")
    preflight = read_json(preflight_path)
    if preflight.get("status") != "passed":
        raise ValueError("preflight status is not passed")
    require_sha(config_path, preflight["frozen_config_sha256"], "frozen config")
    require_sha(config_path, preflight["config_sha256"], "preflight config")
    require_sha(manifest_path, preflight["input_manifest_sha256"], "input manifest")
    require_sha(plan_path, preflight["plan_sha256"], "target plan")
    config = load_config(config_path)
    plan = read_json(plan_path)
    if plan.get("config_sha256") != preflight["config_sha256"]:
        raise ValueError("plan is not bound to the frozen config")
    runtime = verify_clean_wheel(config)
    if runtime["commit"] != preflight["feast_commit"]:
        raise ValueError("verified FEAST commit changed after preflight")
    if runtime["wheel_sha256"] != preflight["wheel_sha256"]:
        raise ValueError("verified FEAST wheel changed after preflight")
    if plan.get("wheel_sha256") != runtime["wheel_sha256"]:
        raise ValueError("target plan is not bound to the verified FEAST wheel")
    import FEAST

    if FEAST.__version__ != config["feast_version"] or FEAST.__version__ != preflight["feast_version"]:
        raise ValueError("FEAST version changed after preflight")
    return config, plan, preflight, runtime


def density_for_gap(plan: Mapping[str, Any], gap: int) -> tuple[str, dict[str, Any]]:
    matches = [
        (name, density)
        for name, density in plan["densities"].items()
        if int(density["spec"]["gap"]) == int(gap)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one frozen density for gap {gap}, found {len(matches)}")
    return matches[0]


def select_targets(
    rows: Sequence[dict[str, Any]],
    target_ids: Sequence[int] | None,
    shard_index: int,
    shard_count: int,
) -> list[dict[str, Any]]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard contract requires 0 <= shard-index < shard-count")
    selected = [row for row in rows if int(row["target_index"]) % shard_count == shard_index]
    if target_ids is not None:
        requested = {int(item) for item in target_ids}
        available = {int(row["target_slice"]) for row in rows}
        missing = requested - available
        if missing:
            raise ValueError(f"requested target IDs are not in this density: {sorted(missing)}")
        selected = [row for row in selected if int(row["target_slice"]) in requested]
    if not selected:
        raise ValueError("target selection is empty")
    return selected


def input_path(plan: Mapping[str, Any], config: Mapping[str, Any], slice_id: int) -> Path:
    prefix = str(config["dataset"]["filename_prefix"])
    return Path(plan["data_dir"]) / f"{prefix}.{int(slice_id):03d}.h5ad"


def verify_target_inputs(row: Mapping[str, Any], plan: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = [
        (
            "target_metadata_only_generation_expression_evaluation_only",
            int(row["target_slice"]),
            row["target_input_sha256"],
        ),
        ("lower_reference_expression", int(row["lower_ref_slice"]), row["lower_reference_sha256"]),
        ("upper_reference_expression", int(row["upper_ref_slice"]), row["upper_reference_sha256"]),
    ]
    records.extend(
        ("declared_label_donor_expression", int(item["donor_slice"]), item["sha256"])
        for item in row["donor_support"]
    )
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for role, slice_id, expected_sha in records:
        key = (role, slice_id)
        if key in seen:
            continue
        seen.add(key)
        path = input_path(plan, config, slice_id)
        require_sha(path, str(expected_sha), f"input slice {slice_id:03d}")
        unique.append(
            {
                "role": role,
                "artifact_id": f"{config['dataset']['filename_prefix']}.{slice_id:03d}",
                "slice_id": slice_id,
                "sha256": str(expected_sha),
            }
        )
    return unique


def read_reference(path: Path, reference_name: str, label_key: str) -> ad.AnnData:
    reference = ad.read_h5ad(path)
    reference.obs = reference.obs.copy()
    reference.obs[label_key] = reference.obs[label_key].astype(str)
    reference.uns["reference_name"] = reference_name
    return reference


def augment_declared_donors(
    lower: ad.AnnData,
    upper: ad.AnnData,
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[ad.AnnData, ad.AnnData]:
    label_key = str(config["dataset"]["label_key"])
    lower_parts = [lower]
    upper_parts = [upper]
    donor_cache: dict[int, ad.AnnData] = {}
    try:
        for record in row["donor_support"]:
            donor_id = int(record["donor_slice"])
            donor = donor_cache.get(donor_id)
            if donor is None:
                donor = read_reference(
                    input_path(plan, config, donor_id),
                    f"{config['dataset']['filename_prefix']}.{donor_id:03d}",
                    label_key,
                )
                donor_cache[donor_id] = donor
            mask = donor.obs[label_key].astype(str).to_numpy() == str(record["label"])
            if int(mask.sum()) != int(record["n_spots"]):
                raise ValueError(f"declared donor support changed for label {record['label']!r}")
            subset = donor[mask, :].copy()
            if record["side"] == "lower":
                lower_parts.append(subset)
            elif record["side"] == "upper":
                upper_parts.append(subset)
            else:
                raise ValueError(f"invalid donor side: {record['side']!r}")
        return (
            concat_reference_parts(lower_parts, str(lower.uns["reference_name"]), label_key),
            concat_reference_parts(upper_parts, str(upper.uns["reference_name"]), label_key),
        )
    finally:
        donor_cache.clear()


def concat_reference_parts(parts: Sequence[ad.AnnData], reference_name: str, label_key: str) -> ad.AnnData:
    if len(parts) == 1:
        return parts[0]
    merged = ad.concat(parts, axis=0, join="inner", merge="same", uns_merge="first")
    merged.obs = merged.obs.copy()
    merged.obs[label_key] = merged.obs[label_key].astype(str)
    merged.obs_names_make_unique()
    merged.uns["reference_name"] = reference_name
    return merged


def target_metadata(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    target = ad.read_h5ad(path, backed="r")
    try:
        # Deliberately do not access target.X or target.layers here.
        return {
            "obs_names": list(map(str, target.obs_names)),
            "var_names": list(map(str, target.var_names)),
            "labels": target.obs[str(dataset["label_key"])].astype(str).to_numpy(),
            "z": np.asarray(target.obs[str(dataset["z_key"])], dtype=float),
            "spatial": np.asarray(target.obsm[str(dataset["spatial_key"])], dtype=float).copy(),
            "spatial_3d": np.asarray(target.obsm[str(dataset["spatial_3d_key"])], dtype=float).copy(),
            "grid_type": target.uns.get("grid_type", "generic"),
            "technology": target.uns.get("technology"),
        }
    finally:
        target.file.close()


def make_blueprint(metadata: Mapping[str, Any], label_key: str):
    import pandas as pd
    from FEAST.de_novo import SimulationBlueprint

    labels = np.asarray(metadata["labels"], dtype=str)
    # SliceBlueprint v1 normalizes its DataFrame index. Preserve source IDs in
    # a column, then restore them explicitly on the generated AnnData.
    obs = pd.DataFrame(
        {
            "_target_obs_name": list(metadata["obs_names"]),
            label_key: labels,
            "domain": labels,
        }
    )
    return SimulationBlueprint(
        coordinates=np.asarray(metadata["spatial_3d"], dtype=float),
        grid_type=metadata["grid_type"],
        domain_map=labels,
        technology=metadata["technology"],
        obs=obs,
        metadata={"source": "target_metadata_only", "label_key": label_key},
    )


def simulation_config(config: Mapping[str, Any], assignment_randomness: float, verbose: bool):
    from FEAST.de_novo import SimulationConfig

    generation = config["generation"]
    transport = config["transport"]
    return SimulationConfig(
        epsilon=float(transport["epsilon"]),
        sinkhorn_iter=int(transport["sinkhorn_iter"]),
        sinkhorn_tol=float(transport["sinkhorn_tol"]),
        unbalanced_transport=bool(transport["unbalanced_transport"]),
        reg_m=float(transport["reg_m"]),
        transport_nonconvergence=str(transport["transport_nonconvergence"]),
        geometry_weight=float(transport["geometry_weight"]),
        boundary_weight=float(transport["boundary_weight"]),
        assignment_randomness=float(assignment_randomness),
        latent_clip_eps=float(transport["latent_clip_eps"]),
        gene_chunk_size=int(transport["gene_chunk_size"]),
        max_transport_pairs=int(transport["max_transport_pairs"]),
        reference_weight_eta=float(generation["reference_weight_eta"]),
        boundary_multiplier=float(generation["boundary_multiplier"]),
        diffusion_level=float(generation["diffusion_level"]),
        boundary_softness=float(generation["boundary_softness"]),
        coordinate_scale=config["reference_fit"]["coordinate_scale"],
        verbose=bool(verbose),
        quantile_field_mode=str(generation["quantile_field_mode"]),
        rank_scope=str(generation["rank_scope"]),
        target_parameter_mode=str(generation["target_parameter_mode"]),
        tie_policy=str(generation["tie_policy"]),
        tie_jitter_scale=float(generation["tie_jitter_scale"]),
        min_rank_scope_size=int(generation["min_rank_scope_size"]),
        store_latent_scores=bool(generation["store_latent_scores"]),
        store_quantiles=generation["store_quantiles"],
        reference_conflict_policy=str(generation["reference_conflict_policy"]),
    )


def convergence_summary(generated: ad.AnnData) -> dict[str, Any]:
    diagnostics = generated.uns.get("de_novo", {}).get("transport_diagnostics", {})
    if not isinstance(diagnostics, Mapping) or not diagnostics:
        raise ValueError("generated output lacks transport diagnostics")
    total = 0
    maximum_error = 0.0
    minimum_mass = float("inf")
    for label, records in diagnostics.items():
        count = int(records.get("n_records", 0))
        if count < 1:
            raise ValueError(f"label {label!r} has no transport records")
        required = (
            "transport_converged",
            "transport_final_error",
            "transport_stop_threshold",
            "transport_max_iterations",
            "transport_nonconvergence_policy",
            "transport_mass",
        )
        if any(key not in records for key in required):
            raise ValueError(f"label {label!r} lacks positive convergence fields")
        for index in range(count):
            converged = str(records["transport_converged"][index]).lower() == "true"
            error = float(records["transport_final_error"][index])
            tolerance = float(records["transport_stop_threshold"][index])
            policy = str(records["transport_nonconvergence_policy"][index])
            mass = float(records["transport_mass"][index])
            if (
                not converged
                or not np.isfinite(error)
                or not error < tolerance
                or policy != "raise"
                or not np.isfinite(mass)
                or mass <= 0.0
            ):
                raise ValueError(f"label {label!r} transport record {index} lacks positive convergence evidence")
            maximum_error = max(maximum_error, error)
            minimum_mass = min(minimum_mass, mass)
            total += 1
    return {
        "transport_records": total,
        "all_converged": True,
        "maximum_final_error": maximum_error,
        "minimum_transport_mass": minimum_mass,
    }


def dense_counts(adata: ad.AnnData) -> np.ndarray:
    matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix)


def validate_generated_in_memory(generated: ad.AnnData, metadata: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    expected_shape = (len(metadata["obs_names"]), len(metadata["var_names"]))
    if generated.shape != expected_shape:
        raise ValueError(f"generated shape {generated.shape} differs from target metadata {expected_shape}")
    if list(map(str, generated.obs_names)) != list(metadata["obs_names"]):
        raise ValueError("generated observation identity/order changed")
    if list(map(str, generated.var_names)) != list(metadata["var_names"]):
        raise ValueError("generated gene identity/order changed")
    if not np.array_equal(generated.obs["class"].astype(str).to_numpy(), metadata["labels"]):
        raise ValueError("generated class labels changed")
    if not np.array_equal(np.asarray(generated.obs["z"], dtype=float), metadata["z"]):
        raise ValueError("generated z metadata changed")
    if not np.array_equal(np.asarray(generated.obsm["spatial"], dtype=float), metadata["spatial"]):
        raise ValueError("generated XY coordinates changed")
    if not np.array_equal(np.asarray(generated.obsm["spatial_3d"], dtype=float), metadata["spatial_3d"]):
        raise ValueError("generated 3D coordinates changed")
    counts = dense_counts(generated)
    if not np.all(np.isfinite(counts)) or np.any(counts < 0) or not np.array_equal(counts, np.rint(counts)):
        raise ValueError("generated counts must be finite, nonnegative integers")
    x_values = generated.X.toarray() if sp.issparse(generated.X) else np.asarray(generated.X)
    if not np.array_equal(counts, x_values):
        raise ValueError("generated X and counts layer differ")
    if int(generated.uns["study06"]["seed"]) != int(row["seed"]):
        raise ValueError("generated seed metadata changed")
    convergence_summary(generated)


def generate_target(
    row: dict[str, Any],
    density: Mapping[str, Any],
    config: dict[str, Any],
    plan: dict[str, Any],
    preflight: dict[str, Any],
    runtime: dict[str, Any],
    output_dir: Path,
    verbose: bool,
) -> dict[str, Any]:
    from FEAST.de_novo import ReferenceFitConfig, fit_reference, simulate_from_reference
    import FEAST

    prefix = str(config["dataset"]["filename_prefix"])
    label_key = str(config["dataset"]["label_key"])
    target_name = f"{prefix}.{int(row['target_slice']):03d}"
    final_dir = output_dir / "targets" / str(row["density_name"]) / target_name
    work_dir = output_dir / ".work" / str(row["density_name"]) / target_name
    if final_dir.exists() or work_dir.exists():
        raise FileExistsError(f"fresh-only target path already exists: {final_dir if final_dir.exists() else work_dir}")
    work_dir.mkdir(parents=True)
    lower: ad.AnnData | None = None
    upper: ad.AnnData | None = None
    try:
        input_artifacts = verify_target_inputs(row, plan, config)
        target_meta = target_metadata(input_path(plan, config, int(row["target_slice"])), config)
        lower_name = f"{prefix}.{int(row['lower_ref_slice']):03d}"
        upper_name = f"{prefix}.{int(row['upper_ref_slice']):03d}"
        lower = read_reference(input_path(plan, config, int(row["lower_ref_slice"])), lower_name, label_key)
        upper = read_reference(input_path(plan, config, int(row["upper_ref_slice"])), upper_name, label_key)
        lower, upper = augment_declared_donors(lower, upper, row, plan, config)
        fit = config["reference_fit"]
        model = fit_reference(
            [lower, upper],
            label_key=label_key,
            config=ReferenceFitConfig(
                min_gene_spots=int(fit["min_gene_spots"]),
                min_gene_mean=float(fit["min_gene_mean"]),
                max_gene_zero_prop=float(fit["max_gene_zero_prop"]),
                boundary_neighbors=int(fit["boundary_neighbors"]),
                coordinate_scale=fit["coordinate_scale"],
            ),
        )
        if set(model.gene_names) != set(target_meta["var_names"]):
            raise ValueError("reference fitting did not retain exactly the target 1,122-gene panel")
        ar = float(density["assignment_randomness_preflight"]["observed_assignment_randomness"])
        generated = simulate_from_reference(
            model,
            make_blueprint(target_meta, label_key),
            config=simulation_config(config, ar, verbose),
            random_seed=int(row["seed"]),
            reference_weights={str(key): float(value) for key, value in row["reference_weights"].items()},
            marginal_model="empirical_reference",
        )
        generated = generated[:, target_meta["var_names"]].copy()
        generated.obs_names = list(target_meta["obs_names"])
        if "_target_obs_name" in generated.obs:
            generated.obs.drop(columns=["_target_obs_name"], inplace=True)
        generated.obs[label_key] = np.asarray(target_meta["labels"], dtype=str)
        generated.obs["z"] = np.asarray(target_meta["z"], dtype=float)
        generated.obsm["spatial"] = np.asarray(target_meta["spatial"], dtype=float)
        generated.obsm["spatial_3d"] = np.asarray(target_meta["spatial_3d"], dtype=float)
        generated.uns["study06"] = {
            "configuration_id": str(config["configuration_id"]),
            "feast_version": str(config["feast_version"]),
            "feast_commit": str(config["required_feast_commit"]),
            "wheel_sha256": str(config["required_wheel_sha256"]),
            "target_slice": int(row["target_slice"]),
            "target_index": int(row["target_index"]),
            "density_gap": int(row["gap"]),
            "density_name": str(row["density_name"]),
            "seed": int(row["seed"]),
            "assignment_randomness": ar,
            "lower_ref_slice": int(row["lower_ref_slice"]),
            "upper_ref_slice": int(row["upper_ref_slice"]),
            "tau": float(row["tau"]),
            "reference_weights": {str(key): float(value) for key, value in row["reference_weights"].items()},
            "donor_support_json": json.dumps(row["donor_support"], sort_keys=True, separators=(",", ":")),
            "target_expression_accessed": False,
            "smoothing": False,
            "z_regularization": False,
        }
        diagnostics = convergence_summary(generated)
        validate_generated_in_memory(generated, target_meta, row)
        h5ad_path = work_dir / "generated.h5ad"
        generated.write_h5ad(h5ad_path, compression="gzip")
        restored = ad.read_h5ad(h5ad_path)
        try:
            validate_generated_in_memory(restored, target_meta, row)
            restored_diagnostics = convergence_summary(restored)
            if restored_diagnostics != diagnostics:
                raise ValueError("transport diagnostics changed during H5AD round-trip")
            if restored.uns.get("study06", {}).get("wheel_sha256") != str(config["required_wheel_sha256"]):
                raise ValueError("FEAST wheel identity changed during H5AD round-trip")
        finally:
            del restored
        output_sha = sha256_file(h5ad_path)
        provenance = {
            "status": "complete",
            "configuration_id": config["configuration_id"],
            "config_sha256": preflight["config_sha256"],
            "plan_sha256": preflight["plan_sha256"],
            "preflight_sha256": sha256_file(output_dir / "preflight.json"),
            "feast_version": FEAST.__version__,
            "feast_commit": preflight["feast_commit"],
            "wheel_sha256": preflight["wheel_sha256"],
            "public_seed": int(config["public_seed"]),
            "target_seed": int(row["seed"]),
            "target_index": int(row["target_index"]),
            "density_gap": int(row["gap"]),
            "density_name": str(row["density_name"]),
            "target_slice": int(row["target_slice"]),
            "lower_ref_slice": int(row["lower_ref_slice"]),
            "upper_ref_slice": int(row["upper_ref_slice"]),
            "tau": float(row["tau"]),
            "reference_weights": row["reference_weights"],
            "donor_support": row["donor_support"],
            "assignment_randomness": ar,
            "assignment_randomness_preflight": density["assignment_randomness_preflight"],
            "target_expression_accessed": False,
            "target_observed_fields": config["generation"]["target_observed_fields"],
            "smoothing": False,
            "z_regularization": False,
            "transport": config["transport"],
            "solver_diagnostics": diagnostics,
            "input_artifacts": input_artifacts,
            "output": {"filename": "generated.h5ad", "sha256": output_sha},
            "runner": {"path": Path(__file__).name, "sha256": sha256_file(Path(__file__))},
            "python": sys.version,
            "platform": platform.platform(),
            "runtime": runtime,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(work_dir / "provenance.json", provenance)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(work_dir, final_dir)
        return {
            "target_slice": int(row["target_slice"]),
            "target_index": int(row["target_index"]),
            "target_dir": str(final_dir.relative_to(output_dir)),
            "output_sha256": output_sha,
            "provenance_sha256": sha256_file(final_dir / "provenance.json"),
        }
    except Exception as exc:
        write_json(
            work_dir / "FAILED.json",
            {
                "status": "failed",
                "target_slice": int(row["target_slice"]),
                "seed": int(row["seed"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
    finally:
        del lower, upper


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    config, plan, preflight, runtime = load_frozen_run(output_dir)
    density_name, density = density_for_gap(plan, args.gap)
    targets = select_targets(
        density["targets"],
        args.target_id,
        int(args.shard_index),
        int(args.shard_count),
    )
    completed: list[dict[str, Any]] = []
    for position, row in enumerate(targets, start=1):
        print(
            f"[{density_name}] {position}/{len(targets)} target {int(row['target_slice']):03d} "
            f"seed={int(row['seed'])}",
            flush=True,
        )
        completed.append(
            generate_target(row, density, config, plan, preflight, runtime, output_dir, bool(args.verbose))
        )
    manifest_name = (
        f"gap_{args.gap}_targets_{'-'.join(str(item) for item in args.target_id)}.json"
        if args.target_id is not None
        else f"gap_{args.gap}_shard_{args.shard_index:03d}_of_{args.shard_count:03d}.json"
    )
    manifest_path = output_dir / "shard_manifests" / manifest_name
    if manifest_path.exists():
        raise FileExistsError(f"fresh-only shard manifest exists: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        manifest_path,
        {
            "configuration_id": config["configuration_id"],
            "density_name": density_name,
            "gap": int(args.gap),
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
            "targets": completed,
        },
    )
    print(f"Completed {len(completed)} fresh targets; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
