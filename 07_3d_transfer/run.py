#!/usr/bin/env python3
"""Run one age-specific, original-z-index shard into the ignored work root."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import FEAST
import numpy as np
import pandas as pd
from FEAST.de_novo import ReferenceFitConfig, SimulationConfig, fit_reference, simulate_from_reference

import workflow


def cache_reference_counts_dense_float32(reference: ad.AnnData) -> dict:
    """Cache immutable reference counts once for FEAST's empirical decoder."""

    if "counts" not in reference.layers:
        raise KeyError("reference is missing the required counts layer")
    original = reference.layers["counts"]
    original_type = f"{type(original).__module__}.{type(original).__name__}"
    if hasattr(original, "toarray"):
        original = original.toarray()
    dense = np.asarray(original, dtype=np.float32)
    if dense.shape != reference.shape:
        raise AssertionError("cached reference counts shape changed")
    if not np.isfinite(dense).all():
        raise ValueError("reference counts must be finite before caching")
    reference.layers["counts"] = dense
    return {
        "strategy": "one_time_dense_float32_public_input_cache",
        "original_type": original_type,
        "cached_dtype": str(dense.dtype),
        "shape": [int(value) for value in dense.shape],
        "value_transform": "none",
    }


def assignment_randomness(config: dict, age: str) -> tuple[float, dict]:
    declaration = config["ages"][age]["assignment_randomness"]
    if declaration["mode"] == "fixed":
        return float(declaration["value"]), {"mode": "fixed", "basis": declaration["basis"]}
    path = workflow.calibration_path(config, age)
    manifest_path = workflow.calibration_manifest_path(config, age)
    if not path.is_file():
        raise FileNotFoundError(f"run reference-only E18.5 calibration first: {path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"calibration manifest is missing: {manifest_path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("configuration_id") != config["configuration_id"]:
        raise ValueError("E18.5 calibration configuration ID mismatch")
    if payload.get("selection_scope") != "reference_only" or payload.get("target_blueprint_access") is not False:
        raise ValueError("E18.5 calibration is not reference-only")
    calibration = config["assignment_randomness_calibration"]
    expected_parameters = {
        "n_genes": int(calibration["n_genes"]),
        "ar_step": float(calibration["step"]),
        "max_ar": float(calibration["maximum"]),
        "n_neighbors": int(calibration["n_neighbors"]),
        "transport_backend": str(calibration["transport_backend"]),
        "transport_device": str(calibration["transport_device"]),
        "transport_dtype": str(calibration["transport_dtype"]),
    }
    if (
        int(payload.get("public_seed", -1)) != int(config["public_seed"])
        or int(payload.get("fitted_reference_count", -1)) != int(config["ages"][age]["expected_references"])
        or int(payload.get("fitted_gene_count", -1)) != int(config["reference_fit"]["expected_genes"])
        or payload.get("estimator_parameters") != expected_parameters
        or payload.get("config_sha256") != workflow.sha256_file(config["_config_path"])
        or payload.get("feast", {}).get("wheel_sha256") != config["required_wheel_sha256"]
        or payload.get("feast", {}).get("commit") != config["required_feast_commit"]
        or payload.get("feast", {}).get("source_patch_sha256")
        != config["required_source_patch_sha256"]
        or payload.get("feast", {}).get("candidate_provenance_sha256")
        != config["required_feast_provenance_sha256"]
    ):
        raise ValueError("E18.5 calibration contract or wheel lineage differs")
    expected_inputs = {path.name: workflow.sha256_file(path) for path in workflow.reference_paths(config, age)}
    if payload.get("reference_inputs") != expected_inputs:
        raise ValueError("E18.5 calibration reference hashes differ from current inputs")
    value = float(payload["assignment_randomness"])
    step = float(calibration["step"])
    if not 0.0 <= value <= float(calibration["maximum"]) or abs(value / step - round(value / step)) > 1e-9:
        raise ValueError("E18.5 calibrated assignment_randomness is outside the declared grid")
    transport = config["transport"]
    expected_transport = {
        "sinkhorn_method": str(transport["sinkhorn_method"]),
        "epsilon": float(transport["epsilon"]),
        "sinkhorn_iter": int(transport["sinkhorn_iter"]),
        "sinkhorn_tol": float(transport["sinkhorn_tol"]),
        "unbalanced_transport": bool(transport["unbalanced_transport"]),
        "reg_m": float(transport["reg_m"]),
        "transport_nonconvergence": str(transport["transport_nonconvergence"]),
        "transport_backend": str(calibration["transport_backend"]),
        "transport_device": str(calibration["transport_device"]),
        "transport_dtype": str(calibration["transport_dtype"]),
        "geometry_weight": float(transport["geometry_weight"]),
        "boundary_weight": float(transport["boundary_weight"]),
        "gene_chunk_size": int(transport["gene_chunk_size"]),
        "max_transport_pairs": int(transport["max_transport_pairs"]),
    }
    expected_manifest = {
        "configuration_id": config["configuration_id"],
        "calibration_file": path.name,
        "calibration_sha256": workflow.sha256_file(path),
        "assignment_randomness": value,
        "public_seed": int(config["public_seed"]),
        "config_sha256": workflow.sha256_file(config["_config_path"]),
        "reference_inputs": expected_inputs,
        "transport_config": expected_transport,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(f"E18.5 calibration manifest differs for {key}")
    solver_gate = manifest.get("solver_diagnostics", {})
    if (
        solver_gate.get("status")
        != "all_estimator_transport_calls_completed_strictly"
        or solver_gate.get("positive_residual_gate") is not True
        or solver_gate.get("nonconvergence_policy") != "raise"
    ):
        raise ValueError("E18.5 calibration lacks strict solver evidence")
    return value, {
        "mode": "reference_calibration",
        "path": str(path.relative_to(workflow.STUDY_ROOT)),
        "sha256": workflow.sha256_file(path),
        "manifest_path": str(manifest_path.relative_to(workflow.STUDY_ROOT)),
        "manifest_sha256": workflow.sha256_file(manifest_path),
    }


def load_references(
    config: dict, age: str
) -> tuple[list[ad.AnnData], list[str], dict[str, dict]]:
    references = []
    cache_records: dict[str, dict] = {}
    input_order = None
    for path in workflow.reference_paths(config, age):
        data = ad.read_h5ad(path)
        data.uns["reference_name"] = path.stem
        current = list(map(str, data.var_names))
        if input_order is None:
            input_order = current
        elif current != input_order:
            raise ValueError(f"gene identity/order differs: {path.name}")
        cache_records[path.name] = cache_reference_counts_dense_float32(data)
        references.append(data)
    return references, sorted(input_order or []), cache_records


def build_blueprint(age: str, entry: dict) -> FEAST.SliceBlueprint:
    ids = workflow.make_spot_ids(age, entry)
    regions = list(map(str, entry["region"]))
    obs = pd.DataFrame({
        "spot_id": ids,
        "region": regions,
        "domain": regions,
        "z_index": int(entry["z_index"]),
        "z": float(entry["z_world"]),
        "voxel_i": np.asarray(entry["voxel_i"], dtype=int),
        "voxel_j": int(entry["voxel_j"]),
        "voxel_k": np.asarray(entry["voxel_k"], dtype=int),
    })
    return FEAST.SliceBlueprint(
        coordinates=np.column_stack([entry["x"], entry["y"]]),
        grid_type="devccf_coronal",
        domain_map=np.asarray(regions),
        technology="MERFISH",
        obs=obs,
        metadata={"age": age, "z_index": int(entry["z_index"]), "z_world": float(entry["z_world"]), "expression_source": "none"},
    )


def simulation_config(config: dict, ar: float) -> SimulationConfig:
    values = config["transport"]
    return SimulationConfig(
        epsilon=float(values["epsilon"]),
        sinkhorn_iter=int(values["sinkhorn_iter"]),
        sinkhorn_tol=float(values["sinkhorn_tol"]),
        unbalanced_transport=bool(values["unbalanced_transport"]),
        reg_m=float(values["reg_m"]),
        transport_nonconvergence=str(values["transport_nonconvergence"]),
        sinkhorn_method=str(values["sinkhorn_method"]),
        transport_backend=str(values["transport_backend"]),
        transport_device=str(values["transport_device"]),
        transport_dtype=str(values["transport_dtype"]),
        geometry_weight=float(values["geometry_weight"]),
        boundary_weight=float(values["boundary_weight"]),
        assignment_randomness=float(ar),
        latent_clip_eps=float(values["latent_clip_eps"]),
        gene_chunk_size=int(values["gene_chunk_size"]),
        max_transport_pairs=int(values["max_transport_pairs"]),
        quantile_field_mode=str(values["quantile_field_mode"]),
        rank_scope=str(values["rank_scope"]),
        tie_policy=str(values["tie_policy"]),
        store_latent_scores=bool(values["store_latent_scores"]),
        store_quantiles=bool(values["store_quantiles"]),
        verbose=False,
    )


def selected_indices(args: argparse.Namespace, n_levels: int) -> list[int]:
    if args.z_indices is not None:
        selected = list(args.z_indices)
        if len(set(selected)) != len(selected):
            raise ValueError("--z-indices must not contain duplicates")
    else:
        start = 0 if args.start_index is None else int(args.start_index)
        stop = n_levels if args.stop_index is None else int(args.stop_index)
        selected = list(range(start, stop))
    excluded = set(getattr(args, "exclude_indices", None) or [])
    if excluded and args.z_indices is not None:
        raise ValueError("--exclude-indices is only valid with an index range")
    selected = [value for value in selected if value not in excluded]
    invalid = [value for value in selected if value < 0 or value >= n_levels]
    if invalid or not selected:
        raise ValueError(f"invalid or empty z-index selection: {invalid}")
    return selected


def valid_shard_id(value: str) -> bool:
    return bool(value) and all(character.isalnum() or character in "_-" for character in value)


def canary_indices(age: str, entries: list[dict]) -> list[int]:
    middle = (len(entries) - 1) // 2
    densest = max(range(len(entries)), key=lambda index: (int(entries[index]["n_spots"]), -index))
    requested = [0, middle, densest, len(entries) - 1]
    if age == "E15.5":
        failure_index = next(
            index for index, entry in enumerate(entries)
            if float(entry["z_world"]) == -3.88
        )
        requested.insert(0, failure_index)
    return list(dict.fromkeys(requested))


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=workflow.STUDY_ROOT / "config.yaml")
    parser.add_argument("--age", choices=workflow.AGE_ORDER, required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--z-indices", type=int, nargs="+")
    parser.add_argument("--canary", action="store_true", help="run the fixed age-specific canary indices")
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--stop-index", type=int, help="exclusive")
    parser.add_argument("--exclude-indices", type=int, nargs="+", help="indices to omit from a range (for completed canaries)")
    args = parser.parse_args()
    selection_modes = int(args.z_indices is not None) + int(args.canary) + int(args.start_index is not None or args.stop_index is not None)
    if selection_modes > 1:
        parser.error("use exactly one of --canary, --z-indices, or --start-index/--stop-index")
    if args.exclude_indices and (args.canary or args.z_indices is not None):
        parser.error("--exclude-indices is only valid with --start-index/--stop-index")
    if not valid_shard_id(args.shard_id):
        parser.error("--shard-id may contain only letters, digits, '_' and '-'")

    config = workflow.load_config(args.config)
    workflow.preflight_inputs(config, inspect_h5ad=True)
    feast = workflow.feast_identity(config)
    _, entries = workflow.load_blueprint(config, args.age)
    indices = canary_indices(args.age, entries) if args.canary else selected_indices(args, len(entries))
    ar, ar_provenance = assignment_randomness(config, args.age)
    shard = config["_work_dir"] / args.age / args.shard_id
    if shard.exists():
        raise FileExistsError(f"refusing to reuse shard; archive it or choose a new ID: {shard}")
    shard.mkdir(parents=True, exist_ok=True)

    references, genes, reference_counts_cache = load_references(config, args.age)
    fit = config["reference_fit"]
    model = fit_reference(references, label_key=fit["label_key"], config=ReferenceFitConfig(
        min_gene_spots=int(fit["min_gene_spots"]),
        min_gene_mean=float(fit["min_gene_mean"]),
        max_gene_zero_prop=float(fit["max_gene_zero_prop"]),
        boundary_neighbors=int(fit["boundary_neighbors"]),
        coordinate_scale=fit["coordinate_scale"],
    ))
    if list(model.gene_names) != genes:
        raise RuntimeError("reference fitting changed the required 550-gene panel")
    sim_config = simulation_config(config, ar)
    transport_contract = workflow.frozen_transport_config(config, ar)
    config_hash = workflow.sha256_file(config["_config_path"])
    blueprint_file = workflow.blueprint_path(config, args.age)
    blueprint_hash = workflow.sha256_file(blueprint_file)
    reference_artifacts = workflow.reference_artifact_contract(config, args.age)
    runner_hash = workflow.sha256_file(Path(__file__).resolve())
    rows = []
    metadata = {
        "schema_version": 1,
        "configuration_id": config["configuration_id"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "age": args.age,
        "shard_id": args.shard_id,
        "z_indices": indices,
        "completed_z_indices": [],
        "assignment_randomness": ar,
        "assignment_randomness_provenance": ar_provenance,
        "blueprint_sha256": blueprint_hash,
        "blueprint_manifest_sha256": workflow.sha256_file(workflow.blueprint_manifest_path(config, args.age)),
        "config_sha256": config_hash,
        "reference_artifacts": reference_artifacts,
        "reference_counts_cache": reference_counts_cache,
        "runner_sha256": runner_hash,
        "transport_config": transport_contract,
        "feast": feast,
    }
    write_json_atomic(shard / "provenance.json", metadata)
    for z_index in indices:
        entry = entries[z_index]
        filename = f"z{z_index:03d}__{float(entry['z_world']):+.3f}.h5ad"
        output = shard / filename
        blueprint = build_blueprint(args.age, entry)
        result = simulate_from_reference(
            model,
            blueprint,
            config=sim_config,
            random_seed=int(config["public_seed"]) + z_index,
            marginal_model="empirical_reference",
        )
        result.obs_names = result.obs["spot_id"].astype(str)
        xy = np.column_stack([entry["x"], entry["y"]]).astype(np.float64)
        result.obsm["spatial"] = xy
        result.obsm["spatial_3d"] = np.column_stack([xy, np.full(len(xy), float(entry["z_world"]))])
        result.obs["z"] = float(entry["z_world"])
        result.uns.setdefault("de_novo", {})["coordinate_system"] = "devccf_world"
        summary = workflow.validate_result_identity(result, args.age, entry, genes, transport_contract)
        result.uns["publication_reproduction"] = {
                "configuration_id": config["configuration_id"],
                "study_id": "07",
                "legacy_study_id": "06",
                "age": args.age,
                "z_index": z_index,
                "z_world": float(entry["z_world"]),
                "public_seed": int(config["public_seed"]) + z_index,
                "assignment_randomness": ar,
                "assignment_randomness_provenance": ar_provenance,
                "config_sha256": config_hash,
                "blueprint_sha256": blueprint_hash,
                "blueprint_manifest_sha256": workflow.sha256_file(workflow.blueprint_manifest_path(config, args.age)),
                "reference_artifacts": reference_artifacts,
                "reference_counts_cache": reference_counts_cache,
                "runner_sha256": runner_hash,
                "feast_wheel_sha256": feast["wheel_sha256"],
                "feast_version": feast["version"],
                "feast_commit": feast["commit"],
                "feast_source_patch_sha256": feast["source_patch_sha256"],
                "feast_candidate_provenance_sha256": feast[
                    "candidate_provenance_sha256"
                ],
                "transport_config": transport_contract,
                "solver_diagnostics": summary,
        }
        workflow.validate_publication_lineage(
            result, config, args.age, entry, transport_contract, reference_artifacts
        )
        temporary = output.with_name(output.stem + ".tmp.h5ad")
        result.write_h5ad(temporary, compression="gzip")
        restored = ad.read_h5ad(temporary)
        workflow.validate_result_identity(restored, args.age, entry, genes, transport_contract)
        workflow.validate_publication_lineage(
            restored, config, args.age, entry, transport_contract, reference_artifacts
        )
        os.replace(temporary, output)
        row = {
            **summary,
            "configuration_id": config["configuration_id"],
            "filename": filename,
            "public_seed": int(config["public_seed"]) + z_index,
            "assignment_randomness": ar,
            "matrix_sha256": workflow.matrix_sha256(restored.X),
            "output_sha256": workflow.sha256_file(output),
        }
        rows.append(row)
        write_json_atomic(output.with_suffix(".record.json"), row)
        manifest_temporary = shard / "manifest.csv.tmp"
        pd.DataFrame(rows).sort_values("z_index").to_csv(manifest_temporary, index=False)
        os.replace(manifest_temporary, shard / "manifest.csv")
        metadata["completed_z_indices"] = [int(item["z_index"]) for item in rows]
        metadata["last_completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(shard / "provenance.json", metadata)
        print(f"completed {args.age} z-index {z_index}/{len(entries)-1}", flush=True)


if __name__ == "__main__":
    main()
