#!/usr/bin/env python3
"""Independently validate every fresh Study 06 target and its provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from workflow import load_config, read_json, require_sha, sha256_file, verify_clean_wheel, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def load_frozen(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    preflight = read_json(output_dir / "preflight.json")
    if preflight.get("status") != "passed":
        raise ValueError("preflight status is not passed")
    config_path = output_dir / "frozen_config.yaml"
    plan_path = output_dir / "plan.json"
    manifest_path = output_dir / "input_manifest.csv"
    require_sha(config_path, preflight["config_sha256"], "frozen config")
    require_sha(config_path, preflight["frozen_config_sha256"], "frozen config copy")
    require_sha(plan_path, preflight["plan_sha256"], "target plan")
    require_sha(manifest_path, preflight["input_manifest_sha256"], "input manifest")
    config = load_config(config_path)
    plan = read_json(plan_path)
    if plan["config_sha256"] != preflight["config_sha256"]:
        raise ValueError("plan/config binding changed")
    runtime = verify_clean_wheel(config)
    if preflight.get("wheel_sha256") != str(config["required_wheel_sha256"]):
        raise ValueError("preflight is not bound to the required FEAST wheel")
    if plan.get("wheel_sha256") != str(config["required_wheel_sha256"]):
        raise ValueError("plan is not bound to the required FEAST wheel")
    if runtime.get("wheel_sha256") != str(config["required_wheel_sha256"]):
        raise ValueError("validator is not running from the required FEAST wheel")
    return config, plan, preflight


def verify_source_files(output_dir: Path, plan: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    manifest = pd.read_csv(output_dir / "input_manifest.csv")
    if len(manifest) != 147 or manifest["slice_id"].nunique() != 147:
        raise ValueError("input manifest must contain 147 unique slices")
    by_id: dict[int, dict[str, Any]] = {}
    data_dir = Path(plan["data_dir"])
    for record in manifest.to_dict("records"):
        slice_id = int(record["slice_id"])
        path = data_dir / str(record["filename"])
        require_sha(path, str(record["sha256"]), f"source input {slice_id:03d}")
        by_id[slice_id] = record
    return by_id


def all_target_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for density in plan["densities"].values()
        for row in density["targets"]
    ]


def validate_target_directory_set(output_dir: Path, rows: Sequence[Mapping[str, Any]], prefix: str) -> None:
    expected = {
        Path("targets") / str(row["density_name"]) / f"{prefix}.{int(row['target_slice']):03d}"
        for row in rows
    }
    targets_root = output_dir / "targets"
    observed = {
        path.relative_to(output_dir)
        for path in targets_root.glob("*/*")
        if path.is_dir()
    } if targets_root.is_dir() else set()
    if observed != expected:
        raise ValueError(
            "final target directory set changed; "
            f"missing={sorted(map(str, expected - observed))}, extra={sorted(map(str, observed - expected))}"
        )
    work_root = output_dir / ".work"
    if work_root.exists() and any(path.is_file() for path in work_root.rglob("*")):
        raise ValueError("production root retains failed or incomplete .work artifacts")


def target_metadata(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    target = ad.read_h5ad(path, backed="r")
    try:
        return {
            "shape": (int(target.n_obs), int(target.n_vars)),
            "obs_names": list(map(str, target.obs_names)),
            "var_names": list(map(str, target.var_names)),
            "labels": target.obs[str(dataset["label_key"])].astype(str).to_numpy(),
            "z": np.asarray(target.obs[str(dataset["z_key"])], dtype=float),
            "spatial": np.asarray(target.obsm[str(dataset["spatial_key"])], dtype=float),
            "spatial_3d": np.asarray(target.obsm[str(dataset["spatial_3d_key"])], dtype=float),
        }
    finally:
        target.file.close()


def dense_matrix(matrix: Any) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def validate_count_matrix(matrix: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    values = dense_matrix(matrix)
    if values.shape != expected_shape:
        raise ValueError(f"count matrix shape changed: {values.shape} != {expected_shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("count matrix contains non-finite values")
    if np.any(values < 0):
        raise ValueError("count matrix contains negative values")
    if not np.array_equal(values, np.rint(values)):
        raise ValueError("count matrix contains non-integral values")
    return values


def column_values(records: Mapping[str, Any], key: str, count: int) -> list[str]:
    if key not in records:
        raise ValueError(f"transport diagnostics lack {key!r}")
    values = np.asarray(records[key]).reshape(-1).tolist()
    if len(values) != count:
        raise ValueError(f"transport diagnostic {key!r} length changed")
    return [str(value) for value in values]


def validate_transport_diagnostics(
    diagnostics: Mapping[str, Any],
    transport: Mapping[str, Any],
    allowed_references: set[str],
) -> tuple[int, float]:
    if not isinstance(diagnostics, Mapping) or not diagnostics:
        raise ValueError("transport diagnostics are missing")
    total = 0
    maximum_error = 0.0
    for label, records in diagnostics.items():
        count = int(records.get("n_records", 0))
        if count < 1 or records.get("format") != "columnar_records_v1":
            raise ValueError(f"label {label!r} has invalid transport records")
        converged = column_values(records, "transport_converged", count)
        errors = column_values(records, "transport_final_error", count)
        tolerances = column_values(records, "transport_stop_threshold", count)
        iterations = column_values(records, "transport_iterations", count)
        maxima = column_values(records, "transport_max_iterations", count)
        policies = column_values(records, "transport_nonconvergence_policy", count)
        references = column_values(records, "reference_name", count)
        masses = column_values(records, "transport_mass", count)
        for index in range(count):
            error = float(errors[index])
            tolerance = float(tolerances[index])
            iteration = int(iterations[index])
            maximum = int(maxima[index])
            if converged[index].lower() != "true":
                raise ValueError(f"label {label!r} record {index} is not converged")
            if not np.isfinite(error) or not error < tolerance:
                raise ValueError(f"label {label!r} record {index} lacks a finite sub-tolerance error")
            if tolerance != float(transport["sinkhorn_tol"]):
                raise ValueError("recorded Sinkhorn tolerance differs from the frozen config")
            if maximum != int(transport["sinkhorn_iter"]) or not 0 <= iteration <= maximum:
                raise ValueError("recorded Sinkhorn iteration contract changed")
            if policies[index] != "raise":
                raise ValueError("transport nonconvergence policy is not strict")
            if references[index] not in allowed_references:
                raise ValueError(f"undeclared logical reference {references[index]!r}")
            mass = float(masses[index])
            if not np.isfinite(mass) or mass <= 0.0:
                raise ValueError(f"label {label!r} record {index} has nonpositive transport mass")
            maximum_error = max(maximum_error, error)
            total += 1
    return total, maximum_error


def expected_input_artifacts(row: Mapping[str, Any], prefix: str) -> set[tuple[str, int, str]]:
    records = {
        ("target_metadata_only_generation_expression_evaluation_only", int(row["target_slice"]), str(row["target_input_sha256"])),
        ("lower_reference_expression", int(row["lower_ref_slice"]), str(row["lower_reference_sha256"])),
        ("upper_reference_expression", int(row["upper_ref_slice"]), str(row["upper_reference_sha256"])),
    }
    records.update(
        ("declared_label_donor_expression", int(item["donor_slice"]), str(item["sha256"]))
        for item in row["donor_support"]
    )
    return records


def validate_provenance(
    provenance: Mapping[str, Any],
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    h5ad_path: Path,
) -> None:
    required_equal = {
        "status": "complete",
        "configuration_id": config["configuration_id"],
        "config_sha256": preflight["config_sha256"],
        "plan_sha256": preflight["plan_sha256"],
        "feast_version": config["feast_version"],
        "feast_commit": config["required_feast_commit"],
        "wheel_sha256": config["required_wheel_sha256"],
        "public_seed": int(config["public_seed"]),
        "target_seed": int(row["seed"]),
        "target_index": int(row["target_index"]),
        "density_gap": int(row["gap"]),
        "density_name": str(row["density_name"]),
        "target_slice": int(row["target_slice"]),
        "lower_ref_slice": int(row["lower_ref_slice"]),
        "upper_ref_slice": int(row["upper_ref_slice"]),
        "target_expression_accessed": False,
        "smoothing": False,
        "z_regularization": False,
        "transport": config["transport"],
    }
    for key, expected in required_equal.items():
        if provenance.get(key) != expected:
            raise ValueError(f"provenance field {key!r} changed")
    if provenance.get("reference_weights") != row["reference_weights"]:
        raise ValueError("provenance reference weights changed")
    if provenance.get("donor_support") != row["donor_support"]:
        raise ValueError("provenance donor support changed")
    if not np.isclose(float(provenance["tau"]), float(row["tau"]), rtol=0.0, atol=1e-15):
        raise ValueError("provenance tau changed")
    if provenance.get("output", {}).get("sha256") != sha256_file(h5ad_path):
        raise ValueError("output H5AD hash does not match provenance")
    observed_inputs = {
        (str(item["role"]), int(item["slice_id"]), str(item["sha256"]))
        for item in provenance.get("input_artifacts", [])
    }
    if observed_inputs != expected_input_artifacts(row, str(config["dataset"]["filename_prefix"])):
        raise ValueError("provenance input artifact set changed")
    solver = provenance.get("solver_diagnostics", {})
    if solver.get("all_converged") is not True or int(solver.get("transport_records", 0)) < 1:
        raise ValueError("provenance lacks positive solver diagnostics")


def validate_one(
    output_dir: Path,
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    prefix = str(config["dataset"]["filename_prefix"])
    target_name = f"{prefix}.{int(row['target_slice']):03d}"
    target_dir = output_dir / "targets" / str(row["density_name"]) / target_name
    if {path.name for path in target_dir.iterdir()} != {"generated.h5ad", "provenance.json"}:
        raise ValueError(f"{target_dir} must contain only generated.h5ad and provenance.json")
    h5ad_path = target_dir / "generated.h5ad"
    provenance_path = target_dir / "provenance.json"
    provenance = read_json(provenance_path)
    validate_provenance(provenance, row, config, preflight, h5ad_path)
    source_path = Path(plan["data_dir"]) / f"{target_name}.h5ad"
    target = target_metadata(source_path, config)
    generated = ad.read_h5ad(h5ad_path)
    try:
        if generated.shape != target["shape"]:
            raise ValueError(f"{target_name} generated shape changed")
        if list(map(str, generated.obs_names)) != target["obs_names"]:
            raise ValueError(f"{target_name} spot identity/order changed")
        if list(map(str, generated.var_names)) != target["var_names"]:
            raise ValueError(f"{target_name} gene identity/order changed")
        if not np.array_equal(generated.obs["class"].astype(str).to_numpy(), target["labels"]):
            raise ValueError(f"{target_name} class labels changed")
        if not np.array_equal(np.asarray(generated.obs["z"], dtype=float), target["z"]):
            raise ValueError(f"{target_name} z values changed")
        if not np.array_equal(np.asarray(generated.obsm["spatial"], dtype=float), target["spatial"]):
            raise ValueError(f"{target_name} XY changed")
        if not np.array_equal(np.asarray(generated.obsm["spatial_3d"], dtype=float), target["spatial_3d"]):
            raise ValueError(f"{target_name} XYZ changed")
        counts = validate_count_matrix(generated.layers["counts"], target["shape"])
        if not np.array_equal(counts, dense_matrix(generated.X)):
            raise ValueError(f"{target_name} X and counts layer differ")
        study = generated.uns.get("study06", {})
        required_study = {
            "configuration_id": config["configuration_id"],
            "feast_version": config["feast_version"],
            "feast_commit": config["required_feast_commit"],
            "wheel_sha256": config["required_wheel_sha256"],
            "target_slice": int(row["target_slice"]),
            "target_index": int(row["target_index"]),
            "density_gap": int(row["gap"]),
            "density_name": str(row["density_name"]),
            "seed": int(row["seed"]),
            "lower_ref_slice": int(row["lower_ref_slice"]),
            "upper_ref_slice": int(row["upper_ref_slice"]),
            "target_expression_accessed": False,
            "smoothing": False,
            "z_regularization": False,
        }
        for key, expected in required_study.items():
            if study.get(key) != expected:
                raise ValueError(f"{target_name} study06 metadata {key!r} changed")
        if json.loads(str(study["donor_support_json"])) != row["donor_support"]:
            raise ValueError(f"{target_name} donor support metadata changed")
        weights = {str(key): float(value) for key, value in study["reference_weights"].items()}
        if weights != {str(key): float(value) for key, value in row["reference_weights"].items()}:
            raise ValueError(f"{target_name} reference weights changed")
        references = {
            f"{prefix}.{int(row['lower_ref_slice']):03d}",
            f"{prefix}.{int(row['upper_ref_slice']):03d}",
        }
        records, max_error = validate_transport_diagnostics(
            generated.uns.get("de_novo", {}).get("transport_diagnostics", {}),
            config["transport"],
            references,
        )
    finally:
        del generated
    return {
        "configuration_id": config["configuration_id"],
        "density_name": row["density_name"],
        "gap": int(row["gap"]),
        "target_index": int(row["target_index"]),
        "target_slice": int(row["target_slice"]),
        "seed": int(row["seed"]),
        "n_obs": int(target["shape"][0]),
        "n_vars": int(target["shape"][1]),
        "donor_labels": len(row["donor_support"]),
        "transport_records": records,
        "maximum_final_error": max_error,
        "output_sha256": sha256_file(h5ad_path),
        "provenance_sha256": sha256_file(provenance_path),
        "valid": True,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    config, plan, preflight = load_frozen(output_dir)
    verify_source_files(output_dir, plan)
    rows = all_target_rows(plan)
    if len(rows) != int(config["validation"]["expected_outputs"]):
        raise ValueError("frozen plan no longer contains exactly 93 targets")
    validate_target_directory_set(output_dir, rows, str(config["dataset"]["filename_prefix"]))
    report_path = args.report.resolve() if args.report else output_dir / "validation.csv"
    summary_path = report_path.with_name("validation_summary.json")
    if report_path.exists() or summary_path.exists():
        raise FileExistsError("validation outputs already exist; validate a fresh root or choose a fresh --report")
    records = [validate_one(output_dir, row, config, plan, preflight) for row in rows]
    report = pd.DataFrame(records).sort_values(["gap", "target_index"])
    report.to_csv(report_path, index=False)
    write_json(
        summary_path,
        {
            "configuration_id": config["configuration_id"],
            "status": "passed",
            "expected_outputs": 93,
            "validated_outputs": int(len(report)),
            "all_positive_convergence": True,
            "validation_csv": report_path.name,
            "validation_csv_sha256": sha256_file(report_path),
        },
    )
    print(f"Validation passed for {len(report)} fresh Study 06 outputs: {report_path}")


if __name__ == "__main__":
    main()
