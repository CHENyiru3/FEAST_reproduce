#!/usr/bin/env python3
"""Validate Study 02 geometry, artifacts, solver evidence, and row coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from FEAST.alignment import apply_spatial_transform


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def expected_artifact_names(method: str) -> set[str]:
    common = {
        "aligned.h5ad",
        "aligned_coordinates.csv",
        "diagnostics.json",
        "transform.json",
    }
    if method == "spateo":
        return common | {"mapping_matrix.npy"}
    return common | {"coupling_matrix.npy", "inner_emd_diagnostics.csv"}


def artifact_manifest_is_valid(output: Path, row: object) -> bool:
    try:
        artifacts = json.loads(row.artifacts)
    except (TypeError, json.JSONDecodeError):
        return False
    expected_names = expected_artifact_names(row.method)
    if not isinstance(artifacts, list) or len(artifacts) != len(expected_names):
        return False
    expected_paths = {
        name: (output / name).resolve() for name in expected_names
    }
    listed_names: set[str] = set()
    for item in artifacts:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256"}
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
        ):
            return False
        path = Path(item["path"])
        name = path.name
        if (
            name in listed_names
            or name not in expected_paths
            or path.resolve() != expected_paths[name]
            or not path.is_file()
            or sha256_file(path) != item["sha256"]
        ):
            return False
        listed_names.add(name)
    try:
        actual_names = {
            path.name
            for path in output.iterdir()
            if path.is_file() and path.name != "runner.log"
        }
    except OSError:
        return False
    runner_log = output / "runner.log"
    return bool(
        listed_names == expected_names
        and actual_names == expected_names
        and runner_log.is_file()
        and sha256_file(runner_log) == row.runner_log_sha256
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rotation-dir", type=Path, required=True)
    parser.add_argument("--methods-dir", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.config.read_text())
    config_hash = sha256_file(args.config)
    tolerance = float(config["geometry_tolerance"])
    reference = ad.read_h5ad(args.reference)
    reference_hash = sha256_file(args.reference)
    reference_coords = np.asarray(reference.obsm["spatial"], dtype=np.float64)

    rotations = pd.read_csv(args.rotation_dir / "rotation_manifest.csv")
    expected_rotations = {
        (alteration, float(angle))
        for alteration in config["alterations"]
        for angle in config["angles"]
    }
    if (
        len(rotations) != 20
        or set(zip(rotations["alteration"], rotations["angle_degrees"].astype(float)))
        != expected_rotations
        or rotations[["alteration", "angle_degrees"]].duplicated().any()
    ):
        raise RuntimeError("rotation manifest is not the exact declared 20-row matrix")
    rotation_rows = []
    for row in rotations.itertuples(index=False):
        source_path, moving_path = Path(row.source_path), Path(row.output_path)
        source, moving = ad.read_h5ad(source_path), ad.read_h5ad(moving_path)
        transform = moving.uns["feast_alignment_transform"]
        restored = apply_spatial_transform(
            moving.obsm["spatial"], transform["inverse_matrix"]
        )
        source_coords = np.asarray(source.obsm["spatial"], dtype=np.float64)
        valid = bool(
            source.obs_names.equals(reference.obs_names)
            and moving.obs_names.equals(source.obs_names)
            and moving.var_names.equals(source.var_names)
            and np.max(np.abs(source_coords - reference_coords)) <= tolerance
            and np.max(np.abs(restored - source_coords)) <= tolerance
            and sha256_file(source_path) == row.source_sha256
            and sha256_file(moving_path) == row.output_sha256
        )
        rotation_rows.append(
            {"kind": "rotation", "job_id": f"{row.alteration}__{row.angle_degrees:g}", "valid": valid}
        )

    methods = pd.read_csv(args.methods_dir / "method_manifest.csv")
    expected_methods = {
        (alteration, float(angle), method)
        for alteration, angle in expected_rotations
        for method in ("spateo", "paste")
    }
    if (
        len(methods) != 40
        or set(
            zip(
                methods["alteration"],
                methods["angle_degrees"].astype(float),
                methods["method"],
            )
        )
        != expected_methods
        or methods["job_id"].duplicated().any()
    ):
        raise RuntimeError("method manifest is not the exact declared 40-row matrix")
    method_rows = []
    for row in methods.itertuples(index=False):
        output = Path(row.output_dir)
        aligned = ad.read_h5ad(output / "aligned.h5ad")
        moving = ad.read_h5ad(Path(row.moving))
        transform = json.loads((output / "transform.json").read_text())
        diagnostics = json.loads((output / "diagnostics.json").read_text())
        expected_configuration_id = (
            f"study02-{row.method}-{row.alteration}-{float(row.angle_degrees):g}"
        )
        matrix_path = output / (
            "mapping_matrix.npy" if row.method == "spateo" else "coupling_matrix.npy"
        )
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        matrix_finite = bool(matrix.size > 0 and np.isfinite(matrix).all())
        matrix_mass = (
            float(np.sum(matrix, dtype=np.float64)) if matrix_finite else np.nan
        )
        matrix_minimum = float(np.min(matrix)) if matrix_finite else np.nan
        if row.method == "spateo":
            negative_tolerance = float(
                config["spateo"]["mapping_negative_tolerance"]
            )
            matrix_evidence_valid = bool(
                matrix_minimum >= -negative_tolerance
                and diagnostics.get("mapping_shape") == list(matrix.shape)
                and diagnostics.get("mapping_all_finite") is True
                and np.isclose(
                    float(diagnostics.get("mapping_minimum", np.nan)),
                    matrix_minimum,
                    rtol=1e-12,
                    atol=1e-12,
                )
                and np.isclose(
                    float(diagnostics.get("mapping_mass", np.nan)),
                    matrix_mass,
                    rtol=1e-12,
                    atol=1e-12,
                )
                and float(
                    diagnostics.get("mapping_negative_tolerance", np.nan)
                )
                == negative_tolerance
            )
        else:
            coupling = diagnostics.get("coupling", {})
            matrix_evidence_valid = bool(
                coupling.get("shape") == list(matrix.shape)
                and np.isclose(
                    float(coupling.get("mass", np.nan)),
                    matrix_mass,
                    rtol=1e-12,
                    atol=1e-12,
                )
            )
        solver_valid = (
            diagnostics.get("status") == "completed"
            and diagnostics.get("cuda_available") is True
            and diagnostics.get("mapping_all_finite") is True
            and diagnostics.get("spot_support_preserved") is True
            and diagnostics.get("gene_order_preserved") is True
            and diagnostics.get("convergence_claim") is False
            and diagnostics.get("iteration_schedule", {}).get("kind") == "fixed"
            and diagnostics.get("iteration_schedule", {}).get("completed_return") is True
            if row.method == "spateo"
            else diagnostics.get("status") == "converged"
            and diagnostics.get("positive_convergence_evidence") is True
            and diagnostics.get("all_inner_emd_optimal") is True
            and diagnostics.get("outer_conditional_gradient", {}).get(
                "positive_convergence_evidence"
            ) is True
        )
        artifact_hashes = transform.get("output_artifacts", {})
        required = [output / "aligned.h5ad", matrix_path, output / "aligned_coordinates.csv"]
        hashes_valid = all(
            artifact_hashes.get(path.name) == sha256_file(path) for path in required
        )
        manifest_artifacts_valid = artifact_manifest_is_valid(output, row)
        valid = bool(
            row.status == "ok"
            and row.configuration_id == expected_configuration_id
            and row.config_sha256 == config_hash
            and int(row.public_seed) == int(config["seed"])
            and row.reference_sha256 == reference_hash
            and row.moving_sha256 == sha256_file(Path(row.moving))
            and transform.get("status") == "ok"
            and transform.get("configuration_id") == expected_configuration_id
            and int(transform.get("public_seed", -1)) == int(config["seed"])
            and transform.get("input_artifacts", {}).get("reference_sha256")
            == reference_hash
            and transform.get("input_artifacts", {}).get("moving_sha256")
            == row.moving_sha256
            and transform.get("source_artifacts", {}).get("config_sha256")
            == config_hash
            and transform.get("source_artifacts", {}).get("wrapper_sha256")
            == sha256_file(
                Path(__file__).parent
                / "methods"
                / ("run_spateo.py" if row.method == "spateo" else "run_paste.py")
            )
            and transform.get("feast", {}).get("commit")
            == config["required_feast_commit"]
            and solver_valid
            and hashes_valid
            and manifest_artifacts_valid
            and aligned.obs_names.equals(moving.obs_names)
            and aligned.var_names.equals(moving.var_names)
            and matrix.shape == (reference.n_obs, moving.n_obs)
            and matrix_finite
            and np.isfinite(matrix_mass)
            and matrix_mass > 0
            and matrix_evidence_valid
        )
        method_rows.append({"kind": "method", "job_id": row.job_id, "valid": valid})

    metrics = pd.read_csv(args.scores_dir / "alignment_metrics.csv")
    score_provenance = json.loads((args.scores_dir / "provenance.json").read_text())
    score_hashes_valid = all(
        score_provenance.get("outputs", {}).get(name) == sha256_file(args.scores_dir / name)
        for name in ("alignment_metrics.csv", "alignment_summary.csv")
    )
    score_valid = bool(
        len(metrics) == 40
        and metrics["job_id"].is_unique
        and (metrics["status"] == "ok").all()
        and metrics["canonical_candidate"].all()
        and np.isfinite(metrics["transport_total_mass"]).all()
        and (metrics["transport_total_mass"] > 0).all()
        and (
            metrics.loc[metrics["method"] == "spateo", "transport_minimum"]
            >= -float(config["spateo"]["mapping_negative_tolerance"])
        ).all()
        and score_hashes_valid
    )
    rows = rotation_rows + method_rows + [
        {"kind": "score_table", "job_id": "alignment_metrics", "valid": score_valid}
    ]
    result = pd.DataFrame(rows)
    if len(rotation_rows) != 20 or len(method_rows) != 40 or not result["valid"].all():
        raise RuntimeError("Study 02 validation failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print("Study 02 validated: 20 rotations, 40 method rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
