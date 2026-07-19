#!/usr/bin/env python3
"""Compute the publication alignment metrics for the fresh 40-row run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.spatial import cKDTree


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


def artifact_manifest_is_valid(output: Path, job: object) -> bool:
    try:
        artifacts = json.loads(job.artifacts)
    except (TypeError, json.JSONDecodeError):
        return False
    expected_names = expected_artifact_names(job.method)
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
        and sha256_file(runner_log) == job.runner_log_sha256
    )


def mean_row_correlation(left, right) -> float:
    left = sparse.csr_matrix(left, dtype=np.float64)
    right = sparse.csr_matrix(right, dtype=np.float64)
    if left.shape != right.shape or left.shape[1] < 2:
        raise RuntimeError("expression matrices do not align for correlation")
    n = left.shape[1]
    left_sum = np.asarray(left.sum(axis=1)).ravel()
    right_sum = np.asarray(right.sum(axis=1)).ravel()
    cross = np.asarray(left.multiply(right).sum(axis=1)).ravel()
    left_sq = np.asarray(left.multiply(left).sum(axis=1)).ravel()
    right_sq = np.asarray(right.multiply(right).sum(axis=1)).ravel()
    numerator = cross - left_sum * right_sum / n
    denominator = np.sqrt(
        np.maximum(left_sq - left_sum**2 / n, 0)
        * np.maximum(right_sq - right_sum**2 / n, 0)
    )
    values = np.divide(
        numerator, denominator, out=np.full(left.shape[0], np.nan),
        where=denominator > 0,
    )
    if not np.isfinite(values).any():
        raise RuntimeError("no finite per-spot expression correlations")
    return float(np.nanmean(values))


def recovered_rotation(source: np.ndarray, target: np.ndarray) -> float:
    source = source - source.mean(axis=0)
    target = target - target.mean(axis=0)
    left, _, right = np.linalg.svd(source.T @ target)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    return float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))


def angular_error(observed: float, expected: float) -> float:
    return float(abs((observed - expected + 180) % 360 - 180))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rotation-dir", type=Path, required=True)
    parser.add_argument("--methods-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    config = yaml.safe_load(args.config.read_text())
    config_hash = sha256_file(args.config)
    rotations = pd.read_csv(args.rotation_dir / "rotation_manifest.csv")
    methods = pd.read_csv(args.methods_dir / "method_manifest.csv")
    expected = {
        (alteration, float(angle), method)
        for alteration in config["alterations"]
        for angle in config["angles"]
        for method in ("spateo", "paste")
    }
    observed = set(
        zip(methods["alteration"], methods["angle_degrees"].astype(float), methods["method"])
    )
    if (
        len(methods) != 40
        or methods["job_id"].duplicated().any()
        or observed != expected
        or set(methods["status"]) != {"ok"}
        or not methods["config_sha256"].eq(config_hash).all()
        or not methods["public_seed"].eq(int(config["seed"])).all()
        or not methods["reference_sha256"].eq(sha256_file(args.reference)).all()
    ):
        raise RuntimeError("method manifest is not the exact successful 40-row matrix")

    reference = ad.read_h5ad(args.reference)
    reference_coords = np.asarray(reference.obsm["spatial"], dtype=np.float64)
    tree = cKDTree(reference_coords)
    rotation_lookup = {
        (row.alteration, float(row.angle_degrees)): Path(row.output_path)
        for row in rotations.itertuples(index=False)
    }
    rows = []
    for job in methods.itertuples(index=False):
        moving_path = rotation_lookup[(job.alteration, float(job.angle_degrees))]
        moving = ad.read_h5ad(moving_path)
        if not moving.obs_names.equals(reference.obs_names):
            raise RuntimeError(f"spot order differs: {job.job_id}")
        common = reference.var_names.intersection(moving.var_names, sort=False)
        if common.empty:
            raise RuntimeError(f"no exact genes shared: {job.job_id}")
        output_dir = Path(job.output_dir)
        if not artifact_manifest_is_valid(output_dir, job):
            raise RuntimeError(f"artifact manifest or runner log is invalid: {job.job_id}")
        coordinates = pd.read_csv(output_dir / "aligned_coordinates.csv")
        if not pd.Index(coordinates["spot_barcode"].astype(str)).equals(moving.obs_names):
            raise RuntimeError(f"coordinate spot order differs: {job.job_id}")
        aligned = coordinates[["x_aligned", "y_aligned"]].to_numpy(np.float64)
        moving_coords = np.asarray(moving.obsm["spatial"], dtype=np.float64)
        if not np.isfinite(aligned).all():
            raise RuntimeError(f"non-finite aligned coordinates: {job.job_id}")
        distances = np.linalg.norm(aligned - reference_coords, axis=1)
        nn_distances, nn_indices = tree.query(aligned, k=1)
        nn_accuracy = float(
            np.mean(reference.obs_names.to_numpy()[nn_indices] == moving.obs_names.to_numpy())
        )
        region_accuracy = np.nan
        if "ground_truth" in reference.obs and "ground_truth" in moving.obs:
            region_accuracy = float(
                np.mean(
                    reference.obs["ground_truth"].astype(str).to_numpy()[nn_indices]
                    == moving.obs["ground_truth"].astype(str).to_numpy()
                )
            )
        expression_correlation = mean_row_correlation(
            reference[:, common].X, moving[:, common].X
        )
        moving_to_aligned = recovered_rotation(moving_coords, aligned)
        residual = recovered_rotation(aligned, reference_coords)
        matrix_path = output_dir / (
            "mapping_matrix.npy" if job.method == "spateo" else "coupling_matrix.npy"
        )
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        if (
            matrix.shape != (reference.n_obs, moving.n_obs)
            or matrix.size == 0
            or not np.isfinite(matrix).all()
        ):
            raise RuntimeError(f"invalid method matrix: {job.job_id}")
        transport_mass = float(np.sum(matrix, dtype=np.float64))
        transport_minimum = float(np.min(matrix))
        if not np.isfinite(transport_mass) or transport_mass <= 0:
            raise RuntimeError(f"non-positive method matrix mass: {job.job_id}")
        forward = np.argmax(matrix, axis=1)
        backward = np.argmax(matrix, axis=0)
        identity_accuracy = float(np.mean(forward == np.arange(reference.n_obs)))
        mutual = float(np.mean(backward[forward] == np.arange(reference.n_obs)))
        diagnostics = json.loads((output_dir / "diagnostics.json").read_text())
        if job.method == "spateo":
            negative_tolerance = float(
                config["spateo"]["mapping_negative_tolerance"]
            )
            matrix_evidence_valid = bool(
                transport_minimum >= -negative_tolerance
                and diagnostics.get("mapping_shape") == list(matrix.shape)
                and diagnostics.get("mapping_all_finite") is True
                and np.isclose(
                    float(diagnostics.get("mapping_minimum", np.nan)),
                    transport_minimum,
                    rtol=1e-12,
                    atol=1e-12,
                )
                and np.isclose(
                    float(diagnostics.get("mapping_mass", np.nan)),
                    transport_mass,
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
                    transport_mass,
                    rtol=1e-12,
                    atol=1e-12,
                )
            )
        solver_ok = (
            diagnostics.get("status") == "completed"
            and diagnostics.get("cuda_available") is True
            and diagnostics.get("mapping_all_finite") is True
            and diagnostics.get("spot_support_preserved") is True
            and diagnostics.get("gene_order_preserved") is True
            and diagnostics.get("convergence_claim") is False
            and diagnostics.get("iteration_schedule", {}).get("kind") == "fixed"
            and diagnostics.get("iteration_schedule", {}).get("completed_return") is True
            and matrix_evidence_valid
            if job.method == "spateo"
            else diagnostics.get("status") == "converged"
            and diagnostics.get("positive_convergence_evidence") is True
            and matrix_evidence_valid
        )
        rows.append(
            {
                "job_id": job.job_id,
                "configuration_id": job.configuration_id,
                "method": job.method,
                "alteration": job.alteration,
                "angle_degrees": float(job.angle_degrees),
                "status": "ok" if solver_ok else "invalid",
                "canonical_candidate": bool(solver_ok),
                "solver_evidence_disposition": (
                    "fixed_schedule_complete_no_convergence_claim"
                    if job.method == "spateo"
                    else "positive_convergence_evidence"
                ),
                "n_aligned": len(aligned),
                "n_exact_join_genes": len(common),
                "mean_spatial_error": float(distances.mean()),
                "median_spatial_error": float(np.median(distances)),
                "rmse_spatial_error": float(np.sqrt(np.mean(distances**2))),
                "max_spatial_error": float(distances.max()),
                "nn_accuracy": nn_accuracy,
                "nn_region_accuracy": region_accuracy,
                "nn_mean_distance": float(np.mean(nn_distances)),
                "ge_correlation_exact_gene_join": expression_correlation,
                "recovered_moving_to_aligned_rotation": moving_to_aligned,
                "expected_moving_to_aligned_rotation": -float(job.angle_degrees),
                "rotation_recovery_error": angular_error(
                    moving_to_aligned, -float(job.angle_degrees)
                ),
                "residual_aligned_to_target_rotation": residual,
                "transport_argmax_identity_accuracy": identity_accuracy,
                "transport_mutual_argmax_consistency": mutual,
                "transport_total_mass": transport_mass,
                "transport_minimum": transport_minimum,
            }
        )

    results = pd.DataFrame(rows)
    if len(results) != 40 or not results["canonical_candidate"].all():
        raise RuntimeError("one or more alignment rows failed the score-time evidence gate")
    summary = (
        results.groupby(["method", "alteration"], as_index=False)
        .agg(
            n_rows=("job_id", "size"),
            mean_spatial_error=("mean_spatial_error", "mean"),
            mean_nn_accuracy=("nn_accuracy", "mean"),
            mean_nn_region_accuracy=("nn_region_accuracy", "mean"),
            mean_ge_correlation=("ge_correlation_exact_gene_join", "mean"),
            mean_rotation_recovery_error=("rotation_recovery_error", "mean"),
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = args.output_dir / "alignment_metrics.csv"
    summary_path = args.output_dir / "alignment_summary.csv"
    results.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    (args.output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "configuration_id": "study02-alignment-score-v2",
                "feast_versions": sorted(set(methods["feast_version"].astype(str))),
                "feast_commits": sorted(set(methods["feast_commit"].astype(str))),
                "rotation_manifest_sha256": sha256_file(args.rotation_dir / "rotation_manifest.csv"),
                "method_manifest_sha256": sha256_file(args.methods_dir / "method_manifest.csv"),
                "outputs": {
                    metrics_path.name: sha256_file(metrics_path),
                    summary_path.name: sha256_file(summary_path),
                },
            },
            indent=2,
        ) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
