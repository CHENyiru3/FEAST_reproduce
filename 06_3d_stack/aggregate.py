#!/usr/bin/env python3
"""Evaluate validated Study 06 outputs and real split-half z continuity."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from workflow import load_config, read_json, require_sha, sha256_file, sha256_json, write_json


METRICS = (
    "mean_per_gene_spot_pearson",
    "median_per_gene_spot_pearson",
    "gene_mean_pearson",
    "log1p_gene_mean_pearson",
    "gene_variance_pearson",
    "gene_zero_fraction_pearson",
    "library_size_pearson",
    "relative_error_gene_mean",
    "zero_pattern_jaccard",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def dense(matrix: Any) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray()
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    if sp.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def counts(adata: ad.AnnData, layer: str = "counts") -> np.ndarray:
    return dense(adata.layers[layer] if layer in adata.layers else adata.X).astype(np.float64, copy=False)


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def column_pearson(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("column Pearson inputs must be matching 2D arrays")
    x = left - left.mean(axis=0, keepdims=True)
    y = right - right.mean(axis=0, keepdims=True)
    denominator = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y, axis=0))
    values = np.full(left.shape[1], np.nan, dtype=np.float64)
    valid = denominator > 0.0
    values[valid] = np.sum(x[:, valid] * y[:, valid], axis=0) / denominator[valid]
    return values


def metric_row(generated: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if generated.shape != target.shape:
        raise ValueError("generated and target expression shapes differ")
    gene_corr = column_pearson(generated, target)
    generated_mean = generated.mean(axis=0)
    target_mean = target.mean(axis=0)
    generated_var = generated.var(axis=0)
    target_var = target.var(axis=0)
    generated_zero = np.mean(generated == 0, axis=0)
    target_zero = np.mean(target == 0, axis=0)
    union = np.count_nonzero((generated == 0) | (target == 0))
    intersection = np.count_nonzero((generated == 0) & (target == 0))
    return {
        "mean_per_gene_spot_pearson": float(np.nanmean(gene_corr)),
        "median_per_gene_spot_pearson": float(np.nanmedian(gene_corr)),
        "gene_mean_pearson": safe_pearson(generated_mean, target_mean),
        "log1p_gene_mean_pearson": safe_pearson(np.log1p(generated_mean), np.log1p(target_mean)),
        "gene_variance_pearson": safe_pearson(generated_var, target_var),
        "gene_zero_fraction_pearson": safe_pearson(generated_zero, target_zero),
        "library_size_pearson": safe_pearson(generated.sum(axis=1), target.sum(axis=1)),
        "relative_error_gene_mean": float(
            np.mean(np.abs(generated_mean - target_mean) / np.maximum(np.abs(target_mean), 1.0e-8))
        ),
        "zero_pattern_jaccard": float(intersection / union) if union else 1.0,
    }


def class_means(matrix: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    label_strings = np.asarray(labels, dtype=str)
    return {
        label: matrix[label_strings == label].mean(axis=0)
        for label in sorted(set(label_strings.tolist()))
    }


def split_half_class_means(
    matrix: np.ndarray,
    labels: np.ndarray,
    seed: int,
    minimum_class_spots: int = 2,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Deterministically split cells within each class of one real slice."""
    label_strings = np.asarray(labels, dtype=str)
    rng = np.random.default_rng(int(seed))
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label in sorted(set(label_strings.tolist())):
        indices = np.flatnonzero(label_strings == label)
        if indices.size < int(minimum_class_spots):
            continue
        shuffled = rng.permutation(indices)
        split_at = int(indices.size // 2)
        left, right = shuffled[:split_at], shuffled[split_at:]
        if left.size and right.size:
            result[label] = (matrix[left].mean(axis=0), matrix[right].mean(axis=0))
    return result


def continuity_row(previous: Mapping[str, np.ndarray], current: Mapping[str, np.ndarray]) -> dict[str, Any]:
    common = sorted(set(previous) & set(current))
    if not common:
        return {"common_classes": 0, "class_gene_mean_pearson": float("nan"), "mean_absolute_change": float("nan")}
    left = np.concatenate([previous[label] for label in common])
    right = np.concatenate([current[label] for label in common])
    return {
        "common_classes": len(common),
        "class_gene_mean_pearson": safe_pearson(left, right),
        "mean_absolute_change": float(np.mean(np.abs(left - right))),
    }


def z_coherence_rows(
    trajectories: Mapping[str, Sequence[Mapping[str, Any]]],
    gene_names: Sequence[str],
    left_key: str,
    right_key: str,
    minimum_z_levels: int,
) -> list[dict[str, Any]]:
    """Correlate paired class-gene trajectories across ordered z levels."""
    rows: list[dict[str, Any]] = []
    for label, records in sorted(trajectories.items()):
        eligible = [record for record in records if record.get(left_key) is not None and record.get(right_key) is not None]
        eligible.sort(key=lambda record: float(record["z"]))
        if len(eligible) < int(minimum_z_levels):
            continue
        left = np.vstack([np.asarray(record[left_key], dtype=np.float64) for record in eligible])
        right = np.vstack([np.asarray(record[right_key], dtype=np.float64) for record in eligible])
        correlations = column_pearson(left, right)
        z_values = [float(record["z"]) for record in eligible]
        for gene, correlation in zip(gene_names, correlations, strict=True):
            if np.isfinite(correlation):
                rows.append(
                    {
                        "class": str(label),
                        "gene": str(gene),
                        "n_z_levels": len(eligible),
                        "z_min": min(z_values),
                        "z_max": max(z_values),
                        "z_coherence": float(correlation),
                    }
                )
    return rows


def continuity_summary(
    density_name: str,
    gap: int,
    generated_rows: pd.DataFrame,
    baseline_rows: pd.DataFrame,
) -> dict[str, Any]:
    generated = generated_rows["z_coherence"].to_numpy(dtype=float)
    baseline = baseline_rows["z_coherence"].to_numpy(dtype=float)
    generated = generated[np.isfinite(generated)]
    baseline = baseline[np.isfinite(baseline)]
    generated_median = float(np.median(generated)) if generated.size else float("nan")
    baseline_median = float(np.median(baseline)) if baseline.size else float("nan")
    normalized = (
        float(generated_median / baseline_median)
        if np.isfinite(generated_median) and np.isfinite(baseline_median) and baseline_median != 0.0
        else float("nan")
    )
    return {
        "density_name": density_name,
        "gap": int(gap),
        "trajectory_unit": "class-gene across ordered z levels",
        "targets_are_biological_replicates": False,
        "generated_target_z_coherence_n_pairs": int(generated.size),
        "generated_target_z_coherence_mean": float(np.mean(generated)) if generated.size else float("nan"),
        "generated_target_z_coherence_median": generated_median,
        "real_split_half_z_coherence_n_pairs": int(baseline.size),
        "real_split_half_z_coherence_mean": float(np.mean(baseline)) if baseline.size else float("nan"),
        "real_split_half_z_coherence_median": baseline_median,
        "normalized_z_coherence": normalized,
    }


def require_validation(output_dir: Path) -> pd.DataFrame:
    summary_path = output_dir / "validation_summary.json"
    validation_path = output_dir / "validation.csv"
    summary = read_json(summary_path)
    if summary.get("status") != "passed" or int(summary.get("validated_outputs", 0)) != 93:
        raise ValueError("complete 93-output validation is required before evaluation")
    require_sha(validation_path, summary["validation_csv_sha256"], "validation table")
    validation = pd.read_csv(validation_path)
    valid = validation["valid"].map(lambda value: str(value).strip().lower() == "true")
    if len(validation) != 93 or not valid.all():
        raise ValueError("validation table does not contain 93 valid outputs")
    return validation


def intended_path(filename: str) -> str:
    return str(Path("evaluation") / filename)


def artifact_record(
    configuration_id: str,
    role: str,
    path: Path,
    recorded_path: str,
    density_name: str = "all",
    target_slice: int = -1,
) -> dict[str, Any]:
    return {
        "configuration_id": configuration_id,
        "density_name": density_name,
        "target_slice": int(target_slice),
        "role": role,
        "path": recorded_path,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def evaluate(output_dir: Path, work_dir: Path) -> None:
    config_path = output_dir / "frozen_config.yaml"
    plan_path = output_dir / "plan.json"
    preflight_path = output_dir / "preflight.json"
    input_manifest_path = output_dir / "input_manifest.csv"
    validation_path = output_dir / "validation.csv"
    validation_summary_path = output_dir / "validation_summary.json"
    config = load_config(config_path)
    plan = read_json(plan_path)
    validation = require_validation(output_dir)

    prefix = str(config["dataset"]["filename_prefix"])
    label_key = str(config["dataset"]["label_key"])
    baseline_contract = config["evaluation"]["real_continuity_baseline"]
    metric_records: list[dict[str, Any]] = []
    continuity_records: list[dict[str, Any]] = []
    generated_z_records: list[dict[str, Any]] = []
    baseline_z_records: list[dict[str, Any]] = []
    continuity_summaries: list[dict[str, Any]] = []

    for density_name, density in plan["densities"].items():
        previous_means: dict[str, np.ndarray] | None = None
        previous_row: Mapping[str, Any] | None = None
        trajectories: dict[str, list[dict[str, Any]]] = {}
        gene_names: list[str] | None = None
        for row in sorted(density["targets"], key=lambda item: int(item["target_index"])):
            target_name = f"{prefix}.{int(row['target_slice']):03d}"
            generated_path = output_dir / "targets" / density_name / target_name / "generated.h5ad"
            target_path = Path(plan["data_dir"]) / f"{target_name}.h5ad"
            generated_adata = ad.read_h5ad(generated_path)
            target_adata = ad.read_h5ad(target_path)
            try:
                observed_genes = list(map(str, generated_adata.var_names))
                if list(map(str, generated_adata.obs_names)) != list(map(str, target_adata.obs_names)):
                    raise ValueError(f"{target_name} spot identity changed after validation")
                if observed_genes != list(map(str, target_adata.var_names)):
                    raise ValueError(f"{target_name} gene identity changed after validation")
                if gene_names is None:
                    gene_names = observed_genes
                elif gene_names != observed_genes:
                    raise ValueError(f"{density_name} gene order changed across targets")
                generated_counts = counts(generated_adata)
                target_counts = counts(target_adata, str(config["dataset"]["counts_layer"]))
                metrics = metric_row(generated_counts, target_counts)
                metric_records.append(
                    {
                        "configuration_id": config["configuration_id"],
                        "density_name": density_name,
                        "gap": int(row["gap"]),
                        "target_index": int(row["target_index"]),
                        "target_slice": int(row["target_slice"]),
                        "target_z": float(row["target_z"]),
                        "seed": int(row["seed"]),
                        "n_obs": int(generated_adata.n_obs),
                        "n_genes": int(generated_adata.n_vars),
                        **metrics,
                    }
                )
                labels = target_adata.obs[label_key].astype(str).to_numpy()
                generated_means = class_means(generated_counts, labels)
                target_means = class_means(target_counts, labels)
                split_means = split_half_class_means(
                    target_counts,
                    labels,
                    int(config["public_seed"]) + int(row["target_slice"]),
                    int(baseline_contract["minimum_class_spots"]),
                )
                for label in sorted(target_means):
                    split = split_means.get(label)
                    trajectories.setdefault(label, []).append(
                        {
                            "z": float(row["target_z"]),
                            "generated": generated_means[label],
                            "target": target_means[label],
                            "split_a": None if split is None else split[0],
                            "split_b": None if split is None else split[1],
                        }
                    )
                if previous_means is not None and previous_row is not None:
                    continuity_records.append(
                        {
                            "configuration_id": config["configuration_id"],
                            "density_name": density_name,
                            "gap": int(row["gap"]),
                            "lower_target_slice": int(previous_row["target_slice"]),
                            "upper_target_slice": int(row["target_slice"]),
                            "lower_z": float(previous_row["target_z"]),
                            "upper_z": float(row["target_z"]),
                            **continuity_row(previous_means, generated_means),
                        }
                    )
                previous_means, previous_row = generated_means, row
            finally:
                del generated_adata, target_adata

        assert gene_names is not None
        generated_density = pd.DataFrame(
            z_coherence_rows(
                trajectories,
                gene_names,
                "generated",
                "target",
                int(baseline_contract["minimum_z_levels"]),
            )
        )
        baseline_density = pd.DataFrame(
            z_coherence_rows(
                trajectories,
                gene_names,
                "split_a",
                "split_b",
                int(baseline_contract["minimum_z_levels"]),
            )
        )
        if generated_density.empty or baseline_density.empty:
            raise ValueError(f"{density_name} lacks valid z-coherence or split-half baseline rows")
        for frame in (generated_density, baseline_density):
            frame.insert(0, "gap", int(density["spec"]["gap"]))
            frame.insert(0, "density_name", density_name)
            frame.insert(0, "configuration_id", config["configuration_id"])
        generated_z_records.extend(generated_density.to_dict("records"))
        baseline_z_records.extend(baseline_density.to_dict("records"))
        continuity_summaries.append(
            continuity_summary(
                density_name,
                int(density["spec"]["gap"]),
                generated_density,
                baseline_density,
            )
        )

    paths = {
        "target_metrics.csv": work_dir / "target_metrics.csv",
        "density_summary.csv": work_dir / "density_summary.csv",
        "adjacent_z_continuity.csv": work_dir / "adjacent_z_continuity.csv",
        "z_coherence.csv": work_dir / "z_coherence.csv",
        "real_split_half_z_coherence.csv": work_dir / "real_split_half_z_coherence.csv",
        "continuity_summary.csv": work_dir / "continuity_summary.csv",
    }
    metrics = pd.DataFrame(metric_records).sort_values(["gap", "target_index"])
    metrics.to_csv(paths["target_metrics.csv"], index=False)
    density_records: list[dict[str, Any]] = []
    for (density_name, gap), frame in metrics.groupby(["density_name", "gap"], sort=False):
        record: dict[str, Any] = {"density_name": density_name, "gap": int(gap), "targets": int(len(frame))}
        for metric in METRICS:
            record[f"mean_{metric}"] = float(frame[metric].mean())
            record[f"median_{metric}"] = float(frame[metric].median())
        density_records.append(record)
    pd.DataFrame(density_records).sort_values("gap").to_csv(paths["density_summary.csv"], index=False)
    pd.DataFrame(continuity_records).sort_values(["gap", "lower_target_slice"]).to_csv(
        paths["adjacent_z_continuity.csv"], index=False
    )
    pd.DataFrame(generated_z_records).sort_values(["gap", "class", "gene"]).to_csv(
        paths["z_coherence.csv"], index=False
    )
    pd.DataFrame(baseline_z_records).sort_values(["gap", "class", "gene"]).to_csv(
        paths["real_split_half_z_coherence.csv"], index=False
    )
    pd.DataFrame(continuity_summaries).sort_values("gap").to_csv(paths["continuity_summary.csv"], index=False)

    output_hashes = {name: sha256_file(path) for name, path in paths.items()}
    score_provenance_path = work_dir / "score_provenance.json"
    write_json(
        score_provenance_path,
        {
            "status": "complete",
            "configuration_id": config["configuration_id"],
            "feast_version": config["feast_version"],
            "feast_commit": config["required_feast_commit"],
            "wheel_sha256": config["required_wheel_sha256"],
            "target_expression_accessed_for_evaluation": True,
            "target_expression_accessed_during_generation": False,
            "metric_contract": {
                "atomic_metrics": list(METRICS),
                "composite_score": "prohibited",
                "winner_ranking": "prohibited",
            },
            "continuity_contract": config["evaluation"],
            "sources": {
                "config_sha256": sha256_file(config_path),
                "plan_sha256": sha256_file(plan_path),
                "preflight_sha256": sha256_file(preflight_path),
                "input_manifest_sha256": sha256_file(input_manifest_path),
                "validation_csv_sha256": sha256_file(validation_path),
                "validation_summary_sha256": sha256_file(validation_summary_path),
                "validated_generated_artifacts_sha256": sha256_json(
                    validation[["density_name", "target_slice", "output_sha256", "provenance_sha256"]].to_dict("records")
                ),
                "aggregate_runner_sha256": sha256_file(Path(__file__)),
            },
            "outputs": output_hashes,
            "numerical_change": {
                "field": "sinkhorn_iter",
                "historical": int(config["historical_comparison"]["historical_sinkhorn_iter"]),
                "fresh": int(config["transport"]["sinkhorn_iter"]),
                "classification": "declared_solver_configuration_change",
            },
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    artifacts: list[dict[str, Any]] = []
    for record in validation.to_dict("records"):
        target_name = f"{prefix}.{int(record['target_slice']):03d}"
        target_dir = output_dir / "targets" / str(record["density_name"]) / target_name
        for role, filename, expected_sha in (
            ("generated_h5ad", "generated.h5ad", record["output_sha256"]),
            ("target_provenance", "provenance.json", record["provenance_sha256"]),
        ):
            path = target_dir / filename
            require_sha(path, str(expected_sha), f"{target_name} {role}")
            artifacts.append(
                artifact_record(
                    str(config["configuration_id"]),
                    role,
                    path,
                    str(path.relative_to(output_dir)),
                    str(record["density_name"]),
                    int(record["target_slice"]),
                )
            )
    source_manifest = pd.read_csv(input_manifest_path)
    for record in source_manifest.to_dict("records"):
        artifacts.append(
            {
                "configuration_id": config["configuration_id"],
                "density_name": "source",
                "target_slice": int(record["slice_id"]),
                "role": "source_input_h5ad",
                "path": str(record["filename"]),
                "sha256": str(record["sha256"]),
                "bytes": int(record["bytes"]),
            }
        )
    for role, path, recorded in (
        ("frozen_config", config_path, "frozen_config.yaml"),
        ("target_plan", plan_path, "plan.json"),
        ("input_manifest", input_manifest_path, "input_manifest.csv"),
        ("validation", validation_path, "validation.csv"),
        ("validation_summary", validation_summary_path, "validation_summary.json"),
        *((name.removesuffix(".csv"), path, intended_path(name)) for name, path in paths.items()),
        ("score_provenance", score_provenance_path, intended_path("score_provenance.json")),
    ):
        artifacts.append(
            artifact_record(str(config["configuration_id"]), role, path, recorded)
        )
    pd.DataFrame(artifacts).to_csv(work_dir / "artifact_manifest.csv", index=False)
    restored_provenance = read_json(score_provenance_path)
    if restored_provenance.get("status") != "complete":
        raise ValueError("score provenance did not round-trip as complete")
    for filename, expected in restored_provenance["outputs"].items():
        require_sha(work_dir / filename, expected, f"fresh score output {filename}")
    restored_manifest = pd.read_csv(work_dir / "artifact_manifest.csv")
    if restored_manifest.empty or restored_manifest["sha256"].isna().any():
        raise ValueError("artifact manifest did not round-trip with complete hashes")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    evaluation_dir = output_dir / "evaluation"
    work_dir = output_dir / ".work" / "evaluation"
    if evaluation_dir.exists() or work_dir.exists():
        raise FileExistsError("fresh-only evaluation path already exists")
    work_dir.mkdir(parents=True)
    try:
        evaluate(output_dir, work_dir)
        os.replace(work_dir, evaluation_dir)
    except Exception as exc:
        write_json(
            work_dir / "FAILED.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
    print(f"Wrote atomic evaluation for 93 validated targets to {evaluation_dir}")


if __name__ == "__main__":
    main()
