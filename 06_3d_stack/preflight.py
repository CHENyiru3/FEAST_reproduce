#!/usr/bin/env python3
"""Freeze Study 06 inputs, target assignments, donors, and AR estimates."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from workflow import (
    CONFIG_PATH,
    SliceInfo,
    adjusted_reference_z,
    build_assignments,
    choose_donor,
    density_specs,
    load_config,
    representative_reference_ids,
    sha256_array,
    sha256_file,
    sha256_strings,
    verify_clean_wheel,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


def inspect_inputs(
    data_dir: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[int, SliceInfo], dict[int, set[str]], dict[int, Counter[str]], list[str]]:
    dataset = config["dataset"]
    prefix = str(dataset["filename_prefix"])
    paths = sorted(data_dir.glob(f"{prefix}.*.h5ad"))
    expected_ids = set(range(int(dataset["first_slice"]), int(dataset["last_slice"]) + 1))
    expected_ids -= {int(item) for item in dataset["intentionally_absent_slices"]}
    observed_ids = {slice_id_from_path(path, prefix) for path in paths}
    if observed_ids != expected_ids:
        raise ValueError(
            "source slice IDs changed; "
            f"missing={sorted(expected_ids - observed_ids)}, extra={sorted(observed_ids - expected_ids)}"
        )
    if len(paths) != int(dataset["expected_files"]):
        raise ValueError(f"expected {dataset['expected_files']} input files, found {len(paths)}")

    records: list[dict[str, Any]] = []
    slice_infos: dict[int, SliceInfo] = {}
    label_sets: dict[int, set[str]] = {}
    label_counts: dict[int, Counter[str]] = {}
    canonical_genes: list[str] | None = None
    for path in paths:
        slice_id = slice_id_from_path(path, prefix)
        artifact_sha = sha256_file(path)
        adata = ad.read_h5ad(path, backed="r")
        try:
            validate_adata_metadata(adata, path, config)
            genes = list(map(str, adata.var_names))
            if canonical_genes is None:
                canonical_genes = genes
            elif genes != canonical_genes:
                raise ValueError(f"{path.name} gene names/order differ from the first input")
            labels = adata.obs[str(dataset["label_key"])].astype(str).tolist()
            counts = Counter(labels)
            z_values = np.asarray(adata.obs[str(dataset["z_key"])], dtype=float)
            z_value = float(z_values[0])
            spatial = np.asarray(adata.obsm[str(dataset["spatial_key"])], dtype=float)
            spatial_3d = np.asarray(adata.obsm[str(dataset["spatial_3d_key"])], dtype=float)
            records.append(
                {
                    "artifact_id": f"{prefix}.{slice_id:03d}",
                    "slice_id": slice_id,
                    "filename": path.name,
                    "sha256": artifact_sha,
                    "bytes": path.stat().st_size,
                    "n_obs": int(adata.n_obs),
                    "n_vars": int(adata.n_vars),
                    "z": z_value,
                    "obs_names_sha256": sha256_strings(adata.obs_names, "study06-obs-names-v1"),
                    "var_names_sha256": sha256_strings(adata.var_names, "study06-var-names-v1"),
                    "class_sha256": sha256_strings(labels, "study06-class-v1"),
                    "spatial_sha256": sha256_array(spatial, "study06-spatial-v1"),
                    "spatial_3d_sha256": sha256_array(spatial_3d, "study06-spatial-3d-v1"),
                }
            )
            slice_infos[slice_id] = SliceInfo(
                slice_id=slice_id,
                path=path.resolve(),
                z=z_value,
                n_obs=int(adata.n_obs),
                n_vars=int(adata.n_vars),
                sha256=artifact_sha,
            )
            label_sets[slice_id] = set(counts)
            label_counts[slice_id] = counts
        finally:
            adata.file.close()
    assert canonical_genes is not None
    return pd.DataFrame(records).sort_values("slice_id"), slice_infos, label_sets, label_counts, canonical_genes


def validate_adata_metadata(adata: ad.AnnData, path: Path, config: dict[str, Any]) -> None:
    dataset = config["dataset"]
    if int(adata.n_vars) != int(dataset["expected_genes"]):
        raise ValueError(f"{path.name} expected {dataset['expected_genes']} genes, found {adata.n_vars}")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError(f"{path.name} has non-unique observation or gene names")
    for key in (dataset["label_key"], dataset["z_key"]):
        if str(key) not in adata.obs:
            raise KeyError(f"{path.name} is missing obs[{key!r}]")
    for key in (dataset["spatial_key"], dataset["spatial_3d_key"]):
        if str(key) not in adata.obsm:
            raise KeyError(f"{path.name} is missing obsm[{key!r}]")
    if str(dataset["counts_layer"]) not in adata.layers:
        raise KeyError(f"{path.name} is missing layers[{dataset['counts_layer']!r}]")
    raw_labels = adata.obs[str(dataset["label_key"])]
    labels = raw_labels.astype(str)
    if raw_labels.isna().any() or (labels.str.len() == 0).any():
        raise ValueError(f"{path.name} contains missing/empty class labels")
    z_values = np.asarray(adata.obs[str(dataset["z_key"])], dtype=float)
    if not np.all(np.isfinite(z_values)) or not np.allclose(z_values, z_values[0], rtol=0.0, atol=1e-8):
        raise ValueError(f"{path.name} has invalid or nonconstant z")
    spatial = np.asarray(adata.obsm[str(dataset["spatial_key"])], dtype=float)
    spatial_3d = np.asarray(adata.obsm[str(dataset["spatial_3d_key"])], dtype=float)
    if spatial.shape != (adata.n_obs, 2) or spatial_3d.shape != (adata.n_obs, 3):
        raise ValueError(f"{path.name} has an invalid spatial coordinate shape")
    if not np.all(np.isfinite(spatial)) or not np.all(np.isfinite(spatial_3d)):
        raise ValueError(f"{path.name} has non-finite spatial coordinates")
    if not np.array_equal(spatial, spatial_3d[:, :2]):
        raise ValueError(f"{path.name} spatial XY differs from spatial_3d XY")
    if not np.allclose(spatial_3d[:, 2], z_values, rtol=0.0, atol=1e-8):
        raise ValueError(f"{path.name} spatial_3d z differs from obs z")


def slice_id_from_path(path: Path, prefix: str) -> int:
    expected = f"{prefix}."
    if not path.stem.startswith(expected):
        raise ValueError(f"unexpected input filename: {path.name}")
    try:
        return int(path.stem[len(expected) :])
    except ValueError as exc:
        raise ValueError(f"cannot parse slice ID from {path.name}") from exc


def plan_density(
    spec,
    assignments,
    reference_ids,
    slice_infos,
    label_sets,
    label_counts,
    prefix: str,
) -> list[dict[str, Any]]:
    label_support: dict[str, list[int]] = {}
    for reference_id in reference_ids:
        for label in label_sets[reference_id]:
            label_support.setdefault(label, []).append(reference_id)
    for values in label_support.values():
        values.sort()

    targets: list[dict[str, Any]] = []
    for assignment in assignments:
        bracket_labels = label_sets[assignment.lower_ref_slice] | label_sets[assignment.upper_ref_slice]
        missing_labels = sorted(label_sets[assignment.target_slice] - bracket_labels)
        donors: list[dict[str, Any]] = []
        for label in missing_labels:
            donor_id = choose_donor(
                label,
                assignment.target_slice,
                assignment.target_z,
                label_support,
                slice_infos,
            )
            donor_z = adjusted_reference_z(donor_id, slice_infos[donor_id].z)
            donors.append(
                {
                    "label": label,
                    "donor_slice": donor_id,
                    "donor_z": donor_z,
                    "side": "lower" if donor_z <= assignment.target_z else "upper",
                    "n_spots": int(label_counts[donor_id][label]),
                    "artifact_id": f"{prefix}.{donor_id:03d}",
                    "sha256": slice_infos[donor_id].sha256,
                }
            )
        row = assignment.to_dict()
        row.update(
            {
                "target_artifact_id": f"{prefix}.{assignment.target_slice:03d}",
                "target_input_sha256": slice_infos[assignment.target_slice].sha256,
                "lower_reference_artifact_id": f"{prefix}.{assignment.lower_ref_slice:03d}",
                "lower_reference_sha256": slice_infos[assignment.lower_ref_slice].sha256,
                "upper_reference_artifact_id": f"{prefix}.{assignment.upper_ref_slice:03d}",
                "upper_reference_sha256": slice_infos[assignment.upper_ref_slice].sha256,
                "reference_weights": {
                    f"{prefix}.{assignment.lower_ref_slice:03d}": float(1.0 - assignment.tau),
                    f"{prefix}.{assignment.upper_ref_slice:03d}": float(assignment.tau),
                },
                "donor_support": donors,
            }
        )
        targets.append(row)
    return targets


def estimate_density_ar(spec, reference_ids, slice_infos, config) -> dict[str, Any]:
    from FEAST import estimate_assignment_randomness

    settings = config["assignment_randomness_preflight"]
    selected = representative_reference_ids(reference_ids, int(settings["representative_reference_slices"]))
    estimates: list[dict[str, Any]] = []
    dataset = config["dataset"]
    for index, reference_id in enumerate(selected):
        reference = ad.read_h5ad(slice_infos[reference_id].path)
        try:
            reference.X = reference.layers[str(dataset["counts_layer"])].copy()
            reference.obsm["spatial"] = np.asarray(reference.obsm[str(dataset["spatial_key"])], dtype=float)
            # The estimator's held-out blueprint is 2D. Remove the parallel
            # 3D key so reference fitting uses the same XY dimensionality.
            if "spatial_3d" in reference.obsm:
                del reference.obsm["spatial_3d"]
            seed = int(config["public_seed"]) + int(spec.gap) * 1_000 + index
            value = estimate_assignment_randomness(
                reference,
                label_key=str(dataset["label_key"]),
                spatial_key="spatial",
                n_genes=int(settings["n_genes"]),
                ar_step=float(settings["step"]),
                max_ar=float(settings["maximum"]),
                n_neighbors=int(settings["spatial_neighbors"]),
                random_seed=seed,
            )
        finally:
            del reference
        estimates.append({"reference_slice": reference_id, "seed": seed, "assignment_randomness": float(value)})
    observed = float(np.median([item["assignment_randomness"] for item in estimates]))
    return {
        "mode": "reference_only_representative_median",
        "representative_estimates": estimates,
        "observed_assignment_randomness": observed,
        "expected_assignment_randomness": float(spec.expected_assignment_randomness),
        "matched": bool(
            np.isclose(
                observed,
                float(spec.expected_assignment_randomness),
                rtol=0.0,
                atol=float(settings["agreement_tolerance"]),
            )
        ),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    config = load_config(config_path)
    if output_dir.exists():
        raise FileExistsError(f"preflight requires a new output root: {output_dir}")
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    runtime = verify_clean_wheel(config)
    import FEAST

    if FEAST.__version__ != str(config["feast_version"]):
        raise ValueError(f"FEAST version mismatch: expected {config['feast_version']}, observed {FEAST.__version__}")

    output_dir.mkdir(parents=True)
    shutil.copyfile(config_path, output_dir / "frozen_config.yaml")
    manifest, slice_infos, label_sets, label_counts, genes = inspect_inputs(data_dir, config)
    manifest_path = output_dir / "input_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    prefix = str(config["dataset"]["filename_prefix"])
    plan: dict[str, Any] = {
        "configuration_id": config["configuration_id"],
        "config_sha256": sha256_file(config_path),
        "feast_version": FEAST.__version__,
        "feast_commit": runtime["commit"],
        "wheel_sha256": runtime["wheel_sha256"],
        "data_dir": str(data_dir),
        "input_manifest_sha256": sha256_file(manifest_path),
        "gene_names": genes,
        "gene_names_sha256": sha256_strings(genes, "study06-var-names-v1"),
        "densities": {},
    }
    for spec in density_specs(config):
        assignments, reference_ids = build_assignments(slice_infos, spec, int(config["public_seed"]))
        ar = estimate_density_ar(spec, reference_ids, slice_infos, config)
        plan["densities"][spec.name] = {
            "spec": {
                "name": spec.name,
                "gap": spec.gap,
                "radius": spec.radius,
                "start": spec.start,
                "expected_targets": spec.expected_targets,
                "expected_references": spec.expected_references,
            },
            "reference_pool": reference_ids,
            "assignment_randomness_preflight": ar,
            "targets": plan_density(
                spec,
                assignments,
                reference_ids,
                slice_infos,
                label_sets,
                label_counts,
                prefix,
            ),
        }
    plan_path = output_dir / "plan.json"
    write_json(plan_path, plan)

    failed = [name for name, density in plan["densities"].items() if not density["assignment_randomness_preflight"]["matched"]]
    common = {
        "configuration_id": config["configuration_id"],
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "frozen_config_sha256": sha256_file(output_dir / "frozen_config.yaml"),
        "input_manifest_sha256": sha256_file(manifest_path),
        "plan_sha256": sha256_file(plan_path),
        "feast_version": FEAST.__version__,
        "feast_commit": runtime["commit"],
        "wheel_sha256": runtime["wheel_sha256"],
        "feast_module_path": str(Path(FEAST.__file__).resolve()),
        "verified_package_files": int(runtime["verified_package_files"]),
        "data_dir": str(data_dir),
        "expected_outputs": int(config["validation"]["expected_outputs"]),
    }
    if failed:
        write_json(
            output_dir / "PREFLIGHT_FAILED.json",
            {**common, "status": "failed", "reason": "assignment_randomness_mismatch", "densities": failed},
        )
        raise RuntimeError(f"assignment-randomness preflight mismatch for {failed}; output root is non-runnable")
    write_json(output_dir / "preflight.json", {**common, "status": "passed"})
    print(f"Preflight passed: {len(manifest)} inputs and 93 target assignments frozen in {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
