#!/usr/bin/env python3
"""Shared, experiment-owned contracts for clean Study 06 execution."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


STUDY_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = STUDY_ROOT / "config.yaml"
ZERO_Z_EPSILON = 1.0e-6
REQUIRED_WHEEL_SHA256 = "9dd912d883a03d51ed7105cecd914cf8f7f25f5355f57dfde1c77b6fef0b056b"


@dataclass(frozen=True)
class SliceInfo:
    slice_id: int
    path: Path
    z: float
    n_obs: int
    n_vars: int
    sha256: str = ""


@dataclass(frozen=True)
class DensitySpec:
    name: str
    gap: int
    radius: int
    start: int
    expected_targets: int
    expected_references: int
    expected_assignment_randomness: float


@dataclass(frozen=True)
class TargetAssignment:
    density_name: str
    gap: int
    target_index: int
    target_slice: int
    lower_ref_slice: int
    upper_ref_slice: int
    target_z: float
    z0: float
    z1: float
    tau: float
    reference_gap_z: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must contain a YAML mapping")
    required = {
        "configuration_id",
        "study_id",
        "public_seed",
        "feast_version",
        "required_feast_commit",
        "required_wheel_sha256",
        "dataset",
        "densities",
        "reference_fit",
        "generation",
        "transport",
        "validation",
        "evaluation",
        "historical_comparison",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"config is missing fields: {sorted(missing)}")
    if str(config["study_id"]) != "06":
        raise ValueError("study_id must remain '06'")
    validate_fixed_contract(config)
    return config


def validate_fixed_contract(config: Mapping[str, Any]) -> None:
    generation = config["generation"]
    transport = config["transport"]
    dataset = config["dataset"]
    if int(dataset["expected_genes"]) != 1122:
        raise ValueError("Study 06 must retain the declared 1,122-gene panel")
    if int(config["reference_fit"]["min_gene_spots"]) != 0:
        raise ValueError("reference_fit.min_gene_spots must remain zero to retain all 1,122 genes")
    if str(config["required_wheel_sha256"]) != REQUIRED_WHEEL_SHA256:
        raise ValueError("required_wheel_sha256 differs from the publication wheel")
    expected = {3: (49, 95, 0.35), 5: (29, 57, 0.25), 10: (15, 16, 0.35)}
    observed = {
        int(row["gap"]): (
            int(row["expected_targets"]),
            int(row["expected_references"]),
            float(row["expected_assignment_randomness"]),
        )
        for row in config["densities"]
    }
    if observed != expected:
        raise ValueError(f"density contract changed: {observed!r}")
    required_transport = {
        "sinkhorn_iter": 1000,
        "sinkhorn_tol": 1.0e-5,
        "transport_nonconvergence": "raise",
        "max_transport_pairs": 25_000_000,
    }
    for key, expected_value in required_transport.items():
        if transport.get(key) != expected_value:
            raise ValueError(f"transport.{key} must remain {expected_value!r}")
    if generation.get("target_expression_access") != "evaluation_only":
        raise ValueError("target expression must remain evaluation-only")
    if generation.get("smoothing") is not False or generation.get("z_regularization") is not False:
        raise ValueError("Study 06 prohibits smoothing and z regularization")
    evaluation = config["evaluation"]
    if evaluation.get("targets_are_biological_replicates") is not False:
        raise ValueError("the 93 targets must not be treated as biological replicates")
    baseline = evaluation.get("real_continuity_baseline", {})
    if int(baseline.get("minimum_class_spots", -1)) != 2 or int(baseline.get("minimum_z_levels", -1)) != 3:
        raise ValueError("real split-half continuity baseline contract changed")
    historical = config["historical_comparison"]
    if historical.get("expected_sha256") != "639b298ed6d093dda24417f1fa04a813557aaab217e338a3ffab9c54456e1b25":
        raise ValueError("historical comparison artifact hash changed")
    if int(historical.get("historical_sinkhorn_iter", -1)) != 200:
        raise ValueError("historical Sinkhorn iteration contract changed")


def density_specs(config: Mapping[str, Any]) -> list[DensitySpec]:
    return [DensitySpec(**row) for row in config["densities"]]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_strings(values: Iterable[Any], prefix: str) -> str:
    digest = hashlib.sha256()
    digest.update(prefix.encode("utf-8") + b"\0")
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def sha256_array(values: Any, prefix: str) -> str:
    array = np.asarray(values)
    if array.dtype.kind in {"O", "U", "S"}:
        return sha256_strings(array.reshape(-1), f"{prefix}:{array.shape}")
    canonical = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(f"{prefix}:{canonical.shape}".encode("utf-8") + b"\0")
    digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def adjusted_reference_z(slice_id: int, z: float) -> float:
    if 1 <= int(slice_id) <= 4 and math.isclose(float(z), 0.0, rel_tol=0.0, abs_tol=1e-12):
        return float((int(slice_id) - 1) * ZERO_Z_EPSILON)
    return float(z)


def build_assignments(
    slice_infos: Mapping[int, SliceInfo],
    spec: DensitySpec,
    public_seed: int,
) -> tuple[list[TargetAssignment], list[int]]:
    available = set(slice_infos)
    if not available:
        raise ValueError("no input slices were discovered")
    minimum, maximum = min(available), max(available)
    target_ids = [item for item in range(spec.start, maximum + 1, spec.gap) if item in available]
    rows: list[TargetAssignment] = []
    references: set[int] = set()
    for target_index, target_id in enumerate(target_ids):
        lower_nominal = target_id - spec.radius
        upper_nominal = target_id + spec.radius
        lower = (
            minimum
            if lower_nominal < minimum and minimum < target_id
            else snap_lower(available, lower_nominal, minimum)
        )
        upper = (
            maximum
            if upper_nominal > maximum and maximum > target_id
            else snap_upper(available, upper_nominal, maximum)
        )
        if lower is None or upper is None or not lower < target_id < upper:
            raise ValueError(f"target {target_id:03d} cannot be strictly bracketed for gap {spec.gap}")
        target_z = float(slice_infos[target_id].z)
        z0 = adjusted_reference_z(lower, slice_infos[lower].z)
        z1 = adjusted_reference_z(upper, slice_infos[upper].z)
        if not z0 < target_z < z1:
            raise ValueError(f"target {target_id:03d} z is not strictly inside its reference interval")
        tau = (target_z - z0) / (z1 - z0)
        rows.append(
            TargetAssignment(
                density_name=spec.name,
                gap=spec.gap,
                target_index=target_index,
                target_slice=target_id,
                lower_ref_slice=lower,
                upper_ref_slice=upper,
                target_z=target_z,
                z0=z0,
                z1=z1,
                tau=float(tau),
                reference_gap_z=float(z1 - z0),
                seed=int(public_seed) + int(spec.gap) * 10_000 + target_index,
            )
        )
        references.update((lower, upper))
    reference_ids = sorted(references)
    if len(rows) != spec.expected_targets:
        raise ValueError(f"{spec.name} expected {spec.expected_targets} targets, found {len(rows)}")
    if len(reference_ids) != spec.expected_references:
        raise ValueError(
            f"{spec.name} expected {spec.expected_references} references, found {len(reference_ids)}"
        )
    return rows, reference_ids


def snap_lower(available: set[int], start: int, minimum: int) -> int | None:
    return next((item for item in range(int(start), int(minimum) - 1, -1) if item in available), None)


def snap_upper(available: set[int], start: int, maximum: int) -> int | None:
    return next((item for item in range(int(start), int(maximum) + 1) if item in available), None)


def representative_reference_ids(reference_ids: Sequence[int], count: int) -> list[int]:
    values = sorted({int(item) for item in reference_ids})
    if len(values) <= count:
        return values
    indices = sorted({int(round(position)) for position in np.linspace(0, len(values) - 1, count)})
    return [values[index] for index in indices]


def choose_donor(
    label: str,
    target_slice: int,
    target_z: float,
    label_support: Mapping[str, Sequence[int]],
    slice_infos: Mapping[int, SliceInfo],
) -> int:
    candidates = [int(item) for item in label_support.get(str(label), []) if int(item) != int(target_slice)]
    if not candidates:
        raise ValueError(f"no declared reference donor supports label {label!r} for target {target_slice:03d}")
    distances = {
        item: abs(adjusted_reference_z(item, slice_infos[item].z) - float(target_z))
        for item in candidates
    }
    minimum = min(distances.values())
    tied = [item for item, distance in distances.items() if math.isclose(distance, minimum, rel_tol=0.0, abs_tol=1e-12)]
    return min(tied)


def verify_clean_wheel(config: Mapping[str, Any]) -> dict[str, Any]:
    verifier = STUDY_ROOT.parent / "scripts" / "verify_feast_install.py"
    completed = subprocess.run(
        [sys.executable, str(verifier)],
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(completed.stdout)
    expected = {
        "status": "OK",
        "version": str(config["feast_version"]),
        "commit": str(config["required_feast_commit"]),
        "wheel_sha256": str(config["required_wheel_sha256"]),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f"clean FEAST wheel verifier field {key!r} changed")
    imported = Path(str(record["import_path"]))
    if "site-packages" not in imported.parts:
        raise RuntimeError(f"FEAST is not imported from an installed wheel: {imported}")
    return record


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_sha(path: Path, expected: str, description: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{description} hash changed: expected {expected}, observed {actual}")
