#!/usr/bin/env python3
"""Verify the frozen cross-study contract for clean Studies 05--07."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.2"
COMMIT = "68816e5c1862a6fa2a49bc30609d617c7fa4b449"
WHEEL_SHA256 = "9dd912d883a03d51ed7105cecd914cf8f7f25f5355f57dfde1c77b6fef0b056b"
SHARED_TRANSPORT = {
    "epsilon": 0.05,
    "sinkhorn_iter": 1000,
    "sinkhorn_tol": 1.0e-5,
    "unbalanced_transport": True,
    "reg_m": 5.0,
    "transport_nonconvergence": "raise",
    "geometry_weight": 1.0,
    "boundary_weight": 0.25,
    "max_transport_pairs": 25_000_000,
}


def read_yaml(relative: str) -> dict[str, Any]:
    payload = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{relative} must contain a mapping")
    return payload


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def check_identity(name: str, config: dict[str, Any], errors: list[str]) -> None:
    version = config.get("required_feast_version", config.get("feast_version"))
    require(str(version) == VERSION, f"{name}: FEAST version changed", errors)
    require(
        str(config.get("required_feast_commit")) == COMMIT,
        f"{name}: FEAST commit changed",
        errors,
    )
    require(
        str(config.get("required_wheel_sha256")) == WHEEL_SHA256,
        f"{name}: installed-wheel hash is not frozen",
        errors,
    )
    require(int(config.get("public_seed", -1)) == 2026, f"{name}: public seed changed", errors)
    for key, expected in SHARED_TRANSPORT.items():
        require(
            config.get("transport", {}).get(key) == expected,
            f"{name}: transport.{key} changed",
            errors,
        )


def main() -> int:
    errors: list[str] = []
    study05 = read_yaml("05_2d_conditional_transfer/config.yaml")
    study06 = read_yaml("06_3d_stack/config.yaml")
    study07 = read_yaml("07_3d_transfer/config.yaml")
    for name, config in (
        ("Study 05", study05),
        ("Study 06", study06),
        ("Study 07", study07),
    ):
        check_identity(name, config, errors)

    study05_datasets = study05.get("datasets", {})
    require(
        set(study05_datasets) == {"dlpfc", "merfish"},
        "Study 05: dataset scope changed",
        errors,
    )
    require(
        len(study05_datasets.get("dlpfc", {}).get("directions", [])) == 6
        and len(study05_datasets.get("merfish", {}).get("directions", [])) == 2,
        "Study 05: direction count changed",
        errors,
    )
    require(
        list(map(float, study05.get("assignment_randomness", []))) == [0.0, 0.1, 0.2, 0.3, 0.5],
        "Study 05: assignment-randomness grid changed",
        errors,
    )
    require(
        int(study05_datasets.get("dlpfc", {}).get("gene_panel", {}).get("expected_genes", -1))
        == 17_391
        and int(
            study05_datasets.get("merfish", {}).get("gene_panel", {}).get(
                "expected_genes", -1
            )
        )
        == 1_122,
        "Study 05: dataset gene-panel sizes changed",
        errors,
    )
    require(
        int(study05.get("expected_jobs", -1)) == 45
        and int(study05.get("expected_cross_slice_jobs", -1)) == 40
        and int(study05.get("expected_mask_half_jobs", -1)) == 5,
        "Study 05: expected job counts changed",
        errors,
    )
    mask_half = study05.get("mask_half", {})
    require(
        int(mask_half.get("split_axis", -1)) == 0
        and float(mask_half.get("split_quantile", -1)) == 0.5
        and mask_half.get("direction") == "low_x_to_high_x"
        and float(mask_half.get("assignment_randomness", -1)) == 0.3,
        "Study 05: half-slice contract changed",
        errors,
    )
    require(
        int(study05.get("label_support", {}).get("min_source_spots", -1)) == 20,
        "Study 05: source label-support threshold changed",
        errors,
    )
    require(
        int(study05.get("reference_fit", {}).get("min_gene_spots", -1)) == 0,
        "Study 05: zero-evidence gene retention changed",
        errors,
    )

    density_contract = {
        int(item["gap"]): (int(item["expected_targets"]), float(item["expected_assignment_randomness"]))
        for item in study06.get("densities", [])
    }
    require(
        density_contract == {3: (49, 0.35), 5: (29, 0.25), 10: (15, 0.35)},
        "Study 06: density scope changed",
        errors,
    )
    require(
        int(study06.get("validation", {}).get("expected_outputs", -1)) == 93,
        "Study 06: expected output count changed",
        errors,
    )
    generation = study06.get("generation", {})
    require(generation.get("smoothing") is False, "Study 06: smoothing was enabled", errors)
    require(generation.get("z_regularization") is False, "Study 06: z regularization was enabled", errors)

    expected_ages = {
        "E15.5": (-4.12, -0.98, 0.02, 158, 2_330_927),
        "E18.5": (-5.56, -1.54, 0.02, 202, 5_213_461),
    }
    observed_ages = {
        age: (
            float(item["z_start"]),
            float(item["z_end"]),
            float(item["z_step"]),
            int(item["expected_levels"]),
            int(item["expected_spots"]),
        )
        for age, item in study07.get("ages", {}).items()
    }
    require(observed_ages == expected_ages, "Study 07: full-axis scope changed", errors)
    require(
        study07.get("ages", {}).get("E18.5", {}).get("assignment_randomness", {}).get("mode")
        == "reference_calibration",
        "Study 07: E18.5 is not reference-only calibrated",
        errors,
    )
    study07_fit = study07.get("reference_fit", {})
    require(
        int(study07_fit.get("expected_genes", -1)) == 550
        and int(study07_fit.get("min_gene_spots", -1)) == 1
        and float(study07_fit.get("min_gene_mean", -1)) == 0.0
        and float(study07_fit.get("max_gene_zero_prop", -1)) == 1.0,
        "Study 07: historical E15.5 550-gene reference-fit contract changed",
        errors,
    )

    publication = json.loads((ROOT / "PUBLICATION_MANIFEST.json").read_text(encoding="utf-8"))
    studies = publication.get("studies", {})
    for study in ("05_2d_conditional_transfer", "06_3d_stack", "07_3d_transfer"):
        record = studies.get(study, {})
        require(bool(record), f"publication manifest lacks {study}", errors)
        require(
            record.get("historical_conditional_outputs_consumed") is False,
            f"{study}: historical conditional outputs were marked as consumed",
            errors,
        )

    if errors:
        print("conditional-workflow contract: FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("conditional-workflow contract: OK (45 + 93 + 360 fresh H5ADs declared)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
