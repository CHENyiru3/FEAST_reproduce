#!/usr/bin/env python3
"""Validate the additive v3 article contracts and their current evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = Path(__file__).resolve().parent / "contracts.json"
EXPECTED_IDS = (
    "FEAST-ART-00-SIM-v3",
    "FEAST-ART-01-CLUSTER-v3",
    "FEAST-ART-02-ALIGN-v3",
    "FEAST-ART-03-DECONV-v3",
    "FEAST-ART-04-2D-XSLICE-v3",
    "FEAST-ART-05-3D-IMPUTE-v3",
    "FEAST-ART-06-DEVCCF-E15-v3",
    "FEAST-ART-08-BATCH-v3",
)
AUTHORIZATION_KEYS = {"publication_claim", "figure", "composite_score", "method_ranking", "winner"}


class ContractError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def validate_record(record: dict) -> None:
    workflow_id = str(record.get("workflow_id"))
    require(record.get("clean_rerun_study"), f"{workflow_id}: clean study missing")
    require(str(record.get("status", "")), f"{workflow_id}: status missing")
    require(str(record.get("configuration_id", "")), f"{workflow_id}: configuration ID missing")

    predecessor = record.get("supersedes", {})
    predecessor_path = str(predecessor.get("path", ""))
    require(predecessor_path.startswith("publication/method_contract_v2/configs/"), f"{workflow_id}: invalid predecessor")
    predecessor_file = ROOT / predecessor_path
    require(predecessor_file.is_file(), f"{workflow_id}: predecessor missing")
    require(sha256(predecessor_file) == predecessor.get("sha256"), f"{workflow_id}: predecessor hash changed")

    evidence = record.get("evidence")
    require(isinstance(evidence, list) and len(evidence) >= 2, f"{workflow_id}: insufficient evidence")
    for item in evidence:
        relative = str(item.get("path", ""))
        require(relative and not Path(relative).is_absolute() and ".." not in Path(relative).parts, f"{workflow_id}: nonportable evidence path")
        path = ROOT / relative
        require(path.is_file(), f"{workflow_id}: evidence missing: {relative}")
        require(sha256(path) == item.get("sha256"), f"{workflow_id}: evidence hash changed: {relative}")

    authorizations = record.get("claim_authorizations", {})
    require(set(authorizations) == AUTHORIZATION_KEYS, f"{workflow_id}: authorization schema changed")
    require(not any(authorizations.values()), f"{workflow_id}: a claim was authorized")
    release = record.get("release", {})
    require(release == {"authorized": False, "tag": None, "pypi_publication": False}, f"{workflow_id}: release state changed")
    require(record.get("claim_limits"), f"{workflow_id}: claim limits missing")


def validate_study00() -> None:
    path = ROOT / "00_simulator_benchmark/outputs/final_metrics/simulator_quality_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 60, "Study 00: final metric row count is not 60")
    selected = [row for row in rows if row["simulator"] == "scCube" and row["sample"] == "Slideseq_001"]
    require(len(selected) == 1, "Study 00: scCube Slide-seq row missing")
    row = selected[0]
    require(int(float(row["n_paired_spots"])) == 35054, "Study 00: scCube Slide-seq support incomplete")
    require(row["spot_pairing"] == "exact_identifier", "Study 00: scCube Slide-seq pairing is not named")
    require(math.isfinite(float(row["cosine_divergence"])), "Study 00: cosine divergence missing")
    require(math.isfinite(float(row["zero_mask_jaccard"])), "Study 00: zero-mask Jaccard missing")


def validate_study06() -> None:
    payload = json.loads((ROOT / "06_3d_stack/outputs/final/validation_summary.json").read_text(encoding="utf-8"))
    require(payload.get("status") == "passed", "Study 06: validation did not pass")
    require(payload.get("expected_outputs") == 93 and payload.get("validated_outputs") == 93, "Study 06: output count changed")
    require(payload.get("all_positive_convergence") is True, "Study 06: positive convergence missing")


def validate_study07() -> None:
    payload = json.loads((ROOT / "07_3d_transfer/outputs/final/validation.json").read_text(encoding="utf-8"))
    require(payload.get("status") == "validated", "Study 07: validation did not pass")
    require(payload.get("publication_canonical") is False, "Study 07: candidate was promoted")
    ages = {item["age"]: item for item in payload.get("ages", [])}
    require(set(ages) == {"E15.5", "E18.5"}, "Study 07: age scope changed")
    require(ages["E15.5"].get("n_z_levels") == 158 and ages["E18.5"].get("n_z_levels") == 202, "Study 07: z scope changed")
    require(sum(int(item["transport_records"]) for item in ages.values()) == 12290, "Study 07: transport count changed")
    require(all(item.get("all_transport_records_converged") and item.get("exact_blueprint_identity") for item in ages.values()), "Study 07: convergence or identity failed")


def validate_manifest(records: list[dict]) -> None:
    manifest = json.loads((ROOT / "PUBLICATION_MANIFEST.json").read_text(encoding="utf-8"))
    require(manifest.get("release_authorized") is False and manifest.get("final_tag") is None, "publication release became authorized")
    studies = manifest.get("studies", {})
    require(studies["06_3d_stack"]["canonical_candidate"]["configuration_id"] == records[5]["configuration_id"], "Study 06 manifest config mismatch")
    require(studies["07_3d_transfer"]["canonical_candidate"]["configuration_id"] == records[6]["configuration_id"], "Study 07 manifest config mismatch")


def main() -> int:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    require(payload.get("schema_version") == 1, "v3 schema version changed")
    require(payload.get("contract_version") == "3.0-addendum", "v3 contract version changed")
    require(payload.get("release_authorized") is False, "v3 release became authorized")
    records = payload.get("workflows")
    require(isinstance(records, list), "v3 workflows must be a list")
    require(tuple(record.get("workflow_id") for record in records) == EXPECTED_IDS, "v3 workflow order/scope changed")
    for record in records:
        validate_record(record)
    validate_study00()
    validate_study06()
    validate_study07()
    validate_manifest(records)
    print("method contract v3: OK (8/8; release and claims remain blocked)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as error:
        print(f"method contract v3: FAILED: {error}")
        raise SystemExit(1)
