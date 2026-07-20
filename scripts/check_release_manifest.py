#!/usr/bin/env python3
"""Validate the clean rerun lineage without requiring scientific outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "PUBLICATION_MANIFEST.json"
BUILD_RECORD_PATH = ROOT / "FEAST_BUILD.txt"
STUDIES = (
    "00_simulator_benchmark",
    "01_clustering",
    "02_alignment",
    "03_deconvolution",
    "04_batch_effect_removal",
    "05_2d_conditional_transfer",
    "06_3d_stack",
    "07_3d_transfer",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_record_fields() -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in BUILD_RECORD_PATH.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            fields[key] = value
    return fields


def required_commit_from_config(path: Path) -> str:
    match = re.search(
        r"(?m)^required_feast_commit:\s*[\"']?([0-9a-f]{40})[\"']?\s*$",
        path.read_text(encoding="utf-8"),
    )
    if match is None:
        raise ValueError(f"{path.relative_to(ROOT)} lacks required_feast_commit")
    return match.group(1)


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("publication manifest schema_version must be 1")
    if manifest.get("status") not in {"in_progress", "approved"}:
        raise ValueError("publication manifest has an invalid status")
    if not isinstance(manifest.get("release_authorized"), bool):
        raise ValueError("release_authorized must be boolean")

    studies = manifest.get("studies", {})
    if tuple(studies) != STUDIES:
        raise ValueError("publication manifest study order/scope is invalid")
    for study in STUDIES:
        if not (ROOT / study).is_dir():
            raise FileNotFoundError(f"missing active study: {study}")

    if manifest["release_authorized"]:
        incomplete = [
            study
            for study, record in studies.items()
            if record.get("status") != "validated_complete"
        ]
        if manifest["status"] != "approved" or incomplete or not manifest.get("final_tag"):
            raise ValueError(
                "authorized release requires approved status, a final tag, and complete studies"
            )
    elif manifest.get("final_tag") is not None:
        raise ValueError("an unauthorized release cannot declare a final tag")

    build = manifest["article_feast_build"]
    record = build_record_fields()
    expected = {
        "version": record["Package version"].split()[0],
        "source_commit": record["Required source commit"],
        "python": record["Required Python"],
        "numpy": record["Required NumPy"],
        "wheel_sha256": record["Wheel SHA-256"],
        "sdist_sha256": record["Source distribution SHA-256"],
    }
    observed = {key: str(build[key]) for key in expected}
    if observed != expected:
        raise ValueError(f"FEAST build record mismatch: {observed} != {expected}")

    for artifact_key, digest_key in (
        ("wheel_file", "wheel_sha256"),
        ("sdist_file", "sdist_sha256"),
    ):
        artifact = ROOT / build[artifact_key]
        if artifact.exists() and sha256(artifact) != build[digest_key]:
            raise ValueError(f"local artifact hash mismatch: {artifact.relative_to(ROOT)}")

    for study in STUDIES:
        config = ROOT / study / "config.yaml"
        if required_commit_from_config(config) != build["source_commit"]:
            raise ValueError(f"{study} is pinned to a different FEAST commit")

    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for rule in ("**/data/local/", "**/outputs/", "**/runs/", "**/logs/"):
        if rule not in ignore:
            raise ValueError(f"missing required artifact ignore rule: {rule}")

    environment_snapshot = (
        ROOT / "environments" / "feast_py311_68816e5" / "pip-list.txt"
    )
    snapshot = environment_snapshot.read_text(encoding="utf-8")
    for package in ("FEAST-py==1.0.2", "numpy==1.26.4", "POT==0.9.7"):
        if package not in snapshot:
            raise ValueError(f"FEAST environment snapshot lacks {package}")

    print(
        "publication manifest: OK "
        f"(status={manifest['status']}, authorized={manifest['release_authorized']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
