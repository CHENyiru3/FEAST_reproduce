#!/usr/bin/env python3
"""Build a fail-closed, metadata-only inventory of publication candidates.

This script hashes the selected artifacts in place.  It never copies, edits, or
deletes scientific outputs, and it refuses to write into an existing freeze
directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import stat
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


FEAST_VERSION = "1.0.2"
FEAST_COMMIT = "68816e5c1862a6fa2a49bc30609d617c7fa4b449"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_CANDIDATE_ROOTS = {
    "study00_run": ("00", "00_simulator_benchmark/outputs/final_rerun_20260718"),
    "study00_metrics": (
        "00",
        "00_simulator_benchmark/outputs/final_rerun_20260718_metrics_v2",
    ),
    "study00_comparison": (
        "00",
        "00_simulator_benchmark/outputs/final_rerun_20260718_comparison",
    ),
    "study01_run": ("01", "01_clustering/outputs/final_rerun_20260718"),
    "study01_report": (
        "01",
        "01_clustering/outputs/final_rerun_20260718_report",
    ),
    "study01_comparison": (
        "01",
        "01_clustering/outputs/final_rerun_20260718_comparison_v2",
    ),
    "study02": ("02", "02_alignment/outputs/final_rerun_20260718_v2"),
    "study03": ("03", "03_deconvolution/outputs/final_rerun_20260718_v2"),
    "study04": (
        "04",
        "04_batch_effect_removal/outputs/final_rerun_20260718_v2",
    ),
}

REQUIRED_RECORD_ROLES = {
    "publication_manifest",
    "run_status",
    "rng_gate",
    "method_contract_v2",
    "decision_final",
    "decision_validation_final",
    "next_stage_status",
    "scientific_disposition",
    "package_readme",
    "docs",
    "checklist",
    "core_v4_declaration",
    "core_v4_results",
    "core_v4_manifest",
    "core_v4_summary",
    "build_evidence",
    "test_evidence",
}

EXPECTED_RECORD_LOCATORS = {
    "publication_manifest": "REPRO:PUBLICATION_MANIFEST.json",
    "run_status": "REPRO:RUN_STATUS.md",
    "rng_gate": "REPRO:validation/rng_article_gate_20260718_v2.json",
    "method_contract_v2": "REPRO:publication/method_contract_v2",
    "next_stage_status": "EXP:SPEC/before_publication/NEXT_STAGE_STATUS.md",
    "scientific_disposition": "PKG:validation/publication/scientific_disposition.json",
    "package_readme": "PKG:README.md",
    "docs": "PKG:docs/source",
    "checklist": "PKG:PUBLICATION_RELEASE_CHECKLIST.md",
    "core_v4_declaration": "PKG:validation/core/prepublication_run_20260719_v4.json",
    "core_v4_results": "PKG:validation/core/runs/20260719_prepublication_v4/core_validation_results.csv",
    "core_v4_manifest": "PKG:validation/core/runs/20260719_prepublication_v4/run_manifest.json",
    "core_v4_summary": "PKG:validation/core/runs/20260719_prepublication_v4/SUMMARY.md",
}

EXPECTED_CONTRACT_STATUSES = {
    "FEAST-ART-00-SIM-v2": "validated_candidate",
    "FEAST-ART-01-CLUSTER-v2": "validated_candidate",
    "FEAST-ART-02-ALIGN-v2": "validated_candidate",
    "FEAST-ART-03-DECONV-v2": "decision_required",
    "FEAST-ART-04-2D-XSLICE-v2": "validation_required",
    "FEAST-ART-05-3D-IMPUTE-v2": "validation_required",
    "FEAST-ART-06-DEVCCF-E15-v2": "validation_required",
    "FEAST-ART-08-BATCH-v2": "validated_candidate",
}

EXPECTED_PUBLIC_SEEDS = {"00": "2026", "01": "2026", "02": "2026", "03": "2026", "04": "42"}

PAYLOAD_SUFFIXES = {
    ".h5",
    ".h5ad",
    ".mtx",
    ".npy",
    ".npz",
    ".pkl",
    ".pt",
    ".pth",
    ".rds",
}
PAYLOAD_DIRS = {"fixed_panels", "methods", "panel", "rotations", "simulations", "truth"}
EVIDENCE_DIRS = {"comparison", "comparison_v2", "metrics", "scores", "validation"}
EVIDENCE_NAME_TOKENS = {
    "audit",
    "comparison",
    "decision",
    "diagnostic",
    "manifest",
    "metadata",
    "metrics",
    "old_vs_new",
    "provenance",
    "score",
    "summary",
    "support",
    "validation",
}
FILELIKE_SUFFIXES = {
    ".csv",
    ".h5ad",
    ".json",
    ".log",
    ".md",
    ".npy",
    ".npz",
    ".pt",
    ".txt",
    ".yaml",
    ".yml",
}


class FreezeError(RuntimeError):
    """A fail-closed freeze precondition failed."""


def sha256_file(path: Path) -> str:
    before = path.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat(follow_symlinks=False)
    snapshot_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    snapshot_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if snapshot_before != snapshot_after:
        raise FreezeError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def parse_bindings(values: list[str], option: str) -> dict[str, Path]:
    bindings: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise FreezeError(f"{option} must use NAME=PATH: {value!r}")
        name, raw_path = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise FreezeError(f"invalid {option} name: {name!r}")
        if name in bindings:
            raise FreezeError(f"duplicate {option} name: {name}")
        path = Path(raw_path).expanduser()
        if path.is_symlink():
            raise FreezeError(f"symlink is not allowed as {option} root: {path}")
        bindings[name] = path.resolve()
    return bindings


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def logical_path(path: Path, repositories: dict[str, Path]) -> str:
    matches = [
        (len(root.parts), label, path.relative_to(root))
        for label, root in repositories.items()
        if is_relative_to(path, root)
    ]
    if not matches:
        raise FreezeError(f"path is outside declared repositories: {path}")
    _, label, relative = max(matches)
    return f"{label}:{relative.as_posix()}"


def resolve_locator(locator: str, repositories: dict[str, Path]) -> Path:
    if ":" not in locator:
        raise FreezeError(f"malformed repository locator: {locator!r}")
    label, relative = locator.split(":", 1)
    if label not in repositories:
        raise FreezeError(f"unknown repository label in locator: {locator!r}")
    path = (repositories[label] / relative).resolve()
    if not is_relative_to(path, repositories[label]):
        raise FreezeError(f"repository locator escapes its root: {locator!r}")
    return path


def iter_regular_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        raise FreezeError(f"missing input root: {root}")
    if root.is_symlink():
        raise FreezeError(f"symlink input is prohibited: {root}")
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        raise FreezeError(f"input is neither a regular file nor directory: {root}")
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in list(dirnames):
            child = current / name
            if child.is_symlink():
                raise FreezeError(f"symlink inside selected root: {child}")
        for name in filenames:
            child = current / name
            mode = child.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                raise FreezeError(f"symlink inside selected root: {child}")
            if not stat.S_ISREG(mode):
                raise FreezeError(f"non-regular file inside selected root: {child}")
            yield child.resolve()


def classify_candidate(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    lowered_parts = [part.casefold() for part in relative.parts]
    lowered = relative.as_posix().casefold()
    if "comparison" in lowered_parts and (root / "comparison_v2").is_dir():
        return "noncanonical_evidence"
    if any("failure" in part or part == "failed" for part in lowered_parts):
        return "failure_evidence"
    if any("archive" in part or "superseded" in part for part in lowered_parts):
        return "noncanonical_evidence"
    if any(token in lowered for token in ("label-informed", "label_informed", "invalid_composite", "historical_composite")):
        return "noncanonical_evidence"
    if path.suffix.casefold() in PAYLOAD_SUFFIXES:
        return "candidate_payload"
    name = path.name.casefold()
    if any(part in EVIDENCE_DIRS for part in lowered_parts):
        return "candidate_evidence"
    if any(token in name for token in EVIDENCE_NAME_TOKENS):
        return "candidate_evidence"
    if path.suffix.casefold() == ".csv" and any(part in PAYLOAD_DIRS for part in lowered_parts):
        return "candidate_payload"
    return "candidate_evidence"


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FreezeError(f"malformed JSON evidence {path}: {exc}") from exc


def read_manifest_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or any(not name for name in reader.fieldnames):
                raise FreezeError(f"manifest has an empty or malformed header: {path}")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise FreezeError(f"malformed CSV manifest {path}: {exc}") from exc
    if not rows:
        raise FreezeError(f"manifest contains no rows: {path}")
    return rows


def walk_json(value: Any) -> Iterable[tuple[tuple[str, ...], Any]]:
    stack: list[tuple[tuple[str, ...], Any]] = [((), value)]
    while stack:
        path, current = stack.pop()
        yield path, current
        if isinstance(current, dict):
            for key, child in current.items():
                stack.append((path + (str(key),), child))
        elif isinstance(current, list):
            for index, child in enumerate(current):
                stack.append((path + (str(index),), child))


def enforce_no_promotion(data: Any, source: Path) -> None:
    for key_path, value in walk_json(data):
        if not key_path:
            continue
        key = key_path[-1].casefold()
        parent = key_path[-2].casefold() if len(key_path) > 1 else ""
        if value is True and (
            key == "release_authorized"
            or key.endswith("_authorized")
            or (key == "authorized" and parent == "release")
            or (
                parent == "claim_authorizations"
                and key in {"publication_claim", "composite_score", "method_ranking", "winner"}
            )
        ):
            raise FreezeError(f"authorization/promotion is prohibited in {source}: {'.'.join(key_path)}")
        if parent == "release" and key == "tag" and value is not None:
            raise FreezeError(f"release tag is prohibited in {source}: {value!r}")
        if parent == "release" and key == "pypi_publication" and value not in (False, None):
            raise FreezeError(f"PyPI publication is prohibited in {source}")


def validate_final_decision_gate(
    decision_paths: list[Path],
    disposition_sha256: str,
    repositories: dict[str, Path],
) -> dict[str, Any]:
    loaded = [(path, load_json(path)) for path in decision_paths]
    summaries = [
        (path, data)
        for path, data in loaded
        if isinstance(data, dict)
        and path.name == "summary.json"
        and "all_gates_passed" in data
        and "freeze_candidate_approved" in data
    ]
    if len(summaries) != 1:
        raise FreezeError(
            "final decision evidence must contain exactly one machine-readable validation summary"
        )
    summary_path, summary = summaries[0]
    expected = {
        "all_gates_passed": True,
        "freeze_candidate_approved": True,
        "scientific_status": "blocked",
        "release_authorized": False,
        "claim_promotion_authorized": False,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise FreezeError(
                f"final decision gate {key} must equal {value!r}: {summary_path}"
            )
    blocker_ids = summary.get("open_blocker_ids")
    if (
        not isinstance(blocker_ids, list)
        or not blocker_ids
        or any(not isinstance(item, str) or not item.strip() for item in blocker_ids)
        or len(set(blocker_ids)) != len(blocker_ids)
    ):
        raise FreezeError("final decision gate requires nonempty, unique open_blocker_ids")

    predecessor_is_v4 = False
    predecessor_hash_bound = False
    disposition_hash_bound = False
    for decision_path, data in loaded:
        enforce_no_promotion(data, decision_path)
        for key_path, value in walk_json(data):
            context = ".".join(key_path).casefold()
            if "predecessor" in context:
                rendered = json.dumps(value, sort_keys=True).casefold()
                if "v4" in context or "v4" in rendered:
                    predecessor_is_v4 = True
                if isinstance(value, str) and SHA256_RE.fullmatch(
                    value.removeprefix("sha256:").casefold()
                ):
                    predecessor_hash_bound = True
            if (
                isinstance(value, str)
                and value.casefold() == disposition_sha256
                and "disposition" in context
            ):
                disposition_hash_bound = True
        for _, value in walk_json(data):
            if not isinstance(value, dict):
                continue
            sibling_text = " ".join(
                str(item).casefold() for item in value.values() if isinstance(item, str)
            )
            if "scientific_disposition" in sibling_text and disposition_sha256 in {
                str(item).casefold() for item in value.values()
            }:
                disposition_hash_bound = True
    if not predecessor_is_v4 or not predecessor_hash_bound:
        raise FreezeError("final decision evidence does not hash-bind an identified v4 predecessor")
    if not disposition_hash_bound:
        raise FreezeError("final decision evidence does not bind the current scientific disposition hash")
    return {
        "validation_summary": logical_path(summary_path, repositories),
        "validation_summary_sha256": sha256_file(summary_path),
        "scientific_status": "blocked",
        "open_blocker_ids": blocker_ids,
        "predecessor_v4_bound": True,
        "scientific_disposition_sha256": disposition_sha256,
        "freeze_candidate_approved": True,
        "release_authorized": False,
        "claim_promotion_authorized": False,
    }


def validate_publication_manifest(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise FreezeError("PUBLICATION_MANIFEST must be a JSON object")
    if data.get("release_authorized") is not False or data.get("final_tag") is not None:
        raise FreezeError("PUBLICATION_MANIFEST authorizes a release or tag")
    build = data.get("article_feast_build")
    if not isinstance(build, dict):
        raise FreezeError("PUBLICATION_MANIFEST is missing article_feast_build")
    if build.get("version") != FEAST_VERSION or build.get("source_commit") != FEAST_COMMIT:
        raise FreezeError("PUBLICATION_MANIFEST FEAST build does not match the frozen version/commit")
    studies = data.get("studies")
    expected = {
        "00_simulator_benchmark",
        "01_clustering",
        "02_alignment",
        "03_deconvolution",
        "04_batch_effect_removal",
    }
    if not isinstance(studies, dict) or set(studies) != expected:
        raise FreezeError("PUBLICATION_MANIFEST must contain exactly clean studies 00 through 04")
    enforce_no_promotion(data, path)
    return data


def validate_contract_root(root: Path) -> None:
    config_dir = root / "configs"
    if not config_dir.is_dir():
        raise FreezeError(f"method-contract root lacks configs/: {root}")
    config_paths = sorted(config_dir.glob("*-v2.json"))
    configs: dict[str, dict[str, Any]] = {}
    for path in config_paths:
        data = load_json(path)
        if not isinstance(data, dict) or not isinstance(data.get("workflow_id"), str):
            raise FreezeError(f"malformed method contract: {path}")
        workflow_id = data["workflow_id"]
        if workflow_id in configs:
            raise FreezeError(f"duplicate method-contract workflow_id: {workflow_id}")
        configs[workflow_id] = data
        enforce_no_promotion(data, path)
    if set(configs) != set(EXPECTED_CONTRACT_STATUSES):
        raise FreezeError("method-contract v2 must contain exactly the eight declared workflows")
    for workflow_id, expected_status in EXPECTED_CONTRACT_STATUSES.items():
        if configs[workflow_id].get("status") != expected_status:
            raise FreezeError(
                f"method-contract status mismatch for {workflow_id}: "
                f"{configs[workflow_id].get('status')!r}"
            )


def candidate_for_reference(
    raw_path: str,
    evidence_path: Path,
    source_root: Path,
    inventory_paths: set[Path],
    repositories: dict[str, Path],
    inventory_hashes: dict[Path, str] | None = None,
    expected_hash: str | None = None,
) -> Path | None:
    possibilities: list[Path] = []
    if re.match(r"^[A-Za-z0-9_.-]+:", raw_path):
        try:
            possibilities.append(resolve_locator(raw_path, repositories))
        except FreezeError:
            pass
    else:
        reference = Path(raw_path).expanduser()
        if reference.is_absolute():
            possibilities.append(reference.resolve())
        else:
            possibilities.append((evidence_path.parent / reference).resolve())
            if source_root.is_dir():
                possibilities.append((source_root / reference).resolve())
            for root in repositories.values():
                if reference.parts and reference.parts[0] == root.name:
                    possibilities.append((root.joinpath(*reference.parts[1:])).resolve())
                if is_relative_to(evidence_path, root):
                    relative_evidence = evidence_path.relative_to(root)
                    if relative_evidence.parts:
                        possibilities.append(
                            (root / relative_evidence.parts[0] / reference).resolve()
                        )
                possibilities.append((root / reference).resolve())
    for possibility in possibilities:
        if possibility in inventory_paths:
            return possibility
    normalized = raw_path.replace("\\", "/").lstrip("./")
    suffix_matches = [
        path for path in inventory_paths if path.as_posix().endswith("/" + normalized)
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        hash_matches = [
            path
            for path in suffix_matches
            if inventory_hashes is not None
            and expected_hash is not None
            and inventory_hashes.get(path) == expected_hash
        ]
        if len(hash_matches) == 1:
            return hash_matches[0]
        raise FreezeError(f"ambiguous embedded artifact path in {evidence_path}: {raw_path}")
    if inventory_hashes is not None and expected_hash is not None:
        hash_identity_matches = sorted(
            path
            for path, digest in inventory_hashes.items()
            if digest == expected_hash
        )
        if hash_identity_matches:
            return hash_identity_matches[0]
    existing = list(
        dict.fromkeys(
            possibility
            for possibility in possibilities
            if possibility.is_file() and not possibility.is_symlink()
        )
    )
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise FreezeError(
            f"ambiguous external embedded artifact path in {evidence_path}: {raw_path}"
        )
    for possibility in possibilities:
        if source_root.is_dir() and is_relative_to(possibility, source_root):
            raise FreezeError(f"embedded artifact is missing from selected root: {raw_path} ({evidence_path})")
    return None


def check_reference(
    raw_path: Any,
    expected_hash: Any,
    evidence_path: Path,
    source_root: Path,
    inventory: dict[Path, dict[str, Any]],
    repositories: dict[str, Path],
) -> int:
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        return 0
    expected_hash = expected_hash.casefold()
    if not SHA256_RE.fullmatch(expected_hash):
        raise FreezeError(f"invalid embedded SHA-256 in {evidence_path}: {expected_hash!r}")
    target = candidate_for_reference(
        raw_path,
        evidence_path,
        source_root,
        set(inventory),
        repositories,
        {path: row["sha256"] for path, row in inventory.items()},
        expected_hash,
    )
    if target is None:
        return 0
    inventory_row = inventory.get(target)
    actual = inventory_row["sha256"] if inventory_row is not None else sha256_file(target)
    if actual != expected_hash:
        raise FreezeError(
            f"embedded hash disagreement: {evidence_path} -> {raw_path}; "
            f"declared {expected_hash}, actual {actual}"
        )
    return 1


def validate_json_references(
    data: Any,
    evidence_path: Path,
    source_root: Path,
    inventory: dict[Path, dict[str, Any]],
    repositories: dict[str, Path],
) -> int:
    checked = 0
    for key_path, value in walk_json(data):
        if not isinstance(value, dict):
            continue
        if any("predecessor" in part.casefold() for part in key_path):
            continue
        pairs: set[tuple[str, str]] = set()
        for key, hash_value in value.items():
            if not key.endswith("_sha256") or not isinstance(hash_value, str):
                continue
            base = key[: -len("_sha256")]
            for path_key in (base, f"{base}_path"):
                if isinstance(value.get(path_key), str):
                    pairs.add((path_key, key))
        if isinstance(value.get("path"), str):
            for hash_key in ("sha256", "file_sha256", "output_sha256", "provenance_sha256"):
                if isinstance(value.get(hash_key), str):
                    pairs.add(("path", hash_key))
                    break
        for path_key, hash_key in pairs:
            checked += check_reference(
                value[path_key],
                value[hash_key],
                evidence_path,
                source_root,
                inventory,
                repositories,
            )
        for possible_path, possible_hash in value.items():
            if (
                isinstance(possible_path, str)
                and isinstance(possible_hash, str)
                and SHA256_RE.fullmatch(possible_hash.casefold())
                and Path(possible_path).suffix.casefold() in FILELIKE_SUFFIXES
            ):
                checked += check_reference(
                    possible_path,
                    possible_hash,
                    evidence_path,
                    source_root,
                    inventory,
                    repositories,
                )
    return checked


def validate_csv_references(
    rows: list[dict[str, str]],
    evidence_path: Path,
    source_root: Path,
    inventory: dict[Path, dict[str, Any]],
    repositories: dict[str, Path],
) -> int:
    checked = 0
    known_pairs = (
        ("path", "sha256"),
        ("file_path", "output_sha256"),
        ("file", "output_sha256"),
        ("output_path", "output_sha256"),
        ("simulation_path", "simulation_sha256"),
        ("truth_path", "truth_sha256"),
        ("metadata_path", "metadata_sha256"),
        ("runner_log_path", "runner_log_sha256"),
        ("log", "log_sha256"),
        ("moving", "moving_sha256"),
        ("source_path", "source_sha256"),
    )
    for row in rows:
        for path_key, hash_key in known_pairs:
            if row.get(path_key) and row.get(hash_key):
                checked += check_reference(
                    row[path_key], row[hash_key], evidence_path, source_root, inventory, repositories
                )
        if row.get("artifacts"):
            try:
                artifacts = json.loads(row["artifacts"])
            except json.JSONDecodeError as exc:
                raise FreezeError(f"malformed artifacts JSON in {evidence_path}: {exc}") from exc
            if not isinstance(artifacts, list):
                raise FreezeError(f"artifacts column is not a list in {evidence_path}")
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    raise FreezeError(f"malformed artifact record in {evidence_path}")
                checked += check_reference(
                    artifact.get("path"),
                    artifact.get("sha256"),
                    evidence_path,
                    source_root,
                    inventory,
                    repositories,
                )
        if row.get("slice_id") and row.get("simulation_id"):
            implied = {
                "clusters_sha256": "clusters.csv",
                "metadata_sha256": "metadata.json",
                "result_h5ad_sha256": "result.h5ad",
                "runner_log_sha256": "runner.log",
            }
            for hash_key, filename in implied.items():
                if row.get(hash_key):
                    implied_path = evidence_path.parent / row["slice_id"] / row["simulation_id"] / filename
                    checked += check_reference(
                        str(implied_path),
                        row[hash_key],
                        evidence_path,
                        source_root,
                        inventory,
                        repositories,
                    )
    return checked


def extract_bindings(
    candidate_evidence: list[tuple[str, str, Path]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    configuration_ids: dict[str, set[str]] = defaultdict(set)
    seeds: dict[str, set[str]] = defaultdict(set)
    for study, classification, path in candidate_evidence:
        if classification != "candidate_evidence":
            continue
        name = path.name.casefold()
        if "manifest" not in name and name != "provenance.json":
            continue
        if path.suffix.casefold() == ".json":
            data = load_json(path)
            for key_path, value in walk_json(data):
                if not key_path:
                    continue
                key = key_path[-1]
                if key == "configuration_id" and isinstance(value, str):
                    configuration_ids[study].add(value)
                if key == "public_seed" and isinstance(value, (int, float, str)):
                    seeds[study].add(str(value))
                if key == "feast_version" and value != FEAST_VERSION:
                    raise FreezeError(f"FEAST version mismatch in {path}: {value!r}")
                if key == "feast_commit" and value != FEAST_COMMIT:
                    raise FreezeError(f"FEAST commit mismatch in {path}: {value!r}")
        elif path.suffix.casefold() == ".csv":
            for row in read_manifest_csv(path):
                if row.get("configuration_id"):
                    configuration_ids[study].add(row["configuration_id"])
                if row.get("public_seed"):
                    seeds[study].add(row["public_seed"])
                if row.get("feast_version") and row["feast_version"] != FEAST_VERSION:
                    raise FreezeError(f"FEAST version mismatch in {path}: {row['feast_version']!r}")
                if row.get("feast_commit") and row["feast_commit"] != FEAST_COMMIT:
                    raise FreezeError(f"FEAST commit mismatch in {path}: {row['feast_commit']!r}")
    for study, expected_seed in EXPECTED_PUBLIC_SEEDS.items():
        if not configuration_ids[study]:
            raise FreezeError(f"study {study} has no manifest-bound configuration ID")
        if seeds[study] != {expected_seed}:
            raise FreezeError(
                f"study {study} public seeds do not equal {{{expected_seed}}}: {sorted(seeds[study])}"
            )
    return (
        {study: sorted(values) for study, values in sorted(configuration_ids.items())},
        {study: sorted(values) for study, values in sorted(seeds.items())},
    )


def build_inventory(
    repositories: dict[str, Path],
    candidate_roots: dict[str, Path],
    records: dict[str, Path],
    output_root: Path,
) -> tuple[dict[Path, dict[str, Any]], list[dict[str, str]]]:
    inventory: dict[Path, dict[str, Any]] = {}
    declarations: list[dict[str, str]] = []
    inputs: list[tuple[str, str, str, Path]] = []
    for alias, root in candidate_roots.items():
        study, _ = EXPECTED_CANDIDATE_ROOTS[alias]
        inputs.append((alias, study, "candidate", root))
    for role, root in records.items():
        inputs.append((role, "publication", "record", root))

    for source_root, study, kind, root in inputs:
        if is_relative_to(output_root, root) or is_relative_to(root, output_root):
            raise FreezeError(f"freeze output overlaps selected input root: {source_root}={root}")
        locator = logical_path(root, repositories)
        declarations.append(
            {
                "source_root": source_root,
                "study": study,
                "kind": kind,
                "locator": locator,
            }
        )
        files = list(iter_regular_files(root))
        if not files:
            raise FreezeError(f"selected root contains no regular files: {source_root}={root}")
        for path in files:
            classification = (
                classify_candidate(path, root) if kind == "candidate" else "publication_evidence"
            )
            if classification not in {
                "candidate_payload",
                "candidate_evidence",
                "failure_evidence",
                "noncanonical_evidence",
                "publication_evidence",
            }:
                raise FreezeError(f"unclassified path: {path}")
            locator_path = logical_path(path, repositories)
            digest = sha256_file(path)
            if not SHA256_RE.fullmatch(digest):
                raise FreezeError(f"unhashed file: {path}")
            row = {
                "study": study,
                "relative_path": locator_path,
                "classification": classification,
                "size": path.stat(follow_symlinks=False).st_size,
                "sha256": digest,
                "source_root": source_root,
            }
            previous = inventory.get(path)
            if previous is not None:
                if previous["sha256"] != digest or previous["classification"] != classification:
                    raise FreezeError(f"conflicting duplicate logical path: {locator_path}")
                continue
            inventory[path] = row
    return inventory, declarations


def write_outputs(
    output_root: Path,
    inventory: dict[Path, dict[str, Any]],
    declarations: list[dict[str, str]],
    repositories: dict[str, Path],
    configuration_ids: dict[str, list[str]],
    public_seeds: dict[str, list[str]],
    embedded_references_checked: int,
    final_decision_gate: dict[str, Any],
) -> None:
    if output_root.exists():
        raise FreezeError(f"refusing to overwrite existing freeze root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    rows = sorted(inventory.values(), key=lambda row: row["relative_path"])
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("study", "relative_path", "classification", "size", "sha256", "source_root"),
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["classification"] for row in rows)
    byte_counts: Counter[str] = Counter()
    study_counts: Counter[str] = Counter()
    study_bytes: Counter[str] = Counter()
    for row in rows:
        byte_counts[row["classification"]] += int(row["size"])
        study_counts[row["study"]] += 1
        study_bytes[row["study"]] += int(row["size"])
    summary = {
        "schema_version": 1,
        "release_authorized": False,
        "file_count": len(rows),
        "total_bytes": sum(int(row["size"]) for row in rows),
        "counts_by_classification": dict(sorted(counts.items())),
        "bytes_by_classification": dict(sorted(byte_counts.items())),
        "counts_by_study": dict(sorted(study_counts.items())),
        "bytes_by_study": dict(sorted(study_bytes.items())),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    source = Path(__file__).resolve()
    provenance = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "release_authorized": False,
        "builder": {
            "path": logical_path(source, repositories),
            "sha256": sha256_file(source),
        },
        "feast": {"version": FEAST_VERSION, "commit": FEAST_COMMIT},
        "repository_labels": sorted(repositories),
        "input_roots": declarations,
        "configuration_ids_by_study": configuration_ids,
        "public_seeds_by_study": public_seeds,
        "embedded_references_checked": embedded_references_checked,
        "final_decision_gate": final_decision_gate,
        "outputs": {
            "manifest.csv": sha256_file(manifest_path),
            "summary.json": sha256_file(output_root / "summary.json"),
        },
    }
    with (output_root / "provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="declared repository root; REPRO and PKG are required",
    )
    parser.add_argument(
        "--candidate-root",
        action="append",
        default=[],
        metavar="ALIAS=PATH",
        help="one of the nine exact clean candidate roots",
    )
    parser.add_argument(
        "--record",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="small publication/package evidence file or directory",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        repositories = parse_bindings(args.repo, "--repo")
        if not {"REPRO", "PKG", "EXP"}.issubset(repositories):
            raise FreezeError("--repo requires at least REPRO=..., PKG=..., and EXP=...")
        for label, path in repositories.items():
            if not path.is_dir():
                raise FreezeError(f"repository root is missing or not a directory: {label}={path}")

        candidate_roots = parse_bindings(args.candidate_root, "--candidate-root")
        if set(candidate_roots) != set(EXPECTED_CANDIDATE_ROOTS):
            missing = sorted(set(EXPECTED_CANDIDATE_ROOTS) - set(candidate_roots))
            extra = sorted(set(candidate_roots) - set(EXPECTED_CANDIDATE_ROOTS))
            raise FreezeError(f"candidate-root aliases mismatch; missing={missing}, extra={extra}")
        for alias, (_, expected_relative) in EXPECTED_CANDIDATE_ROOTS.items():
            expected_path = (repositories["REPRO"] / expected_relative).resolve()
            if candidate_roots[alias] != expected_path:
                raise FreezeError(
                    f"{alias} must resolve to REPRO:{expected_relative}, got {candidate_roots[alias]}"
                )

        records = parse_bindings(args.record, "--record")
        if set(records) != REQUIRED_RECORD_ROLES:
            missing = sorted(REQUIRED_RECORD_ROLES - set(records))
            extra = sorted(set(records) - REQUIRED_RECORD_ROLES)
            raise FreezeError(f"record roles mismatch; missing={missing}, extra={extra}")
        for role, expected_locator in EXPECTED_RECORD_LOCATORS.items():
            expected_path = resolve_locator(expected_locator, repositories)
            if records[role] != expected_path:
                raise FreezeError(
                    f"{role} must resolve to {expected_locator}, got {records[role]}"
                )
        if not records["decision_final"].is_dir():
            raise FreezeError("decision_final must be an explicit finalized-decision directory")
        decision_jsons = [
            path
            for role in ("decision_final", "decision_validation_final")
            for path in iter_regular_files(records[role])
            if path.suffix.casefold() == ".json"
        ]
        if not decision_jsons:
            raise FreezeError("final decision/validation evidence contains no JSON decision record")
        publication_manifest = records["publication_manifest"]
        validate_publication_manifest(publication_manifest)
        validate_contract_root(records["method_contract_v2"])

        output_root = args.output_root.expanduser().resolve()
        if output_root.exists():
            raise FreezeError(f"refusing to overwrite existing freeze root: {output_root}")
        inventory, declarations = build_inventory(
            repositories, candidate_roots, records, output_root
        )

        disposition_path = records["scientific_disposition"]
        final_decision_gate = validate_final_decision_gate(
            decision_jsons,
            inventory[disposition_path]["sha256"],
            repositories,
        )

        candidate_evidence = [
            (row["study"], row["classification"], path)
            for path, row in inventory.items()
            if row["study"] != "publication"
        ]
        configuration_ids, public_seeds = extract_bindings(candidate_evidence)

        embedded_references_checked = 0
        root_by_alias = {**candidate_roots, **records}
        for path, row in inventory.items():
            if row["classification"] not in {"candidate_evidence", "publication_evidence"}:
                continue
            if row["source_root"] in {"build_evidence", "test_evidence"}:
                continue
            source_root = root_by_alias[row["source_root"]]
            suffix = path.suffix.casefold()
            if suffix == ".json":
                data = load_json(path)
                enforce_no_promotion(data, path)
                embedded_references_checked += validate_json_references(
                    data, path, source_root, inventory, repositories
                )
            elif suffix == ".csv" and "manifest" in path.name.casefold():
                rows = read_manifest_csv(path)
                embedded_references_checked += validate_csv_references(
                    rows, path, source_root, inventory, repositories
                )
        if embedded_references_checked == 0:
            raise FreezeError("no embedded manifest/provenance artifact hashes were checked")

        write_outputs(
            output_root,
            inventory,
            declarations,
            repositories,
            configuration_ids,
            public_seeds,
            embedded_references_checked,
            final_decision_gate,
        )
        print(f"final freeze inventory written: {output_root}")
        print(f"files: {len(inventory)}")
        print(f"bytes: {sum(int(row['size']) for row in inventory.values())}")
        print("release_authorized: false")
        return 0
    except FreezeError as exc:
        print(f"final freeze build: FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
