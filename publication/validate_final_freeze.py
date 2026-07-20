#!/usr/bin/env python3
"""Independently validate a final publication freeze inventory."""

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
from pathlib import Path
from typing import Any, Iterable


FEAST_VERSION = "1.0.2"
FEAST_COMMIT = "68816e5c1862a6fa2a49bc30609d617c7fa4b449"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_FIELDS = (
    "study",
    "relative_path",
    "classification",
    "size",
    "sha256",
    "source_root",
)
CLASSIFICATIONS = {
    "candidate_payload",
    "candidate_evidence",
    "failure_evidence",
    "noncanonical_evidence",
    "publication_evidence",
}

EXPECTED_CANDIDATE_ROOTS = {
    "study00_run": ("00", "REPRO:00_simulator_benchmark/outputs/final_rerun_20260718"),
    "study00_metrics": (
        "00",
        "REPRO:00_simulator_benchmark/outputs/final_rerun_20260718_metrics_v2",
    ),
    "study00_comparison": (
        "00",
        "REPRO:00_simulator_benchmark/outputs/final_rerun_20260718_comparison",
    ),
    "study01_run": ("01", "REPRO:01_clustering/outputs/final_rerun_20260718"),
    "study01_report": (
        "01",
        "REPRO:01_clustering/outputs/final_rerun_20260718_report",
    ),
    "study01_comparison": (
        "01",
        "REPRO:01_clustering/outputs/final_rerun_20260718_comparison_v2",
    ),
    "study02": ("02", "REPRO:02_alignment/outputs/final_rerun_20260718_v2"),
    "study03": ("03", "REPRO:03_deconvolution/outputs/final_rerun_20260718_v2"),
    "study04": (
        "04",
        "REPRO:04_batch_effect_removal/outputs/final_rerun_20260718_v2",
    ),
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
REQUIRED_RECORD_ROLES = set(EXPECTED_RECORD_LOCATORS) | {
    "decision_final",
    "decision_validation_final",
    "build_evidence",
    "test_evidence",
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


class ValidationError(RuntimeError):
    """The freeze is incomplete, inconsistent, or tampered."""


def sha256_file(path: Path) -> str:
    before = path.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat(follow_symlinks=False)
    first = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    second = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if first != second:
        raise ValidationError(f"file changed while validating: {path}")
    return digest.hexdigest()


def parse_repositories(values: list[str]) -> dict[str, Path]:
    repositories: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValidationError(f"--repo must use LABEL=PATH: {value!r}")
        label, raw_path = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label) or label in repositories:
            raise ValidationError(f"invalid or duplicate repository label: {label!r}")
        original = Path(raw_path).expanduser()
        if original.is_symlink():
            raise ValidationError(f"repository root may not be a symlink: {original}")
        path = original.resolve()
        if not path.is_dir():
            raise ValidationError(f"repository root is missing: {label}={path}")
        repositories[label] = path
    if not {"REPRO", "PKG", "EXP"}.issubset(repositories):
        raise ValidationError("--repo requires at least REPRO, PKG, and EXP")
    return repositories


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def resolve_locator(locator: str, repositories: dict[str, Path]) -> Path:
    if not isinstance(locator, str) or ":" not in locator:
        raise ValidationError(f"malformed repository locator: {locator!r}")
    label, relative = locator.split(":", 1)
    if label not in repositories or Path(relative).is_absolute():
        raise ValidationError(f"unknown or absolute repository locator: {locator!r}")
    path = (repositories[label] / relative).resolve()
    if not is_relative_to(path, repositories[label]):
        raise ValidationError(f"repository locator escapes its root: {locator!r}")
    return path


def logical_path(path: Path, repositories: dict[str, Path]) -> str:
    matches = [
        (len(root.parts), label, path.relative_to(root))
        for label, root in repositories.items()
        if is_relative_to(path, root)
    ]
    if not matches:
        raise ValidationError(f"path is outside declared repositories: {path}")
    _, label, relative = max(matches)
    return f"{label}:{relative.as_posix()}"


def enumerate_files(root: Path) -> Iterable[Path]:
    if not root.exists() or root.is_symlink():
        raise ValidationError(f"missing or symlinked declared root: {root}")
    if root.is_file():
        yield root.resolve()
        return
    if not root.is_dir():
        raise ValidationError(f"declared root is not a regular file/directory: {root}")
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in dirnames:
            if (current / name).is_symlink():
                raise ValidationError(f"symlink inside declared root: {current / name}")
        for name in filenames:
            path = current / name
            mode = path.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                raise ValidationError(f"symlink inside declared root: {path}")
            if not stat.S_ISREG(mode):
                raise ValidationError(f"non-regular file inside declared root: {path}")
            yield path.resolve()


def classify_candidate(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    parts = [part.casefold() for part in relative.parts]
    lowered = relative.as_posix().casefold()
    if "comparison" in parts and (root / "comparison_v2").is_dir():
        return "noncanonical_evidence"
    if any("failure" in part or part == "failed" for part in parts):
        return "failure_evidence"
    if any("archive" in part or "superseded" in part for part in parts):
        return "noncanonical_evidence"
    if any(token in lowered for token in ("label-informed", "label_informed", "invalid_composite", "historical_composite")):
        return "noncanonical_evidence"
    if path.suffix.casefold() in PAYLOAD_SUFFIXES:
        return "candidate_payload"
    name = path.name.casefold()
    if any(part in EVIDENCE_DIRS for part in parts):
        return "candidate_evidence"
    if any(token in name for token in EVIDENCE_NAME_TOKENS):
        return "candidate_evidence"
    if path.suffix.casefold() == ".csv" and any(part in PAYLOAD_DIRS for part in parts):
        return "candidate_payload"
    return "candidate_evidence"


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"malformed JSON evidence {path}: {exc}") from exc


def read_csv_manifest(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or any(not field for field in reader.fieldnames):
                raise ValidationError(f"manifest has a malformed header: {path}")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValidationError(f"malformed CSV manifest {path}: {exc}") from exc
    if not rows:
        raise ValidationError(f"manifest contains no rows: {path}")
    return rows


def walk_json(value: Any) -> Iterable[tuple[tuple[str, ...], Any]]:
    pending: list[tuple[tuple[str, ...], Any]] = [((), value)]
    while pending:
        path, current = pending.pop()
        yield path, current
        if isinstance(current, dict):
            for key, child in current.items():
                pending.append((path + (str(key),), child))
        elif isinstance(current, list):
            for index, child in enumerate(current):
                pending.append((path + (str(index),), child))


def check_no_authorization(data: Any, source: Path) -> None:
    for key_path, value in walk_json(data):
        if not key_path:
            continue
        key = key_path[-1].casefold()
        parent = key_path[-2].casefold() if len(key_path) > 1 else ""
        forbidden_true = (
            key == "release_authorized"
            or key.endswith("_authorized")
            or (key == "authorized" and parent == "release")
            or (
                parent == "claim_authorizations"
                and key in {"publication_claim", "composite_score", "method_ranking", "winner"}
            )
        )
        if value is True and forbidden_true:
            raise ValidationError(f"prohibited promotion in {source}: {'.'.join(key_path)}")
        if parent == "release" and key == "tag" and value is not None:
            raise ValidationError(f"release tag is present in {source}")
        if parent == "release" and key == "pypi_publication" and value not in (False, None):
            raise ValidationError(f"PyPI publication is authorized in {source}")


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
        raise ValidationError("expected exactly one final-decision validation summary")
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
            raise ValidationError(f"final-decision gate mismatch for {key}: {summary_path}")
    blockers = summary.get("open_blocker_ids")
    if (
        not isinstance(blockers, list)
        or not blockers
        or any(not isinstance(item, str) or not item.strip() for item in blockers)
        or len(set(blockers)) != len(blockers)
    ):
        raise ValidationError("final-decision gate has invalid open_blocker_ids")

    predecessor_v4 = False
    predecessor_hash = False
    disposition_hash = False
    for decision_path, data in loaded:
        check_no_authorization(data, decision_path)
        for key_path, value in walk_json(data):
            context = ".".join(key_path).casefold()
            if "predecessor" in context:
                rendered = json.dumps(value, sort_keys=True).casefold()
                predecessor_v4 |= "v4" in context or "v4" in rendered
                if isinstance(value, str) and SHA256_RE.fullmatch(
                    value.removeprefix("sha256:").casefold()
                ):
                    predecessor_hash = True
            if (
                isinstance(value, str)
                and value.casefold() == disposition_sha256
                and "disposition" in context
            ):
                disposition_hash = True
        for _, value in walk_json(data):
            if not isinstance(value, dict):
                continue
            strings = [str(item).casefold() for item in value.values() if isinstance(item, str)]
            if "scientific_disposition" in " ".join(strings) and disposition_sha256 in strings:
                disposition_hash = True
    if not predecessor_v4 or not predecessor_hash:
        raise ValidationError("final decision does not hash-bind an identified v4 predecessor")
    if not disposition_hash:
        raise ValidationError("final decision does not bind the current disposition hash")
    return {
        "validation_summary": logical_path(summary_path, repositories),
        "validation_summary_sha256": sha256_file(summary_path),
        "scientific_status": "blocked",
        "open_blocker_ids": blockers,
        "predecessor_v4_bound": True,
        "scientific_disposition_sha256": disposition_sha256,
        "freeze_candidate_approved": True,
        "release_authorized": False,
        "claim_promotion_authorized": False,
    }


def validate_publication_manifest(path: Path) -> None:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValidationError("PUBLICATION_MANIFEST is not an object")
    if data.get("release_authorized") is not False or data.get("final_tag") is not None:
        raise ValidationError("PUBLICATION_MANIFEST authorizes release/tag")
    build = data.get("article_feast_build")
    if not isinstance(build, dict):
        raise ValidationError("PUBLICATION_MANIFEST lacks article_feast_build")
    if build.get("version") != FEAST_VERSION or build.get("source_commit") != FEAST_COMMIT:
        raise ValidationError("PUBLICATION_MANIFEST FEAST version/commit mismatch")
    expected_studies = {
        "00_simulator_benchmark",
        "01_clustering",
        "02_alignment",
        "03_deconvolution",
        "04_batch_effect_removal",
    }
    if not isinstance(data.get("studies"), dict) or set(data["studies"]) != expected_studies:
        raise ValidationError("PUBLICATION_MANIFEST clean-study set mismatch")
    check_no_authorization(data, path)


def validate_contracts(root: Path) -> None:
    config_dir = root / "configs"
    if not config_dir.is_dir():
        raise ValidationError("method-contract v2 root lacks configs/")
    contracts: dict[str, dict[str, Any]] = {}
    for path in sorted(config_dir.glob("*-v2.json")):
        data = load_json(path)
        if not isinstance(data, dict) or not isinstance(data.get("workflow_id"), str):
            raise ValidationError(f"malformed method contract: {path}")
        if data["workflow_id"] in contracts:
            raise ValidationError(f"duplicate workflow_id: {data['workflow_id']}")
        contracts[data["workflow_id"]] = data
        check_no_authorization(data, path)
    if set(contracts) != set(EXPECTED_CONTRACT_STATUSES):
        raise ValidationError("method-contract v2 workflow set mismatch")
    for workflow_id, status in EXPECTED_CONTRACT_STATUSES.items():
        if contracts[workflow_id].get("status") != status:
            raise ValidationError(f"method-contract status mismatch: {workflow_id}")


def parse_freeze_manifest(path: Path) -> dict[str, dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
                raise ValidationError(f"freeze manifest fields must be exactly {MANIFEST_FIELDS}")
            raw_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValidationError(f"cannot parse freeze manifest: {exc}") from exc
    if not raw_rows:
        raise ValidationError("freeze manifest contains no rows")
    rows: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        locator = raw["relative_path"]
        if locator in rows:
            raise ValidationError(f"duplicate logical path in freeze manifest: {locator}")
        if raw["classification"] not in CLASSIFICATIONS:
            raise ValidationError(f"unclassified freeze row: {locator}")
        if not SHA256_RE.fullmatch(raw["sha256"]):
            raise ValidationError(f"missing/invalid SHA-256: {locator}")
        try:
            size = int(raw["size"])
        except ValueError as exc:
            raise ValidationError(f"invalid size for {locator}") from exc
        if size < 0:
            raise ValidationError(f"negative size for {locator}")
        row: dict[str, Any] = dict(raw)
        row["size"] = size
        rows[locator] = row
    return rows


def validate_summary(summary: Any, rows: dict[str, dict[str, Any]]) -> None:
    if not isinstance(summary, dict) or summary.get("release_authorized") is not False:
        raise ValidationError("freeze summary is malformed or release-authorized")
    counts = Counter(row["classification"] for row in rows.values())
    byte_counts: Counter[str] = Counter()
    study_counts: Counter[str] = Counter()
    study_bytes: Counter[str] = Counter()
    for row in rows.values():
        byte_counts[row["classification"]] += row["size"]
        study_counts[row["study"]] += 1
        study_bytes[row["study"]] += row["size"]
    expected = {
        "file_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows.values()),
        "counts_by_classification": dict(sorted(counts.items())),
        "bytes_by_classification": dict(sorted(byte_counts.items())),
        "counts_by_study": dict(sorted(study_counts.items())),
        "bytes_by_study": dict(sorted(study_bytes.items())),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValidationError(f"freeze summary mismatch for {key}")


def validate_declarations(
    provenance: dict[str, Any], repositories: dict[str, Path]
) -> list[dict[str, str]]:
    declarations = provenance.get("input_roots")
    if not isinstance(declarations, list) or not declarations:
        raise ValidationError("provenance lacks input_roots")
    seen: set[str] = set()
    normalized: list[dict[str, str]] = []
    for declaration in declarations:
        if not isinstance(declaration, dict):
            raise ValidationError("malformed input-root declaration")
        required = {"source_root", "study", "kind", "locator"}
        if set(declaration) != required or not all(
            isinstance(declaration[key], str) for key in required
        ):
            raise ValidationError(f"malformed input-root declaration: {declaration!r}")
        alias = declaration["source_root"]
        if alias in seen:
            raise ValidationError(f"duplicate input-root alias: {alias}")
        seen.add(alias)
        resolve_locator(declaration["locator"], repositories)
        normalized.append(dict(declaration))

    candidate = {item["source_root"]: item for item in normalized if item["kind"] == "candidate"}
    records = {item["source_root"]: item for item in normalized if item["kind"] == "record"}
    if set(candidate) != set(EXPECTED_CANDIDATE_ROOTS):
        raise ValidationError("candidate-root declaration set mismatch")
    if set(records) != REQUIRED_RECORD_ROLES:
        raise ValidationError("publication-record declaration set mismatch")
    for alias, (study, locator) in EXPECTED_CANDIDATE_ROOTS.items():
        if candidate[alias]["study"] != study or candidate[alias]["locator"] != locator:
            raise ValidationError(f"candidate-root declaration mismatch: {alias}")
    for role, locator in EXPECTED_RECORD_LOCATORS.items():
        if records[role]["study"] != "publication" or records[role]["locator"] != locator:
            raise ValidationError(f"publication-record declaration mismatch: {role}")
    for role in REQUIRED_RECORD_ROLES - set(EXPECTED_RECORD_LOCATORS):
        if records[role]["study"] != "publication":
            raise ValidationError(f"publication record has wrong study: {role}")
    decision_root = resolve_locator(records["decision_final"]["locator"], repositories)
    if not decision_root.is_dir():
        raise ValidationError("decision_final is not a directory")
    decision_jsons = [
        path
        for role in ("decision_final", "decision_validation_final")
        for path in enumerate_files(resolve_locator(records[role]["locator"], repositories))
        if path.suffix.casefold() == ".json"
    ]
    if not decision_jsons:
        raise ValidationError("final decision/validation evidence contains no JSON record")
    return normalized


def rebuild_inventory(
    declarations: list[dict[str, str]], repositories: dict[str, Path]
) -> dict[str, dict[str, Any]]:
    rebuilt: dict[str, dict[str, Any]] = {}
    for declaration in declarations:
        root = resolve_locator(declaration["locator"], repositories)
        files = list(enumerate_files(root))
        if not files:
            raise ValidationError(f"declared root has no regular files: {declaration['source_root']}")
        for path in files:
            classification = (
                classify_candidate(path, root)
                if declaration["kind"] == "candidate"
                else "publication_evidence"
            )
            locator = logical_path(path, repositories)
            row = {
                "study": declaration["study"],
                "relative_path": locator,
                "classification": classification,
                "size": path.stat(follow_symlinks=False).st_size,
                "sha256": sha256_file(path),
                "source_root": declaration["source_root"],
                "path": path,
            }
            previous = rebuilt.get(locator)
            if previous:
                if previous["sha256"] != row["sha256"] or previous["classification"] != classification:
                    raise ValidationError(f"conflicting duplicate logical path: {locator}")
                continue
            rebuilt[locator] = row
    return rebuilt


def resolve_embedded(
    raw_path: str,
    evidence_path: Path,
    source_root: Path,
    by_path: dict[Path, dict[str, Any]],
    repositories: dict[str, Path],
    expected_hash: str | None = None,
) -> Path | None:
    possibilities: list[Path] = []
    if re.match(r"^[A-Za-z0-9_.-]+:", raw_path):
        try:
            possibilities.append(resolve_locator(raw_path, repositories))
        except ValidationError:
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
    for path in possibilities:
        if path in by_path:
            return path
    normalized = raw_path.replace("\\", "/").lstrip("./")
    suffix_matches = [path for path in by_path if path.as_posix().endswith("/" + normalized)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        hash_matches = [
            path
            for path in suffix_matches
            if expected_hash is not None
            and by_path.get(path, {}).get("sha256") == expected_hash
        ]
        if len(hash_matches) == 1:
            return hash_matches[0]
        raise ValidationError(f"ambiguous embedded path in {evidence_path}: {raw_path}")
    if expected_hash is not None:
        hash_identity_matches = sorted(
            path for path, row in by_path.items() if row.get("sha256") == expected_hash
        )
        if hash_identity_matches:
            return hash_identity_matches[0]
    existing = list(
        dict.fromkeys(
            path for path in possibilities if path.is_file() and not path.is_symlink()
        )
    )
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise ValidationError(
            f"ambiguous external embedded path in {evidence_path}: {raw_path}"
        )
    for path in possibilities:
        if source_root.is_dir() and is_relative_to(path, source_root):
            raise ValidationError(f"embedded selected-root artifact is missing: {raw_path}")
    return None


def compare_embedded(
    raw_path: Any,
    expected: Any,
    evidence_path: Path,
    source_root: Path,
    by_path: dict[Path, dict[str, Any]],
    repositories: dict[str, Path],
) -> int:
    if not isinstance(raw_path, str) or not isinstance(expected, str):
        return 0
    expected = expected.casefold()
    if not SHA256_RE.fullmatch(expected):
        raise ValidationError(f"invalid embedded SHA-256 in {evidence_path}: {expected!r}")
    target = resolve_embedded(
        raw_path, evidence_path, source_root, by_path, repositories, expected
    )
    if target is None:
        return 0
    manifest_row = by_path.get(target)
    actual = manifest_row["sha256"] if manifest_row is not None else sha256_file(target)
    if actual != expected:
        raise ValidationError(f"embedded hash disagreement: {evidence_path} -> {raw_path}")
    return 1


def check_json_bindings(
    data: Any,
    evidence_path: Path,
    source_root: Path,
    by_path: dict[Path, dict[str, Any]],
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
            if key.endswith("_sha256") and isinstance(hash_value, str):
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
            checked += compare_embedded(
                value[path_key], value[hash_key], evidence_path, source_root, by_path, repositories
            )
        for possible_path, possible_hash in value.items():
            if (
                isinstance(possible_path, str)
                and isinstance(possible_hash, str)
                and SHA256_RE.fullmatch(possible_hash.casefold())
                and Path(possible_path).suffix.casefold() in FILELIKE_SUFFIXES
            ):
                checked += compare_embedded(
                    possible_path,
                    possible_hash,
                    evidence_path,
                    source_root,
                    by_path,
                    repositories,
                )
    return checked


def check_csv_bindings(
    rows: list[dict[str, str]],
    evidence_path: Path,
    source_root: Path,
    by_path: dict[Path, dict[str, Any]],
    repositories: dict[str, Path],
) -> int:
    checked = 0
    pairs = (
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
        for path_key, hash_key in pairs:
            if row.get(path_key) and row.get(hash_key):
                checked += compare_embedded(
                    row[path_key], row[hash_key], evidence_path, source_root, by_path, repositories
                )
        if row.get("artifacts"):
            try:
                artifacts = json.loads(row["artifacts"])
            except json.JSONDecodeError as exc:
                raise ValidationError(f"malformed artifacts JSON in {evidence_path}") from exc
            if not isinstance(artifacts, list):
                raise ValidationError(f"artifacts field is not a list in {evidence_path}")
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    raise ValidationError(f"malformed artifact in {evidence_path}")
                checked += compare_embedded(
                    artifact.get("path"),
                    artifact.get("sha256"),
                    evidence_path,
                    source_root,
                    by_path,
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
                    target = evidence_path.parent / row["slice_id"] / row["simulation_id"] / filename
                    checked += compare_embedded(
                        str(target), row[hash_key], evidence_path, source_root, by_path, repositories
                    )
    return checked


def extract_manifest_identity(
    rebuilt: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    ids: dict[str, set[str]] = defaultdict(set)
    seeds: dict[str, set[str]] = defaultdict(set)
    for row in rebuilt.values():
        if row["classification"] != "candidate_evidence":
            continue
        path: Path = row["path"]
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
                    ids[row["study"]].add(value)
                if key == "public_seed" and isinstance(value, (int, float, str)):
                    seeds[row["study"]].add(str(value))
                if key == "feast_version" and value != FEAST_VERSION:
                    raise ValidationError(f"FEAST version mismatch in {path}")
                if key == "feast_commit" and value != FEAST_COMMIT:
                    raise ValidationError(f"FEAST commit mismatch in {path}")
        elif path.suffix.casefold() == ".csv":
            for item in read_csv_manifest(path):
                if item.get("configuration_id"):
                    ids[row["study"]].add(item["configuration_id"])
                if item.get("public_seed"):
                    seeds[row["study"]].add(item["public_seed"])
                if item.get("feast_version") and item["feast_version"] != FEAST_VERSION:
                    raise ValidationError(f"FEAST version mismatch in {path}")
                if item.get("feast_commit") and item["feast_commit"] != FEAST_COMMIT:
                    raise ValidationError(f"FEAST commit mismatch in {path}")
    for study, seed in EXPECTED_PUBLIC_SEEDS.items():
        if not ids[study] or seeds[study] != {seed}:
            raise ValidationError(
                f"study {study} manifest identity is incomplete: ids={sorted(ids[study])}, seeds={sorted(seeds[study])}"
            )
    return (
        {study: sorted(values) for study, values in sorted(ids.items())},
        {study: sorted(values) for study, values in sorted(seeds.items())},
    )


def validate_freeze_output_files(freeze_root: Path) -> None:
    if freeze_root.is_symlink() or not freeze_root.is_dir():
        raise ValidationError(f"freeze root is missing or symlinked: {freeze_root}")
    files = list(enumerate_files(freeze_root))
    names = {path.relative_to(freeze_root).as_posix() for path in files}
    expected = {"manifest.csv", "summary.json", "provenance.json"}
    if names != expected:
        raise ValidationError(f"freeze root has missing/extra files: expected={expected}, actual={names}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--freeze-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        repositories = parse_repositories(args.repo)
        freeze_root = args.freeze_root.expanduser().resolve()
        validate_freeze_output_files(freeze_root)

        manifest_path = freeze_root / "manifest.csv"
        summary_path = freeze_root / "summary.json"
        provenance_path = freeze_root / "provenance.json"
        rows = parse_freeze_manifest(manifest_path)
        summary = load_json(summary_path)
        provenance = load_json(provenance_path)
        if not isinstance(provenance, dict):
            raise ValidationError("freeze provenance is not a JSON object")
        if provenance.get("release_authorized") is not False:
            raise ValidationError("freeze provenance authorizes release")
        feast = provenance.get("feast")
        if feast != {"version": FEAST_VERSION, "commit": FEAST_COMMIT}:
            raise ValidationError("freeze provenance FEAST version/commit mismatch")
        if provenance.get("repository_labels") != sorted(repositories):
            raise ValidationError("validator repository labels differ from builder declaration")
        output_hashes = provenance.get("outputs")
        if not isinstance(output_hashes, dict):
            raise ValidationError("freeze provenance lacks output hashes")
        if output_hashes.get("manifest.csv") != sha256_file(manifest_path):
            raise ValidationError("freeze manifest hash does not match provenance")
        if output_hashes.get("summary.json") != sha256_file(summary_path):
            raise ValidationError("freeze summary hash does not match provenance")
        builder = provenance.get("builder")
        if not isinstance(builder, dict) or not SHA256_RE.fullmatch(str(builder.get("sha256", ""))):
            raise ValidationError("freeze provenance lacks builder source binding")
        builder_path = resolve_locator(str(builder.get("path")), repositories)
        if sha256_file(builder_path) != builder["sha256"]:
            raise ValidationError("builder source hash changed after freeze")

        validate_summary(summary, rows)
        declarations = validate_declarations(provenance, repositories)
        rebuilt = rebuild_inventory(declarations, repositories)
        if set(rebuilt) != set(rows):
            missing = sorted(set(rows) - set(rebuilt))[:10]
            extra = sorted(set(rebuilt) - set(rows))[:10]
            raise ValidationError(f"source inventory has missing/extra paths; missing={missing}, extra={extra}")
        for locator, declared in rows.items():
            actual = rebuilt[locator]
            for key in ("study", "classification", "size", "sha256", "source_root"):
                if declared[key] != actual[key]:
                    raise ValidationError(f"inventory mismatch for {locator}: {key}")

        declarations_by_alias = {item["source_root"]: item for item in declarations}
        publication_manifest = resolve_locator(
            declarations_by_alias["publication_manifest"]["locator"], repositories
        )
        contract_root = resolve_locator(
            declarations_by_alias["method_contract_v2"]["locator"], repositories
        )
        validate_publication_manifest(publication_manifest)
        validate_contracts(contract_root)

        decision_paths = [
            path
            for role in ("decision_final", "decision_validation_final")
            for path in enumerate_files(
                resolve_locator(declarations_by_alias[role]["locator"], repositories)
            )
            if path.suffix.casefold() == ".json"
        ]
        disposition_path = resolve_locator(
            declarations_by_alias["scientific_disposition"]["locator"], repositories
        )
        disposition_locator = logical_path(disposition_path, repositories)
        decision_gate = validate_final_decision_gate(
            decision_paths,
            rebuilt[disposition_locator]["sha256"],
            repositories,
        )
        if provenance.get("final_decision_gate") != decision_gate:
            raise ValidationError("final-decision gate binding differs from independent validation")

        ids, seeds = extract_manifest_identity(rebuilt)
        if provenance.get("configuration_ids_by_study") != ids:
            raise ValidationError("configuration-ID binding differs from independently extracted values")
        if provenance.get("public_seeds_by_study") != seeds:
            raise ValidationError("public-seed binding differs from independently extracted values")

        by_path = {row["path"]: row for row in rebuilt.values()}
        checked = 0
        for row in rebuilt.values():
            if row["classification"] not in {"candidate_evidence", "publication_evidence"}:
                continue
            if row["source_root"] in {"build_evidence", "test_evidence"}:
                continue
            path: Path = row["path"]
            source_root = resolve_locator(
                declarations_by_alias[row["source_root"]]["locator"], repositories
            )
            if path.suffix.casefold() == ".json":
                data = load_json(path)
                check_no_authorization(data, path)
                checked += check_json_bindings(data, path, source_root, by_path, repositories)
            elif path.suffix.casefold() == ".csv" and "manifest" in path.name.casefold():
                checked += check_csv_bindings(
                    read_csv_manifest(path), path, source_root, by_path, repositories
                )
        if checked <= 0 or provenance.get("embedded_references_checked") != checked:
            raise ValidationError(
                "embedded manifest/provenance reference count is zero or differs from the builder"
            )

        print(f"final freeze validation: OK: {freeze_root}")
        print(f"files: {len(rows)}")
        print(f"bytes: {sum(row['size'] for row in rows.values())}")
        print(f"embedded references checked: {checked}")
        print("release_authorized: false")
        return 0
    except ValidationError as exc:
        print(f"final freeze validation: FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
