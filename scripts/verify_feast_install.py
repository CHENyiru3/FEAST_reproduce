#!/usr/bin/env python3
"""Verify that this interpreter imports the exact recorded clean FEAST wheel."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BUILD_RECORD = ROOT / "FEAST_BUILD.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fields() -> dict[str, str]:
    result: dict[str, str] = {}
    for line in BUILD_RECORD.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            result[key] = value
    return result


def main() -> int:
    import FEAST

    record = fields()
    required_commit = record["Required source commit"]
    required_version = record["Package version"].split()[0]
    wheel_hash = record["Wheel SHA-256"]
    wheels = sorted((ROOT / "dist").glob("feast_py-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one recorded FEAST wheel, found {len(wheels)}")
    wheel = wheels[0]
    if sha256(wheel) != wheel_hash:
        raise RuntimeError("clean FEAST wheel checksum differs from FEAST_BUILD.txt")

    imported = Path(FEAST.__file__).resolve()
    source_checkout = (ROOT.parent / "FEAST" / "src").resolve()
    if imported.is_relative_to(source_checkout):
        raise RuntimeError(f"FEAST was imported from the mutable source checkout: {imported}")
    if str(FEAST.__version__) != required_version:
        raise RuntimeError(
            f"installed FEAST version {FEAST.__version__} != {required_version}"
        )
    distribution = importlib.metadata.distribution("FEAST-py")
    if distribution.version != required_version:
        raise RuntimeError("installed distribution metadata has the wrong version")

    site_packages = imported.parent.parent
    mismatches: list[str] = []
    checked = 0
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if not member.startswith("FEAST/") or member.endswith("/"):
                continue
            installed = site_packages / member
            checked += 1
            if not installed.is_file() or hashlib.sha256(
                installed.read_bytes()
            ).digest() != hashlib.sha256(archive.read(member)).digest():
                mismatches.append(member)
    if mismatches:
        raise RuntimeError(f"installed wheel files differ: {mismatches[:5]}")

    feast_repo = (ROOT.parent / "FEAST").resolve()
    head = subprocess.run(
        ["git", "-C", str(feast_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != required_commit:
        raise RuntimeError(f"FEAST HEAD {head} != recorded build commit {required_commit}")

    print(
        json.dumps(
            {
                "status": "OK",
                "version": required_version,
                "commit": required_commit,
                "wheel_sha256": wheel_hash,
                "import_path": str(imported),
                "verified_package_files": checked,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
