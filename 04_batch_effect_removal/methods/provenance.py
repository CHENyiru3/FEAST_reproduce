"""Small provenance helpers shared by targeted publication reruns."""

from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


FEAST_REPO = Path(
    os.environ.get(
        "FEAST_REPO",
        Path(__file__).resolve().parents[3] / "FEAST",
    )
).resolve()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def feast_identity() -> dict:
    commit = subprocess.run(
        ["git", "-C", str(FEAST_REPO), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    version = "1.0.2-provisional"
    for version_file in (
        FEAST_REPO / "src" / "FEAST" / "_version.py",
        FEAST_REPO / "src" / "FEAST" / "__init__.py",
    ):
        if not version_file.exists():
            continue
        for line in version_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("__version__") and "=" in line:
                version = line.split("=", 1)[1].strip().strip("'\"")
                break
    return {"version": version, "release_status": "provisional", "commit": commit}


def build_provenance(
    *,
    configuration_id: str,
    seed: int,
    inputs: list[Path],
    outputs: list[Path],
    solver_diagnostics: dict,
    candidate_context: dict | None = None,
    sources: list[Path] | None = None,
) -> dict:
    input_rows = []
    for path in inputs:
        digest = sha256_file(path)
        input_rows.append(
            {
                "artifact_id": f"sha256:{digest}",
                "path": str(path.resolve()),
                "sha256": digest,
            }
        )
    output_rows = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in outputs
    ]
    return {
        "schema_version": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_id": configuration_id,
        "candidate_context": candidate_context or {},
        "feast": feast_identity(),
        "public_seed": int(seed),
        "solver_diagnostics": solver_diagnostics,
        "inputs": input_rows,
        "outputs": output_rows,
        "sources": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in (sources or [])
        ],
    }
