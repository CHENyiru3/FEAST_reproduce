#!/usr/bin/env python3
"""Verify that this interpreter imports the exact recorded clean FEAST wheel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import tempfile
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


def verify_checkout(feast_repo: Path, required_commit: str) -> dict[str, object]:
    """Allow unrelated checkout changes while retaining the recorded build ID."""
    package_paths = (
        "src/FEAST", "pyproject.toml", "setup.py", "setup.cfg",
        "MANIFEST.in", "environment.yml", "requirements.txt",
    )

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(feast_repo), *args],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    # Resolving the recorded commit also rejects missing/invalid build history.
    build_commit = git("rev-parse", "--verify", f"{required_commit}^{{commit}}")
    head = git("rev-parse", "HEAD")
    changed = git("diff", "--name-only", build_commit, "--", *package_paths)
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *package_paths)
    if changed or untracked:
        raise RuntimeError(
            "FEAST checkout package/build files differ from recorded build "
            f"{build_commit}: {(changed + chr(10) + untracked).strip().splitlines()}"
        )
    return {"checkout_head": head, "checkout_package_matches_build": True}


def verify_local_build_source(feast_repo: Path, required_commit: str,
                              wheel: Path, build_dir: Path) -> dict[str, object]:
    """Verify the saved local build source independently of later checkout edits.

    The local build records its tracked changes in source.patch and its new
    module in new_local.py. Reconstruct those inputs from the recorded base;
    never infer the executed source from the current checkout HEAD.
    """
    patch = build_dir / 'source.patch'
    added_module = build_dir / 'new_local.py'
    if not patch.is_file() or not added_module.is_file():
        raise RuntimeError('local build is missing its recorded source inputs')
    base = subprocess.check_output(
        ['git', '-C', str(feast_repo), 'rev-parse', '--verify', f'{required_commit}^{{commit}}'],
        text=True).strip()
    paths = subprocess.check_output(
        ['git', '-C', str(feast_repo), 'ls-tree', '-r', '--name-only', base,
         '--', 'src/FEAST', 'pyproject.toml'], text=True).splitlines()
    with tempfile.TemporaryDirectory(prefix='feast-recorded-source-') as directory:
        snapshot = Path(directory)
        for name in paths:
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(subprocess.check_output(
                ['git', '-C', str(feast_repo), 'show', f'{base}:{name}']))
        subprocess.run(['git', 'apply', str(patch.resolve())], cwd=snapshot,
                       check=True, capture_output=True, text=True)
        module = snapshot / 'src/FEAST/de_novo/local.py'
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_bytes(added_module.read_bytes())
        with zipfile.ZipFile(wheel) as archive:
            wheel_files = {name for name in archive.namelist()
                           if name.startswith('FEAST/') and not name.endswith('/')}
            source_root = snapshot / 'src'
            source_files = {path.relative_to(source_root).as_posix()
                            for path in (source_root / 'FEAST').rglob('*') if path.is_file()}
            if source_files != wheel_files:
                raise RuntimeError('recorded source file set differs from recorded wheel')
            for name in wheel_files:
                if (source_root / name).read_bytes() != archive.read(name):
                    raise RuntimeError(f'recorded source differs from recorded wheel: {name}')
            current_root = feast_repo / 'src'
            current_files = {path.relative_to(current_root).as_posix()
                             for path in (current_root / 'FEAST').rglob('*')
                             if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
            checkout_matches = current_files == wheel_files and all(
                (current_root / name).read_bytes() == archive.read(name) for name in wheel_files)
    head = subprocess.check_output(
        ['git', '-C', str(feast_repo), 'rev-parse', 'HEAD'], text=True).strip()
    return {'checkout_head': head, 'checkout_package_matches_build': checkout_matches,
            'recorded_source_matches_build': True, 'source_mode': 'local_working_tree'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-provenance",
        type=Path,
        help="verify a provisional wheel record instead of FEAST_BUILD.txt",
    )
    args = parser.parse_args()
    import FEAST

    candidate_record: dict | None = None
    if args.candidate_provenance is None:
        record = fields()
        required_commit = record["Required source commit"]
        required_version = record["Package version"].split()[0]
        wheel_hash = record["Wheel SHA-256"]
        wheels = sorted((ROOT / "dist").glob("feast_py-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(
                f"expected one recorded FEAST wheel, found {len(wheels)}"
            )
        wheel = wheels[0]
    else:
        provenance = args.candidate_provenance.resolve()
        candidate_record = json.loads(provenance.read_text(encoding="utf-8"))
        required_commit = str(candidate_record["base_commit"])
        required_version = str(candidate_record["candidate_version"])
        wheel_hash = str(candidate_record["wheel"]["sha256"])
        wheel = provenance.parent / str(candidate_record["wheel"]["path"])
    if sha256(wheel) != wheel_hash:
        raise RuntimeError("FEAST wheel checksum differs from its build record")

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
    if candidate_record is not None and candidate_record.get('source_mode') == 'local_working_tree':
        checkout = verify_local_build_source(feast_repo, required_commit, wheel, provenance.parent)
    else:
        checkout = verify_checkout(feast_repo, required_commit)

    print(
        json.dumps(
            {
                "status": "OK",
                "version": required_version,
                "commit": required_commit,
                **checkout,
                "wheel_sha256": wheel_hash,
                "import_path": str(imported),
                "verified_package_files": checked,
                "candidate_status": (
                    None
                    if candidate_record is None
                    else str(candidate_record["candidate_status"])
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
