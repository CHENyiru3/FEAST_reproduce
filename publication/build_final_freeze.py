"""Create an opt-in, metadata-only publication inventory.

The inventory is intentionally a path-and-size record. Git, version pins, and
the ordinary semantic validators remain the provenance mechanisms; this tool
does not create content digests or freeze scientific artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import stat
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def parse_bindings(values: list[str], option: str) -> dict[str, Path]:
    bindings: dict[str, Path] = {}
    for value in values:
        if '=' not in value:
            raise ValueError(f'{option} must use NAME=PATH: {value!r}')
        name, raw_path = value.split('=', 1)
        if not name or name in bindings:
            raise ValueError(f'invalid or duplicate {option} name: {name!r}')
        path = Path(raw_path).expanduser().resolve()
        if path.is_symlink() or not path.exists():
            raise ValueError(f'{option} root must be an existing non-symlink: {path}')
        bindings[name] = path
    return bindings


def iter_files(root: Path):
    if root.is_file():
        yield root
        return
    for directory, _dirs, names in os.walk(root, followlinks=False):
        for name in sorted(names):
            path = Path(directory) / name
            mode = path.stat(follow_symlinks=False).st_mode
            if not stat.S_ISREG(mode):
                raise ValueError(f'non-regular file in selected root: {path}')
            yield path.resolve()


def classify(path: Path) -> str:
    if path.suffix.casefold() in {'.h5ad', '.h5', '.npy', '.npz', '.pt', '.pkl'}:
        return 'candidate_payload'
    return 'evidence'


def add_rows(rows: list[dict[str, object]], roots: dict[str, Path], *, kind: str) -> list[dict[str, str]]:
    declarations: list[dict[str, str]] = []
    for alias, root in roots.items():
        declarations.append({'source_root': alias, 'kind': kind, 'locator': str(root)})
        for path in iter_files(root):
            rows.append({'relative_path': str(path), 'classification': classify(path), 'size': path.stat().st_size, 'source_root': alias})
    return declarations


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', action='append', default=[], metavar='LABEL=PATH', help='optional repository roots recorded as context')
    parser.add_argument('--candidate-root', action='append', default=[], metavar='ALIAS=PATH')
    parser.add_argument('--record', action='append', default=[], metavar='ROLE=PATH')
    parser.add_argument('--output-root', required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f'refusing to overwrite publication inventory: {output_root}')
    repositories = parse_bindings(args.repo, '--repo')
    candidates = parse_bindings(args.candidate_root, '--candidate-root')
    records = parse_bindings(args.record, '--record')
    if not candidates and not records:
        raise ValueError('select at least one candidate root or record')
    rows: list[dict[str, object]] = []
    declarations = add_rows(rows, candidates, kind='candidate')
    declarations.extend(add_rows(rows, records, kind='record'))
    rows.sort(key=lambda row: str(row['relative_path']))
    if len({row['relative_path'] for row in rows}) != len(rows):
        raise ValueError('a selected file was declared more than once')
    output_root.mkdir(parents=True)
    manifest_path = output_root / 'manifest.csv'
    with manifest_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=('relative_path', 'classification', 'size', 'source_root'))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(str(row['classification']) for row in rows)
    bytes_by_classification: dict[str, int] = defaultdict(int)
    for row in rows:
        bytes_by_classification[str(row['classification'])] += int(row['size'])
    summary = {'schema_version': 2, 'file_count': len(rows), 'total_bytes': sum((int(row['size']) for row in rows)), 'counts_by_classification': dict(sorted(counts.items())), 'bytes_by_classification': dict(sorted(bytes_by_classification.items())), 'release_authorized': False}
    (output_root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    provenance = {'schema_version': 2, 'created_utc': datetime.now(timezone.utc).isoformat(), 'repository_roots': {alias: str(path) for alias, path in repositories.items()}, 'input_roots': declarations, 'builder': {'path': str(Path(__file__).resolve())}, 'outputs': [manifest_path.name, 'summary.json', 'provenance.json'], 'release_authorized': False}
    (output_root / 'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'publication inventory: {len(rows)} files, {summary["total_bytes"]} bytes')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
