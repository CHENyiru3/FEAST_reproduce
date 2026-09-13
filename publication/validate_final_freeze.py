"""Validate a metadata-only publication inventory without content digests."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import build_final_freeze as builder


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'expected a JSON object: {path}')
    return data


def parse_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    required = {'relative_path', 'classification', 'size', 'source_root'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('inventory manifest has an invalid schema')
    for row in rows:
        int(row['size'])
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', action='append', default=[], metavar='LABEL=PATH')
    parser.add_argument('--freeze-root', required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    freeze_root = args.freeze_root.expanduser().resolve()
    expected_files = {'manifest.csv', 'summary.json', 'provenance.json'}
    if not freeze_root.is_dir() or {path.name for path in freeze_root.iterdir()} != expected_files:
        raise ValueError('inventory directory has missing or unexpected files')
    rows = parse_rows(freeze_root / 'manifest.csv')
    summary = read_json(freeze_root / 'summary.json')
    provenance = read_json(freeze_root / 'provenance.json')
    if summary.get('release_authorized') is not False or provenance.get('release_authorized') is not False:
        raise ValueError('inventory must not authorize a release')
    if summary.get('file_count') != len(rows) or summary.get('total_bytes') != sum((int(row['size']) for row in rows)):
        raise ValueError('inventory summary does not match its manifest')
    counts = Counter(row['classification'] for row in rows)
    bytes_by_classification: dict[str, int] = defaultdict(int)
    for row in rows:
        bytes_by_classification[row['classification']] += int(row['size'])
    if summary.get('counts_by_classification') != dict(sorted(counts.items())) or summary.get('bytes_by_classification') != dict(sorted(bytes_by_classification.items())):
        raise ValueError('inventory classification summary does not match its manifest')
    declared = provenance.get('input_roots')
    if not isinstance(declared, list) or not declared:
        raise ValueError('inventory provenance lacks input roots')
    rebuilt: list[dict[str, object]] = []
    for item in declared:
        if not isinstance(item, dict) or set(item) != {'source_root', 'kind', 'locator'}:
            raise ValueError(f'invalid input-root declaration: {item!r}')
        root = Path(item['locator']).expanduser().resolve()
        if not root.exists() or root.is_symlink():
            raise ValueError(f'declared input root is unavailable: {root}')
        for path in builder.iter_files(root):
            rebuilt.append({'relative_path': str(path), 'classification': builder.classify(path), 'size': str(path.stat().st_size), 'source_root': item['source_root']})
    observed = sorted(rows, key=lambda row: row['relative_path'])
    current = sorted(rebuilt, key=lambda row: str(row['relative_path']))
    if observed != current:
        raise ValueError('selected file paths, classifications, or sizes changed; use Git history to inspect the revision')
    print(f'publication inventory validation: OK ({len(rows)} files)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
