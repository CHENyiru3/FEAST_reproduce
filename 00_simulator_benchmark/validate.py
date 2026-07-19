#!/usr/bin/env python3
"""Validate Study 00 inputs, fresh simulations, and atomic metrics."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import anndata as ad
import pandas as pd


ROOT = Path(__file__).parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(data_root: Path) -> None:
    manifest = pd.read_csv(ROOT / "data/input_checksums.csv")
    if len(manifest) != 60 or manifest[["role", "simulator", "sample"]].duplicated().any():
        raise RuntimeError("input manifest must contain 60 unique files")
    for row in manifest.itertuples(index=False):
        path = data_root / row.relative_path
        if not path.is_file() or sha256(path) != row.sha256:
            raise RuntimeError(f"missing or checksum-mismatched input: {path}")


def validate_simulations(root: Path) -> None:
    manifest = pd.read_csv(root / "simulation_manifest.csv")
    if (
        len(manifest) != 12
        or manifest["sample"].nunique() != 12
        or not manifest["status"].eq("ok").all()
    ):
        raise RuntimeError("expected 12 unique successful FEAST simulations")
    for row in manifest.itertuples(index=False):
        path = root / row.file_path
        if sha256(path) != row.output_sha256:
            raise RuntimeError(f"output checksum mismatch: {path}")
        data = ad.read_h5ad(path)
        provenance = data.uns.get("publication_provenance", {})
        if provenance.get("spatial_mode") != "reference_rank":
            raise RuntimeError(f"wrong spatial mode: {path}")
        if provenance.get("assignment_solver") != "scipy" or bool(provenance.get("assignment_blocks")):
            raise RuntimeError(f"wrong assignment contract: {path}")
        if int(provenance.get("public_seed", -1)) != 2026:
            raise RuntimeError(f"wrong public seed: {path}")


def validate_metrics(root: Path) -> None:
    metrics = pd.read_csv(root / "simulator_quality_metrics.csv")
    if len(metrics) != 60 or metrics[["simulator", "sample"]].duplicated().any():
        raise RuntimeError("expected 60 unique simulator/sample metric rows")
    forbidden = [column for column in metrics if "composite" in column.casefold()]
    if forbidden:
        raise RuntimeError(f"composite metrics are prohibited: {forbidden}")
    panel = pd.read_csv(root / "moran_gene_panel.csv")
    required = {"simulator", "sample", "rank", "gene_id"}
    if not required.issubset(panel.columns):
        raise RuntimeError("Moran panel is missing required columns")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inputs", "simulations", "metrics", "all"), default="all")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--simulation-dir", type=Path)
    parser.add_argument("--metrics-dir", type=Path)
    args = parser.parse_args()
    if args.stage in {"inputs", "all"}:
        if args.data_root is None:
            parser.error("--data-root is required")
        validate_inputs(args.data_root)
    if args.stage in {"simulations", "all"}:
        if args.simulation_dir is None:
            parser.error("--simulation-dir is required")
        validate_simulations(args.simulation_dir)
    if args.stage in {"metrics", "all"}:
        if args.metrics_dir is None:
            parser.error("--metrics-dir is required")
        validate_metrics(args.metrics_dir)
    print(f"Study 00 {args.stage}: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
