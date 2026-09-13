#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

SCRIPT_PATH = Path(__file__).resolve()
DATASET_KEY = "GSE269617"
PROCESSED_ROOT = Path(__file__).resolve().parents[2] / "Datasets" / "Processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover GSE269617 broad-region annotations.")
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--dataset-key", default=DATASET_KEY)
    parser.add_argument("--devccf-resource-root", type=Path, default=PROCESSED_ROOT / "DevCCFv1_figshare_26377171" / "coordinate_system")
    parser.add_argument("--input-subdir", default="h5ad")
    parser.add_argument("--output-subdir", default="h5ad_region_annotated")
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--source-annotated-root", type=Path, default=None)
    parser.add_argument("--source-label-table", type=Path, default=None)
    parser.add_argument("--mapping-root", type=Path, default=None)
    parser.add_argument("--schema-path", type=Path, default=None)
    parser.add_argument("--min-non-other-fraction", type=float, default=0.95)
    parser.add_argument("--allow-other", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-qc-figures", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"annotation_recovery_{DATASET_KEY}_{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    return log_path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_matrix(matrix: Any) -> str:
    h = hashlib.sha256()
    if sparse.issparse(matrix):
        csr = matrix.tocsr()
        for arr in (csr.indptr, csr.indices, csr.data):
            h.update(np.asarray(arr).tobytes())
        h.update(str(csr.shape).encode("utf-8"))
    else:
        arr = np.asarray(matrix)
        h.update(arr.tobytes())
        h.update(str(arr.shape).encode("utf-8"))
    return h.hexdigest()


def sha256_index(index: pd.Index) -> str:
    return hashlib.sha256("\n".join(index.astype(str)).encode("utf-8")).hexdigest()


def script_sha256() -> str:
    return hashlib.sha256(SCRIPT_PATH.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def environment_summary() -> str:
    return json.dumps(
        {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": sys.platform,
            "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        },
        sort_keys=True,
    )


def discover_inputs(input_dir: Path, requested: list[str] | None) -> list[Path]:
    paths = sorted(input_dir.glob("*.h5ad"))
    if requested:
        by_sample = {p.stem: p for p in paths}
        missing = sorted(set(requested).difference(by_sample))
        if missing:
            raise ValueError(f"Requested h5ads missing under {input_dir}: {missing}")
        paths = [by_sample[s] for s in requested]
    return paths


def load_schema(schema_path: Path) -> pd.DataFrame:
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing broad-region schema: {schema_path}")
    schema = pd.read_csv(schema_path, sep="\t")
    required = {"region_id", "region_label"}
    if not required.issubset(schema.columns):
        raise ValueError(f"Schema missing columns {required}: {schema_path}")
    return schema


def read_label_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".gz"} or path.name.endswith(".tsv.gz") else ","
    df = pd.read_csv(path, sep=sep)
    if "cell" not in df.columns:
        for candidate in ("cell_id", "EntityID", "obs_name", "barcode"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "cell"})
                break
    if "cell" not in df.columns:
        raise ValueError(f"Label table must contain cell/cell_id/EntityID/obs_name: {path}")
    return df.set_index("cell")


def candidate_annotation_columns(df: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for target, candidates in {
        "region": ["region", "region_pred", "annotation", "ground_truth", "broad_region_label"],
        "region_id": ["region_id", "broad_region_id"],
        "confidence": ["annotation_recovery_confidence", "confidence", "region_confidence", "pred_label_top1_prob"],
        "devccf_x": ["devccf_x"],
        "devccf_y": ["devccf_y"],
        "devccf_z": ["devccf_z"],
        "devccf_struct_id": ["devccf_struct_id", "struct_id"],
        "devccf_struct_acronym": ["devccf_struct_acronym", "struct_acronym"],
        "devccf_struct_name": ["devccf_struct_name", "struct_name"],
    }.items():
        for col in candidates:
            if col in df.columns:
                mapping[target] = col
                break
    return mapping


def load_old_h5ad_annotations(path: Path) -> pd.DataFrame:
    old = ad.read_h5ad(path, backed="r")
    try:
        obs = old.obs.copy()
        if "devccf_3d" in old.obsm:
            xyz = np.asarray(old.obsm["devccf_3d"])
            obs["devccf_x"] = xyz[:, 0]
            obs["devccf_y"] = xyz[:, 1]
            obs["devccf_z"] = xyz[:, 2]
        return obs
    finally:
        if getattr(old, "file", None) is not None:
            old.file.close()


def parse_sample_stage(sample_id: str) -> tuple[str, str]:
    if "_E14" in sample_id:
        return "E14", "E15.5"
    if "_E18" in sample_id:
        return "E18", "E18.5"
    return "unknown", "unknown"


def annotate_one(
    input_path: Path,
    output_path: Path,
    schema: pd.DataFrame,
    label_df: pd.DataFrame | None,
    source_annotated_root: Path | None,
    source_files: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    sample_id = input_path.stem
    adata = ad.read_h5ad(input_path)
    before_counts_hash = sha256_matrix(adata.layers["counts"] if "counts" in adata.layers else adata.X)
    obs_hash = sha256_index(adata.obs_names)

    source = None
    label_source = None
    if label_df is not None:
        source = label_df.reindex(adata.obs_names)
        label_source = "recovered_label_table"
    elif source_annotated_root is not None:
        candidate = source_annotated_root / input_path.name
        if candidate.exists():
            source = load_old_h5ad_annotations(candidate).reindex(adata.obs_names)
            source_files.append(str(candidate))
            label_source = "recovered_lab6_region_annotated_h5ad"

    raw_stage, target_stage = parse_sample_stage(sample_id)
    if source is None:
        return {
            "dataset_key": args.dataset_key,
            "sample_id": sample_id,
            "input_h5ad": str(input_path),
            "output_h5ad": str(output_path),
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "n_labeled": 0,
            "n_unlabeled": int(adata.n_obs),
            "label_source": "not_available_in_raw_tree",
            "annotation_recovery_method": "no_allowed_source_found",
            "devccf_target_stage": target_stage,
            "schema_path": str(args.schema_path),
            "counts_sha256_before": before_counts_hash,
            "counts_sha256_after": before_counts_hash,
            "obs_names_sha256": obs_hash,
            "source_files": ";".join(source_files),
            "created_at": now_iso(),
            "script": str(SCRIPT_PATH),
            "script_sha256": script_sha256(),
            "environment": environment_summary(),
            "status": "blocked",
            "message": "No recovered per-cell labels, annotated h5ad, or mapping table was supplied.",
        }

    colmap = candidate_annotation_columns(source)
    if "region" not in colmap:
        raise ValueError(f"{sample_id}: source annotations lack a region-like column")
    region = source[colmap["region"]].astype("object")
    schema_labels = set(schema["region_label"].astype(str))
    region = region.where(region.astype(str).isin(schema_labels), "Other")
    if "region_id" in colmap:
        region_id = pd.to_numeric(source[colmap["region_id"]], errors="coerce")
    else:
        label_to_id = schema.set_index("region_label")["region_id"].to_dict()
        region_id = region.map(label_to_id)
    labeled = region.notna() & (region.astype(str) != "nan")
    region = region.fillna("Other")
    region_id = region_id.fillna(9).astype(int)

    adata.obs["region_id"] = region_id.to_numpy()
    adata.obs["region"] = pd.Categorical(region.astype(str))
    adata.obs["annotation"] = pd.Categorical(region.astype(str))
    adata.obs["ground_truth"] = pd.Categorical(region.astype(str))
    adata.obs["region_pred"] = pd.Categorical(region.astype(str))
    if "cell_type" not in adata.obs:
        adata.obs["cell_type"] = pd.Categorical(["unknown"] * adata.n_obs)
    adata.obs["label_source"] = label_source
    adata.obs["annotation_recovery_method"] = label_source
    confidence = pd.to_numeric(source[colmap["confidence"]], errors="coerce") if "confidence" in colmap else pd.Series(np.nan, index=adata.obs_names)
    adata.obs["annotation_recovery_confidence"] = confidence.to_numpy()
    adata.obs["devccf_target_stage"] = target_stage
    for key in ["devccf_struct_id", "devccf_struct_acronym", "devccf_struct_name"]:
        if key in colmap:
            adata.obs[key] = source[colmap[key]].to_numpy()
    if {"devccf_x", "devccf_y", "devccf_z"}.issubset(colmap):
        xyz = source[[colmap["devccf_x"], colmap["devccf_y"], colmap["devccf_z"]]].to_numpy(dtype=float)
        adata.obsm["devccf_3d"] = xyz
        adata.obs["devccf_x"] = xyz[:, 0]
        adata.obs["devccf_y"] = xyz[:, 1]
        adata.obs["devccf_z"] = xyz[:, 2]

    non_other = np.mean(region.astype(str).isin(schema.loc[schema["region_id"].between(1, 8), "region_label"].astype(str)))
    if non_other < args.min_non_other_fraction and not args.allow_other:
        raise ValueError(f"{sample_id}: non-Other fraction {non_other:.3f} below {args.min_non_other_fraction:.3f}")

    adata.uns["annotation_recovery"] = {
        "dataset_key": args.dataset_key,
        "schema_path": str(args.schema_path),
        "source_paths": source_files,
        "method": label_source,
        "coordinate_resource": "DevCCFv1_figshare_26377171",
        "stage_pairing": {"raw_stage": raw_stage, "devccf_target_stage": target_stage},
        "counts_preserved": True,
        "obs_names_preserved": True,
        "created_at": now_iso(),
        "script": str(SCRIPT_PATH),
        "script_sha256": script_sha256(),
    }
    after_counts_hash = sha256_matrix(adata.layers["counts"] if "counts" in adata.layers else adata.X)
    if after_counts_hash != before_counts_hash:
        raise ValueError(f"{sample_id}: counts changed during annotation merge")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path, compression="gzip")
    return {
        "dataset_key": args.dataset_key,
        "sample_id": sample_id,
        "input_h5ad": str(input_path),
        "output_h5ad": str(output_path),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_labeled": int(labeled.sum()),
        "n_unlabeled": int((~labeled).sum()),
        "label_source": label_source,
        "annotation_recovery_method": label_source,
        "devccf_target_stage": target_stage,
        "schema_path": str(args.schema_path),
        "counts_sha256_before": before_counts_hash,
        "counts_sha256_after": after_counts_hash,
        "obs_names_sha256": obs_hash,
        "source_files": ";".join(source_files),
        "created_at": now_iso(),
        "script": str(SCRIPT_PATH),
        "script_sha256": script_sha256(),
        "environment": environment_summary(),
        "status": "written",
        "message": f"non_other_fraction={non_other:.4f}",
    }


def main() -> None:
    args = parse_args()
    root = args.processed_root / args.dataset_key
    setup_logging(root / "logs")
    input_dir = root / args.input_subdir
    output_dir = root / args.output_subdir
    manifest_dir = root / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    if args.schema_path is None:
        args.schema_path = args.devccf_resource_root / "GSE269617_region_merged" / "gse269617_region_schema.tsv"
    inputs = discover_inputs(input_dir, args.samples)
    schema = load_schema(args.schema_path)
    source_files: list[str] = [str(args.schema_path)]
    label_df = None
    if args.source_label_table is not None:
        label_df = read_label_table(args.source_label_table)
        source_files.append(str(args.source_label_table))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "inputs": [str(p) for p in inputs],
                    "schema": str(args.schema_path),
                    "source_label_table": str(args.source_label_table) if args.source_label_table else None,
                    "source_annotated_root": str(args.source_annotated_root) if args.source_annotated_root else None,
                },
                indent=2,
            )
        )
        return
    rows = []
    for path in inputs:
        try:
            rows.append(
                annotate_one(
                    path,
                    output_dir / path.name,
                    schema,
                    label_df,
                    args.source_annotated_root,
                    list(source_files),
                    args,
                )
            )
        except Exception as exc:
            logging.exception("Failed %s", path)
            rows.append(
                {
                    "dataset_key": args.dataset_key,
                    "sample_id": path.stem,
                    "input_h5ad": str(path),
                    "output_h5ad": str(output_dir / path.name),
                    "n_obs": "",
                    "n_vars": "",
                    "n_labeled": 0,
                    "n_unlabeled": "",
                    "label_source": "error",
                    "annotation_recovery_method": "error",
                    "devccf_target_stage": parse_sample_stage(path.stem)[1],
                    "schema_path": str(args.schema_path),
                    "counts_sha256_before": "",
                    "counts_sha256_after": "",
                    "obs_names_sha256": "",
                    "source_files": ";".join(source_files),
                    "created_at": now_iso(),
                    "script": str(SCRIPT_PATH),
                    "script_sha256": script_sha256(),
                    "environment": environment_summary(),
                    "status": "error",
                    "message": str(exc),
                }
            )
    manifest_path = manifest_dir / "annotation_recovery_manifest.tsv"
    pd.DataFrame(rows).to_csv(manifest_path, sep="\t", index=False)
    logging.info("Wrote %s", manifest_path)
    blocked = [row for row in rows if row.get("status") in {"blocked", "error"}]
    if blocked:
        raise RuntimeError(f"{len(blocked)} sample(s) did not receive production annotations; see {manifest_path}")


if __name__ == "__main__":
    main()
