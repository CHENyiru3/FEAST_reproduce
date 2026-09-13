from __future__ import annotations

import argparse
import json
import logging
import os
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from preprocess_common import (
    add_common_arguments,
    add_counts_layer,
    add_standard_obs,
    add_var_metadata,
    append_manifest_rows,
    apply_qc,
    ensure_processed_dirs,
    manifest_row,
    preprocessing_uns,
    print_inventory,
    set_labels,
    setup_logging,
    set_spatial,
    to_csr_counts,
    write_processed,
)


DATASET_KEY = "10x_Xenium_human_lymph_node_preview"
SAMPLE_ID = "Xenium_V1_hLymphNode_nondiseased_section"
SCRIPT_PATH = Path(__file__).resolve()


def zip_path(raw_dataset_dir: Path) -> Path:
    return raw_dataset_dir / "samples" / SAMPLE_ID / f"{SAMPLE_ID}_outs.zip"


def extract_member(zf: zipfile.ZipFile, member: str, scratch_dir: Path) -> Path:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    dest = scratch_dir / member
    dest.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Extracting %s", member)
    with zf.open(member) as inp, dest.open("wb") as out:
        out.write(inp.read())
    return dest


def process(args, paths: dict[str, Path]) -> dict:
    raw_dataset_dir = args.raw_root / args.dataset_key
    zpath = zip_path(raw_dataset_dir)
    output_path = paths["h5ad"] / f"{SAMPLE_ID}.h5ad"
    work_dir = args.scratch_dir / DATASET_KEY
    with zipfile.ZipFile(zpath) as zf:
        matrix_h5 = extract_member(zf, "cell_feature_matrix.h5", work_dir)
        cells_csv = extract_member(zf, "cells.csv.gz", work_dir)
        experiment_json = json.loads(zf.read("experiment.xenium").decode("utf-8"))
        metrics_csv = pd.read_csv(zf.open("metrics_summary.csv"))
        gene_panel = json.loads(zf.read("gene_panel.json").decode("utf-8"))

    adata = sc.read_10x_h5(matrix_h5, gex_only=False)
    adata.X = to_csr_counts(adata.X)
    adata.obs_names = adata.obs_names.astype(str)
    add_counts_layer(adata, adata.X)
    gene_ids = adata.var["gene_ids"].astype(str).to_numpy() if "gene_ids" in adata.var else adata.var_names.astype(str)
    add_var_metadata(adata, gene_ids=gene_ids, gene_symbols=adata.var_names.astype(str), species="human")

    cells = pd.read_csv(cells_csv)
    cells = cells.set_index("cell_id")
    missing = adata.obs_names.difference(cells.index)
    if len(missing):
        raise ValueError(f"{len(missing)} matrix cells missing from cells.csv.gz")
    cells = cells.loc[adata.obs_names]
    for col in cells.columns:
        adata.obs[col] = cells[col].to_numpy()

    add_standard_obs(
        adata,
        dataset_key=args.dataset_key,
        sample_id=SAMPLE_ID,
        platform="10x Xenium",
        species="human",
        tissue="non-diseased lymph node",
        condition="normal",
        donor_id="unknown",
        replicate_id="section",
        section_id=SAMPLE_ID,
    )
    set_spatial(adata, cells[["x_centroid", "y_centroid"]].to_numpy(dtype=float))
    labels = pd.Series("unknown", index=adata.obs_names)
    label_source = "not_available_in_raw_tree"
    set_labels(
        adata,
        region=labels,
        cell_type=labels,
        annotation=labels,
        ground_truth=labels,
        label_source=label_source,
    )
    panel_targets = gene_panel.get("payload", {}).get("targets", [])
    adata.uns["xenium_gene_panel_target_count"] = len(panel_targets)
    adata.uns["xenium_metrics_summary_json"] = metrics_csv.to_json(orient="records")
    adata, stats = apply_qc(
        adata,
        min_counts=args.min_counts,
        min_genes=args.min_genes,
        min_cells_per_gene=args.min_cells_per_gene,
        max_pct_mt=args.max_pct_mt,
    )
    adata.uns["preprocessing"] = preprocessing_uns(
        dataset_key=args.dataset_key,
        sample_id=SAMPLE_ID,
        input_files=[zpath],
        counts_source="cell_feature_matrix.h5",
        coordinate_source="cells.csv.gz:x_centroid,y_centroid",
        label_source=label_source,
        seed=args.seed,
        script_path=SCRIPT_PATH,
        extra={
            "xenium_analysis_sw_version": experiment_json.get("analysis_sw_version"),
            "label_status": "annotation_not_required_for_this_dataset",
            "raw_references": {
                "analysis": "analysis.tar.gz",
                "transcripts": "transcripts.csv.gz",
                "cell_boundaries": "cell_boundaries.csv.gz",
                "nucleus_boundaries": "nucleus_boundaries.csv.gz",
                "morphology": experiment_json.get("images", {}),
            },
        },
    )
    write_processed(adata, output_path, overwrite=args.overwrite)
    return manifest_row(
        dataset_key=args.dataset_key,
        sample_id=SAMPLE_ID,
        input_files=[zpath],
        output_h5ad=output_path,
        stats=stats,
        label_source=label_source,
        coordinate_source="cells.csv.gz:x_centroid,y_centroid",
        counts_source="cell_feature_matrix.h5",
        seed=args.seed,
        script_path=SCRIPT_PATH,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser, DATASET_KEY)
    parser.add_argument("--scratch-dir", type=Path, default=Path("/tmp"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ensure_processed_dirs(args.processed_root, args.dataset_key)
    setup_logging(paths["logs"], args.dataset_key)
    zpath = zip_path(args.raw_root / args.dataset_key)
    if args.dry_run:
        print_inventory([{"sample_id": SAMPLE_ID, "input": str(zpath), "exists": zpath.exists()}])
        return
    row = process(args, paths)
    manifest_path = append_manifest_rows(paths["manifest"], [row])
    logging.info("Wrote manifest %s", manifest_path)


if __name__ == "__main__":
    main()
