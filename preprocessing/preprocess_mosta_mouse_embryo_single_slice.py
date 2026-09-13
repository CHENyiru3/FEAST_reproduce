from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from preprocess_common import (
    add_common_arguments,
    add_counts_layer,
    add_standard_obs,
    add_var_metadata,
    append_manifest_rows,
    apply_qc,
    ensure_processed_dirs,
    filter_requested_samples,
    manifest_row,
    preprocessing_uns,
    print_inventory,
    read_raw_manifest,
    set_labels,
    setup_logging,
    set_spatial,
    to_csr_counts,
    write_processed,
)


DATASET_KEY = "MOSTA_mouse_embryo_single_slice_h5ad"
SCRIPT_PATH = Path(__file__).resolve()


def parse_sample_id(sample_id: str) -> tuple[str, str, str]:
    match = re.match(r"^(E\d+(?:\.\d+)?)_(E\d+)S(\d+)$", sample_id)
    if not match:
        return "unknown", "unknown", sample_id
    stage, embryo, section_num = match.groups()
    return stage, embryo, f"S{section_num}"


def discover_samples(raw_dataset_dir: Path) -> list[tuple[str, Path]]:
    manifest = read_raw_manifest(raw_dataset_dir)
    rows = []
    for _, row in manifest.iterrows():
        sample_id = str(row["SampleID"])
        filename = Path(str(row["url"])).name
        path = raw_dataset_dir / "samples" / sample_id / filename
        rows.append((sample_id, path))
    return rows


def process_one(sample_id: str, input_path: Path, args, paths: dict[str, Path]) -> dict:
    output_path = paths["h5ad"] / f"{sample_id}.h5ad"
    logging.info("Processing %s", sample_id)
    adata = ad.read_h5ad(input_path)
    if "count" not in adata.layers:
        raise ValueError(f"{sample_id} is missing layers['count']")
    if "spatial" not in adata.obsm:
        raise ValueError(f"{sample_id} is missing obsm['spatial']")
    if "annotation" not in adata.obs:
        raise ValueError(f"{sample_id} is missing obs['annotation']")

    counts = to_csr_counts(adata.layers["count"])
    adata.X = counts
    add_counts_layer(adata, counts)
    add_var_metadata(adata, species="mouse")
    stage, embryo_id, section_id = parse_sample_id(sample_id)
    add_standard_obs(
        adata,
        dataset_key=args.dataset_key,
        sample_id=sample_id,
        platform="Stereo-seq",
        species="mouse",
        tissue="mouse embryo",
        condition=stage,
        donor_id=embryo_id,
        replicate_id=section_id,
        section_id=section_id,
    )
    set_spatial(adata, np.asarray(adata.obsm["spatial"], dtype=float))
    labels = adata.obs["annotation"].astype(str)
    set_labels(
        adata,
        region=labels,
        cell_type=pd.Series("unknown", index=adata.obs_names),
        annotation=labels,
        ground_truth=labels,
        label_source="obs.annotation",
    )
    adata.obs["development_stage"] = stage
    adata.obs["embryo_id"] = embryo_id
    adata, stats = apply_qc(
        adata,
        min_counts=args.min_counts,
        min_genes=args.min_genes,
        min_cells_per_gene=args.min_cells_per_gene,
        max_pct_mt=args.max_pct_mt,
    )
    adata.uns["preprocessing"] = preprocessing_uns(
        dataset_key=args.dataset_key,
        sample_id=sample_id,
        input_files=[input_path],
        counts_source="layers.count",
        coordinate_source="obsm.spatial",
        label_source="obs.annotation",
        seed=args.seed,
        script_path=SCRIPT_PATH,
        extra={"development_stage": stage, "embryo_id": embryo_id, "section_id": section_id},
    )
    write_processed(adata, output_path, overwrite=args.overwrite)
    return manifest_row(
        dataset_key=args.dataset_key,
        sample_id=sample_id,
        input_files=[input_path],
        output_h5ad=output_path,
        stats=stats,
        label_source="obs.annotation",
        coordinate_source="obsm.spatial",
        counts_source="layers.count",
        seed=args.seed,
        script_path=SCRIPT_PATH,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser, DATASET_KEY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dataset_dir = args.raw_root / args.dataset_key
    paths = ensure_processed_dirs(args.processed_root, args.dataset_key)
    setup_logging(paths["logs"], args.dataset_key)
    samples = discover_samples(raw_dataset_dir)
    selected_ids = filter_requested_samples([sid for sid, _ in samples], args.samples)
    samples = [(sid, path) for sid, path in samples if sid in selected_ids]
    inventory = [{"sample_id": sid, "input": str(path), "exists": path.exists()} for sid, path in samples]
    if args.dry_run:
        print_inventory(inventory)
        return
    rows = [process_one(sid, path, args, paths) for sid, path in samples]
    manifest_path = append_manifest_rows(paths["manifest"], rows)
    logging.info("Wrote manifest %s", manifest_path)


if __name__ == "__main__":
    main()
