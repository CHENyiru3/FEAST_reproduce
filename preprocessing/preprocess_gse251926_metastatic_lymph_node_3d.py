from __future__ import annotations

import argparse
import gzip
import logging
import shutil
from pathlib import Path

import anndata as ad
import numpy as np

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


DATASET_KEY = "GSE251926_metastatic_lymph_node_3d"
SAMPLE_ID = "GSE251926_metastatic_lymph_node_3d"
SCRIPT_PATH = Path(__file__).resolve()


def input_file(raw_dataset_dir: Path) -> Path:
    return raw_dataset_dir / "samples" / "GSE251926" / "GSE251926_metastatic_lymph_node_3d.h5ad.gz"


def inflate_gzip(src: Path, scratch_dir: Path) -> Path:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    dest = scratch_dir / src.name.removesuffix(".gz")
    logging.info("Inflating %s to %s", src, dest)
    with gzip.open(src, "rb") as inp, dest.open("wb") as out:
        shutil.copyfileobj(inp, out, length=1024 * 1024 * 32)
    return dest


def process(args, paths: dict[str, Path]) -> dict:
    raw_dataset_dir = args.raw_root / args.dataset_key
    src = input_file(raw_dataset_dir)
    output_path = paths["h5ad"] / f"{SAMPLE_ID}.h5ad"
    inflated = inflate_gzip(src, args.scratch_dir)
    try:
        adata = ad.read_h5ad(inflated)
        if "raw" not in adata.layers:
            raise ValueError("Input is missing layers['raw']")
        if "spatial" not in adata.obsm:
            raise ValueError("Input is missing obsm['spatial']")
        if "spatial_3d_aligned" not in adata.obsm:
            raise ValueError("Input is missing obsm['spatial_3d_aligned']")
        if "annotation" not in adata.obs:
            raise ValueError("Input is missing obs['annotation']")

        counts = to_csr_counts(adata.layers["raw"])
        adata.X = counts
        add_counts_layer(adata, counts)
        add_var_metadata(adata, species="human")
        section = adata.obs["n_section"].astype(str) if "n_section" in adata.obs else "unknown"
        add_standard_obs(
            adata,
            dataset_key=args.dataset_key,
            sample_id=SAMPLE_ID,
            platform="spatial transcriptomics 3D cell atlas",
            species="human",
            tissue="metastatic lymph node",
            section_id="multi_section",
        )
        adata.obs["section_id"] = section
        set_spatial(
            adata,
            np.asarray(adata.obsm["spatial"], dtype=float),
            xyz=np.asarray(adata.obsm["spatial_3d_aligned"], dtype=float),
        )
        labels = adata.obs["annotation"].astype(str)
        set_labels(
            adata,
            region=labels,
            cell_type=labels,
            annotation=labels,
            ground_truth=labels,
            label_source="obs.annotation",
        )
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
            input_files=[src],
            counts_source="layers.raw",
            coordinate_source="obsm.spatial;obsm.spatial_3d_aligned",
            label_source="obs.annotation",
            seed=args.seed,
            script_path=SCRIPT_PATH,
        )
        write_processed(adata, output_path, overwrite=args.overwrite)
        return manifest_row(
            dataset_key=args.dataset_key,
            sample_id=SAMPLE_ID,
            input_files=[src],
            output_h5ad=output_path,
            stats=stats,
            label_source="obs.annotation",
            coordinate_source="obsm.spatial;obsm.spatial_3d_aligned",
            counts_source="layers.raw",
            seed=args.seed,
            script_path=SCRIPT_PATH,
        )
    finally:
        if args.keep_inflated:
            logging.info("Keeping inflated file %s", inflated)
        else:
            inflated.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser, DATASET_KEY)
    parser.add_argument("--scratch-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--keep-inflated", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ensure_processed_dirs(args.processed_root, args.dataset_key)
    setup_logging(paths["logs"], args.dataset_key)
    src = input_file(args.raw_root / args.dataset_key)
    if args.dry_run:
        print_inventory([{"sample_id": SAMPLE_ID, "input": str(src), "exists": src.exists()}])
        return
    row = process(args, paths)
    manifest_path = append_manifest_rows(paths["manifest"], [row])
    logging.info("Wrote manifest %s", manifest_path)


if __name__ == "__main__":
    main()
