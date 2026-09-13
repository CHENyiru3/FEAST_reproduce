from __future__ import annotations

import argparse
import logging
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


DATASET_KEY = "Tencent_SpatialOmics_dataset119"
SCRIPT_PATH = Path(__file__).resolve()


def drop_upstream_analysis_artifacts(adata: ad.AnnData) -> dict[str, list[str]]:
    removed = {
        "obs": [],
        "obsm": [],
        "varm": [],
        "obsp": [],
        "varp": [],
        "uns": [],
    }
    for key in ["leiden", "louvain", "kmeans", "cluster"]:
        if key in adata.obs:
            del adata.obs[key]
            removed["obs"].append(key)
    for key in list(adata.obsm.keys()):
        if key != "spatial":
            del adata.obsm[key]
            removed["obsm"].append(key)
    for key in list(adata.varm.keys()):
        del adata.varm[key]
        removed["varm"].append(key)
    for key in list(adata.obsp.keys()):
        del adata.obsp[key]
        removed["obsp"].append(key)
    for key in list(adata.varp.keys()):
        del adata.varp[key]
        removed["varp"].append(key)
    for key in list(adata.uns.keys()):
        del adata.uns[key]
        removed["uns"].append(key)
    return {slot: keys for slot, keys in removed.items() if keys}


def parse_sample_id(sample_id: str) -> tuple[str, str, str]:
    parts = sample_id.split("_", 1)
    label = parts[1] if len(parts) == 2 else sample_id
    tokens = label.split("_")
    donor = tokens[0]
    replicate = tokens[1] if len(tokens) > 1 and tokens[1].startswith("rep") else "rep1"
    condition = "MBM" if donor.startswith("MBM") else "ECM" if donor.startswith("ECM") else "unknown"
    return condition, donor, replicate


def discover_samples(raw_dataset_dir: Path) -> list[tuple[str, Path]]:
    manifest = read_raw_manifest(raw_dataset_dir)
    rows = []
    for _, row in manifest.iterrows():
        sample_id = str(row["SampleID"])
        filename = str(row["filename"])
        path = raw_dataset_dir / "samples" / sample_id / filename
        rows.append((sample_id, path))
    return rows


def process_one(sample_id: str, input_path: Path, args, paths: dict[str, Path]) -> dict:
    output_path = paths["h5ad"] / f"{sample_id}.h5ad"
    logging.info("Processing %s", sample_id)
    adata = ad.read_h5ad(input_path)
    if "spatial" not in adata.obsm:
        raise ValueError(f"{sample_id} is missing obsm['spatial']")
    removed_artifacts = drop_upstream_analysis_artifacts(adata)

    adata.X = to_csr_counts(adata.X)
    add_counts_layer(adata, adata.X)
    add_var_metadata(adata, species="human")
    condition, donor_id, replicate_id = parse_sample_id(sample_id)
    add_standard_obs(
        adata,
        dataset_key=args.dataset_key,
        sample_id=sample_id,
        platform="Slide-seqV2",
        species="human",
        tissue="melanoma metastasis",
        condition=condition,
        donor_id=donor_id,
        replicate_id=replicate_id,
    )
    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    set_spatial(adata, xy)
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
        counts_source="X",
        coordinate_source="obsm.spatial",
        label_source=label_source,
        seed=args.seed,
        script_path=SCRIPT_PATH,
        extra={
            "condition": condition,
            "donor_id": donor_id,
            "replicate_id": replicate_id,
            "label_status": "annotation_not_required_for_this_dataset",
            "removed_upstream_analysis_artifacts": removed_artifacts,
        },
    )
    write_processed(adata, output_path, overwrite=args.overwrite)
    return manifest_row(
        dataset_key=args.dataset_key,
        sample_id=sample_id,
        input_files=[input_path],
        output_h5ad=output_path,
        stats=stats,
        label_source=label_source,
        coordinate_source="obsm.spatial",
        counts_source="X",
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
