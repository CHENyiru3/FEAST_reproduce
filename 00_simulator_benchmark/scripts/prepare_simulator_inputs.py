#!/usr/bin/env python3
"""Materialize Figure 2 simulator benchmark input aliases under exper_data/.

Symlinks standalone processed h5ad files. Subsets section-specific slices
from the GSE251926 3D object for OpenST_005 and OpenST_006.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

DATASETS_ROOT = Path("/maiziezhou_lab2/yiru/Datasets")

ALIAS_REGISTRY = {
    "DLPFC_151670": {
        "technology": "10x Visium",
        "source": "spatialLIBD",
        "source_path": DATASETS_ROOT
        / "Processed/spatialLIBD_DLPFC_Visium/h5ad/151670.h5ad",
        "subset": None,
    },
    "DLPFC_151675": {
        "technology": "10x Visium",
        "source": "spatialLIBD",
        "source_path": DATASETS_ROOT
        / "Processed/spatialLIBD_DLPFC_Visium/h5ad/151675.h5ad",
        "subset": None,
    },
    "DLPFC_151676": {
        "technology": "10x Visium",
        "source": "spatialLIBD",
        "source_path": DATASETS_ROOT
        / "Processed/spatialLIBD_DLPFC_Visium/h5ad/151676.h5ad",
        "subset": None,
    },
    "MERFISH_006": {
        "technology": "MERFISH",
        "source": "Allen Brain Atlas",
        "source_path": DATASETS_ROOT
        / "Processed/Allen_Zhuang_ABCA_1/h5ad/Zhuang-ABCA-1.006.h5ad",
        "subset": None,
    },
    "MERFISH_007": {
        "technology": "MERFISH",
        "source": "Allen Brain Atlas",
        "source_path": DATASETS_ROOT
        / "Processed/Allen_Zhuang_ABCA_1/h5ad/Zhuang-ABCA-1.007.h5ad",
        "subset": None,
    },
    "MERFISH_120": {
        "technology": "MERFISH",
        "source": "Allen Brain Atlas",
        "source_path": DATASETS_ROOT
        / "Processed/Allen_Zhuang_ABCA_1/h5ad/Zhuang-ABCA-1.120.h5ad",
        "subset": None,
    },
    "OpenST_005": {
        "technology": "OpenST",
        "source": "GEO: GSE251926",
        "source_path": DATASETS_ROOT
        / "Processed/GSE251926_metastatic_lymph_node_3d/h5ad/GSE251926_metastatic_lymph_node_3d.h5ad",
        "subset": {"column": "n_section", "value": 5},
    },
    "OpenST_006": {
        "technology": "OpenST",
        "source": "GEO: GSE251926",
        "source_path": DATASETS_ROOT
        / "Processed/GSE251926_metastatic_lymph_node_3d/h5ad/GSE251926_metastatic_lymph_node_3d.h5ad",
        "subset": {"column": "n_section", "value": 6},
    },
    "Stereoseq_E9_5_E2S2": {
        "technology": "Stereo-seq",
        "source": "MOSTA",
        "source_path": DATASETS_ROOT
        / "Processed/MOSTA_mouse_embryo_single_slice_h5ad/h5ad/E9.5_E2S2.h5ad",
        "subset": None,
    },
    "Stereoseq_E10_5_E2S1": {
        "technology": "Stereo-seq",
        "source": "MOSTA",
        "source_path": DATASETS_ROOT
        / "Processed/MOSTA_mouse_embryo_single_slice_h5ad/h5ad/E10.5_E2S1.h5ad",
        "subset": None,
    },
    "Stereoseq_E12_5_E2S1": {
        "technology": "Stereo-seq",
        "source": "MOSTA",
        "source_path": DATASETS_ROOT
        / "Processed/MOSTA_mouse_embryo_single_slice_h5ad/h5ad/E12.5_E2S1.h5ad",
        "subset": None,
    },
    "Slideseq_001": {
        "technology": "Slide-seqV2",
        "source": "SODB/Tencent",
        "source_path": DATASETS_ROOT
        / "Processed/Tencent_SpatialOmics_dataset119/h5ad/GSM6025940_MBM08.h5ad",
        "subset": None,
    },
    "Xenium_LymphNode": {
        "technology": "Xenium",
        "source": "10x Genomics",
        "source_path": DATASETS_ROOT
        / "Processed/10x_Xenium_human_lymph_node_preview/h5ad/Xenium_V1_hLymphNode_nondiseased_section.h5ad",
        "subset": None,
    },
}

LABEL_COLUMNS = ["ground_truth", "cell_type", "annotation", "region", "cluster"]


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_label_column(adata: ad.AnnData) -> str | None:
    for col in LABEL_COLUMNS:
        if col in adata.obs.columns:
            return col
    return None


def _validate_adata(adata: ad.AnnData, alias: str) -> list[str]:
    errors = []
    if adata.n_obs == 0 or adata.n_vars == 0:
        errors.append(f"{alias}: empty AnnData ({adata.n_obs} x {adata.n_vars})")
    if "spatial" not in adata.obsm or adata.obsm["spatial"].shape[1] < 2:
        errors.append(f"{alias}: missing or invalid obsm['spatial']")
    if _resolve_label_column(adata) is None:
        errors.append(
            f"{alias}: no label column found among {LABEL_COLUMNS}; "
            f"available: {list(adata.obs.columns)}"
        )
    return errors


def prepare_inputs(output_dir: Path, manifest_path: Path, copy: bool = False) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    errors = []

    for alias, cfg in ALIAS_REGISTRY.items():
        source = Path(cfg["source_path"])
        if not source.exists():
            errors.append(f"{alias}: source not found: {source}")
            manifest_rows.append({**cfg, "alias": alias, "status": "missing_source"})
            continue

        dest = output_dir / f"{alias}.h5ad"

        if cfg["subset"] is None:
            if copy:
                import shutil
                shutil.copy2(source, dest)
            elif not dest.exists():
                os.symlink(source, dest)
            adata = ad.read_h5ad(dest)
        else:
            adata_full = ad.read_h5ad(source)
            col = cfg["subset"]["column"]
            val = cfg["subset"]["value"]
            mask = adata_full.obs[col] == val
            adata = adata_full[mask, :].copy()
            adata.uns["figure2_simulator_input"] = {
                "alias": alias,
                "source_path": str(source),
                "subset_column": col,
                "subset_value": val,
            }
            adata.write_h5ad(dest, compression="gzip")

        validation_errors = _validate_adata(adata, alias)
        label_col = _resolve_label_column(adata)
        checksum = _sha256(dest)

        row = {
            "alias": alias,
            "technology": cfg["technology"],
            "source": cfg["source"],
            "source_path": str(source),
            "subset": str(cfg["subset"]) if cfg["subset"] else "none",
            "n_obs": adata.n_obs,
            "n_vars": adata.n_vars,
            "label_column": label_col or "none",
            "checksum": checksum,
            "status": "valid" if not validation_errors else f"invalid: {'; '.join(validation_errors)}",
        }
        manifest_rows.append(row)

        if validation_errors:
            errors.extend(validation_errors)
            print(f"  WARNING: {alias} — {'; '.join(validation_errors)}")
        else:
            print(f"  {alias}: {adata.n_obs} spots x {adata.n_vars} genes [{label_col}]")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(manifest_path, index=False)
    print(f"\nManifest written to {manifest_path} ({len(manifest)} rows)")

    if errors:
        print(f"\n{len(errors)} validation issue(s):")
        for e in errors:
            print(f"  - {e}")

    return 0 if not errors else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=DATASETS_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for alias .h5ad files (e.g., exper_data)")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Path for input_manifest.csv")
    parser.add_argument("--copy", action="store_true",
                        help="Copy instead of symlinking standalone inputs")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    np.random.seed(args.seed)

    return prepare_inputs(args.output_dir, args.manifest, copy=args.copy)


if __name__ == "__main__":
    raise SystemExit(main())
