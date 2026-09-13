from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    "Tencent_SpatialOmics_dataset119",
    "10x_Xenium_human_lymph_node_preview",
    "GSE251926_metastatic_lymph_node_3d",
    "GSE269617",
    "MOSTA_mouse_embryo_single_slice_h5ad",
    "Allen_Zhuang_ABCA_1",
    "spatialLIBD_DLPFC_Visium",
]

SCRIPTS = {
    "Tencent_SpatialOmics_dataset119": "preprocess_tencent_spatialomics_dataset119.py",
    "10x_Xenium_human_lymph_node_preview": "preprocess_10x_xenium_human_lymph_node.py",
    "GSE251926_metastatic_lymph_node_3d": "preprocess_gse251926_metastatic_lymph_node_3d.py",
    "GSE269617": "preprocess_gse269617.py",
    "MOSTA_mouse_embryo_single_slice_h5ad": "preprocess_mosta_mouse_embryo_single_slice.py",
    "Allen_Zhuang_ABCA_1": "preprocess_allen_zhuang_abca1.py",
    "spatialLIBD_DLPFC_Visium": "preprocess_spatiallibd_dlpfc_visium.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--raw-root", default=str(SCRIPT_DIR.parents[1] / "Datasets" / "Raw"))
    parser.add_argument("--processed-root", default=str(SCRIPT_DIR.parents[1] / "Datasets" / "Processed"))
    parser.add_argument("--seed", default="2026")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-labels", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures = []
    for dataset in args.datasets:
        script = SCRIPTS[dataset]
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / script),
            "--raw-root",
            args.raw_root,
            "--processed-root",
            args.processed_root,
            "--dataset-key",
            dataset,
            "--seed",
            args.seed,
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.allow_missing_labels:
            cmd.append("--allow-missing-labels")
        print("Running:", " ".join(cmd), flush=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode:
            failures.append((dataset, result.returncode))
            if not args.continue_on_error:
                break
    if failures:
        print("Failures:", failures, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
