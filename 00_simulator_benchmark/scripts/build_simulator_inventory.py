#!/usr/bin/env python3
"""Build simulator-sample inventory by scanning outputs/simulation_saved/."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import anndata as ad
import pandas as pd

SIMULATOR_LABELS: dict[str, str] = {
    "FEAST_Rank": "FEAST_Rank",
    "FEAST_OT_Spatial": "FEAST_OT_Spatial",
    "srtsim_updated": "SRTsim",
    "splat": "Splatter",
    "splatSimple": "Splatter_Simple",
    "sccube": "scCube",
}

EXCLUDED = {
    ("FEAST_Rank", "Stereoseq_E9_5_E2S2"),
    ("FEAST_OT_Spatial", "Stereoseq_E9_5_E2S2"),
    ("SRTsim", "Stereoseq_E9_5_E2S2"),
    ("Splatter", "Stereoseq_E9_5_E2S2"),
    ("Splatter_Simple", "Stereoseq_E9_5_E2S2"),
    ("scCube", "Stereoseq_E9_5_E2S2"),
}


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_inventory(
    output_base: Path,
    exper_data: Path,
    inventory_path: Path,
) -> int:
    input_files = sorted(exper_data.glob("*.h5ad"))
    input_aliases = {p.stem for p in input_files}

    rows = []
    for folder_name, label in SIMULATOR_LABELS.items():
        sim_dir = output_base / folder_name
        if not sim_dir.is_dir():
            for alias in sorted(input_aliases):
                rows.append({
                    "simulator": label,
                    "sample": alias,
                    "status": "missing",
                })
            continue

        for h5ad_path in sorted(sim_dir.glob("*.h5ad")):
            stem = h5ad_path.stem
            # Handle scCube naming: <alias>_sccube_sim
            if folder_name == "sccube" and stem.endswith("_sccube_sim"):
                alias = stem[: -len("_sccube_sim")]
            else:
                alias = stem

            if alias not in input_aliases:
                rows.append({
                    "simulator": label, "sample": alias,
                    "status": "unknown_alias",
                })
                continue

            if (label, alias) in EXCLUDED:
                rows.append({
                    "simulator": label, "sample": alias,
                    "status": "excluded_by_config",
                })
                continue

            try:
                adata = ad.read_h5ad(h5ad_path)
                n_obs, n_vars = adata.n_obs, adata.n_vars
                has_spatial = "spatial" in adata.obsm and adata.obsm["spatial"].shape[1] >= 2
                checksum = _sha256(h5ad_path)

                if n_obs == 0 or n_vars == 0:
                    status = "invalid"
                elif not has_spatial:
                    status = "invalid"
                else:
                    status = "matched"

                rows.append({
                    "simulator": label, "sample": alias,
                    "n_obs": n_obs, "n_vars": n_vars,
                    "has_spatial": has_spatial,
                    "checksum": checksum,
                    "file_path": str(h5ad_path),
                    "status": status,
                })
            except Exception as exc:
                rows.append({
                    "simulator": label, "sample": alias,
                    "status": "failed",
                    "error": str(exc)[:200],
                })

        # Check for missing samples
        matched_aliases = {r["sample"] for r in rows if r["simulator"] == label}
        for alias in sorted(input_aliases - matched_aliases):
            if (label, alias) in EXCLUDED:
                rows.append({
                    "simulator": label, "sample": alias,
                    "status": "excluded_by_config",
                })
            else:
                rows.append({
                    "simulator": label, "sample": alias,
                    "status": "missing_simulation",
                })

    inventory = pd.DataFrame(rows)
    inventory.to_csv(inventory_path, index=False)

    status_counts = inventory["status"].value_counts().to_dict()
    print(f"Inventory: {len(inventory)} rows")
    print(f"Status counts: {status_counts}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-base", type=Path, required=True,
                        help="Path to outputs/simulation_saved/")
    parser.add_argument("--exper-data", type=Path, required=True,
                        help="Path to exper_data/")
    parser.add_argument("--inventory", type=Path, required=True,
                        help="Output path for simulator_inventory.csv")
    args = parser.parse_args()

    return build_inventory(args.output_base, args.exper_data, args.inventory)


if __name__ == "__main__":
    raise SystemExit(main())
