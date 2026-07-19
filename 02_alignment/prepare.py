#!/usr/bin/env python3
"""Create the 20 publication alignment rotation inputs in a fresh directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from FEAST import Alteration, __version__ as feast_version, simulate
from FEAST.alignment import apply_spatial_transform, rotate_spatial


CONFIGURATION_ID = "study02-centered-rigid-rotation-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--feast-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.yaml")
    )
    args = parser.parse_args()

    config = load_config(args.config)
    required_commit = str(config["required_feast_commit"])
    if args.feast_commit != required_commit:
        raise RuntimeError(
            f"FEAST commit {args.feast_commit} does not match required {required_commit}"
        )
    if str(feast_version) != "1.0.2":
        raise RuntimeError(f"unexpected FEAST version: {feast_version}")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output root: {args.output_dir}")
    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    input_manifest = pd.read_csv(
        Path(__file__).parent / "data" / "input_checksums.csv"
    )
    if len(input_manifest) != 1 or sha256_file(args.reference) != str(
        input_manifest.iloc[0]["sha256"]
    ):
        raise RuntimeError("alignment reference does not match its input manifest")

    raw_reference = ad.read_h5ad(args.reference)
    reference_coords = np.asarray(raw_reference.obsm["spatial"], dtype=np.float64)
    if not raw_reference.obs_names.is_unique or not np.isfinite(reference_coords).all():
        raise RuntimeError("reference spot IDs/coordinates are invalid")

    tolerance = float(config["geometry_tolerance"])
    seed = int(config["seed"])
    rows: list[dict] = []
    args.output_dir.mkdir(parents=True, exist_ok=False)
    base_dir = args.output_dir / "base_simulations"
    base_dir.mkdir()

    # Match the declared article design: filter the raw reference once, then
    # call only the public FEAST simulation API for each fresh base dataset.
    import scanpy as sc

    reference = raw_reference.copy()
    sc.pp.filter_genes(reference, min_cells=int(config["simulation"]["min_cells"]))
    simulation = config["simulation"]

    for alteration in config["alterations"]:
        parameters = config["alteration_parameters"][alteration]
        if parameters["type"] == "baseline":
            alteration_config = Alteration()
        elif parameters["type"] == "mean":
            alteration_config = Alteration.mean_only(float(parameters["fold_change"]))
        elif parameters["type"] == "variance":
            alteration_config = Alteration.variance_only(float(parameters["fold_change"]))
        elif parameters["type"] == "sparsity_logit":
            alteration_config = Alteration.sparsity_logit(float(parameters["logit_shift"]))
        else:
            raise RuntimeError(f"unsupported alteration type: {parameters['type']}")
        source = simulate(
            reference=reference,
            alteration=alteration_config,
            seed=seed,
            parameter_mode=str(simulation["parameter_mode"]),
            spatial_mode=str(simulation["spatial_mode"]),
            assignment_solver=str(simulation["assignment_solver"]),
            assignment_blocks=bool(simulation["assignment_blocks"]),
            ppf_method=str(simulation["ppf_method"]),
            beta_n_jobs=int(simulation["beta_n_jobs"]),
            beta_early_stopping_patience=int(simulation["beta_early_stopping_patience"]),
            convert_n_jobs=int(simulation["convert_n_jobs"]),
            verbose=True,
        )
        diagnostics = source.uns.get("simulation_diagnostics", {})
        assignment_diagnostics = diagnostics.get("copula_rank_diagnostics", {})
        if diagnostics.get("spatial_mode") != "reference_rank":
            raise RuntimeError(f"{alteration}: FEAST returned the wrong spatial mode")
        if assignment_diagnostics.get("assignment_blocks") is not False:
            raise RuntimeError(f"{alteration}: FEAST did not use global assignment")
        source_path = base_dir / f"{alteration}.h5ad"
        source.uns["feast_reproduce"] = {
            "configuration_id": CONFIGURATION_ID,
            "feast_version": str(feast_version),
            "feast_commit": args.feast_commit,
            "public_seed": seed,
            "alteration": alteration,
            "alteration_parameters": parameters,
            "parameter_mode": str(simulation["parameter_mode"]),
            "spatial_mode": str(simulation["spatial_mode"]),
            "assignment_solver": str(simulation["assignment_solver"]),
            "assignment_blocks": bool(simulation["assignment_blocks"]),
            "solver_diagnostics": {
                "status": "completed",
                "assignment_method": assignment_diagnostics.get("assignment_method"),
                "assignment_blocks": False,
                "assignment_solver_requested": "scipy",
                "model_selection_counts": diagnostics.get(
                    "model_selection_counts", {}
                ),
            },
        }
        source.write_h5ad(source_path, compression="gzip")
        source_coords = np.asarray(source.obsm["spatial"], dtype=np.float64)
        if not source.obs_names.equals(raw_reference.obs_names):
            raise RuntimeError(f"{alteration}: spot identity/order differs from reference")
        if not source.var_names.is_unique:
            raise RuntimeError(f"{alteration}: gene IDs are not unique")
        geometry_error = float(np.max(np.abs(source_coords - reference_coords)))
        if geometry_error > tolerance:
            raise RuntimeError(
                f"{alteration}: spatial geometry differs from reference by {geometry_error}"
            )
        source_hash = sha256_file(source_path)

        for angle_value in config["angles"]:
            angle = float(angle_value)
            rotated = rotate_spatial(source, angle)
            transform = rotated.uns["feast_alignment_transform"]
            restored = apply_spatial_transform(
                rotated.obsm["spatial"], transform["inverse_matrix"]
            )
            inverse_error = float(np.max(np.abs(restored - source_coords)))
            if inverse_error > tolerance:
                raise RuntimeError(
                    f"{alteration}/{angle:g}: inverse error {inverse_error}"
                )
            if not rotated.obs_names.equals(source.obs_names):
                raise RuntimeError(f"{alteration}/{angle:g}: spot order changed")
            if not rotated.var_names.equals(source.var_names):
                raise RuntimeError(f"{alteration}/{angle:g}: gene order changed")

            output = args.output_dir / f"{alteration}_rotated_{angle:g}.h5ad"
            rotated.uns["feast_reproduce"] = {
                "configuration_id": CONFIGURATION_ID,
                "feast_version": str(feast_version),
                "feast_commit": args.feast_commit,
                "public_seed": seed,
                "input_sha256": source_hash,
                "inverse_max_abs_error": inverse_error,
            }
            rotated.write_h5ad(output, compression="gzip")
            rows.append(
                {
                    "configuration_id": CONFIGURATION_ID,
                    "alteration": alteration,
                    "angle_degrees": angle,
                    "public_seed": seed,
                    "feast_version": str(feast_version),
                    "feast_commit": args.feast_commit,
                    "n_spots": rotated.n_obs,
                    "n_genes": rotated.n_vars,
                    "source_path": str(source_path.resolve()),
                    "source_sha256": source_hash,
                    "output_path": str(output.resolve()),
                    "output_sha256": sha256_file(output),
                    "reference_geometry_max_abs_error": geometry_error,
                    "inverse_max_abs_error": inverse_error,
                    "spot_order_preserved": True,
                    "gene_order_preserved": True,
                }
            )

    expected = len(config["alterations"]) * len(config["angles"])
    if len(rows) != expected or expected != 20:
        raise RuntimeError(f"expected 20 rotations, generated {len(rows)}")
    manifest_path = args.output_dir / "rotation_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    provenance = {
        "configuration_id": CONFIGURATION_ID,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "feast_version": str(feast_version),
        "feast_commit": args.feast_commit,
        "public_seed": seed,
        "reference_path": str(args.reference.resolve()),
        "reference_sha256": sha256_file(args.reference),
        "config_sha256": sha256_file(args.config),
        "manifest_sha256": sha256_file(manifest_path),
        "completed_rotations": len(rows),
        "fresh_base_simulations": len(config["alterations"]),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(f"generated {len(rows)} rotations in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
