#!/usr/bin/env python3
"""Calibrate and freeze E18.5 assignment randomness from references only."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import FEAST
from FEAST.de_novo import ReferenceFitConfig, fit_reference

import workflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=workflow.STUDY_ROOT / "config.yaml")
    args = parser.parse_args()
    config = workflow.load_config(args.config)
    workflow.preflight_inputs(config, inspect_h5ad=True)
    feast = workflow.feast_identity(config)
    output = workflow.calibration_path(config)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {output}")
    workflow.verify_prepared_blueprint(config, "E18.5")

    reference_paths = workflow.reference_paths(config, "E18.5")
    references = []
    for path in reference_paths:
        reference = ad.read_h5ad(path)
        reference.uns["reference_name"] = path.stem
        references.append(reference)
    fit = config["reference_fit"]
    model = fit_reference(
        references,
        label_key=fit["label_key"],
        config=ReferenceFitConfig(
            min_gene_spots=int(fit["min_gene_spots"]),
            min_gene_mean=float(fit["min_gene_mean"]),
            max_gene_zero_prop=float(fit["max_gene_zero_prop"]),
            boundary_neighbors=int(fit["boundary_neighbors"]),
            coordinate_scale=fit["coordinate_scale"],
        ),
    )
    if len(model.gene_names) != int(fit["expected_genes"]):
        raise RuntimeError(f"reference fit retained {len(model.gene_names)} genes, expected 550")
    calibration = config["assignment_randomness_calibration"]
    value = FEAST.estimate_assignment_randomness(
        model,
        label_key=config["reference_fit"]["label_key"],
        n_genes=int(calibration["n_genes"]),
        ar_step=float(calibration["step"]),
        max_ar=float(calibration["maximum"]),
        n_neighbors=int(calibration["n_neighbors"]),
        random_seed=int(config["public_seed"]),
    )
    payload = {
        "schema_version": 1,
        "configuration_id": config["configuration_id"],
        "age": "E18.5",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "reference_only",
        "target_blueprint_access": False,
        "prepared_blueprint_integrity_gate": "verified_but_not_used_by_estimator",
        "public_seed": int(config["public_seed"]),
        "fitted_reference_count": len(reference_paths),
        "fitted_gene_count": len(model.gene_names),
        "reference_inputs": {
            path.name: workflow.sha256_file(path)
            for path in reference_paths
        },
        "estimator": "FEAST.estimate_assignment_randomness",
        "estimator_parameters": {
            "n_genes": int(calibration["n_genes"]),
            "ar_step": float(calibration["step"]),
            "max_ar": float(calibration["maximum"]),
            "n_neighbors": int(calibration["n_neighbors"]),
        },
        "assignment_randomness": float(value),
        "config_sha256": workflow.sha256_file(config["_config_path"]),
        "feast": feast,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"E18.5 assignment_randomness={value:.6g} -> {output}")


if __name__ == "__main__":
    main()
