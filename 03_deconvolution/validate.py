#!/usr/bin/env python3
"""Validate the fresh Study 03 run and its matched-support score layer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml


STUDY_ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def method_source_sha256(method: str) -> str:
    script = STUDY_ROOT / "methods" / (
        "run_rctd.py" if method == "rctd" else "run_cell2location.py"
    )
    sources = [script]
    if method == "rctd":
        sources.append(script.with_name("run_rctd_backend.R"))
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(source)))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=STUDY_ROOT / "config.yaml")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    simulations = pd.read_csv(args.run_dir / "simulation_manifest.csv", dtype={"slice_id": str})
    methods = pd.read_csv(args.run_dir / "method_manifest.csv", dtype={"slice_id": str})
    config = yaml.safe_load(args.config.read_text())
    config_hash = sha256_file(args.config)
    expected_pairs = {
        f"{slice_id}__resolution_{float(resolution):g}"
        for slice_id in config["slices"]
        for resolution in config["resolutions"]
    }
    expected_methods = {
        (pair_id, method)
        for pair_id in expected_pairs
        for method in ("rctd", "cell2location")
    }
    if (
        len(simulations) != 6
        or set(simulations["pair_id"]) != expected_pairs
        or simulations["pair_id"].duplicated().any()
        or len(methods) != 12
        or set(zip(methods["pair_id"], methods["method"])) != expected_methods
        or methods[["pair_id", "method"]].duplicated().any()
    ):
        raise RuntimeError("Study 03 manifests do not contain the exact declared job matrix")
    rows = []
    for simulation in simulations.itertuples(index=False):
        path = Path(simulation.simulation_path)
        adata = ad.read_h5ad(path)
        truth = pd.read_csv(simulation.truth_path, index_col=0)
        valid = bool(
            sha256_file(path) == simulation.simulation_sha256
            and sha256_file(Path(simulation.truth_path)) == simulation.truth_sha256
            and adata.obs_names.equals(pd.Index([str(i) for i in range(adata.n_obs)]))
            and simulation.config_sha256 == config_hash
            and simulation.feast_commit == config["required_feast_commit"]
            and int(simulation.public_seed) == int(config["seed"])
            and "n_source_cells" in adata.obs
            and int(adata.obs["n_source_cells"].sum())
            == int(adata.uns["feast_reproduce"]["solver_diagnostics"]["aggregated_source_cell_count"])
            and truth.shape == np.asarray(adata.obsm["cell_type_proportions"]).shape
            and np.allclose(truth.to_numpy(), adata.obsm["cell_type_proportions"], atol=1e-12, rtol=0)
        )
        rows.append({"kind": "simulation", "job_id": simulation.pair_id, "valid": valid})
    for method in methods.itertuples(index=False):
        output, metadata_path = Path(method.output_path), Path(method.metadata_path)
        metadata = json.loads(metadata_path.read_text())
        prediction = pd.read_csv(output, index_col=0)
        simulation = ad.read_h5ad(method.simulation_path)
        libraries = np.asarray(simulation.X.sum(axis=1)).ravel()
        expected = pd.Index(
            [f"spot_{i}" for i in range(simulation.n_obs)]
            if method.method == "cell2location"
            else [f"spot_{i}" for i in np.flatnonzero(libraries > 0)]
        )
        valid = bool(
            method.status == "ok" and metadata.get("status") == "validated_success"
            and sha256_file(output) == method.output_sha256
            and sha256_file(metadata_path) == method.metadata_sha256
            and metadata.get("input_sha256") == method.simulation_sha256
            and metadata.get("reference_sha256") == method.reference_sha256
            and metadata.get("output_sha256") == method.output_sha256
            and int(metadata.get("public_seed", -1)) == int(config["seed"])
            and method.config_sha256 == config_hash
            and method.method_source_sha256 == method_source_sha256(method.method)
            and pd.Index(prediction.index.astype(str)).equals(expected)
            and np.isfinite(prediction.to_numpy(np.float64)).all()
        )
        if method.method == "cell2location":
            valid = bool(
                valid
                and metadata.get("actual_accelerator") == "cuda"
                and metadata.get("cuda_execution_verified") is True
                and str(metadata.get("actual_device", "")).startswith("cuda:")
                and int(metadata.get("peak_gpu_memory_allocated_bytes", 0)) > 0
                and metadata.get(
                    "abundance_named_columns_validated_and_ordered"
                )
                is True
                and metadata.get("abundance_column_schema")
                in {
                    "cell2location_prefixed_exact_order",
                    "factor_names_exact_order",
                }
                and metadata.get("n_cells_per_location_policy", "").startswith(
                    "mean exact high-resolution source-cell assignments"
                )
                and "unsupported" in metadata.get("external_environment_limitation", "")
            )
        else:
            valid = bool(
                valid
                and metadata.get("rng_policy")
                == "R set.seed(public_seed) before createRctd/runRctd"
                and set(
                    ("R", "spacexr", "SummarizedExperiment", "SpatialExperiment")
                ).issubset(metadata.get("environment", {}))
                and metadata.get("solver_diagnostics", {}).get(
                    "weights_all_finite"
                )
                is True
            )
        rows.append({"kind": "method", "job_id": f"{method.pair_id}__{method.method}", "valid": bool(valid)})
    scores = pd.read_csv(args.scores_dir / "deconvolution_scores.csv", dtype={"slice_id": str})
    summary = pd.read_csv(args.scores_dir / "deconvolution_summary.csv")
    decision = json.loads((args.scores_dir / "publication_decision.json").read_text())
    score_provenance = json.loads((args.scores_dir / "provenance.json").read_text())
    score_hashes_valid = all(
        score_provenance.get("outputs", {}).get(name) == sha256_file(args.scores_dir / name)
        for name in (
            "deconvolution_scores.csv", "deconvolution_summary.csv",
            "support_audit.csv", "publication_decision.json",
        )
    )
    score_lineage_valid = bool(
        score_provenance.get("simulation_manifest_sha256")
        == sha256_file(args.run_dir / "simulation_manifest.csv")
        and score_provenance.get("method_manifest_sha256")
        == sha256_file(args.run_dir / "method_manifest.csv")
        and decision.get("historical_direction_audit_sha256")
        == "69e0154042552f86e0cc778ac3963c2de073226a260f2b122d954febbfec58f9"
        and decision.get("status")
        in {
            "candidate_direction_unchanged_pending_author_review",
            "stop_headline_direction_changed",
        }
    )
    score_valid = bool(
        len(scores) == 12 and scores.duplicated(["method", "pair_id"]).sum() == 0
        and set(scores["method"]) == {"rctd", "cell2location"}
        and (scores["status"] == "ok").all() and len(summary) == 2
        and decision.get("publication_claim_authorized") is False
        and decision.get("aggregate_winner_ranking_authorized") is False
        and score_lineage_valid
        and score_hashes_valid
    )
    rows.append({"kind": "score_table", "job_id": "deconvolution_scores", "valid": score_valid})
    result = pd.DataFrame(rows)
    if not result["valid"].all():
        raise RuntimeError("Study 03 validation failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Study 03 validated; publication decision: {decision['status']}")
    return 0 if decision["status"] != "stop_headline_direction_changed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
