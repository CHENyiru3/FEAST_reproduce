#!/usr/bin/env python3
"""Validate one complete fresh Study 04 run and write its decision record."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import run as workflow


STUDY_ROOT = Path(__file__).resolve().parent


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=STUDY_ROOT / "config.yaml")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = workflow.load_config(args.config)
    run_dir = args.run_dir.resolve() if args.run_dir else config["_output_dir"]
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "validation"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite validation output: {output_dir}")

    rows: list[dict[str, object]] = []
    simulation_root = run_dir / "simulations"
    manifest_path = simulation_root / "manifest.csv"
    simulation_provenance_path = simulation_root / "provenance.json"
    manifest = pd.read_csv(manifest_path)
    simulation_provenance = json.loads(
        simulation_provenance_path.read_text(encoding="utf-8")
    )
    expected_cells = {
        (mode, float(alpha))
        for mode in config["simulation"]["modes"]
        for alpha in config["simulation"]["alpha_levels"]
    }
    observed_cells = {
        (str(row.mode), float(row.alpha)) for row in manifest.itertuples(index=False)
    }
    check(len(manifest) == 14, "ladder must contain 14 files including two baselines")
    check(observed_cells == expected_cells, "ladder cell set differs from config")
    check(len(observed_cells) == len(manifest), "ladder contains duplicate cells")
    check(
        simulation_provenance.get("legacy_study_id") == "08",
        "simulation legacy study identity is missing",
    )
    check(
        simulation_provenance.get("solver_diagnostics", {}).get("comparison_conditions")
        == 12,
        "simulation does not declare 12 nonzero comparison conditions",
    )

    baseline_matrix_hashes = set()
    n_spots = None
    source_gene_order = None
    source_spot_order = None
    for row in manifest.itertuples(index=False):
        path = Path(row.file)
        file_hash = workflow.sha256_file(path)
        check(file_hash == str(row.output_sha256), f"simulation hash mismatch: {path}")
        candidate = ad.read_h5ad(path)
        matrix_hash = workflow.matrix_sha256(candidate.X)
        check(matrix_hash == str(row.matrix_sha256), f"matrix hash mismatch: {path}")
        values = candidate.X.data if hasattr(candidate.X, "data") else np.asarray(candidate.X)
        check(np.isfinite(values).all(), f"non-finite simulation: {path}")
        check(np.all(values >= 0), f"negative simulation: {path}")
        check(np.equal(values, np.floor(values)).all(), f"non-integral simulation: {path}")
        if n_spots is None:
            n_spots = int(candidate.n_obs)
            source_spot_order = candidate.obs_names.astype(str).tolist()
            source_gene_order = candidate.var_names.astype(str).tolist()
        check(int(candidate.n_obs) == n_spots, "simulation spot count changed")
        check(candidate.obs_names.astype(str).tolist() == source_spot_order, "spot order changed")
        check(candidate.var_names.astype(str).tolist() == source_gene_order, "gene order changed")
        if float(row.alpha) == 0.0:
            baseline_matrix_hashes.add(str(row.matrix_sha256))
        rows.append(
            {
                "artifact_type": "simulation",
                "method": None,
                "mode": row.mode,
                "alpha": float(row.alpha),
                "path": str(path.resolve()),
                "sha256": file_hash,
                "status": "validated",
            }
        )
    check(
        len(baseline_matrix_hashes) == 1,
        "the two alpha-zero baselines must have identical count matrices",
    )

    panel_path = run_dir / "panel" / "panel.csv"
    panel_provenance_path = run_dir / "panel" / "provenance.json"
    panel = pd.read_csv(panel_path)
    check(len(panel) == int(config["fixed_panel"]["n_genes"]), "panel size changed")
    check(panel["gene"].is_unique, "panel genes are not unique")
    check(
        panel["panel_order"].astype(int).tolist() == list(range(len(panel))),
        "panel order is not contiguous",
    )
    panel_provenance = json.loads(panel_provenance_path.read_text(encoding="utf-8"))
    check(
        panel_provenance.get("solver_diagnostics", {}).get("status") == "completed",
        "panel provenance is incomplete",
    )

    expected_feast_commit = simulation_provenance["feast"]["commit"]
    method_count = 0
    for method in workflow.METHOD_ARTIFACTS:
        for mode in config["simulation"]["modes"]:
            for alpha_value in config["simulation"]["alpha_levels"]:
                alpha = float(alpha_value)
                if alpha == 0.0:
                    continue
                candidate_id = (
                    f"{config['configuration_id']}__{method}__{mode}__alpha_{alpha:.2f}"
                )
                candidate = run_dir / "methods" / method / mode / f"alpha_{alpha:.2f}"
                valid, reason = workflow.candidate_is_valid(
                    candidate,
                    method,
                    candidate_id,
                    config,
                )
                check(valid, f"invalid {method} candidate: {reason}")
                metadata_path = candidate / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                provenance = metadata["provenance"]
                diagnostics = provenance.get("solver_diagnostics", {})
                check(
                    provenance["feast"]["commit"] == expected_feast_commit,
                    "method and simulation FEAST commits differ",
                )
                check(
                    diagnostics.get("cuda_execution_verified") is True
                    and str(diagnostics.get("actual_device", "")).startswith("cuda"),
                    f"{method} candidate does not prove CUDA execution",
                )
                check(
                    metadata.get("runtime_controls", {}).get("pip_check_status")
                    == "ok",
                    f"{method} environment pip check was not recorded as successful",
                )
                method_environment = (
                    metadata.get("execution_environment", {})
                    if method == "scVI"
                    else metadata.get("environment", {})
                )
                numpy_disposition = method_environment.get(
                    "numpy_supported_by_feast"
                )
                check(
                    numpy_disposition in {True, False},
                    f"{method} lacks an explicit FEAST NumPy support disposition",
                )
                if numpy_disposition is False:
                    check(
                        "unsupported"
                        in str(
                            method_environment.get(
                                "external_environment_limitation", ""
                            )
                        ),
                        f"{method} does not report unsupported external NumPy",
                    )
                result = ad.read_h5ad(candidate / "result.h5ad", backed="r")
                try:
                    check(result.n_obs == 2 * n_spots, "method spot support changed")
                    check(result.n_vars == len(panel), "method fixed-panel support changed")
                    check(
                        result.var_names.astype(str).tolist() == panel["gene"].astype(str).tolist(),
                        "method panel gene order changed",
                    )
                    check(result.obs_names.is_unique, "method output spot IDs are not unique")
                finally:
                    result.file.close()
                if method in {"GraphST", "STAMP"}:
                    graph = metadata.get("spatial_graph", {})
                    check(graph.get("schema_version") == 3, "graph schema version changed")
                    check(graph.get("cross_batch_edge_count") == 0, "cross-batch edge found")
                if method == "STAMP":
                    safeguard = metadata.get("zero_library_safeguard", {})
                    check(
                        safeguard.get("strategy")
                        == "raw_zero_likelihood_within_batch_sgc_encoder_v1",
                        "STAMP zero-library strategy changed",
                    )
                    check(
                        safeguard.get("raw_fixed_panel_counts_modified") is False,
                        "STAMP modified raw fixed-panel counts",
                    )
                    check(
                        safeguard.get("likelihood_uses_exact_raw_fixed_panel_counts") is True,
                        "STAMP likelihood counts are not exact",
                    )
                method_count += 1
                rows.append(
                    {
                        "artifact_type": "method_candidate",
                        "method": method,
                        "mode": mode,
                        "alpha": alpha,
                        "path": str(candidate.resolve()),
                        "sha256": workflow.sha256_file(metadata_path),
                        "status": "validated",
                    }
                )
    check(method_count == 36, f"validated {method_count} method candidates; expected 36")
    run_manifest_path = run_dir / "method_run_manifest.csv"
    run_manifest = pd.read_csv(run_manifest_path)
    check(len(run_manifest) == 36, "consolidated method manifest must contain 36 rows")
    check(
        not run_manifest[["method", "mode", "alpha"]].duplicated().any(),
        "consolidated method manifest contains duplicate jobs",
    )
    check(
        set(run_manifest["status"]).issubset({"complete", "skipped_verified"}),
        "consolidated method manifest contains an unsuccessful job",
    )

    metric_root = run_dir / "metrics"
    metric_path = metric_root / "atomic_metrics.csv"
    metric_summary_path = metric_root / "summary.json"
    metric_provenance_path = metric_root / "provenance.json"
    metrics = pd.read_csv(metric_path)
    metric_summary = json.loads(metric_summary_path.read_text(encoding="utf-8"))
    metric_provenance = json.loads(metric_provenance_path.read_text(encoding="utf-8"))
    check(len(metrics) == 60, f"metric table has {len(metrics)} rows; expected 60")
    check(metric_summary.get("rows") == 60, "metric summary row count changed")
    check(
        metric_summary.get("composite_score") == "prohibited_not_computed",
        "composite score was not prohibited",
    )
    check(
        metric_summary.get("winner_ranking") == "prohibited_not_computed",
        "winner ranking was not prohibited",
    )
    check(
        metric_provenance.get("diagnostics", {}).get("validated_method_candidates") == 36,
        "metric provenance does not bind 36 candidates",
    )
    check(
        metric_provenance["feast"]["commit"] == expected_feast_commit,
        "metric and simulation FEAST commits differ",
    )

    output_dir.mkdir(parents=True)
    validation_path = output_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(validation_path, index=False)
    summary = {
        "schema_version": 1,
        "configuration_id": f"{config['configuration_id']}__validation",
        "study_id": "04",
        "legacy_study_id": "08",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "simulation_h5ad": 14,
        "nonzero_comparison_conditions": 12,
        "method_candidates": method_count,
        "method_counts": {"GraphST": 12, "STAMP": 12, "scVI": 12},
        "atomic_metric_rows": len(metrics),
        "cross_batch_edges": 0,
        "decision": "validated_fresh_supplementary_candidate_pending_scientific_review",
        "limitations": [
            "GraphST, STAMP, and scVI run in external method environments that do not expand FEAST support.",
            "Alpha values above 1.00 are extrapolation evidence.",
            "GraphST PCA-10 sensitivity must accompany the PCA-20 primary representation.",
            "No composite, ranking, winner, table, or figure claim is authorized by validation alone.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": 1,
        "configuration_id": summary["configuration_id"],
        "study_id": "04",
        "legacy_study_id": "08",
        "generated_utc": summary["generated_utc"],
        "feast": simulation_provenance["feast"],
        "public_seed": int(config["public_seed"]),
        "sources": [
            {"path": str(Path(__file__).resolve()), "sha256": workflow.sha256_file(Path(__file__).resolve())},
            {"path": str(config["_config_path"]), "sha256": workflow.sha256_file(config["_config_path"])},
        ],
        "inputs": [
            {"path": str(manifest_path.resolve()), "sha256": workflow.sha256_file(manifest_path)},
            {"path": str(simulation_provenance_path.resolve()), "sha256": workflow.sha256_file(simulation_provenance_path)},
            {"path": str(panel_path.resolve()), "sha256": workflow.sha256_file(panel_path)},
            {"path": str(panel_provenance_path.resolve()), "sha256": workflow.sha256_file(panel_provenance_path)},
            {"path": str(metric_path.resolve()), "sha256": workflow.sha256_file(metric_path)},
            {"path": str(metric_provenance_path.resolve()), "sha256": workflow.sha256_file(metric_provenance_path)},
            {"path": str(run_manifest_path.resolve()), "sha256": workflow.sha256_file(run_manifest_path)},
        ],
        "outputs": [
            {"path": str(validation_path.resolve()), "sha256": workflow.sha256_file(validation_path)},
            {"path": str(summary_path.resolve()), "sha256": workflow.sha256_file(summary_path)},
        ],
        "diagnostics": {
            "status": "completed",
            "simulation_h5ad": 14,
            "comparison_conditions": 12,
            "method_candidates": 36,
            "metric_rows": 60,
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
