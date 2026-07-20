#!/usr/bin/env python3
"""Compare fresh atomic Study 06 summaries with one pinned historical table."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from workflow import load_config, read_json, require_sha, sha256_file, write_json


COMPARABLE_METRICS = (
    ("mean_corr", "mean", "mean_corr_mean", "mean_log1p_gene_mean_pearson"),
    ("mean_corr", "median", "mean_corr_median", "median_log1p_gene_mean_pearson"),
    ("var_corr", "mean", "var_corr_mean", "mean_gene_variance_pearson"),
    ("var_corr", "median", "var_corr_median", "median_gene_variance_pearson"),
    ("median_gene_pearson", "mean", "median_gene_pearson_mean", "mean_median_per_gene_spot_pearson"),
    (
        "median_gene_pearson",
        "median",
        "median_gene_pearson_median",
        "median_median_per_gene_spot_pearson",
    ),
)

CONTINUITY_METRICS = (
    ("z_coherence", "mean", "z_coherence_mean", "generated_target_z_coherence_mean"),
    ("z_coherence", "median", "z_coherence_median", "generated_target_z_coherence_median"),
    (
        "real_split_half_z_coherence",
        "mean",
        "real_split_half_z_coherence_mean",
        "real_split_half_z_coherence_mean",
    ),
    (
        "real_split_half_z_coherence",
        "median",
        "real_split_half_z_coherence_median",
        "real_split_half_z_coherence_median",
    ),
    ("normalized_z_coherence", "value", "normalized_z_coherence", "normalized_z_coherence"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Validated Study 06 production root")
    parser.add_argument("--historical-summary", type=Path, required=True)
    return parser.parse_args()


def verify_evaluation(output_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = output_dir / "evaluation"
    provenance = read_json(evaluation / "score_provenance.json")
    if provenance.get("status") != "complete" or provenance.get("configuration_id") != config["configuration_id"]:
        raise ValueError("fresh score provenance is incomplete or belongs to another configuration")
    sources = provenance["sources"]
    require_sha(output_dir / "frozen_config.yaml", sources["config_sha256"], "evaluation config")
    require_sha(output_dir / "plan.json", sources["plan_sha256"], "evaluation plan")
    require_sha(output_dir / "input_manifest.csv", sources["input_manifest_sha256"], "evaluation input manifest")
    require_sha(output_dir / "validation.csv", sources["validation_csv_sha256"], "evaluation validation")
    require_sha(
        output_dir / "validation_summary.json",
        sources["validation_summary_sha256"],
        "evaluation validation summary",
    )
    for filename, expected in provenance["outputs"].items():
        require_sha(evaluation / filename, expected, f"evaluation output {filename}")
    return provenance


def build_comparison(
    historical: pd.DataFrame,
    density_summary: pd.DataFrame,
    continuity_summary: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    expected_gaps = {3, 5, 10}
    for name, frame, gap_column in (
        ("historical", historical, "density_gap"),
        ("fresh density", density_summary, "gap"),
        ("fresh continuity", continuity_summary, "gap"),
    ):
        if set(frame[gap_column].astype(int)) != expected_gaps or len(frame) != 3:
            raise ValueError(f"{name} table must contain exactly gaps 3, 5, and 10")
    records: list[dict[str, Any]] = []
    for gap in sorted(expected_gaps):
        old = historical.loc[historical["density_gap"].astype(int) == gap].iloc[0]
        fresh = density_summary.loc[density_summary["gap"].astype(int) == gap].iloc[0]
        fresh_z = continuity_summary.loc[continuity_summary["gap"].astype(int) == gap].iloc[0]
        if int(old["n_targets"]) != int(fresh["targets"]):
            raise ValueError(f"gap {gap} target count differs between historical and fresh summaries")
        for metric, statistic, old_column, fresh_column in COMPARABLE_METRICS:
            records.append(comparison_record(gap, metric, statistic, old[old_column], fresh[fresh_column]))
        for metric, statistic, old_column, fresh_column in CONTINUITY_METRICS:
            records.append(comparison_record(gap, metric, statistic, old[old_column], fresh_z[fresh_column]))
    historical_iter = int(config["historical_comparison"]["historical_sinkhorn_iter"])
    fresh_iter = int(config["transport"]["sinkhorn_iter"])
    records.append(
        {
            "comparison_kind": "solver_configuration",
            "gap": -1,
            "metric": "sinkhorn_iter",
            "statistic": "declared_value",
            "historical": float(historical_iter),
            "fresh": float(fresh_iter),
            "delta": float(fresh_iter - historical_iter),
            "comparable_formula": False,
            "classification": "declared_solver_configuration_change",
            "publication_decision": "pending_author_review",
        }
    )
    return pd.DataFrame(records)


def comparison_record(gap: int, metric: str, statistic: str, historical: Any, fresh: Any) -> dict[str, Any]:
    old_value = float(historical)
    new_value = float(fresh)
    return {
        "comparison_kind": "atomic_metric",
        "gap": int(gap),
        "metric": metric,
        "statistic": statistic,
        "historical": old_value,
        "fresh": new_value,
        "delta": float(new_value - old_value),
        "comparable_formula": True,
        "classification": "fresh_strict_ot_rerun_vs_historical_noncanonical",
        "publication_decision": "pending_author_review",
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    historical_path = args.historical_summary.resolve()
    config = load_config(output_dir / "frozen_config.yaml")
    verify_evaluation(output_dir, config)
    require_sha(
        historical_path,
        str(config["historical_comparison"]["expected_sha256"]),
        "pinned historical summary",
    )
    final_dir = output_dir / "evaluation" / "old_vs_new"
    work_dir = output_dir / ".work" / "old_vs_new"
    if final_dir.exists() or work_dir.exists():
        raise FileExistsError("fresh-only old-versus-new path already exists")
    work_dir.mkdir(parents=True)
    try:
        historical = pd.read_csv(historical_path)
        density = pd.read_csv(output_dir / "evaluation" / "density_summary.csv")
        continuity = pd.read_csv(output_dir / "evaluation" / "continuity_summary.csv")
        comparison = build_comparison(historical, density, continuity, config)
        comparison_path = work_dir / "old_vs_new.csv"
        comparison.to_csv(comparison_path, index=False)
        write_json(
            work_dir / "comparison_provenance.json",
            {
                "status": "complete_pending_author_review",
                "configuration_id": config["configuration_id"],
                "historical_artifact": {
                    "artifact_id": config["historical_comparison"]["artifact_id"],
                    "path": str(historical_path),
                    "sha256": sha256_file(historical_path),
                    "status": config["historical_comparison"]["status"],
                },
                "fresh_inputs": {
                    "score_provenance_sha256": sha256_file(output_dir / "evaluation" / "score_provenance.json"),
                    "density_summary_sha256": sha256_file(output_dir / "evaluation" / "density_summary.csv"),
                    "continuity_summary_sha256": sha256_file(output_dir / "evaluation" / "continuity_summary.csv"),
                },
                "output": {"old_vs_new.csv": sha256_file(comparison_path)},
                "headline_conclusion_review": "required_before_canonical_or_figure_update",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        restored = read_json(work_dir / "comparison_provenance.json")
        require_sha(comparison_path, restored["output"]["old_vs_new.csv"], "old-versus-new table")
        if len(pd.read_csv(comparison_path)) != len(comparison):
            raise ValueError("old-versus-new table did not round-trip completely")
        os.replace(work_dir, final_dir)
    except Exception as exc:
        write_json(
            work_dir / "FAILED.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
    print(f"Wrote pinned historical comparison to {final_dir}")


if __name__ == "__main__":
    main()
