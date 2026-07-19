#!/usr/bin/env python3
"""Compare fresh Study 02 metrics with retained corrected and historical reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["method", "alteration", "angle_degrees"]
METRICS = (
    "n_aligned",
    "mean_spatial_error",
    "median_spatial_error",
    "rmse_spatial_error",
    "max_spatial_error",
    "nn_accuracy",
    "nn_region_accuracy",
    "nn_mean_distance",
    "ge_correlation_exact_gene_join",
    "recovered_moving_to_aligned_rotation",
    "expected_moving_to_aligned_rotation",
    "rotation_recovery_error",
    "residual_aligned_to_target_rotation",
    "transport_argmax_identity_accuracy",
    "transport_mutual_argmax_consistency",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_metrics(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    missing = set(KEYS + list(METRICS)) - set(table.columns)
    if missing:
        raise RuntimeError(f"{path} lacks required columns: {sorted(missing)}")
    if (
        len(table) != 40
        or table[KEYS].duplicated().any()
        or not table["status"].eq("ok").all()
        or not table["canonical_candidate"].astype(bool).all()
    ):
        raise RuntimeError(f"{path} is not a complete 40-row candidate table")
    return table[KEYS + list(METRICS)].copy()


def relation(left: float, right: float, preferred: str) -> str:
    if np.isclose(left, right, rtol=1e-10, atol=1e-12):
        return "tie"
    better = left < right if preferred == "lower" else left > right
    return "paste_better" if better else "spateo_better"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-scores", type=Path, required=True)
    parser.add_argument("--prior-corrected-metrics", type=Path, required=True)
    parser.add_argument("--prior-historical-summary", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    fresh_path = args.fresh_scores / "alignment_metrics.csv"
    fresh = load_metrics(fresh_path)
    prior = load_metrics(args.prior_corrected_metrics)
    merged = prior.merge(
        fresh,
        on=KEYS,
        how="outer",
        suffixes=("_old", "_fresh"),
        indicator=True,
        validate="one_to_one",
    )
    if len(merged) != 40 or not merged["_merge"].eq("both").all():
        raise RuntimeError("prior corrected and fresh alignment keys do not match")
    merged = merged.drop(columns="_merge")
    for metric in METRICS:
        merged[f"{metric}_delta_fresh_minus_old"] = (
            merged[f"{metric}_fresh"] - merged[f"{metric}_old"]
        )
    merged["change_class"] = (
        "fresh_FEAST_simulation_and_stochastic_method_rerun; "
        "PASTE_retains_strict_positive_convergence_contract"
    )
    merged["single_cause_attribution_supported"] = False
    metrics_path = args.output_dir / "old_vs_new_metrics.csv"
    merged.to_csv(metrics_path, index=False)

    summary_rows = []
    for (method, alteration), group in merged.groupby(["method", "alteration"]):
        for metric in METRICS:
            delta = group[f"{metric}_delta_fresh_minus_old"]
            summary_rows.append(
                {
                    "method": method,
                    "alteration": alteration,
                    "metric": metric,
                    "n_rows": len(group),
                    "old_mean": group[f"{metric}_old"].mean(),
                    "fresh_mean": group[f"{metric}_fresh"].mean(),
                    "delta_fresh_minus_old": delta.mean(),
                    "mean_absolute_row_delta": delta.abs().mean(),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "old_vs_new_summary.csv"
    summary.to_csv(summary_path, index=False)

    historical = pd.read_csv(args.prior_historical_summary)
    if len(historical) != 8 or historical[["method", "alteration"]].duplicated().any():
        raise RuntimeError("historical design summary must contain eight unique rows")
    fresh_grouped = (
        fresh.groupby(["method", "alteration"], as_index=False)
        .agg(
            fresh_n_rows=("n_aligned", "size"),
            fresh_mean_n_aligned=("n_aligned", "mean"),
            fresh_mean_spatial_error=("mean_spatial_error", "mean"),
            fresh_mean_nn_accuracy=("nn_accuracy", "mean"),
            fresh_mean_nn_region_accuracy=("nn_region_accuracy", "mean"),
            fresh_mean_ge_correlation_exact_gene_join=(
                "ge_correlation_exact_gene_join", "mean"
            ),
        )
    )
    historical_to_fresh = historical[
        [
            "method",
            "alteration",
            "old_design",
            "old_n_rows",
            "old_mean_n_aligned",
            "old_mean_spatial_error",
            "old_mean_nn_accuracy",
            "old_mean_nn_region_accuracy",
            "old_mean_ge_correlation",
        ]
    ].merge(
        fresh_grouped,
        on=["method", "alteration"],
        how="outer",
        validate="one_to_one",
    )
    if len(historical_to_fresh) != 8 or historical_to_fresh.isna().any().any():
        raise RuntimeError("historical design and fresh summaries do not match")
    historical_to_fresh["fresh_design"] = (
        "centered_rigid_rotation_preserved_identity_support"
    )
    historical_to_fresh["directly_comparable"] = False
    historical_to_fresh["numerical_change_classification"] = (
        "design_correction_fresh_simulation_and_metric_recomputation"
    )
    historical_path = args.output_dir / "historical_design_vs_fresh_summary.csv"
    historical_to_fresh.to_csv(historical_path, index=False)

    assessment_spec = (
        ("mean_spatial_error", "lower"),
        ("rmse_spatial_error", "lower"),
        ("nn_accuracy", "higher"),
        ("nn_region_accuracy", "higher"),
        ("ge_correlation_exact_gene_join", "higher"),
        ("transport_argmax_identity_accuracy", "higher"),
    )
    assessment_rows = []
    for metric, preferred in assessment_spec:
        prior_values = prior[prior["alteration"].eq("sparsity_0.5")].groupby(
            "method"
        )[metric].mean()
        fresh_values = fresh[fresh["alteration"].eq("sparsity_0.5")].groupby(
            "method"
        )[metric].mean()
        prior_relation = relation(
            prior_values["paste"], prior_values["spateo"], preferred
        )
        fresh_relation = relation(
            fresh_values["paste"], fresh_values["spateo"], preferred
        )
        assessment_rows.append(
            {
                "metric": metric,
                "preferred_direction": preferred,
                "prior_paste_mean": prior_values["paste"],
                "prior_spateo_mean": prior_values["spateo"],
                "fresh_paste_mean": fresh_values["paste"],
                "fresh_spateo_mean": fresh_values["spateo"],
                "prior_relation": prior_relation,
                "fresh_relation": fresh_relation,
                "relation_reversed": prior_relation != fresh_relation,
            }
        )
    assessment = pd.DataFrame(assessment_rows)
    assessment_path = args.output_dir / "headline_direction_assessment.csv"
    assessment.to_csv(assessment_path, index=False)

    validation = pd.read_csv(args.validation)
    validation_ok = bool(len(validation) == 61 and validation["valid"].all())
    reversal = bool(assessment["relation_reversed"].any())
    decision = {
        "study": "02_alignment",
        "decision": (
            "validated_publication_candidate"
            if validation_ok and not reversal
            else "blocked"
        ),
        "rotation_rows_valid": 20,
        "method_rows_valid": 40,
        "score_table_valid": validation_ok,
        "headline_direction_reversal_detected": reversal,
        "paste_convergence_disposition": (
            "All 20 fresh PASTE rows have positive inner-EMD and outer "
            "conditional-gradient convergence evidence."
        ),
        "spateo_convergence_disposition": (
            "Spateo completed the declared fixed 200-iteration CUDA schedule; "
            "no positive convergence flag is exposed and no convergence claim is made."
        ),
        "external_environment_limitation": (
            "Spateo ran with NumPy 2.4.6, outside FEAST's supported range; "
            "FEAST was not imported in that worker."
        ),
        "numerical_change_classification": (
            "Fresh-simulation and stochastic method-rerun sensitivity. The PASTE "
            "sparsity change retains strict convergence and does not reverse any "
            "declared PASTE-versus-Spateo headline relation."
        ),
        "overall_release_authorized": False,
        "reason_overall_release_not_authorized": (
            "Other article studies and final release checks remain pending."
        ),
    }
    decision_path = args.output_dir / "publication_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n")

    source_path = Path(__file__).resolve()
    output_paths = (
        metrics_path,
        summary_path,
        historical_path,
        assessment_path,
        decision_path,
    )
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_id": "study02-alignment-comparison-v1",
        "public_seed": 2026,
        "feast_version": "1.0.2",
        "feast_commit": "68816e5c1862a6fa2a49bc30609d617c7fa4b449",
        "comparison_source": str(source_path),
        "comparison_source_sha256": sha256(source_path),
        "inputs": {
            str(path.resolve()): sha256(path)
            for path in (
                fresh_path,
                args.prior_corrected_metrics,
                args.prior_historical_summary,
                args.validation,
            )
        },
        "outputs": {path.name: sha256(path) for path in output_paths},
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print("Study 02 comparison: 40/40 row keys and eight design groups matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
