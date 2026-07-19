#!/usr/bin/env python3
"""Compare fresh Study 00 atomic metrics with the corrected historical table."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_DIRECTIONS = {
    "input_mean_corr": "higher",
    "input_variance_corr": "higher",
    "cosine_divergence": "lower",
    "relative_error_mean": "lower",
    "zero_mask_jaccard": "higher",
    "gene_zero_fraction_wasserstein": "lower",
    "gene_mean_wasserstein": "lower",
    "gene_variance_wasserstein": "lower",
    "library_size_wasserstein": "lower",
    "moran_i_correlation": "higher",
}
CONTRACT_COLUMNS = [
    "identity_flag",
    "n_common_genes",
    "n_paired_spots",
    "spot_pairing",
    "n_moran_genes",
    "moran_panel_sha256",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-metrics", type=Path, required=True)
    parser.add_argument("--historical-sha256", required=True)
    parser.add_argument("--fresh-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.historical_metrics) != args.historical_sha256:
        raise RuntimeError("historical metric checksum mismatch")
    historical = pd.read_csv(args.historical_metrics)
    fresh = pd.read_csv(args.fresh_metrics)
    historical = historical[
        historical["simulator"] != "FEAST_OT_Spatial"
    ].copy()
    historical.loc[
        historical["simulator"] == "FEAST_Rank", "simulator"
    ] = "FEAST_reference_rank"

    keys = ["simulator", "sample"]
    if (
        len(historical) != 60
        or len(fresh) != 60
        or historical[keys].duplicated().any()
        or fresh[keys].duplicated().any()
        or set(map(tuple, historical[keys].to_numpy()))
        != set(map(tuple, fresh[keys].to_numpy()))
    ):
        raise RuntimeError("old/fresh tables do not have the exact 60-row key set")
    required = set(keys + CONTRACT_COLUMNS + list(METRIC_DIRECTIONS))
    if not required.issubset(historical) or not required.issubset(fresh):
        raise RuntimeError("old/fresh tables lack required atomic columns")

    fresh = fresh.copy()
    fresh["_fresh_order"] = np.arange(len(fresh))
    joined = fresh.merge(
        historical,
        on=keys,
        how="inner",
        suffixes=("_fresh", "_old"),
        validate="one_to_one",
    ).sort_values("_fresh_order")
    joined.insert(
        2,
        "change_class",
        np.where(
            joined["simulator"] == "FEAST_reference_rank",
            "fresh_feast_rng_fixed_global_assignment_rerun",
            "same_frozen_external_artifact_metric_recalculation",
        ),
    )
    joined.drop(columns="_fresh_order", inplace=True)
    for metric in METRIC_DIRECTIONS:
        joined[f"{metric}_delta"] = (
            joined[f"{metric}_fresh"] - joined[f"{metric}_old"]
        )

    summary_rows: list[dict] = []
    for metric, direction in METRIC_DIRECTIONS.items():
        old_means = joined.groupby("simulator")[f"{metric}_old"].mean()
        fresh_means = joined.groupby("simulator")[f"{metric}_fresh"].mean()
        ascending = direction == "lower"
        old_ranks = old_means.rank(method="min", ascending=ascending)
        fresh_ranks = fresh_means.rank(method="min", ascending=ascending)
        for simulator in sorted(old_means.index):
            old_rank = old_ranks.loc[simulator]
            fresh_rank = fresh_ranks.loc[simulator]
            rank_unchanged = bool(
                (pd.isna(old_rank) and pd.isna(fresh_rank))
                or old_rank == fresh_rank
            )
            summary_rows.append(
                {
                    "metric": metric,
                    "preferred_direction": direction,
                    "simulator": simulator,
                    "old_mean": old_means.loc[simulator],
                    "fresh_mean": fresh_means.loc[simulator],
                    "mean_delta": fresh_means.loc[simulator]
                    - old_means.loc[simulator],
                    "old_rank": old_rank,
                    "fresh_rank": fresh_rank,
                    "rank_unchanged": rank_unchanged,
                }
            )
    summary = pd.DataFrame(summary_rows)
    headline_unchanged = bool(summary["rank_unchanged"].all())

    feast = summary[summary["simulator"] == "FEAST_reference_rank"].copy()
    improved = (
        ((feast["preferred_direction"] == "higher") & (feast["mean_delta"] > 0))
        | ((feast["preferred_direction"] == "lower") & (feast["mean_delta"] < 0))
    )
    worsened = (
        ((feast["preferred_direction"] == "higher") & (feast["mean_delta"] < 0))
        | ((feast["preferred_direction"] == "lower") & (feast["mean_delta"] > 0))
    )
    identity_changes = joined[
        joined["identity_flag_fresh"] != joined["identity_flag_old"]
    ][
        ["simulator", "sample", "identity_flag_old", "identity_flag_fresh"]
    ].to_dict("records")
    decision = {
        "status": (
            "candidate_headline_direction_unchanged"
            if headline_unchanged
            else "stop_aggregate_atomic_metric_ranking_changed"
        ),
        "criterion": (
            "old versus fresh rank equality for every simulator across all ten "
            "aggregate mean atomic metrics; no composite score"
        ),
        "headline_direction_unchanged": headline_unchanged,
        "publication_claim_authorized": False,
        "fresh_feast_rows": int(
            (joined["simulator"] == "FEAST_reference_rank").sum()
        ),
        "same_frozen_external_rows": int(
            (joined["simulator"] != "FEAST_reference_rank").sum()
        ),
        "feast_aggregate_metrics_improved": feast.loc[improved, "metric"].tolist(),
        "feast_aggregate_metrics_unchanged": feast.loc[
            feast["mean_delta"] == 0, "metric"
        ].tolist(),
        "feast_aggregate_metrics_worsened": feast.loc[worsened, "metric"].tolist(),
        "identity_flag_changes": identity_changes,
        "disposition": (
            "Fresh per-sample values and every signed delta are retained. The "
            "worsened gene-variance Wasserstein result is not averaged away or hidden."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    comparison_path = args.output_dir / "old_vs_new_metrics.csv"
    summary_path = args.output_dir / "aggregate_metric_summary.csv"
    decision_path = args.output_dir / "publication_decision.json"
    joined.to_csv(comparison_path, index=False)
    summary.to_csv(summary_path, index=False)
    decision_path.write_text(json.dumps(decision, indent=2) + "\n")
    source = Path(__file__).resolve()
    provenance = {
        "configuration_id": "study00-fresh-vs-corrected-historical-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "historical_role": "audit_only_noncanonical_comparator",
        "historical_metrics": {
            "path": str(args.historical_metrics.resolve()),
            "sha256": sha256(args.historical_metrics),
        },
        "fresh_metrics": {
            "path": str(args.fresh_metrics.resolve()),
            "sha256": sha256(args.fresh_metrics),
        },
        "source": {"path": str(source), "sha256": sha256(source)},
        "outputs": {
            path.name: sha256(path)
            for path in (comparison_path, summary_path, decision_path)
        },
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(decision["status"])
    return 0 if headline_unchanged else 2


if __name__ == "__main__":
    raise SystemExit(main())
