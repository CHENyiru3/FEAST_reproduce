#!/usr/bin/env python3
"""Compare the fresh Study 01 benchmark with both retained prior tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


METRICS = (
    "ARI",
    "NMI",
    "AMI",
    "Homogeneity",
    "Completeness",
    "V_measure",
    "CHAOS",
    "PAS",
)
KEYS = ["slice_id", "simulation_id", "method"]
METHOD_MAP = {"Leiden": "Leiden_unsupervised"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(
        path,
        dtype={"slice_id": str, "simulation_id": str, "method": str},
    )
    table["method"] = table["method"].replace(METHOD_MAP)
    missing = set(KEYS + list(METRICS)) - set(table.columns)
    if missing:
        raise RuntimeError(f"{path} lacks required columns: {sorted(missing)}")
    if table[KEYS].duplicated().any():
        raise RuntimeError(f"{path} contains duplicate benchmark keys")
    return table[KEYS + ["alteration_type", *METRICS]].copy()


def comparison(
    old: pd.DataFrame,
    fresh: pd.DataFrame,
    name: str,
    expected_rows: int,
) -> pd.DataFrame:
    merged = old.merge(
        fresh,
        on=KEYS,
        how="inner",
        suffixes=("_old", "_fresh"),
        validate="one_to_one",
    )
    if len(merged) != expected_rows:
        raise RuntimeError(
            f"{name} matched {len(merged)} rows; expected {expected_rows}"
        )
    merged.insert(0, "comparison", name)
    merged["alteration_type"] = merged.pop("alteration_type_fresh")
    merged = merged.drop(columns="alteration_type_old")
    for metric in METRICS:
        merged[f"{metric}_delta_fresh_minus_old"] = (
            merged[f"{metric}_fresh"] - merged[f"{metric}_old"]
        )
    merged["change_class"] = merged["method"].map(
        {
            "GraphST": "mixed_workflow_change_fixed_panel_fresh_simulation_and_method_rerun",
            "STAGATE_mclust": "mixed_workflow_change_fixed_panel_fresh_simulation_and_method_rerun",
            "Leiden_unsupervised": "mixed_workflow_change_fixed_panel_fresh_simulation_and_label_free_selection",
        }
    )
    merged["single_cause_attribution_supported"] = False
    return merged


def summarize(comparisons: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (name, method), group in comparisons.groupby(["comparison", "method"]):
        for metric in METRICS:
            delta = group[f"{metric}_delta_fresh_minus_old"]
            rows.append(
                {
                    "comparison": name,
                    "method": method,
                    "metric": metric,
                    "n_rows": len(group),
                    "old_mean": group[f"{metric}_old"].mean(),
                    "fresh_mean": group[f"{metric}_fresh"].mean(),
                    "delta_fresh_minus_old": delta.mean(),
                    "mean_absolute_row_delta": delta.abs().mean(),
                }
            )
    return pd.DataFrame(rows)


def baseline_gaps(prior: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, table in (("prior_fixed_panel", prior), ("fresh", fresh)):
        for method, group in table.groupby("method"):
            feast = group[group["simulation_id"].eq("baseline")]
            real = group[group["simulation_id"].eq("real_baseline")]
            if len(feast) != 3 or len(real) != 3:
                raise RuntimeError(f"{source}/{method} lacks three paired baselines")
            rows.append(
                {
                    "source": source,
                    "method": method,
                    **{
                        f"{metric}_feast_minus_real": (
                            feast[metric].mean() - real[metric].mean()
                        )
                        for metric in ("ARI", "NMI", "AMI")
                    },
                }
            )
    return pd.DataFrame(rows)


def baseline_gaps_by_slice(fresh: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, slice_id), group in fresh.groupby(["method", "slice_id"]):
        feast = group[group["simulation_id"].eq("baseline")]
        real = group[group["simulation_id"].eq("real_baseline")]
        if len(feast) != 1 or len(real) != 1:
            raise RuntimeError(f"fresh {method}/{slice_id} lacks paired baselines")
        rows.append(
            {
                "method": method,
                "slice_id": slice_id,
                **{
                    f"{metric}_feast_minus_real": (
                        float(feast.iloc[0][metric]) - float(real.iloc[0][metric])
                    )
                    for metric in ("ARI", "NMI", "AMI")
                },
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-report", type=Path, required=True)
    parser.add_argument("--prior-fixed-report", type=Path, required=True)
    parser.add_argument("--historical-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    fresh_path = args.fresh_report / "fixed_panel_benchmark_metrics.csv"
    prior_path = args.prior_fixed_report / "fixed_panel_benchmark_metrics.csv"
    fresh = load_table(fresh_path)
    prior = load_table(prior_path)
    historical = load_table(args.historical_table)
    if len(fresh) != 252 or len(prior) != 252 or len(historical) != 243:
        raise RuntimeError("unexpected source benchmark row count")

    historical_comparison = comparison(
        historical,
        fresh,
        "historical_independent_hvg_to_fresh_fixed_panel",
        243,
    )
    prior_comparison = comparison(
        prior,
        fresh,
        "prior_fixed_panel_to_fresh_fixed_panel",
        252,
    )
    combined = pd.concat(
        [historical_comparison, prior_comparison], ignore_index=True
    )
    combined_path = args.output_dir / "old_vs_new_metrics.csv"
    combined.to_csv(combined_path, index=False)

    summary = summarize(combined)
    summary_path = args.output_dir / "old_vs_new_summary.csv"
    summary.to_csv(summary_path, index=False)

    gaps = baseline_gaps(prior, fresh)
    gaps_path = args.output_dir / "baseline_gap_comparison.csv"
    gaps.to_csv(gaps_path, index=False)
    slice_gaps = baseline_gaps_by_slice(fresh)
    slice_gaps_path = args.output_dir / "baseline_gap_by_slice.csv"
    slice_gaps.to_csv(slice_gaps_path, index=False)
    fresh_gaps = gaps[gaps["source"].eq("fresh")]
    max_gap = float(
        fresh_gaps[
            [
                "ARI_feast_minus_real",
                "NMI_feast_minus_real",
                "AMI_feast_minus_real",
            ]
        ]
        .abs()
        .to_numpy()
        .max()
    )
    max_slice_gap = float(
        slice_gaps[
            [
                "ARI_feast_minus_real",
                "NMI_feast_minus_real",
                "AMI_feast_minus_real",
            ]
        ]
        .abs()
        .to_numpy()
        .max()
    )

    decision = {
        "study": "01_clustering",
        "decision": "validated_publication_candidate",
        "headline_direction": "unchanged",
        "headline_conclusion": (
            "Averaged across three slices under identical raw-derived fixed gene "
            "panels, FEAST baseline and raw-slice clustering performance remain "
            "closely matched; this is not a uniform per-slice equivalence claim."
        ),
        "maximum_absolute_three_slice_method_mean_baseline_gap_across_ari_nmi_ami": max_gap,
        "maximum_absolute_slice_level_baseline_gap_across_ari_nmi_ami": max_slice_gap,
        "leiden_disposition": (
            "Retain only the declared label-free modularity-selection rerun; "
            "the label-informed historical Leiden table is sensitivity evidence."
        ),
        "numerical_change_classification": (
            "Workflow-level mixed changes. Fixed panels, freshly regenerated FEAST "
            "simulations, and stochastic method reruns changed together; Leiden also "
            "changed to label-free resolution selection. Row deltas do not support "
            "single-cause attribution."
        ),
        "external_environment_limitation": (
            "GraphST and STAGATE ran in external method environments whose NumPy "
            "versions are unsupported by FEAST; FEAST was not imported there."
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
        combined_path,
        summary_path,
        gaps_path,
        slice_gaps_path,
        decision_path,
    )
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_id": "study01-reference-rank-fixed-panel-v1",
        "public_seed": 2026,
        "feast_version": "1.0.2",
        "feast_commit": "68816e5c1862a6fa2a49bc30609d617c7fa4b449",
        "comparison_source": str(source_path),
        "comparison_source_sha256": sha256(source_path),
        "inputs": {
            str(path.resolve()): sha256(path)
            for path in (fresh_path, prior_path, args.historical_table)
        },
        "outputs": {path.name: sha256(path) for path in output_paths},
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print("Study 01 comparison: 243 historical and 252 prior-fixed rows matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
