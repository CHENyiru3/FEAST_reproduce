#!/usr/bin/env python3
"""Compare only the ten historically comparable Study 05 cross-slice rows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from score import panel_summary
from workflow import STUDY_DIR, load_config, sha256_file, sha256_lines


AR_VALUES = (0.0, 0.1, 0.2, 0.3, 0.5)
HISTORICAL_DIRECTIONS = {"151675_to_151676", "151676_to_151675"}
IDENTITY = ["mode", "dataset", "direction", "assignment_randomness"]
METRICS = (
    "mean_corr",
    "var_corr",
    "moran_corr",
    "zero_ks",
    "median_gene_pearson",
    "median_gene_spearman",
)
HISTORICAL_SUMMARY_SHA256 = {
    0.0: "8b0932fdf3b1d4b43c5b6d3802732d22f9384892f40c6e593b1350e5a3cc8660",
    0.1: "7d8e68250728c1bfda7a19add867e268b1f8a35b78e6ff2adee6b4fd3f963bae",
    0.2: "185f18e3455468fcc36799f6740da161f31e92c22b4fa00d8b273b2b3c215172",
    0.3: "cf54142c6a3afc58ffa344da1a22e260738edcbf4c4816564cfbf9bdf60e566d",
    0.5: "05cb9dadc232b3ef80ca7e1fd3470263801db8656162ad2c8811a82940870915",
}
EXPECTED_HISTORICAL_SUMMARY_MANIFEST_SHA256 = (
    "eaeaef9ccea2c790a83c7c33a023e0a29ab1c889347e8691d5aee9dbb72c57f1"
)
EXPECTED_HISTORICAL_PER_GENE_MANIFEST_SHA256 = (
    "28e0d573d841cd396a8e3f15b40700b82f093e84e248cd64619a209585c8a615"
)


def load_historical(
    root: Path,
    fresh_genes: list[str],
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    frames = []
    summary_hashes: dict[str, str] = {}
    per_gene_hashes: dict[str, str] = {}
    for randomness in AR_VALUES:
        path = root / f"ar_{randomness:.1f}" / "summary.csv"
        relative_summary = f"ar_{randomness:.1f}/summary.csv"
        summary_hash = sha256_file(path)
        if summary_hash != HISTORICAL_SUMMARY_SHA256[randomness]:
            raise RuntimeError(f"historical summary hash mismatch: {relative_summary}")
        summary_hashes[relative_summary] = summary_hash
        frame = pd.read_csv(path)
        frame["assignment_randomness"] = randomness
        frame["mode"] = "cross_slice"
        frame["dataset"] = "dlpfc"
        for direction in sorted(HISTORICAL_DIRECTIONS):
            per_gene_path = (
                root
                / f"ar_{randomness:.1f}"
                / direction
                / "per_gene_metrics.csv"
            )
            relative_per_gene = (
                f"ar_{randomness:.1f}/{direction}/per_gene_metrics.csv"
            )
            per_gene_hashes[relative_per_gene] = sha256_file(per_gene_path)
            per_gene = pd.read_csv(per_gene_path, dtype={"gene": str})
            if not (
                len(per_gene) == 17_924
                and per_gene["gene"].is_unique
                and set(fresh_genes).issubset(set(per_gene["gene"]))
            ):
                raise RuntimeError(
                    f"historical per-gene support mismatch: {relative_per_gene}"
                )
            common = per_gene.set_index("gene").loc[fresh_genes].reset_index()
            common_summary = panel_summary(common)
            row_mask = frame["direction"].astype(str) == direction
            if int(row_mask.sum()) != 1:
                raise RuntimeError(f"historical direction mismatch: {relative_summary}")
            for metric, value in common_summary.items():
                frame.loc[row_mask, f"{metric}_common_panel"] = value
        frames.append(frame)
    summary_manifest_hash = sha256_lines(
        f"{path},{digest}" for path, digest in summary_hashes.items()
    )
    per_gene_manifest_hash = sha256_lines(
        f"{path},{digest}" for path, digest in per_gene_hashes.items()
    )
    if summary_manifest_hash != EXPECTED_HISTORICAL_SUMMARY_MANIFEST_SHA256:
        raise RuntimeError("historical summary manifest hash mismatch")
    if per_gene_manifest_hash != EXPECTED_HISTORICAL_PER_GENE_MANIFEST_SHA256:
        raise RuntimeError("historical per-gene manifest hash mismatch")
    historical = pd.concat(frames, ignore_index=True)
    if not (
        len(historical) == 10
        and not historical[IDENTITY].duplicated().any()
        and set(historical["direction"]) == HISTORICAL_DIRECTIONS
    ):
        raise RuntimeError("historical summaries are not the exact comparable ten-row matrix")
    return historical, summary_hashes, per_gene_hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--fresh-score-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=STUDY_DIR / "config.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance_path = args.output.with_name(f"{args.output.stem}_provenance.json")
    if args.output.exists() or provenance_path.exists():
        raise FileExistsError(args.output if args.output.exists() else provenance_path)
    config = load_config(args.config)
    fresh_summary = args.fresh_score_dir / "summary.csv"
    provenance = json.loads(
        (args.fresh_score_dir / "provenance.json").read_text(encoding="utf-8")
    )
    if not (
        provenance.get("status") == "ok"
        and provenance.get("configuration_id")
        == config["scoring"]["configuration_id"]
        and provenance.get("config_sha256") == sha256_file(args.config)
        and provenance.get("outputs", {}).get("summary.csv")
        == sha256_file(fresh_summary)
    ):
        raise RuntimeError("fresh score provenance/hash validation failed")
    all_new = pd.read_csv(fresh_summary)
    if len(all_new) != int(config["expected_jobs"]) or not all_new["job_id"].is_unique:
        raise RuntimeError("fresh summary is not the exact declared 45-row matrix")
    new = all_new[
        (all_new["mode"] == "cross_slice")
        & (all_new["dataset"] == "dlpfc")
        & all_new["direction"].isin(HISTORICAL_DIRECTIONS)
    ].copy()
    if len(new) != 10 or new[IDENTITY].duplicated().any():
        raise RuntimeError("fresh historical-overlap subset is not exactly ten rows")
    fresh_per_gene_path = args.fresh_score_dir / "per_gene_metrics.csv"
    panel_job_id = str(new.sort_values(IDENTITY).iloc[0]["job_id"])
    fresh_panel_rows = pd.read_csv(
        fresh_per_gene_path,
        usecols=["job_id", "gene"],
        dtype={"job_id": str, "gene": str},
    )
    fresh_genes = fresh_panel_rows.loc[
        fresh_panel_rows["job_id"] == panel_job_id, "gene"
    ].tolist()
    expected_panel = config["datasets"]["dlpfc"]["gene_panel"]
    if not (
        len(fresh_genes) == int(expected_panel["expected_genes"])
        and len(set(fresh_genes)) == len(fresh_genes)
        and sha256_lines(fresh_genes) == str(expected_panel["sha256"])
    ):
        raise RuntimeError("fresh DLPFC panel support/hash mismatch")
    old, historical_summary_hashes, historical_per_gene_hashes = load_historical(
        args.historical_root,
        fresh_genes,
    )
    if not set(METRICS).issubset(old.columns) or not set(METRICS).issubset(new.columns):
        raise RuntimeError("old or new summary lacks a declared historical metric")
    merged = old.merge(
        new,
        on=IDENTITY,
        suffixes=("_historical_full_panel", "_fresh"),
        validate="one_to_one",
    )
    rows = []
    for row in merged.itertuples(index=False):
        values = row._asdict()
        for metric in METRICS:
            historical_full = float(values[f"{metric}_historical_full_panel"])
            historical_common = float(values[f"{metric}_common_panel"])
            fresh = float(values[f"{metric}_fresh"])
            rows.append(
                {
                    "mode": values["mode"],
                    "dataset": values["dataset"],
                    "direction": values["direction"],
                    "assignment_randomness": values["assignment_randomness"],
                    "metric": metric,
                    "historical_full_panel": historical_full,
                    "historical_common_panel": historical_common,
                    "fresh": fresh,
                    "panel_scope_delta": historical_common - historical_full,
                    "method_delta_common_panel": fresh - historical_common,
                    "total_delta_vs_historical_full_panel": fresh - historical_full,
                    "classification": (
                        "panel_scope_effect_separated_then_fresh_unified_ot_"
                        "vs_historical_common_panel"
                    ),
                    "historical_overlap_jobs": 10,
                    "fresh_only_jobs_not_numerically_compared": 35,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.output, index=False)
    provenance = {
        "status": "ok",
        "study_configuration_id": config["configuration_id"],
        "classification": (
            "panel_scope_effect_separated_then_fresh_unified_ot_"
            "vs_historical_common_panel"
        ),
        "historical_scope": {
            "study": "frozen_restored_legacy_04_2d_conditional_transfer",
            "directions": sorted(HISTORICAL_DIRECTIONS),
            "assignment_randomness": list(AR_VALUES),
            "full_panel_genes": 17_924,
            "common_panel_genes": len(fresh_genes),
            "removed_for_common_panel": 17_924 - len(fresh_genes),
            "common_panel_sha256": sha256_lines(fresh_genes),
            "summary_sha256": historical_summary_hashes,
            "summary_manifest_sha256": (
                EXPECTED_HISTORICAL_SUMMARY_MANIFEST_SHA256
            ),
            "per_gene_sha256": historical_per_gene_hashes,
            "per_gene_manifest_sha256": (
                EXPECTED_HISTORICAL_PER_GENE_MANIFEST_SHA256
            ),
        },
        "fresh_scope": {
            "summary_sha256": sha256_file(fresh_summary),
            "per_gene_metrics_sha256": sha256_file(fresh_per_gene_path),
            "score_provenance_sha256": sha256_file(
                args.fresh_score_dir / "provenance.json"
            ),
            "overlap_jobs": 10,
            "fresh_only_jobs_not_numerically_compared": 35,
        },
        "rows": len(comparison),
        "sources": {"comparison_script_sha256": sha256_file(Path(__file__))},
        "outputs": {args.output.name: sha256_file(args.output)},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(rows)} old-versus-new metric rows; "
        "35 expanded-scope jobs have no historical numerical counterpart; "
        f"provenance: {provenance_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
