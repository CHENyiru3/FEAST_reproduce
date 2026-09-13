"""Build an age-and-region-conditioned whole-spot baseline for Study 07.

Study 07 has an expression-free target blueprint, so this is a descriptive
continuity control rather than an accuracy benchmark.  For every blueprint
spot, the baseline samples one complete expression vector with replacement
from pooled references of the same age and region.  Only sufficient summaries
are retained; a second full 7.5-million-spot expression volume is not written.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import workflow
from evaluate import adjacent_metrics, require_validation


CONTINUITY_METRICS = (
    "median_adjacent_gene_mean_pearson",
    "median_adjacent_region_gene_mean_pearson",
    "minimum_adjacent_gene_mean_pearson",
    "minimum_adjacent_region_gene_mean_pearson",
)


def pooled_region_matrices(
    paths: list[Path], label_key: str, included_regions: set[str]
) -> tuple[list[str], dict[str, sparse.csr_matrix], dict[str, int]]:
    """Load pooled reference count rows, separated by conditional region."""
    parts: dict[str, list[sparse.csr_matrix]] = {
        region: [] for region in sorted(included_regions)
    }
    genes: list[str] | None = None
    for path in paths:
        data = ad.read_h5ad(path)
        observed_genes = list(map(str, data.var_names))
        if genes is None:
            genes = observed_genes
        elif genes != observed_genes:
            raise ValueError(f"gene order differs: {path.name}")
        matrix = data.layers["counts"] if "counts" in data.layers else data.X
        if not sparse.issparse(matrix):
            matrix = sparse.csr_matrix(np.asarray(matrix))
        else:
            matrix = matrix.tocsr()
        labels = data.obs[label_key].astype(str).to_numpy()
        for region in sorted(set(labels) & included_regions):
            parts[region].append(matrix[labels == region].copy())
        del data, matrix
    if genes is None:
        raise ValueError("no reference inputs")
    missing = sorted(region for region, blocks in parts.items() if not blocks)
    if missing:
        raise ValueError(f"reference cohort lacks blueprint regions: {missing}")
    pools = {
        region: sparse.vstack(blocks, format="csr")
        for region, blocks in sorted(parts.items())
    }
    support = {region: matrix.shape[0] for region, matrix in pools.items()}
    return genes, pools, support


def bootstrap_region_mean(
    matrix: sparse.csr_matrix, n_spots: int, rng: np.random.Generator
) -> np.ndarray:
    """Mean of ``n_spots`` complete donor rows sampled with replacement."""
    if n_spots <= 0:
        raise ValueError("blueprint region must contain at least one spot")
    donor_indices = rng.integers(0, matrix.shape[0], size=int(n_spots))
    return np.asarray(matrix[donor_indices].mean(axis=0), dtype=np.float64).reshape(-1)


def sampled_state(
    coverage: pd.DataFrame,
    pools: dict[str, sparse.csr_matrix],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Sample target-sized region profiles and combine by blueprint abundance."""
    region_means: dict[str, np.ndarray] = {}
    total_spots = int(coverage["n_spots"].sum())
    gene_mean = np.zeros(next(iter(pools.values())).shape[1], dtype=np.float64)
    for record in coverage.sort_values("region").itertuples(index=False):
        region = str(record.region)
        if region not in pools:
            raise ValueError(f"blueprint region lacks reference support: {region}")
        n_spots = int(record.n_spots)
        mean = bootstrap_region_mean(pools[region], n_spots, rng)
        region_means[region] = mean
        gene_mean += (n_spots / total_spots) * mean
    return {
        "z_index": int(coverage["z_index"].iloc[0]),
        "z_world": float(coverage["z_world"].iloc[0]),
        "gene_mean": gene_mean,
        "region_means": region_means,
    }


def aggregate_age_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for age, group in frame.groupby("age", sort=False):
        row: dict[str, Any] = {
            "age": age,
            "replicates": len(group),
            "n_z_levels": int(group["n_z_levels"].iloc[0]),
            "n_genes": int(group["n_genes"].iloc[0]),
            "n_reference_spots": int(group["n_reference_spots"].iloc[0]),
        }
        for metric in CONTINUITY_METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_sd"] = float(np.nanstd(values, ddof=1))
            row[f"{metric}_p05"] = float(np.nanquantile(values, 0.05))
            row[f"{metric}_p95"] = float(np.nanquantile(values, 0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_frame(aggregate: pd.DataFrame, feast_path: Path) -> pd.DataFrame:
    feast = pd.read_csv(feast_path)
    merged = aggregate.merge(
        feast[["age", *CONTINUITY_METRICS]], on="age", validate="one_to_one"
    )
    rows = []
    for record in merged.to_dict("records"):
        for metric in CONTINUITY_METRICS:
            baseline_mean = float(record[f"{metric}_mean"])
            rows.append(
                {
                    "age": record["age"],
                    "metric": metric,
                    "baseline_mean": baseline_mean,
                    "baseline_sd": float(record[f"{metric}_sd"]),
                    "baseline_p05": float(record[f"{metric}_p05"]),
                    "baseline_p95": float(record[f"{metric}_p95"]),
                    "feast": float(record[metric]),
                    "feast_minus_baseline_mean": float(record[metric]) - baseline_mean,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=workflow.STUDY_ROOT / "config.yaml"
    )
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: FINAL_DIR/baselines/conditional_whole_spot_resampling",
    )
    args = parser.parse_args()
    if args.replicates < 2:
        raise ValueError("at least two replicates are required to estimate uncertainty")
    config = workflow.load_config(args.config)
    decision = require_validation(config)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else config["_final_dir"] / "baselines" / "conditional_whole_spot_resampling"
    )
    if output_dir.exists():
        raise FileExistsError(f"baseline output already exists: {output_dir}")
    coverage = pd.read_csv(config["_final_dir"] / "evaluation" / "region_coverage.csv")
    included_regions = set(workflow.EXPECTED_REGIONS)
    seeds = [int(config["public_seed"]) + index for index in range(args.replicates)]
    rngs = [np.random.default_rng(seed) for seed in seeds]
    work_parent = config["_work_dir"]
    work_parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="study07-resampling-", dir=work_parent))
    adjacent_rows: list[dict[str, Any]] = []
    age_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    try:
        for age in workflow.AGE_ORDER:
            genes, pools, support = pooled_region_matrices(
                workflow.reference_paths(config, age),
                str(config["reference_fit"]["label_key"]),
                included_regions,
            )
            support_rows.extend(
                {
                    "age": age,
                    "region": region,
                    "n_reference_spots": n_spots,
                }
                for region, n_spots in sorted(support.items())
            )
            age_coverage = coverage[coverage["age"] == age]
            previous: list[dict[str, Any] | None] = [None for _ in rngs]
            local_adjacent: list[list[dict[str, Any]]] = [[] for _ in rngs]
            n_z_levels = 0
            for _, frame in age_coverage.groupby("z_index", sort=True):
                n_z_levels += 1
                for replicate, (seed, rng) in enumerate(zip(seeds, rngs, strict=True)):
                    state = sampled_state(frame, pools, rng)
                    if previous[replicate] is not None:
                        adjacent = {
                            "age": age,
                            "replicate": replicate,
                            "seed": seed,
                            **adjacent_metrics(previous[replicate], state),
                        }
                        adjacent_rows.append(adjacent)
                        local_adjacent[replicate].append(adjacent)
                    previous[replicate] = state
            for replicate, seed in enumerate(seeds):
                local = local_adjacent[replicate]
                age_rows.append(
                    {
                        "age": age,
                        "method": "conditional_whole_spot_resampling",
                        "replicate": replicate,
                        "seed": seed,
                        "n_z_levels": n_z_levels,
                        "n_genes": len(genes),
                        "n_reference_spots": sum(support.values()),
                        "median_adjacent_gene_mean_pearson": float(
                            np.nanmedian([row["gene_mean_pearson"] for row in local])
                        ),
                        "median_adjacent_region_gene_mean_pearson": float(
                            np.nanmedian(
                                [row["region_gene_mean_pearson"] for row in local]
                            )
                        ),
                        "minimum_adjacent_gene_mean_pearson": float(
                            np.nanmin([row["gene_mean_pearson"] for row in local])
                        ),
                        "minimum_adjacent_region_gene_mean_pearson": float(
                            np.nanmin(
                                [row["region_gene_mean_pearson"] for row in local]
                            )
                        ),
                    }
                )
            del pools

        replicate_frame = pd.DataFrame(age_rows).sort_values(["age", "replicate"])
        aggregate_frame = aggregate_age_summaries(replicate_frame)
        pd.DataFrame(support_rows).sort_values(["age", "region"]).to_csv(
            work_dir / "reference_support.csv", index=False
        )
        pd.DataFrame(adjacent_rows).sort_values(
            ["age", "lower_z_index", "replicate"]
        ).to_csv(work_dir / "adjacent_z_continuity.csv", index=False)
        replicate_frame.to_csv(work_dir / "age_replicate_summary.csv", index=False)
        aggregate_frame.to_csv(work_dir / "age_aggregate_summary.csv", index=False)
        comparison_frame(
            aggregate_frame, config["_final_dir"] / "evaluation" / "age_summary.csv"
        ).to_csv(work_dir / "feast_comparison.csv", index=False)
        provenance = {
            "status": "complete",
            "configuration_id": config["configuration_id"],
            "method": "same_age_same_region_whole_reference_spot_resampling_with_replacement",
            "replicates": args.replicates,
            "seeds": seeds,
            "target_expression_exists": False,
            "accuracy_claim_authorized": False,
            "interpretation": "descriptive_continuity_control_only",
            "target_geometry_source": "validated_expression_free_blueprint_region_coverage",
            "full_expression_volume_written": False,
            "summary_is_exact_for_sampled_spot_means": True,
            "validation_status": decision["status"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (work_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(work_dir, output_dir)
    except Exception:
        raise
    print(f"wrote Study 07 resampling continuity control to {output_dir}")


if __name__ == "__main__":
    main()
