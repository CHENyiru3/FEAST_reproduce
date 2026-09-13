"""Evaluate a leakage-free label-only linear baseline for Study 05.

The one-hot ordinary least-squares fit is the source mean expression for each
supported conditional label.  It intentionally ignores spatial coordinates.
Only the primary FEAST assignment-randomness setting is compared because the
deterministic baseline has no assignment-randomness parameter.
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

from score import (
    _moran_columns,
    _pearson_columns,
    donor_stratum,
    neighbor_indices,
    panel_summary,
)
from workflow import (
    STUDY_DIR,
    Job,
    declared_jobs,
    direction_id,
    inspect_inputs,
    prepare_job,
)


METRICS = (
    "mean_corr",
    "var_corr",
    "moran_corr",
    "zero_ks",
    "median_gene_pearson",
    "median_gene_spearman",
)


def dense_chunk(matrix: Any, start: int, stop: int) -> np.ndarray:
    values = matrix[:, start:stop]
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float64)


def fit_label_means(matrix: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Fit one-hot OLS coefficients (one mean vector per label)."""
    values = np.asarray(matrix, dtype=np.float64)
    label_strings = np.asarray(labels, dtype=str)
    if values.ndim != 2 or values.shape[0] != label_strings.size:
        raise ValueError("expression rows and labels must align")
    return {
        label: values[label_strings == label].mean(axis=0)
        for label in sorted(set(label_strings.tolist()))
    }


def selected_jobs(config: dict[str, Any]) -> list[Job]:
    primary = float(config["primary_assignment_randomness"])
    return [
        job
        for job in declared_jobs(config)
        if job.mode == "mask_half" or np.isclose(job.assignment_randomness, primary)
    ]


def evaluate_job(
    config: dict[str, Any],
    paths: dict[tuple[str, str], Path],
    panels: dict[str, list[str]],
    job: Job,
    targets: dict[tuple[str, str], ad.AnnData],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prepared = prepare_job(config, paths, panels, job)
    annotation_key = str(config["datasets"][job.dataset]["annotation_key"])
    genes = prepared.genes
    target_full = targets[job.dataset, job.target]
    target = target_full[prepared.target_contract.obs_names.astype(str).tolist(), genes].copy()
    target_labels = target.obs[annotation_key].astype(str).to_numpy()
    source_labels = prepared.reference.obs[annotation_key].astype(str).to_numpy()
    missing = sorted(set(target_labels) - set(source_labels))
    if missing:
        raise ValueError(f"target labels lack source support: {missing}")

    source_matrix = (
        prepared.reference.layers["counts"]
        if "counts" in prepared.reference.layers
        else prepared.reference.X
    )
    target_matrix = target.layers["counts"] if "counts" in target.layers else target.X
    neighbors = neighbor_indices(
        target.obsm["spatial"], int(config["scoring"]["neighbors_including_self"])
    )
    chunk_size = int(config["scoring"]["gene_chunk_size"])
    columns = {
        name: np.full(len(genes), np.nan, dtype=np.float64)
        for name in (
            "pearson",
            "spearman",
            "generated_mean",
            "target_mean",
            "generated_variance",
            "target_variance",
            "generated_zero_prop",
            "target_zero_prop",
            "generated_moran_i",
            "target_moran_i",
        )
    }
    for start in range(0, len(genes), chunk_size):
        stop = min(start + chunk_size, len(genes))
        coefficients = fit_label_means(
            dense_chunk(source_matrix, start, stop), source_labels
        )
        predicted = np.vstack([coefficients[label] for label in target_labels])
        observed = dense_chunk(target_matrix, start, stop)
        columns["pearson"][start:stop] = _pearson_columns(predicted, observed)
        columns["spearman"][start:stop] = _pearson_columns(
            pd.DataFrame(predicted).rank(method="average").to_numpy(),
            pd.DataFrame(observed).rank(method="average").to_numpy(),
        )
        columns["generated_mean"][start:stop] = predicted.mean(axis=0)
        columns["target_mean"][start:stop] = observed.mean(axis=0)
        columns["generated_variance"][start:stop] = predicted.var(axis=0)
        columns["target_variance"][start:stop] = observed.var(axis=0)
        columns["generated_zero_prop"][start:stop] = np.mean(predicted <= 0, axis=0)
        columns["target_zero_prop"][start:stop] = np.mean(observed <= 0, axis=0)
        columns["generated_moran_i"][start:stop] = _moran_columns(predicted, neighbors)
        columns["target_moran_i"][start:stop] = _moran_columns(observed, neighbors)

    per_gene = pd.DataFrame({"gene": genes, **columns})
    zero_genes = set(map(str, prepared.support["reference_zero_genes"]))
    per_gene.insert(0, "reference_observed", ~per_gene["gene"].isin(zero_genes))
    identity = {
        "method": "label_mean_linear",
        "mode": job.mode,
        "dataset": job.dataset,
        "direction": direction_id(job),
        "source": job.source,
        "target": job.target,
        "donor_stratum": donor_stratum(job),
    }
    for name, value in reversed(tuple(identity.items())):
        per_gene.insert(0, name, value)
    observed_summary = {
        f"observed_{name}": value
        for name, value in panel_summary(per_gene[per_gene["reference_observed"]]).items()
    }
    summary = {
        **identity,
        "n_target_spots": int(target.n_obs),
        "target_retained_fraction": float(prepared.support["target_retained_fraction"]),
        "n_genes": len(genes),
        "n_reference_observed_genes": int(
            prepared.support["reference_observed_gene_count"]
        ),
        **panel_summary(per_gene),
        **observed_summary,
    }
    return per_gene, summary


def feast_comparison(summary: pd.DataFrame, feast_path: Path) -> pd.DataFrame:
    feast = pd.read_csv(feast_path)
    feast = feast[feast["primary_setting"].astype(bool)].copy()
    keys = ["mode", "dataset", "direction", "source", "target", "donor_stratum"]
    merged = summary.merge(feast[keys + list(METRICS)], on=keys, suffixes=("_baseline", "_feast"), validate="one_to_one")
    rows = []
    for record in merged.to_dict("records"):
        for metric in METRICS:
            baseline_value = float(record[f"{metric}_baseline"])
            feast_value = float(record[f"{metric}_feast"])
            rows.append(
                {
                    **{key: record[key] for key in keys},
                    "metric": metric,
                    "baseline": baseline_value,
                    "feast": feast_value,
                    "feast_minus_baseline": feast_value - baseline_value,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=STUDY_DIR / "config.yaml")
    parser.add_argument(
        "--manifest", type=Path, default=STUDY_DIR / "data" / "input_checksums.csv"
    )
    parser.add_argument("--input-dir", type=Path, default=STUDY_DIR / "data" / "local")
    parser.add_argument(
        "--feast-summary", type=Path, default=STUDY_DIR / "outputs" / "scores" / "summary.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STUDY_DIR / "outputs" / "baselines" / "label_mean_linear",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"baseline output already exists: {args.output_dir}")
    config, paths, panels = inspect_inputs(args.config, args.manifest, args.input_dir)
    jobs = selected_jobs(config)
    targets = {key: ad.read_h5ad(path) for key, path in paths.items()}
    work_root = STUDY_DIR / ".work"
    work_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="study05-linear-baseline-", dir=work_root))
    try:
        summaries = []
        first = True
        for job in jobs:
            per_gene, summary = evaluate_job(config, paths, panels, job, targets)
            per_gene.to_csv(
                work_dir / "per_gene_metrics.csv",
                mode="w" if first else "a",
                header=first,
                index=False,
            )
            first = False
            summaries.append(summary)
        summary_frame = pd.DataFrame(summaries)
        summary_frame.to_csv(work_dir / "summary.csv", index=False)
        feast_comparison(summary_frame, args.feast_summary).to_csv(
            work_dir / "feast_comparison.csv", index=False
        )
        provenance = {
            "status": "complete",
            "configuration_id": config["configuration_id"],
            "method": "one_hot_ordinary_least_squares_equivalent_to_source_label_means",
            "spatial_coordinates_used": False,
            "target_expression_access": "evaluation_only",
            "compared_feast_setting": float(config["primary_assignment_randomness"]),
            "jobs": len(jobs),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (work_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(work_dir, args.output_dir)
    except Exception:
        raise
    print(f"wrote {len(jobs)} Study 05 baseline comparisons to {args.output_dir}")


if __name__ == "__main__":
    main()
