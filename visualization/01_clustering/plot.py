#!/usr/bin/env python3
"""Render the registered Study 01 fixed-panel baseline comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("SOURCE_DATE_EPOCH", "1784419200")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / "PUBLICATION_MANIFEST.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

METRICS = ("ARI", "NMI", "AMI")
METHOD_ORDER = ("GraphST", "STAGATE_mclust", "Leiden_unsupervised")
METHOD_LABELS = {
    "GraphST": "GraphST",
    "STAGATE_mclust": "STAGATE +\nmclust",
    "Leiden_unsupervised": "Label-free\nLeiden",
}
BASELINE_ORDER = ("real_baseline", "baseline")
BASELINE_LABELS = {
    "real_baseline": "Raw slice",
    "baseline": "FEAST baseline",
}
BASELINE_COLORS = {
    "real_baseline": "#4C78A8",
    "baseline": "#E45756",
}
SLICE_COLORS = ("#59A14F", "#F28E2B", "#76B7B2")
MEAN_COLOR = "#6F4E7C"
FIXED_PDF_TIME = datetime(2026, 7, 19, tzinfo=timezone.utc)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT))


def registered_sources() -> tuple[Path, str, Path, str, dict[str, object]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    study = manifest["studies"]["01_clustering"]
    candidate = study["canonical_candidate"]

    if study["status"] != "validated_complete":
        raise ValueError("Study 01 is not registered as validated_complete.")
    if candidate["headline_scope"] != "three_slice_method_means":
        raise ValueError("Unexpected Study 01 headline scope in PUBLICATION_MANIFEST.json.")
    if candidate["slice_level_uniformity_claimed"] is not False:
        raise ValueError("The registered Study 01 decision must reject slice-level uniformity.")

    metrics_path = REPOSITORY_ROOT / candidate["metrics"]
    decision_path = REPOSITORY_ROOT / candidate["decision"]
    return (
        metrics_path,
        str(candidate["metrics_sha256"]),
        decision_path,
        str(candidate["decision_sha256"]),
        candidate,
    )


def verify_registered_file(path: Path, expected_hash: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Registered input is missing: {path}")
    observed_hash = sha256(path)
    if observed_hash != expected_hash:
        raise ValueError(
            f"Registered SHA-256 mismatch for {path}: "
            f"expected {expected_hash}, observed {observed_hash}"
        )


def load_baselines(
    metrics_path: Path, candidate: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(metrics_path)
    required = {
        "slice_id",
        "simulation_id",
        "alteration_type",
        "method",
        "status",
        *METRICS,
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"Registered metric table is missing columns: {missing}")
    if len(metrics) != 252 or set(metrics["status"]) != {"ok"}:
        raise ValueError("Expected 252 successful rows in the registered Study 01 table.")
    if set(metrics["method"]) != set(METHOD_ORDER):
        raise ValueError("Registered metric table has an unexpected method set.")

    baseline = metrics.loc[
        metrics["simulation_id"].isin(BASELINE_ORDER),
        ["slice_id", "simulation_id", "alteration_type", "method", *METRICS],
    ].copy()
    expected_keys = len(METHOD_ORDER) * 3 * len(BASELINE_ORDER)
    if len(baseline) != expected_keys:
        raise ValueError(f"Expected {expected_keys} baseline rows, found {len(baseline)}.")
    if baseline.duplicated(["method", "slice_id", "simulation_id"]).any():
        raise ValueError("Duplicate method/slice/baseline rows found.")
    if baseline[list(METRICS)].isna().any().any() or not np.isfinite(
        baseline[list(METRICS)].to_numpy(dtype=float)
    ).all():
        raise ValueError("Baseline metrics contain missing or non-finite values.")

    slices = sorted(baseline["slice_id"].astype(str).unique())
    if len(slices) != 3:
        raise ValueError(f"Expected exactly three slices, found {slices}.")

    long_table = baseline.melt(
        id_vars=["method", "slice_id", "simulation_id"],
        value_vars=list(METRICS),
        var_name="metric",
        value_name="score",
    ).rename(columns={"simulation_id": "baseline_source"})
    long_table["slice_id"] = long_table["slice_id"].astype(str)
    long_table["baseline_label"] = long_table["baseline_source"].map(BASELINE_LABELS)
    long_table["_metric_order"] = long_table["metric"].map(dict(zip(METRICS, range(3))))
    long_table["_method_order"] = long_table["method"].map(
        dict(zip(METHOD_ORDER, range(3)))
    )
    long_table["_baseline_order"] = long_table["baseline_source"].map(
        dict(zip(BASELINE_ORDER, range(2)))
    )
    long_table = long_table.sort_values(
        ["_metric_order", "_method_order", "slice_id", "_baseline_order"], kind="stable"
    ).drop(columns=["_metric_order", "_method_order", "_baseline_order"])
    long_table = long_table.reset_index(drop=True)

    means = (
        long_table.groupby(["method", "metric", "baseline_source"], sort=False)["score"]
        .mean()
        .unstack("baseline_source")
        .reset_index()
        .rename(
            columns={
                "baseline": "feast_baseline_mean",
                "real_baseline": "raw_slice_mean",
            }
        )
    )
    means["feast_minus_raw"] = means["feast_baseline_mean"] - means["raw_slice_mean"]
    means["n_slices"] = len(slices)
    means["_metric_order"] = means["metric"].map(dict(zip(METRICS, range(3))))
    means["_method_order"] = means["method"].map(dict(zip(METHOD_ORDER, range(3))))
    means = means.sort_values(["_metric_order", "_method_order"], kind="stable").drop(
        columns=["_metric_order", "_method_order"]
    )
    means = means.reset_index(drop=True)

    gaps = (
        long_table.pivot(
            index=["method", "slice_id", "metric"],
            columns="baseline_source",
            values="score",
        )
        .reset_index()
        .rename(columns={"baseline": "feast_baseline", "real_baseline": "raw_slice"})
    )
    gaps["feast_minus_raw"] = gaps["feast_baseline"] - gaps["raw_slice"]
    gaps["_metric_order"] = gaps["metric"].map(dict(zip(METRICS, range(3))))
    gaps["_method_order"] = gaps["method"].map(dict(zip(METHOD_ORDER, range(3))))
    gaps = gaps.sort_values(
        ["_metric_order", "_method_order", "slice_id"], kind="stable"
    ).drop(columns=["_metric_order", "_method_order"])
    gaps = gaps.reset_index(drop=True)

    mean_gap = float(means["feast_minus_raw"].abs().max())
    slice_gap = float(gaps["feast_minus_raw"].abs().max())
    expected_mean_gap = float(candidate["maximum_absolute_three_slice_method_mean_gap"])
    expected_slice_gap = float(candidate["maximum_absolute_slice_level_gap"])
    if not np.isclose(mean_gap, expected_mean_gap, rtol=0.0, atol=1e-12):
        raise ValueError(f"Method-mean gap {mean_gap} does not match {expected_mean_gap}.")
    if not np.isclose(slice_gap, expected_slice_gap, rtol=0.0, atol=1e-12):
        raise ValueError(f"Slice-level gap {slice_gap} does not match {expected_slice_gap}.")
    return long_table, means, gaps


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "feast-study01-publication",
        }
    )


def draw_figure(
    means: pd.DataFrame, gaps: pd.DataFrame, output_dir: Path, dpi: int
) -> list[Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.4), sharex="col")
    x = np.arange(len(METHOD_ORDER), dtype=float)
    offsets = {"real_baseline": -0.10, "baseline": 0.10}
    slices = sorted(gaps["slice_id"].unique())
    slice_offsets = np.linspace(-0.12, 0.12, len(slices))

    for column, metric in enumerate(METRICS):
        top = axes[0, column]
        bottom = axes[1, column]
        metric_means = means.loc[means["metric"] == metric].set_index("method")
        metric_gaps = gaps.loc[gaps["metric"] == metric]

        for method_index, method in enumerate(METHOD_ORDER):
            raw = float(metric_means.loc[method, "raw_slice_mean"])
            feast = float(metric_means.loc[method, "feast_baseline_mean"])
            top.plot(
                [method_index + offsets["real_baseline"], method_index + offsets["baseline"]],
                [raw, feast],
                color="#B8B8B8",
                linewidth=1.0,
                zorder=1,
            )
            top.scatter(
                method_index + offsets["real_baseline"],
                raw,
                color=BASELINE_COLORS["real_baseline"],
                edgecolor="white",
                linewidth=0.5,
                s=42,
                zorder=2,
            )
            top.scatter(
                method_index + offsets["baseline"],
                feast,
                color=BASELINE_COLORS["baseline"],
                edgecolor="white",
                linewidth=0.5,
                s=42,
                zorder=2,
            )

            method_gaps = metric_gaps.loc[metric_gaps["method"] == method].set_index(
                "slice_id"
            )
            for slice_index, slice_id in enumerate(slices):
                bottom.scatter(
                    method_index + slice_offsets[slice_index],
                    float(method_gaps.loc[slice_id, "feast_minus_raw"]),
                    color=SLICE_COLORS[slice_index],
                    edgecolor="white",
                    linewidth=0.4,
                    s=28,
                    zorder=2,
                )
            bottom.scatter(
                method_index,
                float(metric_means.loc[method, "feast_minus_raw"]),
                marker="D",
                color=MEAN_COLOR,
                edgecolor="white",
                linewidth=0.5,
                s=48,
                zorder=3,
            )

        top.set_title(metric, fontweight="bold", pad=7)
        top.set_ylim(0.15, 0.70)
        top.grid(axis="y", color="#E6E6E6", linewidth=0.7)
        top.tick_params(axis="x", length=0, labelbottom=False)
        bottom.axhline(0.0, color="#666666", linewidth=0.8, zorder=0)
        bottom.set_ylim(-0.09, 0.09)
        bottom.grid(axis="y", color="#E6E6E6", linewidth=0.7)
        bottom.set_xticks(x, [METHOD_LABELS[method] for method in METHOD_ORDER])
        bottom.tick_params(axis="x", length=0)

    axes[0, 0].set_ylabel("Baseline clustering score\n(three-slice mean)")
    axes[1, 0].set_ylabel("FEAST baseline − raw slice")
    axes[0, 0].text(-0.26, 1.10, "A", transform=axes[0, 0].transAxes, fontweight="bold", size=12)
    axes[1, 0].text(-0.26, 1.10, "B", transform=axes[1, 0].transAxes, fontweight="bold", size=12)

    baseline_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=BASELINE_COLORS[source],
            markeredgecolor="white",
            markersize=7,
            label=BASELINE_LABELS[source],
        )
        for source in BASELINE_ORDER
    ]
    slice_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=SLICE_COLORS[index],
            markeredgecolor="white",
            markersize=6,
            label=f"Slice {slice_id}",
        )
        for index, slice_id in enumerate(slices)
    ]
    slice_handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor=MEAN_COLOR,
            markeredgecolor="white",
            markersize=6,
            label="Three-slice mean",
        )
    )
    fig.legend(
        handles=baseline_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.995),
        ncol=2,
        frameon=False,
    )
    fig.legend(
        handles=slice_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.015),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.075,
        "Method means summarize n = 3 slices; slice-level points disclose heterogeneity "
        "and are not an equivalence claim.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.20, hspace=0.34, wspace=0.24)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "baseline_clustering_comparison"
    paths = [stem.with_suffix(suffix) for suffix in (".pdf", ".png", ".svg")]
    fig.savefig(
        paths[0],
        metadata={
            "Title": "Study 01 fixed-panel baseline clustering comparison",
            "Creator": "FEAST Study 01 plot.py",
            "CreationDate": FIXED_PDF_TIME,
            "ModDate": FIXED_PDF_TIME,
        },
    )
    fig.savefig(
        paths[1],
        dpi=dpi,
        metadata={"Software": "FEAST Study 01 plot.py"},
    )
    fig.savefig(
        paths[2],
        metadata={
            "Title": "Study 01 fixed-panel baseline clustering comparison",
            "Creator": "FEAST Study 01 plot.py",
            "Date": "2026-07-19",
        },
    )
    plt.close(fig)
    return paths


def write_provenance(
    output_dir: Path,
    outputs: list[Path],
    metrics_path: Path,
    metrics_hash: str,
    decision_path: Path,
    decision_hash: str,
    means: pd.DataFrame,
    gaps: pd.DataFrame,
) -> Path:
    payload = {
        "schema_version": 1,
        "figure_set": "study01_fixed_panel_baseline_comparison",
        "source": {
            "publication_manifest": repository_path(MANIFEST_PATH),
            "publication_manifest_sha256": sha256(MANIFEST_PATH),
            "metrics_csv": repository_path(metrics_path),
            "metrics_sha256": metrics_hash,
            "publication_decision": repository_path(decision_path),
            "publication_decision_sha256": decision_hash,
            "plot_script": repository_path(Path(__file__)),
            "plot_script_sha256": sha256(Path(__file__)),
        },
        "design": {
            "headline_scope": "three_slice_method_means",
            "slice_level_uniformity_claimed": False,
            "winner_claimed": False,
            "methods": list(METHOD_ORDER),
            "metrics": list(METRICS),
            "slice_ids": sorted(gaps["slice_id"].unique().tolist()),
            "n_slices": 3,
            "maximum_absolute_three_slice_method_mean_gap": float(
                means["feast_minus_raw"].abs().max()
            ),
            "maximum_absolute_slice_level_gap": float(gaps["feast_minus_raw"].abs().max()),
        },
        "scientific_disposition": {
            "figure_promotion_authorized": False,
            "aggregate_method_ranking_authorized": False,
            "winner_claim_authorized": False,
            "slice_level_equivalence_authorized": False,
            "supported_interpretation": (
                "FEAST-baseline and raw-slice clustering scores are closely matched only "
                "as method means across the three registered slices."
            ),
        },
        "render_environment": {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "outputs": {path.name: sha256(path) for path in sorted(outputs)},
    }
    provenance_path = output_dir / "figure_provenance.json"
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n")
    return provenance_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_path, metrics_hash, decision_path, decision_hash, candidate = registered_sources()
    verify_registered_file(metrics_path, metrics_hash)
    verify_registered_file(decision_path, decision_hash)
    decision = json.loads(decision_path.read_text())
    if decision.get("decision") != "validated_publication_candidate":
        raise ValueError("The registered Study 01 decision is not a publication candidate.")

    long_table, means, gaps = load_baselines(metrics_path, candidate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.output_dir / "baseline_slice_values.csv"
    means_path = args.output_dir / "baseline_method_means.csv"
    gaps_path = args.output_dir / "baseline_slice_gaps.csv"
    long_table.to_csv(long_path, index=False, float_format="%.12f")
    means.to_csv(means_path, index=False, float_format="%.12f")
    gaps.to_csv(gaps_path, index=False, float_format="%.12f")

    outputs = [long_path, means_path, gaps_path]
    outputs.extend(draw_figure(means, gaps, args.output_dir, args.dpi))
    provenance_path = write_provenance(
        args.output_dir,
        outputs,
        metrics_path,
        metrics_hash,
        decision_path,
        decision_hash,
        means,
        gaps,
    )
    for path in [*outputs, provenance_path]:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
