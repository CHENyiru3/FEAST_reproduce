#!/usr/bin/env python3
"""Build the validated Study 05 publication figure and plotting data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VIS_DIR = Path(__file__).resolve().parent
REPRO_ROOT = VIS_DIR.parents[1]
STUDY_DIR = REPRO_ROOT / "05_2d_conditional_transfer"

COLORS = {
    "DLPFC cross-slice, within donor": "#276FBF",
    "DLPFC cross-slice, cross donor": "#6A4C93",
    "DLPFC half-slice": "#E07A5F",
    "MERFISH cross-slice": "#2A9D8F",
    "MERFISH half-slice": "#E9A23B",
}
MARKERS = {
    "DLPFC cross-slice, within donor": "o",
    "DLPFC cross-slice, cross donor": "s",
    "DLPFC half-slice": "^",
    "MERFISH cross-slice": "D",
    "MERFISH half-slice": "v",
}
GROUP_ORDER = list(COLORS)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_label(row: pd.Series) -> str:
    if row["dataset"] == "dlpfc" and row["mode"] == "cross_slice":
        suffix = "within donor" if row["donor_stratum"] == "within_donor" else "cross donor"
        return f"DLPFC cross-slice, {suffix}"
    if row["dataset"] == "dlpfc":
        return "DLPFC half-slice"
    if row["mode"] == "cross_slice":
        return "MERFISH cross-slice"
    return "MERFISH half-slice"


def short_direction(row: pd.Series) -> str:
    direction = str(row["direction"])
    if row["dataset"] == "merfish":
        direction = direction.replace("Zhuang-ABCA-1.", "")
    return direction.replace("_low_x_to_high_x", " half").replace("_to_", "→")


def verify_inputs() -> tuple[pd.DataFrame, dict[str, Path], dict]:
    paths = {
        "summary": STUDY_DIR / "outputs" / "scores" / "summary.csv",
        "score_provenance": STUDY_DIR / "outputs" / "scores" / "provenance.json",
        "validation": STUDY_DIR / "outputs" / "study05_complete_validation.json",
        "comparison": STUDY_DIR / "outputs" / "old_vs_new_metrics.csv",
        "comparison_provenance": (
            STUDY_DIR / "outputs" / "old_vs_new_metrics_provenance.json"
        ),
        "decision": STUDY_DIR / "PUBLICATION_DECISION.md",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Study 05 evidence: {missing}")
    score_provenance = json.loads(paths["score_provenance"].read_text(encoding="utf-8"))
    validation = json.loads(paths["validation"].read_text(encoding="utf-8"))
    comparison_provenance = json.loads(
        paths["comparison_provenance"].read_text(encoding="utf-8")
    )
    if not (
        score_provenance.get("status") == "ok"
        and score_provenance.get("outputs", {}).get("summary.csv")
        == sha256_file(paths["summary"])
        and validation.get("status") == "ok"
        and int(validation.get("validated_jobs", -1)) == 45
        and validation.get("scores", {}).get("valid") is True
        and comparison_provenance.get("status") == "ok"
        and comparison_provenance.get("outputs", {}).get("old_vs_new_metrics.csv")
        == sha256_file(paths["comparison"])
    ):
        raise RuntimeError("Study 05 evidence hashes or validation status do not agree")
    summary = pd.read_csv(paths["summary"], dtype={"source": str, "target": str})
    if not (
        len(summary) == 45
        and summary["job_id"].is_unique
        and int(summary["primary_setting"].sum()) == 13
        and set(summary["donor_stratum"])
        == {"within_donor", "cross_donor", "within_slice", "donor_not_declared"}
    ):
        raise RuntimeError("Study 05 summary is not the declared 45-row stratified table")
    return summary, paths, validation


def prepare_plotting_data(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = summary[summary["primary_setting"]].copy()
    primary["group"] = primary.apply(group_label, axis=1)
    primary["direction_label"] = primary.apply(short_direction, axis=1)
    primary["zero_fidelity"] = 1.0 - primary["zero_ks"]
    primary["reference_observed_fraction"] = (
        primary["n_reference_observed_genes"] / primary["n_genes"]
    )
    primary["evaluation_scope"] = np.where(
        primary["target_retained_fraction"] < 0.999999,
        "filtered_supported_target",
        "complete_target",
    )

    sensitivity = summary[summary["mode"] == "cross_slice"].copy()
    sensitivity["group"] = sensitivity.apply(group_label, axis=1)
    sensitivity["direction_label"] = sensitivity.apply(short_direction, axis=1)
    ar_summary = (
        sensitivity.groupby(["group", "assignment_randomness"], sort=False)
        .agg(
            moran_corr_mean=("moran_corr", "mean"),
            moran_corr_min=("moran_corr", "min"),
            moran_corr_max=("moran_corr", "max"),
            direction_count=("direction", "size"),
        )
        .reset_index()
    )
    support = primary[
        [
            "job_id",
            "dataset",
            "mode",
            "group",
            "direction",
            "direction_label",
            "donor_stratum",
            "target_retained_fraction",
            "reference_observed_fraction",
            "n_target_spots",
            "n_genes",
            "n_reference_observed_genes",
            "evaluation_scope",
        ]
    ].copy()
    return primary, ar_summary, support


def plot_fidelity(ax: plt.Axes, primary: pd.DataFrame) -> None:
    metrics = ["mean_corr", "var_corr", "moran_corr", "zero_fidelity"]
    labels = ["Mean\nprofile r", "Variance\nprofile r", "Moran\nprofile r", "1 − zero\nKS"]
    offsets = np.linspace(-0.24, 0.24, len(GROUP_ORDER))
    for offset, group in zip(offsets, GROUP_ORDER):
        rows = primary[primary["group"] == group]
        for index, metric in enumerate(metrics):
            values = rows[metric].to_numpy(float)
            jitter = np.linspace(-0.035, 0.035, len(values)) if len(values) > 1 else np.zeros(1)
            ax.scatter(
                index + offset + jitter,
                values,
                s=22,
                marker=MARKERS[group],
                facecolor="white",
                edgecolor=COLORS[group],
                linewidth=0.8,
                alpha=0.85,
                zorder=2,
            )
            ax.scatter(
                index + offset,
                float(np.mean(values)),
                s=54,
                marker=MARKERS[group],
                color=COLORS[group],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
    ax.set_xticks(range(len(metrics)), labels)
    ax.set_ylim(0.68, 1.015)
    ax.set_ylabel("Fidelity (higher is better)")
    ax.set_title("A  Aggregate distribution and spatial-statistic fidelity", loc="left")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.7)


def plot_spotwise(ax: plt.Axes, primary: pd.DataFrame) -> None:
    metrics = ["median_gene_pearson", "median_gene_spearman"]
    labels = ["Median gene\nPearson", "Median gene\nSpearman"]
    offsets = np.linspace(-0.24, 0.24, len(GROUP_ORDER))
    for offset, group in zip(offsets, GROUP_ORDER):
        rows = primary[primary["group"] == group]
        for index, metric in enumerate(metrics):
            values = rows[metric].to_numpy(float)
            jitter = np.linspace(-0.035, 0.035, len(values)) if len(values) > 1 else np.zeros(1)
            ax.scatter(
                index + offset + jitter,
                values,
                s=22,
                marker=MARKERS[group],
                facecolor="white",
                edgecolor=COLORS[group],
                linewidth=0.8,
                alpha=0.85,
            )
            ax.scatter(
                index + offset,
                float(np.mean(values)),
                s=54,
                marker=MARKERS[group],
                color=COLORS[group],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
    ax.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(metrics)), labels)
    ax.set_ylim(-0.012, 0.062)
    ax.set_ylabel("Spotwise correlation")
    ax.set_title("B  Coordinate-wise expression correspondence", loc="left")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.7)
    ax.text(
        0.02,
        0.96,
        "Near-zero DLPFC medians limit spot-level prediction claims",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#333333",
    )


def plot_ar(ax: plt.Axes, ar_summary: pd.DataFrame) -> None:
    groups = [
        "DLPFC cross-slice, within donor",
        "DLPFC cross-slice, cross donor",
        "MERFISH cross-slice",
    ]
    for group in groups:
        rows = ar_summary[ar_summary["group"] == group].sort_values(
            "assignment_randomness"
        )
        x = rows["assignment_randomness"].to_numpy(float)
        y = rows["moran_corr_mean"].to_numpy(float)
        ax.fill_between(
            x,
            rows["moran_corr_min"].to_numpy(float),
            rows["moran_corr_max"].to_numpy(float),
            color=COLORS[group],
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            x,
            y,
            color=COLORS[group],
            marker=MARKERS[group],
            markersize=5,
            linewidth=1.8,
            label=group,
        )
    ax.axvline(0.3, color="#555555", linewidth=0.8, linestyle="--")
    ax.text(0.305, 0.56, "predeclared primary", rotation=90, va="bottom", fontsize=8)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.5])
    ax.set_xlabel("Assignment randomness")
    ax.set_ylabel("Moran profile correlation")
    ax.set_ylim(0.52, 0.95)
    ax.set_title("C  Cross-slice spatial sensitivity", loc="left")
    ax.grid(color="#DDDDDD", linewidth=0.5, alpha=0.7)


def plot_support(ax: plt.Axes, support: pd.DataFrame) -> None:
    order = support.sort_values(
        ["dataset", "mode", "donor_stratum", "direction"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    y = np.arange(len(order))
    ax.scatter(
        order["target_retained_fraction"],
        y,
        s=38,
        color="#333333",
        marker="o",
        label="Retained target spots",
        zorder=3,
    )
    ax.scatter(
        order["reference_observed_fraction"],
        y,
        s=38,
        facecolor="white",
        edgecolor="#C44E52",
        linewidth=1.1,
        marker="s",
        label="Globally observed reference genes",
        zorder=3,
    )
    for index, row in order.iterrows():
        low = min(row["target_retained_fraction"], row["reference_observed_fraction"])
        ax.plot([low, 1.0], [index, index], color="#D9D9D9", linewidth=0.7, zorder=1)
    ax.set_yticks(y, order["direction_label"].tolist(), fontsize=7.5)
    ax.set_xlim(0.80, 1.005)
    ax.set_xlabel("Fraction of declared support")
    ax.set_title("D  Target support and reference-gene evidence", loc="left")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5, alpha=0.7)
    ax.invert_yaxis()
    ax.legend(loc="lower left", fontsize=7.5, frameon=False)


def build_figure(primary: pd.DataFrame, ar_summary: pd.DataFrame, support: pd.DataFrame):
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "feast-study05-conditional-transfer",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.2), constrained_layout=False)
    plt.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.92, wspace=0.30, hspace=0.40)
    plot_fidelity(axes[0, 0], primary)
    plot_spotwise(axes[0, 1], primary)
    plot_ar(axes[1, 0], ar_summary)
    plot_support(axes[1, 1], support)

    handles = [
        plt.Line2D(
            [],
            [],
            color=COLORS[group],
            marker=MARKERS[group],
            linestyle="none",
            markersize=6,
            label=group,
        )
        for group in GROUP_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=VIS_DIR / "figures")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"fresh figure contract forbids replacing {args.output_dir}")

    summary, input_paths, validation = verify_inputs()
    primary, ar_summary, support = prepare_plotting_data(summary)
    work_root = VIS_DIR / ".work"
    work_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="study05-figure-", dir=work_root))
    primary.to_csv(work_dir / "primary_metrics.csv", index=False)
    ar_summary.to_csv(work_dir / "ar_sensitivity.csv", index=False)
    support.to_csv(work_dir / "support_and_panel.csv", index=False)

    figure = build_figure(primary, ar_summary, support)
    stem = "conditional_transfer_fidelity_and_limits"
    figure.savefig(work_dir / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(work_dir / f"{stem}.svg", bbox_inches="tight")
    figure.savefig(work_dir / f"{stem}.png", dpi=int(args.dpi), bbox_inches="tight")
    plt.close(figure)

    generation_hashes = {
        row["configuration_id"]: row["generated_sha256"]
        for row in validation["jobs"]
    }
    source_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    output_hashes = {
        path.name: sha256_file(path)
        for path in sorted(work_dir.iterdir())
        if path.is_file()
    }
    provenance = {
        "status": "ok",
        "configuration_id": "study05-conditional-transfer-figure-v1",
        "scientific_disposition": "validated_evidence_claim_reframe_required",
        "primary_assignment_randomness": 0.3,
        "input_sha256": source_hashes,
        "generation_h5ad_sha256": generation_hashes,
        "plot_script_sha256": sha256_file(Path(__file__)),
        "rows": {
            "score_summary": len(summary),
            "primary": len(primary),
            "ar_summary": len(ar_summary),
            "support": len(support),
        },
        "claim_limits": [
            "conditional on observed target XY and labels",
            "aggregate distribution/spatial-statistic fidelity is distinct from spotwise recovery",
            "DLPFC spotwise median gene correlations are near zero",
            "151670 outgoing directions evaluate only supported target labels",
            "MERFISH donor identity is not declared",
        ],
        "vector_text": {
            "pdf_fonttype": 42,
            "svg_fonttype": "none",
            "adobe_illustrator_editable_text": True,
        },
        "outputs": output_hashes,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (work_dir / "figure_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(work_dir, args.output_dir)
    print(
        json.dumps(
            {"status": "ok", "output_dir": str(args.output_dir), "outputs": output_hashes},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
