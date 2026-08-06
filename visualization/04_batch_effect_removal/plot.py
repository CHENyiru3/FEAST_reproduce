#!/usr/bin/env python3
"""Render validated Study 04 supplementary atomic-metric diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/feast-reproduce-matplotlib")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / "PUBLICATION_MANIFEST.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

METHOD_ORDER = ["GraphST", "STAMP", "scVI"]
METHOD_COLORS = {"GraphST": "#3B6FB6", "STAMP": "#D26A35", "scVI": "#539D68"}
MODE_STYLES = {"shift_only": "-", "diagonal_affine": "--"}
MODE_LABELS = {"shift_only": "Shift only", "diagonal_affine": "Diagonal affine"}
METRICS = [
    ("batch_asw", "Batch ASW"),
    ("batch_mixing_entropy_k30", "Batch mixing entropy (k=30)"),
    ("paired_retrieval_top1", "Paired retrieval top-1"),
    ("paired_retrieval_median_rank", "Paired retrieval median rank"),
    ("domain_asw", "Domain ASW"),
    ("domain_purity_k30", "Domain purity (k=30)"),
    ("centroid_distance", "Centroid distance"),
    ("covariance_distance", "Covariance distance"),
    ("mmd2_rbf_biased", "Biased RBF MMD²"),
    ("paired_expression_correlation", "Paired expression correlation"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT))


def selected_source(entry: dict[str, object], key: str) -> Path:
    path = REPOSITORY_ROOT / str(entry[key])
    expected = str(entry[f"{key}_sha256"])
    if not path.is_file():
        raise FileNotFoundError(f"Manifest-selected source is missing: {path}")
    observed = sha256(path)
    if observed != expected:
        raise ValueError(
            f"Manifest hash mismatch for {key}: expected {expected}, observed {observed}"
        )
    return path


def load_inputs() -> tuple[pd.DataFrame, dict[str, object], dict[str, object], dict[str, Path]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    study = manifest["studies"]["04_batch_effect_removal"]
    candidate = study["canonical_candidate"]
    if study["status"] != "validated_complete_author_review_required":
        raise ValueError("Study 04 is not in the validated author-review state")
    expected_false = [
        "composite_score_present",
        "winner_ranking_authorized",
        "figure_promotion_authorized",
    ]
    for field in expected_false:
        if candidate[field] is not False:
            raise ValueError(f"Study 04 governance field must be false: {field}")
    if candidate["active_manuscript_claim"] is not None:
        raise ValueError("Study 04 must not have an active manuscript claim")

    sources = {
        "atomic_metrics": selected_source(candidate, "atomic_metrics"),
        "validation_summary": selected_source(candidate, "validation_summary"),
        "stamp_numerical_retry_policy": selected_source(
            candidate, "stamp_numerical_retry_policy"
        ),
    }
    metrics = pd.read_csv(sources["atomic_metrics"])
    validation = json.loads(sources["validation_summary"].read_text())

    required = {
        "method",
        "representation",
        "primary_representation",
        "mode",
        "alpha",
        "alpha_stratum",
        "support_variant",
        "primary_support",
        "analysis_role",
        *(column for column, _ in METRICS),
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"Atomic metric table lacks columns: {missing}")
    if len(metrics) != 60 or int(candidate["atomic_metric_rows"]) != 60:
        raise ValueError("Expected the selected 60-row Study 04 atomic table")
    if int(candidate["cross_batch_edges"]) != 0 or int(validation["cross_batch_edges"]) != 0:
        raise ValueError("Study 04 within-batch graph contract is not satisfied")
    if validation["active_manuscript_claim"] is not None:
        raise ValueError("Validation unexpectedly binds an active manuscript claim")
    if validation["decision"] != "validated_fresh_supplementary_candidate_author_review_required":
        raise ValueError("Study 04 validation is not supplementary/author-review only")
    if not any("PCA-20/PCA-10" in item for item in validation["limitations"]):
        raise ValueError("GraphST PCA sensitivity limitation is missing")
    if set(metrics["method"]) != set(METHOD_ORDER):
        raise ValueError("Study 04 method set differs from the manifest selection")

    primary = metrics.loc[metrics["analysis_role"].eq("primary")].copy()
    if len(primary) != 36 or primary.groupby("method").size().to_dict() != {
        method: 12 for method in METHOD_ORDER
    }:
        raise ValueError("Expected 12 primary atomic rows per method")
    if not primary["primary_representation"].astype(bool).all():
        raise ValueError("Primary rows include a representation sensitivity")
    if not primary["primary_support"].astype(bool).all():
        raise ValueError("Primary rows include a support sensitivity")

    return metrics, validation, candidate, sources


def build_long_table(primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in primary.sort_values(["method", "mode", "alpha"]).itertuples(index=False):
        for metric, label in METRICS:
            value = getattr(row, metric)
            rows.append(
                {
                    "method": row.method,
                    "mode": row.mode,
                    "alpha": float(row.alpha),
                    "alpha_stratum": row.alpha_stratum,
                    "metric": metric,
                    "metric_label": label,
                    "value": float(value) if pd.notna(value) else np.nan,
                    "analysis_role": "primary",
                    "support_variant": row.support_variant,
                    "representation": row.representation,
                    "supplementary_diagnostic_only": True,
                    "active_manuscript_claim": False,
                    "winner_ranking_authorized": False,
                    "composite_score_used": False,
                }
            )
    return pd.DataFrame(rows)


def build_availability(long_table: pd.DataFrame) -> pd.DataFrame:
    return (
        long_table.assign(available=lambda frame: frame["value"].notna())
        .groupby(["metric", "metric_label", "method"], sort=True, as_index=False)
        .agg(n_expected=("available", "size"), n_available=("available", "sum"))
    )


def build_graphst_sensitivity(metrics: pd.DataFrame) -> pd.DataFrame:
    graphst = metrics.loc[metrics["method"].eq("GraphST")].copy()
    keys = ["mode", "alpha", "alpha_stratum", "support_variant"]
    primary_rep = graphst.loc[
        graphst["representation"].eq("pca20_randomized_seed42")
    ]
    sensitivity_rep = graphst.loc[
        graphst["representation"].eq("pca10_randomized_seed42_sensitivity")
    ]
    merged = primary_rep.merge(
        sensitivity_rep,
        on=keys,
        suffixes=("_pca20", "_pca10"),
        validate="one_to_one",
    )
    if len(merged) != 15:
        raise ValueError("Expected all 15 paired GraphST PCA-20/PCA-10 sensitivity rows")

    rows: list[dict[str, object]] = []
    for row in merged.sort_values(keys).itertuples(index=False):
        for metric, label in METRICS:
            pca20 = getattr(row, f"{metric}_pca20")
            pca10 = getattr(row, f"{metric}_pca10")
            rows.append(
                {
                    "mode": row.mode,
                    "alpha": float(row.alpha),
                    "alpha_stratum": row.alpha_stratum,
                    "support_variant": row.support_variant,
                    "metric": metric,
                    "metric_label": label,
                    "pca20_value": float(pca20) if pd.notna(pca20) else np.nan,
                    "pca10_value": float(pca10) if pd.notna(pca10) else np.nan,
                    "pca10_minus_pca20": (
                        float(pca10 - pca20)
                        if pd.notna(pca10) and pd.notna(pca20)
                        else np.nan
                    ),
                    "supplementary_sensitivity_only": True,
                    "winner_ranking_authorized": False,
                }
            )
    return pd.DataFrame(rows)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "feast-study04-atomic-diagnostic-candidate",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    base = output_dir / stem
    outputs: list[Path] = []
    formats = {
        ".pdf": {
            "metadata": {
                "Creator": "FEAST reproduction Study 04 plot.py",
                "CreationDate": None,
                "ModDate": None,
            }
        },
        ".svg": {
            "metadata": {
                "Creator": "FEAST reproduction Study 04 plot.py",
                "Date": "1970-01-01T00:00:00Z",
            }
        },
        ".png": {
            "dpi": 360,
            "metadata": {"Software": "FEAST reproduction Study 04 plot.py"},
        },
    }
    for suffix, kwargs in formats.items():
        path = base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    return outputs


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(0.75)


def draw_primary_figure(long_table: pd.DataFrame, output_dir: Path) -> list[Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 5, figsize=(17.2, 7.2))
    axes = axes.ravel()
    for panel_index, (ax, (metric, label)) in enumerate(zip(axes, METRICS)):
        subset = long_table.loc[long_table["metric"].eq(metric)]
        for method in METHOD_ORDER:
            for mode, linestyle in MODE_STYLES.items():
                rows = subset.loc[
                    subset["method"].eq(method) & subset["mode"].eq(mode)
                ].sort_values("alpha")
                rows = rows.dropna(subset=["value"])
                if rows.empty:
                    continue
                ax.plot(
                    rows["alpha"],
                    rows["value"],
                    color=METHOD_COLORS[method],
                    linestyle=linestyle,
                    marker="o",
                    markersize=3,
                    linewidth=1.2,
                )
        ax.axvspan(1.0, 1.55, color="#E6E6E6", alpha=0.45, zorder=-2)
        ax.axvline(1.0, color="#999999", linewidth=0.7, linestyle=":", zorder=-1)
        ax.set_title(label, fontsize=9.2, fontweight="bold", pad=6)
        ax.set_xticks([0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
        ax.tick_params(axis="x", labelrotation=35)
        ax.text(
            -0.12,
            1.08,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        if metric == "paired_expression_correlation":
            ax.text(
                0.04,
                0.05,
                "Available for scVI only",
                transform=ax.transAxes,
                fontsize=7,
                color="#555555",
            )
        style_axis(ax)

    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], lw=2, label=method)
        for method in METHOD_ORDER
    ]
    mode_handles = [
        Line2D([0], [0], color="#444444", lw=1.5, ls=style, label=MODE_LABELS[mode])
        for mode, style in MODE_STYLES.items()
    ]
    fig.legend(
        handles=method_handles + mode_handles,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        frameon=False,
        fontsize=8.5,
    )
    fig.text(
        0.5,
        0.006,
        "Primary representations and all-spot support only. Gray region: α > 1 extrapolation. "
        "No active manuscript claim; figure promotion is not authorized.",
        ha="center",
        fontsize=8.5,
        color="#5B4315",
    )
    fig.supxlabel("Perturbation strength α", fontsize=10, y=0.035)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96), w_pad=1.7, h_pad=2.0)
    return save_figure(fig, output_dir, "atomic_metric_diagnostics")


def draw_sensitivity_figure(
    sensitivity: pd.DataFrame, output_dir: Path
) -> list[Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(3, 4, figsize=(13.5, 9.4))
    axes = axes.ravel()
    mode_markers = {"shift_only": "o", "diagonal_affine": "s"}
    support_colors = {
        "all_spots_primary": "#3B6FB6",
        "paired_zero_panel_spots_excluded_sensitivity": "#B45D26",
    }
    for panel_index, (ax, (metric, label)) in enumerate(zip(axes[:10], METRICS)):
        subset = sensitivity.loc[sensitivity["metric"].eq(metric)].dropna(
            subset=["pca20_value", "pca10_value"]
        )
        if subset.empty:
            ax.text(0.5, 0.5, "Not available", ha="center", va="center", color="#666666")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            values = pd.concat([subset["pca20_value"], subset["pca10_value"]])
            minimum, maximum = float(values.min()), float(values.max())
            span = max(maximum - minimum, abs(maximum) * 0.08, 1e-8)
            lower, upper = minimum - 0.08 * span, maximum + 0.08 * span
            ax.plot([lower, upper], [lower, upper], color="#999999", ls=":", lw=0.8)
            for row in subset.itertuples(index=False):
                ax.scatter(
                    row.pca20_value,
                    row.pca10_value,
                    s=22,
                    marker=mode_markers[row.mode],
                    color=support_colors[row.support_variant],
                    edgecolor="white",
                    linewidth=0.35,
                )
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
        ax.set_title(label, fontsize=9, fontweight="bold", pad=5)
        ax.text(
            -0.12,
            1.08,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        style_axis(ax)

    axes[10].axis("off")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="#555555",
            markeredgecolor="white",
            label=MODE_LABELS[mode],
        )
        for mode, marker in mode_markers.items()
    ] + [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            label=("All spots" if support == "all_spots_primary" else "Paired-support sensitivity"),
        )
        for support, color in support_colors.items()
    ]
    axes[10].legend(handles=legend_handles, loc="center", frameon=False, fontsize=8.5)
    axes[11].axis("off")
    axes[11].text(
        0.02,
        0.95,
        "15 paired GraphST rows\n\n"
        "Diagonal: PCA-10 = PCA-20\n\n"
        "Sensitivity evidence must\n"
        "accompany any future use.\n\n"
        "No aggregate ranking or\n"
        "figure promotion authorized.",
        ha="left",
        va="top",
        fontsize=8.5,
        linespacing=1.35,
        color="#4F4F4F",
    )
    fig.supxlabel("PCA-20 metric value", fontsize=10, y=0.025)
    fig.supylabel("PCA-10 metric value", fontsize=10, x=0.015)
    fig.tight_layout(rect=(0.03, 0.04, 1, 0.98), w_pad=1.8, h_pad=1.8)
    return save_figure(fig, output_dir, "graphst_pca_sensitivity")


def write_provenance(
    sources: dict[str, Path],
    validation: dict[str, object],
    candidate: dict[str, object],
    long_table: pd.DataFrame,
    sensitivity: pd.DataFrame,
    output_paths: list[Path],
) -> Path:
    payload = {
        "schema_version": 1,
        "figure_set": "study04_supplementary_atomic_diagnostic_candidate",
        "publication_status": "supplementary_unpromoted_author_review_required",
        "scientific_disposition": {
            "active_manuscript_claim": None,
            "supplementary_candidate_only": True,
            "composite_score_present": False,
            "winner_ranking_authorized": False,
            "figure_promotion_authorized": False,
        },
        "sources": {
            "publication_manifest": {
                "path": repository_path(MANIFEST_PATH),
                "sha256": sha256(MANIFEST_PATH),
            },
            **{
                role: {"path": repository_path(path), "sha256": sha256(path)}
                for role, path in sorted(sources.items())
            },
            "plot_script": {
                "path": repository_path(Path(__file__)),
                "sha256": sha256(Path(__file__)),
            },
        },
        "design": {
            "source_atomic_rows": int(candidate["atomic_metric_rows"]),
            "primary_rows": 36,
            "primary_long_rows": int(len(long_table)),
            "graphst_pca_sensitivity_pairs": 15,
            "graphst_pca_sensitivity_long_rows": int(len(sensitivity)),
            "methods": METHOD_ORDER,
            "metrics": [column for column, _ in METRICS],
            "cross_batch_edges": int(candidate["cross_batch_edges"]),
            "ranking_or_composite_used": False,
        },
        "limitations": validation["limitations"],
        "external_numpy_limitation": candidate["external_numpy_limitation"],
        "outputs": {
            path.name: sha256(path)
            for path in sorted(output_paths, key=lambda item: item.name)
        },
    }
    path = OUTPUT_DIR / "figure_provenance.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def main() -> int:
    metrics, validation, candidate, sources = load_inputs()
    primary = metrics.loc[metrics["analysis_role"].eq("primary")].copy()
    long_table = build_long_table(primary)
    availability = build_availability(long_table)
    sensitivity = build_graphst_sensitivity(metrics)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table_paths = [
        OUTPUT_DIR / "atomic_metrics_primary_long.csv",
        OUTPUT_DIR / "atomic_metric_availability.csv",
        OUTPUT_DIR / "graphst_pca_sensitivity_plot.csv",
    ]
    long_table.to_csv(table_paths[0], index=False, na_rep="NA", float_format="%.12g")
    availability.to_csv(table_paths[1], index=False)
    sensitivity.to_csv(table_paths[2], index=False, na_rep="NA", float_format="%.12g")

    figure_paths = draw_primary_figure(long_table, OUTPUT_DIR)
    figure_paths.extend(draw_sensitivity_figure(sensitivity, OUTPUT_DIR))
    write_provenance(
        sources,
        validation,
        candidate,
        long_table,
        sensitivity,
        table_paths + figure_paths,
    )
    for path in table_paths + figure_paths + [OUTPUT_DIR / "figure_provenance.json"]:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
