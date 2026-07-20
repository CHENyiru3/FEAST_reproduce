#!/usr/bin/env python3
"""Render the validated Study 03 common-support diagnostic candidate."""

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
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / "PUBLICATION_MANIFEST.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

METHOD_ORDER = ["RCTD", "Cell2location"]
METHOD_LABELS = {"rctd": "RCTD", "cell2location": "Cell2location"}
COLORS = {"RCTD": "#365C8D", "Cell2location": "#E07A3F"}
METRICS = [
    ("jsd_mean", "Mean Jensen–Shannon divergence ↓"),
    ("pearson_mean", "Mean Pearson correlation ↑"),
    ("summed_rmse", "Summed RMSE ↓"),
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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, Path]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    study = manifest["studies"]["03_deconvolution"]
    evidence = study["validated_evidence"]

    if study["status"] != "decision_required_headline_direction_changed":
        raise ValueError("Study 03 is not in the required headline-direction-changed state")
    if evidence["publication_claim_authorized"] is not False:
        raise ValueError("Study 03 publication claim must remain unauthorized")
    if evidence["aggregate_winner_ranking_authorized"] is not False:
        raise ValueError("Study 03 aggregate ranking must remain unauthorized")

    sources = {
        "scores": selected_source(evidence, "scores"),
        "support_audit": selected_source(evidence, "support_audit"),
        "decision": selected_source(evidence, "decision"),
    }
    scores = pd.read_csv(sources["scores"])
    support = pd.read_csv(sources["support_audit"])
    decision = json.loads(sources["decision"].read_text())

    required_score_columns = {
        "method",
        "pair_id",
        "slice_id",
        "resolution",
        "status",
        "comparison_scope",
        "n_input_spots",
        "n_scored_spots",
        "n_zero_library_spots_excluded",
        "n_common_named_cell_types",
        "n_scored_cell_types_including_other",
        *(column for column, _ in METRICS),
    }
    missing = sorted(required_score_columns - set(scores.columns))
    if missing:
        raise ValueError(f"Common-support score table lacks columns: {missing}")
    if len(scores) != 12 or set(scores["method"]) != set(METHOD_LABELS):
        raise ValueError("Expected 12 rows split across RCTD and Cell2location")
    if not scores["status"].eq("ok").all():
        raise ValueError("Every selected Study 03 score row must have status=ok")
    expected_scope = "same_expression_exact_spots_common_named_types_plus_other"
    if not scores["comparison_scope"].eq(expected_scope).all():
        raise ValueError("Study 03 scores do not use the declared common-support scope")
    if scores.duplicated(["method", "pair_id"]).any():
        raise ValueError("Duplicate method/pair rows in Study 03 scores")

    required_support_true = [
        "exact_positive_spot_support_between_methods",
        "cell2location_full_spot_support_exact",
        "rctd_type_set_equals_truth_ordered_common_type_set",
        "reference_common_type_set_equals_canonical_common_type_set",
        "cell2location_common_type_set_equals_canonical_common_type_set",
        "other_residual_included",
    ]
    if len(support) != 6 or set(support["pair_id"]) != set(scores["pair_id"]):
        raise ValueError("Support audit must contain exactly the six scored pairs")
    for column in required_support_true:
        if column not in support or not support[column].astype(bool).all():
            raise ValueError(f"Support-audit assertion failed: {column}")

    if decision["status"] != "stop_headline_direction_changed":
        raise ValueError("Study 03 decision does not record the scientific stop")
    if decision["direction_changed"] is not True:
        raise ValueError("Study 03 decision must record a changed headline direction")
    if decision["publication_claim_authorized"] is not False:
        raise ValueError("Decision unexpectedly authorizes a publication claim")
    if decision["aggregate_winner_ranking_authorized"] is not False:
        raise ValueError("Decision unexpectedly authorizes an aggregate ranking")

    return scores, support, decision, sources


def build_tables(
    scores: pd.DataFrame,
    support: pd.DataFrame,
    decision: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered_pairs = (
        scores[["pair_id", "slice_id", "resolution"]]
        .drop_duplicates()
        .sort_values(["slice_id", "resolution"])
    )
    pair_order = ordered_pairs["pair_id"].tolist()

    plot_rows: list[dict[str, object]] = []
    for row in scores.itertuples(index=False):
        for column, label in METRICS:
            plot_rows.append(
                {
                    "pair_id": row.pair_id,
                    "pair_order": pair_order.index(row.pair_id) + 1,
                    "slice_id": int(row.slice_id),
                    "resolution": float(row.resolution),
                    "method": METHOD_LABELS[row.method],
                    "metric": column,
                    "metric_label": label,
                    "value": float(getattr(row, column)),
                    "comparison_scope": row.comparison_scope,
                    "scientific_stop": True,
                    "publication_claim_authorized": False,
                    "aggregate_winner_ranking_authorized": False,
                }
            )
    plot_table = pd.DataFrame(plot_rows).sort_values(
        ["metric", "pair_order", "method"], kind="stable"
    )

    support_columns = [
        "pair_id",
        "slice_id",
        "resolution",
        "n_full_spots",
        "n_scored_spots",
        "n_zero_library_spots",
        "n_common_named_types",
        "exact_positive_spot_support_between_methods",
        "cell2location_full_spot_support_exact",
        "other_residual_included",
    ]
    support_table = support[support_columns].sort_values(
        ["slice_id", "resolution"], kind="stable"
    )

    historical = decision["historical_metric_winners"]
    fresh = decision["metric_winners"]
    differences = decision["metric_differences"]
    difference_keys = {
        "jsd_mean": "jsd_mean_rctd_minus_cell2location",
        "pearson_mean": "pearson_mean_rctd_minus_cell2location",
        "summed_rmse": "summed_rmse_rctd_minus_cell2location",
    }
    changed = set(decision["changed_metric_directions"])
    direction_table = pd.DataFrame(
        [
            {
                "metric": metric,
                "historical_direction": historical[metric],
                "fresh_direction": fresh[metric],
                "fresh_rctd_minus_cell2location": differences[difference_keys[metric]],
                "headline_direction_changed": metric in changed,
                "scientific_stop": True,
                "publication_claim_authorized": False,
                "aggregate_winner_ranking_authorized": False,
            }
            for metric, _ in METRICS
        ]
    )
    return plot_table, support_table, direction_table


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "feast-study03-common-support-candidate",
            "font.size": 9,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path) -> list[Path]:
    stem = output_dir / "common_support_diagnostic"
    outputs: list[Path] = []
    formats = {
        ".pdf": {
            "metadata": {
                "Creator": "FEAST reproduction Study 03 plot.py",
                "CreationDate": None,
                "ModDate": None,
            }
        },
        ".svg": {
            "metadata": {
                "Creator": "FEAST reproduction Study 03 plot.py",
                "Date": "1970-01-01T00:00:00Z",
            }
        },
        ".png": {
            "dpi": 360,
            "metadata": {"Software": "FEAST reproduction Study 03 plot.py"},
        },
    }
    for suffix, kwargs in formats.items():
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    return outputs


def draw_figure(
    plot_table: pd.DataFrame,
    support_table: pd.DataFrame,
    direction_table: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    labels = (
        support_table.assign(
            label=lambda frame: frame.apply(
                lambda row: f"S{int(row.slice_id):03d}\nr={row.resolution:g}", axis=1
            )
        )
        .sort_values(["slice_id", "resolution"])["label"]
        .tolist()
    )
    x = list(range(len(labels)))
    direction_lookup = direction_table.set_index("metric")

    for panel_index, (ax, (metric, label)) in enumerate(zip(axes, METRICS)):
        subset = plot_table.loc[plot_table["metric"] == metric]
        for method in METHOD_ORDER:
            rows = subset.loc[subset["method"] == method].sort_values("pair_order")
            ax.plot(
                x,
                rows["value"],
                color=COLORS[method],
                marker="o",
                markersize=4.5,
                linewidth=1.5,
                label=method,
            )
        changed = bool(direction_lookup.loc[metric, "headline_direction_changed"])
        status = "Historical direction changed" if changed else "Historical direction retained"
        ax.set_title(label, fontsize=10.5, fontweight="bold", pad=25)
        ax.text(
            0.5,
            1.03,
            status,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#A12622" if changed else "#555555",
            fontweight="bold" if changed else "normal",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.text(
            -0.10,
            1.12,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )
        for spine in ax.spines.values():
            spine.set_color("#666666")
            spine.set_linewidth(0.8)

    axes[0].legend(loc="best", frameon=False, fontsize=8.5)
    fig.suptitle(
        "Study 03 corrected common-support evidence — SCIENTIFIC STOP",
        fontsize=14,
        fontweight="bold",
        color="#8D1D18",
        y=1.03,
    )
    fig.text(
        0.5,
        -0.01,
        "Exact positive-spot intersection; common named cell types plus Other. "
        "Headline direction changed. No publication claim, aggregate winner, or method ranking is authorized.",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#5A1714",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94), w_pad=2.0)
    return save_figure(fig, output_dir)


def write_provenance(
    sources: dict[str, Path],
    plot_table: pd.DataFrame,
    support_table: pd.DataFrame,
    output_paths: list[Path],
) -> Path:
    payload = {
        "schema_version": 1,
        "figure_set": "study03_common_support_diagnostic_candidate",
        "publication_status": "scientific_stop_headline_direction_changed",
        "scientific_disposition": {
            "scientific_stop": True,
            "headline_direction_changed": True,
            "publication_claim_authorized": False,
            "aggregate_winner_ranking_authorized": False,
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
            "score_rows": 12,
            "plot_rows": int(len(plot_table)),
            "support_pairs": int(len(support_table)),
            "methods": METHOD_ORDER,
            "metrics": [column for column, _ in METRICS],
            "comparison_scope": "same_expression_exact_spots_common_named_types_plus_other",
            "ranking_or_composite_used": False,
        },
        "outputs": {
            path.name: sha256(path)
            for path in sorted(output_paths, key=lambda item: item.name)
        },
    }
    path = OUTPUT_DIR / "figure_provenance.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def main() -> int:
    scores, support, decision, sources = load_inputs()
    plot_table, support_table, direction_table = build_tables(scores, support, decision)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table_paths = [
        OUTPUT_DIR / "common_support_scores_plot.csv",
        OUTPUT_DIR / "common_support_audit_plot.csv",
        OUTPUT_DIR / "headline_direction_audit_plot.csv",
    ]
    plot_table.to_csv(table_paths[0], index=False, float_format="%.12g")
    support_table.to_csv(table_paths[1], index=False, float_format="%.12g")
    direction_table.to_csv(table_paths[2], index=False, float_format="%.12g")

    figure_paths = draw_figure(plot_table, support_table, direction_table, OUTPUT_DIR)
    write_provenance(sources, plot_table, support_table, table_paths + figure_paths)
    for path in table_paths + figure_paths + [OUTPUT_DIR / "figure_provenance.json"]:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
