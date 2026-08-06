#!/usr/bin/env python3
"""Render Study 01 alteration-level clustering sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("SOURCE_DATE_EPOCH", "1784419200")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / "PUBLICATION_MANIFEST.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
EXTENSION_REPORT_DIR = (
    REPOSITORY_ROOT
    / "01_clustering"
    / "outputs"
    / "expanded_mean_variance_fc_0_20_5_20260806"
    / "report"
)
EXTENSION_METRICS_PATH = EXTENSION_REPORT_DIR / "expanded_benchmark_metrics.csv"
EXTENSION_DIAGNOSTICS_PATH = EXTENSION_REPORT_DIR / "realized_alteration_diagnostics.csv"
EXTENSION_VALIDATION_PATH = EXTENSION_REPORT_DIR / "validation.json"
EXTENSION_CONFIGURATION_ID = "study01-expanded-mean-variance-extremes-v1"
EXTENSION_CONDITIONS = {
    ("mean_0_20", "mean", 0.2),
    ("mean_5", "mean", 5.0),
    ("variance_0_20", "variance", 0.2),
    ("variance_5", "variance", 5.0),
}

METRICS = ("ARI", "NMI", "AMI")
METHOD_ORDER = ("GraphST", "STAGATE_mclust", "Leiden_unsupervised")
METHOD_LABELS = {
    "GraphST": "GraphST",
    "STAGATE_mclust": "STAGATE + mclust",
    "Leiden_unsupervised": "Label-free Leiden",
}
# Wong colorblind-friendly palette
METHOD_COLORS = {
    "GraphST": "#0072B2",
    "STAGATE_mclust": "#E69F00",
    "Leiden_unsupervised": "#009E73",
}
METHOD_MARKERS = {
    "GraphST": "o",
    "STAGATE_mclust": "s",
    "Leiden_unsupervised": "^",
}

ORDERED_ALTERATIONS = ("mean", "variance", "sparsity")
ALTERATION_LABELS = {
    "mean": "Mean",
    "variance": "Variance",
    "sparsity": "Sparsity",
}
# simulation_id prefix → plot group (checked longest-first)
_ALT_PREFIX_MAP = [
    ("sparsity_neg", "sparsity"),
    ("sparsity_pos", "sparsity"),
    ("variance", "variance"),
    ("mean", "mean"),
]

BASELINE_REF = "real_baseline"
FEAST_BASELINE = "baseline"
# Neutral (no-perturbation) level for each ordered alteration type
NEUTRAL_LEVEL = {
    "mean": 1.0,
    "variance": 1.0,
    "sparsity": 0.0,
}
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


def validated_extension_sources() -> dict[str, object]:
    if not EXTENSION_VALIDATION_PATH.is_file():
        raise FileNotFoundError(
            f"Expanded Study 01 validation is missing: {EXTENSION_VALIDATION_PATH}"
        )
    validation = json.loads(EXTENSION_VALIDATION_PATH.read_text())
    if validation.get("configuration_id") != EXTENSION_CONFIGURATION_ID:
        raise ValueError("Expanded Study 01 configuration identity does not match.")
    if validation.get("all_status_ok") is not True:
        raise ValueError("Expanded Study 01 validation did not report all jobs successful.")
    if (
        int(validation.get("simulation_jobs", -1)) != 12
        or int(validation.get("fixed_panel_jobs", -1)) != 12
        or int(validation.get("metric_rows", -1)) != 36
        or sum(int(value) for value in validation.get("method_jobs", {}).values()) != 36
    ):
        raise ValueError("Expanded Study 01 validation counts do not match the extension contract.")

    metrics_hash = str(validation["expanded_benchmark_metrics_sha256"])
    diagnostics_hash = str(validation["realized_alteration_diagnostics_sha256"])
    verify_registered_file(EXTENSION_METRICS_PATH, metrics_hash)
    verify_registered_file(EXTENSION_DIAGNOSTICS_PATH, diagnostics_hash)
    return {
        "metrics_path": EXTENSION_METRICS_PATH,
        "metrics_sha256": metrics_hash,
        "diagnostics_path": EXTENSION_DIAGNOSTICS_PATH,
        "diagnostics_sha256": diagnostics_hash,
        "validation_path": EXTENSION_VALIDATION_PATH,
        "validation_sha256": sha256(EXTENSION_VALIDATION_PATH),
    }


def classify_alteration(sim_id: str) -> str | None:
    for prefix, group in _ALT_PREFIX_MAP:
        if sim_id.startswith(prefix):
            return group
    return None


def _extract_level(sim_id: str) -> float | str:
    """Extract numeric level or string label from a simulation_id.
    Uses the matched prefix to strip, then converts underscore-separated
    digits to a decimal (e.g. '0_50' → 0.50)."""
    for prefix, _group in _ALT_PREFIX_MAP:
        if sim_id.startswith(prefix):
            suffix = sim_id[len(prefix) + 1:]  # +1 for the trailing '_'
            try:
                return float(suffix.replace("_", "."))
            except ValueError:
                return suffix
    raise ValueError(f"Cannot classify simulation_id: {sim_id}")


def _signed_level(sim_id: str, group: str, raw_level: float) -> float:
    """Apply sign convention for merged alteration groups."""
    if group == "sparsity":
        if sim_id.startswith("sparsity_neg"):
            return -abs(raw_level)
        return abs(raw_level)
    return raw_level


def load_alteration_data(
    metrics_path: Path,
    extension_metrics_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    metrics = pd.read_csv(metrics_path)
    required = {"slice_id", "simulation_id", "alteration_type", "method", "status", *METRICS}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"Registered metric table is missing columns: {missing}")
    if len(metrics) != 252 or set(metrics["status"]) != {"ok"}:
        raise ValueError("Expected 252 successful rows in the registered Study 01 table.")
    if set(metrics["method"]) != set(METHOD_ORDER):
        raise ValueError("Registered metric table has an unexpected method set.")
    metrics["source_set"] = "frozen_canonical"

    extension = pd.read_csv(extension_metrics_path)
    extension_required = required | {"alteration_value"}
    extension_missing = sorted(extension_required - set(extension.columns))
    if extension_missing:
        raise ValueError(f"Expanded metric table is missing columns: {extension_missing}")
    if len(extension) != 36 or set(extension["status"]) != {"ok"}:
        raise ValueError("Expected 36 successful rows in the expanded Study 01 table.")
    if set(extension["method"]) != set(METHOD_ORDER):
        raise ValueError("Expanded metric table has an unexpected method set.")
    conditions = {
        (str(row.simulation_id), str(row.alteration_type), float(row.alteration_value))
        for row in extension[["simulation_id", "alteration_type", "alteration_value"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    if conditions != EXTENSION_CONDITIONS:
        raise ValueError(f"Expanded metric conditions do not match: {conditions}")
    key_columns = ["slice_id", "simulation_id", "method"]
    if extension.duplicated(key_columns).any():
        raise ValueError("Expanded metric table contains duplicate slice-condition-method rows.")
    extension["source_set"] = "expanded_extremes"
    metrics = pd.concat([metrics, extension], ignore_index=True, sort=False)

    baseline_ref = metrics.loc[
        metrics["simulation_id"] == BASELINE_REF,
        ["slice_id", "method", *METRICS],
    ].copy()
    baseline_ref = baseline_ref.rename(
        columns={m: f"{m}_ref" for m in METRICS}
    )

    feast_baseline = metrics.loc[
        metrics["simulation_id"] == FEAST_BASELINE,
        ["slice_id", "method", *METRICS],
    ].copy()

    alteration_rows = metrics.loc[
        ~metrics["simulation_id"].isin([BASELINE_REF, FEAST_BASELINE])
    ].copy()
    alteration_rows["plot_group"] = alteration_rows["simulation_id"].apply(classify_alteration)
    alteration_rows = alteration_rows.loc[alteration_rows["plot_group"].notna()].copy()
    alteration_rows["raw_level"] = alteration_rows["simulation_id"].apply(_extract_level)
    alteration_rows["level"] = alteration_rows.apply(
        lambda r: _signed_level(r["simulation_id"], r["plot_group"], r["raw_level"]), axis=1
    )

    alteration_data: dict[str, pd.DataFrame] = {}
    for alt_type in ORDERED_ALTERATIONS:
        subset = alteration_rows.loc[alteration_rows["plot_group"] == alt_type].copy()
        if subset.empty:
            continue
        subset["level_num"] = pd.to_numeric(subset["level"])
        neutral = NEUTRAL_LEVEL[alt_type]
        anchor = feast_baseline.copy()
        anchor["level_num"] = neutral
        anchor["level"] = str(neutral)
        anchor["plot_group"] = alt_type
        subset = pd.concat([subset, anchor], ignore_index=True)
        subset = subset.sort_values(["method", "slice_id", "level_num"])
        alteration_data[alt_type] = subset

    return baseline_ref, alteration_rows, alteration_data


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "feast-study01-publication-v3",
        }
    )


def _panel_y_range(
    alteration_data: dict[str, pd.DataFrame],
    baseline_ref: pd.DataFrame,
    metric: str,
) -> tuple[float, float]:
    """Compute shared y-range for a metric across all panels."""
    all_vals = [baseline_ref[f"{metric}_ref"].to_numpy()]
    for data in alteration_data.values():
        if data is not None and not data.empty and metric in data.columns:
            all_vals.append(data[metric].dropna().to_numpy())
    combined = np.concatenate([v for v in all_vals if len(v) > 0])
    if len(combined) == 0:
        return 0.0, 1.0
    pad = 0.08 * (combined.max() - combined.min()) if combined.max() > combined.min() else 0.05
    return max(0.0, combined.min() - pad), min(1.0, combined.max() + pad)


def draw_ordered_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    y_lim: tuple[float, float],
    neutral_level: float,
    x_label: str,
    show_xlabel: bool,
) -> None:
    levels = sorted(data["level_num"].unique())
    for method in METHOD_ORDER:
        method_data = data.loc[data["method"] == method]
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]

        # The transparent traces preserve the three-slice variation without
        # mixing it with a separate min--max band and point cloud.
        for _slice_id, slice_data in method_data.groupby("slice_id", sort=True):
            slice_data = slice_data.sort_values("level_num")
            ax.plot(
                slice_data["level_num"],
                slice_data[metric],
                color=color,
                linewidth=0.85,
                alpha=0.28,
                zorder=1,
            )

        grouped = method_data.groupby("level_num")[metric]
        means = grouped.mean()

        ordered_means = [means.get(lev, np.nan) for lev in levels]
        valid = [(l, m) for l, m in zip(levels, ordered_means) if not np.isnan(m)]

        if len(valid) >= 2:
            x_vals = [v[0] for v in valid]
            y_vals = [v[1] for v in valid]
            ax.plot(
                x_vals,
                y_vals,
                color=color,
                linewidth=2.0,
                marker=marker,
                markersize=5.8,
                markeredgecolor="white",
                markeredgewidth=0.7,
                zorder=3,
            )

    ax.set_ylim(*y_lim)
    fold_change_axis = x_label.startswith("Fold")
    if fold_change_axis:
        ax.set_xscale("log")
        ax.set_xlim(min(levels) / 1.15, max(levels) * 1.15)
        ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1.0, 2.0, 5.0]))
        ax.xaxis.set_major_formatter(FixedFormatter(["0.2", "0.5", "1", "2", "5"]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_xlim(
            min(levels) - 0.06 * (max(levels) - min(levels) or 1.0),
            max(levels) + 0.06 * (max(levels) - min(levels) or 1.0),
        )
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.axvline(neutral_level, color="#777777", linewidth=0.8, linestyle="--", alpha=0.8, zorder=0)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    if show_xlabel:
        ax.set_xlabel(x_label, fontsize=10, color="#333333")


def draw_figure(
    alteration_data: dict[str, pd.DataFrame],
    baseline_ref: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    configure_matplotlib()
    panel_types = list(ORDERED_ALTERATIONS)
    n_rows = len(METRICS)
    n_cols = len(panel_types)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.8, 9.2))

    for row, metric in enumerate(METRICS):
        y_lim = _panel_y_range(alteration_data, baseline_ref, metric)
        for col, alt_type in enumerate(panel_types):
            ax = axes[row, col]
            data = alteration_data.get(alt_type)
            if data is None or data.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="#999999")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            show_xlabel = (row == n_rows - 1)
            x_label = (
                "Fold change (neutral = 1)"
                if alt_type in {"mean", "variance"}
                else "Signed logit shift (neutral = 0)"
            )
            draw_ordered_panel(
                ax,
                data,
                metric,
                y_lim,
                NEUTRAL_LEVEL[alt_type],
                x_label,
                show_xlabel,
            )

            if col == 0:
                ax.set_ylabel(f"{metric} ↑", fontsize=11.5, fontweight="bold", labelpad=5)
            if row == 0:
                ax.set_title(ALTERATION_LABELS.get(alt_type, alt_type),
                             fontsize=12.5, fontweight="bold", pad=10)

    method_handles = []
    for method in METHOD_ORDER:
        method_handles.append(
            plt.Line2D(
                [0], [0],
                marker=METHOD_MARKERS[method],
                linestyle="-",
                markerfacecolor=METHOD_COLORS[method],
                markeredgecolor="white",
                color=METHOD_COLORS[method],
                markersize=7.5,
                linewidth=1.8,
                label=METHOD_LABELS[method],
            )
        )
    fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.76, 0.985),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=1.2,
        handletextpad=0.6,
        columnspacing=1.5,
    )

    fig.text(
        0.075,
        0.985,
        "Clustering sensitivity to controlled alterations",
        ha="left",
        va="center",
        fontsize=13.5,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.945,
        "Thin traces: individual DLPFC slices; solid lines and markers: three-slice means. Mean/variance axes show requested FC on a log scale.",
        ha="left",
        va="center",
        fontsize=8.9,
        color="#666666",
    )

    fig.subplots_adjust(left=0.085, right=0.995, top=0.90, bottom=0.09,
                        hspace=0.34, wspace=0.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "alteration_sensitivity"
    paths = [stem.with_suffix(suffix) for suffix in (".pdf", ".png", ".svg")]
    fig.savefig(paths[0], metadata={"Creator": "FEAST Study 01 plot.py"})
    fig.savefig(paths[1], dpi=dpi, metadata={"Software": "FEAST Study 01 plot.py"})
    fig.savefig(paths[2], metadata={"Creator": "FEAST Study 01 plot.py"})
    plt.close(fig)
    return paths


def write_provenance(
    output_dir: Path,
    outputs: list[Path],
    metrics_path: Path,
    metrics_hash: str,
    decision_path: Path,
    decision_hash: str,
    extension_sources: dict[str, object],
    baseline_ref: pd.DataFrame,
    alteration_rows: pd.DataFrame,
) -> Path:
    payload = {
        "schema_version": 4,
        "figure_set": "study01_alteration_sensitivity",
        "source": {
            "publication_manifest": repository_path(MANIFEST_PATH),
            "publication_manifest_sha256": sha256(MANIFEST_PATH),
            "metrics_csv": repository_path(metrics_path),
            "metrics_sha256": metrics_hash,
            "publication_decision": repository_path(decision_path),
            "publication_decision_sha256": decision_hash,
            "expanded_metrics_csv": repository_path(extension_sources["metrics_path"]),
            "expanded_metrics_sha256": extension_sources["metrics_sha256"],
            "expanded_diagnostics_csv": repository_path(extension_sources["diagnostics_path"]),
            "expanded_diagnostics_sha256": extension_sources["diagnostics_sha256"],
            "expanded_validation": repository_path(extension_sources["validation_path"]),
            "expanded_validation_sha256": extension_sources["validation_sha256"],
            "plot_script": repository_path(Path(__file__)),
            "plot_script_sha256": sha256(Path(__file__)),
        },
        "design": {
            "metrics": list(METRICS),
            "methods": list(METHOD_ORDER),
            "ordered_alterations": list(ORDERED_ALTERATIONS),
            "baseline_reference": BASELINE_REF,
            "sparsity_merged": True,
            "var_hetero_removed": True,
            "combined_removed": True,
            "shared_y_per_metric": True,
            "mean_variance_x_scale": "log",
            "mean_variance_nominal_levels_added": [0.2, 5.0],
            "mean_variance_levels_are_requested_not_realized": True,
            "slice_variation_encoding": "thin individual-slice traces; solid three-slice mean traces",
            "neutral_setting_encoding": "dashed vertical line",
            "raw_slice_reference_lines_drawn": False,
            "slice_ids": sorted(baseline_ref["slice_id"].unique().tolist()),
            "n_slices": int(baseline_ref["slice_id"].nunique()),
            "n_alteration_rows": int(len(alteration_rows)),
            "n_expanded_alteration_rows": int(
                alteration_rows["source_set"].eq("expanded_extremes").sum()
            ),
            "winner_claimed": False,
        },
        "scientific_disposition": {
            "figure_promotion_authorized": False,
            "aggregate_method_ranking_authorized": False,
            "winner_claim_authorized": False,
            "supported_interpretation": (
                "Clustering sensitivity across ordered requested alteration levels, "
                "including the validated mean and variance extreme-level extension. "
                "Nominal fold changes must be interpreted with the realized diagnostics."
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
    extension_sources = validated_extension_sources()
    decision = json.loads(decision_path.read_text())
    if decision.get("decision") != "validated_publication_candidate":
        raise ValueError("The registered Study 01 decision is not a publication candidate.")

    baseline_ref, alteration_rows, alteration_data = load_alteration_data(
        metrics_path, extension_sources["metrics_path"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ref_path = args.output_dir / "baseline_reference.csv"
    rows_path = args.output_dir / "alteration_plot_data.csv"
    baseline_ref.to_csv(ref_path, index=False, float_format="%.12f")
    alteration_rows.to_csv(rows_path, index=False, float_format="%.12f")

    outputs = [ref_path, rows_path]
    outputs.extend(draw_figure(alteration_data, baseline_ref, args.output_dir, args.dpi))
    provenance_path = write_provenance(
        args.output_dir, outputs,
        metrics_path, metrics_hash, decision_path, decision_hash,
        extension_sources,
        baseline_ref, alteration_rows,
    )
    for path in [*outputs, provenance_path]:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
