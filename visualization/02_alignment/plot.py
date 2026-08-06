#!/usr/bin/env python3
"""Render the unpromoted Study 02 alignment sensitivity diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ["SOURCE_DATE_EPOCH"] = "0"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "PUBLICATION_MANIFEST.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

METHODS = ("paste", "spateo")
METHOD_LABELS = {"paste": "PASTE", "spateo": "Spateo"}
ALTERATIONS = ("baseline", "mean_0.5", "variance_2.0", "sparsity_0.5")
ALTERATION_LABELS = {
    "baseline": "Baseline",
    "mean_0.5": "Mean ×0.5",
    "variance_2.0": "Variance ×2.0",
    "sparsity_0.5": "Sparsity ×0.5",
}
ALTERATION_COLORS = {
    "baseline": "#666666",
    "mean_0.5": "#0072B2",
    "variance_2.0": "#CC79A7",
    "sparsity_0.5": "#E69F00",
}
ALTERATION_MARKERS = {
    "baseline": "o",
    "mean_0.5": "s",
    "variance_2.0": "D",
    "sparsity_0.5": "^",
}
ANGLES = (1.0, 5.0, 10.0, 30.0, 45.0)
METRICS = (
    {
        "key": "mean_spatial_error",
        "label": "Mean spatial error",
        "direction": "lower_is_better",
        "scale": "log",
    },
    {
        "key": "rotation_recovery_error",
        "label": "Rotation recovery error (deg)",
        "direction": "lower_is_better",
        "scale": "linear",
    },
    {
        "key": "ge_correlation_exact_gene_join",
        "label": "GE correlation",
        "direction": "higher_is_better",
        "scale": "linear",
    },
)

# GE correlation is retained in the source-derived CSV for completeness, but
# it is invariant to alignment method and rotation in this design.  It is an
# expression-simulation property rather than an alignment diagnostic, so the
# figure focuses on the two directly interpretable alignment errors.
DISPLAY_METRICS = METRICS[:2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT))


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def registered_sources(manifest_path: Path) -> tuple[dict[str, object], dict[str, Path]]:
    manifest = load_json(manifest_path)
    try:
        candidate = manifest["studies"]["02_alignment"]["canonical_candidate"]
    except (KeyError, TypeError) as error:
        raise ValueError("Manifest lacks the Study 02 canonical candidate") from error
    if not isinstance(candidate, dict):
        raise ValueError("Study 02 canonical candidate must be an object")

    required_disposition = {
        "figure_promotion_authorized": False,
        "aggregate_method_ranking_authorized": False,
        "winner_claim_authorized": False,
    }
    for field, expected in required_disposition.items():
        if candidate.get(field) is not expected:
            raise ValueError(f"Refusing to plot: manifest {field} is not {expected}")

    sources: dict[str, Path] = {}
    for name in ("metrics", "validation", "decision"):
        relative_path = candidate.get(name)
        expected_hash = candidate.get(f"{name}_sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise ValueError(f"Manifest lacks registered {name} path/hash")
        path = REPOSITORY_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Registered {name} does not exist: {path}")
        observed_hash = sha256(path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"Registered {name} hash mismatch: expected {expected_hash}, "
                f"observed {observed_hash}"
            )
        sources[name] = path
    return candidate, sources


def load_validated_metrics(metrics_path: Path, validation_path: Path) -> list[dict[str, object]]:
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    required = {
        "job_id", "method", "alteration", "angle_degrees",
        "status", "canonical_candidate", "solver_evidence_disposition",
    } | {str(metric["key"]) for metric in METRICS}
    missing = sorted(required - set(source_rows[0] if source_rows else ()))
    if missing:
        raise ValueError(f"Alignment metrics lack required columns: {missing}")
    if len(source_rows) != 40:
        raise ValueError(f"Expected 40 alignment rows, found {len(source_rows)}")

    with validation_path.open(newline="", encoding="utf-8") as handle:
        validation_rows = list(csv.DictReader(handle))
    validated_jobs = {
        row["job_id"]
        for row in validation_rows
        if row.get("kind") == "method" and row.get("valid") == "True"
    }

    rows: list[dict[str, object]] = []
    seen_jobs: set[str] = set()
    observed_cells: set[tuple[str, str, float]] = set()
    for source in source_rows:
        job_id = source["job_id"]
        if job_id in seen_jobs:
            raise ValueError(f"Duplicate alignment job: {job_id}")
        if source["status"] != "ok" or source["canonical_candidate"] != "True":
            raise ValueError(f"Nonvalidated/noncandidate row: {job_id}")
        if job_id not in validated_jobs:
            raise ValueError(f"Alignment row lacks positive validation: {job_id}")

        method = source["method"]
        alteration = source["alteration"]
        angle = float(source["angle_degrees"])
        if method not in METHODS or alteration not in ALTERATIONS or angle not in ANGLES:
            raise ValueError(f"Unexpected Study 02 design cell: {job_id}")
        cell = (method, alteration, angle)
        if cell in observed_cells:
            raise ValueError(f"Duplicate Study 02 design cell: {cell}")

        row: dict[str, object] = {
            "job_id": job_id, "method": method,
            "alteration": alteration, "angle_degrees": angle,
            "solver_evidence_disposition": source["solver_evidence_disposition"],
        }
        for metric in METRICS:
            key = str(metric["key"])
            value = float(source[key])
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite {key} for {job_id}")
            row[key] = value
        rows.append(row)
        seen_jobs.add(job_id)
        observed_cells.add(cell)

    expected_cells = {
        (method, alteration, angle)
        for method in METHODS for alteration in ALTERATIONS for angle in ANGLES
    }
    if observed_cells != expected_cells or validated_jobs != seen_jobs:
        raise ValueError("Study 02 validation/design coverage is not exact")
    return rows


def build_plot_data(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    plot_rows: list[dict[str, object]] = []
    for alteration in ALTERATIONS:
        for metric in METRICS:
            key = str(metric["key"])
            for method in METHODS:
                selected = sorted(
                    (r for r in rows
                     if r["alteration"] == alteration and r["method"] == method),
                    key=lambda r: float(r["angle_degrees"]),
                )
                for row in selected:
                    plot_rows.append({
                        "job_id": row["job_id"], "method": method,
                        "alteration": alteration,
                        "angle_degrees": row["angle_degrees"],
                        "metric": key, "metric_label": metric["label"],
                        "direction": metric["direction"],
                        "value": row[key],
                        "solver_evidence_disposition": row["solver_evidence_disposition"],
                    })
    return plot_rows


def write_plot_data(plot_rows: list[dict[str, object]], output_dir: Path) -> Path:
    path = output_dir / "plot_data.csv"
    fieldnames = list(plot_rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(plot_rows)
    return path


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 11, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "feast-study02-alignment-diagnostic",
    })


def draw_figure(rows: list[dict[str, object]], output_dir: Path, dpi: int) -> list[Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(
        len(METHODS), len(DISPLAY_METRICS),
        figsize=(10.8, 6.25), sharex=True, squeeze=False,
    )

    for row_index, method in enumerate(METHODS):
        for column_index, metric in enumerate(DISPLAY_METRICS):
            ax = axes[row_index][column_index]
            key = str(metric["key"])
            for alteration in ALTERATIONS:
                selected = sorted(
                    (r for r in rows
                     if r["alteration"] == alteration and r["method"] == method),
                    key=lambda r: float(r["angle_degrees"]),
                )
                ax.plot(
                    [float(r["angle_degrees"]) for r in selected],
                    [float(r[key]) for r in selected],
                    color=ALTERATION_COLORS[alteration],
                    marker=ALTERATION_MARKERS[alteration],
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    markersize=5.7,
                    linewidth=1.8,
                    label=ALTERATION_LABELS[alteration],
                )

            if metric["scale"] == "log":
                ax.set_yscale("log")
                # Same absolute range for the two methods keeps the magnitude
                # comparison honest while retaining the 12-order dynamic range.
                ax.set_ylim(1e-12, 50.0)
            else:
                values = [float(r[key]) for r in rows if r["method"] == method]
                ymax = max(values)
                ax.set_ylim(-0.035 * ymax, 1.10 * ymax)
            ax.set_xticks(ANGLES)
            ax.set_xlim(0.0, 46.5)
            ax.grid(axis="y", color="#E0E0E0", linewidth=0.55, alpha=0.75)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8.5)
            for spine in ax.spines.values():
                spine.set_color("#555555")
                spine.set_linewidth(0.8)

            if row_index == 0:
                title = f"{metric['label']} ↓"
                if metric["scale"] == "log":
                    title += " (log scale)"
                ax.set_title(title, fontsize=11.5, fontweight="bold", pad=9)
            if column_index == 0:
                ax.set_ylabel("Mean spatial error", fontsize=9.5)
            else:
                ax.set_ylabel("Rotation recovery error (degrees)", fontsize=9.5)
            if row_index == len(METHODS) - 1:
                ax.set_xlabel("Input rotation (degrees)", fontsize=9.5)

    row_centers = (0.648, 0.301)
    for center, method in zip(row_centers, METHODS):
        fig.text(
            0.014, center, METHOD_LABELS[method], rotation=90,
            ha="center", va="center", fontsize=11, fontweight="bold",
        )
    handles = [
        Line2D(
            [0], [0], color=ALTERATION_COLORS[alteration],
            marker=ALTERATION_MARKERS[alteration], markerfacecolor="white",
            markeredgewidth=1.0, markersize=5.7, linewidth=1.8,
            label=ALTERATION_LABELS[alteration],
        )
        for alteration in ALTERATIONS
    ]
    fig.legend(
        handles, [handle.get_label() for handle in handles], loc="upper center", ncol=4,
        frameon=False, bbox_to_anchor=(0.56, 0.915), fontsize=9,
        handlelength=1.8, columnspacing=1.3,
    )
    fig.text(
        0.075, 0.975, "Alignment residuals across input rotations",
        ha="left", va="center", fontsize=14, fontweight="bold",
    )
    fig.text(
        0.075, 0.938,
        "Each marker is one validated run. Colours denote simulated conditions; no cross-condition aggregation or method ranking. "
        "Mean spatial-error panels share a log scale; rotation-error rows use their own raw linear limits.",
        ha="left", va="center", fontsize=8.1, color="#666666",
    )
    fig.subplots_adjust(left=0.105, right=0.99, top=0.825, bottom=0.105, hspace=0.32, wspace=0.28)

    stem = "alignment_sensitivity_diagnostic"
    output_paths: list[Path] = []
    formats = {
        "png": {"dpi": 600, "metadata": {"Software": "FEAST Study 02 plot.py"}},
        "pdf": {"metadata": {"CreationDate": None, "ModDate": None}},
        "svg": {"metadata": {"Date": None}},
    }
    for suffix, kwargs in formats.items():
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        output_paths.append(path)
    plt.close(fig)
    return output_paths


def write_provenance(
    manifest_path: Path, candidate: dict[str, object],
    sources: dict[str, Path], decision: dict[str, object],
    rows: list[dict[str, object]], plot_rows: list[dict[str, object]],
    output_paths: list[Path], output_dir: Path,
) -> Path:
    payload = {
        "schema_version": 2,
        "figure_set": "study02_alignment_sensitivity_diagnostic",
        "promotion_status": "unpromoted_author_selection_required",
        "publication_figure_authorized": False,
        "aggregate_method_ranking_authorized": False,
        "winner_claim_authorized": False,
        "author_action_required": candidate["author_action_required"],
        "scientific_disposition": {
            "status": "author_scope_required",
            "figure_promotion_authorized": False,
            "aggregate_method_ranking_authorized": False,
            "winner_claim_authorized": False,
        },
        "source": {
            "publication_manifest": repository_path(manifest_path),
            "publication_manifest_sha256": sha256(manifest_path),
            "metrics": repository_path(sources["metrics"]),
            "metrics_sha256": sha256(sources["metrics"]),
            "validation": repository_path(sources["validation"]),
            "validation_sha256": sha256(sources["validation"]),
            "decision": repository_path(sources["decision"]),
            "decision_sha256": sha256(sources["decision"]),
            "plot_script": repository_path(Path(__file__)),
            "plot_script_sha256": sha256(Path(__file__)),
        },
        "design": {
            "validated_atomic_rows": len(rows),
            "plot_data_rows": len(plot_rows),
            "methods": list(METHODS), "alterations": list(ALTERATIONS),
            "angles_degrees": list(ANGLES),
            "source_metrics": [
                {"key": m["key"], "label": m["label"], "direction": m["direction"]}
                for m in METRICS
            ],
            "display_metrics": [
                {"key": m["key"], "label": m["label"], "direction": m["direction"]}
                for m in DISPLAY_METRICS
            ],
            "aggregation": "none; every plotted point is one validated method job",
            "display_scope": (
                "GE correlation remains in plot_data.csv but is excluded from the main diagnostic "
                "because it is invariant across methods and rotations in the validated design."
            ),
            "changes_v3": (
                "Replaced the 4×3 matrix with a 2×2 method-by-error layout; simulation condition "
                "is encoded by colour and marker, all 40 atomic rows remain in the companion data CSV."
            ),
        },
        "limitations": {
            "manuscript_or_figure_claim_direction": decision["manuscript_or_figure_claim_direction"],
            "paste_convergence_disposition": decision["paste_convergence_disposition"],
            "spateo_convergence_disposition": decision["spateo_convergence_disposition"],
            "external_environment_limitation": decision["external_environment_limitation"],
        },
        "render_environment": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
        },
        "outputs": {
            path.name: sha256(path)
            for path in sorted(output_paths, key=lambda item: item.name)
        },
    }
    path = output_dir / "figure_provenance.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate, sources = registered_sources(args.manifest)
    decision = load_json(sources["decision"])
    if decision.get("manuscript_or_figure_claim_direction") != "author_decision_required":
        raise ValueError("Decision evidence no longer requires author selection")

    rows = load_validated_metrics(sources["metrics"], sources["validation"])
    plot_rows = build_plot_data(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_data_path = write_plot_data(plot_rows, args.output_dir)
    output_paths = [plot_data_path, *draw_figure(rows, args.output_dir, args.dpi)]
    provenance_path = write_provenance(
        args.manifest, candidate, sources, decision,
        rows, plot_rows, output_paths, args.output_dir,
    )
    for path in [*output_paths, provenance_path]:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
