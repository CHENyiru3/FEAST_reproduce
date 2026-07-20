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


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "PUBLICATION_MANIFEST.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

METHODS = ("paste", "spateo")
METHOD_LABELS = {"paste": "PASTE", "spateo": "Spateo"}
METHOD_COLORS = {"paste": "#287D8E", "spateo": "#D77A32"}
METHOD_MARKERS = {"paste": "o", "spateo": "s"}
ALTERATIONS = ("baseline", "mean_0.5", "variance_2.0", "sparsity_0.5")
ALTERATION_LABELS = {
    "baseline": "Baseline",
    "mean_0.5": "Mean x0.5",
    "variance_2.0": "Variance x2.0",
    "sparsity_0.5": "Sparsity x0.5",
}
ANGLES = (1.0, 5.0, 10.0, 30.0, 45.0)
METRICS = (
    {
        "key": "rotation_recovery_error",
        "label": "Rotation recovery error (degrees)",
        "direction": "lower_is_better",
        "scale": "symlog",
        "linthresh": 1e-10,
    },
    {
        "key": "mean_spatial_error",
        "label": "Mean spatial error (coordinate units)",
        "direction": "lower_is_better",
        "scale": "symlog",
        "linthresh": 1e-9,
    },
    {
        "key": "transport_argmax_identity_accuracy",
        "label": "Transport argmax identity accuracy",
        "direction": "higher_is_better",
        "scale": "linear",
    },
)


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
        "job_id",
        "method",
        "alteration",
        "angle_degrees",
        "status",
        "canonical_candidate",
        "solver_evidence_disposition",
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
    score_table_valid = any(
        row.get("kind") == "score_table"
        and row.get("job_id") == "alignment_metrics"
        and row.get("valid") == "True"
        for row in validation_rows
    )

    rows: list[dict[str, object]] = []
    seen_jobs: set[str] = set()
    observed_cells: set[tuple[str, str, float]] = set()
    for source in source_rows:
        job_id = source["job_id"]
        if job_id in seen_jobs:
            raise ValueError(f"Duplicate alignment job: {job_id}")
        if source["status"] != "ok" or source["canonical_candidate"] != "True":
            raise ValueError(f"Nonvalidated/noncandidate row encountered: {job_id}")
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
            "job_id": job_id,
            "method": method,
            "alteration": alteration,
            "angle_degrees": angle,
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
        for method in METHODS
        for alteration in ALTERATIONS
        for angle in ANGLES
    }
    if observed_cells != expected_cells or validated_jobs != seen_jobs or not score_table_valid:
        raise ValueError("Study 02 validation/design coverage is not exact")
    return rows


def build_plot_data(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    plot_rows: list[dict[str, object]] = []
    for alteration in ALTERATIONS:
        for metric in METRICS:
            key = str(metric["key"])
            for method in METHODS:
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["alteration"] == alteration and row["method"] == method
                    ),
                    key=lambda row: float(row["angle_degrees"]),
                )
                for row in selected:
                    plot_rows.append(
                        {
                            "job_id": row["job_id"],
                            "method": method,
                            "alteration": alteration,
                            "angle_degrees": row["angle_degrees"],
                            "metric": key,
                            "metric_label": metric["label"],
                            "direction": metric["direction"],
                            "value": row[key],
                            "solver_evidence_disposition": row[
                                "solver_evidence_disposition"
                            ],
                        }
                    )
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
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "feast-study02-alignment-diagnostic",
        }
    )


def draw_figure(rows: list[dict[str, object]], output_dir: Path, dpi: int) -> list[Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(
        len(ALTERATIONS),
        len(METRICS),
        figsize=(12.8, 11.2),
        sharex=True,
        squeeze=False,
    )

    for row_index, alteration in enumerate(ALTERATIONS):
        for column_index, metric in enumerate(METRICS):
            ax = axes[row_index][column_index]
            key = str(metric["key"])
            for method in METHODS:
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["alteration"] == alteration and row["method"] == method
                    ),
                    key=lambda row: float(row["angle_degrees"]),
                )
                ax.plot(
                    [float(row["angle_degrees"]) for row in selected],
                    [float(row[key]) for row in selected],
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    markersize=4.5,
                    linewidth=1.35,
                    label=METHOD_LABELS[method],
                )

            if metric["scale"] == "symlog":
                ax.set_yscale("symlog", linthresh=float(metric["linthresh"]), linscale=0.6)
            else:
                ax.set_ylim(0.88, 1.01)
            ax.set_xticks(ANGLES)
            ax.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.8)
            ax.tick_params(labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#666666")
                spine.set_linewidth(0.7)

            if row_index == 0:
                ax.set_title(str(metric["label"]), fontsize=10.5, fontweight="bold", pad=7)
            if column_index == 0:
                ax.set_ylabel(
                    f"{ALTERATION_LABELS[alteration]}\nMetric value",
                    fontsize=9.5,
                    fontweight="bold",
                )
            if row_index == len(ALTERATIONS) - 1:
                ax.set_xlabel("Input rotation (degrees)", fontsize=9.5)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(
        "Study 02 alignment sensitivity diagnostics (unpromoted)",
        fontsize=14,
        fontweight="bold",
        y=0.992,
    )
    fig.text(
        0.5,
        0.012,
        "Raw validated rows only; no cross-alteration aggregation or overall ranking. "
        "AUTHOR SELECTION REQUIRED.",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#7A2E2E",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.025, 0.045, 0.99, 0.935), h_pad=1.35, w_pad=1.0)

    stem = "alignment_sensitivity_diagnostic"
    output_paths: list[Path] = []
    formats = {
        "png": {"dpi": dpi, "metadata": {"Software": "FEAST Study 02 plot.py"}},
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
    manifest_path: Path,
    candidate: dict[str, object],
    sources: dict[str, Path],
    decision: dict[str, object],
    rows: list[dict[str, object]],
    plot_rows: list[dict[str, object]],
    output_paths: list[Path],
    output_dir: Path,
) -> Path:
    payload = {
        "schema_version": 1,
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
            "methods": list(METHODS),
            "alterations": list(ALTERATIONS),
            "angles_degrees": list(ANGLES),
            "metrics": [
                {
                    "key": metric["key"],
                    "label": metric["label"],
                    "direction": metric["direction"],
                }
                for metric in METRICS
            ],
            "aggregation": "none; every plotted point is one validated method job",
        },
        "limitations": {
            "manuscript_or_figure_claim_direction": decision[
                "manuscript_or_figure_claim_direction"
            ],
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
    parser.add_argument("--dpi", type=int, default=300)
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
        args.manifest,
        candidate,
        sources,
        decision,
        rows,
        plot_rows,
        output_paths,
        args.output_dir,
    )
    for path in [*output_paths, provenance_path]:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
