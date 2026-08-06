#!/usr/bin/env python3
"""Run the additive Study 01 mean/variance extreme-level extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import yaml

from run_methods import METHOD_CONFIG_KEYS, METHOD_SCRIPTS, method_metadata_is_valid
from score import METHODS, metrics, spatial_cache


EXPECTED_CONDITIONS = {
    ("mean_0_20", "mean", 0.2),
    ("mean_5", "mean", 5.0),
    ("variance_0_20", "variance", 0.2),
    ("variance_5", "variance", 5.0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    actual = {
        (str(row["simulation_id"]), str(row["alteration_type"]), float(row["value"]))
        for row in config["conditions"]
    }
    if actual != EXPECTED_CONDITIONS:
        raise RuntimeError(f"expanded condition contract mismatch: {actual}")
    if [str(value) for value in config["slices"]] != ["151508", "151670", "151676"]:
        raise RuntimeError("expanded sensitivity requires the three frozen Study 01 slices")
    feast = config["feast"]
    if (
        feast["spatial_mode"] != "reference_rank"
        or feast["assignment_solver"] != "scipy"
        or feast["assignment_blocks"] is not False
    ):
        raise RuntimeError("expanded sensitivity requires global SciPy reference-rank assignment")
    return config


def resolve_from_config(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.resolve().parent / path


def expected_keys(config: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(slice_id), str(condition["simulation_id"]))
        for slice_id in config["slices"]
        for condition in config["conditions"]
    }


def require_manifest(table: pd.DataFrame, config: dict[str, Any], description: str) -> None:
    keys = list(
        table[["slice_id", "simulation_id"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    if len(keys) != len(set(keys)) or set(keys) != expected_keys(config):
        raise RuntimeError(f"{description} does not contain the 12 unique extension jobs")
    if "status" in table and not table["status"].eq("ok").all():
        raise RuntimeError(f"{description} contains failed jobs")


def build_alteration(condition: dict[str, Any], config: dict[str, Any]):
    from FEAST import Alteration

    value = float(condition["value"])
    if condition["alteration_type"] == "mean":
        return Alteration.mean_only(
            fold_change=value,
            variance_coupling=config["mean_variance_coupling"],
        )
    return Alteration.variance_only(
        fold_change=value,
        dispersion=float(config["variance_heterogeneity_scale"]),
    )


def simulate(args: argparse.Namespace) -> int:
    import FEAST

    config = load_config(args.config)
    if str(FEAST.__version__) != "1.0.2":
        raise RuntimeError(f"unexpected FEAST version: {FEAST.__version__}")
    stage = args.output_root / "simulations"
    manifest_path = stage / "simulation_manifest.csv"
    previous: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume and manifest_path.is_file():
        old = pd.read_csv(manifest_path, dtype={"slice_id": str, "simulation_id": str})
        previous = {
            (row["slice_id"], row["simulation_id"]): row
            for row in old.to_dict("records")
        }
    elif stage.exists() and not args.resume:
        raise FileExistsError(stage)
    stage.mkdir(parents=True, exist_ok=True)
    config_hash = sha256(args.config)
    runner_hash = sha256(Path(__file__).resolve())
    input_checksums = pd.read_csv(
        resolve_from_config(args.config, config["base_artifacts"]["input_checksums"]),
        dtype={"slice_id": str},
    ).set_index("slice_id")
    rows: list[dict[str, Any]] = []
    total = len(expected_keys(config))
    index = 0
    for slice_value in config["slices"]:
        slice_id = str(slice_value)
        source = args.raw_dir / f"{slice_id}.h5ad"
        source_hash = sha256(source)
        if source_hash != input_checksums.loc[slice_id, "sha256"]:
            raise RuntimeError(f"raw slice checksum mismatch: {source}")
        reference = ad.read_h5ad(source)
        for condition in config["conditions"]:
            index += 1
            simulation_id = str(condition["simulation_id"])
            relative = Path("data") / slice_id / f"{simulation_id}.h5ad"
            output = stage / relative
            old = previous.get((slice_id, simulation_id), {})
            if (
                args.resume
                and old.get("status") == "ok"
                and old.get("config_sha256") == config_hash
                and old.get("runner_source_sha256") == runner_hash
                and old.get("input_sha256") == source_hash
                and output.is_file()
                and sha256(output) == old.get("output_sha256")
            ):
                rows.append(old)
                print(f"[{index}/{total}] {slice_id}/{simulation_id}: skipped verified", flush=True)
                continue
            if output.exists():
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                archived = stage / "failures" / slice_id / f"{simulation_id}__{stamp}.h5ad"
                archived.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(output), str(archived))
            started = time.time()
            simulated = FEAST.simulate(
                reference,
                seed=int(config["public_seed"]),
                alteration=build_alteration(condition, config),
                parameter_mode=config["feast"]["parameter_mode"],
                spatial_mode="reference_rank",
                annotation_key=config["annotation_key"],
                assignment_method=config["feast"]["assignment_method"],
                assignment_solver="scipy",
                assignment_blocks=False,
                use_distributional_alteration=True,
                ppf_method=config["feast"]["ppf_method"],
                beta_n_jobs=int(config["feast"]["beta_n_jobs"]),
                beta_early_stopping_patience=int(
                    config["feast"]["beta_early_stopping_patience"]
                ),
                convert_n_jobs=int(config["feast"]["convert_n_jobs"]),
                n_jobs=int(config["feast"]["n_jobs"]),
                use_heuristic_search=False,
                clip_overshoot_factor=0.0,
                boundary_multiplier=1.1,
                verbose=True,
            )
            diagnostics = simulated.uns.get("simulation_diagnostics", {})
            assignment = diagnostics.get("copula_rank_diagnostics", {})
            if diagnostics.get("spatial_mode") != "reference_rank":
                raise RuntimeError("FEAST returned the wrong spatial mode")
            if assignment.get("assignment_blocks") is not False:
                raise RuntimeError("FEAST did not use exact global assignment")
            simulated.uns["expanded_sensitivity_provenance"] = {
                "configuration_id": config["configuration_id"],
                "config_sha256": config_hash,
                "runner_source_sha256": runner_hash,
                "extends_configuration_id": config["extends_configuration_id"],
                "public_seed": int(config["public_seed"]),
                "feast_version": str(FEAST.__version__),
                "feast_commit": config["required_feast_commit"],
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            simulated.write_h5ad(output, compression="gzip")
            rows.append(
                {
                    "configuration_id": config["configuration_id"],
                    "config_sha256": config_hash,
                    "runner_source_sha256": runner_hash,
                    "slice_id": slice_id,
                    "simulation_id": simulation_id,
                    "alteration_type": condition["alteration_type"],
                    "alteration_value": float(condition["value"]),
                    "file_path": relative.as_posix(),
                    "input_sha256": source_hash,
                    "output_sha256": sha256(output),
                    "public_seed": config["public_seed"],
                    "feast_version": str(FEAST.__version__),
                    "feast_commit": config["required_feast_commit"],
                    "spatial_mode": "reference_rank",
                    "assignment_solver": "scipy",
                    "assignment_blocks": False,
                    "elapsed_seconds": round(time.time() - started, 2),
                    "status": "ok",
                }
            )
            pd.DataFrame(rows).to_csv(manifest_path, index=False)
            print(f"[{index}/{total}] {slice_id}/{simulation_id}: ok", flush=True)
    table = pd.DataFrame(rows)
    require_manifest(table, config, "simulation manifest")
    table.to_csv(manifest_path, index=False)
    (stage / "provenance.json").write_text(
        json.dumps(
            {
                "configuration_id": config["configuration_id"],
                "config_sha256": config_hash,
                "runner_source_sha256": runner_hash,
                "python": sys.version,
                "executable": sys.executable,
                "job_count": len(table),
                "manifest_sha256": sha256(manifest_path),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


def panels(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    stage = args.output_root / "fixed_panels"
    if stage.exists():
        raise FileExistsError(stage)
    simulations = pd.read_csv(
        args.output_root / "simulations" / "simulation_manifest.csv",
        dtype={"slice_id": str, "simulation_id": str},
    )
    require_manifest(simulations, config, "simulation manifest")
    base_panel = resolve_from_config(
        args.config, config["base_artifacts"]["fixed_panel_dir"]
    )
    rows: list[dict[str, Any]] = []
    for row in simulations.itertuples(index=False):
        source = args.output_root / "simulations" / row.file_path
        if sha256(source) != row.output_sha256:
            raise RuntimeError(f"simulation hash mismatch: {source}")
        panel_path = base_panel / "panels" / f"{row.slice_id}.csv"
        genes = pd.Index(pd.read_csv(panel_path)["gene_id"].astype(str))
        data = ad.read_h5ad(source)
        missing = genes.difference(data.var_names)
        if len(missing):
            raise RuntimeError(f"{row.slice_id}/{row.simulation_id} lacks panel genes")
        prepared = data[:, genes].copy()
        sc.pp.normalize_total(prepared, target_sum=float(config["fixed_panel"]["target_sum"]))
        sc.pp.log1p(prepared)
        prepared.uns["expanded_sensitivity_preprocessing"] = {
            "panel_path": str(panel_path),
            "panel_sha256": sha256(panel_path),
            "source_sha256": row.output_sha256,
            "preprocessing": "normalize_total(target_sum=10000) then log1p exactly once",
        }
        relative = Path("inputs") / row.slice_id / f"{row.simulation_id}.h5ad"
        output = stage / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        prepared.write_h5ad(output, compression="gzip")
        rows.append(
            {
                "slice_id": row.slice_id,
                "simulation_id": row.simulation_id,
                "alteration_type": row.alteration_type,
                "alteration_value": row.alteration_value,
                "file_path": relative.as_posix(),
                "source_sha256": row.output_sha256,
                "panel_sha256": sha256(panel_path),
                "output_sha256": sha256(output),
                "n_spots": prepared.n_obs,
                "n_genes": prepared.n_vars,
                "status": "ok",
            }
        )
    manifest = stage / "fixed_panel_manifest.csv"
    table = pd.DataFrame(rows)
    require_manifest(table, config, "fixed-panel manifest")
    table.to_csv(manifest, index=False)
    (stage / "provenance.json").write_text(
        json.dumps(
            {
                "configuration_id": config["configuration_id"],
                "job_count": len(table),
                "manifest_sha256": sha256(manifest),
                "base_panel_dir": str(base_panel),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


def canonical_parameters(settings: dict[str, Any]) -> str:
    return json.dumps(settings, sort_keys=True, separators=(",", ":"))


def method(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    stage = args.output_root / "methods" / args.method
    manifest_path = stage / "run_manifest.csv"
    previous: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume and manifest_path.is_file():
        old = pd.read_csv(
            manifest_path,
            dtype={"slice_id": str, "simulation_id": str},
            keep_default_na=False,
        )
        previous = {
            (row["slice_id"], row["simulation_id"]): row
            for row in old.to_dict("records")
        }
    elif stage.exists() and not args.resume:
        raise FileExistsError(stage)
    stage.mkdir(parents=True, exist_ok=True)
    panels_table = pd.read_csv(
        args.output_root / "fixed_panels" / "fixed_panel_manifest.csv",
        dtype={"slice_id": str, "simulation_id": str},
    )
    require_manifest(panels_table, config, "fixed-panel manifest")
    script = (Path(__file__).parent / "methods" / METHOD_SCRIPTS[args.method]).resolve()
    settings = config["methods"][METHOD_CONFIG_KEYS[args.method]]
    python = args.python.resolve(strict=True)
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "") if args.method == "STAGATE_mclust" else ""
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(panels_table.itertuples(index=False), start=1):
        source = args.output_root / "fixed_panels" / row.file_path
        source_hash = sha256(source)
        output = stage / row.slice_id / row.simulation_id
        old = previous.get((row.slice_id, row.simulation_id), {})
        artifacts = {
            "clusters_sha256": output / "clusters.csv",
            "metadata_sha256": output / "metadata.json",
            "result_h5ad_sha256": output / "result.h5ad",
            "runner_log_sha256": output / "runner.log",
        }
        if (
            args.resume
            and old.get("status") == "ok"
            and old.get("input_sha256") == source_hash
            and all(path.is_file() and sha256(path) == old.get(field) for field, path in artifacts.items())
            and method_metadata_is_valid(
                output / "metadata.json",
                args.method,
                settings,
                int(config["public_seed"]),
                ld_library_path,
            )[0]
        ):
            rows.append(old)
            print(f"[{index}/12] {args.method} {row.slice_id}/{row.simulation_id}: skipped verified", flush=True)
            continue
        if output.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            archived = args.output_root / "failures" / args.method / row.slice_id / f"{row.simulation_id}__{stamp}"
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(output), str(archived))
        output.mkdir(parents=True, exist_ok=False)
        command = [
            str(python), str(script), "--input", str(source), "--output-dir", str(output),
            "--seed", str(config["public_seed"]),
        ]
        if args.method == "GraphST":
            command += ["--device", str(settings["device"]), "--epochs", str(settings["epochs"]), "--radius", str(settings["radius"])]
        elif args.method == "STAGATE_mclust":
            command += ["--rad-cutoff", str(settings["rad_cutoff"]), "--n-epochs", str(settings["n_epochs"])]
        else:
            command += ["--resolutions", ",".join(map(str, settings["resolutions"])), "--n-pcs", str(settings["n_pcs"]), "--n-neighbors", str(settings["n_neighbors"])]
        started = time.time()
        completed = subprocess.run(command, text=True, capture_output=True)
        runner_log = output / "runner.log"
        runner_log.write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr)
        metadata_valid, metadata_error = method_metadata_is_valid(
            output / "metadata.json", args.method, settings, int(config["public_seed"]), ld_library_path
        )
        files_complete = all(path.is_file() for path in list(artifacts.values())[:-1])
        ok = completed.returncode == 0 and files_complete and metadata_valid
        metadata: dict[str, Any] = {}
        if (output / "metadata.json").is_file():
            metadata = json.loads((output / "metadata.json").read_text())
        rows.append(
            {
                "configuration_id": config["configuration_id"],
                "slice_id": row.slice_id,
                "simulation_id": row.simulation_id,
                "method": args.method,
                "method_parameters_json": canonical_parameters(settings),
                "method_python": str(python),
                "input_sha256": source_hash,
                **{field: sha256(path) if path.is_file() else "" for field, path in artifacts.items()},
                "public_seed": config["public_seed"],
                "cuda_execution_verified": metadata.get("cuda_execution_verified", ""),
                "returncode": completed.returncode,
                "elapsed_seconds": round(time.time() - started, 2),
                "validation_error": "" if ok else metadata_error or f"method process returned {completed.returncode}",
                "status": "ok" if ok else "failed",
            }
        )
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        print(f"[{index}/12] {args.method} {row.slice_id}/{row.simulation_id}: {rows[-1]['status']}", flush=True)
    table = pd.DataFrame(rows)
    require_manifest(table, config, f"{args.method} run manifest")
    table.to_csv(manifest_path, index=False)
    (stage / "provenance.json").write_text(
        json.dumps(
            {
                "configuration_id": config["configuration_id"],
                "method": args.method,
                "method_parameters": settings,
                "method_python": str(python),
                "wrapper_source_sha256": sha256(script),
                "job_count": len(table),
                "successful_jobs": int(table.status.eq("ok").sum()),
                "run_manifest_sha256": sha256(manifest_path),
                **({"ld_library_path": ld_library_path} if args.method == "STAGATE_mclust" else {}),
            },
            indent=2,
        )
        + "\n"
    )
    return 0 if table.status.eq("ok").all() else 1


def score(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    report = args.output_root / "report"
    if report.exists():
        raise FileExistsError(report)
    panels_table = pd.read_csv(
        args.output_root / "fixed_panels" / "fixed_panel_manifest.csv",
        dtype={"slice_id": str, "simulation_id": str},
    )
    require_manifest(panels_table, config, "fixed-panel manifest")
    method_runs = {}
    for method_name in METHODS:
        table = pd.read_csv(
            args.output_root / "methods" / method_name / "run_manifest.csv",
            dtype={"slice_id": str, "simulation_id": str},
        )
        require_manifest(table, config, f"{method_name} run manifest")
        method_runs[method_name] = table.set_index(["slice_id", "simulation_id"])
    rows: list[dict[str, Any]] = []
    for panel_row in panels_table.itertuples(index=False):
        input_path = args.output_root / "fixed_panels" / panel_row.file_path
        if sha256(input_path) != panel_row.output_sha256:
            raise RuntimeError(f"fixed-panel hash mismatch: {input_path}")
        data = ad.read_h5ad(input_path)
        truth = data.obs[config["annotation_key"]].astype(str).to_numpy()
        distances, neighbors = spatial_cache(np.asarray(data.obsm["spatial"]))
        for method_name in METHODS:
            output = args.output_root / "methods" / method_name / panel_row.slice_id / panel_row.simulation_id
            run = method_runs[method_name].loc[(panel_row.slice_id, panel_row.simulation_id)]
            cluster_path = output / "clusters.csv"
            if sha256(cluster_path) != run.clusters_sha256:
                raise RuntimeError(f"cluster hash mismatch: {cluster_path}")
            clusters = pd.read_csv(cluster_path, dtype={"spot_barcode": str}).set_index("spot_barcode")
            if not clusters.index.equals(data.obs_names):
                raise RuntimeError(f"spot order mismatch: {cluster_path}")
            labels = clusters["predicted_cluster"].astype(str).to_numpy()
            rows.append(
                {
                    "slice_id": panel_row.slice_id,
                    "simulation_id": panel_row.simulation_id,
                    "alteration_type": panel_row.alteration_type,
                    "alteration_value": float(panel_row.alteration_value),
                    "method": method_name,
                    **metrics(truth, labels, distances, neighbors),
                    "status": "ok",
                }
            )
    report.mkdir(parents=True, exist_ok=False)
    expanded = pd.DataFrame(rows)
    expanded_path = report / "expanded_benchmark_metrics.csv"
    expanded.to_csv(expanded_path, index=False)
    base_metrics = pd.read_csv(resolve_from_config(args.config, config["base_artifacts"]["benchmark_metrics"]), dtype={"slice_id": str})
    base_manifest = pd.read_csv(resolve_from_config(args.config, config["base_artifacts"]["simulation_manifest"]), dtype={"slice_id": str})
    values = base_manifest[["simulation_id", "alteration_value"]].drop_duplicates()
    base = base_metrics[base_metrics.alteration_type.isin(["mean", "variance", "baseline"])].merge(values, on="simulation_id", how="left")
    base["alteration_value"] = base["alteration_value"].fillna(1.0)
    combined = pd.concat([base, expanded], ignore_index=True, sort=False)
    combined_path = report / "combined_mean_variance_metrics.csv"
    combined.to_csv(combined_path, index=False)
    metric_columns = ["ARI", "NMI", "AMI", "CHAOS", "PAS"]
    summary = combined.groupby(["alteration_type", "alteration_value", "method"], as_index=False)[metric_columns].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(str(part) for part in column if part).rstrip("_") if isinstance(column, tuple) else column for column in summary.columns]
    summary_path = report / "combined_mean_variance_summary.csv"
    summary.to_csv(summary_path, index=False)
    (report / "provenance.json").write_text(
        json.dumps(
            {
                "configuration_id": config["configuration_id"],
                "expanded_row_count": len(expanded),
                "combined_row_count": len(combined),
                "outputs": {path.name: sha256(path) for path in (expanded_path, combined_path, summary_path)},
            },
            indent=2,
        )
        + "\n"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("expanded_config.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    simulation = subparsers.add_parser("simulate")
    simulation.add_argument("--raw-dir", type=Path, required=True)
    simulation.add_argument("--resume", action="store_true")
    subparsers.add_parser("panels")
    method_parser = subparsers.add_parser("method")
    method_parser.add_argument("--method", choices=METHOD_SCRIPTS, required=True)
    method_parser.add_argument("--python", type=Path, required=True)
    method_parser.add_argument("--resume", action="store_true")
    subparsers.add_parser("score")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return {"simulate": simulate, "panels": panels, "method": method, "score": score}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
