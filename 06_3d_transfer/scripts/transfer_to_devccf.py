#!/usr/bin/env python
"""transfer_to_devccf.py -- Transfer GSE269617 reference expression onto DevCCF 3D coordinate blueprints.

This is the main pipeline script that uses FEAST de novo conditional generation to
synthesise MERFISH-like expression profiles at every z-level of a DevCCF blueprint,
producing a full 3D spatial transcriptomics volume.

Pipeline
--------
1. Load reference slices from GSE269617 region-annotated h5ad files.
2. Fit a SimulationReference model from all loaded reference slices.
3. Load the DevCCF blueprint JSON (one entry per z-level).
4. For each z-level: simulate from reference, assign 3D coordinates, save.
5. Write a manifest CSV mapping z_world -> output file path.

Usage
-----
    python transfer_to_devccf.py \
        --references /path/to/h5ad_region_annotated/ \
        --reference-pattern "E14*" \
        --blueprints outputs/E15.5_blueprints.json \
        --label-key region \
        --output-dir outputs/E15.5_generated/ \
        --seed 2026 \
        --z-subsample 5

By default, ``assignment_randomness`` is auto-selected once from the fitted
reference cohort and then held fixed for every generated z-level in the volume.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure FEAST is on the import path
# ---------------------------------------------------------------------------
_FEAST_SRC = Path("/maiziezhou_lab2/yiru/FEAST/src")
if str(_FEAST_SRC) not in sys.path:
    sys.path.insert(0, str(_FEAST_SRC))

from FEAST.de_novo.conditional import fit_reference, simulate_from_reference  # noqa: E402
from FEAST import ReferenceFitConfig, SimulationConfig  # noqa: E402
from FEAST import estimate_assignment_randomness  # noqa: E402
from FEAST import SliceBlueprint  # noqa: E402
from FEAST.de_novo.core import assign_generated_coordinates  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("transfer_to_devccf")

# Known region labels in GSE269617 (all 9).  "Other" is kept in references
# for parameter fitting but is masked out of blueprints by the extractor.
ALL_REGIONS = [
    "BT", "CP", "DorsalVZ", "GE", "IZ",
    "Other", "SeptalVZ", "Septum", "VentralVZ",
]

# ---------------------------------------------------------------------------
# Step 1 -- Load reference slices
# ---------------------------------------------------------------------------

def _set_reference_name(adata: ad.AnnData) -> ad.AnnData:
    """Store ``sample_id`` as ``uns['reference_name']`` so ``fit_reference``
    can pick it up without requiring extra columns."""
    if "sample_id" in adata.obs:
        adata.uns["reference_name"] = str(adata.obs["sample_id"].iloc[0])
    elif "reference_name" not in adata.uns:
        adata.uns["reference_name"] = "unnamed_reference"
    return adata


def _subsample_spots(adata: ad.AnnData, max_spots: int, seed: int) -> ad.AnnData:
    """Randomly subsample spots if the slice exceeds *max_spots*."""
    if adata.n_obs <= max_spots:
        return adata
    rng = np.random.default_rng(seed)
    indices = rng.choice(adata.n_obs, size=max_spots, replace=False)
    indices = np.sort(indices)
    logger.info(
        "Subsampled %s from %d -> %d spots.",
        adata.uns.get("reference_name", "?"),
        adata.n_obs,
        max_spots,
    )
    return adata[indices, :].copy()


def _collect_h5ad_paths(reference_dir: str, pattern: str) -> List[Path]:
    """Return sorted list of h5ad paths whose stem contains *pattern*.

    Because GSE269617 filenames are prefixed with GSM accession IDs
    (e.g. ``GSM8323035_E14M_1.h5ad``), the pattern is matched as a
    *substring* rather than a strict fnmatch from the start.  Patterns
    already beginning with a wildcard are used as-is for fnmatch.
    """
    base = Path(reference_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {base}")
    paths = sorted(base.glob("*.h5ad"))
    if not paths:
        raise FileNotFoundError(f"No .h5ad files found in {base}")
    if pattern in {"", "*"}:
        return paths
    # If the pattern already contains a leading wildcard use fnmatch as-is.
    if pattern[0] in ("*", "?", "["):
        matched = [p for p in paths if fnmatch.fnmatch(p.stem, pattern)]
    else:
        # Substring match: the pattern should appear somewhere in the stem.
        matched = [p for p in paths if fnmatch.fnmatch(p.stem, f"*{pattern}")]
    if not matched:
        raise FileNotFoundError(
            f"No .h5ad files matched pattern '{pattern}' in {base}"
        )
    return matched


def load_reference_slices(
    reference_dir: str,
    pattern: str,
    label_key: str = "region",
    max_reference_spots: Optional[int] = None,
    seed: int = 2026,
) -> List[ad.AnnData]:
    """Load and validate GSE269617 reference slices.

    Parameters
    ----------
    reference_dir : str
        Directory containing region-annotated h5ad files.
    pattern : str
        fnmatch pattern for filenames (e.g. ``"E14*"``, ``"E18M*"``).
    label_key : str
        Observation column containing region labels (default ``"region"``).
    max_reference_spots : int, optional
        If set, subsample slices that exceed this many spots.
    seed : int
        Random seed for subsampling.

    Returns
    -------
    List of AnnData objects ready for ``fit_reference``.
    """
    paths = _collect_h5ad_paths(reference_dir, pattern)
    logger.info("Found %d reference h5ad file(s) matching '%s'.", len(paths), pattern)

    slices: List[ad.AnnData] = []
    for p in paths:
        logger.info("  Loading %s ...", p.name)
        adata = ad.read_h5ad(p)
        _set_reference_name(adata)

        # Validate label column
        if label_key not in adata.obs:
            raise KeyError(
                f"{p.name} is missing label_key '{label_key}'. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        labels = adata.obs[label_key].astype(str)
        unknown = sorted(set(labels.unique()) - set(ALL_REGIONS))
        if unknown:
            raise ValueError(
                f"{p.name} contains unknown region labels: {unknown}. "
                f"Known labels: {ALL_REGIONS}"
            )

        # Subsample if needed
        if max_reference_spots is not None:
            adata = _subsample_spots(adata, max_reference_spots, seed + len(slices))

        slices.append(adata)

    # Cross-validate gene intersection
    gene_sets = [set(map(str, s.var_names)) for s in slices]
    common = gene_sets[0]
    for gs in gene_sets[1:]:
        common &= gs
    logger.info(
        "Loaded %d slices, %d common genes across all slices.",
        len(slices),
        len(common),
    )
    for i, (p, s) in enumerate(zip(paths, slices)):
        logger.info(
            "  [%d] %s  spots=%d  genes=%d  name=%s",
            i,
            p.name,
            s.n_obs,
            s.n_vars,
            s.uns.get("reference_name", "?"),
        )

    return slices


# ---------------------------------------------------------------------------
# Step 3 -- Load blueprints
# ---------------------------------------------------------------------------

def _validate_blueprint_entry(entry: Dict[str, Any], idx: int) -> None:
    """Quick sanity checks on a single blueprint entry."""
    required = {"x", "y", "region", "z_world"}
    missing = required - set(entry.keys())
    if missing:
        raise KeyError(f"Blueprint entry {idx} is missing keys: {sorted(missing)}")
    n = len(entry["x"])
    if len(entry["y"]) != n:
        raise ValueError(
            f"Blueprint entry {idx}: 'x' and 'y' lengths differ ({len(entry['x'])} vs {len(entry['y'])})"
        )
    if len(entry["region"]) != n:
        raise ValueError(
            f"Blueprint entry {idx}: 'region' length ({len(entry['region'])}) "
            f"does not match coordinate count ({n})"
        )


def load_blueprints(blueprint_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load and validate the DevCCF blueprint JSON.

    Supports two JSON layouts:

    1.  Dict-of-entries (produced by extract_devccf_blueprints.py)::

            {"metadata": {...}, "blueprints": {"-1.20": {...}, ...}}

    2.  List-of-entries::

            {"metadata": {...}, "slices": [{...}, ...]}

    Returns
    -------
    metadata : dict
        Top-level metadata (age, stage, etc.)
    entries : list of dict
        One dict per z-level with keys ``x``, ``y``, ``region``, ``z_world``,
        ``voxel_j``.  Sorted by z_world ascending.
    """
    path = Path(blueprint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Blueprint JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    metadata = data.get("metadata", {})
    raw_blueprints = data.get("blueprints", data.get("slices", data.get("entries", None)))

    if raw_blueprints is None:
        raise ValueError(
            "Blueprint JSON must contain a 'blueprints', 'slices', or 'entries' key."
        )

    # Normalise to a list of entry dicts
    if isinstance(raw_blueprints, dict):
        # Dict-of-entries format: {"-1.20": {...}, ...}
        entries: List[Dict[str, Any]] = []
        for key, value in raw_blueprints.items():
            if isinstance(value, dict):
                entries.append(value)
            elif isinstance(value, (list, str)):
                # Nested list-of-entries or string reference; skip non-dict values
                logger.debug("Skipping non-dict blueprint key %r", key)
    elif isinstance(raw_blueprints, list):
        entries = list(raw_blueprints)
    else:
        raise TypeError(
            f"Unsupported blueprints type: {type(raw_blueprints).__name__}"
        )

    if not entries:
        raise ValueError("Blueprint JSON contains no slice entries.")

    for i, entry in enumerate(entries):
        _validate_blueprint_entry(entry, i)

    # Sort by z_world
    entries.sort(key=lambda e: float(e["z_world"]))

    logger.info(
        "Loaded blueprint: age=%s, %d z-levels (z range %.3f .. %.3f).",
        metadata.get("age", "unknown"),
        len(entries),
        float(entries[0]["z_world"]),
        float(entries[-1]["z_world"]),
    )

    # Log region presence across z-levels
    all_regions: set[str] = set()
    for e in entries:
        all_regions.update(e["region"])
    logger.info("Blueprint region labels: %s", sorted(all_regions))

    return metadata, entries


def entry_to_blueprint(entry: Dict[str, Any], metadata: Dict[str, Any]) -> SliceBlueprint:
    """Convert a single blueprint JSON entry into a SliceBlueprint."""
    coords = np.column_stack([
        np.asarray(entry["x"], dtype=float),
        np.asarray(entry["y"], dtype=float),
    ])
    regions = [str(r) for r in entry["region"]]

    return SliceBlueprint(
        coordinates=coords,
        grid_type="devccf_coronal",
        domain_map=regions,
        technology="MERFISH",
        obs=pd.DataFrame({"domain": regions}),
        metadata={
            "age": metadata.get("age", "unknown"),
            "z_world": float(entry["z_world"]),
            "voxel_j": int(entry.get("voxel_j", -1)),
        },
    )


# ---------------------------------------------------------------------------
# Step 5 -- Assign 3D coordinates
# ---------------------------------------------------------------------------

def _assign_3d(result: ad.AnnData, blueprint: SliceBlueprint, z_world: float) -> None:
    """Attach 2D + 3D spatial coordinates and z metadata."""
    xyz = np.column_stack([
        blueprint.coordinates,
        np.full(blueprint.n_active_spots, z_world, dtype=float),
    ])
    assign_generated_coordinates(result, xyz, z_value=z_world, coordinate_system="devccf_world")
    logger.debug("Assigned 3D coordinates (z=%.3f) to %d spots.", z_world, result.n_obs)


# ---------------------------------------------------------------------------
# Step 6 -- Save and manifest
# ---------------------------------------------------------------------------

def _z_filename(z_world: float) -> str:
    """Format z_world as a filename-safe string, e.g. '-0.560.h5ad'."""
    return f"{z_world:+.3f}.h5ad"


def _clean_value(value: Any) -> Any:
    """Recursively sanitize a value for h5ad storage."""
    if isinstance(value, dict):
        return {str(k): _clean_value(v) for k, v in value.items()}
    if isinstance(value, list):
        items = [_clean_value(v) for v in value]
        if _is_h5ad_scalar_list_safe(items):
            return items
        return json.dumps(_json_ready(value), sort_keys=True, allow_nan=False)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"O", "U"}:
            return _clean_value(value.tolist())
        return value
    if isinstance(value, tuple):
        return _clean_value(list(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else "null"
    if isinstance(value, (str, int, float, bool, type(None))):
        if isinstance(value, float) and not np.isfinite(value):
            return "null"
        return value
    return str(value)


def _is_h5ad_scalar(value: Any) -> bool:
    return isinstance(value, (str, bytes, bool, int, float, np.integer, np.floating, np.bool_))


def _is_h5ad_scalar_list_safe(items: Sequence[Any]) -> bool:
    if not all(_is_h5ad_scalar(item) for item in items):
        return False
    if not items:
        return True
    if all(isinstance(item, (str, bytes)) for item in items):
        return True
    if all(isinstance(item, (bool, np.bool_)) for item in items):
        return True
    return all(isinstance(item, (int, float, np.integer, np.floating, bool, np.bool_)) for item in items)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _json_ready(float(value))
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool, type(None))):
        return value
    return str(value)


def _sanitize_uns(adata: ad.AnnData) -> None:
    """Remove uns entries that can't be stored in h5ad."""
    uns_clean = {}
    for key, value in adata.uns.items():
        try:
            uns_clean[key] = _clean_value(value)
        except Exception:
            uns_clean[key] = str(value)
    adata.uns = uns_clean


def save_result(
    result: ad.AnnData,
    z_world: float,
    output_dir: str,
) -> Path:
    """Save a single z-level AnnData and return the output path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = _z_filename(z_world)
    out_path = out_dir / fname
    _sanitize_uns(result)
    result.write_h5ad(out_path)
    logger.info("Saved z=%+.3f -> %s  (%d spots, %d genes)", z_world, out_path, result.n_obs, result.n_vars)
    return out_path


def write_manifest(records: List[Dict[str, Any]], output_dir: str) -> None:
    """Write manifest.csv mapping z_world to file path and spot count."""
    out_dir = Path(output_dir)
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["z_world", "filename", "n_spots", "z_idx", "assignment_randomness"],
        )
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    logger.info("Manifest written -> %s  (%d entries)", manifest_path, len(records))


def write_transfer_metadata(payload: Dict[str, Any], output_dir: str) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Metadata written -> %s", metadata_path)


def select_assignment_randomness(
    model: Any,
    label_key: str,
    *,
    requested: str,
    seed: int,
    n_genes: int,
    ar_step: float,
    max_ar: float,
    n_neighbors: int,
) -> Tuple[float, Dict[str, Any]]:
    value_text = str(requested).strip().lower()
    if value_text not in {"", "auto"}:
        value = float(value_text)
        if not 0.0 <= value <= 1.0:
            raise ValueError("--assignment-randomness must be 'auto' or a value in [0, 1].")
        return value, {
            "mode": "manual",
            "assignment_randomness": value,
            "requested": str(requested),
        }

    value = estimate_assignment_randomness(
        model,
        label_key=label_key,
        n_genes=int(n_genes),
        ar_step=float(ar_step),
        max_ar=float(max_ar),
        n_neighbors=int(n_neighbors),
        random_seed=int(seed),
    )
    return float(value), {
        "mode": "auto_reference_mini_sweep",
        "assignment_randomness": float(value),
        "n_genes": int(n_genes),
        "ar_step": float(ar_step),
        "max_ar": float(max_ar),
        "n_neighbors": int(n_neighbors),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def transfer(
    reference_dir: str,
    reference_pattern: str,
    blueprint_path: str,
    label_key: str = "region",
    output_dir: str = "outputs/generated/",
    seed: int = 2026,
    z_subsample: int = 1,
    limit: Optional[int] = None,
    max_reference_spots: Optional[int] = None,
    fit_config: Optional[ReferenceFitConfig] = None,
    sim_config: Optional[SimulationConfig] = None,
    assignment_randomness: str = "auto",
    ar_estimate_genes: int = 20,
    ar_step: float = 0.05,
    ar_max: float = 0.5,
    ar_neighbors: int = 6,
) -> None:
    """Run the full 3D transfer pipeline.

    Parameters
    ----------
    reference_dir : str
        Directory of region-annotated GSE269617 h5ad files.
    reference_pattern : str
        fnmatch pattern to select reference slices.
    blueprint_path : str
        Path to the DevCCF blueprint JSON.
    label_key : str
        Observation column for region labels.
    output_dir : str
        Directory to write generated h5ad files + manifest.
    seed : int
        Base random seed (incremented per z-level).
    z_subsample : int
        Process every Nth z-level (1 = all).
    limit : int, optional
        Stop after N z-levels (useful for testing).
    max_reference_spots : int, optional
        If set, subsample reference slices to this many spots.
    fit_config : ReferenceFitConfig, optional
        Override default reference fitting parameters.
    sim_config : SimulationConfig, optional
        Override default simulation parameters.
    assignment_randomness : str
        "auto" or a fixed numeric value for conditional assignment randomness.
    """
    # ---- Step 1: Load references ----
    logger.info("=" * 60)
    logger.info("STEP 1: Loading reference slices")
    logger.info("=" * 60)
    reference_slices = load_reference_slices(
        reference_dir,
        reference_pattern,
        label_key=label_key,
        max_reference_spots=max_reference_spots,
        seed=seed,
    )

    # ---- Step 2: Fit reference model ----
    logger.info("=" * 60)
    logger.info("STEP 2: Fitting reference model")
    logger.info("=" * 60)
    fit_cfg = fit_config or ReferenceFitConfig(
        min_gene_spots=1,
        min_gene_mean=0.0,
        max_gene_zero_prop=1.0,
        coordinate_scale=None,
    )
    model = fit_reference(reference_slices, label_key=label_key, config=fit_cfg)
    logger.info(
        "Reference model: %d genes, %d labels, %d reference slices, coord_dim=%d.",
        len(model.gene_names),
        len(model.reference_metadata.get("labels", [])),
        len(model.references),
        model.coordinate_dim,
    )
    logger.info("Model labels: %s", model.reference_metadata.get("labels", []))

    logger.info("=" * 60)
    logger.info("STEP 2b: Selecting assignment_randomness")
    logger.info("=" * 60)
    selected_ar, ar_metadata = select_assignment_randomness(
        model,
        label_key,
        requested=assignment_randomness,
        seed=seed,
        n_genes=ar_estimate_genes,
        ar_step=ar_step,
        max_ar=ar_max,
        n_neighbors=ar_neighbors,
    )
    logger.info(
        "Selected assignment_randomness=%.3g (%s).",
        selected_ar,
        ar_metadata.get("mode"),
    )

    # ---- Step 3: Load blueprints ----
    logger.info("=" * 60)
    logger.info("STEP 3: Loading blueprints")
    logger.info("=" * 60)
    metadata, entries = load_blueprints(blueprint_path)

    # Filter by subsample / limit
    selected_entries = entries[::z_subsample]
    if limit is not None:
        selected_entries = selected_entries[:limit]
    logger.info(
        "Processing %d / %d z-levels (subsample=%d, limit=%s).",
        len(selected_entries),
        len(entries),
        z_subsample,
        limit,
    )

    # ---- Step 4: Generate per-z slices ----
    logger.info("=" * 60)
    logger.info("STEP 4: Generating per-z slices")
    logger.info("=" * 60)
    sim_cfg_base = sim_config or SimulationConfig(
        coordinate_scale=None,
        verbose=False,
    )
    sim_cfg = replace(sim_cfg_base, assignment_randomness=float(selected_ar))

    transfer_metadata = {
        "reference_dir": str(reference_dir),
        "reference_pattern": str(reference_pattern),
        "blueprint_path": str(blueprint_path),
        "label_key": str(label_key),
        "seed": int(seed),
        "z_subsample": int(z_subsample),
        "limit": None if limit is None else int(limit),
        "max_reference_spots": None if max_reference_spots is None else int(max_reference_spots),
        "n_reference_slices": int(len(reference_slices)),
        "n_reference_genes": int(len(model.gene_names)),
        "reference_metadata": dict(model.reference_metadata),
        "assignment_randomness": float(selected_ar),
        "assignment_randomness_selection": dict(ar_metadata),
    }

    manifest_records: List[Dict[str, Any]] = []
    for z_idx, entry in enumerate(selected_entries):
        z_world = float(entry["z_world"])
        logger.info("--- z_idx=%d  z_world=%+.3f  voxel_j=%d ---", z_idx, z_world, entry.get("voxel_j", -1))

        blueprint = entry_to_blueprint(entry, metadata)
        logger.info(
            "Blueprint: %d total spots, %d active spots.",
            blueprint.n_spots,
            blueprint.n_active_spots,
        )

        result = simulate_from_reference(
            model,
            blueprint,
            parameter_cloud=None,
            config=sim_cfg,
            random_seed=seed + z_idx,
            reference_weights=None,
        )

        # Assign 3D coordinates
        _assign_3d(result, blueprint, z_world)
        result.uns.setdefault("de_novo", {}).setdefault("experiment", {})
        result.uns["de_novo"]["experiment"].update(
            {
                "name": "gse269617_to_devccf_3d_transfer",
                "reference_pattern": str(reference_pattern),
                "blueprint_age": metadata.get("age", "unknown"),
                "z_world": float(z_world),
                "z_idx": int(z_idx),
                "random_seed": int(seed + z_idx),
                "assignment_randomness": float(selected_ar),
                "assignment_randomness_selection": dict(ar_metadata),
            }
        )

        # Save
        out_path = save_result(result, z_world, output_dir)
        manifest_records.append({
            "z_world": f"{z_world:+.3f}",
            "filename": out_path.name,
            "n_spots": result.n_obs,
            "z_idx": z_idx,
            "assignment_randomness": f"{selected_ar:.6g}",
        })

    # ---- Step 6: Manifest ----
    logger.info("=" * 60)
    logger.info("STEP 6: Writing manifest")
    logger.info("=" * 60)
    write_manifest(manifest_records, output_dir)
    transfer_metadata["n_generated_slices"] = int(len(manifest_records))
    write_transfer_metadata(transfer_metadata, output_dir)

    logger.info("=" * 60)
    logger.info("Transfer complete. %d slices generated.", len(manifest_records))
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transfer GSE269617 reference expression onto DevCCF 3D blueprints.",
    )
    parser.add_argument(
        "--references",
        required=True,
        help="Directory containing GSE269617 region-annotated h5ad files.",
    )
    parser.add_argument(
        "--reference-pattern",
        default="*",
        help="fnmatch pattern to filter reference files (default: '*').",
    )
    parser.add_argument(
        "--blueprints",
        required=True,
        help="Path to the DevCCF blueprint JSON file.",
    )
    parser.add_argument(
        "--label-key",
        default="region",
        help="Observation column for region labels (default: 'region').",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/generated/",
        help="Directory to write generated h5ad files + manifest (default: outputs/generated/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Base random seed (default: 2026).",
    )
    parser.add_argument(
        "--z-subsample",
        type=int,
        default=1,
        help="Process every Nth z-level (default: 1).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N z-levels (useful for testing).",
    )
    parser.add_argument(
        "--max-reference-spots",
        type=int,
        default=None,
        help="Subsample reference slices to this many spots (default: no subsampling).",
    )
    parser.add_argument(
        "--assignment-randomness",
        default="auto",
        help="'auto' or a fixed conditional assignment_randomness value in [0, 1].",
    )
    parser.add_argument(
        "--ar-estimate-genes",
        type=int,
        default=20,
        help="High-variance genes for automatic AR estimation (default: 20).",
    )
    parser.add_argument(
        "--ar-step",
        type=float,
        default=0.05,
        help="Candidate AR step size for automatic estimation (default: 0.05).",
    )
    parser.add_argument(
        "--ar-max",
        type=float,
        default=0.5,
        help="Maximum candidate AR for automatic estimation (default: 0.5).",
    )
    parser.add_argument(
        "--ar-neighbors",
        type=int,
        default=6,
        help="Spatial neighbors for automatic AR Moran's-I scoring (default: 6).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logger.info("transfer_to_devccf.py starting.")
    logger.info("  references      = %s", args.references)
    logger.info("  reference_pat   = %s", args.reference_pattern)
    logger.info("  blueprints      = %s", args.blueprints)
    logger.info("  label_key       = %s", args.label_key)
    logger.info("  output_dir      = %s", args.output_dir)
    logger.info("  seed            = %d", args.seed)
    logger.info("  z_subsample     = %d", args.z_subsample)
    logger.info("  limit           = %s", args.limit)
    logger.info("  max_ref_spots   = %s", args.max_reference_spots)
    logger.info("  assign_random   = %s", args.assignment_randomness)

    transfer(
        reference_dir=args.references,
        reference_pattern=args.reference_pattern,
        blueprint_path=args.blueprints,
        label_key=args.label_key,
        output_dir=args.output_dir,
        seed=args.seed,
        z_subsample=args.z_subsample,
        limit=args.limit,
        max_reference_spots=args.max_reference_spots,
        assignment_randomness=args.assignment_randomness,
        ar_estimate_genes=args.ar_estimate_genes,
        ar_step=args.ar_step,
        ar_max=args.ar_max,
        ar_neighbors=args.ar_neighbors,
    )


if __name__ == "__main__":
    main()
