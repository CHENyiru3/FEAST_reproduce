#!/usr/bin/env python
"""Extract 2D cross-section blueprints from DevCCF NIfTI broad-region volumes.

Each z-level (voxel j index) that contains non-background tissue is extracted
as a flat-listed blueprint: arrays of x, y world coordinates and region labels.
These blueprints are used as target geometries for FEAST de novo generation.
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# NIfTI-1 datatype codes (adapted from build_devccfv1_coordinate_resource.py)
# ---------------------------------------------------------------------------
DTYPE_BY_NIFTI: dict[int, np.dtype] = {
    2: np.dtype("u1"),    # uint8
    4: np.dtype("i2"),    # int16
    8: np.dtype("i4"),    # int32
    16: np.dtype("f4"),   # float32
    64: np.dtype("f8"),   # float64
    256: np.dtype("i1"),  # int8
    512: np.dtype("u2"),  # uint16
    768: np.dtype("u4"),  # uint32
}


# ---------------------------------------------------------------------------
# NIfTI-1 byte reader
#   Adapted from read_nifti_bytes() in:
#   /maiziezhou_lab2/yiru/SPEC/staged_scripts/build_devccfv1_coordinate_resource.py
#   (original function at approx. line 336)
# ---------------------------------------------------------------------------
def read_nifti_bytes(raw: bytes, source_label: str = "<bytes>") -> dict[str, Any]:
    """Parse NIfTI-1 header and return data array + affine.

    Parameters
    ----------
    raw : bytes
        Decompressed NIfTI-1 bytes (not gzip-compressed).
    source_label : str
        Label used in error messages.

    Returns
    -------
    dict with keys ``data`` (np.ndarray), ``affine`` ((4,4) np.ndarray),
    ``shape``, ``pixdim``, etc.
    """
    # Detect endianness from sizeof_hdr (must be 348)
    endian = "<"
    if struct.unpack("<i", raw[:4])[0] != 348:
        if struct.unpack(">i", raw[:4])[0] == 348:
            endian = ">"
        else:
            raise ValueError(f"{source_label} is not a NIfTI-1 file")

    # Dimensions
    dim = struct.unpack(endian + "8h", raw[40:56])
    ndim = int(dim[0])
    shape = tuple(int(x) for x in dim[1 : 1 + ndim])

    # Datatype
    datatype = struct.unpack(endian + "h", raw[70:72])[0]
    bitpix = struct.unpack(endian + "h", raw[72:74])[0]

    # Pixel dimensions
    pixdim = struct.unpack(endian + "8f", raw[76:108])

    # Voxel offset
    vox_offset = int(round(struct.unpack(endian + "f", raw[108:112])[0]))

    # Scaling
    scl_slope = struct.unpack(endian + "f", raw[112:116])[0]
    scl_inter = struct.unpack(endian + "f", raw[116:120])[0]

    # sform code
    sform_code = struct.unpack(endian + "h", raw[254:256])[0]

    if datatype not in DTYPE_BY_NIFTI:
        raise ValueError(f"Unsupported NIfTI datatype {datatype} in {source_label}")
    dtype = DTYPE_BY_NIFTI[datatype].newbyteorder(endian)

    n_values = int(np.prod(shape))
    start = vox_offset
    stop = start + n_values * dtype.itemsize
    data = np.frombuffer(raw[start:stop], dtype=dtype, count=n_values).reshape(
        shape, order="F"
    )

    if scl_slope not in (0.0, 1.0) or scl_inter != 0.0:
        data = data.astype(np.float32) * (
            scl_slope if scl_slope != 0 else 1.0
        ) + scl_inter

    # Build affine
    affine = np.eye(4, dtype=float)
    if sform_code > 0:
        affine[0, :] = struct.unpack(endian + "4f", raw[280:296])
        affine[1, :] = struct.unpack(endian + "4f", raw[296:312])
        affine[2, :] = struct.unpack(endian + "4f", raw[312:328])
    else:
        affine[0, 0] = pixdim[1] or 1.0
        affine[1, 1] = pixdim[2] or 1.0
        affine[2, 2] = pixdim[3] or 1.0

    return {
        "data": data,
        "affine": affine,
        "shape": shape,
        "pixdim": pixdim,
        "endian": endian,
        "vox_offset": vox_offset,
    }


def read_nifti_gz(path: str | Path) -> dict[str, Any]:
    """Read a .nii.gz file, decompress, and return data + affine."""
    path = Path(path)
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    return read_nifti_bytes(raw, source_label=str(path))


# ---------------------------------------------------------------------------
# Region schema
# ---------------------------------------------------------------------------
def load_region_schema(tsv_path: str | Path) -> dict[int, dict[str, Any]]:
    """Parse the DevCCF region schema TSV.

    Returns dict mapping region_id (int) → dict with keys:
        region_label, hex_color, include_in_final_generation
    """
    schema: dict[int, dict[str, Any]] = {}
    path = Path(tsv_path)
    with open(path, "r") as fh:
        header = fh.readline().strip().split("\t")
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fields = line.split("\t")
            record = dict(zip(header, fields))
            rid = int(record["region_id"])
            schema[rid] = {
                "region_label": record["region_label"],
                "hex_color": record["hex_color"],
                "include_in_final_generation": record[
                    "include_in_final_generation"
                ].strip().upper()
                == "TRUE",
            }
    return schema


# ---------------------------------------------------------------------------
# Blueprint extraction
# ---------------------------------------------------------------------------
def extract_blueprints(
    data: np.ndarray,
    affine: np.ndarray,
    region_schema: dict[int, dict[str, Any]],
    mask_other: bool = True,
    z_step: int = 1,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Extract 2D slice blueprints at every ``z_step``-th voxel j level.

    Parameters
    ----------
    data : np.ndarray
        3D volume of shape (ni, nj, nk), uint8 region IDs.
    affine : np.ndarray
        (4,4) affine matrix.
    region_schema : dict
        Region ID → metadata dict (from load_region_schema).
    mask_other : bool
        If True, drop spots labeled "Other" (include_in_final_generation=False).
    z_step : int
        Step size along j axis (1 = every level).

    Returns
    -------
    blueprints : dict
        Blueprint dictionary keyed by z_world (str).
    region_labels : list[str]
        Region labels included in output (from schema, excluding masked).
    masked_labels : list[str]
        Region labels excluded from output.
    """
    ni, nj, nk = data.shape

    # Collect all region labels present in the schema
    all_regions: dict[int, str] = {}
    for rid, rec in region_schema.items():
        all_regions[rid] = rec["region_label"]

    # Determine which labels are masked
    region_labels: list[str] = []
    masked_labels: list[str] = []
    for rid, rec in region_schema.items():
        label = rec["region_label"]
        if rec["include_in_final_generation"]:
            if label not in region_labels:
                region_labels.append(label)
        else:
            if label not in masked_labels:
                masked_labels.append(label)

    # Add any region IDs present in data but not in schema (should not happen)
    unique_ids = np.unique(data)
    for uid in unique_ids:
        if uid not in all_regions:
            all_regions[uid] = f"Unknown_{uid}"
            if f"Unknown_{uid}" not in masked_labels:
                masked_labels.append(f"Unknown_{uid}")

    # Build mask map: region_id → True if should be DROPPED
    drop_map: dict[int, bool] = {}
    for rid in all_regions:
        rec = region_schema.get(rid)
        if rec is None or not rec["include_in_final_generation"]:
            drop_map[rid] = True
        else:
            drop_map[rid] = False

    blueprints: dict[str, Any] = {}

    j_range = range(0, nj, z_step)

    # Pre-compute i and k index grids for fast world-coordinate conversion
    # For a given j, x_world = affine[0,0]*i + affine[0,1]*j + affine[0,2]*k + affine[0,3]
    #               y_world = affine[1,0]*i + affine[1,1]*j + affine[1,2]*k + affine[1,3]
    # Since the DevCCF affine is diagonal-permuted, many terms are zero.
    # We compute the full matrix product to be safe.
    i_idx = np.arange(ni, dtype=np.float64)
    k_idx = np.arange(nk, dtype=np.float64)
    ii, kk = np.meshgrid(i_idx, k_idx, indexing="ij")  # shapes (ni, nk)

    for j in j_range:
        slice_data = data[:, j, :]  # shape (ni, nk)

        # Build mask of spots to KEEP (not background, not dropped region)
        keep_mask = np.zeros_like(slice_data, dtype=bool)
        for rid in unique_ids:
            if rid == 0:  # Background
                continue
            if drop_map.get(rid, True):
                continue
            keep_mask |= slice_data == rid

        if not np.any(keep_mask):
            continue  # no non-background tissue at this z-level

        # Flatten and filter
        keep_i = ii[keep_mask]
        keep_k = kk[keep_mask]
        keep_regions = slice_data[keep_mask]

        # World coordinates using full affine
        # voxel_coords = [i, j, k, 1] for each spot
        j_f = float(j)
        ones = np.ones_like(keep_i)
        voxel_coords = np.stack([keep_i, np.full_like(keep_i, j_f), keep_k, ones], axis=0)  # (4, n)
        world_coords = affine @ voxel_coords  # (4, n)

        x_world = world_coords[0, :]
        y_world = world_coords[1, :]
        z_world = float(world_coords[2, 0])  # Should be constant for fixed j

        # Map region IDs to string labels
        region_names = np.array([all_regions.get(int(rid), f"Unknown_{rid}") for rid in keep_regions])

        n_spots = int(len(x_world))
        z_key = f"{z_world:.2f}"

        blueprints[z_key] = {
            "z_world": z_world,
            "voxel_j": j,
            "x": x_world.tolist(),
            "y": y_world.tolist(),
            "region": region_names.tolist(),
            "n_spots": n_spots,
        }

    return blueprints, region_labels, masked_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 2D cross-section blueprints from DevCCF NIfTI volumes."
    )
    parser.add_argument(
        "--volume",
        required=True,
        type=str,
        help="Path to .nii.gz broad-region annotation volume.",
    )
    parser.add_argument(
        "--region-schema",
        required=True,
        type=str,
        help="Path to gse269617_region_schema.tsv.",
    )
    parser.add_argument(
        "--age",
        required=True,
        type=str,
        help="Age label (e.g. E15.5, E18.5).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--mask-other",
        action="store_true",
        default=True,
        help="Drop spots labeled 'Other' per schema (default: True).",
    )
    parser.add_argument(
        "--no-mask-other",
        dest="mask_other",
        action="store_false",
        help="Keep 'Other' spots in output.",
    )
    parser.add_argument(
        "--z-step",
        type=int,
        default=1,
        help="Extract every Nth z-level (default: 1 = all levels).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    volume_path = Path(args.volume)
    schema_path = Path(args.region_schema)
    output_path = Path(args.output)

    # Validate inputs
    if not volume_path.exists():
        print(f"ERROR: volume file not found: {volume_path}", file=sys.stderr)
        sys.exit(1)
    if not schema_path.exists():
        print(f"ERROR: region schema not found: {schema_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading volume: {volume_path}")
    try:
        nifti = read_nifti_gz(volume_path)
    except (ValueError, struct.error, OSError) as e:
        print(f"ERROR: failed to read NIfTI volume: {e}", file=sys.stderr)
        sys.exit(1)

    data: np.ndarray = nifti["data"]
    affine: np.ndarray = nifti["affine"]
    shape = nifti["shape"]

    print(f"  shape: {shape}")
    print(f"  unique labels: {sorted(np.unique(data).tolist())}")
    print(f"  affine:\n{affine}")

    # Load region schema
    print(f"Reading region schema: {schema_path}")
    try:
        region_schema = load_region_schema(schema_path)
    except (OSError, ValueError) as e:
        print(f"ERROR: failed to read region schema: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(region_schema)} region entries loaded")

    # Extract blueprints
    print(f"Extracting blueprints (z_step={args.z_step}, mask_other={args.mask_other})...")
    try:
        blueprints, region_labels, masked_labels = extract_blueprints(
            data=data,
            affine=affine,
            region_schema=region_schema,
            mask_other=args.mask_other,
            z_step=args.z_step,
        )
    except Exception as e:
        print(f"ERROR: blueprint extraction failed: {e}", file=sys.stderr)
        sys.exit(1)

    n_z = len(blueprints)
    if n_z == 0:
        print("ERROR: no non-background z-levels found in volume", file=sys.stderr)
        sys.exit(1)

    spot_counts = [bp["n_spots"] for bp in blueprints.values()]
    min_spots = min(spot_counts)
    max_spots = max(spot_counts)
    print(f"  {n_z} z-levels extracted")
    print(f"  spot count range: {min_spots} – {max_spots}")

    # Estimate voxel size (diagonal entries of affine, max absolute)
    voxel_diag = np.max(np.abs(np.diag(affine)[:3]))

    # Build output
    output = {
        "metadata": {
            "age": args.age,
            "affine": affine.tolist(),
            "n_z_levels": n_z,
            "voxel_size_mm": float(voxel_diag),
            "region_labels": region_labels,
            "masked_labels": masked_labels,
        },
        "blueprints": blueprints,
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing output: {output_path}")
    with open(output_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"  done ({output_path.stat().st_size:,} bytes)")

    # Summary
    print(f"\nSummary for {args.age}:")
    print(f"  Volume shape: {shape}")
    print(f"  Z-levels extracted: {n_z}")
    print(f"  Spot count range per slice: {min_spots:,} – {max_spots:,}")
    print(f"  Region labels: {region_labels}")
    print(f"  Masked labels: {masked_labels}")


if __name__ == "__main__":
    main()
