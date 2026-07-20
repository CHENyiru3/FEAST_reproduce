#!/usr/bin/env python3
"""Fail on historical or machine-specific material in the clean repository."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDIES = (
    "00_simulator_benchmark",
    "01_clustering",
    "02_alignment",
    "03_deconvolution",
    "04_batch_effect_removal",
    "05_2d_conditional_transfer",
    "06_3d_stack",
    "07_3d_transfer",
)
TEXT_SUFFIXES = {".md", ".py", ".sh", ".yaml", ".yml", ".json", ".txt", ".csv"}
FORBIDDEN_ALL = (
    "/" + "maiziezhou_lab2/",
    "archive_" + "202",
    "publication_repair_" + "202",
)
FORBIDDEN_STUDY_00_01 = (
    "PrefitSimulator",
    'spatial_mode="ot_spatial"',
    "spatial_mode='ot_spatial'",
    "TransportConfig(",
    "assignment_blocks=True",
)


def main() -> int:
    errors: list[str] = []
    for study in STUDIES:
        path = ROOT / study
        if not path.is_dir():
            errors.append(f"missing study directory: {study}")

    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if (
            "data/local" in path.as_posix()
            or "outputs" in path.parts
            or ".work" in path.parts
            or ".archive" in path.parts
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(ROOT)
        for token in FORBIDDEN_ALL:
            if token in text:
                errors.append(f"{relative}: contains forbidden token {token!r}")
        if relative.parts and relative.parts[0] in STUDIES[:2]:
            for token in FORBIDDEN_STUDY_00_01:
                if token in text:
                    errors.append(f"{relative}: contains forbidden token {token!r}")

    if errors:
        print("clean-repository check: FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("clean-repository check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
