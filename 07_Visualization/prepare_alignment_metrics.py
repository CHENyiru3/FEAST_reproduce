"""Transform alignment_benchmark_results.csv into long-format metrics CSV for plotting.

Maps benchmark columns to display metric names:
    nn_accuracy           -> Accuracy
    morpho_precision      -> Precision
    morpho_f1             -> F1 Score
    ge_correlation        -> Gene Expression Correlation
    nn_region_accuracy    -> Adjusted Region Accuracy

Filters to a single alteration_id (default: 'baseline') for a clean
robustness-to-rotation-angle curve.  morpho_* metrics are spateo-only;
spacel rows carry NaN and will be skipped automatically at plot time.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_CSV = ROOT / "02_alignment" / "results" / "alignment_benchmark_results.csv"
OUT_CSV = Path(__file__).resolve().parent / "alignment_alteration_metrics.csv"

COLUMN_MAP = {
    "nn_accuracy": "Accuracy",
    "morpho_precision": "Precision",
    "morpho_f1": "F1 Score",
    "ge_correlation": "Gene Expression Correlation",
    "nn_region_accuracy": "Adjusted Region Accuracy",
}


def prepare(alteration_id: str = "baseline") -> pd.DataFrame:
    df = pd.read_csv(BENCHMARK_CSV)
    df = df[df["alteration_id"] == alteration_id].copy()

    # Normalize method names for display
    _METHOD_DISPLAY = {"spateo": "Spateo", "paste": "PASTE"}
    df["method"] = df["method"].map(_METHOD_DISPLAY).fillna(df["method"].str.capitalize())

    records = []
    for _, row in df.iterrows():
        for src_col, display_name in COLUMN_MAP.items():
            score = row[src_col]
            records.append(
                {
                    "method": row["method"],
                    "angle": row["angle"],
                    "metric": display_name,
                    "score": score,
                }
            )

    out = pd.DataFrame(records)
    # Drop NaN scores (spacel morpho_*)
    out = out.dropna(subset=["score"])
    out.to_csv(OUT_CSV, index=False)

    print(f"Wrote {len(out)} rows to {OUT_CSV}")
    print(f"  Alteration filter: {alteration_id}")
    print(f"  Methods: {out['method'].unique().tolist()}")
    print(f"  Metrics: {out['metric'].unique().tolist()}")
    for metric in COLUMN_MAP.values():
        sub = out[out["metric"] == metric]
        print(f"    {metric}: {len(sub)} rows, range [{sub['score'].min():.4f}, {sub['score'].max():.4f}]")
    return out


if __name__ == "__main__":
    prepare()
