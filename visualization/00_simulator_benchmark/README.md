# Study 00: simulator benchmark visualization

The figure source is the validated 60-row atomic metric table:

```text
../../00_simulator_benchmark/outputs/final_rerun_20260718_metrics_v2/
    simulator_quality_metrics.csv
```

It contains the fresh FEAST reference-rank arm and four frozen external
simulators. The historical FEAST OT-spatial arm is outside the clean rerun and
is not plotted.

Rebuild from the repository root:

```bash
python visualization/00_simulator_benchmark/plot.py
```

Figures, plot-ready tables, and `figure_provenance.json` are written to the
local `figures/` directory. Use `--metrics-csv` and `--output-dir` only when
validating a new candidate run.
