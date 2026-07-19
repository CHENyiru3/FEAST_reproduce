# Study 01: fixed-panel clustering

This workflow regenerates every active result: 81 FEAST simulations (3 slices
times 27 non-OT conditions), 84 fixed-panel inputs after adding one real
baseline per slice, and 252 clustering outputs from GraphST, STAGATE+mclust,
and publication-valid unsupervised Leiden.

Place the three checked source slices at `data/local/<slice>.h5ad`; their
expected hashes are in `data/input_checksums.csv`. No previous FEAST
simulation, fixed-panel file, or method result is used.

## Run order

Use a new directory for each stage:

```bash
python run_simulations.py \
  --config config.yaml --input-dir data/local \
  --output-dir outputs/simulations

python build_fixed_panels.py \
  --config config.yaml --raw-dir data/local \
  --simulation-dir outputs/simulations \
  --output-dir outputs/fixed_panels

python run_methods.py \
  --config config.yaml --manifest outputs/fixed_panels/fixed_panel_manifest.csv \
  --panel-dir outputs/fixed_panels --method GraphST \
  --python "$GRAPHST_PYTHON" --output-dir outputs/methods

python run_methods.py \
  --config config.yaml --manifest outputs/fixed_panels/fixed_panel_manifest.csv \
  --panel-dir outputs/fixed_panels --method STAGATE_mclust \
  --python "$STAGATE_PYTHON" --output-dir outputs/methods

python run_methods.py \
  --config config.yaml --manifest outputs/fixed_panels/fixed_panel_manifest.csv \
  --panel-dir outputs/fixed_panels --method Leiden_unsupervised \
  --python "$FEAST_PYTHON" --output-dir outputs/methods

python score.py \
  --panel-dir outputs/fixed_panels --method-dir outputs/methods \
  --output-dir outputs/report

python validate.py --config config.yaml --raw-dir data/local \
  --simulation-dir outputs/simulations --panel-dir outputs/fixed_panels \
  --method-dir outputs/methods --report-dir outputs/report
```

`run_simulations.py --dry-run` prints the exact 81 simulations and
`run_methods.py --dry-run` prints the selected 84-job method matrix. Scientific
output roots are never overwritten. If a method process is interrupted, rerun
that same command with `--resume`; a candidate is skipped only when its
configuration, runner and wrapper sources, interpreter binary, seed,
parameters, input, outputs, runner log, and method diagnostics all validate.
Incomplete or stale candidates are preserved under
`outputs/methods/failures/`.

FEAST simulations use public `FEAST.simulate()` independently for every
condition, seed 2026, reference-rank spatial assignment, and the full global
SciPy assignment. The old fitted-object shortcut is not present.
An interrupted simulation stage can be continued by repeating its command with
`--resume`; verified H5ADs are not regenerated.

The 3,000-gene Seurat-v3 panel is selected once from each raw slice and applied
in identical order to all corresponding simulations and the real baseline.
Every input receives exactly one `normalize_total(target_sum=1e4)` and `log1p`.

GraphST and STAGATE both require a visible CUDA GPU and positive peak GPU
memory allocation during the method run; they fail rather than silently
falling back to CPU. The STAGATE run also records and hashes the inherited
`LD_LIBRARY_PATH` used to load CUDA libraries. Their external method
environments use NumPy
1.23.x, which is below FEAST's supported range; FEAST is neither imported nor
executed in those workers, and the limitation is retained in each candidate's
metadata. STAGATE jobs run sequentially to prevent shared-R state races. Both
methods use the true slice cluster count as the declared clustering target.
Leiden selects the maximum weighted Newman-Girvan modularity at fixed gamma 1,
breaking exact ties toward the lower resolution; labels are never read during
its resolution sweep.
