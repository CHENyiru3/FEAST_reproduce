# Study 03: deconvolution

## Cell-class rerun (current)

The current registered rerun uses `cell_class` consistently for FEAST truth,
RCTD, and Cell2location. Both methods retain the same slice-specific classes
with at least 50 reference cells; truth mass from rarer classes is evaluated as
`Other`. Cell2location fits only positive-library locations and reinserts empty
locations as zero rows. The previous fine-`cell_type` evidence remains on disk
for audit but is not the source of the current figures.

The versioned configuration is `config_cell_class.yaml`, and the output root is
`outputs/cell_class_rerun_20260810_v1`. Run or resume the exact six-pair,
twelve-job design with:

```bash
python 03_deconvolution/run.py \
  --input-dir 03_deconvolution/data/local \
  --output-dir 03_deconvolution/outputs/cell_class_rerun_20260810_v1 \
  --feast-commit 68816e5c1862a6fa2a49bc30609d617c7fa4b449 \
  --rscript /path/to/envs/rctd_bioc/bin/Rscript \
  --rctd-python /path/to/envs/feast-prepublication-py311/bin/python \
  --cell2location-python /path/to/envs/cell2loc_env/bin/python \
  --config 03_deconvolution/config_cell_class.yaml \
  --methods rctd cell2location --resume
```

Cell2location remains CUDA-only. Score and independently validate the completed
matrix with:

```bash
python 03_deconvolution/score_cell_class.py \
  --run-dir 03_deconvolution/outputs/cell_class_rerun_20260810_v1 \
  --config 03_deconvolution/config_cell_class.yaml \
  --output-dir 03_deconvolution/outputs/cell_class_rerun_20260810_v1/scores_cell_class

python 03_deconvolution/validate_cell_class.py \
  --run-dir 03_deconvolution/outputs/cell_class_rerun_20260810_v1 \
  --config 03_deconvolution/config_cell_class.yaml \
  --scores-dir 03_deconvolution/outputs/cell_class_rerun_20260810_v1/scores_cell_class \
  --output 03_deconvolution/outputs/cell_class_rerun_20260810_v1/validation_cell_class.csv
```


Inputs are raw integer-count `Zhuang-ABCA-1.007.h5ad`, `.050.h5ad`, and
`.100.h5ad`, with `cell_class` labels and `obsm['spatial']`; see
`data/input_checksums.csv`. Use a fresh run root for new work. The older
fine-cell-type `config.yaml`, `score.py`, and `validate.py` are retained for
historical comparisons. `config_120.yaml` is a separate slice-120 diagnostic;
its preview must be reviewed before starting Cell2location. Neither defines
the current cell-class figures.

## Figures

Run from the repository root after validation.

```bash
python visualization/03_deconvolution/plot.py
python visualization/03_deconvolution/plot_spatial.py
```

The numerical source is `outputs/cell_class_rerun_20260810_v1/scores_cell_class/`.
Both methods share slice-specific classes with at least 50 reference cells;
rarer truth mass is `Other`. The spatial panel uses S100 at resolutions 0.25
and 0.10 with the same 15 classes plus `Other`. Atomic results are mixed;
no aggregate method ranking is authorized.
