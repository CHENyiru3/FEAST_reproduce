# Dataset preprocessing

Upstream data preparation for the FEAST reproduction studies. This directory
contains the 12 Python scripts originally maintained in
`Reproduce/Preprocess_helper`, plus eight dataset download scripts and their
URL tables. Processing rules, QC thresholds, label logic, seeds, overwrite
checks, and existing provenance recording are preserved. Machine-specific
path defaults were replaced with paths relative to the checkout.

No raw data, processed matrices, annotation tables, or execution logs are
included. The scripts were copied and checked without rerunning preprocessing.

## Dataset-to-study map

| Dataset | Preprocessing entry point | Used by |
|---|---|---|
| spatialLIBD DLPFC Visium | `preprocess_spatiallibd_dlpfc_visium.py` | Studies 00, 01, 02, 04, 05 |
| Allen Zhuang-ABCA-1 MERFISH | `preprocess_allen_zhuang_abca1.py` | Studies 00, 03, 05, 06 |
| 10x Xenium lymph node | `preprocess_10x_xenium_human_lymph_node.py` | Study 00 |
| Tencent SpatialOmics dataset 119 / Slide-seq | `preprocess_tencent_spatialomics_dataset119.py` | Study 00 |
| MOSTA / Stereo-seq | `preprocess_mosta_mouse_embryo_single_slice.py` | Study 00 |
| GSE251926 / OpenST | `preprocess_gse251926_metastatic_lymph_node_3d.py` | Study 00 |
| GSE269617 | `preprocess_gse269617.py`, then `annotate_telencephalon_regions.py` | Study 07 references |
| DevCCFv1 | `build_devccfv1_coordinate_resource.py` | Study 07 atlas; workflow illustrations |

`preprocess_common.py` supplies shared count handling, metadata, QC, and output
validation. `run_preprocess_all.py` dispatches the seven expression converters;
it does not build DevCCF resources or run the GSE269617 annotation stage.
`recover_gse269617_annotations.py` is an alternative for transferring existing
barcode-keyed labels or annotated references; it does not infer missing labels.
The recorded GSE269617 annotation manifest identifies the marker-rule
`annotate_telencephalon_regions.py` workflow as the source of the current labels.

## Environment and paths

Use Python 3.11 with `anndata`, `numpy`, `pandas`, `scipy`, `scanpy`, and `h5py`.
Optional DevCCF QC plots also use `matplotlib`. Downloads require Bash and
standard shell utilities, `curl` (some scripts also support `wget`), and `jq`
for Tencent and Figshare metadata. This is a dependency list, not an exact
cross-platform environment lock.

By default the Python scripts read `../Datasets/Raw/` and write
`../Datasets/Processed/`, relative to the repository root, regardless of the
shell's working directory. Prefer explicit roots for a new reproduction:

```bash
PREPROCESS_PY=/path/to/preprocessing-environment/bin/python
DATA_ROOT=/path/to/new/Datasets
RAW="$DATA_ROOT/Raw"
PROCESSED="$DATA_ROOT/Processed"
```

Download preparation is documented in [downloads/README.md](downloads/README.md).
The copied URL tables identify the recorded upstream files; remote availability
has not been rechecked during this code migration.

## Expression datasets

From the repository root, inspect a selected dataset and then process it:

```bash
"$PREPROCESS_PY" preprocessing/preprocess_spatiallibd_dlpfc_visium.py \
  --raw-root "$RAW" --processed-root "$PROCESSED" \
  --samples 151670 151675 151676 --dry-run

"$PREPROCESS_PY" preprocessing/preprocess_spatiallibd_dlpfc_visium.py \
  --raw-root "$RAW" --processed-root "$PROCESSED" \
  --samples 151670 151675 151676
```

For multiple datasets, pass the keys from `run_preprocess_all.py`:

```bash
"$PREPROCESS_PY" preprocessing/run_preprocess_all.py \
  --raw-root "$RAW" --processed-root "$PROCESSED" \
  --datasets spatialLIBD_DLPFC_Visium Allen_Zhuang_ABCA_1
```

The dispatcher processes all samples for each selected dataset. Use individual
scripts for `--samples` or dataset-specific options. Outputs normally appear
under `<processed-root>/<dataset>/h5ad/`, with local `manifest/` and `logs/`.
Some inventory-only `--dry-run` paths also create log/manifest directories.
Existing H5AD files are protected unless `--overwrite` is explicitly supplied.

The common defaults are seed 2026, minimum total count 1, minimum detected gene
count 1, and genes present in at least 3 cells; mitochondrial-percentage
filtering is disabled unless requested. Dataset-specific processing also
applies. These scripts preserve raw-count matrices rather than applying the
normalization used later by individual benchmark methods.

DLPFC requires barcode-keyed layer labels such as
`<raw-root>/spatialLIBD_DLPFC_Visium/metadata/DLPFC_annotations/151675_truth.txt`.
These labels are not included in the source upload and are not fetched by the
copied vendor-data downloader. Supply the original annotation tables.
`--allow-missing-labels` produces an explicitly incomplete annotation state;
it is not a substitute for study ground truth.

## GSE269617 and DevCCF sequence

GSE269617 raw references generally lack the broad-region labels required by
Study 07. Create the count-preserving intermediate files, build the atlas
crosswalk, then apply the recorded annotation workflow:

```bash
"$PREPROCESS_PY" preprocessing/preprocess_gse269617.py \
  --raw-root "$RAW" --processed-root "$PROCESSED" --allow-missing-labels

"$PREPROCESS_PY" preprocessing/build_devccfv1_coordinate_resource.py \
  --raw-root "$RAW" --processed-root "$PROCESSED" \
  --ages E15.5 E18.5 --build-gse269617-crosswalk

"$PREPROCESS_PY" preprocessing/annotate_telencephalon_regions.py \
  --processed-root "$PROCESSED" \
  --raw-tar "$RAW/GSE269617/components/GSE269617_RAW.tar" \
  --schema-path "$PROCESSED/DevCCFv1_figshare_26377171/coordinate_system/GSE269617_region_merged/gse269617_region_schema.tsv"
```

The annotation code uses its existing marker rules, spatial smoothing, and
available author-label anchors. Study 07 consumes
`GSE269617/h5ad_region_annotated/` and the atlas resources in
`DevCCFv1_figshare_26377171/coordinate_system/GSE269617_region_merged/`.
Expression-free per-age blueprints are then prepared by the study's
[`prepare.py`](../07_3d_transfer/prepare.py).

If using `recover_gse269617_annotations.py` instead, provide the existing label
source and set `--devccf-resource-root` explicitly when changing
`--processed-root`; that option has its own default.

## Hand off to the studies

Use the sample IDs and input requirements in each study's README and
`data/input_checksums.csv`. Place local links under `<study>/data/local/` or
adjust the supported input-path arguments. Study 00 uses benchmark aliases such
as `DLPFC_151675.h5ad` and `MERFISH_007.h5ad`; preprocessing retains dataset-native
filenames. The study's input table records the required correspondence.

Copying the scripts does not establish byte-identical regeneration of the
existing article inputs. The original processed manifests remain local in
`Datasets/Processed/*/manifest/` and record the settings and code used at that
time. No datasets, labels, or scientific outputs were changed by this migration.
