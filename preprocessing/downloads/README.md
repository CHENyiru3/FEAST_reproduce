# Raw-data acquisition scripts

Each dataset folder contains its original Bash downloader and
`manifest/raw_urls.tsv`, the small URL table used to identify upstream inputs.
The download scripts and URL tables are copied from `Datasets/Raw`; only
trailing blank lines in three scripts were removed.
They retain their existing remote checks, path checks, and checksum behavior.

Copy this source layout into a new raw-data directory before downloading. This
keeps downloaded files, extracted content, metadata responses, and logs outside
the code repository and preserves the layout expected by the preprocessing code.
From the repository root:

```bash
RAW=/path/to/new/Datasets/Raw
mkdir -p "$RAW"
for dataset in preprocessing/downloads/*/; do
  cp -R "$dataset" "$RAW/"
done

bash "$RAW/spatialLIBD_DLPFC_Visium/scripts/download_spatialLIBD_raw_data.sh"
```

The default destination is `<dataset>/samples/` or `<dataset>/components/`:

| Dataset folder | Script under `scripts/` | Destination |
|---|---|---|
| `spatialLIBD_DLPFC_Visium` | `download_spatialLIBD_raw_data.sh` | `samples/` |
| `Allen_Zhuang_ABCA_1` | `download_zhuang_abca1_components.sh` | `components/` |
| `10x_Xenium_human_lymph_node_preview` | `download_10x_xenium_human_lymph_node.sh` | `samples/` |
| `Tencent_SpatialOmics_dataset119` | `download_tencent_spatialomics_dataset119.sh` | `samples/` |
| `MOSTA_mouse_embryo_single_slice_h5ad` | `download_mosta_single_slice_h5ad.sh` | `samples/` |
| `GSE251926_metastatic_lymph_node_3d` | `download_GSE251926_metastatic_lymph_node_3d.sh` | `samples/` |
| `GSE269617` | `download_gse269617_geo.sh` | `components/` |
| `DevCCFv1_figshare_26377171` | `download_devccfv1_figshare.sh` | `components/` |

`CHECK_ONLY=1 bash <script>` checks remote availability/sizes without downloading
the dataset payloads. It still makes network requests and may save metadata or
logs. Tencent/DevCCF scripts may refresh their local URL tables from upstream
metadata. Preserve the recorded input versions when reproducing an existing run.

The downloaders do not supply every annotation needed by the studies. In
particular, DLPFC barcode-keyed layer labels must be supplied separately; see
[the preprocessing guide](../README.md). No download was executed during this
migration, and no remote URL availability claim is made here.
