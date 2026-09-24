# Study 07 visualization

The active figures use the completed `generative_transfer_two_reference_float32_v1`
reconstruction for E15.5 and E18.5. Earlier scripts default to `outputs/final`;
pass the active input explicitly when reproducing those figures.

## Sections through different directions

The [multi-plane figures](figures/multiplane_sections/README.md) show XY, XZ,
YZ and two 45-degree oblique sections, with a neutral 3D plane locator
and Slc17a7, Gad2, Gfap and Reln expression for each section.

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../envs/feast-study06-1.0.6-precision/bin/python visualization/07_3d_transfer/plot_multiplane_sections.py
```

The script defaults to the active two-reference float32 run. Use `--output-dir`
to render another copy, and `--dpi 600` for 600-dpi PNGs.
PDF and SVG contain editable text and rasterized point layers.

The cutting planes pass through the middle unique coordinate on each native
axis within each age. The XZ 45° plane is translated +0.4 along its unit normal
(-1/√2, 0, -1/√2), selecting another parallel section. Each slab has thickness 0.02 in stored DevCCF world
coordinates, matching native voxel spacing. All selected voxels are projected
onto the plane and displayed without interpolation, averaging or smoothing.
Plane basis vectors, centers, selection counts and input paths are recorded
in `provenance.json`; `section_plot_data.csv.gz` retains source z/spot indices,
native coordinates, projected coordinates, labels and raw marker counts.

Expression uses log1p generated counts with gene-wise upper limits at the
pooled positive 99th percentile across displayed ages and planes. This is
display saturation only. The spatial scale is shared across all section
panels. Gray voxels have zero expression. The 3D locators sample up to 700
voxels per original z slice using the existing gene-volume sampling convention;
the section panels use every voxel within the slab.

Axis labels refer to stored coordinates rather than inferred anatomical
directions. Central sections are selected independently by age and are not
homologous cross-age positions. Target expression was generated, not observed.

## Existing figures

The [existing eight-page collection](figures/study07_figures.pdf) remains available.
The scripts `plot_3d_region_four_views.py` and `plot_axis_cross_sections.py`
provide the original z-section views. `plot_3d_gene_distribution.py` renders
whole-volume expression; the other entry points render expression summaries,
transfer diagnostics and resampling benchmarks.
