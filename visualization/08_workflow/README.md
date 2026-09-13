# Study 08 workflow illustration components

This directory builds standalone, data-first assets for manual assembly of the
two FEAST 3D workflows. It does not regenerate either study or replace their
validated diagnostic figures.

It now provides two builders:

- `build_components.py` retains the original spot- and slice-level component
  set.
- `build_anatomical_workflow.py` builds the affine-aware anatomical review set
  and an assembled graphical abstract. It replaces the Study 07 point-cloud
  volume with a DevCCF segmentation mesh while keeping Study 06 as an exploded
  stack of transcriptomic sections.

- **Study 06** uses expression-bearing reference slices inside the native
  coordinate system to reconstruct an intervening target plane.
- **Study 07** transfers expression to an expression-free DevCCF coordinate
  system. Its source cohorts are deliberately shown outside the DevCCF volume.

The default Study 06 case is the validated bracket-only triplet
`082 -> 083 -> 084`. All expression maps show all-gene log-library size, not a
selected gene. Study 06 target counts are not read when rendering its blueprint
asset, and Study 07 assets do not support a target-expression accuracy claim.

Run from `FEAST_reproduce`:

```bash
python visualization/08_workflow/build_components.py
```

Assets are written below `figures/illustration_components/` as editable PDF and
SVG plus 600-DPI PNG. Dense point layers are rasterized; labels, outlines,
legends, and colour bars stay vector-editable. Use `--output-dir` for a fresh
alternate directory and `--study06-target` to choose another validated target.

## Affine-aware anatomical workflow

Run with the recorded Python 3.11 environment, which includes compatible
`scikit-image` and `trimesh` versions:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
../envs/feast-prepublication-py311/bin/python3.11 \
  visualization/08_workflow/build_anatomical_workflow.py
```

The default review uses Study 06 slices `082 -> 083 -> 084` and the E15.5
DevCCF volume with generated plane index 78. Outputs are written below
`figures/anatomical_workflow_review_v6/`:

- `feast_workflow_graphical_abstract.{pdf,svg,png}`;
- measured/target and generated Study 06 stack components;
- expression-free and generated E15.5 DevCCF atlas components;
- mesh, dataset-summary, and figure-provenance records.

To extract only the reusable, single-page component PDFs (without the composite
graphical abstract), run:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
../envs/feast-prepublication-py311/bin/python3.11 \
  visualization/08_workflow/build_anatomical_workflow.py --components-only
```

This writes six component sets below `figures/anatomical_workflow_components_v6/`:
the Study 06 input/output stacks, E15.5 source cohort, expression-free atlas,
generated atlas, and the FEAST transfer module. Each component has its own
single-page PDF, SVG, and 600-DPI PNG.

The atlas renderer extracts surfaces from the labeled NIfTI volume in voxel
space and then applies its affine to every vertex. Labels 1--8 are the FEAST
target regions, label 9 supplies a translucent anatomical context shell, and
label 0 remains background. The central expression cutaway uses occupied
generated voxels only and shows `log1p` total counts on a 2nd--98th percentile
scale. The DevCCF view is an atlas segmentation rendering, not measured MRI.

The E15.5 DevCCF input is the moderately opaque gray atlas shell. The output
intentionally renders only the retained final-generation labels 1--8, omitting
the non-generated atlas context, and adds the generated-expression cutaway.
For Study 06, the pre-FEAST target is a pale outline only, while the generated
target is expression-filled with a thin blue boundary.

The Study 06 display uses local slices `080`--`086` so it reads as a genuine
native stack. All six observed neighboring slices are expression-rendered for
local stack context, but only `082` and `084` are FEAST inputs; the additional
displayed expression does not expand conditional-generation support.
The displayed 082--083 and 083--084 gaps are deliberately enlarged relative to
the other local planes, making the missing-plane operation visible; this is a
visual spacing convention, not a physical-z claim.

Study 06 deliberately remains an exploded slice stack because it does not have
a continuous anatomical target volume. Its common XY system is physical; z is
expanded only for display and is labeled as such. Study 07 remains descriptive:
there is no target-expression ground truth or accuracy claim.

Geometry checks are available as:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
../envs/feast-prepublication-py311/bin/python3.11 -m pytest -q \
  visualization/08_workflow/test_anatomical_workflow.py
```
