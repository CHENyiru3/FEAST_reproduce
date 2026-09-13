# Visualization code

Figure builders are organized by study (`00_simulator_benchmark` through
`07_3d_transfer`); `08_workflow` contains workflow illustration builders.
See the [figure source map](../FIGURE_SOURCE_MAP.md) for the matching analysis
inputs and links to each study's commands.

Run preparation, evaluation, and validation before plotting. Some composite
figures also consume CSVs produced by earlier plotting commands. Figure scripts
write PDF/SVG/PNG, plotting tables, and provenance below their local `figures/`
directories. These generated files are excluded from Git.

Study 07's plotting code now includes the scripts actually used for the
September 9 two-reference float32 report. Use its
[reproduction commands](07_3d_transfer/README.md), which pass the input root
explicitly. Study 06's older defaults do not establish completion of the latest
precision-build run; select validated inputs explicitly.

The established [figure style](STYLE.md) is retained separately. A rendered
figure does not itself establish a scientific claim or manuscript selection.
