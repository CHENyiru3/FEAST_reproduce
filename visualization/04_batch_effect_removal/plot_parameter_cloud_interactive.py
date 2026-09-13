"""Create an interactive Study 04A parameter-cloud viewer with Plotly."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plot_parameter_cloud import (
    ALPHA_COLORS,
    ALPHAS,
    DISPLAY_LIMITS,
    MODES,
    OUTPUT_DIR,
    REFERENCE_DISPLAY_REGION,
    load_cloud,
)


OUTPUT_PATH = OUTPUT_DIR / "parameter_cloud_batch_simulation_3d_interactive.html"


def selected_gene_index(clouds: dict[tuple[str, float], object]):
    reference = clouds[("shift_only", 0.0)]
    selected = np.logical_and.reduce([
        reference[coordinate].between(lower, upper).to_numpy()
        for coordinate, (lower, upper) in REFERENCE_DISPLAY_REGION.items()
    ])
    return reference.index[selected]


def scene_layout() -> dict:
    return {
        "xaxis": {"title": {"text": "log mean", "font": {"size": 16}}, "range": DISPLAY_LIMITS["log_mean"], "showbackground": True, "backgroundcolor": "#F7F7F7", "gridcolor": "#D0D0D0"},
        "yaxis": {"title": {"text": "log variance", "font": {"size": 16}}, "range": DISPLAY_LIMITS["log_variance"], "showbackground": True, "backgroundcolor": "#F7F7F7", "gridcolor": "#D0D0D0"},
        "zaxis": {"title": {"text": "zero proportion", "font": {"size": 16}}, "range": DISPLAY_LIMITS["zero_proportion"], "showbackground": True, "backgroundcolor": "#F7F7F7", "gridcolor": "#D0D0D0"},
        "aspectmode": "cube",
        "camera": {"eye": {"x": 1.45, "y": -1.45, "z": 0.75}},
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clouds = {}
    for mode in MODES:
        for alpha in ALPHAS:
            clouds[(mode, alpha)], _ = load_cloud(mode, alpha)
    gene_index = selected_gene_index(clouds)
    if len(gene_index) == 0:
        raise ValueError("The illustrative reference-cloud region contains no genes")

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("shift-only: translation only", "diagonal-affine: translation + axis-specific scaling"),
        horizontal_spacing=0.04,
    )
    figure.update_annotations(font={"size": 18}, xanchor="center")
    for column, mode in enumerate(MODES, start=1):
        for alpha in ALPHAS:
            cloud = clouds[(mode, alpha)].loc[gene_index]
            figure.add_trace(
                go.Scatter3d(
                    x=cloud["log_mean"],
                    y=cloud["log_variance"],
                    z=cloud["zero_proportion"],
                    mode="markers",
                    name=f"α = {alpha:g}",
                    legendgroup=f"alpha-{alpha:g}",
                    showlegend=column == 1,
                    marker={"size": 2.3, "opacity": 0.32, "color": ALPHA_COLORS[alpha]},
                    customdata=np.asarray(cloud.index, dtype=str),
                    hovertemplate="gene: %{customdata}<br>log mean: %{x:.3f}<br>log variance: %{y:.3f}<br>zero proportion: %{z:.3f}<extra></extra>",
                ),
                row=1,
                col=column,
            )

    visibility = []
    for alpha in ALPHAS:
        visibility.append([trace_alpha == alpha for _ in MODES for trace_alpha in ALPHAS])
    all_visible = [True] * (len(MODES) * len(ALPHAS))
    buttons = [{"label": "all α", "method": "update", "args": [{"visible": all_visible}]}]
    buttons.extend(
        {"label": f"α = {alpha:g}", "method": "update", "args": [{"visible": values}]}
        for alpha, values in zip(ALPHAS, visibility, strict=True)
    )
    figure.update_layout(
        title={"text": f"FEAST batch-effect simulation — interactive local parameter-cloud view ({len(gene_index):,} genes)", "x": 0.5, "xanchor": "center", "font": {"size": 22}},
        template="plotly_white",
        height=760,
        width=1400,
        margin={"l": 20, "r": 20, "t": 110, "b": 30},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.02, "yanchor": "bottom"},
        updatemenus=[{"type": "buttons", "direction": "right", "x": 0.5, "xanchor": "center", "y": 1.13, "yanchor": "top", "buttons": buttons}],
        scene=scene_layout(),
        scene2=scene_layout(),
    )
    figure.add_annotation(
        text="Drag to rotate · scroll to zoom · use the mode-bar home icon to reset · selector controls which α levels are visible",
        x=0.5,
        y=1.065,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 13, "color": "#4D4D4D"},
    )
    figure.write_html(OUTPUT_PATH, include_plotlyjs=True, full_html=True, config={"scrollZoom": True, "displaylogo": False})
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
