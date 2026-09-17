"""Pathway plotting utilities, including rank-overlap plots."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd


def _pathway_connections(right_ids, left_ids, connect_superpathways, hierarchy):
    """Match identical IDs or ancestor/descendant IDs in a directed hierarchy."""
    children = {}
    if connect_superpathways:
        if hierarchy is None:
            hierarchy = Path(__file__).with_name("pathway_hierarchy.csv")
        table = hierarchy if isinstance(hierarchy, pd.DataFrame) else pd.read_csv(hierarchy)
        if not {"superpathway", "subpathway"}.issubset(table.columns):
            raise ValueError("pathway_hierarchy requires superpathway and subpathway columns.")
        if table[["superpathway", "subpathway"]].isna().any().any():
            raise ValueError("Pathway hierarchy IDs must not be missing.")
        for parent, child in table[["superpathway", "subpathway"]].itertuples(index=False, name=None):
            children.setdefault(parent, set()).add(child)

    def descendants(parent):
        seen = set()
        pending = list(children.get(parent, ()))
        while pending:
            child = pending.pop()
            if child == parent:
                raise ValueError("Pathway hierarchy contains a cycle.")
            if child not in seen:
                seen.add(child)
                pending.extend(children.get(child, ()))
        return seen

    closure = {parent: descendants(parent) for parent in children}
    return [
        (left, right, "exact" if left == right else "superpathway")
        for right in right_ids for left in left_ids
        if (left == right or right in closure.get(left, ())
            or left in closure.get(right, ()))
    ]


def plot_rank_overlap(
    series1: pd.Series,
    series2: pd.Series,
    fig_path=None,
    fig_name=None,
    figsize=(6, 5),
    line_alpha=0.8,
    column_gap=0.30,
    label_map=None,
    label_fontsize=10,
    header_fontsize=11,
    ascending=True,
    line_color=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    colorbar_label=None,
    curvature=0.0,
    connect_superpathways=True,
    pathway_hierarchy=None,
):
    """Connect shared IDs between two independently sorted Series.

    series1 contains epistatic pathways (right); series2 contains main effect
    pathways (left).
    Smaller column_gap brings columns closer. Reduce figsize[1] for closer
    rows. label_map changes display labels only; matching uses original IDs.
    connect_superpathways adds ancestor/descendant matches from the bundled
    SGD hierarchy. Set False for exact-ID matches only. pathway_hierarchy can
    supply a CSV path or DataFrame with superpathway and subpathway columns.
    The bundled table covers verified relationships for the local pathway
    catalog, not the complete SGD hierarchy. No live network calls are made.
    ax.pathway_connections records main_effect, epistatic and relationship.
    ascending=True puts the smallest values first. Headers align with the
    inner edges of the corresponding pathway labels. Connectors use rainbow
    colors by default; set line_color to give all connectors one color.
    line_alpha controls connector opacity. Returns (fig, ax).
    Squares inside each label column encode each series' values using cmap.
    vmin/vmax default to the combined series minimum/maximum. Squares match
    their label's rendered height; shared pathways are connected by lines.
    A vertical colorbar occupies the gap beneath the right column when it
    fits, otherwise it is placed below the plot at the lower right.
    colorbar_label optionally labels the colorbar.
    Connectors have two elbows with horizontal segments next to the squares.
    curvature controls corner rounding: 0 gives sharp elbows; magnitudes up
    to 0.5 increasingly round both corners (the sign is ignored).
    """
    if not 0 < column_gap < 1:
        raise ValueError("column_gap must be between 0 and 1.")
    if not np.isscalar(curvature) or not np.isfinite(curvature):
        raise ValueError("curvature must be a finite number.")
    if not series1.index.is_unique or not series2.index.is_unique:
        raise ValueError("Each series must have unique pathway IDs.")
    if series1.empty and series2.empty:
        raise ValueError("At least one series must be nonempty.")
    if fig_path is not None and fig_name is None:
        raise ValueError("Provide fig_name when specifying fig_path.")

    series1 = pd.to_numeric(series1, errors="raise")
    series2 = pd.to_numeric(series2, errors="raise")
    values = np.concatenate([series1.to_numpy(dtype=float), series2.to_numpy(dtype=float)])
    if not np.isfinite(values).all():
        raise ValueError("Series values must be finite numbers.")
    vmin = float(values.min()) if vmin is None else float(vmin)
    vmax = float(values.max()) if vmax is None else float(vmax)
    if not np.isfinite([vmin, vmax]).all() or vmin > vmax:
        raise ValueError("Color limits must be finite with vmin <= vmax.")
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    color_map = plt.get_cmap(cmap)
    label_map = {} if label_map is None else label_map
    s1 = series1.sort_values(ascending=ascending, kind="stable")
    s2 = series2.sort_values(ascending=ascending, kind="stable")
    rank1 = {idx: i for i, idx in enumerate(s1.index)}
    rank2 = {idx: i for i, idx in enumerate(s2.index)}
    connections = _pathway_connections(
        rank1, rank2, connect_superpathways, pathway_hierarchy)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.pathway_connections = pd.DataFrame(
        connections, columns=["main_effect", "epistatic", "relationship"])
    x1 = 0.5 - column_gap / 2
    x2 = 0.5 + column_gap / 2
    label_markers = []
    for series, ranks, x, alignment, direction in (
        (s2, rank2, x1, "right", -1),
        (s1, rank1, x2, "left", 1),
    ):
        for idx, rank in ranks.items():
            label = ax.annotate(
                str(label_map.get(idx, idx)), (x, rank),
                xytext=(direction * (label_fontsize / 2 + 4), 0),
                textcoords="offset points",
                ha=alignment, multialignment=alignment, va="center",
                fontsize=label_fontsize,
            )
            marker, = ax.plot(
                x, rank, marker="s", linestyle="none", markersize=label_fontsize,
                markerfacecolor=color_map(norm(series.loc[idx])),
                markeredgewidth=0, zorder=3,
            )
            label_markers.append((label, marker, direction))

    rainbow = plt.get_cmap("rainbow")
    for position, (left, right, relationship) in enumerate(connections):
        start = np.array([x1, rank2[left]])
        end = np.array([x2, rank1[right]])
        elbow1 = np.array([x1 + column_gap / 3, rank2[left]])
        elbow2 = np.array([x2 - column_gap / 3, rank1[right]])
        rounding = min(abs(float(curvature)), 0.5)
        vertices = [
            start,
            elbow1 + rounding * (start - elbow1),
            elbow1,
            elbow1 + rounding * (elbow2 - elbow1),
            elbow2 + rounding * (elbow1 - elbow2),
            elbow2,
            elbow2 + rounding * (end - elbow2),
            end,
        ]
        codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3,
                 MplPath.CURVE3, MplPath.LINETO, MplPath.CURVE3,
                 MplPath.CURVE3, MplPath.LINETO]
        ax.add_patch(PathPatch(
            MplPath(vertices, codes), facecolor="none",
            edgecolor=(rainbow(position / max(len(connections) - 1, 1))
                   if line_color is None else line_color),
            alpha=line_alpha, linewidth=1.5, zorder=2,
        ))

    # Share label x positions and alignment, including each multiline header.
    transform = ax.get_xaxis_transform()
    for x, title, alignment in (
        (x1, "Main effect\npathways", "right"),
        (x2, "Epistatic\npathways", "left"),
    ):
        ax.text(
            x, 1.03, title, transform=transform,
            ha=alignment, multialignment=alignment, va="bottom",
            fontsize=header_fontsize,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(max(len(s1), len(s2)) - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Marker dimensions and label offsets are in points, independent of axis
    # scaling. Match actual text height, including multiline pathway labels.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label, marker, direction in label_markers:
        height = label.get_window_extent(renderer).height * 72 / fig.dpi
        marker.set_markersize(height)
        label.set_position((direction * (height / 2 + 4), 0))

    # Measure the free space below the epistatic labels, including multiline
    # text. Keep the bar to the right of all connector endpoints.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    right_bottom = min(
        (ax.transAxes.inverted().transform(
            (0, label.get_window_extent(renderer).y0))[1]
         for label, _, direction in label_markers if direction == 1),
        default=1.0,
    )
    available_height = right_bottom - .12
    if len(s1) < len(s2) and available_height >= .18:
        bar_bottom, bar_height = .05, min(.30, available_height)
    else:
        bar_bottom, bar_height = -.40, .30
    cax = ax.inset_axes([x2 + .07, bar_bottom, .025, bar_height])
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=color_map), cax=cax,
        orientation="vertical",
    )
    colorbar.ax.tick_params(labelsize=label_fontsize)
    if colorbar_label is not None:
        colorbar.set_label(colorbar_label, fontsize=label_fontsize)
    colorbar.outline.set_visible(False)
    ax.pathway_colorbar = colorbar
    # Colorbar expands a constant-valued normalization; refresh square colors.
    for series, direction in ((s2, -1), (s1, 1)):
        markers = [marker for _, marker, side in label_markers if side == direction]
        for marker, value in zip(markers, series):
            marker.set_markerfacecolor(color_map(norm(value)))

    if fig_path is not None:
        fig.savefig(Path(fig_path) / fig_name, bbox_inches="tight")
    plt.show()
    return fig, ax
