"""Public plotting API: draw a circular ("fan") dendrogram."""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .clustering import ArrayLike, DendrogramLayout, build_layout
from .geometry import PolarTransform, arc_points, radial_label_alignment
from .styling import ColorMap, resolve_leaf_colors


@dataclasses.dataclass
class CircularDendrogramResult:
    """Everything a caller might need to annotate a circular dendrogram further.

    Attributes
    ----------
    fig, ax : matplotlib Figure / Axes
        The figure and axes the tree was drawn on.
    layout : DendrogramLayout
        The rectangular-dendrogram layout the tree was built from.
    transform : PolarTransform
        The polar transform used, so callers can map their own extra
        rectangular coordinates (or additional linkage heights) into
        the same coordinate system, e.g. to add a scale ring with
        :func:`dendrofan.annotations.add_scale_ring`.
    leaf_theta : dict of {str: float}
        Angle (radians) of each leaf label's centre, keyed by the
        (possibly duplicated) display label as drawn. For unique
        per-leaf angles regardless of label text, use ``leaf_theta_by_index``.
    leaf_theta_by_index : list of float
        Angle (radians) of each leaf, in the same left-to-right order
        as ``layout.leaf_labels`` / ``layout.leaves``.
    """

    fig: Figure
    ax: Axes
    layout: DendrogramLayout
    transform: PolarTransform
    leaf_theta: Dict[str, float]
    leaf_theta_by_index: List[float]


def circular_dendrogram(
    data: Optional[ArrayLike] = None,
    *,
    Z: Optional[np.ndarray] = None,
    condensed_distances: Optional[ArrayLike] = None,
    labels: Optional[Sequence[str]] = None,
    metric: str = "euclidean",
    method: str = "ward",
    color_threshold: Optional[float] = None,
    link_color_func=None,
    truncate_mode: Optional[str] = None,
    p: int = 30,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 10),
    start_angle: float = 90.0,
    span: float = 360.0,
    clockwise: bool = True,
    inner_radius: float = 0.12,
    outer_radius: float = 1.0,
    radius_scale: str = "linear",
    branch_color: Optional[str] = "black",
    branch_linewidth: float = 1.0,
    n_arc_points: int = 50,
    show_leaf_labels: bool = True,
    label_colors: Optional[ColorMap] = None,
    label_default_color: str = "black",
    label_formatter=None,
    label_fontsize: float = 10,
    label_offset: float = 0.04,
    label_fontstyle: str = "normal",
    label_fontweight: str = "normal",
    leaf_marker: Optional[str] = None,
    leaf_marker_size: float = 20,
) -> CircularDendrogramResult:
    """Draw a circular ("fan"/"ape"-style) dendrogram.

    This is the circular analogue of
    ``scipy.cluster.hierarchy.dendrogram``: it takes the same kinds of
    inputs (raw observations, a condensed distance vector, or a
    precomputed linkage matrix), builds the same tree, and draws it
    bent onto an annulus instead of a rectangle -- in the visual style
    of ``ape::plot.phylo(type = "fan")`` in R (Paradis & Schliep, 2019),
    computed and rendered entirely with SciPy and Matplotlib.

    Exactly one of ``data``, ``Z``, or ``condensed_distances`` must be
    given; see :func:`dendrofan.clustering.build_layout` for details on
    each. All clustering/tree-building parameters
    (``metric``, ``method``, ``color_threshold``, ``link_color_func``,
    ``truncate_mode``, ``p``) are forwarded there unchanged.

    Parameters
    ----------
    labels : sequence of str, optional
        One label per leaf/observation. Defaults to stringified indices.
    ax : matplotlib Axes, optional
        Axes to draw on. If omitted, a new square figure is created.
        The axes are always plain Cartesian (not a matplotlib polar
        projection) so that label rotation, partial fans, and inward
        roots are all under direct control; the polar math itself is
        handled by :class:`dendrofan.geometry.PolarTransform` and
        applied before anything is plotted.
    figsize : (float, float), default (10, 10)
        Figure size when ``ax`` is not given. Ignored if ``ax`` is given.
    start_angle, span, clockwise, inner_radius, outer_radius, radius_scale
        Forwarded to :class:`dendrofan.geometry.PolarTransform`; see
        its docstring for the full explanation of each. In short:
        ``span=350`` (instead of 360) opens a gap for labels at the
        seam, and ``radius_scale="sqrt"`` trades quantitative radial
        distance for better legibility on trees with many leaves.
    branch_color : str, optional
        Color for every branch. If ``None``, each merge is colored
        according to SciPy's own ``color_list`` (meaningful only when
        ``color_threshold`` or ``link_color_func`` is also set --
        otherwise SciPy colors everything the same default color).
    branch_linewidth : float, default 1.0
        Line width for radial segments and arcs.
    n_arc_points : int, default 50
        Number of straight-line segments used to approximate each
        curved merge arc. Higher values give smoother arcs at a small
        performance cost; 50 is smooth even for large figures.
    show_leaf_labels : bool, default True
        If False, no leaf labels are drawn (useful when leaves will be
        labeled by an external legend or when there are too many to
        read individually).
    label_colors : dict or callable, optional
        Per-leaf label color; see :func:`dendrofan.styling.resolve_leaf_colors`.
        Typically a ``{group_name: color}``-style dict combined with a
        ``label -> group_name`` lookup, or directly a
        ``{label: color}`` dict.
    label_default_color : str, default "black"
        Fallback color for labels not covered by ``label_colors``.
    label_formatter : callable, optional
        ``label -> display_string``, applied to each leaf label right
        before drawing (e.g. to append an annotation such as
        ``" (excl.)"``). The unmodified label is still used to look up
        ``label_colors``.
    label_fontsize, label_fontstyle, label_fontweight
        Passed straight through to ``ax.text``.
    label_offset : float, default 0.04
        Radial gap between the outer rim (``outer_radius``) and the
        start of each leaf label, in the same units as ``outer_radius``.
    leaf_marker : str, optional
        Matplotlib marker style (e.g. ``"o"``) drawn at each leaf tip.
        ``None`` (default) draws no marker, matching the reference
        figure style of the source manuscript.
    leaf_marker_size : float, default 20
        Marker size (``s`` in ``ax.scatter`` units), used only if
        ``leaf_marker`` is not ``None``.

    Returns
    -------
    CircularDendrogramResult
        The figure/axes plus the layout and transform needed to add
        further annotations (scale rings, sector highlights, etc. --
        see :mod:`dendrofan.annotations`).

    Examples
    --------
    >>> import numpy as np
    >>> from dendrofan import circular_dendrogram
    >>> rng = np.random.default_rng(0)
    >>> data = rng.normal(size=(12, 5))
    >>> result = circular_dendrogram(
    ...     data,
    ...     labels=[f"sample_{i}" for i in range(12)],
    ...     span=350,               # leave a small gap at the seam
    ...     inner_radius=0.15,
    ... )
    >>> result.fig.savefig("tree.png", dpi=200)
    """
    layout = build_layout(
        data=data,
        Z=Z,
        condensed_distances=condensed_distances,
        labels=labels,
        metric=metric,
        method=method,
        color_threshold=color_threshold,
        link_color_func=link_color_func,
        truncate_mode=truncate_mode,
        p=p,
    )

    transform = PolarTransform(
        x_min=layout.x_min,
        x_max=layout.x_max,
        y_max=layout.y_max,
        start_angle=start_angle,
        span=span,
        clockwise=clockwise,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        radius_scale=radius_scale,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.set_aspect("equal")

    _draw_branches(
        ax,
        layout,
        transform,
        branch_color=branch_color,
        branch_linewidth=branch_linewidth,
        n_arc_points=n_arc_points,
    )

    leaf_colors = resolve_leaf_colors(
        layout.leaf_labels, colors=label_colors, default=label_default_color
    )

    leaf_theta_by_index = _leaf_angles(layout, transform)

    if leaf_marker is not None:
        xs, ys = transform.to_cartesian(
            np.array([_leaf_x(layout, i) for i in range(layout.n_leaves)]),
            np.zeros(layout.n_leaves),
        )
        ax.scatter(xs, ys, s=leaf_marker_size, marker=leaf_marker, color=leaf_colors, zorder=3)

    leaf_theta: Dict[str, float] = {}
    if show_leaf_labels:
        for i, (label, color, theta) in enumerate(
            zip(layout.leaf_labels, leaf_colors, leaf_theta_by_index)
        ):
            display = label_formatter(label) if label_formatter else label
            rotation, ha = radial_label_alignment(theta)
            r_text = outer_radius + label_offset
            x_txt, y_txt = r_text * np.cos(theta), r_text * np.sin(theta)
            ax.text(
                x_txt,
                y_txt,
                display,
                rotation=rotation,
                rotation_mode="anchor",
                ha=ha,
                va="center",
                fontsize=label_fontsize,
                fontstyle=label_fontstyle,
                fontweight=label_fontweight,
                color=color,
            )
            leaf_theta[label] = theta

    margin = outer_radius * 0.85 + label_offset
    limit = outer_radius + margin
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.axis("off")

    return CircularDendrogramResult(
        fig=fig,
        ax=ax,
        layout=layout,
        transform=transform,
        leaf_theta=leaf_theta,
        leaf_theta_by_index=leaf_theta_by_index,
    )


def _leaf_x(layout: DendrogramLayout, i: int) -> float:
    """x-coordinate (rectangular dendrogram axis) of the i-th leaf, left to right.

    SciPy always spaces leaves 10 units apart starting at 5 (5, 15, 25, ...);
    this is reconstructed rather than assumed, by anchoring to the observed
    ``icoord`` extent, so it stays correct under ``truncate_mode``.
    """
    if layout.n_leaves == 1:
        return (layout.x_min + layout.x_max) / 2.0
    step = (layout.x_max - layout.x_min) / (layout.n_leaves - 1)
    return layout.x_min + i * step


def _leaf_angles(layout: DendrogramLayout, transform: PolarTransform) -> List[float]:
    xs = [_leaf_x(layout, i) for i in range(layout.n_leaves)]
    return [float(transform.theta(x)) for x in xs]


def _draw_branches(
    ax: Axes,
    layout: DendrogramLayout,
    transform: PolarTransform,
    *,
    branch_color: Optional[str],
    branch_linewidth: float,
    n_arc_points: int,
) -> None:
    for row, (xs, ys) in enumerate(zip(layout.icoord, layout.dcoord)):
        x1, x2, x3, x4 = xs
        y1, y2, y3, y4 = ys
        color = branch_color if branch_color is not None else layout.color_list[row]

        # Left radial segment: from the left child up to the merge height.
        xr, yr = transform.to_cartesian(np.array([x1, x2]), np.array([y1, y2]))
        ax.plot(xr, yr, color=color, linewidth=branch_linewidth)

        # Right radial segment: from the right child up to the merge height.
        xr, yr = transform.to_cartesian(np.array([x4, x3]), np.array([y4, y3]))
        ax.plot(xr, yr, color=color, linewidth=branch_linewidth)

        # Arc joining the two children at the (constant) merge height.
        xr, yr = arc_points(transform, x2, x3, y2, n_points=n_arc_points)
        ax.plot(xr, yr, color=color, linewidth=branch_linewidth)
