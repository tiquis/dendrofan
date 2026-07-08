"""Optional decorations for a circular dendrogram: scale rings and sector highlights.

These operate on the ``ax``/``transform`` pair returned by
:func:`dendrofan.plotting.circular_dendrogram`, so they can be added or
omitted independently of the tree itself.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from matplotlib.axes import Axes

from .geometry import PolarTransform


def add_scale_ring(
    ax: Axes,
    transform: PolarTransform,
    heights: Sequence[float],
    *,
    color: str = "gray",
    linewidth: float = 0.6,
    linestyle: str = "--",
    label_heights: bool = True,
    label_fontsize: float = 8,
    label_angle: float = 90.0,
) -> None:
    """Draw concentric guide circles at the given linkage heights.

    The circular analogue of the distance axis ``ape::axisPhylo()``
    draws on a rectangular/fan tree: a light dashed ring at each height
    in ``heights``, optionally labeled with the height value, so
    readers can gauge merge distances even though a circular layout
    has no literal axis line to read off.

    Parameters
    ----------
    heights : sequence of float
        Linkage heights (in the original units of the tree, i.e. of
        ``dcoord`` / the linkage matrix) at which to draw a ring.
        Heights outside ``[0, transform.y_max]`` are skipped.
    label_angle : float, default 90
        Angle, in degrees, at which each ring's height label is placed
        (90 = top of the circle, matching ``start_angle``'s default).
    """
    theta = np.radians(label_angle)
    for h in heights:
        if h < 0 or h > transform.y_max:
            continue
        r = float(transform.radius(h))
        circle = np.linspace(0, 2 * np.pi, 200)
        ax.plot(
            r * np.cos(circle),
            r * np.sin(circle),
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=0,
        )
        if label_heights:
            ax.text(
                r * np.cos(theta),
                r * np.sin(theta),
                f"{h:g}",
                fontsize=label_fontsize,
                color=color,
                ha="center",
                va="bottom",
            )


def highlight_sector(
    ax: Axes,
    transform: PolarTransform,
    theta_start: float,
    theta_end: float,
    *,
    r_inner: Optional[float] = None,
    r_outer: Optional[float] = None,
    color: str = "gray",
    alpha: float = 0.15,
    n_points: int = 100,
    zorder: float = -1,
) -> None:
    """Shade a wedge of the plot behind a clade, the way ``ape``'s clade
    highlighting (e.g. via ``ape::ring`` or manually drawn background
    wedges) marks a group of related leaves.

    Parameters
    ----------
    theta_start, theta_end : float
        Angular bounds of the wedge, in radians (e.g. from
        ``CircularDendrogramResult.leaf_theta_by_index``).
    r_inner, r_outer : float, optional
        Radial bounds of the wedge. Default to
        ``transform.inner_radius`` and ``transform.outer_radius``
        respectively, i.e. the whole tree depth.
    """
    r_inner = transform.inner_radius if r_inner is None else r_inner
    r_outer = transform.outer_radius if r_outer is None else r_outer

    thetas = np.linspace(theta_start, theta_end, n_points)
    outer_x, outer_y = r_outer * np.cos(thetas), r_outer * np.sin(thetas)
    inner_x, inner_y = r_inner * np.cos(thetas[::-1]), r_inner * np.sin(thetas[::-1])

    xs = np.concatenate([outer_x, inner_x])
    ys = np.concatenate([outer_y, inner_y])
    ax.fill(xs, ys, color=color, alpha=alpha, linewidth=0, zorder=zorder)
