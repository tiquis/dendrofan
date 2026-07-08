"""
Core polar-coordinate geometry for circular ("fan") dendrograms.

This module contains the one genuinely reusable idea behind a circular
dendrogram: a pair of coordinate maps that take the rectangular
coordinates produced by ``scipy.cluster.hierarchy.dendrogram`` --
``icoord`` (leaf positions along x) and ``dcoord`` (linkage heights along
y) -- and bend them onto an annulus.

Everything else in dendrofan (clustering, styling, plotting) builds on
top of :class:`PolarTransform`.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Callable, Tuple, Union

import numpy as np

RadiusScaleName = str  # "linear" or "sqrt"
RadiusScale = Union[RadiusScaleName, Callable[[np.ndarray], np.ndarray]]


@dataclasses.dataclass(frozen=True)
class PolarTransform:
    """Maps rectangular dendrogram coordinates onto an annulus.

    A rectangular dendrogram, as produced by
    ``scipy.cluster.hierarchy.dendrogram``, lives on a plane where the
    x-axis enumerates leaves (evenly spaced, 10 units apart by SciPy's
    convention) and the y-axis is the linkage/merge height, with 0 at
    the leaves and increasing towards the root.

    This class re-maps that plane onto polar coordinates:

    - x (leaf position)  -> theta (angle), so leaves spread around a
      circle instead of sitting on a line.
    - y (linkage height) -> r (radius), so the root sits near the
      centre and leaves sit on the outer rim (or the reverse, if
      ``invert_radius=False``).

    Parameters
    ----------
    x_min, x_max : float
        The extent of the x-axis in the source rectangular dendrogram
        (``icoord.min()`` / ``icoord.max()``).
    y_max : float
        The maximum linkage height in the source dendrogram
        (``dcoord.max()``). If this is ``0`` (a degenerate tree where
        every merge happens at height 0), the transform falls back to
        placing every internal node at the outer radius instead of
        raising a ZeroDivisionError.
    start_angle : float, default 90
        Angle, in degrees, of the *first* leaf, measured counter-
        clockwise from the positive x-axis (standard math convention).
        90 puts the first leaf pointing straight up, matching the
        default orientation used by ``ape::plot.phylo(type = "fan")``.
    span : float, default 360
        Total angular extent, in degrees, that the leaves are spread
        across. Use a value below 360 (e.g. 350) to leave an angular
        gap between the first and last leaf, which is useful when
        labels would otherwise collide at the seam -- this mirrors the
        partial-fan idiom available in ``ape``.
    clockwise : bool, default True
        If True, leaves are laid out clockwise from ``start_angle``
        (the conventional reading direction for circular dendrograms
        and the ``ape`` default). If False, they are laid out counter-
        clockwise.
    inner_radius : float, default 0.0
        Radius of the root (or of the centre, if ``inner_radius`` is 0).
        A small positive value reproduces the central gap conventional
        in ``ape``'s fan style and avoids a cluttered centre.
    outer_radius : float, default 1.0
        Radius of the leaves.
    invert_radius : bool, default True
        If True (default), y=0 (leaves) maps to ``outer_radius`` and
        y=y_max (root) maps to ``inner_radius`` -- the usual dendrogram
        reading, root at the centre. If False, the mapping is reversed
        (root at the rim) -- rarely useful, provided for completeness.
    radius_scale : {"linear", "sqrt"} or callable, default "linear"
        How linkage height is mapped to radius.

        - ``"linear"``: radius is a linear function of height. This
          matches the original ad hoc script and is the correct choice
          when the *radial distance* should be read quantitatively
          (e.g. with a distance-scale ring).
        - ``"sqrt"``: radius is proportional to the square root of the
          (rescaled) height. Because the area of an annulus grows with
          the square of its radius, a linear height mapping visually
          compresses shallow, near-root merges into a tiny central
          area and exaggerates the area given to merges near the rim.
          The square-root mapping keeps the *area* allotted to a given
          range of merge heights roughly constant across radii, which
          is often more legible for trees with many leaves and a long
          tail of shallow merges. This is a purely cosmetic remapping
          and should not be used if radial distances will be read
          quantitatively.
        - a callable ``f(y_norm) -> r_norm`` mapping normalised height
          in [0, 1] to normalised radius in [0, 1], for full control.

    Notes
    -----
    All angles are handled internally in radians; degrees are only
    used at the public boundary (``start_angle``, ``span``) because
    that is what most users reason in.
    """

    x_min: float
    x_max: float
    y_max: float
    start_angle: float = 90.0
    span: float = 360.0
    clockwise: bool = True
    inner_radius: float = 0.0
    outer_radius: float = 1.0
    invert_radius: bool = True
    radius_scale: RadiusScale = "linear"

    def __post_init__(self) -> None:
        if self.x_max < self.x_min:
            raise ValueError("x_max must be >= x_min")
        if self.y_max < 0:
            raise ValueError("y_max must be >= 0")
        if not (0 < self.span <= 360):
            raise ValueError("span must be in (0, 360]")
        if self.outer_radius <= self.inner_radius:
            raise ValueError("outer_radius must be > inner_radius")
        if self.inner_radius < 0:
            raise ValueError("inner_radius must be >= 0")

    # -- angle ---------------------------------------------------------

    def theta(self, x):
        """Map rectangular x-coordinate(s) (leaf axis) to angle(s), in radians."""
        x = np.asarray(x, dtype=float)
        x_range = self.x_max - self.x_min
        if x_range == 0:
            # A single leaf: park it at start_angle.
            frac = np.zeros_like(x)
        else:
            frac = (x - self.x_min) / x_range
        span_rad = math.radians(self.span)
        start_rad = math.radians(self.start_angle)
        signed_span = -span_rad if self.clockwise else span_rad
        return start_rad + frac * signed_span

    # -- radius ----------------------------------------------------------

    def radius(self, y):
        """Map rectangular y-coordinate(s) (linkage height) to radius/radii."""
        y = np.asarray(y, dtype=float)
        if self.y_max == 0:
            # Degenerate tree: every merge happens at height 0. There is
            # no meaningful radial ordering, so place every node at the
            # leaf radius rather than dividing by zero.
            y_norm = np.zeros_like(y)
        else:
            y_norm = np.clip(y / self.y_max, 0.0, 1.0)

        r_norm = self._apply_radius_scale(y_norm)

        if self.invert_radius:
            r_norm = 1.0 - r_norm

        return self.inner_radius + (self.outer_radius - self.inner_radius) * r_norm

    def _apply_radius_scale(self, y_norm):
        scale = self.radius_scale
        if callable(scale):
            return scale(y_norm)
        if scale == "linear":
            return y_norm
        if scale == "sqrt":
            return np.sqrt(y_norm)
        raise ValueError(
            f"radius_scale must be 'linear', 'sqrt', or a callable, got {scale!r}"
        )

    # -- combined --------------------------------------------------------

    def to_cartesian(self, x, y):
        """Map rectangular (x, y) dendrogram coordinates straight to (X, Y)."""
        t = self.theta(x)
        r = self.radius(y)
        return r * np.cos(t), r * np.sin(t)


def arc_points(transform, x_start, x_end, y, n_points=50):
    """Cartesian points tracing a constant-radius arc between two leaf positions.

    Used to draw the horizontal bar of a dendrogram merge (a straight
    line in rectangular coordinates) as a properly curved arc once
    bent onto the annulus.
    """
    xs = np.linspace(x_start, x_end, n_points)
    ys = np.full(n_points, y)
    return transform.to_cartesian(xs, ys)


def radial_label_alignment(theta_rad: float) -> Tuple[float, str]:
    """Rotation (degrees) and horizontal alignment for a label at angle ``theta_rad``.

    Labels placed just outside the rim of a circular dendrogram should
    read left-to-right when possible instead of upside down. This
    returns the text rotation (in degrees, as expected by
    ``matplotlib``'s ``rotation`` kwarg with ``rotation_mode="anchor"``)
    and the horizontal alignment (``"left"`` or ``"right"``) so that
    text on the left half of the circle is flipped 180 degrees and
    right-aligned, keeping it upright and pointing away from the tree.
    """
    angle_deg = math.degrees(theta_rad) % 360.0
    # Normalise to (-180, 180] for the upright/flipped test below.
    signed_deg = angle_deg if angle_deg <= 180 else angle_deg - 360
    if -90 <= signed_deg <= 90:
        return signed_deg, "left"
    flipped = signed_deg + 180 if signed_deg < 0 else signed_deg - 180
    return flipped, "right"
