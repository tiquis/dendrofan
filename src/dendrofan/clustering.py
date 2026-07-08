"""
Turn raw data, a condensed distance vector, or a precomputed linkage
matrix into the rectangular-dendrogram layout that :mod:`dendrofan.geometry`
bends into a circle.

This module is a thin, validating wrapper around
``scipy.cluster.hierarchy`` -- dendrofan does not reimplement clustering,
it only makes the SciPy output safe and convenient to consume for
circular rendering.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence, Union

import numpy as np
from scipy.cluster.hierarchy import dendrogram as _scipy_dendrogram
from scipy.cluster.hierarchy import is_valid_linkage, linkage as _scipy_linkage
from scipy.spatial.distance import is_valid_y, pdist

from .exceptions import DegenerateTreeError, InvalidLinkageError, LabelMismatchError


@dataclasses.dataclass
class DendrogramLayout:
    """Rectangular-coordinate layout of a dendrogram, ready to be bent into a circle.

    This mirrors (a validated, typed subset of) the dict returned by
    ``scipy.cluster.hierarchy.dendrogram``.

    Attributes
    ----------
    Z : ndarray
        The linkage matrix underlying the tree.
    icoord, dcoord : ndarray
        Per-merge x (leaf-axis) and y (height-axis) coordinates, shape
        ``(n_merges, 4)``, exactly as returned by SciPy: for merge *i*,
        ``(icoord[i, 0], dcoord[i, 0])`` and ``(icoord[i, 3], dcoord[i, 3])``
        are the two outer (leaf-ward) corners of the merge's "U" shape,
        and ``dcoord[i, 1] == dcoord[i, 2]`` is the merge height.
    leaves : list of int
        Original observation indices, in left-to-right leaf order.
    leaf_labels : list of str
        Display labels, in the same left-to-right order as ``leaves``.
    color_list : list of str
        Per-merge color, as assigned by SciPy's ``color_threshold`` /
        ``link_color_func`` machinery (one entry per row of ``icoord``).
    n_leaves : int
        Number of leaves in the tree.
    """

    Z: np.ndarray
    icoord: np.ndarray
    dcoord: np.ndarray
    leaves: List[int]
    leaf_labels: List[str]
    color_list: List[str]

    @property
    def n_leaves(self) -> int:
        return len(self.leaves)

    @property
    def x_min(self) -> float:
        return float(self.icoord.min())

    @property
    def x_max(self) -> float:
        return float(self.icoord.max())

    @property
    def y_max(self) -> float:
        return float(self.dcoord.max())


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


def _validate_linkage_index_bounds(Z: np.ndarray) -> None:
    """Check that every merge in ``Z`` only references clusters formed so far.

    ``scipy.cluster.hierarchy.is_valid_linkage`` is not sufficient on its
    own: at least as of SciPy 1.17, it fails to flag a linkage matrix
    whose child indices point past the clusters available at that row
    (verified to correctly reject the same input under SciPy 1.13). Left
    unchecked, such a matrix passes validation but then crashes
    ``dendrogram()`` deep inside SciPy with a bare ``IndexError`` instead
    of a clear dendrofan exception. This redoes that specific check
    ourselves so behavior does not depend on the installed SciPy version.
    """
    n = Z.shape[0] + 1
    for i, row in enumerate(Z):
        left, right = row[0], row[1]
        upper_bound = n + i  # valid cluster ids at merge i: 0 .. n+i-1
        if not (0 <= left < upper_bound and 0 <= right < upper_bound):
            raise InvalidLinkageError(
                f"row {i} of `Z` references cluster index {row[:2].tolist()}, "
                f"but only clusters 0..{upper_bound - 1} exist at that point"
            )


def _validate_labels(labels: Optional[Sequence[str]], n_leaves: int) -> List[str]:
    if labels is None:
        return [str(i) for i in range(n_leaves)]
    labels = list(labels)
    if len(labels) != n_leaves:
        raise LabelMismatchError(
            f"got {len(labels)} labels for {n_leaves} leaves; "
            "labels must have exactly one entry per observation"
        )
    return labels


def build_layout(
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
) -> DendrogramLayout:
    """Build a validated :class:`DendrogramLayout` from one of three inputs.

    Exactly one of ``data``, ``Z``, or ``condensed_distances`` should be
    given:

    - ``data``: an ``(n_observations, n_features)`` array of raw
      observations. Pairwise distances (``metric``) and linkage
      (``method``) are computed for you, mirroring what the original
      ad hoc script did with ``pdist`` + ``linkage``.
    - ``Z``: a precomputed linkage matrix (e.g. your own call to
      ``scipy.cluster.hierarchy.linkage``, or ``R``'s ``hclust`` output
      converted with ``scipy``-compatible tooling). Validated with
      ``scipy.cluster.hierarchy.is_valid_linkage`` before use.
    - ``condensed_distances``: a precomputed condensed distance vector
      (as returned by ``scipy.spatial.distance.pdist``); ``method`` is
      applied to it to obtain ``Z``.

    Parameters
    ----------
    labels : sequence of str, optional
        One label per leaf, in the same order as the rows of ``data``
        (or the original observations behind ``Z`` /
        ``condensed_distances``). Defaults to stringified indices.
    color_threshold, link_color_func, truncate_mode, p
        Passed straight through to ``scipy.cluster.hierarchy.dendrogram``;
        see its docstring. Use ``color_threshold`` to have SciPy assign
        per-cluster colors automatically (a common way to visually mark
        clades below a chosen cut height).

    Raises
    ------
    DendrofanError
        If none or more than one of ``data``/``Z``/``condensed_distances``
        is given, if a given ``Z`` fails SciPy's validity check, if
        ``labels`` does not have one entry per leaf, or if the resulting
        tree has fewer than 2 leaves (nothing meaningful to draw).
    """
    n_inputs_given = sum(x is not None for x in (data, Z, condensed_distances))
    if n_inputs_given != 1:
        raise ValueError(
            "pass exactly one of `data`, `Z`, or `condensed_distances` "
            f"(got {n_inputs_given})"
        )

    if data is not None:
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError("`data` must be a 2-D array (n_observations, n_features)")
        if data.shape[0] < 2:
            raise DegenerateTreeError(
                f"need at least 2 observations to build a tree, got {data.shape[0]}"
            )
        condensed = pdist(data, metric=metric)
        Z = _scipy_linkage(condensed, method=method)
    elif condensed_distances is not None:
        condensed = np.asarray(condensed_distances, dtype=float)
        if not is_valid_y(condensed):
            raise InvalidLinkageError(
                "`condensed_distances` is not a valid condensed distance vector "
                "(check its length matches n*(n-1)/2 for some integer n)"
            )
        Z = _scipy_linkage(condensed, method=method)
    else:
        Z = np.asarray(Z, dtype=float)
        if not is_valid_linkage(Z):
            raise InvalidLinkageError(
                "`Z` is not a valid SciPy linkage matrix; see "
                "scipy.cluster.hierarchy.is_valid_linkage for the required format"
            )
        _validate_linkage_index_bounds(Z)

    n_leaves = Z.shape[0] + 1
    if n_leaves < 2:
        raise DegenerateTreeError("need at least 2 leaves to build a tree")

    resolved_labels = _validate_labels(labels, n_leaves)

    dendro = _scipy_dendrogram(
        Z,
        labels=resolved_labels,
        no_plot=True,
        color_threshold=color_threshold,
        link_color_func=link_color_func,
        truncate_mode=truncate_mode,
        p=p,
    )

    return DendrogramLayout(
        Z=Z,
        icoord=np.asarray(dendro["icoord"], dtype=float),
        dcoord=np.asarray(dendro["dcoord"], dtype=float),
        leaves=list(dendro["leaves"]),
        leaf_labels=list(dendro["ivl"]),
        color_list=list(dendro["color_list"]),
    )
