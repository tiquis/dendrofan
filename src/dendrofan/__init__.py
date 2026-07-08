"""
dendrofan: circular ("fan") dendrograms for Python.

dendrofan draws dendrograms bent onto an annulus, in the visual style
of ``ape::plot.phylo(type = "fan")`` in R (Paradis & Schliep, 2019),
using only NumPy, SciPy, and Matplotlib -- no R dependency, no
reimplementation of clustering. It exists because
``scipy.cluster.hierarchy`` has no circular layout mode: the usual
workaround is a one-off script that manually re-projects
``dendrogram()``'s rectangular ``icoord``/``dcoord`` output into polar
coordinates. dendrofan generalizes that idea into a small, tested
library: a documented coordinate transform
(:class:`dendrofan.geometry.PolarTransform`) plus a plotting API that
handles the edge cases a one-off script usually skips (single-leaf and
two-leaf trees, all-zero merge heights, mismatched labels, partial fans
with an angular gap, inward vs. outward root, etc).

Quick start
-----------
>>> import numpy as np
>>> from dendrofan import circular_dendrogram
>>> rng = np.random.default_rng(0)
>>> data = rng.normal(size=(12, 5))
>>> result = circular_dendrogram(data, labels=[f"s{i}" for i in range(12)])
>>> result.fig.savefig("tree.png", dpi=200)

See :func:`circular_dendrogram` for the full parameter reference, and
the ``examples/`` directory in the source distribution for worked
examples, including one that reproduces a colored-by-group station
dendrogram end to end.
"""
from ._version import __version__
from .annotations import add_scale_ring, highlight_sector
from .clustering import DendrogramLayout, build_layout
from .exceptions import (
    DegenerateTreeError,
    DendrofanError,
    InvalidLinkageError,
    LabelMismatchError,
)
from .geometry import PolarTransform, arc_points, radial_label_alignment
from .plotting import CircularDendrogramResult, circular_dendrogram
from .styling import legend_handles, resolve_leaf_colors

__all__ = [
    "__version__",
    "circular_dendrogram",
    "CircularDendrogramResult",
    "build_layout",
    "DendrogramLayout",
    "PolarTransform",
    "arc_points",
    "radial_label_alignment",
    "add_scale_ring",
    "highlight_sector",
    "resolve_leaf_colors",
    "legend_handles",
    "DendrofanError",
    "InvalidLinkageError",
    "LabelMismatchError",
    "DegenerateTreeError",
]
