"""Color and label-placement helpers for circular dendrograms."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

ColorMap = Union[Dict[str, str], Callable[[str], str]]


def resolve_leaf_colors(
    leaf_labels: Sequence[str],
    colors: Optional[ColorMap] = None,
    default: str = "black",
) -> List[str]:
    """Resolve a per-leaf color for each label in ``leaf_labels``.

    Parameters
    ----------
    leaf_labels : sequence of str
        Leaf labels, in left-to-right (or any) order -- one color is
        produced per entry, in the same order.
    colors : dict or callable, optional
        - dict mapping label -> color. Labels missing from the dict
          fall back to ``default`` rather than raising, since dendrograms
          routinely mix "classified" and "unclassified" leaves (e.g. the
          excluded stations in the source manuscript).
        - callable ``label -> color``.
        - ``None``: every leaf gets ``default``.
    default : str, default "black"
        Fallback color for labels not covered by ``colors``.

    Returns
    -------
    list of str
        One color per entry of ``leaf_labels``.
    """
    if colors is None:
        return [default] * len(leaf_labels)
    if callable(colors):
        return [colors(label) for label in leaf_labels]
    return [colors.get(label, default) for label in leaf_labels]


def legend_handles(
    color_map: Dict[str, str], marker: str = "s", markersize: float = 10
) -> Tuple[list, list]:
    """Build ``(handles, labels)`` for ``ax.legend()`` from a ``{group: color}`` map.

    Groups are emitted in the order given by ``color_map`` (a plain
    ``dict`` preserves insertion order in Python 3.7+), so callers
    control legend ordering by controlling dict construction order.
    """
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=markersize,
        )
        for color in color_map.values()
    ]
    labels = list(color_map.keys())
    return handles, labels
