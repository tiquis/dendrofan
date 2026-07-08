import numpy as np
import pytest

from dendrofan import circular_dendrogram
from dendrofan.annotations import add_scale_ring, highlight_sector
from dendrofan.exceptions import DegenerateTreeError


def test_circular_dendrogram_basic(small_data):
    labels = [f"s{i}" for i in range(small_data.shape[0])]
    result = circular_dendrogram(small_data, labels=labels)
    assert result.layout.n_leaves == small_data.shape[0]
    assert set(result.leaf_theta.keys()) == set(labels)
    assert len(result.leaf_theta_by_index) == small_data.shape[0]
    # All angles should be finite, distinct-ish values inside one turn.
    thetas = np.array(result.leaf_theta_by_index)
    assert np.all(np.isfinite(thetas))


def test_circular_dendrogram_two_leaves():
    data = np.array([[0.0, 0.0], [5.0, 5.0]])
    result = circular_dendrogram(data, labels=["a", "b"])
    assert result.layout.n_leaves == 2


def test_circular_dendrogram_single_observation_raises():
    with pytest.raises(DegenerateTreeError):
        circular_dendrogram(np.zeros((1, 3)))


def test_circular_dendrogram_zero_variance_data_does_not_raise():
    # Every pairwise distance is 0 -> merge heights are all 0. The
    # original ad hoc polar transform divided by dcoord.max() here and
    # would raise ZeroDivisionError / emit NaNs.
    data = np.zeros((5, 3))
    result = circular_dendrogram(data, labels=list("abcde"))
    assert result.layout.n_leaves == 5


def test_circular_dendrogram_partial_span_gap():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(8, 3))
    result = circular_dendrogram(data, span=270)
    assert result.transform.span == 270


def test_circular_dendrogram_label_colors_dict(small_data):
    labels = [f"s{i}" for i in range(small_data.shape[0])]
    colors = {label: "red" for label in labels[:3]}
    result = circular_dendrogram(small_data, labels=labels, label_colors=colors)
    assert result.ax is not None  # smoke test: no exception with partial color map


def test_circular_dendrogram_reuses_given_axes(small_data):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result = circular_dendrogram(small_data, ax=ax)
    assert result.ax is ax
    assert result.fig is fig


def test_add_scale_ring_and_highlight_sector_smoke(small_data):
    result = circular_dendrogram(small_data)
    add_scale_ring(result.ax, result.transform, heights=[0, result.transform.y_max / 2])
    highlight_sector(result.ax, result.transform, 0.0, 1.0)  # smoke test, no assertion needed
