import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from dendrofan.clustering import build_layout
from dendrofan.exceptions import (
    DegenerateTreeError,
    InvalidLinkageError,
    LabelMismatchError,
)


def test_build_layout_from_raw_data(small_data):
    layout = build_layout(small_data, method="ward")
    assert layout.n_leaves == small_data.shape[0]
    assert layout.icoord.shape[1] == 4
    assert layout.dcoord.shape[1] == 4
    assert len(layout.leaf_labels) == small_data.shape[0]


def test_build_layout_from_condensed_distances(small_data):
    condensed = pdist(small_data)
    layout = build_layout(condensed_distances=condensed, method="average")
    assert layout.n_leaves == small_data.shape[0]


def test_build_layout_from_precomputed_linkage(small_data):
    Z = linkage(small_data, method="ward")
    layout = build_layout(Z=Z)
    assert layout.n_leaves == small_data.shape[0]


def test_build_layout_requires_exactly_one_input(small_data):
    Z = linkage(small_data, method="ward")
    with pytest.raises(ValueError):
        build_layout(data=small_data, Z=Z)
    with pytest.raises(ValueError):
        build_layout()


def test_build_layout_rejects_invalid_linkage_matrix():
    # n=2 observations (1 merge row) but the row references cluster index 5,
    # which does not exist yet -- structurally invalid per SciPy's own check.
    bad_Z = np.array([[0.0, 5.0, 0.5, 2.0]])
    with pytest.raises(InvalidLinkageError):
        build_layout(Z=bad_Z)


def test_build_layout_rejects_out_of_range_index_even_if_is_valid_linkage_is_not(monkeypatch):
    # SciPy's own is_valid_linkage does not reliably catch this on every
    # version (observed: catches it under SciPy 1.13, misses it under
    # SciPy 1.17). Force the "it was missed" branch to make sure
    # dendrofan's own bounds check still catches it regardless of which
    # SciPy version is installed.
    import dendrofan.clustering as clustering_module

    monkeypatch.setattr(clustering_module, "is_valid_linkage", lambda Z: True)
    bad_Z = np.array([[0.0, 5.0, 0.5, 2.0]])
    with pytest.raises(InvalidLinkageError):
        build_layout(Z=bad_Z)


def test_build_layout_rejects_invalid_condensed_vector():
    with pytest.raises(InvalidLinkageError):
        # Length 4 is not n*(n-1)/2 for any integer n (3 -> n=3, 6 -> n=4, ...).
        build_layout(condensed_distances=np.array([1.0, 2.0, 3.0, 4.0]))


def test_build_layout_rejects_mismatched_labels(small_data):
    with pytest.raises(LabelMismatchError):
        build_layout(small_data, labels=["only", "two"])


def test_build_layout_rejects_single_observation():
    with pytest.raises(DegenerateTreeError):
        build_layout(np.zeros((1, 3)))


def test_build_layout_accepts_two_observations():
    layout = build_layout(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert layout.n_leaves == 2
    assert layout.y_max > 0


def test_build_layout_labels_default_to_indices(small_data):
    layout = build_layout(small_data)
    assert set(layout.leaf_labels) == {str(i) for i in range(small_data.shape[0])}


def test_build_layout_color_threshold_populates_color_list(small_data):
    layout = build_layout(small_data, color_threshold=0.0)
    assert len(layout.color_list) == layout.icoord.shape[0]
