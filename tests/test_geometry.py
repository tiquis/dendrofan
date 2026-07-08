import math

import numpy as np
import pytest

from dendrofan.geometry import PolarTransform, radial_label_alignment


def test_theta_maps_endpoints_full_circle():
    t = PolarTransform(x_min=0, x_max=100, y_max=1, start_angle=90, span=360, clockwise=False)
    assert math.isclose(t.theta(0), math.radians(90))
    assert math.isclose(t.theta(100), math.radians(90 + 360))


def test_theta_clockwise_direction():
    t = PolarTransform(x_min=0, x_max=100, y_max=1, start_angle=0, span=180, clockwise=True)
    assert math.isclose(t.theta(100), math.radians(-180))


def test_theta_partial_span_leaves_gap():
    t = PolarTransform(x_min=0, x_max=100, y_max=1, start_angle=90, span=350, clockwise=False)
    # A full circle would put the last leaf at the same angle as the
    # first; a 350-degree span should stop 10 degrees short of that.
    assert not math.isclose(t.theta(100) % (2 * math.pi), t.theta(0) % (2 * math.pi))


def test_theta_single_leaf_does_not_divide_by_zero():
    t = PolarTransform(x_min=5, x_max=5, y_max=1, start_angle=45)
    assert math.isclose(float(t.theta(5)), math.radians(45))


def test_radius_root_at_centre_leaves_at_rim():
    t = PolarTransform(x_min=0, x_max=1, y_max=10, inner_radius=0.1, outer_radius=1.0)
    assert math.isclose(float(t.radius(0)), 1.0)  # leaves: y=0 -> outer rim
    assert math.isclose(float(t.radius(10)), 0.1)  # root: y=y_max -> inner radius


def test_radius_degenerate_zero_height_tree_does_not_raise():
    t = PolarTransform(x_min=0, x_max=1, y_max=0, inner_radius=0.1, outer_radius=1.0)
    r = t.radius(np.array([0.0, 0.0]))
    assert np.all(np.isfinite(r))
    assert np.allclose(r, 1.0)  # falls back to leaf radius everywhere


def test_radius_sqrt_scale_matches_linear_at_endpoints():
    linear = PolarTransform(x_min=0, x_max=1, y_max=1, radius_scale="linear")
    sqrt_ = PolarTransform(x_min=0, x_max=1, y_max=1, radius_scale="sqrt")
    for y in (0.0, 1.0):
        assert math.isclose(float(linear.radius(y)), float(sqrt_.radius(y)), abs_tol=1e-9)


def test_radius_sqrt_scale_differs_at_midpoint():
    linear = PolarTransform(x_min=0, x_max=1, y_max=1, radius_scale="linear")
    sqrt_ = PolarTransform(x_min=0, x_max=1, y_max=1, radius_scale="sqrt")
    assert not math.isclose(float(linear.radius(0.5)), float(sqrt_.radius(0.5)))


def test_radius_custom_callable_scale():
    t = PolarTransform(x_min=0, x_max=1, y_max=1, radius_scale=lambda y: y ** 2)
    assert math.isclose(float(t.radius(0.5)), 1.0 - 0.25)  # invert_radius=True by default


def test_invalid_span_raises():
    with pytest.raises(ValueError):
        PolarTransform(x_min=0, x_max=1, y_max=1, span=0)
    with pytest.raises(ValueError):
        PolarTransform(x_min=0, x_max=1, y_max=1, span=361)


def test_invalid_radii_raise():
    with pytest.raises(ValueError):
        PolarTransform(x_min=0, x_max=1, y_max=1, inner_radius=1.0, outer_radius=1.0)
    with pytest.raises(ValueError):
        PolarTransform(x_min=0, x_max=1, y_max=1, inner_radius=-0.1)


def test_radial_label_alignment_right_half_upright():
    rotation, ha = radial_label_alignment(math.radians(30))
    assert rotation == pytest.approx(30)
    assert ha == "left"


def test_radial_label_alignment_left_half_flipped():
    rotation, ha = radial_label_alignment(math.radians(150))
    assert rotation == pytest.approx(150 - 180)
    assert ha == "right"


def test_radial_label_alignment_boundary_at_90_degrees():
    rotation, ha = radial_label_alignment(math.radians(90))
    assert ha == "left"
    rotation, ha = radial_label_alignment(math.radians(90.0001))
    assert ha == "right"
