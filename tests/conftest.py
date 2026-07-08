import matplotlib

matplotlib.use("Agg")  # headless test environment

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def small_data(rng):
    return rng.normal(size=(10, 4))
