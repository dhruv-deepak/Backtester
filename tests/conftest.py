import pytest

from backtester.data import generate_synthetic


@pytest.fixture(scope="session")
def random_walk():
    return generate_synthetic(days=900, trend=0.0, seed=1)


@pytest.fixture(scope="session")
def trending():
    return generate_synthetic(days=2500, trend=0.15, seed=7)
