import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pytest_configure(config):
    config.addinivalue_line("markers", "model: needs the frozen GP-MoLFormer checkpoint")


@pytest.fixture(scope="session")
def generator():
    """The real frozen checkpoint, loaded once for the whole test session."""
    from property_to_go.config import load_config
    from property_to_go.model_io import load_generator

    return load_generator(load_config("model"))
