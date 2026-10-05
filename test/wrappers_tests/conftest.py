import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher

import doubles


@pytest.fixture(autouse=True)
def reset_choregrapher():
    """Wrapper tests mutate the Choregrapher singleton: give each test a clean run state."""
    Choregrapher().reset()
    yield
    Choregrapher().reset()


@pytest.fixture
def translator_path(tmp_path):
    return doubles.write_translator(tmp_path / "coupling_translator.yaml")
