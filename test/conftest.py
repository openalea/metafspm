"""
Test-suite wide setup: make the shared test helper modules importable (solver_specs, doubles, doubles_ds,
simple_seedling, the UC1 components, the growth helper) whatever the pytest import mode, and when the suite runs from test/
without pyproject.toml (conda recipe).
"""
import os
import sys

_HERE = os.path.dirname(__file__)
for _folder in ("", "mpg_tests", "solver_tests", "wrappers_tests", "graph_system_tests", "structure_tests"):
    _path = os.path.join(_HERE, _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher


@pytest.fixture(autouse=True)
def _fresh_choregrapher():
    """Each test starts with the Choregrapher's run state cleared (bound schedules, time steps, DataStructures)."""
    Choregrapher().reset()
    yield
    Choregrapher().reset()
