"""
Test-suite wide setup: make the shared helper modules of test/helpers importable (simple_seedling, growth, anatomy,
scene_doubles, doubles, doubles_ds, solver_specs, the plotting helpers) whatever the pytest import mode, and when the
suite runs from test/ without pyproject.toml (conda recipe); reset the Choregrapher between tests.
"""
import os
import sys

_HERE = os.path.dirname(__file__)
for _folder in ("", "helpers"):
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
