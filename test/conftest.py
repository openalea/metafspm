"""
Test-suite wide setup: make the shared helper modules of test/helpers importable (simple_seedling, growth, anatomy,
scene_doubles, doubles, doubles_ds, solver_specs, the plotting helpers) whatever the pytest import mode, and when the
suite runs from test/ without pyproject.toml (conda recipe).
"""
import os
import sys

_HERE = os.path.dirname(__file__)
for _folder in ("", "helpers"):
    _path = os.path.join(_HERE, _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)
