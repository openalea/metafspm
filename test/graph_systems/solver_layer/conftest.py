"""Pytest fixtures for the tests of the internal solver layer (graph systems are tested through components elsewhere); the spec builders live in solver_specs.py."""
import pytest

from solver_specs import build_spec_nonlinear_1node, build_spec_decay_1node, build_spec_linear_2node


# ── Pytest fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def spec_nonlinear_1node():
    return build_spec_nonlinear_1node()


@pytest.fixture
def spec_decay_1node():
    return build_spec_decay_1node()


@pytest.fixture
def spec_linear_2node():
    return build_spec_linear_2node()
