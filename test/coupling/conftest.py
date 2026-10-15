import pytest

import doubles


@pytest.fixture
def translator_path(tmp_path):
    return doubles.write_translator(tmp_path / "coupling_translator.yaml")
