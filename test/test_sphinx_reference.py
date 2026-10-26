"""A component's reference text from its declarations (openalea.metafspm.sphinx), valid reStructuredText."""
import sys

import pytest

from openalea.metafspm.sphinx import component_rst


def test_component_reference():
    sys.path.insert(0, "components")
    from test_settings import Lamp
    text = component_rst(Lamp)
    assert "State variables" in text and "Settings" in text and "``latitude``" in text and "Schedule" in text
    docutils = pytest.importorskip("docutils.core")
    docutils.publish_doctree("Lamp\n====\n\n" + text, settings_overrides={"report_level": 2, "halt_level": 2})


def test_a_model_reference():
    from openalea.metafspm.scene.environment import EnvironmentModel

    class Light(EnvironmentModel):
        """A light model."""
        interval = 3600.

        def between(self):
            """Only PARa."""
    text = component_rst(Light)
    assert "A light model." in text and "every 3600 s" in text and "``between``: Only PARa." in text


def test_a_component_without_declarations():
    from dataclasses import dataclass
    from openalea.metafspm.coupling.component import FunctionalComponent

    @dataclass
    class Bare(FunctionalComponent):
        """Nothing declared."""
    assert component_rst(Bare).startswith("Nothing declared.")
