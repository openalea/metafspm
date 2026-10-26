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
