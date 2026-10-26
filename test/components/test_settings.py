"""Settings: a model's values, one for all (not stored per plant): listed by declarations(), passed whole to the
steps naming them, readable as attributes, set by keywords and scenarios, checked against their choices."""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, setting, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.solve.decorator import rate

from growth import DOC, make_chain


@dataclass
class Lamp(FunctionalComponent):
    glow: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)
    seen: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)
    latitude: float = setting(unit="°", description="The site's latitude", default=48.85)
    model: str = setting(description="The light model", default="direct", choices=("direct", "nested"))
    sky: tuple = setting(description="The sky's sectors", default=(4, 5))

    @rate
    def _glow(self, glow, latitude, model, sky) -> dict:
        assert isinstance(latitude, float) and model in ("direct", "nested") and sky == (4, 5)
        return {"glow": np.full(glow.size, latitude) + (1. if self.model == "nested" else 0.)}

    @rate(vectorized=False)
    def _seen(self, seen, latitude, sky):
        return latitude * sky[0]


def lamp(**settings):
    Choregrapher().reset()
    _, ds, _ = make_chain()
    return ds, Lamp(data_structure=ds, **settings)


def test_settings_are_values_of_the_model():
    ds, component = lamp(latitude=45.)
    component()
    np.testing.assert_allclose(ds.get("glow"), 45.)
    np.testing.assert_allclose(ds.get("seen"), 180.)                 # a per-element step: the tuple kept whole
    assert not ds.has("latitude") and not ds.has("model")
    table = Lamp.declarations().set_index("name")
    assert table.loc["latitude", "kind"] == "setting" and table.loc["latitude", "default"] == 48.85
    assert Lamp.declared_settings() == ("latitude", "model", "sky")
    component.model = "nested"
    component()
    np.testing.assert_allclose(ds.get("glow"), 46.)


def test_a_setting_outside_its_choices_raises():
    with pytest.raises(ValueError, match="Lamp.model"):
        lamp(model="radiosity")
