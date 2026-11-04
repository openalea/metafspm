"""
A checkpoint loaded in place into a DataStructure its models coupled keeps the models' links, those of their current
translator, not the checkpoint's: a run resumed with a changed formula uses it (recomputed from the loaded sources at
its next read), a new link's receiving variable is registered, an unchanged one is kept as saved; a changed alias is
refused. A DataStructure no model coupled (DataStructure.restore) takes the checkpoint's links.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.component import StructuralComponent, input_variable, state_variable
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene.population import population_data_structure
from openalea.metafspm.solve.decorator import rate

DOC = dict(unit="", unit_comment="", description="", min_value=None, max_value=None, value_comment="", references="",
           DOI=[])
AREAS = (1., 2., 3., 4.)


@dataclass
class Elements(StructuralComponent):
    """A stem of four elements, their areas."""
    element_area: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        s, L = g.scales, g.labels
        complex_ = plant
        for scale, label in ((s.Axis, L.Axis.Shoot), (s.GrowthUnit, L.GrowthUnit.Shoot), (s.Phytomer, L.Phytomer.Shoot),
                             (s.Organ, L.Organ.StemInternode)):
            complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=label))
        for area in AREAS:
            element = g.add_component(complex_, **PropsConfig(scale=s.SubOrgan, edge_type='/',
                                                              label=L.SubOrgan.StemElement))
            g.property("element_area")[element] = area


@dataclass
class Exchange(StructuralComponent):
    """Its flux, twice the area it exchanges over (a link from Elements' areas); an optional input too."""
    exchanging_area: float = input_variable(**DOC, by="Elements", default=0., scale=scales.SubOrgan,
                                            kind="extensive")
    emerged_area: float = input_variable(**DOC, by="Elements", default=np.nan, optional=True,
                                         scale=scales.SubOrgan, kind="extensive")
    flux: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")

    @rate
    def _flux(self, exchanging_area):
        return {"flux": 2. * np.asarray(exchanging_area)}


def half(element_area):                       # module-level formulas: a checkpoint keeps them by reference
    return 0.5 * element_area


def quarter(element_area):
    return 0.25 * element_area


def tenth(element_area):
    return 0.1 * element_area


def _model(translator):
    class Model(CompositeModel):
        initiators = (Elements,)

        def __init__(self, data_structure, time_step):
            self.elements = Elements(data_structure=data_structure, time_step=time_step)
            self.exchange = Exchange(data_structure=data_structure, time_step=time_step)
            self.couple(self.elements, self.exchange)

        def run(self):
            self.elements()
            self.exchange()

    Model.translator = translator
    return Model


def _population(translator):
    model = _model(translator)
    table = pd.DataFrame([dict(plant="p0", model=model, x=0., y=0., z=0., rotation=0., scenario={})])
    return population_data_structure(table, model)


def _link(formula=half, **sources):
    if sources:
        return Translator().link("Exchange", "exchanging_area", "Elements", sources)
    return Translator().link("Exchange", "exchanging_area", "Elements", ("element_area",), formula=formula)


def _checkpointed(tmp_path, translator):
    population = _population(translator)
    population.instance.run()
    path = str(tmp_path / "checkpoint")
    population.data_structure.checkpoint(path)
    return population, path


def test_a_resumed_run_uses_its_translator_s_changed_formula(tmp_path):
    saved, path = _checkpointed(tmp_path, _link(half))
    np.testing.assert_allclose(saved.data_structure.get("flux"), np.asarray(AREAS))      # 2 × half the area
    resumed = _population(_link(quarter))
    resumed.data_structure.load_checkpoint(path)
    ds = resumed.data_structure
    assert ds.derived()["exchanging_area"]["formula"] is quarter
    np.testing.assert_allclose(ds.get("flux"), np.asarray(AREAS))             # the saved state, as it was
    np.testing.assert_allclose(ds.get("exchanging_area"), 0.25 * np.asarray(AREAS))      # recomputed when read
    resumed.instance.run()
    np.testing.assert_allclose(ds.get("flux"), 0.5 * np.asarray(AREAS))


def test_an_unchanged_link_is_kept_as_saved(tmp_path):
    saved, path = _checkpointed(tmp_path, _link(half))
    resumed = _population(_link(half))
    resumed.data_structure.load_checkpoint(path)
    ds = resumed.data_structure
    assert not ds.is_stale("exchanging_area")                 # its saved values up to date with the loaded sources
    np.testing.assert_array_equal(ds.get("exchanging_area"), saved.data_structure.get("exchanging_area"))
    resumed.instance.run()
    saved.instance.run()
    np.testing.assert_array_equal(ds.get("flux"), saved.data_structure.get("flux"))


def test_a_new_link_and_a_dropped_one(tmp_path):
    """A link the checkpoint has not (a receiving variable it does not hold) is registered and computed; one the
    translator no longer has leaves its variable as saved, no longer derived."""
    _, path = _checkpointed(tmp_path, _link(half))
    translator = Translator().link("Exchange", "emerged_area", "Elements", ("element_area",), formula=tenth)
    resumed = _population(translator)
    resumed.data_structure.load_checkpoint(path)
    ds = resumed.data_structure
    assert set(ds.derived()) == {"emerged_area"}
    np.testing.assert_allclose(ds.get("emerged_area"), 0.1 * np.asarray(AREAS))
    np.testing.assert_allclose(ds.get("exchanging_area"), 0.5 * np.asarray(AREAS))        # as saved
    ds.validate_variables(strict=True)


def test_a_changed_alias_is_refused(tmp_path):
    _, path = _checkpointed(tmp_path, _link(element_area=1.))                 # an alias of Elements' areas
    assert _population(_link(element_area=1.)).data_structure.aliases() == {"exchanging_area": "element_area"}
    resumed = _population(_link(quarter))
    with pytest.raises(ValueError, match="aliases differ"):
        resumed.data_structure.load_checkpoint(path)


def test_a_data_structure_no_model_coupled_takes_the_checkpoint_s_links(tmp_path):
    _, path = _checkpointed(tmp_path, _link(half))
    restored = MPGDataStructure.restore(path)
    assert restored.derived()["exchanging_area"]["formula"] is half
