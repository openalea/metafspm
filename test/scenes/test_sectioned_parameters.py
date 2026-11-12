"""
Sectioned parameters: a parameter whose value is {section: {field: value}} (e.g. an anatomy's parametrization),
its fields listed one by one ("name.section.field") by declarations, parameter_table and variants (their defaults and
bounds), and a scenario's dotted entries folded into it.
"""
from dataclasses import dataclass

import pytest

from openalea.metafspm.coupling.component import StructuralComponent, parameter
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.scene.population import _flat_scenario
from openalea.metafspm.scene.scenarios import parameter_table, variants


def _fields():
    return {"cortex.n_layers": dict(default=3, unit="", description="the cortex' layers", min_value=1, max_value=8),
            "stele.radius": dict(default=0.2, unit="mm", description="the stele's radius", min_value=0.05,
                                 max_value=1.)}


@dataclass
class Anatomy(StructuralComponent):
    anatomy: object = parameter(unit="", description="the anatomy's sections", default=None, dtype="object",
                                scale=scales.Plant, sections=_fields)


def test_the_fields_are_declared_one_by_one():
    table = parameter_table(Anatomy).set_index("name")
    assert {"anatomy", "anatomy.cortex.n_layers", "anatomy.stele.radius"} <= set(table.index)
    assert table.at["anatomy.cortex.n_layers", "value"] == 3 and table.at["anatomy.stele.radius", "max_value"] == 1.


def test_variants_range_a_field_within_its_bounds():
    found = variants({}, {"anatomy.cortex.n_layers": None}, design="grid", levels=3, declared=[Anatomy])
    assert [s["anatomy.cortex.n_layers"] for s in found.values()] == [1, 4, 8]       # an integer stays one
    with pytest.raises(ValueError, match="outside its declared bounds"):
        variants({}, {"anatomy.stele.radius": [2.]}, declared=[Anatomy])


def test_a_scenarios_dotted_entries_fold_into_the_parameter():
    given = {"anatomy": {"preset": "for_root", "cortex": {"n_layers": 2}}, "anatomy.cortex.n_layers": 5,
             "anatomy.stele.radius": 0.3, "other": 1.}
    folded = _flat_scenario(given)
    assert folded == {"anatomy": {"preset": "for_root", "cortex": {"n_layers": 5}, "stele": {"radius": 0.3}},
                      "other": 1.}
    assert given["anatomy"]["cortex"]["n_layers"] == 2                               # the given one unchanged
