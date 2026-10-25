"""The plain forms of devplan_porting §16: named step outputs (a dict), declarations documented by keywords,
parameter_value, cached, run_stage, ds.structure, planting; and display.draw."""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm import FunctionalComponent, Planting, planting
from openalea.metafspm.coupling.component import input_variable, parameter, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from simple_seedling import generate_simple_mpg_seedling
from openalea.metafspm.solve.decorator import postsegmentation, rate


@dataclass
class Named(FunctionalComponent):
    length: float = input_variable(unit="m", description="A length", by="Test", initialize=2., location="node")
    double: float = state_variable(unit="m", description="Twice the length", initialize=0., location="node",
                                   state_variable_type="extensive")
    square: float = state_variable(unit="m2", description="The length squared", initialize=0., location="node",
                                   state_variable_type="extensive")
    later: float = state_variable(unit="m", description="Written at postsegmentation", initialize=0.,
                                  location="node", state_variable_type="extensive")
    factor: float = parameter(unit="adim", description="A factor", by="Test", default=3.)

    @rate
    def _measures(self, length) -> dict:
        return {"double": 2. * length, "square": length ** 2}

    @postsegmentation
    def _later(self, length, factor):
        return factor * length


@dataclass
class Scalar(FunctionalComponent):
    length: float = input_variable(unit="m", description="A length", by="Test", initialize=2., location="node")
    double: float = state_variable(unit="m", description="Twice", initialize=0., location="node",
                                   state_variable_type="extensive")

    @rate(vectorized=False)
    def _per_element(self, length) -> dict:
        return {"double": 2. * length}


@dataclass
class Wrong(FunctionalComponent):
    length: float = input_variable(unit="m", description="A length", by="Test", initialize=2., location="node")

    @rate
    def _wrong(self, length) -> dict:
        return length


def graph():
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return MPGDataStructure(g, from_scale=g.scales.SubOrgan)


def test_named_outputs():
    ds = graph()
    component = Named(data_structure=ds, time_step=1.)
    component()
    np.testing.assert_array_equal(ds.get("double"), 4.)
    np.testing.assert_array_equal(ds.get("square"), 4.)
    assert not ds.has("measures")                      # the method's name only names the step
    scalar_ds = graph()
    Scalar(data_structure=scalar_ds, time_step=1.)()
    np.testing.assert_array_equal(scalar_ds.get("double"), 4.)
    with pytest.raises(ValueError, match="annotated -> dict"):
        Wrong(data_structure=graph(), time_step=1.)()


def test_documentation_by_keywords():
    metadata = Named.__dataclass_fields__["length"].metadata
    assert metadata["unit"] == "m" and metadata["description"] == "A length" and metadata["unit_comment"] == ""


def test_parameter_value_cached_and_run_stage():
    ds = graph()
    component = Named(data_structure=ds, time_step=1.)
    assert component.parameter_value("factor") == 3.
    builds = []
    assert component.cached("thing", lambda: builds.append(1) or 7) == 7
    assert component.cached("thing", lambda: builds.append(1) or 8) == 7 and len(builds) == 1
    assert component.cached("thing", lambda: 9, key="another") == 9
    assert ds.structure is ds._mtg                                  # the MTG, unsynchronised
    component.run_stage("postsegmentation")
    np.testing.assert_array_equal(ds.get("later"), 6.)
    np.testing.assert_array_equal(ds.get("double"), 0.)            # the rate step did not run
    with pytest.raises(ValueError, match="no schedule row"):
        component.run_stage("nothing")


def test_planting():
    table = planting([Planting(0., 0., -0.02, 10.), Planting(0.3, 0.1)], model=object, parameters=dict(a=1),
                     plant_parameters=[dict(b=1), dict(b=2)])
    assert table["z"].tolist() == [-0.02, 0.] and table["rotation"].tolist() == [10., 0.]
    assert [row["parameters"] for row in table["scenario"]] == [dict(a=1, b=1), dict(a=1, b=2)]
    assert table.attrs["xrange"] == pytest.approx(0.3) and table.attrs["yrange"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        planting([Planting()], model=object, plant_parameters=[{}, {}])


def test_draw():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from openalea.metafspm.coupling.geometry import Triangles
    from openalea.metafspm.display import draw, value_colors
    triangles = Triangles(np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 1.]]] * 2), np.array([0, 1]))
    figure, axes = plt.subplots(subplot_kw=dict(projection="3d"))
    draw(axes, triangles, colors=value_colors(np.array([0., 1.])))
    assert len(axes.collections) == 1
    plt.close(figure)
