"""
Live reading of DataStructure variables by components and the solver: the solver reads the DataStructure at each
solve and writes the results in place, steps are evaluated on its arrays (vectorised, with a per-element opt-in),
previous() is managed by the framework, and the component follows topology growth.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.solve.decorator import rate

from nitrogen import DT, K, Transport, TransportWithAmount, TransportWithOutput, incidence, nitrogen_ds
from simple_seedling import generate_simple_mpg_seedling


@pytest.fixture(autouse=True)
def _simulation_time_step():
    Choregrapher().add_simulation_time_step(DT)


def _model(component_class=Transport, seed=42):
    ds = nitrogen_ds()
    model = component_class(data_structure=ds)
    rng = np.random.default_rng(seed)
    c_old = 0.10 + 0.40 * rng.random(ds.n_nodes())
    J = 0.01 * rng.random(ds.n_nodes())
    ds.set("concentration", c_old)
    ds.set("radial_solute_input", J)
    ds.set("K_axial", np.full(ds.n_edges(), K))
    return ds, model, c_old, J


def _balance_residual(ds, c_old, J):
    B = incidence(ds)
    return (np.asarray(ds.get("concentration")) - c_old) / DT + B @ np.asarray(ds.get("axial_flux")) - J


# ---------------------------------------------------------------- solver path

def test_the_solution_is_written_in_place():
    ds, model, c_old, J = _model()
    concentration = ds.get("concentration")
    model()
    assert ds.get("concentration") is concentration
    np.testing.assert_allclose(_balance_residual(ds, c_old, J), 0., atol=1e-10)


def test_parameter_changes_are_read_at_the_next_solve():
    ds, model, _, _ = _model()
    ds.set("K_axial", np.zeros(ds.n_edges()))
    model()
    np.testing.assert_allclose(ds.get("axial_flux"), 0., atol=1e-15)


@pytest.mark.parametrize("component_class, name, location",
                         [(TransportWithAmount, "axial_flux_amount", "edge"),
                          (TransportWithOutput, "axial_divergence", "node")])
def test_integrated_amounts_and_outputs_are_registered(component_class, name, location):
    ds, model, _, _ = _model(component_class)
    model()
    assert ds.location(name) == location


# ---------------------------------------------------------------- Choregrapher steps

@dataclass
class VectorisedProbe(FunctionalComponent):
    level: float = state_variable(unit="", unit_comment="", description="", min_value="", max_value="",
                                  value_comment="", references="", DOI="", initialize=1., scale=scales.SubOrgan)
    doubled: float = state_variable(unit="", unit_comment="", description="", min_value="", max_value="",
                                    value_comment="", references="", DOI="", initialize=0., scale=scales.SubOrgan)
    clipped: float = state_variable(unit="", unit_comment="", description="", min_value="", max_value="",
                                    value_comment="", references="", DOI="", initialize=0., scale=scales.SubOrgan)
    threshold: float = parameter(unit="", unit_comment="", description="", min_value="", max_value="",
                                 value_comment="", references="", DOI="", by="", default=2.)

    @rate
    def _doubled(self, level):
        self.vectorised_calls = getattr(self, "vectorised_calls", 0) + 1
        return 2. * level

    @rate(vectorized=False)
    def _clipped(self, level, threshold):
        self.scalar_calls = getattr(self, "scalar_calls", 0) + 1
        if level > threshold:               # scalar logic: needs the per-element opt-in
            return threshold
        return level


def test_steps_are_vectorised_with_a_per_element_opt_in():
    ds = nitrogen_ds()
    ds.register("level", np.linspace(0., 4., ds.n_nodes()), location="node")
    model = VectorisedProbe(data_structure=ds)

    model()

    np.testing.assert_allclose(ds.get("doubled"), 2. * np.linspace(0., 4., ds.n_nodes()))
    np.testing.assert_allclose(ds.get("clipped"), np.minimum(np.linspace(0., 4., ds.n_nodes()), 2.))
    assert model.vectorised_calls == 1
    assert model.scalar_calls == ds.n_nodes()


# ---------------------------------------------------------------- previous state

def test_previous_state_is_managed_by_the_framework():
    """previous() is the state at the start of the last solve: consecutive calls advance it by themselves."""
    ds, model, c_old, J = _model(seed=7)
    model()
    np.testing.assert_array_equal(model.previous("concentration"), c_old)
    first = np.array(ds.get("concentration"))
    model()
    np.testing.assert_array_equal(model.previous("concentration"), first)
    np.testing.assert_allclose(_balance_residual(ds, first, J), 0., atol=1e-10)
    assert not np.allclose(ds.get("concentration"), first)


def test_previous_state_outside_a_solve_raises():
    model = Transport(data_structure=nitrogen_ds())
    with pytest.raises(KeyError, match="previous"):
        model.previous("concentration")


def test_the_component_follows_topology_growth():
    """After update_topology(), the graph view, the carried-over variables and the solve use the new topology."""
    g, seedling = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    model = Transport(data_structure=ds)
    ds.set("radial_solute_input", np.full(ds.n_nodes(), 0.01))
    model()
    n_before = model._graph_view.n_nodes

    g.add_child(seedling.root_segment6, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<',
                                                     label=g.labels.SubOrgan.RootSegment))
    ds.update_topology()
    c_old, J = np.array(ds.get("concentration")), np.array(ds.get("radial_solute_input"))
    model()

    assert model._graph_view.n_nodes == n_before + 1 == ds.n_nodes()
    assert ds.get("concentration").shape == (ds.n_nodes(),)
    np.testing.assert_allclose(_balance_residual(ds, c_old, J), 0., atol=1e-10)
