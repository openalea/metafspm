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


def _model(component_class=Transport, seed=42):
    data_structure = nitrogen_ds()
    model = component_class(data_structure=data_structure)
    rng = np.random.default_rng(seed)
    c_old = 0.10 + 0.40 * rng.random(data_structure.n_nodes())
    J = 0.01 * rng.random(data_structure.n_nodes())
    data_structure.set("concentration", c_old)
    data_structure.set("radial_solute_input", J)
    data_structure.set("K_axial", np.full(data_structure.n_edges(), K))
    return data_structure, model, c_old, J


def _balance_residual(data_structure, c_old, J):
    B = incidence(data_structure)
    return (np.asarray(data_structure.get("concentration")) - c_old) / DT + B @ np.asarray(data_structure.get("axial_flux")) - J


# ---------------------------------------------------------------- solver path

def test_the_solution_is_written_in_place():
    data_structure, model, c_old, J = _model()
    concentration = data_structure.get("concentration")
    model()
    assert data_structure.get("concentration") is concentration
    np.testing.assert_allclose(_balance_residual(data_structure, c_old, J), 0., atol=1e-10)


def test_parameter_changes_are_read_at_the_next_solve():
    data_structure, model, _, _ = _model()
    data_structure.set("K_axial", np.zeros(data_structure.n_edges()))
    model()
    np.testing.assert_allclose(data_structure.get("axial_flux"), 0., atol=1e-15)


@pytest.mark.parametrize("component_class, name, location",
                         [(TransportWithAmount, "axial_flux_amount", "edge"),
                          (TransportWithOutput, "axial_divergence", "node")])
def test_integrated_amounts_and_outputs_are_registered(component_class, name, location):
    data_structure, model, _, _ = _model(component_class)
    model()
    assert data_structure.location(name) == location


# ---------------------------------------------------------------- Choregrapher steps

@dataclass
class VectorisedProbe(FunctionalComponent):
    level: float = state_variable(unit="", unit_comment="", description="", min_value="", max_value="",
                                  value_comment="", references="", DOI="", default=1., scale=scales.SubOrgan)
    doubled: float = state_variable(unit="", unit_comment="", description="", min_value="", max_value="",
                                    value_comment="", references="", DOI="", default=0., scale=scales.SubOrgan)
    clipped: float = state_variable(unit="", unit_comment="", description="", min_value="", max_value="",
                                    value_comment="", references="", DOI="", default=0., scale=scales.SubOrgan)
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
    data_structure = nitrogen_ds()
    data_structure.register("level", np.linspace(0., 4., data_structure.n_nodes()), location="node")
    model = VectorisedProbe(data_structure=data_structure)

    model()

    np.testing.assert_allclose(data_structure.get("doubled"), 2. * np.linspace(0., 4., data_structure.n_nodes()))
    np.testing.assert_allclose(data_structure.get("clipped"), np.minimum(np.linspace(0., 4., data_structure.n_nodes()), 2.))
    assert model.vectorised_calls == 1
    assert model.scalar_calls == data_structure.n_nodes()


# ---------------------------------------------------------------- previous state

def test_previous_state_is_managed_by_the_framework():
    """previous() is the state at the start of the last solve: consecutive calls advance it by themselves."""
    data_structure, model, c_old, J = _model(seed=7)
    model()
    np.testing.assert_array_equal(model.previous("concentration"), c_old)
    first = np.array(data_structure.get("concentration"))
    model()
    np.testing.assert_array_equal(model.previous("concentration"), first)
    np.testing.assert_allclose(_balance_residual(data_structure, first, J), 0., atol=1e-10)
    assert not np.allclose(data_structure.get("concentration"), first)


def test_previous_state_outside_a_solve_raises():
    model = Transport(data_structure=nitrogen_ds())
    with pytest.raises(KeyError, match="previous"):
        model.previous("concentration")


def test_the_component_follows_topology_growth():
    """After update_topology(), the graph view, the carried-over variables and the solve use the new topology."""
    mtg, seedling = generate_simple_mpg_seedling()
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
    model = Transport(data_structure=data_structure)
    data_structure.set("radial_solute_input", np.full(data_structure.n_nodes(), 0.01))
    model()
    n_before = model._graph_view.n_nodes

    mtg.add_child(seedling.root_segment6, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<',
                                                     label=mtg.labels.SubOrgan.RootSegment))
    data_structure.update_topology()
    c_old, J = np.array(data_structure.get("concentration")), np.array(data_structure.get("radial_solute_input"))
    model()

    assert model._graph_view.n_nodes == n_before + 1 == data_structure.n_nodes()
    assert data_structure.get("concentration").shape == (data_structure.n_nodes(),)
    np.testing.assert_allclose(_balance_residual(data_structure, c_old, J), 0., atol=1e-10)
