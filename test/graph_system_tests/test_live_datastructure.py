"""
Live reading of DataStructure variables by components and the solver (devplan WD.2, design note §8).

Components no longer work on a props snapshot copied at construction: the solver snapshots the
DataStructure at each solve, results are written to it in place, and Choregrapher steps are
evaluated on its arrays (vectorised, with a per-element opt-in).
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.solve.decorator import rate

from test_uc1_nitrogen_transport import NitrogenAxialTransport, _make_ds, _setup_nitrogen_model


@pytest.fixture(autouse=True)
def _fresh_choregrapher_run_state():
    Choregrapher().reset()
    yield
    Choregrapher().reset()


def _model(rng_seed=42, K=0.07, dt=0.5):
    ds = _make_ds()
    rng = np.random.default_rng(rng_seed)
    c_old = 0.10 + 0.40 * rng.random(ds.n_nodes())
    model = _setup_nitrogen_model(ds, c_old=c_old, J_radial=0.01 * rng.random(ds.n_nodes()),
                                  K_axial_vals=np.full(ds.n_edges(), K), dt=dt)
    return ds, model, c_old


# ---------------------------------------------------------------- solver path

def test_solution_is_written_to_the_data_structure():
    ds, model, _ = _model()
    concentration = ds.get("concentration")

    model._invoke_graph_system("_transport_solve")

    node_u, edge_u = model._last_graph_system.unpack_unknowns(model._last_graph_solution)
    np.testing.assert_allclose(ds.get("concentration"), node_u["concentration"], atol=1e-15)
    np.testing.assert_allclose(ds.get("axial_flux"), edge_u["axial_flux"], atol=1e-15)
    assert ds.get("concentration") is concentration          # written in place


def test_parameter_changes_are_read_live():
    """A parameter changed on the DataStructure after construction is used by the next solve."""
    ds, model, _ = _model(K=0.07)
    ds.set("K_axial", 0.)

    model._invoke_graph_system("_transport_solve")

    _, edge_u = model._last_graph_system.unpack_unknowns(model._last_graph_solution)
    np.testing.assert_allclose(edge_u["axial_flux"], 0., atol=1e-15)


def test_integrated_amount_and_outputs_are_registered():
    ds, model, _ = _model()
    model._invoke_graph_system("_transport_solve_with_amount")
    assert ds.location("axial_flux_amount") == "edge"
    model._invoke_graph_system("_transport_solve_with_output")
    assert ds.location("axial_divergence") == "node"


def test_props_is_a_read_only_view_of_the_data_structure():
    ds, model, _ = _model()
    model._invoke_graph_system("_transport_solve")
    vid = ds._idx_to_vid[0]
    assert model.props["concentration"][vid] == ds.get("concentration")[0]
    assert "axial_flux" in model.props and len(model.props["axial_flux"]) == ds.n_edges()
    with pytest.raises(TypeError):
        model.props["concentration"][vid] = 1.


# ---------------------------------------------------------------- Choregrapher steps

def test_rate_output_lands_in_the_data_structure():
    ds = _make_ds()
    ds.set_node_property("concentration", np.full(ds.n_nodes(), 0.3))
    model = NitrogenAxialTransport(data_structure=ds)
    model.k_radial, model.c_ext = 0.2, 1.0
    model._previous_fields = {"concentration": np.full(ds.n_nodes(), 0.3)}
    model.time_step = 0.5

    model()

    np.testing.assert_allclose(ds.get("radial_solute_input"), 0.2 * (1.0 - 0.3), atol=1e-15)


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
    def _clipped(self, level):
        self.scalar_calls = getattr(self, "scalar_calls", 0) + 1
        if level > self.threshold:          # scalar logic: needs the per-element opt-in
            return self.threshold
        return level


def test_steps_are_vectorised_with_a_per_element_opt_in():
    ds = _make_ds()
    ds.set_node_property("level", np.linspace(0., 4., ds.n_nodes()))
    Choregrapher().add_simulation_time_step(1)
    model = VectorisedProbe(data_structure=ds)

    model()

    np.testing.assert_allclose(ds.get("doubled"), 2. * np.linspace(0., 4., ds.n_nodes()))
    np.testing.assert_allclose(ds.get("clipped"), np.minimum(np.linspace(0., 4., ds.n_nodes()), 2.))
    assert model.vectorised_calls == 1
    assert model.scalar_calls == ds.n_nodes()


# ---------------------------------------------------------------- previous state (Q21)

from openalea.metafspm.solve.decorator import edge_law, graph_system, node_balance
from openalea.metafspm.solve.solver import NewtonSolver


@dataclass
class SelfSteppingTransport(NitrogenAxialTransport):
    """UC1 transport whose equations read the framework-managed previous state instead of _previous_fields."""

    @graph_system(node_unknowns=["concentration"], edge_unknowns=["axial_flux"], solver=NewtonSolver,
                  max_iter=15, schedule_as="state")
    class _self_stepping_solve:
        @node_balance(field="concentration")
        def _concentration_balance(self, concentration, axial_flux, radial_solute_input):
            B = self._graph_view.incidence
            return ((concentration - self.previous("concentration")) / self.time_step
                    + np.asarray(B @ axial_flux).reshape(-1) - radial_solute_input)

        @edge_law(field="axial_flux", explicit=False, integrate=False)
        def _axial_flux_law(self, concentration, axial_flux, K_axial):
            B = self._graph_view.incidence
            return axial_flux - K_axial * np.asarray(B.T @ concentration).reshape(-1)


def test_previous_state_is_managed_by_the_framework():
    """Two solves in a row advance c_old automatically, as the manual _previous_fields bookkeeping does."""
    ds, reference, c_old = _model(rng_seed=7)
    ds_self, _, _ = _model(rng_seed=7)
    model = SelfSteppingTransport(data_structure=ds_self)
    model.time_step = reference.time_step

    c_prev = c_old.copy()
    for _ in range(2):
        reference._previous_fields = {"concentration": c_prev.copy()}
        reference._invoke_graph_system("_transport_solve")
        c_prev = ds.get("concentration").copy()
        model._invoke_graph_system("_self_stepping_solve")

    np.testing.assert_allclose(ds_self.get("concentration"), ds.get("concentration"), atol=1e-12)
    assert not np.allclose(ds.get("concentration"), c_old)


def test_component_follows_topology_growth():
    """After ds.update_topology(), the component's graph view, the carried-over variables and the solve use the new topology."""
    from openalea.metafspm.data_structure.configs import PropsConfig
    g, seedling = __import__("simple_seedling").generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    from openalea.metafspm.data_structure.data_api import MPGDataStructure
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    model = SelfSteppingTransport(data_structure=ds)
    model.time_step = 0.5
    ds.set("radial_solute_input", 0.01)
    model._invoke_graph_system("_self_stepping_solve")
    n_before = model._graph_view.n_nodes

    g.add_child(seedling.root_segment6, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<', label=g.labels.SubOrgan.RootSegment))
    ds.update_topology()
    model._invoke_graph_system("_self_stepping_solve")

    assert model._graph_view.n_nodes == n_before + 1 == ds.n_nodes()
    residual = model._last_graph_system.residual(model._last_graph_solution)
    np.testing.assert_allclose(residual, 0., atol=1e-10)
    assert ds.get("concentration").shape == (ds.n_nodes(),)
