"""
UC3 — a steady hydraulic network with conductances typed by tissue, Robin exchanges with the soil (at the root tips)
and the xylem (at the collar) given as boundary sets, an analytic Jacobian and an edge-flux @graph_output, on the
seedling's root-system graph.
UC4 — a Laplacian on a 3-segment chain with a Dirichlet or a Neumann boundary set at the collar.

Everything runs through the components' calls and is checked by hand on the DataStructure's values.
"""
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.sparse import diags

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, parameter, state_variable
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.solve.decorator import boundary_set, graph_jacobian, graph_output, graph_system, node_balance
from simple_seedling import generate_simple_mpg_seedling

DOC = dict(unit="", unit_comment="", description="", min_value=-10., max_value=10., value_comment="", references="",
           DOI=[])
SOIL_WEIGHT, XYLEM_WEIGHT = 0.6, 1.0


@pytest.fixture(autouse=True)
def _simulation_time_step():
    Choregrapher().add_simulation_time_step(1)


# ══════════════════════════════════════════════════════════════════════════════
# UC3 — typed conductances and Robin exchanges
# ══════════════════════════════════════════════════════════════════════════════

def _total_conductance(K_membrane, K_symplastic, K_apoplastic):
    return K_membrane + K_symplastic + K_apoplastic


class _Hydraulics:
    @node_balance(field="water_potential")
    def _balance(self, water_potential, K_membrane, K_symplastic, K_apoplastic):
        B = self._graph_view.incidence
        K = _total_conductance(K_membrane, K_symplastic, K_apoplastic)
        return np.asarray((B @ diags(K) @ B.T) @ water_potential).reshape(-1)

    # the exchanges, w (p - v), added to the balance by the framework (and to the analytic Jacobian)
    soil = boundary_set(filters={"is_soil": ">0"}, kind="robin", value="soil_water_potential", weight=SOIL_WEIGHT)
    collar = boundary_set(filters={"is_collar": ">0"}, kind="robin", value="xylem_water_potential", weight=XYLEM_WEIGHT)

    @graph_output(name="edge_water_flux", location="edge")
    def _edge_flux(self, water_potential, K_membrane, K_symplastic, K_apoplastic):
        B = self._graph_view.incidence
        return _total_conductance(K_membrane, K_symplastic, K_apoplastic) * np.asarray(B.T @ water_potential).reshape(-1)


class _HydraulicsWithJacobian(_Hydraulics):
    @graph_jacobian
    def _jacobian(self, K_membrane, K_symplastic, K_apoplastic):
        B = self._graph_view.incidence
        return (B @ diags(_total_conductance(K_membrane, K_symplastic, K_apoplastic)) @ B.T).toarray()


@dataclass
class TissueHydraulics(FunctionalComponent):
    water_potential: float = state_variable(**DOC, initialize=0., location="node", state_variable_type="intensive")
    K_membrane: float = parameter(**DOC, by="TissueHydraulics", default=0., location="edge")
    K_symplastic: float = parameter(**DOC, by="TissueHydraulics", default=0., location="edge")
    K_apoplastic: float = parameter(**DOC, by="TissueHydraulics", default=0., location="edge")
    soil_water_potential: float = input_variable(**DOC, by="Soil", initialize=0., location="node")
    xylem_water_potential: float = input_variable(**DOC, by="Xylem", initialize=-1., location="node")
    is_soil: float = input_variable(**DOC, by="Setup", initialize=0., location="node")
    is_collar: float = input_variable(**DOC, by="Setup", initialize=0., location="node")
    edge_water_flux: float = state_variable(**DOC, initialize=0., location="edge", state_variable_type="extensive")

    @graph_system(node_unknowns=["water_potential"], solver="newton", max_iter=5)
    class _pressure_solve(_HydraulicsWithJacobian):
        pass


@dataclass
class TissueHydraulicsOneStep(TissueHydraulics):
    """The analytic Jacobian with one Newton step and its convergence check."""
    steps_removed = ("pressure_solve",)

    @graph_system(node_unknowns=["water_potential"], solver="newton", max_iter=2)
    class _pressure_solve_one_step(_HydraulicsWithJacobian):
        pass


@dataclass
class TissueHydraulicsFD(TissueHydraulics):
    """The finite-difference Jacobian."""
    steps_removed = ("pressure_solve",)

    @graph_system(node_unknowns=["water_potential"], solver="newton_fd", max_iter=5)
    class _pressure_solve_fd(_Hydraulics):
        pass


def _anatomy(component_class=TissueHydraulics):
    """
    The seedling root-system graph, with conductances typed by the child segment's label (stem: symplastic, root:
    transmembrane, leaf: apoplastic), the soil at the root tips and the xylem at the collar. Returns (ds, soil, collar).
    """
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    model = component_class(data_structure=ds)

    labels = g.property("label")
    root = g.labels.SubOrgan.RootSegment
    collar = [int(ds.roots()[0])]
    soil = [int(i) for i in ds.tips() if labels[ds.entity_ids("node")[i]] == root]
    for name, nodes in (("is_soil", soil), ("is_collar", collar)):
        flags = np.zeros(ds.n_nodes())
        flags[nodes] = 1.
        ds.set(name, flags)
    child = np.array([labels[b] for _, b in ds.edges()])
    K_sym = np.where(child == g.labels.SubOrgan.StemElement, 0.80, 0.)
    K_mem = np.where(child == root, 0.35, 0.)
    ds.set("K_symplastic", K_sym)
    ds.set("K_membrane", K_mem)
    ds.set("K_apoplastic", np.where((K_sym == 0.) & (K_mem == 0.), 1.10, 0.))
    model()
    return ds, soil, collar


def _conductances(ds):
    return np.asarray(ds.get("K_membrane")) + np.asarray(ds.get("K_symplastic")) + np.asarray(ds.get("K_apoplastic"))


def test_uc3_the_balance_holds_and_water_flows_to_the_xylem():
    ds, soil, collar = _anatomy()
    B, p = ds.incidence_matrix().toarray(), np.asarray(ds.get("water_potential"))
    residual = B @ (_conductances(ds) * (B.T @ p))
    residual[soil] += SOIL_WEIGHT * (p[soil] - 0.)
    residual[collar] += XYLEM_WEIGHT * (p[collar] - (-1.))
    np.testing.assert_allclose(residual, 0., atol=1e-10)
    assert -1. - 1e-10 <= p.min() and p.max() <= 1e-10                          # bounded by the external values
    assert p[collar].mean() < p[soil].mean()


def test_uc3_the_edge_flux_output_is_written():
    ds, _, _ = _anatomy()
    B, p = ds.incidence_matrix().toarray(), np.asarray(ds.get("water_potential"))
    flux = np.asarray(ds.get("edge_water_flux"))
    np.testing.assert_allclose(flux, _conductances(ds) * (B.T @ p), atol=1e-12)
    assert np.abs(flux).max() > 0


def test_uc3_the_analytic_jacobian_gives_the_solution_in_one_newton_step():
    """A linear system and an exact Jacobian (boundary sets included): one step from zero reaches the solution."""
    one_step, _, _ = _anatomy(TissueHydraulicsOneStep)
    fd, _, _ = _anatomy(TissueHydraulicsFD)
    np.testing.assert_allclose(one_step.get("water_potential"), fd.get("water_potential"), atol=1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# UC4 — Dirichlet and Neumann boundary sets
# ══════════════════════════════════════════════════════════════════════════════

def _laplacian(self, pressure, K):
    B = self._graph_view.incidence
    return np.asarray((B @ diags(K) @ B.T) @ pressure).reshape(-1)


@dataclass
class ChainFields(FunctionalComponent):
    pressure: float = state_variable(**DOC, initialize=0., location="node", state_variable_type="intensive")
    is_collar: float = input_variable(**DOC, by="Setup", initialize=0., location="node")
    K: float = parameter(**DOC, by="ChainFields", default=1., location="edge")


@dataclass
class ChainDirichlet(ChainFields):
    @graph_system(node_unknowns=["pressure"], solver="newton", max_iter=50, tol=1e-12)
    class _solve:
        _balance = node_balance(field="pressure")(_laplacian)
        collar = boundary_set(filters={"is_collar": ">0"}, kind="dirichlet", value=2.0)


@dataclass
class ChainNeumann(ChainFields):
    @graph_system(node_unknowns=["pressure"], solver="newton", max_iter=50, tol=1e-12)
    class _solve:
        @node_balance(field="pressure")
        def _balance(self, pressure, K):
            return _laplacian(self, pressure, K) + 0.5 * pressure          # a leakage, so that the system is regular

        collar = boundary_set(filters={"is_collar": ">0"}, kind="neumann", value=1.0)   # an inflow of 1.0


def _chain(component_class):
    """3 segments (3 nodes, 2 edges), K = 1, the collar at the graph root."""
    g = MPG()
    scale = g.scales.SubOrgan
    vid = g.add_system_root_at_scale(scale, label=g.labels.SubOrgan.RootSegment)
    for _ in range(2):
        vid = g.add_component_with_topo(g.scales.anchors[scale], vid, **PropsConfig(
            scale=scale, edge_type="<", label=g.labels.SubOrgan.RootSegment))
    g.populate_graph(scale)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=scale)
    model = component_class(data_structure=ds)
    collar = np.zeros(ds.n_nodes())
    collar[ds.roots()[0]] = 1.
    ds.set("is_collar", collar)
    model()
    return ds


def test_uc4_a_dirichlet_set_brings_the_whole_chain_to_its_value():
    ds = _chain(ChainDirichlet)
    np.testing.assert_allclose(ds.get("pressure"), 2.0, atol=1e-10)


def test_uc4_a_neumann_inflow_is_balanced_by_the_leakage():
    ds = _chain(ChainNeumann)
    B, p = ds.incidence_matrix().toarray(), np.asarray(ds.get("pressure"))
    collar = int(ds.roots()[0])
    inflow = np.zeros(ds.n_nodes())
    inflow[collar] = 1.
    np.testing.assert_allclose(B @ (B.T @ p) + 0.5 * p - inflow, 0., atol=1e-10)
    assert p.sum() * 0.5 == pytest.approx(1.)                                   # the leakage carries the inflow out
    assert p[collar] == p.max()
