"""
UC5, leaf transpiration: boundary sets assembled by the
framework. A steady water potential on the seedling's shoot and root graph, with
  * axial conductance on edges;
  * leaves: Robin to a per-leaf air water potential (microclimate input), with a leaf conductance computed by the
    Stomata component (a @rate) as stomatal conductance x exchange surface;
  * roots: Robin to the soil water potential.
A use case of the tool, not a validated plant model. Results are checked against the direct linear solve of
(B diag(K) B^T + diag(w_b)) psi = w_b v_b.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, parameter, state_variable
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.solve.decorator import (boundary_set, edge_law, graph_jacobian, graph_system, node_balance,
                                               rate)
from simple_seedling import generate_simple_mpg_seedling

DOC = dict(unit="", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])


def _is_leaf(ds):
    return ds.get("label") == ds.mtg.labels.SubOrgan.LeafElement


def _water(jacobian=False, **options):
    @graph_system(node_unknowns=["water_potential"], edge_unknowns=["axial_flux"], solver="newton",
                  schedule_as="state", **options)
    class _water:
        leaves = boundary_set(filters=_is_leaf, kind="robin", value="air_water_potential", weight="leaf_conductance")
        roots = boundary_set(filters={"root_surface": ">0"}, kind="robin", value="soil_water_potential",
                             weight="radial_conductance")

        @node_balance(field="water_potential")
        def _balance(self, axial_flux):
            return np.asarray(self._graph_view.incidence @ axial_flux).reshape(-1)

        @edge_law(field="axial_flux")
        def _darcy(self, water_potential, axial_flux, K_axial):
            return axial_flux - K_axial * np.asarray(self._graph_view.incidence.T @ water_potential).reshape(-1)

        if jacobian:
            @graph_jacobian
            def _jacobian(self, K_axial):
                """The equations only: the framework adds the boundary sets' terms."""
                B = self._graph_view.incidence.toarray()
                n, m = B.shape
                J = np.zeros((n + m, n + m))
                J[:n, n:] = B
                J[n:, :n] = -K_axial[:, None] * B.T
                J[n:, n:] = np.eye(m)
                return J

    return _water


@dataclass
class TranspirationFields(FunctionalComponent):
    water_potential: float = state_variable(**DOC, initialize=-0.5, scale=scales.SubOrgan,
                                            state_variable_type="intensive")
    axial_flux: float = state_variable(**DOC, initialize=0., location="edge")
    K_axial: float = parameter(**DOC, by="", default=2., location="edge")
    label: float = parameter(**DOC, by="", default=0., scale=scales.SubOrgan)
    root_surface: float = parameter(**DOC, by="", default=0., scale=scales.SubOrgan)
    air_water_potential: float = input_variable(**DOC, by="Microclimate", initialize=-2., scale=scales.SubOrgan)
    soil_water_potential: float = input_variable(**DOC, by="Soil", initialize=-0.3, scale=scales.SubOrgan)
    stomatal_conductance: float = input_variable(**DOC, by="Stomata", initialize=0.01, scale=scales.SubOrgan)
    exchange_surface: float = parameter(**DOC, by="", default=2., scale=scales.SubOrgan)
    leaf_conductance: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan,
                                             state_variable_type="NonInertialIntensive")
    radial_conductance: float = parameter(**DOC, by="", default=0.5, scale=scales.SubOrgan)


@dataclass
class Stomata(FunctionalComponent):
    """Leaf conductance from stomatal conductance and exchange surface, for the transpiration component."""
    stomatal_conductance: float = input_variable(**DOC, by="Microclimate", initialize=0.01, scale=scales.SubOrgan)
    exchange_surface: float = parameter(**DOC, by="", default=2., scale=scales.SubOrgan)
    leaf_conductance: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan,
                                             state_variable_type="NonInertialIntensive")

    @rate
    def _leaf_conductance(self, stomatal_conductance, exchange_surface):
        return stomatal_conductance * exchange_surface


@dataclass
class LeafTranspiration(TranspirationFields):
    _water = _water()


@dataclass
class LeafTranspirationWithJacobian(TranspirationFields):
    _water = _water(jacobian=True)


@dataclass
class ActiveLeafTranspiration(TranspirationFields):
    _water = _water(filters="active")



def _plant():
    g, s = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    label = g.property("label")
    g.properties()["root_surface"] = {v: float(label[v] == g.labels.SubOrgan.RootSegment) for v in ds.entity_ids("node")}
    return g, s, ds


def _microclimate(ds):
    """Per-leaf air water potential and stomatal conductance."""
    leaf = _is_leaf(ds)
    air = np.where(leaf, -2. - 0.1 * np.arange(ds.n_nodes()), -2.)
    ds.set("air_water_potential", air)
    ds.set("stomatal_conductance", np.where(leaf, 0.01 * (1. + np.arange(ds.n_nodes())), 0.01))


def _direct_solution(ds):
    B = ds.incidence_matrix().toarray()
    K = ds.get("K_axial")
    leaf, root = _is_leaf(ds), ds.get("root_surface") > 0
    w = np.where(leaf, ds.get("stomatal_conductance") * ds.get("exchange_surface"), 0.) \
        + np.where(root, ds.get("radial_conductance"), 0.)
    v = np.where(leaf, ds.get("air_water_potential"), 0.) + np.where(root, ds.get("soil_water_potential"), 0.)
    return np.linalg.solve(B @ np.diag(K) @ B.T + np.diag(w), w * v)


@pytest.mark.parametrize("model_class", [LeafTranspiration, LeafTranspirationWithJacobian])
def test_the_water_potential_matches_the_direct_solve(model_class):
    _, _, ds = _plant()
    model, stomata = model_class(data_structure=ds), Stomata(data_structure=ds)
    _microclimate(ds)
    stomata(); model()
    np.testing.assert_allclose(ds.get("water_potential"), _direct_solution(ds), rtol=1e-8, atol=1e-10)
    psi = ds.get("water_potential")
    assert np.all(psi[_is_leaf(ds)] < psi[ds.get("root_surface") > 0].max())   # water flows from roots to leaves


def test_a_leaf_microclimate_change_is_seen_at_the_next_solve_without_rebuilding():
    _, _, ds = _plant()
    model, stomata = LeafTranspiration(data_structure=ds), Stomata(data_structure=ds)
    _microclimate(ds)
    stomata(); model()
    view = model._graph_view
    leaf = int(np.flatnonzero(_is_leaf(ds))[0])
    air = np.array(ds.get("air_water_potential")); air[leaf] = -5.
    ds.set("air_water_potential", air)
    model()
    assert model._graph_view is view
    np.testing.assert_allclose(ds.get("water_potential"), _direct_solution(ds), rtol=1e-8, atol=1e-10)


def test_a_grown_leaf_joins_the_leaf_set():
    g, s, ds = _plant()
    model, stomata = LeafTranspiration(data_structure=ds), Stomata(data_structure=ds)
    _microclimate(ds)
    stomata(); model()
    new_leaf = g.add_child(s.leafelement3, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<',
                                                         label=g.labels.SubOrgan.LeafElement, root_surface=0.))
    ds.update_topology()
    stomata(); model()                    # the label of the new leaf is read from the MTG before the solve
    assert _is_leaf(ds)[ds.index_of(new_leaf)]
    np.testing.assert_allclose(ds.get("water_potential"), _direct_solution(ds), rtol=1e-8, atol=1e-10)


def test_robin_sets_anchor_steady_pieces_of_an_active_subgraph():
    """Cutting the stem base leaves a shoot piece anchored by its leaves and a root piece by its roots."""
    _, s, ds = _plant()
    ds.register("alive", 1., location="node")
    alive = np.ones(ds.n_nodes()); alive[ds.index_of(s.internodeelement)] = 0.
    ds.set("alive", alive)
    ds.define_mask("active", {"alive": ">0"})
    model, stomata = ActiveLeafTranspiration(data_structure=ds), Stomata(data_structure=ds)
    _microclimate(ds)
    stomata(); model()                                            # steady, no Dirichlet: Robin weights anchor both pieces
    assert np.isfinite(ds.get("water_potential")).all()


def test_boundary_set_declarations_are_checked():
    with pytest.raises(ValueError, match="kind must be one of"):
        boundary_set(filters="x", kind="flux", value=0.)
    with pytest.raises(TypeError, match="filters must be"):
        boundary_set(filters=3, kind="robin", value=0.)
