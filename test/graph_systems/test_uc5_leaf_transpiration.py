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

DOC = dict(unit="adim", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])


def _is_leaf(data_structure):
    return data_structure.get("label") == data_structure.mtg.labels.SubOrgan.LeafElement


def _water(jacobian=False, **options):
    @graph_system(node_unknowns=["water_potential"], edge_unknowns=["axial_flux"], solver="newton",
                  schedule_as="state", **options)
    class _water:
        leaves = boundary_set(where=_is_leaf, condition="robin", value="air_water_potential", weight="leaf_conductance")
        roots = boundary_set(where={"root_surface": ">0"}, condition="robin", value="soil_water_potential",
                             weight="radial_conductance")

        @node_balance(field="water_potential")
        def _balance(self, axial_flux):
            return np.asarray(self.graph_view.incidence @ axial_flux).reshape(-1)

        @edge_law(field="axial_flux")
        def _darcy(self, water_potential, axial_flux, K_axial):
            return axial_flux - K_axial * np.asarray(self.graph_view.incidence.T @ water_potential).reshape(-1)

        if jacobian:
            @graph_jacobian
            def _jacobian(self, K_axial):
                """The equations only: the framework adds the boundary sets' terms."""
                B = self.graph_view.incidence.toarray()
                n, m = B.shape
                J = np.zeros((n + m, n + m))
                J[:n, n:] = B
                J[n:, :n] = -K_axial[:, None] * B.T
                J[n:, n:] = np.eye(m)
                return J

    return _water


@dataclass
class TranspirationFields(FunctionalComponent):
    water_potential: float = state_variable(**DOC, default=-0.5, scale=scales.SubOrgan,
                                            kind="intensive")
    axial_flux: float = state_variable(**DOC, default=0., location="edge")
    K_axial: float = parameter(**DOC, by="", default=2., location="edge")
    label: float = parameter(**DOC, by="", default=0., scale=scales.SubOrgan)
    root_surface: float = parameter(**DOC, by="", default=0., scale=scales.SubOrgan)
    air_water_potential: float = input_variable(**DOC, by="Microclimate", default=-2., scale=scales.SubOrgan)
    soil_water_potential: float = input_variable(**DOC, by="Soil", default=-0.3, scale=scales.SubOrgan)
    stomatal_conductance: float = input_variable(**DOC, by="Stomata", default=0.01, scale=scales.SubOrgan)
    exchange_surface: float = parameter(**DOC, by="", default=2., scale=scales.SubOrgan)
    leaf_conductance: float = state_variable(**DOC, default=0., scale=scales.SubOrgan,
                                             kind="non_inertial_intensive")
    radial_conductance: float = parameter(**DOC, by="", default=0.5, scale=scales.SubOrgan)


@dataclass
class Stomata(FunctionalComponent):
    """Leaf conductance from stomatal conductance and exchange surface, for the transpiration component."""
    stomatal_conductance: float = input_variable(**DOC, by="Microclimate", default=0.01, scale=scales.SubOrgan)
    exchange_surface: float = parameter(**DOC, by="", default=2., scale=scales.SubOrgan)
    leaf_conductance: float = state_variable(**DOC, default=0., scale=scales.SubOrgan,
                                             kind="non_inertial_intensive")

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
    _water = _water(where="active")



def _plant():
    mtg, s = generate_simple_mpg_seedling()
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
    label = mtg.property("label")
    mtg.properties()["root_surface"] = {v: float(label[v] == mtg.labels.SubOrgan.RootSegment) for v in data_structure.entity_ids("node")}
    return mtg, s, data_structure


def _microclimate(data_structure):
    """Per-leaf air water potential and stomatal conductance."""
    leaf = _is_leaf(data_structure)
    air = np.where(leaf, -2. - 0.1 * np.arange(data_structure.n_nodes()), -2.)
    data_structure.set("air_water_potential", air)
    data_structure.set("stomatal_conductance", np.where(leaf, 0.01 * (1. + np.arange(data_structure.n_nodes())), 0.01))


def _direct_solution(data_structure):
    B = data_structure._incidence_matrix().toarray()
    K = data_structure.get("K_axial")
    leaf, root = _is_leaf(data_structure), data_structure.get("root_surface") > 0
    w = np.where(leaf, data_structure.get("stomatal_conductance") * data_structure.get("exchange_surface"), 0.) \
        + np.where(root, data_structure.get("radial_conductance"), 0.)
    v = np.where(leaf, data_structure.get("air_water_potential"), 0.) + np.where(root, data_structure.get("soil_water_potential"), 0.)
    return np.linalg.solve(B @ np.diag(K) @ B.T + np.diag(w), w * v)


@pytest.mark.parametrize("model_class", [LeafTranspiration, LeafTranspirationWithJacobian])
def test_the_water_potential_matches_the_direct_solve(model_class):
    _, _, data_structure = _plant()
    model, stomata = model_class(data_structure=data_structure), Stomata(data_structure=data_structure)
    _microclimate(data_structure)
    stomata(); model()
    np.testing.assert_allclose(data_structure.get("water_potential"), _direct_solution(data_structure), rtol=1e-8, atol=1e-10)
    psi = data_structure.get("water_potential")
    assert np.all(psi[_is_leaf(data_structure)] < psi[data_structure.get("root_surface") > 0].max())   # water flows from roots to leaves


def test_a_leaf_microclimate_change_is_seen_at_the_next_solve_without_rebuilding():
    _, _, data_structure = _plant()
    model, stomata = LeafTranspiration(data_structure=data_structure), Stomata(data_structure=data_structure)
    _microclimate(data_structure)
    stomata(); model()
    view = model._graph_view
    leaf = int(np.flatnonzero(_is_leaf(data_structure))[0])
    air = np.array(data_structure.get("air_water_potential")); air[leaf] = -5.
    data_structure.set("air_water_potential", air)
    model()
    assert model._graph_view is view
    np.testing.assert_allclose(data_structure.get("water_potential"), _direct_solution(data_structure), rtol=1e-8, atol=1e-10)


def test_a_grown_leaf_joins_the_leaf_set():
    mtg, s, data_structure = _plant()
    model, stomata = LeafTranspiration(data_structure=data_structure), Stomata(data_structure=data_structure)
    _microclimate(data_structure)
    stomata(); model()
    new_leaf = mtg.add_child(s.leafelement3, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<',
                                                         label=mtg.labels.SubOrgan.LeafElement, root_surface=0.))
    data_structure.update_topology()
    stomata(); model()                    # the label of the new leaf is read from the MTG before the solve
    assert _is_leaf(data_structure)[data_structure.index_of(new_leaf)]
    np.testing.assert_allclose(data_structure.get("water_potential"), _direct_solution(data_structure), rtol=1e-8, atol=1e-10)


def test_robin_sets_anchor_steady_pieces_of_an_active_subgraph():
    """Cutting the stem base leaves a shoot piece anchored by its leaves and a root piece by its roots."""
    _, s, data_structure = _plant()
    data_structure.register("alive", 1., location="node")
    alive = np.ones(data_structure.n_nodes()); alive[data_structure.index_of(s.internodeelement)] = 0.
    data_structure.set("alive", alive)
    data_structure.define_mask("active", {"alive": ">0"})
    model, stomata = ActiveLeafTranspiration(data_structure=data_structure), Stomata(data_structure=data_structure)
    _microclimate(data_structure)
    stomata(); model()                                            # steady, no Dirichlet: Robin weights anchor both pieces
    assert np.isfinite(data_structure.get("water_potential")).all()


def test_boundary_set_declarations_are_checked():
    with pytest.raises(ValueError, match="condition must be one of"):
        boundary_set(where="x", condition="flux", value=0.)
    with pytest.raises(TypeError, match="where must be"):
        boundary_set(where=3, condition="robin", value=0.)
