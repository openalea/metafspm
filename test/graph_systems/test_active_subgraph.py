"""
Graph systems on the active subgraph: where= restricts
a solve to the nodes of a mask and the edges between them; inactive nodes are frozen, dropped edges carry no flux,
and steady systems need an anchor in every connected piece.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.solve.decorator import boundary_set, edge_law, graph_system, node_balance
from simple_seedling import generate_simple_mpg_seedling

DOC = dict(unit="adim", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])



def _diffusion(**options):
    @graph_system(node_unknowns=["concentration"], edge_unknowns=["flux"], solver="newton", **options)
    class _diffusion:
        @node_balance(field="concentration")
        def _balance(self, concentration, flux, source):
            B = self.graph_view.incidence
            return (concentration - self.previous("concentration")) / self.step_duration \
                + np.asarray(B @ flux).reshape(-1) - source

        @edge_law(field="flux")
        def _fick(self, concentration, flux, K):
            B = self.graph_view.incidence
            return flux - K * np.asarray(B.T @ concentration).reshape(-1)

    return _diffusion


@dataclass
class DiffusionFields(FunctionalComponent):
    concentration: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)
    flux: float = state_variable(**DOC, default=0., location="edge")
    K: float = parameter(**DOC, by="", default=0.5, location="edge")
    source: float = parameter(**DOC, by="", default=0., location="node")
    time_step = 1.


@dataclass
class WholeDiffusion(DiffusionFields):
    _diffusion = _diffusion(transient=True)


@dataclass
class ActiveDiffusion(DiffusionFields):
    _diffusion = _diffusion(where="active", transient=True)


def _ds(alive=None):
    mtg, s = generate_simple_mpg_seedling()
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
    data_structure.register("concentration", np.linspace(1., 2., data_structure.n_nodes()), location="node")
    data_structure.register("source", np.linspace(0., 0.3, data_structure.n_nodes()), location="node")
    data_structure.register("alive", 1. if alive is None else alive, location="node")
    data_structure.define_mask("active", {"alive": ">0"})
    return s, data_structure


def test_all_active_equals_the_whole_graph_solve():
    _, whole_ds = _ds()
    _, active_ds = _ds()
    whole, active = WholeDiffusion(data_structure=whole_ds), ActiveDiffusion(data_structure=active_ds)
    for _ in range(2):
        whole(); active()
    np.testing.assert_array_equal(active_ds.get("concentration"), whole_ds.get("concentration"))
    np.testing.assert_array_equal(active_ds.get("flux"), whole_ds.get("flux"))


def test_inactive_nodes_are_frozen_and_dropped_edges_carry_no_flux():
    s, data_structure = _ds()
    dead = data_structure.index_of(s.root_segment2)            # cuts the root axis below it and the lateral branch off
    alive = np.ones(data_structure.n_nodes()); alive[dead] = 0.
    data_structure.set("alive", alive)
    model = ActiveDiffusion(data_structure=data_structure)
    before = np.array(data_structure.get("concentration"))
    model()
    after = data_structure.get("concentration")
    assert after[dead] == before[dead]
    tail, head = model._graph_view.tail, model._graph_view.head
    touching = (tail == dead) | (head == dead)
    np.testing.assert_array_equal(data_structure.get("flux")[touching], 0.)
    active = alive > 0                             # mass balance over the active nodes (dt = 1)
    assert (after - before)[active].sum() == pytest.approx(data_structure.get("source")[active].sum())


def test_a_node_activated_between_two_steps_joins_the_solve():
    s, data_structure = _ds()
    tip = data_structure.index_of(s.root_segment6)
    alive = np.ones(data_structure.n_nodes()); alive[tip] = 0.
    data_structure.set("alive", alive)
    model = ActiveDiffusion(data_structure=data_structure)
    model()
    frozen = float(data_structure.get("concentration")[tip])
    data_structure.set("alive", 1.)                            # it emerges: the mask changes through its variable
    before = np.array(data_structure.get("concentration"))
    model()
    after = data_structure.get("concentration")
    assert after[tip] != frozen                    # solved again
    assert (after - before).sum() == pytest.approx(data_structure.get("source").sum())   # previous() = its value at activation


def test_an_empty_mask_skips_the_solve():
    _, data_structure = _ds(alive=0.)
    model = ActiveDiffusion(data_structure=data_structure)
    data_structure.set("flux", 1.)
    before = np.array(data_structure.get("concentration"))
    model()
    np.testing.assert_array_equal(data_structure.get("concentration"), before)
    np.testing.assert_array_equal(data_structure.get("flux"), 0.)


# ---------------------------------------------------------------- steady systems need anchors

@dataclass
class SteadyPotential(FunctionalComponent):
    potential: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)
    water_flux: float = state_variable(**DOC, default=0., location="edge")
    K: float = parameter(**DOC, by="", default=1., location="edge")
    is_collar: float = parameter(**DOC, by="", default=0., location="node")

    @graph_system(node_unknowns=["potential"], edge_unknowns=["water_flux"], solver="newton", where="active")
    class _steady:
        @node_balance(field="potential")
        def _balance(self, water_flux):
            return np.asarray(self.graph_view.incidence @ water_flux).reshape(-1)

        collar = boundary_set(where={"is_collar": [1]}, condition="dirichlet", value=-0.1)

        @edge_law(field="water_flux")
        def _darcy(self, potential, water_flux, K):
            return water_flux - K * np.asarray(self.graph_view.incidence.T @ potential).reshape(-1)


def s_vid(name):
    """Vertex id of a named seedling vertex (the seedling is rebuilt identically)."""
    _, s = generate_simple_mpg_seedling()
    return getattr(s, name)


def _steady_ds(dead_vid=None):
    s, data_structure = _ds()
    collar = np.zeros(data_structure.n_nodes()); collar[data_structure.roots()[0]] = 1.
    data_structure.register("is_collar", collar, location="node")
    if dead_vid is not None:
        alive = np.ones(data_structure.n_nodes()); alive[data_structure.index_of(dead_vid)] = 0.
        data_structure.set("alive", alive)
    return s, data_structure


def test_a_steady_connected_subgraph_with_an_anchor_solves():
    _, data_structure = _steady_ds()
    SteadyPotential(data_structure=data_structure)()
    np.testing.assert_allclose(data_structure.get("potential"), -0.1)     # no flux: uniform at the anchor's value


def test_a_steady_piece_without_anchor_raises_with_its_nodes():
    s, data_structure = _steady_ds(dead_vid=s_vid("root_segment2"))
    model = SteadyPotential(data_structure=data_structure)
    with pytest.raises(ValueError, match=r"_steady: piece of \d+ nodes .* has no Dirichlet or positive-weight Robin anchor"):
        model()


def test_transient_defaults_from_the_solver():
    systems = WholeDiffusion._graph_system_specs
    assert systems["_diffusion"]["transient"] is True                       # declared
    assert SteadyPotential._graph_system_specs["_steady"]["transient"] is False   # newton: steady by default

    @graph_system(node_unknowns=["c"], solver="explicit_euler")
    class _explicit:
        pass
    assert _explicit._spec["transient"] is True


