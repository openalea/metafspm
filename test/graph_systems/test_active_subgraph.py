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
from openalea.metafspm.data_structure.data_api import BoundaryPort, MPGDataStructure
from openalea.metafspm.solve.decorator import boundary_set, edge_law, graph_system, node_balance
from simple_seedling import generate_simple_mpg_seedling

DOC = dict(unit="", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])



def _diffusion(**options):
    @graph_system(node_unknowns=["concentration"], edge_unknowns=["flux"], solver="newton", **options)
    class _diffusion:
        @node_balance(field="concentration")
        def _balance(self, concentration, flux, source):
            B = self._graph_view.incidence
            return (concentration - self.previous("concentration")) / self.dt \
                + np.asarray(B @ flux).reshape(-1) - source

        @edge_law(field="flux")
        def _fick(self, concentration, flux, K):
            B = self._graph_view.incidence
            return flux - K * np.asarray(B.T @ concentration).reshape(-1)

    return _diffusion


@dataclass
class DiffusionFields(FunctionalComponent):
    concentration: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)
    flux: float = state_variable(**DOC, initialize=0., location="edge")
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
    g, s = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    ds.register("concentration", np.linspace(1., 2., ds.n_nodes()), location="node")
    ds.register("source", np.linspace(0., 0.3, ds.n_nodes()), location="node")
    ds.register("alive", 1. if alive is None else alive, location="node")
    ds.define_mask("active", {"alive": ">0"})
    return s, ds


def test_all_active_equals_the_whole_graph_solve():
    _, whole_ds = _ds()
    _, active_ds = _ds()
    whole, active = WholeDiffusion(data_structure=whole_ds), ActiveDiffusion(data_structure=active_ds)
    for _ in range(2):
        whole(); active()
    np.testing.assert_array_equal(active_ds.get("concentration"), whole_ds.get("concentration"))
    np.testing.assert_array_equal(active_ds.get("flux"), whole_ds.get("flux"))


def test_inactive_nodes_are_frozen_and_dropped_edges_carry_no_flux():
    s, ds = _ds()
    dead = ds.index_of(s.root_segment2)            # cuts the root axis below it and the lateral branch off
    alive = np.ones(ds.n_nodes()); alive[dead] = 0.
    ds.set("alive", alive)
    model = ActiveDiffusion(data_structure=ds)
    before = np.array(ds.get("concentration"))
    model()
    after = ds.get("concentration")
    assert after[dead] == before[dead]
    tail, head = model._graph_view.tail, model._graph_view.head
    touching = (tail == dead) | (head == dead)
    np.testing.assert_array_equal(ds.get("flux")[touching], 0.)
    active = alive > 0                             # mass balance over the active nodes (dt = 1)
    assert (after - before)[active].sum() == pytest.approx(ds.get("source")[active].sum())


def test_a_node_activated_between_two_steps_joins_the_solve():
    s, ds = _ds()
    tip = ds.index_of(s.root_segment6)
    alive = np.ones(ds.n_nodes()); alive[tip] = 0.
    ds.set("alive", alive)
    model = ActiveDiffusion(data_structure=ds)
    model()
    frozen = float(ds.get("concentration")[tip])
    ds.set("alive", 1.)                            # it emerges: the mask changes through its variable
    before = np.array(ds.get("concentration"))
    model()
    after = ds.get("concentration")
    assert after[tip] != frozen                    # solved again
    assert (after - before).sum() == pytest.approx(ds.get("source").sum())   # previous() = its value at activation


def test_an_empty_mask_skips_the_solve():
    _, ds = _ds(alive=0.)
    model = ActiveDiffusion(data_structure=ds)
    ds.set("flux", 1.)
    before = np.array(ds.get("concentration"))
    model()
    np.testing.assert_array_equal(ds.get("concentration"), before)
    np.testing.assert_array_equal(ds.get("flux"), 0.)


# ---------------------------------------------------------------- steady systems need anchors

@dataclass
class SteadyPotential(FunctionalComponent):
    potential: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)
    water_flux: float = state_variable(**DOC, initialize=0., location="edge")
    K: float = parameter(**DOC, by="", default=1., location="edge")
    is_collar: float = parameter(**DOC, by="", default=0., location="node")

    @graph_system(node_unknowns=["potential"], edge_unknowns=["water_flux"], solver="newton", where="active")
    class _steady:
        @node_balance(field="potential")
        def _balance(self, water_flux):
            return np.asarray(self._graph_view.incidence @ water_flux).reshape(-1)

        collar = boundary_set(select={"is_collar": [1]}, kind="dirichlet", value=-0.1)

        @edge_law(field="water_flux")
        def _darcy(self, potential, water_flux, K):
            return water_flux - K * np.asarray(self._graph_view.incidence.T @ potential).reshape(-1)


def s_vid(name):
    """Vertex id of a named seedling vertex (the seedling is rebuilt identically)."""
    _, s = generate_simple_mpg_seedling()
    return getattr(s, name)


def _steady_ds(dead_vid=None):
    s, ds = _ds()
    collar = np.zeros(ds.n_nodes()); collar[ds.roots()[0]] = 1.
    ds.register("is_collar", collar, location="node")
    if dead_vid is not None:
        alive = np.ones(ds.n_nodes()); alive[ds.index_of(dead_vid)] = 0.
        ds.set("alive", alive)
    return s, ds


def test_a_steady_connected_subgraph_with_an_anchor_solves():
    _, ds = _steady_ds()
    SteadyPotential(data_structure=ds)()
    np.testing.assert_allclose(ds.get("potential"), -0.1)     # no flux: uniform at the anchor's value


def test_a_steady_piece_without_anchor_raises_with_its_nodes():
    s, ds = _steady_ds(dead_vid=s_vid("root_segment2"))
    model = SteadyPotential(data_structure=ds)
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


def test_hand_set_boundary_ports_cannot_follow_an_active_subgraph():
    _, ds = _ds()
    model = ActiveDiffusion(data_structure=ds)
    model._boundary_ports = (BoundaryPort(name="p", node_id=int(ds.entity_ids("node")[0]), kind="robin", value=0.),)
    with pytest.raises(NotImplementedError, match="use boundary sets"):
        model()
