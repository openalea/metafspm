"""
Graph-system options through components: the solver argument and its errors, solver keys giving the same steady
solution, previous() at the start of the solve or of the sub-step, boundary_set variants (Neumann, Robin with a
constant or per-node weight, per-node kinds, field= with several unknowns), pool unknowns given by their location
only, forcings given as callables or (times, values) pairs, @graph_output write-back (location checks, active
subgraphs), integrated edge amounts on an active subgraph, and filtered node balances.
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene.population import build_population
from openalea.metafspm.solve.decorator import (boundary_set, edge_law, graph_output, graph_system, node_balance,
                                               pool_balance)
from openalea.metafspm.solve.solver import NewtonSolver

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mpg_tests"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "structure_tests"))
from simple_seedling import generate_simple_mpg_seedling
from growth import RootGrowthProbe

DOC = dict(unit="", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])
DT, K = 2., 0.3


@pytest.fixture(autouse=True)
def _simulation_time_step():
    Choregrapher().add_simulation_time_step(DT)


def _ds():
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    collar = np.zeros(ds.n_nodes())
    collar[ds.roots()[0]] = 1.
    ds.register("is_collar", collar, location="node")
    return ds


def _laplacian(ds):
    B = ds.incidence_matrix().toarray()
    return B @ B.T


def _diffusion_residual(self, u):
    B = self._graph_view.incidence
    return (u - self.previous("u")) / self.dt + K * np.asarray(B @ (B.T @ u)).reshape(-1)


def _implicit_step(ds, u0, robin_weight=None, robin_value=None):
    """One implicit Euler step of du/dt + K L u + w (u - v) = 0, solved by hand."""
    A = np.eye(ds.n_nodes()) / DT + K * _laplacian(ds)
    b = u0 / DT
    if robin_weight is not None:
        A += np.diag(robin_weight)
        b = b + robin_weight * robin_value
    return np.linalg.solve(A, b)


# ---------------------------------------------------------------- the solver argument

def test_solver_and_method_cannot_both_be_given():
    with pytest.raises(TypeError, match="both 'solver' and 'method'"):
        graph_system(node_unknowns=["u"], solver="implicit_euler", method="newton")


def test_an_unknown_solver_key_is_refused():
    with pytest.raises(ValueError, match="Unknown solver 'bogus'"):
        graph_system(node_unknowns=["u"], solver="bogus")


def test_a_solver_instance_is_refused():
    with pytest.raises(TypeError, match="must be a str or an uninstantiated"):
        graph_system(node_unknowns=["u"], solver=NewtonSolver())


# ---------------------------------------------------------------- solver keys give the same steady solution

class _SteadyEquations:
    @node_balance(field="u")
    def _balance(self, u, source):
        B = self._graph_view.incidence
        return K * np.asarray(B @ (B.T @ u)).reshape(-1) + u + 0.1 * u ** 3 - source


@dataclass
class SteadyFields(FunctionalComponent):
    u: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)
    source: float = parameter(**DOC, by="", default=1., location="node")
    time_step = DT


def _steady_class(name, solver):
    return dataclass(type(name, (SteadyFields,), {
        "__qualname__": name,
        "_solve": graph_system(node_unknowns=["u"], solver=solver)(type("_solve", (_SteadyEquations,), {}))}))


STEADY = {"newton": _steady_class("SteadyNewton", "newton"),
          "newton_fd": _steady_class("SteadyNewtonFD", "newton_fd"),
          "scipy_krylov": _steady_class("SteadyKrylov", "scipy_krylov"),
          "scipy_hybr": _steady_class("SteadyHybr", "scipy_hybr"),
          "NewtonSolver": _steady_class("SteadyNewtonClass", NewtonSolver)}


def _steady(solver):
    ds = _ds()
    model = STEADY[solver](data_structure=ds)
    ds.set("source", np.linspace(0., 2., ds.n_nodes()))
    model()
    return ds.get("u").copy()


@pytest.mark.parametrize("solver", ["newton_fd", "scipy_krylov", "scipy_hybr", "NewtonSolver"])
def test_solver_keys_and_classes_reach_the_same_steady_solution(solver):
    reference = _steady("newton")
    assert np.abs(reference).max() > 0.1
    np.testing.assert_allclose(_steady(solver), reference, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------- previous() with sub-steps

@dataclass
class Decay(FunctionalComponent):
    u: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)
    time_step = DT
    seen = None

    @graph_system(node_unknowns=["u"], transient=True, integrate="substeps", n_substeps=4)
    class _solve:
        @node_balance(field="u")
        def _balance(self, u):
            type(self).seen.append((self.previous("u", at="solve").copy(), self.previous("u").copy()))
            return (u - self.previous("u")) / self.dt + u


def test_previous_at_solve_is_the_start_of_the_call_and_at_substep_the_start_of_each_substep():
    ds = _ds()
    model = Decay(data_structure=ds)
    ds.set("u", np.linspace(1., 2., ds.n_nodes()))
    u0 = ds.get("u").copy()
    Decay.seen = []
    model()
    for at_solve, _ in Decay.seen:
        np.testing.assert_array_equal(at_solve, u0)
    starts = {tuple(np.round(at_substep, 12)) for _, at_substep in Decay.seen}
    assert len(starts) == 4                                                  # one start per sub-step
    np.testing.assert_allclose(ds.get("u"), u0 / (1. + DT / 4.) ** 4)


# ---------------------------------------------------------------- boundary_set variants

@dataclass
class Neumann(FunctionalComponent):
    u: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)
    time_step = DT

    @graph_system(node_unknowns=["u"], transient=True)
    class _solve:
        collar = boundary_set(select="is_collar", kind="neumann", value=0.25)
        _balance = node_balance(field="u")(_diffusion_residual)


def test_a_neumann_set_is_an_inflow_of_its_value():
    ds = _ds()
    model = Neumann(data_structure=ds)
    total = ds.get("u").sum()
    model()
    assert ds.get("u").sum() == pytest.approx(total + 0.25 * DT)


@dataclass
class ConstantRobin(FunctionalComponent):
    u: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)
    time_step = DT

    @graph_system(node_unknowns=["u"], transient=True)
    class _solve:
        collar = boundary_set(select="is_collar", kind="robin", value=0.5, weight=2.)
        _balance = node_balance(field="u")(_diffusion_residual)


def test_a_robin_set_with_a_constant_weight_is_an_exchange_towards_its_value():
    ds = _ds()
    model = ConstantRobin(data_structure=ds)
    u0 = np.linspace(1., 2., ds.n_nodes())
    ds.set("u", u0)
    model()
    collar = ds.get("is_collar") > 0
    np.testing.assert_allclose(ds.get("u"), _implicit_step(ds, u0, 2. * collar, 0.5), rtol=1e-10)


@dataclass
class RobinByCode(FunctionalComponent):
    u: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)
    kind_code: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, dtype="int")
    conductance: float = parameter(**DOC, by="", default=0., location="node")
    time_step = DT

    @graph_system(node_unknowns=["u"], transient=True)
    class _solve:
        tips = boundary_set(select=lambda ds: np.ones(ds.n_nodes(), dtype=bool), kinds="kind_code", value=3.,
                            weight="conductance")
        _balance = node_balance(field="u")(_diffusion_residual)


def test_per_node_kinds_apply_robin_where_the_code_says_so_with_per_node_weights():
    ds = _ds()
    model = RobinByCode(data_structure=ds)
    u0 = np.linspace(1., 2., ds.n_nodes())
    ds.set("u", u0)
    leaves = np.setdiff1d(np.arange(ds.n_nodes()), ds.parents())             # nodes no other node has as parent
    codes = np.zeros(ds.n_nodes(), dtype=np.int64)
    codes[leaves] = boundary_set.CODES["robin"]
    weights = np.linspace(0.5, 1.5, ds.n_nodes())
    ds.set("kind_code", codes)
    ds.set("conductance", weights)
    model()
    expected = _implicit_step(ds, u0, np.where(codes == 3, weights, 0.), 3.)
    np.testing.assert_allclose(ds.get("u"), expected, rtol=1e-10)


@dataclass
class TwoUnknowns(FunctionalComponent):
    u: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)
    w: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)
    time_step = DT

    @graph_system(node_unknowns=["u", "w"], transient=True)
    class _solve:
        collar = boundary_set(select="is_collar", kind="dirichlet", value=4., field="w")
        _u = node_balance(field="u")(_diffusion_residual)

        @node_balance(field="w")
        def _w(self, w):
            B = self._graph_view.incidence
            return (w - self.previous("w")) / self.dt + K * np.asarray(B @ (B.T @ w)).reshape(-1)


@dataclass
class TwoUnknownsWithoutField(FunctionalComponent):
    u: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)
    w: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)
    time_step = DT

    @graph_system(node_unknowns=["u", "w"], transient=True)
    class _solve:
        collar = boundary_set(select="is_collar", kind="dirichlet", value=4.)
        _u = node_balance(field="u")(_diffusion_residual)

        @node_balance(field="w")
        def _w(self, w):
            return w - self.previous("w")


def test_field_chooses_the_unknown_of_a_set_and_is_required_with_several_unknowns():
    ds = _ds()
    TwoUnknowns(data_structure=ds)()
    collar = ds.get("is_collar") > 0
    np.testing.assert_allclose(ds.get("w")[collar], 4.)
    np.testing.assert_allclose(ds.get("u"), 1.)                               # uniform, no condition on u
    with pytest.raises(ValueError, match="give field=, the system has several node unknowns"):
        TwoUnknownsWithoutField(data_structure=_ds())()


def test_boundary_set_arguments_are_checked():
    with pytest.raises(ValueError, match="not both"):
        boundary_set(select="is_collar", kind="robin", kinds="kind_code")
    with pytest.raises(ValueError, match="kind must be one of"):
        boundary_set(select="is_collar", kind="flux")
    with pytest.raises(TypeError, match="select must be"):
        boundary_set(select=3, kind="dirichlet")


# ---------------------------------------------------------------- pools given by their location

EXCHANGE = 0.8


def _population(sizes=(3, 4)):
    table = pd.DataFrame([dict(plant=f"p{i}", model=None, x=0.1 * i, y=0., z=0., rotation=0.,
                               scenario={"parameters": {"n_segments": n}}) for i, n in enumerate(sizes)])
    g, _ = build_population(table, initiators=(RootGrowthProbe,))
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return MPGDataStructure(g, from_scale=g.scales.SubOrgan)


class _PoolEquations:
    @node_balance(field="sugar")
    def _balance(self, sugar, shoot):
        P = self.pool_exchange("shoot")
        exchange = P @ shoot - np.asarray(P.sum(axis=1)).reshape(-1) * sugar
        return (sugar - self.previous("sugar")) / self.dt - EXCHANGE * exchange

    @pool_balance(field="shoot")
    def _shoot(self, sugar, shoot):
        P = self.pool_exchange("shoot")
        exchange = P @ shoot - np.asarray(P.sum(axis=1)).reshape(-1) * sugar
        return (shoot - self.previous("shoot")) / self.dt + EXCHANGE * np.asarray(P.T @ exchange).reshape(-1)


@dataclass
class EveryNodePool(FunctionalComponent):
    sugar: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)
    shoot: float = state_variable(**DOC, initialize=0., scale=scales.Plant)
    time_step = DT
    _solve = graph_system(node_unknowns=["sugar"], transient=True, pool_unknowns={"shoot": "Plant"}
                          )(type("_solve", (_PoolEquations,), {}))


def test_a_pool_given_by_its_location_exchanges_with_every_node_of_its_plant():
    ds = _population()
    model = EveryNodePool(data_structure=ds)
    ds.set("shoot", [6., 0.])
    model()
    n, owner = ds.n_nodes(), np.asarray(ds.owner("Plant"))
    P = np.zeros((n, 2))
    P[np.arange(n), owner] = 1.
    A = np.zeros((n + 2, n + 2))
    A[:n, :n] = np.eye(n) / DT + EXCHANGE * np.eye(n)
    A[:n, n:] = -EXCHANGE * P
    A[n:, :n] = -EXCHANGE * P.T
    A[n:, n:] = np.eye(2) / DT + EXCHANGE * np.diag(P.sum(axis=0))
    x = np.linalg.solve(A, np.r_[np.zeros(n), 6., 0.] / DT)
    np.testing.assert_allclose(ds.get("sugar"), x[:n], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(ds.get("shoot"), x[n:], rtol=1e-8)
    assert (ds.get("sugar")[owner == 0] > 0.).all() and (ds.get("sugar")[owner == 1] == 0.).all()


def test_pool_exchange_names_a_pool_of_the_solve():
    model = EveryNodePool(data_structure=_population())
    with pytest.raises(KeyError, match="'other' is not a pool unknown"):
        model.pool_exchange("other")


@dataclass
class UndeclaredExchange(FunctionalComponent):
    sugar: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)
    shoot: float = state_variable(**DOC, initialize=0., scale=scales.Plant)
    time_step = DT
    _solve = graph_system(node_unknowns=["sugar"], transient=True,
                          pool_unknowns={"shoot": {"location": "Plant", "exchange": "collar"}}
                          )(type("_solve", (_PoolEquations,), {}))


@dataclass
class PoolWithoutBalance(FunctionalComponent):
    sugar: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)
    shoot: float = state_variable(**DOC, initialize=0., scale=scales.Plant)
    time_step = DT

    @graph_system(node_unknowns=["sugar"], transient=True, pool_unknowns={"shoot": "Plant"})
    class _solve:
        @node_balance(field="sugar")
        def _balance(self, sugar):
            return sugar - self.previous("sugar")


def test_pool_declarations_are_checked():
    with pytest.raises(ValueError, match="needs a location"):
        graph_system(node_unknowns=["sugar"], pool_unknowns={"shoot": {"exchange": "collar"}})
    with pytest.raises(KeyError, match="which the graph system does not declare"):
        UndeclaredExchange(data_structure=_population())()
    with pytest.raises(ValueError, match="have no @pool_balance"):
        PoolWithoutBalance(data_structure=_population())()


# ---------------------------------------------------------------- forcings

@dataclass
class Supplied(FunctionalComponent):
    u: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)
    time_step = DT

    @graph_system(node_unknowns=["u"], transient=True)
    class _solve:
        @node_balance(field="u")
        def _balance(self, u):
            return (u - self.previous("u")) / self.dt - self.forcing("supply")


@pytest.mark.parametrize("supply", [lambda t: 3. * t,
                                    pd.Series([0., 30.], index=[0., 10.]),
                                    ([0., 10.], [0., 30.])],
                         ids=["callable", "series", "times_values"])
def test_forcings_given_as_callables_or_times_and_values(supply):
    ds = _ds()
    model = Supplied(data_structure=ds)
    model.forcings = {"supply": supply}
    model()
    np.testing.assert_allclose(ds.get("u"), DT * 3. * DT)                   # read at the end of the step, t = DT


def test_a_missing_forcing_raises():
    model = Supplied(data_structure=_ds())
    with pytest.raises(KeyError, match="has no forcing 'absent'"):
        model.forcing("absent")


# ---------------------------------------------------------------- @graph_output write-back

@dataclass
class WrongOutputLocation(FunctionalComponent):
    u: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)
    doubled: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)
    time_step = DT

    @graph_system(node_unknowns=["u"], transient=True)
    class _solve:
        _balance = node_balance(field="u")(_diffusion_residual)

        @graph_output("doubled", location="edge")
        def _doubled(self, u):
            return 2. * u


def test_an_output_location_must_be_the_registered_one():
    with pytest.raises(ValueError, match="is registered at node"):
        WrongOutputLocation(data_structure=_ds())()


@dataclass
class ActiveOutputs(FunctionalComponent):
    u: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)
    flux: float = state_variable(**DOC, initialize=0., location="edge")
    doubled: float = state_variable(**DOC, initialize=-1., scale=scales.SubOrgan)
    time_step = DT

    @graph_system(node_unknowns=["u"], edge_unknowns=["flux"], transient=True, where="active")
    class _solve:
        @node_balance(field="u")
        def _balance(self, u, flux):
            return (u - self.previous("u")) / self.dt + np.asarray(self._graph_view.incidence @ flux).reshape(-1)

        @edge_law(field="flux", integrate=True)
        def _fick(self, u, flux):
            return flux - K * np.asarray(self._graph_view.incidence.T @ u).reshape(-1)

        @graph_output("doubled")
        def _doubled(self, u):
            return 2. * u

        @graph_output("edge_marker", location="edge")
        def _edge_marker(self, flux):
            return np.ones_like(flux)


def _active_ds(dead):
    ds = _ds()
    ds.register("u", np.linspace(1., 2., ds.n_nodes()), location="node")
    alive = np.ones(ds.n_nodes())
    alive[dead] = 0.
    ds.register("alive", alive, location="node")
    ds.define_mask("active", {"alive": ">0"})
    return ds


def _dropped_edges(ds, dead):
    vids = ds.entity_ids("node")
    dead_vids = set(vids[np.atleast_1d(dead)].tolist())
    return np.array([a in dead_vids or b in dead_vids for a, b in ds.edges()])


def test_outputs_on_an_active_subgraph_leave_the_other_entities_alone():
    dead = np.array([5, 6])
    ds = _active_ds(dead)
    ActiveOutputs(data_structure=ds)()
    active = ds.get("alive") > 0
    np.testing.assert_allclose(ds.get("doubled")[active], 2. * ds.get("u")[active])
    np.testing.assert_array_equal(ds.get("doubled")[~active], -1.)          # inactive nodes keep their values
    dropped = _dropped_edges(ds, dead)
    assert dropped.any() and not dropped.all()
    np.testing.assert_array_equal(ds.get("edge_marker")[~dropped], 1.)
    np.testing.assert_array_equal(ds.get("edge_marker")[dropped], 0.)       # dropped edges get no value


def test_integrated_amounts_on_an_active_subgraph_are_kept_on_dropped_edges():
    ds = _active_ds(np.array([], dtype=np.int64))
    model = ActiveOutputs(data_structure=ds)
    model()
    first = ds.get("flux_amount").copy()
    np.testing.assert_allclose(first, ds.get("flux") * DT)
    dead = np.array([5, 6])
    alive = np.ones(ds.n_nodes())
    alive[dead] = 0.
    ds.set("alive", alive)
    model()
    dropped = _dropped_edges(ds, dead)
    np.testing.assert_array_equal(ds.get("flux_amount")[dropped], first[dropped])
    np.testing.assert_array_equal(ds.get("flux")[dropped], 0.)
    np.testing.assert_allclose(ds.get("flux_amount")[~dropped], first[~dropped] + ds.get("flux")[~dropped] * DT)


# ---------------------------------------------------------------- filtered node balances

@dataclass
class ZonedSource(FunctionalComponent):
    u: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)
    zone: float = parameter(**DOC, by="", default=0., location="node")
    source: float = parameter(**DOC, by="", default=0., location="node")
    time_step = DT

    @graph_system(node_unknowns=["u"], transient=True)
    class _solve:
        _balance = node_balance(field="u")(_diffusion_residual)

        @node_balance(field="u", filters={"zone": [1]})
        def _zone_source(self, u, source):
            return -source


@dataclass
class FilterOnMissingVariable(FunctionalComponent):
    u: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)
    time_step = DT

    @graph_system(node_unknowns=["u"], transient=True)
    class _solve:
        _balance = node_balance(field="u")(_diffusion_residual)

        @node_balance(field="u", filters={"nowhere": [1]})
        def _zone_source(self, u):
            return np.ones_like(u)


def test_a_filtered_balance_adds_its_terms_on_the_selected_nodes_only():
    ds = _ds()
    zone = (np.arange(ds.n_nodes()) % 2).astype(float)
    source = np.linspace(1., 2., ds.n_nodes())
    model = ZonedSource(data_structure=ds)
    ds.set("zone", zone)
    ds.set("source", source)
    model()
    A = np.eye(ds.n_nodes()) / DT + K * _laplacian(ds)
    np.testing.assert_allclose(ds.get("u"), np.linalg.solve(A, np.where(zone == 1, source, 0.)), rtol=1e-10)
    with pytest.raises(KeyError, match="'nowhere' is used by a graph system but is not registered"):
        FilterOnMissingVariable(data_structure=_ds())()
