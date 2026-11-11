"""
Graph-system extensions: pool unknowns at a coarse scale solved with the graph (a shoot
pool per plant exchanging with its collar), boundary conditions chosen per node at each solve, and forcings read at
the end of each (sub-)step.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene.population import _build_population
from openalea.metafspm.solve.decorator import boundary_set, graph_system, node_balance, pool_balance
from growth import DOC, RootGrowthProbe

DT, K, EXCHANGE = 2., 0.3, 0.8


def _population(sizes=(3, 5)):
    table = pd.DataFrame([dict(plant=f"p{i}", model=None, x=0.1 * i, y=0., depth=0., rotation=0.,
                               scenario={"n_segments": n}) for i, n in enumerate(sizes)])
    mtg, plants = _build_population(table, initiators=(RootGrowthProbe,))
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
    data_structure.register("is_collar", (data_structure.parents() < 0).astype(float), location="node")
    return data_structure, plants


class _Equations:
    collar = boundary_set(where={"is_collar": ">0"})                     # a selection: the nodes exchanging with the pool

    @node_balance(field="sugar")
    def _balance(self, sugar, shoot_sugar):
        B, P = self.graph_view.incidence, self.pool_exchange("shoot_sugar")
        exchange = P @ shoot_sugar - np.asarray(P.sum(axis=1)).reshape(-1) * sugar
        return (sugar - self.previous("sugar")) / self.step_duration + K * np.asarray(B @ (B.T @ sugar)).reshape(-1) \
            - EXCHANGE * exchange

    @pool_balance(field="shoot_sugar")
    def _shoot(self, sugar, shoot_sugar):
        P = self.pool_exchange("shoot_sugar")
        exchange = P @ shoot_sugar - np.asarray(P.sum(axis=1)).reshape(-1) * sugar
        return (shoot_sugar - self.previous("shoot_sugar")) / self.step_duration + EXCHANGE * np.asarray(P.T @ exchange).reshape(-1)


def _fields():
    return dict(sugar=state_variable(**DOC, default=0., scale=scales.SubOrgan),
                shoot_sugar=state_variable(**DOC, default=0., scale=scales.Plant))


@dataclass
class RootWithShootPool(FunctionalComponent):
    sugar: float = _fields()["sugar"]
    shoot_sugar: float = _fields()["shoot_sugar"]
    time_step = DT
    _solve = graph_system(node_unknowns=["sugar"], transient=True,
                          pool_unknowns={"shoot_sugar": {"location": "Plant", "exchange": "collar"}}
                          )(type("_solve", (_Equations,), {}))


@dataclass
class SplitRootWithShootPool(FunctionalComponent):
    sugar: float = _fields()["sugar"]
    shoot_sugar: float = _fields()["shoot_sugar"]
    time_step = DT
    _solve = graph_system(node_unknowns=["sugar"], transient=True, split="components",
                          pool_unknowns={"shoot_sugar": {"location": "Plant", "exchange": "collar"}}
                          )(type("_solve", (_Equations,), {}))


def _start(model_class):
    data_structure, plants = _population()
    model = model_class(data_structure=data_structure)
    data_structure.set("sugar", np.linspace(0., 1., data_structure.n_nodes()))
    data_structure.set("shoot_sugar", [10., 0.])
    return data_structure, plants, model


def _reference(data_structure, sugar0, shoot0):
    """The same implicit step as one hand-assembled linear system."""
    B = data_structure._incidence_matrix().toarray()
    n = data_structure.n_nodes()
    owner = np.asarray(data_structure.owner("Plant"))
    collar = np.flatnonzero(data_structure.parents() < 0)
    P = np.zeros((n, 2))
    P[collar, owner[collar]] = 1.
    d = P.sum(axis=1)
    A = np.zeros((n + 2, n + 2))
    A[:n, :n] = np.eye(n) / DT + K * B @ B.T + EXCHANGE * np.diag(d)
    A[:n, n:] = -EXCHANGE * P
    A[n:, :n] = -EXCHANGE * P.T
    A[n:, n:] = np.eye(2) / DT + EXCHANGE * np.diag(P.sum(axis=0))
    x = np.linalg.solve(A, np.r_[sugar0, shoot0] / DT)
    return x[:n], x[n:]


def test_a_pool_per_plant_is_solved_with_the_graph():
    data_structure, plants, model = _start(RootWithShootPool)
    sugar0, shoot0 = data_structure.get("sugar").copy(), data_structure.get("shoot_sugar").copy()
    model()
    sugar, shoot = _reference(data_structure, sugar0, shoot0)
    np.testing.assert_allclose(data_structure.get("sugar"), sugar, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(data_structure.get("shoot_sugar"), shoot, rtol=1e-8, atol=1e-10)
    assert data_structure.get("sugar").sum() + data_structure.get("shoot_sugar").sum() == pytest.approx(sugar0.sum() + shoot0.sum())
    owner = np.asarray(data_structure.owner("Plant"))
    assert data_structure.get("sugar")[owner == 0].sum() > sugar0[owner == 0].sum()        # fed by its own shoot only
    assert data_structure.get("sugar")[owner == 1].sum() < sugar0[owner == 1].sum()


def test_pools_of_split_solves_agree_with_the_whole_solve():
    whole, _, model = _start(RootWithShootPool)
    split, _, split_model = _start(SplitRootWithShootPool)
    model()
    split_model()
    np.testing.assert_allclose(split.get("shoot_sugar"), whole.get("shoot_sugar"), rtol=1e-8)
    np.testing.assert_allclose(split.get("sugar"), whole.get("sugar"), rtol=1e-8, atol=1e-12)


@dataclass
class SwitchingCollar(FunctionalComponent):
    sugar: float = state_variable(**DOC, default=1., scale=scales.SubOrgan)
    collar_condition: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, dtype="int")
    collar_value: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)
    time_step = DT

    @graph_system(node_unknowns=["sugar"], transient=True)
    class _solve:
        collar = boundary_set(where={"is_collar": ">0"}, condition_variable="collar_condition", value="collar_value")

        @node_balance(field="sugar")
        def _balance(self, sugar):
            B = self.graph_view.incidence
            return (sugar - self.previous("sugar")) / self.step_duration + K * np.asarray(B @ (B.T @ sugar)).reshape(-1)


def test_boundary_conditions_are_read_per_node_at_each_solve():
    data_structure, plants = _population()
    model = SwitchingCollar(data_structure=data_structure)
    collar = np.flatnonzero(data_structure.parents() < 0)
    owner = np.asarray(data_structure.owner("Plant"))
    conditions, values = np.zeros(data_structure.n_nodes(), dtype=np.int64), np.zeros(data_structure.n_nodes())
    conditions[collar[owner[collar] == 0]], values[collar[owner[collar] == 0]] = 1, 5.        # Dirichlet: a pressure
    conditions[collar[owner[collar] == 1]], values[collar[owner[collar] == 1]] = 2, 0.25      # Neumann: an inflow
    data_structure.set("collar_condition", conditions)
    data_structure.set("collar_value", values)
    total = data_structure.get("sugar")[owner == 1].sum()
    model()
    np.testing.assert_allclose(data_structure.get("sugar")[collar[owner[collar] == 0]], 5.)
    assert data_structure.get("sugar")[owner == 1].sum() == pytest.approx(total + 0.25 * DT)        # the inflow, integrated
    data_structure.set("collar_condition", 0)                                                           # no condition any more
    total = data_structure.get("sugar").sum()
    model()
    assert data_structure.get("sugar").sum() == pytest.approx(total)


@dataclass
class Forced(FunctionalComponent):
    sugar: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)
    time_step = DT
    seen = None

    @graph_system(node_unknowns=["sugar"], transient=True, integrate="substeps", substeps=4)
    class _solve:
        @node_balance(field="sugar")
        def _balance(self, sugar):
            supply = self.forcing("supply")
            type(self).seen.add((self.forcing_time(), supply, self.forcing("supply", over="step")))
            return (sugar - self.previous("sugar")) / self.step_duration - supply


def test_forcings_are_their_mean_over_each_substep():
    data_structure, _ = _population()
    model = Forced(data_structure=data_structure)
    model.set_forcings({"supply": pd.Series([0., 100.], index=[0., 100.])})                      # supply(t) = t
    Forced.seen = set()
    model()
    model()
    times = sorted(t for t, _, _ in Forced.seen)
    np.testing.assert_allclose(times, DT * np.arange(1, 9) / 4.)                  # two calls of 4 sub-steps
    assert all(supply == pytest.approx(t - DT / 8.) for t, supply, _ in Forced.seen)   # the mean over [t - h, t]
    assert all(whole == pytest.approx(DT / 2. if t <= DT else 1.5 * DT) for t, _, whole in Forced.seen)  # over="step"
    np.testing.assert_allclose(data_structure.get("sugar"), (2. * DT) ** 2 / 2.)            # the integral of t, exactly


def test_pools_need_a_newton_solver():
    with pytest.raises(ValueError, match="need a Newton solver"):
        graph_system(node_unknowns=["sugar"], solver="explicit_euler", pool_unknowns={"shoot": "Plant"})
