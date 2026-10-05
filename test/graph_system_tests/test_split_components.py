"""
Graph systems solved per connected piece: split="components" solves
each plant of a population on its own, with its own Newton convergence and integration steps, as if it were alone.
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
from openalea.metafspm.solve.decorator import edge_law, graph_system, node_balance

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "structure_tests"))
from growth import DOC, RootGrowthProbe

DT = 10.


@pytest.fixture(autouse=True)
def _simulation_time_step():
    Choregrapher().add_simulation_time_step(DT)


class _Diffusion:
    @node_balance(field="sugar")
    def _balance(self, sugar, sugar_flux):
        return (sugar - self.previous("sugar")) / self.dt + np.asarray(self._graph_view.incidence @ sugar_flux).reshape(-1)

    @edge_law(field="sugar_flux")
    def _law(self, sugar, sugar_flux, conductance):
        return sugar_flux - conductance * np.asarray(self._graph_view.incidence.T @ sugar).reshape(-1)


def _fields():
    return dict(sugar=state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="intensive"),
                sugar_flux=state_variable(**DOC, initialize=0., scale=scales.SubOrgan, location="edge",
                                          mapping="child", state_variable_type="extensive"))


@dataclass
class WholeDiffusion(FunctionalComponent):
    sugar: float = _fields()["sugar"]
    sugar_flux: float = _fields()["sugar_flux"]
    conductance: float = parameter(**DOC, by="", default=0.05)
    time_step = DT
    _solve = graph_system(node_unknowns=["sugar"], edge_unknowns=["sugar_flux"], transient=True,
                          integrate="adaptive", rtol=1e-3, atol=1e-6)(type("_solve", (_Diffusion,), {}))


@dataclass
class SplitDiffusion(FunctionalComponent):
    sugar: float = _fields()["sugar"]
    sugar_flux: float = _fields()["sugar_flux"]
    conductance: float = parameter(**DOC, by="", default=0.05)
    time_step = DT
    _solve = graph_system(node_unknowns=["sugar"], edge_unknowns=["sugar_flux"], transient=True,
                          integrate="adaptive", rtol=1e-3, atol=1e-6, split="components")(type("_solve", (_Diffusion,), {}))


@dataclass
class ActiveSplitDiffusion(FunctionalComponent):
    sugar: float = _fields()["sugar"]
    sugar_flux: float = _fields()["sugar_flux"]
    conductance: float = parameter(**DOC, by="", default=0.05)
    time_step = DT
    _solve = graph_system(node_unknowns=["sugar"], edge_unknowns=["sugar_flux"], transient=True,
                          where="active", split="components")(type("_solve", (_Diffusion,), {}))


def _population(n_segments, conductances):
    table = pd.DataFrame([dict(plant=f"p{i}", model=None, x=0.1 * i, y=0., z=0., rotation=0.,
                               scenario={"parameters": {"n_segments": n}}) for i, n in enumerate(n_segments)])
    g, plants = build_population(table, initiators=(RootGrowthProbe,))
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    return ds, plants


def _initial_sugar(ds):
    """By rank within each plant (vids differ between a plant alone and in a population)."""
    owner, ids = ds.entity_ids("Plant")[ds.owner("Plant")], ds.entity_ids("node")
    rank = np.empty(ids.size, dtype=np.int64)
    for plant in np.unique(owner):
        members = np.flatnonzero(owner == plant)
        rank[members[np.argsort(ids[members])]] = np.arange(members.size)
    return np.where(rank % 3 == 0, 1., 0.) + 0.01 * (rank % 7)


def _run(model_class, n_segments, conductances, steps=3):
    ds, plants = _population(n_segments, conductances)
    model = model_class(data_structure=ds)
    ds.set("conductance", conductances)
    ds.set("sugar", _initial_sugar(ds))
    for _ in range(steps):
        model()
    return ds, plants, model


def _of_plant(ds, plant, name, location="node"):
    owner = ds.entity_ids("Plant")[ds.owner("Plant")]
    if location == "node":
        return ds.get(name)[owner == plant]
    child_owner = owner[ds.index_of(np.array([b for _, b in ds.edges()]))]
    return ds.get(name)[child_owner == plant]


def test_each_piece_is_solved_as_if_its_plant_were_alone():
    sizes, conductances = [4, 7, 5], [0.01, 0.5, 0.08]          # slow and fast plants: their own adaptive steps
    population, plants, model = _run(SplitDiffusion, sizes, conductances)
    assert len(model.__dict__["_pieces_cache"][1]) == 3
    for plant, size, k in zip(plants, sizes, conductances):
        alone, (alone_plant,), _ = _run(SplitDiffusion, [size], [k])
        np.testing.assert_allclose(_of_plant(population, plant, "sugar"), _of_plant(alone, alone_plant, "sugar"),
                                   rtol=0, atol=1e-15)
        np.testing.assert_allclose(_of_plant(population, plant, "sugar_flux", "edge"),
                                   _of_plant(alone, alone_plant, "sugar_flux", "edge"), rtol=0, atol=1e-15)


def test_split_and_whole_solves_agree():
    sizes, conductances = [4, 7, 5], [0.01, 0.5, 0.08]
    split, _, _ = _run(SplitDiffusion, sizes, conductances)
    whole, _, _ = _run(WholeDiffusion, sizes, conductances)
    np.testing.assert_allclose(split.get("sugar"), whole.get("sugar"), rtol=1e-2, atol=1e-5)   # within the adaptive tolerances
    assert split.get("sugar").sum() == pytest.approx(_initial_sugar(split).sum(), rel=1e-9)     # conserved


def test_pieces_of_an_active_subgraph():
    ds, plants = _population([4, 6], [0.1, 0.1])
    model = ActiveSplitDiffusion(data_structure=ds)
    owner = ds.entity_ids("Plant")[ds.owner("Plant")]
    ds.register("growing", (owner == plants[1]).astype(float) * (ds.entity_ids("node") % 2 + 1.), location="node")
    ds.define_mask("active", {"growing": ">0"})
    ds.set("sugar", _initial_sugar(ds))
    before = ds.get("sugar").copy()
    model()
    np.testing.assert_array_equal(ds.get("sugar")[owner == plants[0]], before[owner == plants[0]])   # frozen
    assert not np.array_equal(ds.get("sugar")[owner == plants[1]], before[owner == plants[1]])
    assert len(model.__dict__["_pieces_cache"][1]) == 1


def test_split_is_checked():
    with pytest.raises(ValueError, match="split must be"):
        graph_system(node_unknowns=["sugar"], split="plants")
