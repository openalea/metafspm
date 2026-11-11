"""
Compute(linear_solver=) per system: a name (the same backend everywhere, unchanged) or a mapping {key: backend},
resolved for each graph system by its qualified name "Component._system" (its class's, then a base's), else a
property it declares (symmetric=True), else "default"; checked when the Compute is made (its backends, its keys) and
when the Scene is built (the packages, each qualified name one of the scene's systems), naming the key. The static
condensation's and the Kron reduction's factorisations and solves go through the backend resolved for their system
(a spy backend registered here records them).
"""
import sys
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

from openalea.metafspm.compute import Compute
from openalea.metafspm.coupling.component import FunctionalComponent, state_variable
from openalea.metafspm.scene.scene import Scene
from openalea.metafspm.solve import linear
from openalea.metafspm.solve.decorator import graph_jacobian, graph_system, linear_system, node_balance
from openalea.metafspm.solve.linear import LINEAR_SOLVERS, LinearSystem, SuperLU, resolve
from openalea.metafspm.solve.network import condensed_solve, kron_reduction

from growth import DOC
from scene_doubles import DT, RootPopulation, SceneSoil, Seedlings, doubles_planting, soil_translator


class Spy(SuperLU):
    """SuperLU (the same bits), recording each call: (what, symmetric)."""
    name = "spy"
    calls = []

    def solve(self, matrix, rhs, key=None, symmetric: bool = False):
        type(self).calls.append(("solve", symmetric))
        return super().solve(matrix, rhs, key=key, symmetric=symmetric)

    def solve_once(self, matrix, rhs, symmetric: bool = False):
        type(self).calls.append(("solve_once", symmetric))
        return super().solve_once(matrix, rhs, symmetric)

    def factorize(self, matrix, symmetric: bool = False):
        type(self).calls.append(("factorize", symmetric))
        return super().factorize(matrix, symmetric)


@pytest.fixture
def spy(monkeypatch):
    monkeypatch.setitem(LINEAR_SOLVERS, "spy", Spy)
    Spy.calls = []
    return Spy


# ── the resolution ──

PLANT = LinearSystem(key="plant", names=("MyHydraulics._plant", "Hydraulics._plant"))
STANDARD = LinearSystem(key="standard", names=("Hydraulics._standard",), properties=frozenset({"symmetric"}))
SOIL = LinearSystem(key="soil", names=("Soil._flow",))


def test_a_name_is_every_systems_backend():
    compute = Compute(linear_solver="superlu_reuse")
    assert compute.linear_solver == "superlu_reuse"
    for system in (None, PLANT, STANDARD, SOIL):
        assert resolve(compute.linear_solver, system) == "superlu_reuse"
        assert compute.linear(system).name == "superlu_reuse"
    assert compute.linear(PLANT) is compute.linear(SOIL)              # one instance per backend, its caches kept
    assert Compute().linear_solver == "superlu"


def test_each_level_of_a_mapping_resolves():
    setting = {"default": "superlu_reuse", "symmetric": "spy", "Hydraulics._plant": "superlu"}
    assert resolve(setting, PLANT) == "superlu"             # its qualified name (a base class's)
    assert resolve(setting, STANDARD) == "spy"              # its declared property
    assert resolve(setting, SOIL) == "superlu_reuse"        # the default
    assert resolve(setting, None) == "superlu_reuse"
    assert resolve({**setting, "Hydraulics._standard": "superlu"}, STANDARD) == "superlu"     # the name first
    assert resolve({**setting, "MyHydraulics._plant": "spy"}, PLANT) == "spy"             # its own class first


def test_a_mappings_default_is_the_devices(spy):
    assert Compute(linear_solver={"symmetric": "spy"}).linear_solver == {"symmetric": "spy", "default": "superlu"}


@pytest.mark.parametrize("setting, message", [
    ({"default": "umfpack"}, "'default'.*umfpack"),
    ({"symmetric": "cupy"}, "'symmetric'.*GPU"),
    ({"diagonal": "superlu"}, "'diagonal'"),
    ({"Hydraulics": "superlu"}, "'Hydraulics'"),
    ({"Hydraulics._plant": 3}, "'Hydraulics._plant'"),
])
def test_a_mappings_mistake_raises_naming_its_key(setting, message):
    with pytest.raises((ValueError, TypeError), match=message):
        Compute(linear_solver=setting)


# ── at the Scene's build ──

@dataclass
class GridWater(FunctionalComponent):
    """A symmetric graph system on the soil's grid: (I + L) w = 1."""
    water: float = state_variable(**DOC, default=0., location="cell")
    time_step = DT

    @graph_system(node_unknowns=["water"], solver="newton", symmetric=True)
    class _water:
        @node_balance(field="water")
        def _balance(self, water):
            B = self.graph_view.incidence
            return water + np.asarray(B @ np.asarray(B.T @ water).reshape(-1)).reshape(-1) - 1.

        @graph_jacobian
        def _jacobian(self):
            B = self.graph_view.incidence
            return (diags(np.ones(B.shape[0])) + B @ B.T).tocsr()


class WaterSoil(SceneSoil):
    def __init__(self, plant_data_structures, x_range, y_range, time_step):
        super().__init__(plant_data_structures, x_range, y_range, time_step)
        self.water = GridWater(data_structure=self.grid)
        self.components = [self.nitrate, self.water]


def _scene(linear_solver):
    return Scene(doubles_planting([RootPopulation, Seedlings]), environment=[WaterSoil],
                 translator=soil_translator("SceneExudation", "SeedlingExudation"), time_step=DT,
                 compute=Compute(threads=1, linear_solver=linear_solver))


def test_a_scenes_systems_get_their_backends(spy):
    scene = _scene({"GridWater._water": "spy"})
    scene.run()
    assert ("solve", True) in Spy.calls
    assert np.allclose(scene.environment[0].grid.get("water"), scene.environment[0].grid.get("water").mean())
    Spy.calls = []
    _scene({"symmetric": "spy"}).run()
    assert ("solve", True) in Spy.calls
    Spy.calls = []
    _scene({"default": "spy", "GridWater._water": "superlu"}).run()
    assert not Spy.calls


def test_a_qualified_name_that_is_none_of_the_scenes_systems_raises_at_its_build():
    with pytest.raises(ValueError, match=r"'GridWater._flow'.*GridWater._water"):
        _scene({"GridWater._flow": "superlu"})


def test_cholmod_for_a_system_not_declared_symmetric_raises_at_the_build(monkeypatch):
    import types
    module, cholmod = types.ModuleType("sksparse"), types.ModuleType("sksparse.cholmod")   # a stand-in: present
    module.cholmod = cholmod
    monkeypatch.setitem(sys.modules, "sksparse", module)
    monkeypatch.setitem(sys.modules, "sksparse.cholmod", cholmod)
    monkeypatch.setattr(GridWater, "_graph_system_specs",
                        {"_water": {**GridWater._graph_system_specs["_water"], "symmetric": False}})
    with pytest.raises(ValueError, match=r"'GridWater._water'.*symmetric"):
        _scene({"GridWater._water": "cholmod"})


def test_a_missing_optional_package_raises_at_the_build_naming_its_key(monkeypatch):
    monkeypatch.setitem(sys.modules, "sksparse", None)
    monkeypatch.setitem(sys.modules, "sksparse.cholmod", None)
    with pytest.raises(ImportError, match=r"'symmetric'.*scikit-sparse>=0.5"):
        _scene({"symmetric": "cholmod"})


def test_a_components_systems_are_named_by_its_class_and_bases():
    @dataclass
    class MoreWater(GridWater):
        pass
    system = linear_system(MoreWater(data_structure=WaterSoil([], 0.8, 0.4, DT).grid), "_water")
    assert system.names == ("MoreWater._water", "GridWater._water") and system.symmetric


# ── the static condensation and the Kron reduction ──

def _network(n=30, seed=0):
    rng = np.random.default_rng(seed)
    a = np.r_[np.arange(n - 1), rng.integers(0, n, n)]
    b = np.r_[np.arange(1, n), rng.integers(0, n, n)]
    keep = a != b
    adjacency = csr_matrix((rng.uniform(0.5, 2., keep.sum()), (a[keep], b[keep])), shape=(n, n))
    adjacency = adjacency + adjacency.T
    return (diags(np.asarray(adjacency.sum(axis=1)).reshape(-1) + rng.uniform(0.1, 1., n)) - adjacency).tocsr()


def test_a_kron_reduction_factorises_with_its_systems_backend(spy):
    A = _network()
    kept = np.arange(A.shape[0]) % 5 == 0
    with Compute(linear_solver={"Hydraulics._standard": "spy"}).limits():
        expected = kron_reduction(A, kept)                      # no system: "default", SuperLU
        assert not Spy.calls
        reduced = kron_reduction(A, kept, STANDARD)
        recovered = reduced.recover(np.ones(kept.sum()))
    assert Spy.calls == [("factorize", True)]
    np.testing.assert_array_equal(reduced.reduced, expected.reduced)          # the same bits as the default
    np.testing.assert_array_equal(recovered, expected.recover(np.ones(kept.sum())))


def test_a_static_condensation_solves_with_its_systems_backend(spy):
    from test_condensed_solve import _plant
    A, groups, kept = _plant(np.random.default_rng(0), factors=[1., 1., 2.5, 1.])
    b = np.ones(A.shape[0])
    with Compute(linear_solver={"symmetric": "spy"}).limits():
        expected = condensed_solve(A, b, groups, kept)
        assert not Spy.calls
        x = condensed_solve(A, b, groups, kept, system=STANDARD)
    assert {call for call in Spy.calls} == {("factorize", True), ("solve_once", True)}
    np.testing.assert_array_equal(x, expected)


def test_a_reductions_kinds_factorise_with_the_reduced_systems_backend(spy):
    """The kinds of a Reduction (test_reduction's toy sections) through the backend mapped to the reduced system: its
    Kron reductions' factorisations and its Newton solves; the same potentials, bit for bit, as the default's."""
    from test_reduction import _solved
    with Compute(threads=1).limits():
        expected = _solved(True).data_structure.get("water_potential")
    with Compute(threads=1, linear_solver={"NodeHydraulics._hydraulics": "spy"}).limits():
        found = _solved(True).data_structure.get("water_potential")
    assert ("factorize", False) in Spy.calls and ("solve", False) in Spy.calls
    np.testing.assert_array_equal(found, expected)


def test_the_systems_linear_system_reaches_its_solves(spy):
    """A Newton step's solve receives its system's declaration (symmetric) through the mapping's backend."""
    grid = WaterSoil([], 0.8, 0.4, DT).grid
    component = GridWater(data_structure=grid)
    with Compute(linear_solver={"GridWater._water": "spy"}).limits():
        component()
    assert Spy.calls and all(call == ("solve", True) for call in Spy.calls)
    assert linear.SYSTEM_PROPERTIES == ("symmetric",)
