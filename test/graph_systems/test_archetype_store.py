"""
solve.archetype_store: the archetypes a run's processes share (Compute(processes=n)): a kind evaluated by one process
(or population) written once to the store's folder and read by every other one needing it, the same bits as its
evaluation in place; within a process one object per kind, never released in place while another population may
hold it.
"""
import os
import time

import numpy as np
import pytest

from openalea.metafspm.compute import Compute
from openalea.metafspm.scene.population import Planting, planting, population_data_structure
from openalea.metafspm.solve import archetype_store
from openalea.metafspm.solve.reduction import Reduction, _difference

from test_reduction import _model, _soil_walls, _xylem

workers_processes = pytest.mark.skipif(not hasattr(os, "fork"),
                                       reason="the store's files and claims serve the forked workers (POSIX only)")


@pytest.fixture
def store(tmp_path):
    with archetype_store.using(archetype_store.ArchetypeStore(str(tmp_path / "archetypes"))) as shared:
        yield shared


def _kinded_model():
    class KindedPlant(_model(False)):
        reduction = Reduction(location="SubOrgan", where={"label": "RootSegment"}, ports=_xylem, keep=_soil_walls,
                              archetype_variables={"soil_water_potential": None}, systems=("NodeHydraulics._hydraulics",))
    return KindedPlant


def _solved(model, soil_water_potential=-0.2):
    population = population_data_structure(planting([Planting()], model, {}), model)
    data_structure = population.data_structure
    data_structure.set("soil_water_potential", np.full(data_structure.n_nodes(), soil_water_potential))
    population.instance.run()
    return population


def _archetypes(population) -> dict:
    return population.data_structure.structure.reduction._archetypes


def test_a_kind_is_evaluated_once_and_read_by_another_population_with_the_same_bits(store):
    model = _kinded_model()
    with archetype_store.using(None):
        alone = _solved(model)                                    # without the store: the reference
    with Compute(checks={"archetypes"}).limits():                 # every kind compared with its evaluation alone
        first = _solved(model)
        second = _solved(model)
    assert store.written == 1 and store.loaded == 0 and len(os.listdir(store.folder)) == 1
    (key, kind), = _archetypes(second).items()
    assert all(archetype is _archetypes(first)[key][system] for system, archetype in kind.items())   # one object
    for system, archetype in kind.items():
        assert _difference(archetype, _archetypes(alone)[key][system]) is None
    np.testing.assert_array_equal(second.data_structure.get("water_potential"),
                                  alone.data_structure.get("water_potential"))


@workers_processes
def test_a_kind_written_by_another_process_is_read_from_its_file(store):
    model = _kinded_model()
    pid = os.fork()
    if pid == 0:                                                 # another process of the run evaluates the kind
        status = 1
        try:
            _solved(model)
            status = 0 if store.written == 1 else 2
        finally:
            os._exit(status)
    _, status = os.waitpid(pid, 0)
    assert os.WEXITSTATUS(status) == 0
    population = _solved(model)
    assert store.loaded == 1 and store.written == 0
    with archetype_store.using(None):
        alone = _solved(model)
    (key, kind), = _archetypes(population).items()
    for system, archetype in kind.items():
        assert _difference(archetype, _archetypes(alone)[key][system]) is None


def test_a_shared_kind_released_by_one_population_stays_whole_for_another(store):
    model = _kinded_model()
    first, second = _solved(model), _solved(model)
    shared = [archetype for kind in _archetypes(second).values() for archetype in kind.values()]
    data_structure = first.data_structure
    data_structure.set("soil_water_potential", np.full(data_structure.n_nodes(), -0.3))
    first.instance.run()                                        # its former kind let go, not released in place
    assert all(archetype_store.is_shared(archetype) and archetype.D is not None for archetype in shared)
    assert store.written == 2


def test_in_memory_the_populations_of_one_process_share_their_kinds():
    model = _kinded_model()
    with archetype_store.using(archetype_store.ArchetypeStore()) as store:
        first, second = _solved(model), _solved(model)
    assert store.folder is None and store.written == 0 and store.loaded == 0
    (key, kind), = _archetypes(second).items()
    assert all(archetype is _archetypes(first)[key][system] for system, archetype in kind.items())


def test_two_templates_of_one_key_but_other_vertices_are_two_kinds(store):
    model = _kinded_model()
    population = _solved(model)
    reduction = population.data_structure.structure.reduction
    archetype_key = next(iter(reduction._archetypes))
    name = store.name(reduction, archetype_key)
    other = reduction.for_population(reduction.mtg)
    full = reduction._full[archetype_key[0]]
    changed = type(full)(key=full.key, scale=full.scale, complex=full.complex, parent=full.parent,
                         properties={**full.properties, "K": (np.zeros(1, dtype=np.int64), np.ones(1))},
                         links=full.links)
    other._full, other._reduced = {archetype_key[0]: changed}, {full.key: reduction._reduced[full.key]}
    assert store.name(other, archetype_key) != name


@workers_processes
def test_a_kind_claimed_by_another_process_is_awaited_not_evaluated_twice(store):
    model = _kinded_model()
    population = _solved(model)                                  # the kind, to be written by a "claimer"
    reduction = population.data_structure.structure.reduction
    (archetype_key, kind), = reduction._archetypes.items()
    name = store.name(reduction, archetype_key)
    other = archetype_store.ArchetypeStore(os.path.join(os.path.dirname(store.folder), "other"))
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:                                                 # claims the kind, then writes it
        try:
            assert other.claim(name)
            os.write(write, b"x")
            time.sleep(0.2)
            other.write(name, kind)
        finally:
            os._exit(0)
    os.read(read, 1)
    waiting = archetype_store.ArchetypeStore(other.folder)
    assert not waiting.claim(name)
    awaited = waiting.wait(name)
    os.waitpid(pid, 0)
    assert waiting.loaded == 1 and all(_difference(awaited[system], kind[system]) is None for system in kind)
    assert not os.path.exists(os.path.join(other.folder, name + ".claim"))


@workers_processes
def test_a_kind_whose_claimer_ended_without_it_is_left_to_the_waiter(tmp_path):
    import subprocess
    ended = subprocess.Popen(["true"])
    ended.wait()
    store = archetype_store.ArchetypeStore(str(tmp_path))
    with open(tmp_path / "kind.claim", "w") as file:
        file.write(str(ended.pid))
    assert not store.claim("kind") and store.wait("kind") is None
