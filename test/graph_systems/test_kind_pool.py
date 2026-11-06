"""
A reduction's kinds evaluated in the Compute's pool (compute.WorkerPool, Compute.workers): the same kinds as in the
process, bit for bit, whatever the number of processes, forked (Linux) or spawned (macOS, Windows: tested here
through the doubles, on any platform); the pool kept between batches, each process sent only what it lacks (the
templates added since), closed by Compute.close (Scene.finish); a failure in a process raised in the main one; and
"auto" starts no pool for a small batch. The toy: reduced_sections' sections, kinds by length.
"""
import numpy as np
import pytest

from openalea.metafspm import compute as compute_module
from openalea.metafspm.compute import Compute, WorkerError, WorkerPool
from openalea.metafspm.solve import reduction as reduction_module

from reduced_sections import N_SEGMENTS, SYSTEM, ClassLengthPlant, ExactLengthPlant, length_population

LENGTHS = [0.3 + 0.37 * i for i in range(N_SEGMENTS)]


def _kinds(population) -> dict:
    """{group: the plant system's Kind} of every reduced group (the missing evaluated)."""
    ds = population.data_structure
    reduction = ds.structure.reduction
    return {i: by_system[SYSTEM] for i, by_system in
            reduction.group_kinds(ds, reduction.reduced_groups(ds), release=True).items()}


def _assert_same(kinds, reference) -> None:
    assert kinds.keys() == reference.keys()
    for i in reference:
        for name in ("D", "e", "interior_source", "kept_rows", "interior_rows"):
            np.testing.assert_array_equal(getattr(kinds[i], name), getattr(reference[i], name))
        assert kinds[i].key == reference[i].key


def _pool_kinds(compute, model=ExactLengthPlant, lengths=LENGTHS, start_method=None):
    population = length_population(model, lengths)
    if start_method is not None:
        compute.pool(1, start_method=start_method)
    with compute.limits():
        return population, _kinds(population)


@pytest.fixture
def reference():
    with Compute(workers=1).limits():
        return _kinds(length_population(ExactLengthPlant, LENGTHS))


@pytest.mark.parametrize("workers", [2, 3])
def test_the_pool_gives_the_kinds_of_the_process_bit_for_bit(reference, workers):
    compute = Compute(workers=workers)
    try:
        population, kinds = _pool_kinds(compute)
        assert compute.pool_info() == {"workers": workers, "start_method": compute_module.START_METHOD}
        reduction = population.data_structure.structure.reduction
        assert not reduction._libraries                    # evaluated in the processes' libraries, not here
        _assert_same(kinds, reference)
    finally:
        compute.close()
    assert compute.pool_info() is None


def test_a_spawned_pool_gives_the_same_kinds(reference):
    """Spawned processes (macOS, Windows) receive the reduction whole at their first task (its declaration, plant
    model by name, templates): the same kinds."""
    compute = Compute(workers=2)
    try:
        _, kinds = _pool_kinds(compute, start_method="spawn")
        assert compute.pool_info()["start_method"] == "spawn"
        _assert_same(kinds, reference)
    finally:
        compute.close()


@pytest.mark.parametrize("start_method", [compute_module.START_METHOD, "spawn"])
def test_the_pool_is_kept_and_each_process_sent_only_what_it_lacks(start_method):
    """A second batch runs in the same processes, which received the templates once; the class kinds of growing
    sections (a blend of two each) are those of the process."""
    compute = Compute(workers=2)
    try:
        population, _ = _pool_kinds(compute, model=ClassLengthPlant, start_method=start_method)
        processes = [worker.process.pid for worker in compute._pool.workers]
        reduction = population.data_structure.structure.reduction
        token = reduction.__dict__["_token"]
        sent = [worker.known[("reduction", token)] for worker in compute._pool.workers]
        ds = population.data_structure
        ds.set("length", np.asarray(ds.get("length")) * 1.7)
        with compute.limits():
            grown = _kinds(population)
        assert [worker.process.pid for worker in compute._pool.workers] == processes
        sync = reduction_module._sync_for(reduction, compute._pool.workers[0])
        assert sync["declaration"] is None and not any(sync["tables"].values())     # nothing new to send
        assert all(known["_full"] for known in sent)
    finally:
        compute.close()
    with Compute(workers=1).limits():
        alone = length_population(ClassLengthPlant, LENGTHS)
        _kinds(alone)
        ds = alone.data_structure
        ds.set("length", np.asarray(ds.get("length")) * 1.7)
        expected = _kinds(alone)
    for i in expected:
        np.testing.assert_array_equal(grown[i].D, expected[i].D)
        np.testing.assert_array_equal(grown[i].e, expected[i].e)


def test_auto_starts_no_pool_for_a_small_batch(reference):
    compute = Compute()                           # "auto": a few toy kinds are far below a second of work
    _, kinds = _pool_kinds(compute)
    assert compute.pool_info() is None
    _assert_same(kinds, reference)


def test_a_failure_in_a_process_is_raised_here():
    pool = WorkerPool(Compute(workers=2))
    try:
        pool.grow(2)
        assert pool.run([(int, ("7",)), (int, ("8",))]) == [7, 8]
        with pytest.raises(WorkerError, match="invalid literal"):
            pool.run([(int, ("7",)), (int, ("not a number",))])
        assert pool.closed                        # a failed run closes the pool
    finally:
        pool.close()


def test_a_scene_finish_closes_its_pool(tmp_path):
    """run.json records the pool used; Scene.finish stops its processes."""
    import json
    from scene_doubles import DT, RootPopulation, SceneSoil, Seedlings, doubles_planting, soil_translator
    from openalea.metafspm.scene.recorder import SceneRecorder
    from openalea.metafspm.scene.scene import Scene
    compute = Compute(workers=2)
    scene = Scene(doubles_planting([RootPopulation, Seedlings]), environment=[SceneSoil],
                  translator=soil_translator("SceneExudation", "SeedlingExudation"), time_step=DT, compute=compute,
                  recorder=SceneRecorder(str(tmp_path / "run")))
    pool = compute.pool(2)
    processes = [worker.process for worker in pool.workers]
    assert all(process.is_alive() for process in processes)
    scene.run()
    scene.finish()
    assert pool.closed and compute.pool_info() is None
    assert not any(process.is_alive() for process in processes)
    with open(tmp_path / "run" / "run.json") as file:
        recorded = json.load(file)["compute"]["pool"]
    assert recorded == {"workers": 2, "start_method": compute_module.START_METHOD}
