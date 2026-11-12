"""
The module-level state shared by a process's threads, made per context (docs/conventions.md, "Threads and forks"):
two populations' steps run in two threads at once give the results of the sequential steps, bit for bit (a reduction's
archetype extraction in one thread no longer strips the other's boundary sets: "piece has no Dirichlet anchor");
Compute.current(), the extraction flag and the skipped systems are each thread's own; the shared positions' cache hits
only its own arrays; a forked child starts with no timings, no frame writer and numba's default threads.
"""
import os
import sys
import threading

import numpy as np
import pytest

from openalea.metafspm import compute as compute_module
from openalea.metafspm.compute import Compute, fork_safe_threading
from openalea.metafspm.data_structure import columns
from openalea.metafspm.solve import decorator
from openalea.metafspm.solve.reduction import _extracting

from reduced_sections import N_SEGMENTS, ExactLengthPlant, _length_plant, length_population

WholePlant = _length_plant(False)       # the whole network: a steady system anchored at the collar (Dirichlet)
LENGTHS = [0.3 + 0.37 * i for i in range(N_SEGMENTS)]
ROUNDS = 60


def _steps(model, lengths, barrier=None) -> list:
    """ROUNDS fresh plants of *model* (sections of *lengths*), each stepped once: their water potentials."""
    if barrier is not None:
        barrier.wait()
    solved = []
    with Compute(threads=2).limits():
        for _ in range(ROUNDS):
            population = length_population(model, lengths)
            population.instance.run()
            solved.append(np.asarray(population.data_structure.get("water_potential")).copy())
    return solved


@pytest.fixture
def switching():
    """The interpreter's thread switches made frequent, for the two populations' steps to interleave."""
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(interval)


def test_two_populations_steps_in_two_threads_equal_the_sequential_steps(switching):
    """A reduced population (its archetypes extracted at each fresh plant's step) and a whole one (a steady system
    anchored by its collar's Dirichlet set) stepped in two threads at once: no "piece has no Dirichlet anchor", the
    same bits as the sequential steps."""
    work = ((ExactLengthPlant, LENGTHS), (WholePlant, LENGTHS[::-1]))
    expected = [_steps(model, lengths) for model, lengths in work]
    barrier = threading.Barrier(2)
    found, errors = [None, None], []

    def run(k):
        try:
            found[k] = _steps(*work[k], barrier)
        except BaseException as error:              # raised again in the test's thread
            errors.append(error)
    threads = [threading.Thread(target=run, args=(k,)) for k in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    for solved, reference in zip(found, expected):
        for one, other in zip(solved, reference):
            np.testing.assert_array_equal(one, other)


def test_the_extraction_flag_the_skipped_systems_and_the_compute_are_each_threads_own():
    seen = {}
    inside, done = threading.Event(), threading.Event()

    def other():
        inside.wait()
        seen.update(extracting=decorator._EXTRACTING.get(), skipped=decorator._SKIPPED.get(),
                    compute=Compute.current())
        done.set()
    thread = threading.Thread(target=other)
    thread.start()
    mine = Compute(threads=1)
    token = decorator._SKIPPED.set(frozenset({"A._system"}))
    try:
        with mine.limits(), _extracting():
            assert decorator._EXTRACTING.get() and Compute.current() is mine
            inside.set()
            done.wait()
    finally:
        decorator._SKIPPED.reset(token)
    thread.join()
    assert seen["extracting"] is False and seen["skipped"] == frozenset()
    assert seen["compute"] is not mine and seen["compute"] == Compute()
    assert not decorator._EXTRACTING.get() and Compute.current() == Compute()


def test_the_blas_hold_lasts_while_any_thread_is_inside_limits():
    first_in, second_in, first_out = threading.Event(), threading.Event(), threading.Event()

    def first():
        with Compute(threads=1).limits():
            first_in.set()
            second_in.wait()
        first_out.set()
    thread = threading.Thread(target=first)
    thread.start()
    first_in.wait()
    with Compute(threads=1).limits():
        second_in.set()
        first_out.wait()
        assert compute_module._BLAS_HOLD["depth"] == 1 and compute_module._BLAS_HOLD["limiter"] is not None
    thread.join()
    assert compute_module._BLAS_HOLD == {"depth": 0, "limiter": None}


def test_the_shared_positions_hit_only_their_own_arrays():
    keys = np.arange(0, 20, 2, dtype=np.int64)
    values = np.array([4, 6, 7], dtype=np.int64)
    keys.flags.writeable = values.flags.writeable = False
    at, found = columns._positions_shared(keys, values)
    again = columns._positions_shared(keys, values)
    assert again[0] is at and again[1] is found
    entry = columns._SHARED_POSITIONS[(id(keys), id(values))]
    assert entry[0]() is keys and entry[1]() is values          # held weakly
    other = np.array([4, 6, 8], dtype=np.int64)
    other.flags.writeable = False
    assert columns._positions_shared(keys, other)[1].tolist() == [True, True, True]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="os.fork: POSIX only")
def test_a_forked_child_starts_without_timings_writer_and_with_numbas_default_threads():
    import numba
    from openalea.metafspm.scene.recorder import SceneRecorder
    assert fork_safe_threading() in (None, "tbb", "omp")
    compute = Compute(threads=1)
    compute.count("probe", 1.)
    recorder = SceneRecorder()
    recorder._writer = object()                     # as a writer thread made in the parent
    numba_threads = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        read, write = os.pipe()
        child = os.fork()
        if child == 0:                              # the child: its verdict through the pipe, then out
            ok = (compute._timings == {} and recorder._writer is None
                  and numba.get_num_threads() == numba.config.NUMBA_NUM_THREADS)
            os.write(write, b"1" if ok else b"0")
            os._exit(0)
        os.close(write)
        verdict = os.read(read, 1)
        os.close(read)
        os.waitpid(child, 0)
    finally:
        numba.set_num_threads(numba_threads)
    assert verdict == b"1"
    assert compute._timings and recorder._writer is not None   # the parent's kept


def test_the_workqueue_layer_is_refused_before_a_fork(monkeypatch):
    import numba
    monkeypatch.setattr(numba, "threading_layer", lambda: "workqueue")
    with pytest.raises(RuntimeError, match="NUMBA_THREADING_LAYER"):
        fork_safe_threading()
