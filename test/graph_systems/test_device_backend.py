"""
Compute(device="cuda") without a GPU: the backend's dispatch, its transfers and its results against the cpu path,
on a fake CuPy (fake_cupy: numpy under cupy's interface, its device arrays refused by numpy, every crossing logged).
CuPy itself is optional and absent here: nothing imports it unless device="cuda", and a scene asking for it without
it, or without a device, raises at its build with the install line. GPU runs are by hand (docs/gpu.md).
"""
import subprocess
import sys

import numpy as np
import pytest
from scipy.sparse import block_diag, csr_matrix, diags, random as sparse_random

from openalea.metafspm import backend as backend_module
from openalea.metafspm.compute import Compute
from openalea.metafspm.scene.scene import Scene
from openalea.metafspm.solve.network import kron_reduction

import fake_cupy
from reduced_sections import N_SEGMENTS, SYSTEM, ExactLengthPlant, length_population
from scene_doubles import DT, RootPopulation, SceneSoil, Seedlings, doubles_planting, soil_translator


@pytest.fixture
def fake(monkeypatch):
    """The fake CuPy in backend.cupy_modules' place, every right-hand side and system offloaded."""
    modules = fake_cupy.modules()
    monkeypatch.setattr(backend_module, "cupy_modules", lambda: modules)
    monkeypatch.setattr(backend_module, "DEVICE_MIN_VALUES", 0)
    monkeypatch.setattr(backend_module, "DEVICE_MIN_NONZEROS", 0)
    return modules


def _network(n=80, seed=0):
    rng = np.random.default_rng(seed)
    a = np.r_[np.arange(n - 1), rng.integers(0, n, n)]
    b = np.r_[np.arange(1, n), rng.integers(0, n, n)]
    keep = a != b
    a, b = a[keep], b[keep]
    k = rng.uniform(0.5, 2., a.size)
    edge = np.arange(a.size)
    incidence = csr_matrix((np.r_[np.ones(a.size), -np.ones(a.size)], (np.r_[a, b], np.r_[edge, edge])),
                           shape=(n, a.size))
    terms = np.zeros(n)
    terms[[3, 7]] = [0.3, 0.8]
    return (incidence @ diags(k) @ incidence.T + diags(terms)).tocsr()


def _kron_values(matrix, kept):
    kron = kron_reduction(matrix, kept)
    interior = kron.interior_nodes.size
    source = np.linspace(0.1, 1., interior)
    values = {"reduced": kron.reduced, "inject": kron.inject(source), "injection": kron.injection(),
              "shares": kron.shares(np.ones(kron.kept_nodes.size, dtype=bool), source),
              "recover": kron.recover(np.linspace(1., 2., kron.kept_nodes.size), source)}
    kron.release()
    values["recover_released"] = kron.recover(np.linspace(1., 2., kron.kept_nodes.size), source)
    return values


def _scene(**options):
    return Scene(doubles_planting([RootPopulation, Seedlings]), environment=[SceneSoil],
                 translator=soil_translator("SceneExudation", "SeedlingExudation"), time_step=DT, **options)


# ── no CuPy: nothing imported, a clear error at the scene's build ──

def test_nothing_imports_cupy_unless_the_device_is_cuda():
    code = ("import sys\n"
            "import openalea.metafspm\n"
            "from openalea.metafspm.compute import Compute\n"
            "from openalea.metafspm.solve import linear, network\n"
            "compute = Compute()\n"
            "compute.check()\n"
            "with compute.limits():\n"
            "    network.kron_reduction(linear.sp.eye(4).tocsr() * 2., [True, False, False, True])\n"
            "assert compute.backend().name == 'cpu' and compute.device_info() is None\n"
            "assert not [m for m in sys.modules if m.split('.')[0] in ('cupy', 'cupyx', 'nvmath')], 'imported'\n")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_a_missing_cupy_raises_at_the_scenes_build_with_its_install_line():
    try:
        import cupy  # noqa: F401
        pytest.skip("CuPy is installed here")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="pip install cupy-cuda12x"):
        _scene(compute=Compute(device="cuda"))


def test_no_device_raises_at_the_scenes_build(monkeypatch):
    modules = fake_cupy.modules(devices=0)
    monkeypatch.setattr(backend_module, "cupy_modules", lambda: modules)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        _scene(compute=Compute(device="cuda"))


def test_the_devices_linear_solver():
    assert Compute().linear_solver == "superlu"
    assert Compute(device="cuda").linear_solver == "cupy"
    assert Compute(device="cuda", linear_solver="superlu").linear_solver == "superlu"
    with pytest.raises(ValueError, match="device='cuda'"):
        Compute(linear_solver="cupy")


# ── the archetypes' Kron reductions on the device ──

def test_a_kron_reduction_on_the_device_equals_the_cpus_its_transfers_explicit(fake):
    matrix = _network()
    kept = np.zeros(matrix.shape[0], dtype=bool)
    kept[[0, 5, 17, 33, 59, 79]] = True
    with Compute().limits():
        reference = _kron_values(matrix, kept)
    assert not fake.log                                  # the cpu: nothing on the device
    compute = Compute(device="cuda")
    with compute.limits():
        values = _kron_values(matrix, kept)
    for name in reference:                               # the same SuperLU: the same numbers
        np.testing.assert_array_equal(values[name], reference[name])
    calls = [call for call, _ in fake.log]
    assert calls.count("SuperLU") == 2                   # the factors copied once per factorisation (and after release)
    transfers = compute.device_info()["transfers"]
    assert transfers["to_device"] == calls.count("asarray") + calls.count("SuperLU")    # the factors' copies too
    assert transfers["to_host"] == calls.count("asnumpy") == calls.count("lu_solve")
    assert transfers["bytes_to_host"] == sum(size for call, size in fake.log if call == "asnumpy")
    assert compute.device_info()["solver"] == "cupyx"


def test_a_small_right_hand_side_stays_on_the_host(monkeypatch):
    modules = fake_cupy.modules()
    monkeypatch.setattr(backend_module, "cupy_modules", lambda: modules)
    matrix = _network()
    kept = np.zeros(matrix.shape[0], dtype=bool)
    kept[[0, 79]] = True
    with Compute(device="cuda").limits():
        _kron_values(matrix, kept)
    assert not modules.log                               # below DEVICE_MIN_VALUES: no copy is worth it


def test_a_reductions_kinds_on_the_device_equal_the_cpus(fake):
    lengths = [0.3 + 0.37 * i for i in range(N_SEGMENTS)]

    def archetypes(compute):
        population = length_population(ExactLengthPlant, lengths)
        data_structure = population.data_structure
        reduction = data_structure.structure.reduction
        with compute.limits():
            return {i: by_system[SYSTEM] for i, by_system in
                    reduction.group_archetypes(data_structure, reduction.reduced_groups(data_structure), release=True).items()}

    reference = archetypes(Compute())
    compute = Compute(device="cuda")
    found = archetypes(compute)
    assert found.keys() == reference.keys()
    for i in reference:
        for name in ("D", "e", "interior_source", "kept_rows", "interior_rows"):
            np.testing.assert_array_equal(getattr(found[i], name), getattr(reference[i], name))
    assert compute.device_info()["transfers"]["to_host"] > 0


# ── the linear solves ──

def _blocks(count=3, size=40):
    blocks = [(sparse_random(size, size, density=0.1, random_state=seed) + diags(np.full(size, 4.)))
              for seed in range(count)]
    return block_diag(blocks).tocsc()


def test_the_cupy_solver_solves_the_whole_system_in_one_call_its_pattern_copied_once(fake, monkeypatch):
    from openalea.metafspm.solve import linear
    monkeypatch.setattr(linear, "BLOCKS_MIN_NONZEROS", 0)       # the cpu would split its blocks
    matrix = _blocks()
    rhs = np.linspace(1., 2., matrix.shape[0])
    compute = Compute(device="cuda")
    with compute.limits():
        first = compute.linear().solve(matrix, rhs, key="system")
        crossings = len(fake.log)
        second = compute.linear().solve(matrix * 2., rhs, key="system")
    reference = linear.linear_solver("superlu").solve(matrix, rhs)
    np.testing.assert_allclose(first, reference, rtol=1e-12)
    np.testing.assert_allclose(second, reference / 2., rtol=1e-12)
    calls = [call for call, _ in fake.log]
    assert calls.count("spsolve") == 2                   # the three blocks together: one call each solve
    second_calls = [call for call, _ in fake.log[crossings:]]
    assert second_calls.count("asarray") == 2            # the same pattern: only the values and the rhs copied
    assert second_calls.count("asnumpy") == 1


def test_a_small_system_is_solved_on_the_host(monkeypatch):
    modules = fake_cupy.modules()
    monkeypatch.setattr(backend_module, "cupy_modules", lambda: modules)
    compute = Compute(device="cuda")
    matrix = _blocks(count=1)
    with compute.limits():
        x = compute.linear().solve(matrix, np.ones(matrix.shape[0]))
    np.testing.assert_allclose(matrix @ x, 1., rtol=1e-12)
    assert not modules.log


def test_cudss_is_used_when_installed(monkeypatch):
    modules = fake_cupy.modules(cudss=True)
    monkeypatch.setattr(backend_module, "cupy_modules", lambda: modules)
    monkeypatch.setattr(backend_module, "DEVICE_MIN_NONZEROS", 0)
    compute = Compute(device="cuda")
    matrix = _blocks(count=2)
    with compute.limits():
        x = compute.linear().solve(matrix, np.ones(matrix.shape[0]))
    np.testing.assert_allclose(matrix @ x, 1., rtol=1e-12)
    assert ("cudss", matrix.shape[0]) in modules.log and compute.device_info()["solver"] == "cudss"


# ── the scene ──

def test_a_scene_on_the_device_records_it(fake, tmp_path):
    import json
    from openalea.metafspm.scene.recorder import SceneRecorder
    scene = _scene(compute=Compute(device="cuda"), recorder=SceneRecorder(str(tmp_path)))
    scene.run()
    scene.finish()
    with open(tmp_path / "run.json") as file:
        device = json.load(file)["compute"]["device"]
    assert device["name"] == "cuda" and device["solver"] == "cupyx"
