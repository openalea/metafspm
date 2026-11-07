"""
The sparse solvers of the graph systems' linear steps (solve.linear, Compute(linear_solver=)): SuperLU, SuperLU
reusing its ordering and its factorisation (the same solution, bit for bit), CHOLMOD (scikit-sparse, optional: its
tests skipped without it, its adapter tested on a stand-in), PARDISO and MUMPS (optional, registered when importable:
their tests skipped without them), a missing package raising at the Scene's build with its install line; a system declared @graph_system(symmetric=True) checked once per pattern (its held rows, a Dirichlet
node's, aside: CHOLMOD solves them first); a matrix of independent blocks solved block by block (each block's matrix
gathered from the whole one's values), in Compute.threads threads, the same whatever the threads.
"""
import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from openalea.metafspm.compute import Compute
from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.scene.population import Planting, planting, population_data_structure
from openalea.metafspm.solve import linear
from openalea.metafspm.solve.decorator import boundary_set, graph_jacobian, graph_system, node_balance
from openalea.metafspm.solve.linear import LINEAR_SOLVERS, SymmetryError, linear_solver

from growth import DOC
from layered_anatomy import wiring
from reduced_sections import LengthSegments, _collar, _conductances


def _matrix(n=200, seed=0, symmetric=True, pieces=1):
    """A conductance network's matrix (a chain and random links, a positive diagonal), *pieces* of them on the
    diagonal; not symmetric: an upwind advection added."""
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(pieces):
        rows = np.r_[np.arange(n - 1), rng.integers(0, n, n)]
        cols = np.r_[np.arange(1, n), rng.integers(0, n, n)]
        keep = rows != cols
        rows, cols = rows[keep], cols[keep]
        weights = rng.uniform(0.5, 2., rows.size)
        adjacency = sp.coo_matrix((weights, (rows, cols)), shape=(n, n))
        adjacency = (adjacency + adjacency.T).tocsr()
        laplacian = sp.diags(np.asarray(adjacency.sum(axis=1)).reshape(-1) + rng.uniform(0.1, 1., n)) - adjacency
        if not symmetric:
            laplacian = laplacian + sp.diags(rng.uniform(0.1, 0.5, n - 1), 1)
        blocks.append(laplacian)
    return sp.block_diag(blocks).tocsc(), rng.normal(size=n * pieces)


def test_the_registry_and_the_compute():
    assert {"superlu", "superlu_reuse", "cholmod"} <= set(LINEAR_SOLVERS)
    assert Compute(linear_solver="superlu_reuse").linear().name == "superlu_reuse"
    with pytest.raises(ValueError):
        Compute(linear_solver="umfpack")


@pytest.mark.parametrize("solver", linear.OPTIONAL_SOLVERS, ids=lambda solver: solver.name)
def test_an_optional_solver_is_registered_when_its_package_is_importable(solver, monkeypatch):
    from importlib.util import find_spec
    assert (solver.name in LINEAR_SOLVERS) == (find_spec(solver.package) is not None)
    monkeypatch.delitem(LINEAR_SOLVERS, solver.name, raising=False)             # as without its package
    with pytest.raises(ValueError, match=solver.package):
        Compute(linear_solver=solver.name)


def test_superlu_reuse_gives_superlus_solution_bit_for_bit():
    """Its ordering reused on new values of the same pattern, its factorisation on the same matrix: SuperLU's
    solution each time."""
    A, b = _matrix()
    reuse = linear_solver("superlu_reuse")
    for values in (A, A, A * 1.7 + sp.diags(np.full(A.shape[0], 0.3)), A):
        expected = spsolve(values.tocsc(), b, permc_spec="MMD_AT_PLUS_A")
        np.testing.assert_array_equal(reuse.solve(values, b, key="system"), expected)
    patterns = reuse._patterns["system"]
    assert len(patterns) == 1 and patterns[0].state[0]["permuted"]


def test_a_symmetric_declaration_is_checked_once_per_pattern():
    A, b = _matrix(symmetric=False)
    solver = linear_solver("superlu")
    with pytest.raises(SymmetryError):
        solver.solve(A, b, key="advection", symmetric=True)
    solver.solve(A, b, key="advection")                     # undeclared: solved
    S, c = _matrix()
    solver.solve(S, c, key="diffusion", symmetric=True)
    assert solver._patterns["diffusion"][0].symmetric_checked


def test_independent_blocks_are_solved_apart_the_same_whatever_the_threads(monkeypatch):
    monkeypatch.setattr(linear, "BLOCKS_MIN_NONZEROS", 100)
    A, b = _matrix(n=150, pieces=5)
    solutions = []
    for threads in (1, 3):
        with Compute(threads=threads).limits():
            solver = linear_solver("superlu")
            solutions.append(solver.solve(A, b, key="plants"))
            assert len(solver._patterns["plants"][0].blocks) == 5
    np.testing.assert_array_equal(solutions[0], solutions[1])
    np.testing.assert_allclose(A @ solutions[0], b, atol=1e-10)


def test_the_blocks_threads_are_the_computes_threads(monkeypatch):
    """At most one a block, and at most the solver's block_threads_max: SuperLU's factorisations are bandwidth
    bound, slower beyond SUPERLU_BLOCK_THREADS threads."""
    for threads, blocks, most, expected in ((1, 8, None, 1), (4, 8, None, 4), (12, 8, None, 8), (12, 8, 4, 4),
                                            (2, 8, 4, 2), (12, 8, 1, 1)):
        with Compute(threads=threads, workers=1).limits():
            assert linear._block_threads(blocks, most) == expected
    assert linear.SuperLU.block_threads_max == linear.SUPERLU_BLOCK_THREADS
    assert linear.Mumps.block_threads_max == 1                     # MUMPS is not thread-safe


def test_a_blocks_matrix_is_the_slice_of_the_matrix_bit_for_bit(monkeypatch):
    """Its matrix gathered from the whole one's values at the positions found once per pattern: the same pattern
    and values as matrix[rows][:, rows], explicit zeros included."""
    monkeypatch.setattr(linear, "BLOCKS_MIN_NONZEROS", 100)
    A, _ = _matrix(n=60, pieces=4, symmetric=False)
    A = sp.csc_matrix(A)
    A.data[::7] = 0.                                                 # explicit zeros kept in the pattern
    rng = np.random.default_rng(1)
    order = rng.permutation(A.shape[0])                              # the pieces' rows interleaved
    A = A[order][:, order].tocsc()
    A.sort_indices()
    blocks = linear._blocks(A)
    assert len(blocks) == 4
    for block in blocks:
        expected = A[block.rows][:, block.rows].tocsc()
        expected.sort_indices()
        found = block.matrix(A.data)
        for name in ("indptr", "indices", "data"):
            np.testing.assert_array_equal(getattr(found, name), getattr(expected, name))
    assert linear._blocks(sp.csc_matrix(_matrix(n=10)[0])) == []     # below BLOCKS_MIN_NONZEROS: solved whole


@pytest.mark.parametrize("name", ["pardiso", "mumps"])
def test_an_optional_solver_itself_keeps_its_analysis_the_same_whatever_the_threads(name, monkeypatch):
    if name not in LINEAR_SOLVERS:
        pytest.skip(f"{name}'s package is not installed")
    monkeypatch.setattr(linear, "BLOCKS_MIN_NONZEROS", 100)
    A, b = _matrix(n=150, pieces=3, symmetric=False)
    solutions = []
    for threads in (1, 3):
        with Compute(threads=threads, linear_solver=name).limits() as compute:
            solver = compute.linear()
            for values in (A, A * 1.7 + sp.diags(np.full(A.shape[0], 0.3))):
                x = solver.solve(values, b, key="plants")
                np.testing.assert_allclose(x, spsolve(sp.csc_matrix(values), b), rtol=1e-10, atol=1e-12)
                factors = [solver._patterns["plants"][0].state[k]["factor"] for k in range(3)]
                if values is A:
                    analysed = factors
                else:
                    assert all(f is g for f, g in zip(factors, analysed))          # the analysis kept
            solutions.append(x)
    np.testing.assert_array_equal(solutions[0], solutions[1])


def test_many_pieces_are_gathered_into_at_most_blocks_max(monkeypatch):
    monkeypatch.setattr(linear, "BLOCKS_MIN_NONZEROS", 10)
    monkeypatch.setattr(linear, "BLOCKS_MAX", 4)
    A, b = _matrix(n=20, pieces=11)
    solver = linear_solver("superlu")
    x = solver.solve(A, b, key="many")
    assert len(solver._patterns["many"][0].blocks) <= 4
    np.testing.assert_allclose(A @ x, b, atol=1e-10)


def _missing_sksparse(monkeypatch):
    monkeypatch.setitem(sys.modules, "sksparse", None)            # import sksparse raises ImportError
    monkeypatch.setitem(sys.modules, "sksparse.cholmod", None)


def test_a_missing_optional_package_raises_at_the_scenes_build_with_its_install_line(monkeypatch):
    from scene_doubles import DT, RootPopulation, SceneSoil, Seedlings, doubles_planting, soil_translator
    from openalea.metafspm.scene.scene import Scene
    _missing_sksparse(monkeypatch)
    with pytest.raises(ImportError, match="pip install scikit-sparse"):
        Scene(doubles_planting([RootPopulation, Seedlings]), environment=[SceneSoil],
              translator=soil_translator("SceneExudation", "SeedlingExudation"), time_step=DT,
              compute=Compute(linear_solver="cholmod"))


class _StandInFactor:
    """scikit-sparse's Factor, its symbolic analysis kept (a stand-in: scipy's LU of the matrix it is given)."""
    analyses = 0

    def __init__(self, matrix):
        type(self).analyses += 1
        self.pattern = (matrix.indptr.copy(), matrix.indices.copy())

    def cholesky_inplace(self, matrix):
        assert np.array_equal(matrix.indptr, self.pattern[0]) and np.array_equal(matrix.indices, self.pattern[1])
        np.testing.assert_allclose(matrix.toarray(), matrix.toarray().T)
        assert matrix.diagonal().min() > 0                      # definite: a negative one given as its opposite
        self.lu = sp.linalg.splu(matrix.tocsc())

    def __call__(self, rhs):
        return self.lu.solve(rhs)


def test_cholmods_adapter_keeps_its_analysis_and_solves_a_negative_definite_system_as_its_opposite(monkeypatch):
    module = types.ModuleType("sksparse")
    cholmod = types.ModuleType("sksparse.cholmod")
    cholmod.analyze = _StandInFactor
    module.cholmod = cholmod
    monkeypatch.setitem(sys.modules, "sksparse", module)
    monkeypatch.setitem(sys.modules, "sksparse.cholmod", cholmod)
    _StandInFactor.analyses = 0
    A, b = _matrix()
    solver = linear_solver("cholmod")
    for values in (-A, -1.3 * A, -A):                             # a Newton Jacobian: negative definite
        np.testing.assert_allclose(solver.solve(values, b, key="water", symmetric=True),
                                   spsolve(values.tocsc(), b), rtol=1e-10)
    assert _StandInFactor.analyses == 1
    N, c = _matrix(symmetric=False)
    np.testing.assert_allclose(solver.solve(N, c, key="solutes"), spsolve(N.tocsc(), c), rtol=1e-10)   # SuperLU's
    assert _StandInFactor.analyses == 1


def test_cholmod_itself():
    pytest.importorskip("sksparse.cholmod")
    A, b = _matrix()
    solver = linear_solver("cholmod")
    np.testing.assert_allclose(solver.solve(-A, b, key="water", symmetric=True), spsolve(-A.tocsc(), b),
                               rtol=1e-8)


# ── A graph system's declaration reaches its solves ──

@dataclass
class _Chain(FunctionalComponent):
    """The toy sections' network, its walls joined to the soil (Robin), its collar held (Dirichlet) when *held*."""
    water_potential: float = state_variable(**DOC, default=-0.5, location="node")
    K: float = parameter(**DOC, by="", default=1., location="edge")
    label: float = parameter(**DOC, by="", default=0., scale=scales.Compartment)
    kind: float = parameter(**DOC, by="", default=0., scale=scales.Compartment)
    length: float = parameter(**DOC, by="", default=1., location="node")
    soil_water_potential: float = parameter(**DOC, by="", default=-0.2, location="node")
    xylem_water_potential: float = parameter(**DOC, by="", default=-1., location="node")


def _balance(self, water_potential, K, kind, length):
    B = self.graph_view.incidence
    conductance = _conductances(self.graph_view, K, kind, length)
    return np.asarray(B @ (conductance * np.asarray(B.T @ water_potential).reshape(-1))).reshape(-1)


def _jacobian(self, K, kind, length):
    B = self.graph_view.incidence
    return (B @ sp.diags(_conductances(self.graph_view, K, kind, length)) @ B.T).tocsr()


@dataclass
class SoilOnly(_Chain):
    @graph_system(node_unknowns=["water_potential"], solver="newton", symmetric=True)
    class _hydraulics:
        soil = boundary_set(filters={"kind": "==1"}, kind="robin", value="soil_water_potential", weight=0.6)
        _balance = node_balance(field="water_potential")(_balance)
        _jacobian = graph_jacobian(_jacobian)


@dataclass
class HeldCollar(_Chain):
    @graph_system(node_unknowns=["water_potential"], solver="newton", symmetric=True)
    class _hydraulics:
        soil = boundary_set(filters={"kind": "==1"}, kind="robin", value="soil_water_potential", weight=0.6)
        collar = boundary_set(filters=_collar, kind="dirichlet", value="xylem_water_potential")
        _balance = node_balance(field="water_potential")(_balance)
        _jacobian = graph_jacobian(_jacobian)


def _advected_balance(self, water_potential, K, kind, length):
    return _balance(self, water_potential, K, kind, length) + 0.3 * (_upwind(self) @ water_potential)


def _upwind(self):
    """Each node's outflow to its next node: an advection, not symmetric."""
    view = self.graph_view
    n = view.incidence.shape[0]
    return sp.coo_matrix((np.ones(view.tail.size), (view.tail, view.tail)), shape=(n, n)).tocsr() - \
        sp.coo_matrix((np.ones(view.tail.size), (view.head, view.tail)), shape=(n, n)).tocsr()


@dataclass
class Advected(_Chain):
    @graph_system(node_unknowns=["water_potential"], solver="newton", symmetric=True)
    class _hydraulics:
        soil = boundary_set(filters={"kind": "==1"}, kind="robin", value="soil_water_potential", weight=0.6)
        _balance = node_balance(field="water_potential")(_advected_balance)

        @graph_jacobian
        def _jacobian(self, K, kind, length):
            return (_jacobian(self, K, kind, length) + 0.3 * _upwind(self)).tocsr()


def _run(component_class):
    class ChainPlant(CompositeModel):
        from_scale, nodes, initiators = "SubOrgan", "Compartment", (LengthSegments,)

        @staticmethod
        def wiring(g):
            return wiring(g)

        parts = {"hydraulics": component_class}
    population = population_data_structure(planting([Planting()], ChainPlant, {}), ChainPlant)
    with Compute(linear_solver="superlu_reuse").limits() as compute:
        population.instance.run()
        population.instance.run()
    return population, compute


@pytest.mark.parametrize("component_class", [SoilOnly, HeldCollar])
def test_a_symmetric_system_is_solved_with_its_declaration(component_class):
    """A conductance network, with or without a held (Dirichlet) collar: symmetric, its held rows aside."""
    population, compute = _run(component_class)
    patterns = [pattern for key, patterns in compute.linear()._patterns.items() if key[0] == component_class.__name__
                for pattern in patterns]
    assert patterns and all(pattern.symmetric_checked for pattern in patterns)
    assert np.all(np.isfinite(population.data_structure.get("water_potential")))


def test_a_system_declared_symmetric_whose_matrix_is_not_raises():
    with pytest.raises(SymmetryError, match="Advected"):
        _run(Advected)


def test_cholmod_solves_the_held_rows_first(monkeypatch):
    """A Dirichlet row (its diagonal alone, its column coupled): solved first, the symmetric rest by CHOLMOD."""
    module = types.ModuleType("sksparse")
    cholmod = types.ModuleType("sksparse.cholmod")
    cholmod.analyze = _StandInFactor
    module.cholmod = cholmod
    monkeypatch.setitem(sys.modules, "sksparse", module)
    monkeypatch.setitem(sys.modules, "sksparse.cholmod", cholmod)
    A, b = _matrix()
    A = (-A).tolil()
    A[3, :] = 0.
    A[3, 3] = 1.                                                      # held: its column keeps its couplings
    A = A.tocsc()
    solver = linear_solver("cholmod")
    np.testing.assert_allclose(solver.solve(A, b, key="held", symmetric=True), spsolve(A, b), rtol=1e-10)


def test_superlu_reuse_keeps_one_factorisation_per_system():
    """A system whose pattern changes (a growing plant) keeps at most PATTERNS_KEPT orderings and one
    factorisation, its last pattern's; at most SYSTEMS_KEPT systems are kept."""
    reuse = linear_solver("superlu_reuse")
    for n in range(100, 100 + 3 * linear.PATTERNS_KEPT):
        A, b = _matrix(n=n)
        reuse.solve(A, b, key="growing")
    patterns = reuse._patterns["growing"]
    assert len(patterns) == linear.PATTERNS_KEPT
    assert sum("factor" in pattern.state[0] for pattern in patterns) == 1
    for k in range(linear.SYSTEMS_KEPT + 5):
        reuse.solve(*_matrix(n=50), key=("system", k))
    assert len(reuse._patterns) == linear.SYSTEMS_KEPT


def test_the_patterns_kept_are_bounded_in_bytes(monkeypatch):
    """A growing plant's pattern changes at every step: its former ones are dropped beyond PATTERN_BYTES_KEPT (the
    least recently solved first, the one being solved kept), whatever their number; the solutions unchanged."""
    A, b = _matrix(n=400)
    one = linear._Pattern(sp.csc_matrix(A)).nbytes()
    monkeypatch.setattr(linear, "PATTERN_BYTES_KEPT", 3 * one)
    solver = linear_solver("superlu")
    for n in range(400, 410):
        A, b = _matrix(n=n)
        np.testing.assert_array_equal(solver.solve(A, b, key="growing"), spsolve(sp.csc_matrix(A), b,
                                                                                  permc_spec="MMD_AT_PLUS_A"))
    kept = solver._patterns["growing"]
    assert len(kept) < linear.PATTERNS_KEPT
    assert sum(pattern.nbytes() for pattern in kept) <= 3 * one + kept[-1].nbytes()
    assert kept[-1].shape == (409, 409)                          # the last solved kept
    monkeypatch.setattr(linear, "PATTERN_BYTES_KEPT", 0)
    solver.solve(*_matrix(n=50), key="other")
    assert list(solver._patterns) == ["other"] and len(solver._patterns["other"]) == 1
