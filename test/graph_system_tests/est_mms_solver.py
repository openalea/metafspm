"""
Phase 1 — Method of Manufactured Solutions (MMS) solver tests.

Strategy: MMS on a ring-with-chord graph (cyclic, non-tree) or a pure ring
(for PDE-convergence tests).  A known analytic solution u*(i) is chosen; the
source f* is back-calculated so the exact discrete balance holds for u*.
Every solver method is then verified to recover u* within tolerance.

Graph-system tests (no MTG / decorator dependency — GraphSystem is built
directly from GraphView + equation blocks):

  Fix 1 (sparse Newton linear step)      → test_p1_sparse_jacobian_newton_step
  Fix 2 (graph-structured FD colouring)  → test_p1_jac_sparsity_*
  Fix 3 (Armijo line search)             → test_p1_linesearch_*
  Fix 4 (scipy dispatcher)               → test_p1_scipy_nonlinear_cross_solver
  Cross-solver regression                → test_p1_mms_linear_cross_solver*
  MMS convergence O(N⁻²)                → test_p1_mms_graph_refinement_convergence
  explicit=True decorator semantics      → test_explicit_edge_law_*
                                            test_explicit_node_balance_*
"""

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, diags

from openalea.metafspm.solve.system_specs import (
    EquationBlock,
    FieldState,
    GraphSystem,
    GraphView,
    SolverSpec,
    UnknownLayout,
)
from openalea.metafspm.solve.decorator import edge_law, node_balance


# ══════════════════════════════════════════════════════════════════════════════
# Shared graph builders
# ══════════════════════════════════════════════════════════════════════════════

def _ring_chord_graph(N: int):
    """
    Ring-with-chord GraphView with uniform K=1 and its dense Laplacian L.

    Topology: nodes 0..N-1 on a ring (edges i→(i+1)%N), plus chord 0→N//2.
    The chord guarantees at least two independent cycles (non-tree).
    """
    tails   = np.array(list(range(N)) + [0],                   dtype=np.int64)
    heads   = np.array([(i + 1) % N for i in range(N)] + [N // 2], dtype=np.int64)
    n_edges = len(tails)

    rows = np.concatenate([tails, heads])
    cols = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
    data = np.concatenate([np.ones(n_edges), -np.ones(n_edges)])
    inc  = coo_matrix((data, (rows, cols)), shape=(N, n_edges)).tocsc()

    gv = GraphView(
        node_ids=np.arange(N, dtype=np.int64),
        edge_ids=np.arange(n_edges, dtype=np.int64),
        tail=tails, head=heads,
        incidence=inc,
        boundary_incidence=csc_matrix((N, 0)),
        boundary_names=(),
    )
    L = (inc @ diags(np.ones(n_edges)) @ inc.T).toarray()
    return gv, L


def _pure_ring_graph(N: int, K_edge: float):
    """
    Pure ring GraphView (no chord) with uniform edge weight K_edge.

    Used for PDE convergence tests where K = (N/2π)² approximates −d²/dθ².
    """
    tails   = np.arange(N, dtype=np.int64)
    heads   = np.roll(np.arange(N, dtype=np.int64), -1)
    n_edges = N
    rows = np.concatenate([tails, heads])
    cols = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
    data = np.concatenate([np.ones(n_edges), -np.ones(n_edges)])
    inc  = coo_matrix((data, (rows, cols)), shape=(N, n_edges)).tocsc()
    gv = GraphView(
        node_ids=np.arange(N, dtype=np.int64),
        edge_ids=np.arange(n_edges, dtype=np.int64),
        tail=tails, head=heads,
        incidence=inc,
        boundary_incidence=csc_matrix((N, 0)),
        boundary_names=(),
    )
    K = np.full(n_edges, K_edge)
    L = (inc @ diags(K) @ inc.T).toarray()
    return gv, L


# ══════════════════════════════════════════════════════════════════════════════
# MMS system builders
# ══════════════════════════════════════════════════════════════════════════════

def _linear_mms_system(N, alpha=1.0, method="newton", linesearch=False):
    """
    Linear MMS: (L + α·I)·u = f*,  u*(i) = cos(2πi/N).
    f* computed from u* so the exact solution is u*.
    """
    gv, L  = _ring_chord_graph(N)
    u_star = np.cos(2 * np.pi * np.arange(N) / N)
    A      = L + alpha * np.eye(N)
    f_star = A @ u_star

    def ev(ctx, _A=A, _f=f_star):
        return _A @ ctx.node_unknowns["u"] - _f

    system = GraphSystem(
        graph=gv,
        node_fields={"u": FieldState("u", "node", np.zeros(N))},
        edge_fields={},
        boundary_ports=(),
        unknowns=UnknownLayout(node_fields=("u",), edge_fields=()),
        solver=SolverSpec(method=method, tol=1e-10, max_iter=100, linesearch=linesearch),
        equation_blocks=(EquationBlock(name="balance", evaluator=ev),),
        parameters={},
    )
    return system, u_star


def _linear_mms_direct(N, alpha=1.0):
    """Same MMS problem via the linear-assembly path (matrix_evaluator / rhs_evaluator)."""
    gv, L  = _ring_chord_graph(N)
    u_star = np.cos(2 * np.pi * np.arange(N) / N)
    A_sp   = csr_matrix(L + alpha * np.eye(N))
    f_star = A_sp @ u_star

    def matrix_ev(ctx, _A=A_sp): return _A
    def rhs_ev(ctx, _f=f_star):  return _f

    system = GraphSystem(
        graph=gv,
        node_fields={"u": FieldState("u", "node", np.zeros(N))},
        edge_fields={},
        boundary_ports=(),
        unknowns=UnknownLayout(node_fields=("u",), edge_fields=()),
        solver=SolverSpec(method="linear_direct", tol=1e-10),
        equation_blocks=(),
        matrix_evaluator=matrix_ev,
        rhs_evaluator=rhs_ev,
        parameters={},
    )
    return system, u_star


def _nonlinear_mms_system(N, alpha=1.0, Vmax=0.5, Km=2.0, method="newton", linesearch=False):
    """
    Nonlinear MMS: L·u + α·u + Vmax·u/(Km+u) = f*
    u*(i) = cos(2πi/N) + 3.0  (offset keeps u* > 0 and well-conditioned).
    f* manufactured from u* — exact solution is u*.
    """
    gv, L  = _ring_chord_graph(N)
    u_star = np.cos(2 * np.pi * np.arange(N) / N) + 3.0
    f_star = L @ u_star + alpha * u_star + Vmax * u_star / (Km + u_star)

    def ev(ctx, _L=L, _f=f_star):
        u = ctx.node_unknowns["u"]
        return _L @ u + alpha * u + Vmax * u / (Km + u) - _f

    system = GraphSystem(
        graph=gv,
        node_fields={"u": FieldState("u", "node", np.full(N, 3.0))},
        edge_fields={},
        boundary_ports=(),
        unknowns=UnknownLayout(node_fields=("u",), edge_fields=()),
        solver=SolverSpec(method=method, tol=1e-8, max_iter=100, linesearch=linesearch),
        equation_blocks=(EquationBlock(name="balance", evaluator=ev),),
        parameters={},
    )
    return system, u_star


# ══════════════════════════════════════════════════════════════════════════════
# Fix 2: jac_sparsity_matrix
# ══════════════════════════════════════════════════════════════════════════════

def test_p1_jac_sparsity_matrix_shape():
    """jac_sparsity_matrix returns the correct shape for 1-field and 2-field systems."""
    N = 8
    sys1, _ = _linear_mms_system(N)
    S1 = sys1.jac_sparsity_matrix()
    assert S1.shape == (N, N), f"1-field: expected ({N},{N}), got {S1.shape}"

    gv, _ = _ring_chord_graph(N)
    sys2 = GraphSystem(
        graph=gv,
        node_fields={
            "a": FieldState("a", "node", np.zeros(N)),
            "b": FieldState("b", "node", np.zeros(N)),
        },
        edge_fields={},
        boundary_ports=(),
        unknowns=UnknownLayout(node_fields=("a", "b"), edge_fields=()),
        solver=SolverSpec(method="newton"),
        equation_blocks=(),
        parameters={},
    )
    S2 = sys2.jac_sparsity_matrix()
    assert S2.shape == (2 * N, 2 * N), f"2-field: expected ({2*N},{2*N}), got {S2.shape}"


def test_p1_jac_sparsity_is_sparser_than_dense():
    """
    For a ring+chord of N=20, nnz(jac_sparsity_matrix) ≪ N².
    Ring+chord has max degree 3, so nnz ≤ N (diagonal) + 2·n_edges.
    """
    N = 20
    sys1, _ = _linear_mms_system(N)
    S = sys1.jac_sparsity_matrix()
    n_edges      = N + 1   # N ring edges + 1 chord
    expected_nnz = N + 2 * n_edges
    assert S.nnz <= expected_nnz + 1, (
        f"nnz={S.nnz} exceeds expected {expected_nnz} (N + 2·n_edges)"
    )
    assert S.nnz < N * N // 4, f"nnz={S.nnz} not significantly sparser than N²={N*N}"


# ══════════════════════════════════════════════════════════════════════════════
# Fix 1: sparse Newton linear step
# ══════════════════════════════════════════════════════════════════════════════

def test_p1_sparse_jacobian_newton_step():
    """
    Fix 1: matrix_evaluator returning a sparse CSR matrix is solved via
    spsolve in the Newton loop (not np.linalg.solve).
    Verified by correctness: residual < tol and solution matches u*.
    """
    N     = 10
    alpha = 1.0
    gv, L = _ring_chord_graph(N)
    u_star = np.cos(2 * np.pi * np.arange(N) / N)
    A_sp   = csr_matrix(L + alpha * np.eye(N))
    f_star = A_sp @ u_star

    system = GraphSystem(
        graph=gv,
        node_fields={"u": FieldState("u", "node", np.zeros(N))},
        edge_fields={},
        boundary_ports=(),
        unknowns=UnknownLayout(node_fields=("u",), edge_fields=()),
        solver=SolverSpec(method="newton", tol=1e-10),
        equation_blocks=(),
        matrix_evaluator=lambda ctx, _A=A_sp: _A,
        rhs_evaluator=lambda ctx, _f=f_star: _f,
        parameters={},
    )
    x = system.solve()
    np.testing.assert_allclose(x, u_star, atol=1e-8,
                                err_msg="sparse-Jacobian Newton failed MMS recovery")
    assert np.linalg.norm(system.residual(x), np.inf) < 1e-8


# ══════════════════════════════════════════════════════════════════════════════
# Cross-solver regression on linear MMS
# ══════════════════════════════════════════════════════════════════════════════

def test_p1_mms_linear_cross_solver():
    """
    All six solver paths recover u*(i) = cos(2πi/N) on the ring-with-chord
    graph within tolerance.
    """
    N = 12

    sys_ld, u_star = _linear_mms_direct(N)
    np.testing.assert_allclose(sys_ld.solve(), u_star, atol=1e-8,
                                err_msg="linear_direct MMS failed")

    for method in ("newton", "newton_fd", "scipy_krylov", "scipy_anderson", "scipy_hybr"):
        sys_m, _ = _linear_mms_system(N, method=method)
        x   = sys_m.solve()
        err = np.max(np.abs(x - u_star))
        assert err < 1e-6, f"{method}: max_err = {err:.2e} > 1e-6; solution={x}"


def test_p1_mms_linear_cross_solver_agreement():
    """
    Every pair of methods must agree to ‖y_A − y_B‖∞ < 1e-6.
    """
    N = 16
    results = {}
    sys_ld, _ = _linear_mms_direct(N)
    results["linear_direct"] = sys_ld.solve()
    for method in ("newton", "newton_fd", "scipy_krylov", "scipy_anderson"):
        sys_m, _ = _linear_mms_system(N, method=method)
        results[method] = sys_m.solve()

    ref     = results["newton"]
    tol_cross = 1e-6
    for name, x in results.items():
        if name == "newton":
            continue
        diff = np.max(np.abs(x - ref))
        assert diff < tol_cross, (
            f"Cross-solver disagreement newton vs {name}: ‖Δ‖∞ = {diff:.2e} ≥ {tol_cross}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MMS convergence on pure ring: O(N⁻²)
# ══════════════════════════════════════════════════════════════════════════════

def test_p1_mms_graph_refinement_convergence():
    """
    PDE: (−d²u/dθ² + α)u = (1+α)cos(θ),  u(θ) = cos(θ).
    Graph approximation: (L_scaled + α·I)·u = (1+α)·cos(2πi/N)
    with K = (N/2π)² so L_scaled → −d²/dθ² as N→∞.

    L∞ error between graph solution and cos(2πi/N) scales as O(N⁻²):
    error halves by at least ×3.5 each time N doubles.
    """
    alpha  = 1.0
    errors = {}
    for N in (8, 16, 32, 64):
        K_edge  = (N / (2 * np.pi)) ** 2
        gv, L   = _pure_ring_graph(N, K_edge)
        theta   = 2 * np.pi * np.arange(N) / N
        u_pde   = np.cos(theta)
        f_pde   = (1 + alpha) * np.cos(theta)
        A       = L + alpha * np.eye(N)

        def ev(ctx, _A=A, _f=f_pde):
            return _A @ ctx.node_unknowns["u"] - _f

        system = GraphSystem(
            graph=gv,
            node_fields={"u": FieldState("u", "node", np.zeros(N))},
            edge_fields={},
            boundary_ports=(),
            unknowns=UnknownLayout(node_fields=("u",), edge_fields=()),
            solver=SolverSpec(method="newton", tol=1e-12),
            equation_blocks=(EquationBlock(name="b", evaluator=ev),),
            parameters={},
        )
        x        = system.solve()
        errors[N] = np.max(np.abs(x - u_pde))

    for N_c, N_f in ((8, 16), (16, 32), (32, 64)):
        ratio = errors[N_c] / errors[N_f]
        assert ratio >= 3.5, (
            f"N={N_c}→{N_f}: error ratio {ratio:.2f} < 3.5 "
            f"(expected ≥3.5 for O(N⁻²)); errors {errors[N_c]:.2e} → {errors[N_f]:.2e}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Fix 3: Armijo line search
# ══════════════════════════════════════════════════════════════════════════════

def test_p1_linesearch_same_solution_as_newton():
    """
    Newton with linesearch=True must recover the same MMS solution as plain
    Newton on both a linear and a nonlinear problem.
    """
    N = 10

    sys_no, u_star = _linear_mms_system(N, method="newton", linesearch=False)
    sys_ls, _      = _linear_mms_system(N, method="newton", linesearch=True)
    x_no = sys_no.solve()
    x_ls = sys_ls.solve()
    np.testing.assert_allclose(x_no, x_ls, atol=1e-8,
                                err_msg="linesearch changed linear MMS solution")
    np.testing.assert_allclose(x_ls, u_star, atol=1e-8,
                                err_msg="linesearch+newton failed linear MMS recovery")

    sys_nl_no, u_nl = _nonlinear_mms_system(N, method="newton", linesearch=False)
    sys_nl_ls, _    = _nonlinear_mms_system(N, method="newton", linesearch=True)
    np.testing.assert_allclose(sys_nl_no.solve(), sys_nl_ls.solve(), rtol=1e-5,
                                err_msg="linesearch changed nonlinear MMS solution")
    np.testing.assert_allclose(sys_nl_ls.solve(), u_nl, rtol=1e-5,
                                err_msg="linesearch+newton failed nonlinear MMS recovery")


def test_p1_linesearch_residual_converges_to_tol():
    """After linesearch-Newton solve, ‖R‖∞ < tol."""
    N = 14
    sys_ls, _ = _linear_mms_system(N, method="newton", linesearch=True)
    x     = sys_ls.solve()
    r_norm = np.linalg.norm(sys_ls.residual(x), ord=np.inf)
    assert r_norm < 1e-9, f"linesearch final ‖R‖∞ = {r_norm:.2e} ≥ 1e-9"


# ══════════════════════════════════════════════════════════════════════════════
# Fix 4: scipy.optimize.root dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def test_p1_scipy_nonlinear_cross_solver():
    """
    Fix 4: scipy_krylov and scipy_anderson recover the nonlinear MMS solution
    (Michaelis-Menten source) on the ring-with-chord graph.
    """
    N = 10
    _, u_star = _nonlinear_mms_system(N)

    for method in ("scipy_krylov", "scipy_anderson"):
        sys_m, _ = _nonlinear_mms_system(N, method=method)
        x   = sys_m.solve()
        err = np.max(np.abs(x - u_star))
        assert err < 1e-5, f"{method}: nonlinear MMS max_err = {err:.2e} > 1e-5"


def test_p1_scipy_hybr_uses_analytic_jacobian():
    """
    scipy_hybr picks up the analytic Jacobian when one is registered, and
    recovers the MMS solution on the ring-with-chord graph.
    """
    N     = 8
    alpha = 1.0
    gv, L = _ring_chord_graph(N)
    u_star = np.cos(2 * np.pi * np.arange(N) / N)
    A      = L + alpha * np.eye(N)
    f_star = A @ u_star

    def ev(ctx, _A=A, _f=f_star):
        return _A @ ctx.node_unknowns["u"] - _f

    def jac_ev(ctx, _A=A):
        return _A

    system = GraphSystem(
        graph=gv,
        node_fields={"u": FieldState("u", "node", np.zeros(N))},
        edge_fields={},
        boundary_ports=(),
        unknowns=UnknownLayout(node_fields=("u",), edge_fields=()),
        solver=SolverSpec(method="scipy_hybr", tol=1e-10),
        equation_blocks=(EquationBlock(name="balance", evaluator=ev),),
        jacobian_evaluator=jac_ev,
        parameters={},
    )
    np.testing.assert_allclose(system.solve(), u_star, atol=1e-8,
                                err_msg="scipy_hybr with analytic Jacobian failed MMS")


# ══════════════════════════════════════════════════════════════════════════════
# explicit=True for edge_law and node_balance
# ══════════════════════════════════════════════════════════════════════════════

def test_explicit_edge_law_matches_residual_form():
    """
    @edge_law(explicit=True): framework generates R = edge_unknown − formula.

    Two GraphSystem objects for the same diffusion problem (B·q + α·c = f*,
    q = K·Bᵀ·c):
      - residual form:  edge evaluator returns q − K·Bᵀ·c  (classic)
      - explicit form:  edge evaluator wrapped by framework as q − K·Bᵀ·c

    Both must give the same c*, and the decorator tag must carry explicit=True.
    """
    N     = 6
    alpha = 1.0
    gv, L = _ring_chord_graph(N)
    B     = gv.incidence
    K     = np.ones(gv.n_edges)
    u_star = np.cos(2 * np.pi * np.arange(N) / N)
    f_star = L @ u_star + alpha * u_star

    def node_ev(ctx):
        return (
            np.asarray(B @ ctx.edge_unknowns["q"]).reshape(-1)
            + alpha * ctx.node_unknowns["c"]
            - f_star
        )

    def edge_ev_residual(ctx):
        return (
            ctx.edge_unknowns["q"]
            - K * np.asarray(B.T @ ctx.node_unknowns["c"]).reshape(-1)
        )

    def edge_ev_explicit(ctx):
        formula = K * np.asarray(B.T @ ctx.node_unknowns["c"]).reshape(-1)
        return ctx.edge_unknowns["q"] - formula

    def _make_sys(edge_ev):
        return GraphSystem(
            graph=gv,
            node_fields={"c": FieldState("c", "node", np.zeros(N))},
            edge_fields={"q": FieldState("q", "edge", np.zeros(gv.n_edges))},
            boundary_ports=(),
            unknowns=UnknownLayout(node_fields=("c",), edge_fields=("q",)),
            solver=SolverSpec(method="newton_fd", max_iter=50, tol=1e-10),
            equation_blocks=(
                EquationBlock(name="node_c", evaluator=node_ev),
                EquationBlock(name="edge_q", evaluator=edge_ev),
            ),
            parameters={},
        )

    packed_r = _make_sys(edge_ev_residual).solve()
    packed_e = _make_sys(edge_ev_explicit).solve()
    np.testing.assert_allclose(packed_r, packed_e, atol=1e-8,
                                err_msg="explicit edge wrapping differs from residual form")

    sys_r = _make_sys(edge_ev_residual)
    node_r, _ = sys_r.unpack_unknowns(sys_r.solve())
    np.testing.assert_allclose(node_r["c"], u_star, atol=1e-6,
                                err_msg="explicit edge_law MMS: c does not match u*")

    @edge_law(explicit=True)
    def _dummy(self, c): return K * c
    assert _dummy.__graph_tag__["explicit"] is True
    assert _dummy.__graph_tag__["kind"] == "edge_law"


def test_explicit_node_balance_matches_residual_form():
    """
    @node_balance(field=..., explicit=True): framework generates R = node_unknown − formula.

    Problem: (α·I + L)·u = f*  →  u* = cos(2πi/N).

    Residual form:  R = (α·I + L)·u − f*
    Explicit form:  formula returns (f*−L·u)/α;
                    framework wraps as R = u − (f*−L·u)/α  ≡ (α·u + L·u − f*)/α

    Both must converge to u* and the decorator tag must carry explicit=True.
    """
    N     = 6
    alpha = 2.0
    gv, L = _ring_chord_graph(N)
    u_star = np.cos(2 * np.pi * np.arange(N) / N)
    f_star = (alpha * np.eye(N) + L) @ u_star

    def node_ev_residual(ctx):
        u = ctx.node_unknowns["u"]
        return (alpha * np.eye(N) + L) @ u - f_star

    def node_ev_explicit(ctx):
        u = ctx.node_unknowns["u"]
        return u - (f_star - L @ u) / alpha   # R = u − formula

    def _make_sys(node_ev):
        return GraphSystem(
            graph=gv,
            node_fields={"u": FieldState("u", "node", np.zeros(N))},
            edge_fields={},
            boundary_ports=(),
            unknowns=UnknownLayout(node_fields=("u",), edge_fields=()),
            solver=SolverSpec(method="newton_fd", max_iter=50, tol=1e-10),
            equation_blocks=(EquationBlock(name="node_u", evaluator=node_ev),),
            parameters={},
        )

    packed_r = _make_sys(node_ev_residual).solve()
    packed_e = _make_sys(node_ev_explicit).solve()
    np.testing.assert_allclose(packed_r, packed_e, atol=1e-8,
                                err_msg="explicit node wrapping differs from residual form")

    sys_r = _make_sys(node_ev_residual)
    node_r, _ = sys_r.unpack_unknowns(sys_r.solve())
    np.testing.assert_allclose(node_r["u"], u_star, atol=1e-6,
                                err_msg="explicit node_balance MMS: u does not match u*")

    @node_balance(field="u", explicit=True)
    def _dummy(self, u): return u
    assert _dummy.__graph_tag__["explicit"] is True
    assert _dummy.__graph_tag__["kind"] == "node_balance"
