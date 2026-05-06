# Solver Architecture for `metafspm` — Revised Implementation Plan

> Based on direct audit of `graph_system.py` and `graph_system_decorators.py` (branch `graph`).
> Three revisions from v1: (1) no tree-specific solvers — the graph is always non-oriented and cyclic-capable; (2) solver options organised by **system linearity**, not by physical problem class; (3) JAX-first stack (CPU-compatible, GPU-optional).

---

## 1. Executive summary

The codebase is architecturally sound at the residual-assembly level: `GraphView` already holds a CSC incidence matrix, `EquationBlock` evaluators compose cleanly, and the decorator layer is expressive and well-structured. The problems are entirely in the **solve loop** of `GraphSystem.solve()`:

- **The Newton linear step is dense** (`np.linalg.solve` on line 505), even though the Jacobian was assembled as sparse. This hard-caps scalability at ~10³–10⁴ unknowns.
- **The FD Jacobian has no sparsity colouring** — it evaluates 2N residuals column-by-column. On a graph of degree d, colouring reduces this to 2(d+1) evaluations. That is a free 100–1000× speedup never taken.
- **There is no time integration** — `previous_node_fields` and `dt` flow through the `EquationContext`, but `GraphSystem` offers no integrator. Every transient model reimplements time-stepping externally, without implicit stiffness handling.
- **Newton has no globalization** — no line search, no damping, no trust region. It diverges on poorly-scaled nonlinear problems (sigmoidal loaders, near-cavitation hydraulics) without any diagnostic.
- **No backend abstraction** — solver method is a string literal. Swapping in a better solver requires rewriting `GraphSystem.solve()`.

The fixes are surgical and non-breaking. The highest-leverage change is a two-line fix to the Newton linear step (sparse `spsolve` instead of dense `linalg.solve`), followed by graph-structured Jacobian colouring and a pluggable solver backend enum. A JAX residual path then makes Diffrax and Optimistix available without changing the decorator API.

---

## 2. Exact audit of the current solver

### 2.1 What is correctly implemented

| Component | Status | Notes |
|---|---|---|
| `GraphView` incidence matrix | ✓ | CSC, built from `tail`/`head` arrays; `weighted_laplacian` utility available |
| `FieldState` / `UnknownLayout` pack/unpack | ✓ | Flat vector layout, field-named slicing |
| `BoundaryPort` + `boundary_incidence` | ✓ | Dirichlet/Neumann distinction at decorator level |
| `EquationBlock` additive accumulation | ✓ | Multiple blocks per field, type-filtered sub-arrays |
| `linear_direct` path | ✓ | Correctly uses `spsolve` via `_solve_linear` |
| `@graph_jacobian` analytic path | ✓ | Bypasses FD, used when provided |
| `EquationContext` with `previous_node_fields` + `dt` | ✓ | Infrastructure for transient residuals exists |
| Decorator MRO scan + prop snapshot | ✓ | Robust, lazy, correct |

### 2.2 Exact bugs / gaps in the solve path

**Bug 1 — Dense Newton linear step (`graph_system.py` line 505)**

```python
# CURRENT — always dense, O(N³):
correction = np.linalg.solve(np.asarray(jac, dtype=np.float64), -residual)
```

`GraphSystem.jacobian()` returns a NumPy 2-D array even when the analytic evaluator or the linear-assembly path returns a sparse matrix (line 462: `return matrix.toarray()`). The Newton step then calls `np.linalg.solve` on the dense conversion. This is the single worst-scaling decision in the file.

**Bug 2 — FD Jacobian has no sparsity structure (`graph_system.py` lines 423-445)**

The loop iterates over every column independently. A graph of N nodes with average degree d has a Jacobian with at most (d+2)N non-zeros. Graph-distance colouring reduces 2N perturbation evaluations to 2(d+1) — a ~500× reduction for d=2 (linear pipe network) or ~100× for d=5 (anatomical cell graph).

**Gap 1 — No time integration driver**

`previous_node_fields` and `dt` are correctly threaded through `EquationContext`, meaning modellers can write backward-Euler residuals manually. But there is no integrator that manages step-size control, error estimation, or stiffness detection. Every caller reimplements this.

**Gap 2 — No Newton globalization**

The Newton loop has no line search. For nonlinear BCs (sigmoidal loading, Michaelis-Menten saturation) the initial Newton step can overshoot into a region where the residual is larger. Without backtracking the loop diverges with an opaque `AssertionError`.

**Gap 3 — No matrix-free Krylov option**

The Jacobian is always formed explicitly (analytic or FD). On large sparse systems a matrix-free JFNK (Jacobian-Free Newton-Krylov) using only residual evaluations is far more memory-efficient and avoids the FD cost entirely.

---

## 3. Solver options by system linearity

The three physical problem classes (steady nonlinear, ODE/DAE, PDE-on-graph) all reduce to the same mathematical question at each time step: **how nonlinear is the system F(x) = 0 we need to solve right now?** Organising options around this question is cleaner than organising by physics.

### 3.1 Linear systems — `F(x) = Ax - b`

These arise from Hagen-Poiseuille xylem, linearised diffusion, or any model where conductances are treated as fixed during a step.

**Already handled:** the `linear_direct` method path uses `spsolve` correctly.

**What is missing:**
- For large graphs (10⁶ nodes), direct sparse LU (`spsolve` / UMFPACK) will run out of fill-in memory. An iterative Krylov solver (GMRES, CG for SPD systems like the weighted Laplacian) with a preconditioner is needed.
- Recommended preconditioners for graph Laplacians: **incomplete Cholesky (IC)** for symmetric positive-definite systems (pressure solve with fixed conductances); **ILU(k)** for non-symmetric systems. Both available via `scipy.sparse.linalg` and `pyamg`.
- For SPD systems, **algebraic multigrid (AMG via `pyamg`)** is theoretically optimal (O(N) solve) and works on any graph topology including cycles. Algebraic multigrid requires no mesh structure — only the matrix — making it ideal here.

**Recommended addition to `SolverSpec.method`:** `"linear_iterative"` using `scipy.sparse.linalg.gmres` or `cg` with `pyamg.smoothed_aggregation_solver` as preconditioner. This is a pure-Python addition with no new dependencies beyond `pyamg`.

### 3.2 Smooth nonlinear systems — `F(x) = 0`, `F` differentiable

These arise from any model with Michaelis-Menten kinetics, sigmoidal transporters, pressure-dependent conductances, or coupled water-carbon balances at steady state.

**What is missing:** globalized Newton (line search), sparsity-aware Jacobian, Krylov linear step.

#### 3.2.1 scipy path (Phase 1, zero new dependencies)

`scipy.optimize.root` offers several methods:

| `method` | Jacobian needed? | Sparsity? | Globalized? | Best for |
|---|---|---|---|---|
| `'hybr'` | FD (can supply) | no | yes (trust region) | small–medium N |
| `'lm'` | FD | no | yes (LM) | overdetermined, medium N |
| `'krylov'` | **no** (JFNK) | N/A | limited | **large N, no Jacobian** |
| `'broyden1'` | no (quasi-Newton) | no | no | medium N, cheap residual |
| `'anderson'` | no (fixed-point accel.) | N/A | no | **weakly coupled systems, large N** |
| `'df-sane'` | no | N/A | yes | derivative-free, robust |

For the FSPM use case:
- **`'krylov'`** (Newton-GMRES, matrix-free) is the best drop-in for large graphs when no analytic Jacobian is available. It calls `F` once per GMRES iteration, never forms `J`.
- **`'anderson'`** is the recommended fallback for fixed-point iterations (e.g. iterating on fluxes between coupled water and carbon pools). Anderson acceleration on 6–10 past iterates typically converges in 10–20 evaluations for weakly coupled FSPM systems.

The current `"newton"` path in `GraphSystem.solve()` should be replaced by a thin wrapper around `scipy.optimize.root`, making the method a proper solver choice rather than a hand-rolled loop.

#### 3.2.2 JAX / Optimistix path (Phase 2)

[Optimistix](https://arxiv.org/abs/2402.09983) (Rader, Lyons, Kidger 2024) is a JAX-based modular optimisation and root-finding library. Its root-finder API:

```python
import optimistix as optx
import jax.numpy as jnp

solver = optx.Newton(rtol=1e-8, atol=1e-10, linear_solver=optx.AutoLinearSolver(well_posed=True))
# or for nonpositive-definite: optx.Chord, optx.LevenbergMarquardt
sol = optx.root_find(F_jax, solver, y0, args=params, max_steps=30)
```

Key properties relevant to `metafspm`:
- **Works on CPU without a GPU.** `jax.jit` compiles to efficient CPU code (XLA/LLVM) by default; GPU is additive.
- **End-to-end differentiable** — `jax.grad` through the entire solve (implicit differentiation via implicit function theorem). This gives free sensitivity analysis later.
- **`linear_solver` is pluggable** — use `lineax.GMRES()` for graph-sparse systems, `lineax.CG()` for SPD (weighted Laplacian), or `lineax.Tridiagonal()` for banded structures. Lineax handles sparsity via JAX's `BCOO` format.
- **`jax.vmap`** over the outer `solve` call gives free ensemble/parameter-sweep parallelism on CPU or GPU.

The main requirement: residual functions must be written as pure JAX functions (no in-place NumPy mutation, no Python side effects). Since the current evaluators already return new arrays from named-arg methods, this is achievable by replacing `np` with `jnp` inside the method bodies — which is a model-author concern, not a framework concern.

#### 3.2.3 PETSc SNES path (Phase 3, for distributed scale)

`petsc4py.PETSc.SNES` provides:
- Globalized Newton (line search: `bt`, `l2`, `cp`; trust region: `newtontr`)
- Full Krylov menu (GMRES, BiCGStab, etc.) + Krylov preconditioners (ASM, HYPRE-BoomerAMG, GAMG)
- MPI-native: the graph can be partitioned across processes using `DMPlex` or `DMNetwork`

For the `metafspm` use case, a minimal petsc4py adapter calls `SNES.setFunction(F_petsc)` and `SNES.setJacobian(J_petsc)`, where `J_petsc` can be a PETSc `Mat` with the graph-derived sparsity pattern pre-filled (once, at assembly time). This is Phase 3 territory but the adapter is small (~100 lines).

### 3.3 Non-smooth or discontinuous systems

Cavitation in xylem (abrupt drop in conductance), stomatal closure (piecewise), organ death/emergence. These are technically nonlinear systems with discontinuous right-hand sides.

**Recommended approach:** handle discontinuities in the residual function as smooth approximations (tanh switching functions) wherever possible. For hard discontinuities (organ emergence), treat architectural change as an outer event loop and re-assemble the `GraphView` rather than embedding the discontinuity in `F`. This is already implicit in the current architecture.

For ODE/DAE integration (section below), SUNDIALS CVODE event detection (`rootfn`) is the right tool for hard switches during integration.

### 3.4 Time-dependent systems — `M ẏ = f(t, y)` or `F(t, y, ẏ) = 0`

This is a new solver mode that the current `GraphSystem` supports at the residual level (via `dt` and `previous_node_fields` in the context) but does not integrate. **The framework needs a `solve_ivp` driver.**

Three options, ordered by capability:

#### 3.4.1 scipy `solve_ivp` with `BDF` / `Radau` (Phase 1 extension)

```python
from scipy.integrate import solve_ivp
import scipy.sparse

def rhs(t, y):
    # wrap GraphSystem.residual as an ODE rhs (explicit form)
    ...

jac_sparsity = scipy.sparse.csr_matrix(...)  # from graph incidence
sol = solve_ivp(rhs, t_span, y0, method='BDF',
                jac_sparsity=jac_sparsity, rtol=1e-6, atol=1e-9)
```

The `jac_sparsity` argument to `solve_ivp` triggers finite-difference Jacobian colouring internally. This is the same colouring that the current `finite_difference_jacobian` method should also implement. A utility `GraphSystem.jac_sparsity_matrix()` that returns a CSR mask derived from the incidence matrix covers both.

**Limitation:** `scipy.solve_ivp` is pure Python overhead per step; will cap at ~10⁵ d.o.f. for practical runtime.

#### 3.4.2 SUNDIALS CVODE/IDA via scikit-SUNDAE (Phase 2)

[scikit-SUNDAE](https://scikit-sundae.readthedocs.io/) (NREL, BSD-3) provides Python bindings to CVODE (stiff ODE) and IDA (DAE index 1). Relevant features:

```python
from sksundae.cvode import CVODE

def rhs(t, y, yp, userdata):
    # yp[:] = GraphSystem.rhs_as_ode(t, y, userdata)
    ...

solver = CVODE(rhs, userdata=system)
solver.init(t0, y0)
solver.set_jac_type('sparse')        # banded or direct-sparse (SuperLU_MT)
solver.set_jac_pattern(jac_pattern)  # CSR sparsity mask
```

Key properties:
- **Adaptive step-size BDF** with variable order (1–5) — handles stiffness from Michaelis-Menten saturation near equilibrium
- **IDA** for genuine DAE (e.g. instantaneous water balance constraint + slow carbon balance)
- Ships as conda-forge binary wheels (no compilation needed)
- Note: scikit-SUNDAE does not wrap KINSOL (their docs confirm this) — use `scipy.optimize.root(method='krylov')` for the steady-state path instead

#### 3.4.3 Diffrax (JAX) — unified ODE/DAE path (Phase 2, preferred for JAX stack)

[Diffrax](https://docs.kidger.site/diffrax/) (Kidger 2021–2024) provides JAX-based ODE/SDE solvers. Recommended solvers for FSPM:

| Solver | Stiff? | DAE? | GPU? | Best for |
|---|---|---|---|---|
| `diffrax.Tsit5` | no | no | yes | explicit, weakly coupled fast dynamics |
| `diffrax.Kvaerno5` | **yes** | no | yes | stiff transport-reaction |
| `diffrax.KenCarp4` | **IMEX** | no | yes | split explicit (slow C-balance) + implicit (fast hydraulics) |
| `diffrax.ImplicitEuler` | yes | semi | yes | quick prototype, low order |

```python
import diffrax, jax.numpy as jnp, optimistix as optx

def vector_field(t, y, args):
    # pure-JAX version of GraphSystem residual as explicit rhs
    return graph_rhs_jax(t, y, args)

solver  = diffrax.Kvaerno5()
stepctrl = diffrax.PIDController(rtol=1e-6, atol=1e-9)
root_finder = optx.Newton(rtol=1e-8, atol=1e-10,
                          linear_solver=lineax.GMRES())

sol = diffrax.diffeqsolve(
    diffrax.ODETerm(vector_field),
    solver, t0=0., t1=T, dt0=None,
    y0=y0, args=params,
    stepsize_controller=stepctrl,
    nonlinear_solver=root_finder,
)
```

**CPU without GPU:** `jax.jit` compiles to CPU XLA/LLVM by default. The same code runs on CPU, GPU, TPU — no conditional logic needed. On a 6-core CPU, `jax.vmap` over multiple independent root systems (e.g. parallel plant instances) provides near-linear speedup.

**IMEX splitting** (`KenCarp4`) is the recommended strategy for FSPM: the carbon pool balances (slow, mildly nonlinear) are treated explicitly; xylem/phloem hydraulics (fast, linear or mildly nonlinear when conductances are fixed) are treated implicitly. This avoids the tight step-size constraints that pure explicit methods require for stiff systems.

---

## 4. Architectural proposal

### 4.1 Fixes to existing `GraphSystem` (non-breaking, immediate)

**Fix 1 — Sparse Newton linear step** (2-line change):

```python
# graph_system.py, in the Newton loop — replace line 505:
# BEFORE:
correction = np.linalg.solve(np.asarray(jac, dtype=np.float64), -residual)

# AFTER:
if issparse(jac):
    correction = spsolve(jac.tocsr(), -residual)
else:
    correction = np.linalg.solve(jac, -residual)
```

And in `GraphSystem.jacobian()`, stop calling `matrix.toarray()` when the matrix is sparse (line 462):

```python
# BEFORE:
if issparse(matrix):
    return matrix.toarray()
# AFTER:
if issparse(matrix):
    return matrix  # keep sparse
```

**Fix 2 — Graph-structured FD colouring** (new utility method):

```python
def jac_sparsity_matrix(self) -> csc_matrix:
    """
    CSR/CSC sparsity mask for the Jacobian, derived from graph topology.

    For a system with node unknowns only:
      ∂F_i/∂x_j ≠ 0  iff  i == j  or  (i, j) share a graph edge.
    This corresponds to the adjacency matrix + identity.

    For mixed node+edge unknowns the pattern is the incidence-derived
    block structure [[A_nn, A_ne], [A_en, A_ee]].
    """
    B = self.graph.incidence           # n × m CSC
    A_nn = (B @ B.T).tocsr()           # n × n: i,j share an edge iff A_nn[i,j] > 0
    # add identity (self-coupling)
    A_nn = (A_nn + diags(np.ones(self.graph.n_nodes))).astype(bool).tocsr()
    # TODO: extend to mixed node+edge blocks when edge_unknowns is non-empty
    return A_nn
```

With this mask, replace the FD Jacobian column loop with scipy's `_numdiff.approx_derivative` which accepts a `sparsity` argument and does colouring internally:

```python
from scipy.optimize._numdiff import approx_derivative

def finite_difference_jacobian(self, packed, ...):
    sparsity = self.jac_sparsity_matrix()
    return approx_derivative(
        lambda x: self.residual(x, ...),
        packed, method='3-point', sparsity=sparsity,
    )
```

**Fix 3 — Newton with line search** (Armijo backtracking, ~15 lines):

```python
def _newton_step_with_linesearch(self, packed, residual, jac,
                                  previous_node_fields, dt,
                                  alpha=1.0, c=0.5, rho=0.5, maxback=10):
    if issparse(jac):
        delta = spsolve(jac.tocsr(), -residual)
    else:
        delta = np.linalg.solve(jac, -residual)
    r0_norm = np.linalg.norm(residual)
    step = alpha
    for _ in range(maxback):
        trial = packed + step * delta
        r_trial = self.residual(trial, previous_node_fields, dt)
        if np.linalg.norm(r_trial) <= (1 - c * step) * r0_norm:
            return trial
        step *= rho
    return packed + step * delta   # accept best found
```

### 4.2 Backend abstraction

Add a `SolverBackend` enum and a thin dispatcher in `GraphSystem.solve()`:

```python
# graph_system.py additions

from enum import Enum

class SolverBackend(Enum):
    SCIPY_DIRECT    = "scipy_direct"      # spsolve, for linear systems
    SCIPY_NEWTON    = "scipy_newton"      # scipy.optimize.root, method='krylov' or 'hybr'
    SCIPY_ANDERSON  = "scipy_anderson"    # fixed-point acceleration
    SCIPY_IVP_BDF   = "scipy_ivp_bdf"    # solve_ivp BDF for stiff ODE
    SUNDAE_CVODE    = "sundae_cvode"      # CVODE via scikit-SUNDAE (optional dep)
    SUNDAE_IDA      = "sundae_ida"        # IDA via scikit-SUNDAE (optional dep)
    JAX_OPTIMISTIX  = "jax_optimistix"   # Optimistix Newton/LM (optional dep)
    JAX_DIFFRAX     = "jax_diffrax"      # Diffrax ODE/DAE (optional dep)
    PETSC_SNES      = "petsc_snes"       # PETSc SNES (optional dep)
    PETSC_TS        = "petsc_ts"         # PETSc TS (optional dep)
```

The `SolverSpec` dataclass gains a `backend` field that defaults to `SolverBackend.SCIPY_NEWTON`, and `GraphSystem.solve()` dispatches to the appropriate `_solve_*` private method. Each backend is in a separate submodule (`metafspm.solvers.scipy_backend`, `.sundae_backend`, `.jax_backend`, `.petsc_backend`) imported lazily at call time, so optional dependencies are not required at import.

### 4.3 Decorator additions for time integration

Add `@time_derivative` and `@mass_matrix` to `graph_system_decorators.py`:

```python
def time_derivative(field, scheme="backward_euler"):
    """
    Tag a method as the time-derivative contribution M * (y - y_prev) / dt
    for *field*. Combined with @node_balance blocks this produces a
    backward-Euler or Crank-Nicolson residual automatically.

    scheme: "backward_euler" | "crank_nicolson" | "none" (pure residual, caller handles dt)
    """
    def decorator(func):
        func.__graph_tag__ = {
            "kind": "time_derivative",
            "field": field,
            "scheme": scheme,
        }
        return func
    return decorator


def mass_matrix(field):
    """
    Tag a method returning a per-node capacity coefficient M_i for *field*.
    Used to construct M * (y - y_prev) / dt  in the residual.
    If not provided, M = I (identity).
    """
    def decorator(func):
        func.__graph_tag__ = {"kind": "mass_matrix", "field": field}
        return func
    return decorator
```

Usage example — transient water balance with backward Euler:

```python
@graph_system(node_unknowns=["xylem_pressure"],
              method="newton", backend="jax_optimistix",
              schedule_as="hydraulics")
class _hydraulics_solve:

    @node_balance(field="xylem_pressure")
    def _kirchhoff(self, xylem_pressure, axial_conductance):
        # Kirchhoff flux balance: B K B^T psi = source
        # Returns residual for each node
        B = self._graph_view.incidence
        L = B @ (axial_conductance[:, None] * B.T)  # weighted Laplacian
        return L @ xylem_pressure

    @mass_matrix(field="xylem_pressure")
    def _capacitance(self, hydraulic_capacitance):
        return hydraulic_capacitance

    @time_derivative(field="xylem_pressure", scheme="backward_euler")
    def _storage(self, xylem_pressure, xylem_pressure_prev, hydraulic_capacitance):
        return hydraulic_capacitance * (xylem_pressure - xylem_pressure_prev)

    @boundary_condition(location="node", kind="dirichlet",
                        field="xylem_pressure",
                        types={"organ_type": ["leaf_tip"]})
    def _transpiration_bc(self, transpiration_demand):
        return -transpiration_demand

    @graph_output(name="axial_flux")
    def _flux(self, xylem_pressure, axial_conductance):
        B = self._graph_view.incidence
        return axial_conductance * (B.T @ xylem_pressure)
```

The `_invoke_graph_system` function in the decorators file handles `time_derivative` blocks by folding them into the `node_balance` residual for the relevant field, using `previous_node_fields` and `dt` from `EquationContext`.

### 4.4 JAX residual path

The cleanest way to make residual functions JAX-compatible without forcing model authors to change their code is to provide a **JAX-aware context** that mirrors the NumPy context:

```python
# metafspm/solvers/jax_backend.py

import jax.numpy as jnp
from jax.experimental.sparse import BCOO

def numpy_to_jax_context(ctx: EquationContext) -> EquationContext:
    """
    Return a copy of ctx with all arrays converted to jnp.ndarray.
    Works only if no evaluator uses in-place mutation.
    """
    import dataclasses
    def _to_jax(v):
        if isinstance(v, np.ndarray):
            return jnp.asarray(v)
        return v
    ...

def build_jax_residual(system: GraphSystem):
    """
    Wrap system.residual as a pure JAX function F(y) → r.
    Uses the JAX context; incidence matrix converted to BCOO once.
    """
    B_bcoo = BCOO.from_scipy_sparse(system.graph.incidence)
    # Attach to system for reuse
    system._jax_incidence = B_bcoo

    def F_jax(y, args):
        ctx = system.make_context(np.asarray(y), ...)
        ctx_jax = numpy_to_jax_context(ctx)
        return jnp.concatenate([
            jnp.asarray(block.evaluator(ctx_jax)).ravel()
            for block in system.equation_blocks
        ])

    return F_jax
```

Model authors who want the full JAX path (autodiff, GPU) write their evaluators using `jnp` throughout. Authors who use NumPy still work, but the JAX backend will convert arrays and incur a small overhead. This is the same progressive-adoption pattern as JAX's own `jax.pure_callback`.

---

## 5. Phased implementation roadmap

### Phase 1 — Fix the critical bugs, add scipy backend (1–3 weeks)

- [ ] **Fix sparse Newton linear step** (2-line change in `GraphSystem.solve()`).
- [ ] **Fix Jacobian sparsity**: add `GraphSystem.jac_sparsity_matrix()` from incidence; route FD Jacobian through `approx_derivative(..., sparsity=...)`.
- [ ] **Add Armijo line-search** to the Newton loop (< 20 lines).
- [ ] **Add `scipy.optimize.root` dispatcher**: replace the hand-rolled Newton loop with a thin wrapper; expose `method='krylov'` (JFNK) and `method='anderson'` via `SolverSpec`.
- [ ] **Add `scipy.integrate.solve_ivp` dispatcher** using `jac_sparsity_matrix()` for BDF/Radau.
- [ ] **Add `@time_derivative` and `@mass_matrix` decorators** + handling in `_invoke_graph_system`.
- [ ] **MMS test harness**: Y-with-a-cycle graph (not a tree), analytic steady solution with Kirchhoff + Michaelis-Menten source, verify residual → 0 under grid refinement.
- [ ] **Cross-solver regression**: run same problem through `linear_direct`, `newton`, `scipy_newton(krylov)`. Results must match to `< 10 * tol`.

**Exit criterion:** RhizoDep runs end-to-end through the new framework. A 10⁴-node xylem solve that previously took >10 s now takes <0.5 s (sparse linear step).

### Phase 2 — JAX backend + SUNDIALS bindings (6–10 weeks)

- [ ] **JAX context conversion** utility; `build_jax_residual()` wrapper.
- [ ] **Optimistix Newton solver** as `jax_optimistix` backend for steady-state. Plug `lineax.GMRES()` as the linear solver.
- [ ] **Diffrax `Kvaerno5` / `KenCarp4`** as `jax_diffrax` backend for transient. Wire `@time_derivative` blocks to the ODE right-hand side.
- [ ] **scikit-SUNDAE CVODE/IDA** as `sundae_cvode` / `sundae_ida` backends (optional dep, conda-forge). This is the fallback for users who cannot or do not want JAX.
- [ ] **`pyamg` AMG preconditioner** for the `linear_iterative` path (pure Python, optional dep). Use for large symmetric systems (weighted Laplacian from xylem conductances).
- [ ] **Ensemble / parameter-sweep API** via `jax.vmap` over `solve()`.

**Exit criterion:** 10⁵-node coupled xylem-phloem transient solve completes in < 60 s/simulated-day on a workstation CPU. `jax.grad` through the steady xylem solve computes ∂(leaf water potential)/∂(per-segment conductance) correctly.

### Phase 3 — PETSc DMNetwork backend (3–6 months, optional, for HPC)

- [ ] Implement `petsc_snes` backend: register `DMNetwork`, map `GraphView` vertices/edges to DMNetwork network components, call `SNES.solve()`.
- [ ] Implement `petsc_ts` backend for transient, using `ARKIMEX` time stepper.
- [ ] Use `PCGAMG` or `PCHYPRE` as the Krylov preconditioner via PETSc options.
- [ ] Add MPI partitioning guide: `DMNetworkDistribute` partitions the plant graph across ranks; each rank runs on a plant sub-graph.

**Exit criterion:** 10⁷-segment whole-plant hydraulics solve scales with >50% efficiency from 8 to 64 MPI ranks.

---

## 6. Sparsity colouring details

This is the most impactful immediate optimisation after the sparse linear-step fix, so it deserves elaboration.

The graph-structured Jacobian sparsity for a system with node unknowns is:

```
J[i,j] ≠ 0  iff  i == j  (diagonal, always)
              or  (i,j) is a graph edge  (nearest-neighbour coupling)
              or  (i,j) share a boundary port
```

For a plant graph with average degree d ≈ 2–3, the maximum number of distinct non-zero columns is d+1 ≈ 3–4. **Graph distance-1 colouring** partitions the N columns into d+1 colour groups where no two columns in the same group share a non-zero row. The FD Jacobian then needs 2(d+1) residual evaluations instead of 2N — a 100–1000× speedup.

`scipy.optimize._numdiff.approx_derivative` does this automatically when `sparsity` is a `csr_matrix`. The only work needed in `metafspm` is computing that sparsity pattern from the `GraphView` incidence matrix, which `jac_sparsity_matrix()` above provides in two lines.

---

## 7. What NOT to do

- ❌ **Do not use `bvpy`** as the solver engine. It is a FEniCS wrapper for 2D/3D boundary-value problems in biomechanics. The cell-type interface and the FEniCS dependency are both wrong for 1D-or-non-oriented graph problems.
- ❌ **Do not implement your own BDF.** scipy's BDF with `jac_sparsity` is correct, well-tested, and adequate to 10⁵ d.o.f. SUNDIALS IDA handles 10⁶. Do not reinvent either.
- ❌ **Do not keep `np.linalg.solve` in the Newton loop.** This is a showstopper bug already documented above. It must be fixed before any performance work.
- ❌ **Do not assume the graph is a tree in any solver path.** The incidence matrix `B` is always rectangular (n_nodes × n_edges) and does not factorise as a tree adjacency. Use CSR/CSC sparse operations uniformly.
- ❌ **Do not use `torchdiffeq`.** It has no implicit/stiff solver; is slower than Diffrax on CPU; and adds a PyTorch dependency for no gain.
- ❌ **Do not use FEniCSx, Firedrake, or NGSolve** for graph problems. They target volumetric FEM on 2D/3D meshes. Their setup overhead (mesh, function space, form compilation) is unnecessary for graph-structured residuals that `metafspm` already assembles correctly.
- ❌ **Do not use `diffeqpy` (Julia)** as anything other than an optional experimental backend. A Julia JIT dependency is a distribution problem for OpenAlea users.
- ❌ **Do not expose PETSc `Options` strings in the decorator API.** Keep all backend tuning in a `SolverConfig` dict passed separately from the model equations. Model code should not know what solver it is running on.

---

## 8. Testing and validation strategy

### 8.1 Method of Manufactured Solutions on a cyclic graph

Design a test graph that is **not** a tree: a ring + chords (or a 2D hex grid cropped to a plant-like topology), with at least one cycle. Choose an analytic solution `u*(v)` and back-calculate the source term. Verify convergence order under graph refinement.

### 8.2 Cross-solver agreement

Every test case runs through `scipy_direct`, `scipy_newton(krylov)`, `jax_optimistix`, and (when installed) `sundae_cvode`. Acceptable: `‖y_A − y_B‖∞ < 10 · max(tol_A, tol_B)`. Discrepancies expose Jacobian bugs or BC implementation differences.

### 8.3 Conservation diagnostics

The `@conserved` decorator (proposed) triggers a post-solve audit: `|Σ_i y_i(t) − Σ_i y_i(0) − ∫ source(t') dt'| < ε`. For mass balance (carbon, water) this is a necessary — if not sufficient — correctness check.

### 8.4 Jacobian correctness

After any analytic `@graph_jacobian` is added, compare it against `finite_difference_jacobian` using `np.testing.assert_allclose(J_analytic, J_fd, rtol=1e-5)`. Run this as a unit test, not just a manual check.

### 8.5 Scalability benchmark

Track wall-time for a 10⁴ and 10⁵-node uniform xylem solve as a CI benchmark. A regression of >20% fails the build. This prevents silent performance regressions from numpy/scipy upgrades.

---

## 9. Dependency matrix

| Capability | Dependency | Availability | Phase |
|---|---|---|---|
| Sparse Newton linear step | `scipy.sparse` (already imported) | standard | 1 (bug fix) |
| FD sparsity colouring | `scipy.optimize._numdiff` (scipy internal) | standard | 1 |
| AMG preconditioner | `pyamg` | conda-forge, pip | 1 |
| JFNK / Anderson | `scipy.optimize.root` (already a dep) | standard | 1 |
| BDF/Radau ODE | `scipy.integrate.solve_ivp` | standard | 1 |
| CVODE / IDA | `scikit-sundae` | conda-forge, pip | 2 |
| Optimistix (JAX) | `optimistix`, `equinox`, `jax` | pip | 2 |
| Diffrax (JAX) | `diffrax` | pip | 2 |
| Lineax (JAX) | `lineax` | pip | 2 |
| PETSc SNES/TS + DMNetwork | `petsc4py` | conda-forge | 3 |

All Phase 1 fixes require only existing `scipy` and `numpy` — no new dependencies.

---

## 10. References

- Abhyankar S. et al. (2020). *PETSc DMNetwork: A Library for Scalable Network PDE-Based Multiphysics Simulations.* ACM TOMS 46(1). doi:10.1145/3344587.
- Bauget F. et al. (2023). *A root functional-structural model allows assessment of the effects of water deficit on water and solute transport parameters.* J. Exp. Bot. 74(5):1594–1608.
- Heymans A. et al. (2023). *CPlantBox: a fully coupled modelling platform for the water and carbon fluxes in the soil–plant–atmosphere continuum.* in silico Plants 5(2):diad009.
- Kidger P. (2021–2024). *Diffrax.* docs.kidger.site/diffrax.
- NREL. *scikit-SUNDAE.* scikit-sundae.readthedocs.io.
- Rader H., Lyons D., Kidger P. (2024). *Optimistix: modular optimisation in JAX and Equinox.* arXiv:2402.09983.
- Vassiliev D., Lyons D., Kidger P. (2023). *Lineax: unified linear solves and linear least-squares in JAX and Equinox.* arXiv:2311.17283.
- Rees F. et al. (2025). *Deciphering spatiotemporal patterns of rhizodeposition with a functional-structural root model: RhizoDep.* Plant and Soil. doi:10.1007/s11104-025-07766-z.
