
"""
solver.py
─────────
Solver hierarchy for metafspm graph models.

SolverConfig / SolverSpec       numerical settings

AbstractSolver (abstract)       step_once() interface + shared numerics
  └── ODESolver (abstract)      adaptive time loop, _integrate_step abstract
        └── DAESolver (abstract) BC injection, algebraic recovery, _recover_algebraic abstract
              ├── NewtonSolver              quasi-static  (method="newton"/"newton_fd")
              ├── ImplicitEulerSolver       backward-Euler (method="implicit_euler")
              ├── LinearDirectSolver        Ax=b           (method="linear_direct")
              ├── ScipyRootSolver           scipy.root     (method="scipy_krylov/anderson/hybr")
              └── ScipyIVPSolver            solve_ivp      (method="scipy_ivp_bdf/radau")

Design rationale
────────────────
AbstractSolver   — pure interface, shared numerics (_linear_step, _armijo_linesearch).
ODESolver        — adds the adaptive time loop (solve → SolverResult) and the
                   step-size control utilities.  _integrate_step is abstract so
                   subclasses supply the actual numerical method.
DAESolver        — extends ODESolver for DAE systems (GraphDAESpec):
                   • _update_p   injects BoundaryConditions into p before every f/g call.
                   • _recover_algebraic  recovers y from g(x,y,p)=0 after each accepted step.
                     For solvers that handle x and y jointly in one Newton loop
                     (implicit path), _recover_algebraic is a no-op.
                   • step_once   per-tick Choregrapher interface; calls _integrate_step
                     with the previous-field state for time-derivative terms.
                   • solve       overrides ODESolver.solve() to thread prev_fields
                     through the loop and call _recover_algebraic after each step.
NewtonSolver     — _integrate_step = Newton on R(x,y)=0 (quasi-static, ignores h as dt).
ImplicitEulerSolver — _integrate_step = backward-Euler Newton; h IS the dt.
LinearDirectSolver  — _integrate_step = direct sparse A x = b.
ScipyRootSolver     — _integrate_step = scipy.optimize.root.
ScipyIVPSolver      — _integrate_step = one solve_ivp step from t to t+h;
                      _recover_algebraic = inner Newton for edge algebraics (explicit path).
"""

from __future__ import annotations

from abc         import ABC, abstractmethod
from dataclasses import dataclass, replace as _dc_replace
from typing      import Optional

import numpy as np
from scipy.sparse import eye, issparse
from scipy.sparse.linalg import spsolve


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SolverConfig:
    """Numerical settings, decoupled from system description."""
    method       : str   = "newton"
    max_iter     : int   = 15
    tol          : float = 1e-10
    fd_eps       : float = 1e-8
    prefer_sparse: bool  = True
    linesearch   : bool  = False
    # ODESolver time-loop settings
    rtol         : float = 1e-6
    atol         : float = 1e-9
    max_step     : float = np.inf
    max_steps    : int   = 100_000
    dense_output : bool  = False


# Backward-compatibility alias
SolverSpec = SolverConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Level 1 — AbstractSolver
# ═══════════════════════════════════════════════════════════════════════════════

class AbstractSolver(ABC):
    """
    Minimal solver interface + shared low-level numerics.

    Every solver exposes step_once() for the Choregrapher per-tick pattern.
    Subclasses that own a time loop also implement solve() → SolverResult.

    Shared utilities
    ────────────────
    _linear_step          sparse-aware J δ = −R  (spsolve when J is sparse).
    _armijo_linesearch    Armijo sufficient-decrease backtracking for Newton.
    """

    def __init__(self, config: SolverConfig = None):
        self.config = config or SolverConfig()

    @abstractmethod
    def step_once(self, spec, previous_node_fields=None,
                  dt=None) -> np.ndarray:
        """
        One Choregrapher tick → converged packed unknown vector.

        Parameters
        ----------
        spec                 : GraphDAESpec or ODESystemSpec
        previous_node_fields : dict[str, np.ndarray] | None
        dt                   : float | None
        """
        ...

    def derive_outputs(self, spec, packed: np.ndarray,
                       previous_node_fields=None, dt=None) -> dict:
        """Evaluate output blocks on the converged solution."""
        return spec.derive_outputs(packed, previous_node_fields, dt)

    def solve(self, spec, t_span=None):
        """Full time integration. Override in subclasses that own a time loop."""
        from .system_specs import SolverResult
        raise NotImplementedError(
            f"{type(self).__name__} does not implement a time loop. "
            "Use a DAESolver subclass (e.g. ScipyIVPSolver) for full integration."
        )

    # ── Shared numerics ────────────────────────────────────────────────────────

    def _linear_step(self, jac, residual: np.ndarray) -> np.ndarray:
        """
        Solve J δ = −R, sparse-aware.

        Sparse J → spsolve (O(N log N)).
        Dense J  → numpy.linalg.solve (O(N³)).
        """
        if issparse(jac):
            return np.asarray(spsolve(jac.tocsr(), -residual), dtype=np.float64)
        return np.linalg.solve(np.asarray(jac, dtype=np.float64), -residual)

    def _armijo_linesearch(self, spec, packed: np.ndarray,
                            delta: np.ndarray, r_norm: float,
                            prev_fields, dt,
                            c: float = 0.5, rho: float = 0.5,
                            max_back: int = 10) -> np.ndarray:
        """Armijo sufficient-decrease backtracking."""
        for k in range(max_back):
            step  = rho ** k
            trial = packed + step * delta
            r_new = np.linalg.norm(
                spec.residual(trial, prev_fields, dt), ord=np.inf
            )
            if r_new <= (1.0 - c * step) * r_norm:
                return trial
        return packed + (rho ** (max_back - 1)) * delta


# ═══════════════════════════════════════════════════════════════════════════════
# Level 2 — ODESolver
# ═══════════════════════════════════════════════════════════════════════════════

class ODESolver(AbstractSolver):
    """
    Adds the adaptive time integration loop over AbstractSolver.

    Subclasses implement _integrate_step; ODESolver.solve() calls it
    repeatedly with Hairer & Wanner step-size control.

    Used directly only for pure ODE specs (ODESystemSpec + FieldDataStructure).
    For graph-based DAE specs, see DAESolver which extends this level.
    """

    @property
    def order(self) -> int:
        """Convergence order of the method. Override in concrete subclasses."""
        return 1

    @abstractmethod
    def _integrate_step(self, spec, t: float, x: np.ndarray,
                        h: float, p: dict) -> tuple:
        """
        Advance x from t to t+h.

        Returns (x_new, error_estimate), both shape (n_states,).
        error_estimate approximates the local truncation error, typically
        the difference between a high-order and a low-order embedded solution.
        """
        ...

    def solve(self, spec, t_span=None):
        """
        Adaptive time integration from t_span[0] to t_span[1].

        Accept/reject based on the Hairer & Wanner error norm:
          norm ≤ 1  → accept, grow h.
          norm > 1  → reject, shrink h and retry.
        """
        from .system_specs import SolverResult, ODESystemSpec

        t_span = self._resolve_t_span(spec, t_span)
        t      = float(t_span[0])
        t_end  = float(t_span[1])
        x      = (spec.x0.copy() if isinstance(spec, ODESystemSpec)
                  else spec.pack_unknowns())
        p      = self._current_p(spec, t)
        h      = self._initial_step(spec, p, x)

        ts, xs     = [t], [x.copy()]
        n_steps    = 0
        n_rejected = 0

        while t < t_end and n_steps < self.config.max_steps:
            h = min(h, t_end - t, self.config.max_step)

            x_new, err_est = self._integrate_step(spec, t, x, h, p)
            err_norm       = self._error_norm(x_new, x, err_est)

            if err_norm <= 1.0 or h <= 1e-12:
                t += h
                x  = x_new
                p  = self._current_p(spec, t)
                ts.append(t);  xs.append(x.copy())
                n_steps  += 1
                h = self._grow_step(h, err_norm)
            else:
                n_rejected += 1
                h = self._shrink_step(h, err_norm)
                if h < 1e-12:
                    return SolverResult(
                        t=np.array(ts), x=np.array(xs), success=False,
                        message=f"Step size collapsed at t={t:.4e}.",
                        n_steps=n_steps, n_rejected=n_rejected,
                    )

        return SolverResult(
            t=np.array(ts), x=np.array(xs),
            success=(t >= t_end),
            message="OK" if t >= t_end else "max_steps reached.",
            n_steps=n_steps, n_rejected=n_rejected,
        )

    def step_once(self, spec, previous_node_fields=None, dt=None) -> np.ndarray:
        """
        Per-tick fallback: one step of size dt (or heuristic h₀).
        Lets ODESolver subclasses work in the Choregrapher loop.
        """
        from .system_specs import ODESystemSpec
        x = (spec.x0.copy() if isinstance(spec, ODESystemSpec)
             else spec.pack_unknowns())
        p = self._current_p(spec, 0.0)
        h = dt if dt is not None else self._initial_step(spec, p, x)
        x_new, _ = self._integrate_step(spec, 0.0, x, h, p)
        return x_new

    # ── Step-size control utilities ───────────────────────────────────────────

    def _error_norm(self, x: np.ndarray, x_ref: np.ndarray,
                    err: np.ndarray) -> float:
        """
        Mixed atol/rtol norm (Hairer & Wanner)::

            NORM = sqrt( mean( (err / scale)² ) )
            scale_i = atol + rtol × max(|x_i|, |x_ref_i|)
        """
        scale = (self.config.atol
                 + self.config.rtol * np.maximum(np.abs(x), np.abs(x_ref)))
        return float(np.sqrt(np.mean((err / scale) ** 2)))

    def _initial_step(self, spec, p: dict, x: np.ndarray) -> float:
        """
        Hairer heuristic for h₀::

            d0 = ‖x₀‖,  d1 = ‖f(x₀)‖
            h0 = 1e-6           if d0 or d1 near zero
               = 0.01 × d0/d1   otherwise
        """
        from .system_specs import ODESystemSpec
        if isinstance(spec, ODESystemSpec):
            f0 = spec.eval_f(x, p)
        else:
            f0 = -spec.residual(spec.pack_unknowns(), None, None)
        d0 = float(np.linalg.norm(x))
        d1 = float(np.linalg.norm(f0))
        h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
        return float(np.clip(h0, 1e-12, self.config.max_step))

    def _grow_step(self, h: float, err_norm: float) -> float:
        factor = (min(5.0, 0.9 * err_norm ** (-1.0 / (self.order + 1)))
                  if err_norm > 0 else 5.0)
        return min(h * factor, self.config.max_step)

    def _shrink_step(self, h: float, err_norm: float) -> float:
        factor = max(0.2, 0.9 * err_norm ** (-1.0 / self.order))
        return max(h * factor, 1e-12)

    def _current_p(self, spec, t: float) -> dict:
        """Parameters merged with BoundaryCondition values at time t."""
        from .system_specs import ODESystemSpec
        p  = getattr(spec, "parameters", {}).copy()
        bc = getattr(spec, "boundary_conditions", None)
        if bc is not None:
            p.update(bc.update(t))
        return p

    @staticmethod
    def _resolve_t_span(spec, t_span):
        if t_span is not None:
            return t_span
        ts = getattr(spec, "t_span", None) or spec.parameters.get("t_span")
        if ts is None:
            raise ValueError(
                "ODESolver.solve() requires t_span.  "
                "Pass it directly or set spec.t_span / spec.parameters['t_span']."
            )
        return ts


# ═══════════════════════════════════════════════════════════════════════════════
# Level 3 — DAESolver
# ═══════════════════════════════════════════════════════════════════════════════

class DAESolver(ODESolver):
    """
    Extends ODESolver for graph-based DAE systems (GraphDAESpec).

    Adds three capabilities over ODESolver:

    1. _update_p(spec, t)
       Injects BoundaryConditions into p before every f/g evaluation.
       f and g remain autonomous — they never see t directly.

    2. _recover_algebraic(spec, x, y_prev, p) [abstract]
       Recovers algebraic variables y from g(x,y,p) = 0 after an
       accepted differential step.

       Two paths:
         Explicit path (NewtonSolver/ScipyRootSolver): x is stepped without
           y, then _recover_algebraic solves g=0 to find y.
         Implicit path (ImplicitEulerSolver and others): x and y are solved
           jointly in one Newton loop; _recover_algebraic is a no-op
           (return empty array).

    3. step_once(spec, previous_node_fields, dt)
       Per-tick Choregrapher interface.  Calls _update_p then
       _integrate_step with previous_node_fields threaded through.

    Concrete subclasses implement _integrate_step and _recover_algebraic.
    The _integrate_step signature is extended with prev_fields over ODESolver:
        _integrate_step(spec, t, x, h, p, prev_fields=None) → (x_new, err)
    """

    @abstractmethod
    def _integrate_step(self, spec, t: float, x: np.ndarray,
                        h: float, p: dict,
                        prev_fields=None) -> tuple:
        """
        Advance the packed unknown vector x by one step of size h.

        Parameters
        ----------
        spec       : GraphDAESpec
        t          : current time (used for BC injection; passed for context)
        x          : current packed unknowns [n_node_dof | n_edge_dof]
        h          : step size; serves as dt for time-derivative terms
        p          : parameters (already updated with BCs at time t)
        prev_fields: dict[field_name, np.ndarray] | None
                     Node field values from the previous accepted step.
                     Required by ImplicitEulerSolver; None for quasi-static.

        Returns (x_new, error_estimate), both shape (n_unknowns,).
        """
        ...

    @abstractmethod
    def _recover_algebraic(self, spec, x: np.ndarray,
                            y_prev: np.ndarray, p: dict) -> np.ndarray:
        """
        Recover algebraic variables from g(x, y, p) = 0 given current x.

        For implicit solvers (x and y solved jointly): return empty array.
        For explicit solvers: Newton on the algebraic sub-system.

        Parameters
        ----------
        x      : converged node state
        y_prev : algebraic values from previous step (warm start for Newton)
        p      : current parameters

        Returns y_new (shape n_edge_unknowns,) or empty array.
        """
        ...

    # ── Concrete: BC injection ────────────────────────────────────────────────

    def _update_p(self, spec, t: float) -> dict:
        """
        Return spec.parameters merged with BoundaryConditions.update(t).

        This is the ONLY place where t enters the equation system.
        All evaluators (f, g, node_balance, edge_law) receive p, not t.
        """
        p  = spec.parameters.copy()
        bc = getattr(spec, "boundary_conditions", None)
        if bc is not None:
            p.update(bc.update(t))
        return p

    # ── Per-tick interface ────────────────────────────────────────────────────

    def step_once(self, spec, previous_node_fields=None,
                  dt=None) -> np.ndarray:
        """
        One Choregrapher tick → converged packed unknown vector.

        Calls _update_p at t=0 (time is managed by Choregrapher, not solver),
        then delegates to _integrate_step with previous_node_fields.
        """
        p     = self._update_p(spec, 0.0)
        x     = spec.pack_unknowns()
        h     = dt if dt is not None else self._initial_step(spec, p, x)
        x_new, _ = self._integrate_step(spec, 0.0, x, h, p, previous_node_fields)
        return x_new

    # ── Override solve() to thread prev_fields and call _recover_algebraic ───

    def solve(self, spec, t_span=None):
        """
        Adaptive DAE integration.

        Extends ODESolver.solve() with:
          - _update_p before each step
          - _recover_algebraic after each accepted step
          - prev_fields maintenance across the loop
        """
        from .system_specs import SolverResult

        t_span = self._resolve_t_span(spec, t_span)
        t      = float(t_span[0])
        t_end  = float(t_span[1])
        x      = spec.pack_unknowns()
        p      = self._update_p(spec, t)
        h      = self._initial_step(spec, p, x)

        # Initialise prev_fields from current node state
        node_u, edge_u = spec.unpack_unknowns(x)
        prev_fields    = {fn: node_u[fn].copy() for fn in spec.unknowns.node_fields}
        y_prev         = np.concatenate(
            [edge_u[fn] for fn in spec.unknowns.edge_fields]
        ) if spec.unknowns.edge_fields else np.zeros(0)

        ts, xs     = [t], [x.copy()]
        n_steps    = 0
        n_rejected = 0

        while t < t_end and n_steps < self.config.max_steps:
            h = min(h, t_end - t, self.config.max_step)

            x_new, err_est = self._integrate_step(
                spec, t, x, h, p, prev_fields
            )
            err_norm = self._error_norm(x_new, x, err_est)

            if err_norm <= 1.0 or h <= 1e-12:
                # Accept — recover algebraic at new time
                p_new = self._update_p(spec, t + h)
                y_new = self._recover_algebraic(spec, x_new, y_prev, p_new)

                # Advance prev_fields
                node_u_new, _ = spec.unpack_unknowns(x_new)
                prev_fields   = {fn: node_u_new[fn].copy()
                                 for fn in spec.unknowns.node_fields}
                y_prev = y_new

                t += h
                x  = x_new
                p  = p_new
                ts.append(t);  xs.append(x.copy())
                n_steps  += 1
                h = self._grow_step(h, err_norm)
            else:
                n_rejected += 1
                h = self._shrink_step(h, err_norm)
                if h < 1e-12:
                    return SolverResult(
                        t=np.array(ts), x=np.array(xs), success=False,
                        message=f"Step size collapsed at t={t:.4e}.",
                        n_steps=n_steps, n_rejected=n_rejected,
                    )

        return SolverResult(
            t=np.array(ts), x=np.array(xs),
            success=(t >= t_end),
            message="OK" if t >= t_end else "max_steps reached.",
            n_steps=n_steps, n_rejected=n_rejected,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Level 4 — Concrete DAE solvers
# ═══════════════════════════════════════════════════════════════════════════════

class NewtonSolver(DAESolver):
    """
    Quasi-static Newton–Raphson DAE solver.

    Finds R(x, y) = 0 with no time-derivative term.  Both node (x) and edge
    (y) unknowns are packed into one vector and solved jointly — this is the
    implicit DAE path: _recover_algebraic is a no-op.

    h (step size) is ignored: the solver always finds the static equilibrium
    at the current parameter state p.  When called from the Choregrapher
    per tick, p carries the updated BoundaryConditions for that tick.

    method="newton"    — analytic Jacobian if available, FD otherwise.
    method="newton_fd" — always finite-difference Jacobian.
    """

    @property
    def order(self) -> int:
        return 1

    def _integrate_step(self, spec, t, x, h, p, prev_fields=None):
        packed   = spec.pack_unknowns()
        force_fd = (self.config.method == "newton_fd")

        for _ in range(self.config.max_iter):
            residual = spec.residual(packed, prev_fields, h)
            if np.linalg.norm(residual, ord=np.inf) < self.config.tol:
                return packed, np.zeros_like(packed)

            jac = (spec.finite_difference_jacobian(packed, prev_fields, h)
                   if force_fd else
                   spec.jacobian(packed, prev_fields, h))

            delta = self._linear_step(jac, residual)

            if self.config.linesearch:
                packed = self._armijo_linesearch(
                    spec, packed, delta,
                    np.linalg.norm(residual, ord=np.inf),
                    prev_fields, h,
                )
            else:
                packed = packed + delta

        raise AssertionError(
            f"Newton did not converge in {self.config.max_iter} iterations; "
            f"final ||R||inf = "
            f"{np.linalg.norm(spec.residual(packed, prev_fields, h), ord=np.inf):.3e}"
        )

    def _recover_algebraic(self, spec, x, y_prev, p):
        """No-op: y is already in the converged packed vector from _integrate_step."""
        return np.zeros(0)


class ImplicitEulerSolver(DAESolver):
    """
    Backward-Euler transient DAE solver.

    Augments the spatial residual with (u − u_prev)/h (where h serves as dt)
    and the Jacobian with I/h, then drives the augmented system to zero via
    Newton.  Both node and edge unknowns are solved jointly (implicit path).

    On the first call (prev_fields=None) falls back to NewtonSolver so the
    initial state is the quasi-static equilibrium.

    method="implicit_euler".
    """

    @property
    def order(self) -> int:
        return 1

    def _integrate_step(self, spec, t, x, h, p, prev_fields=None):
        if h is None or h <= 0:
            raise ValueError(
                "ImplicitEulerSolver requires h > 0 (h serves as dt). "
                "Set self.time_step on the model or pass dt to step_once()."
            )
        if prev_fields is None:
            # First call: no history → quasi-static initialisation
            return NewtonSolver(self.config)._integrate_step(
                spec, t, x, h, p, None
            )

        u_prev = spec.pack_unknowns(node_overrides=prev_fields)
        packed = spec.pack_unknowns()
        N      = len(packed)

        for step_i in range(self.config.max_iter):
            R = spec.residual(packed, prev_fields, h) + (packed - u_prev) / h
            # Always take at least one step so we compute the backward-Euler
            # solution rather than just confirming the initial guess.
            if step_i > 0 and np.linalg.norm(R, ord=np.inf) < self.config.tol:
                return packed, np.zeros_like(packed)

            J_s = spec.jacobian(packed, prev_fields, h)
            J   = (J_s + eye(N, format="csr") * (1.0 / h)
                   if issparse(J_s) else
                   np.asarray(J_s, dtype=np.float64) + np.eye(N) / h)

            packed = packed + self._linear_step(J, R)

        raise AssertionError(
            f"ImplicitEuler Newton did not converge in {self.config.max_iter} "
            f"iterations; final ||R||inf = {np.linalg.norm(R, ord=np.inf):.3e}"
        )

    def _recover_algebraic(self, spec, x, y_prev, p):
        """No-op: x and y are solved jointly."""
        return np.zeros(0)


class ExplicitEulerSolver(DAESolver):
    """
    Forward (explicit) Euler time integration.

    Advances node unknowns explicitly::

        x_{n+1} = x_n + h · f(x_n, y_n, p)

    without any linear solve for the node update — just one function
    evaluation per step.  This is the explicit DAE path:

      1. _integrate_step  — recovers y_n algebraically, evaluates the explicit
                            RHS, steps x forward.
      2. _recover_algebraic — Newton on edge sub-system at the new x_{n+1}.

    The explicit RHS is obtained from:
      ``spec.rhs_evaluator``   if set — user-provided ẋ = f(x, p) callable.
      ``-spec.residual[:n_node]``  otherwise — quasi-static proxy that assumes
           the node balance is ``C·ẋ + R_spatial(x,y) = 0`` with C = I.
           Supply ``rhs_evaluator`` (via ``@graph_output`` or equivalent) for
           correct scaling when C ≠ I.

    The error estimate returned is **zero** — ``DAESolver.solve()`` always
    accepts the step and uses the step size h as-is (no adaptivity).
    Use ``ScipyIVPSolver`` when adaptive stepping is needed.

    Not suitable for stiff systems (e.g. coupled fast hydraulics + slow growth).
    Use ``ImplicitEulerSolver`` or ``ScipyIVPSolver`` for stiff problems.

    method="explicit_euler"
    """

    @property
    def order(self) -> int:
        return 1

    def _integrate_step(self, spec, t, x, h, p, prev_fields=None):
        n_node_dof = spec.graph.n_nodes * len(spec.unknowns.node_fields)

        if spec.rhs_evaluator is not None:
            # User supplied explicit ODE form ẋ = f(x, p)
            ctx      = spec.make_context(x, prev_fields, None)
            rhs_node = np.asarray(
                spec.rhs_evaluator(ctx), dtype=np.float64
            ).reshape(-1)[:n_node_dof]
        else:
            # Proxy: ẋ ≈ −R_spatial(x, y)  (assumes unit mass matrix)
            R        = spec.residual(x, prev_fields, None)
            rhs_node = -R[:n_node_dof]

        x_new                = x.copy()
        x_new[:n_node_dof]   = x[:n_node_dof] + h * rhs_node
        # Edge unknowns are stale after the explicit step —
        # _recover_algebraic updates them at the new node state.
        return x_new, np.zeros_like(x_new)   # zero error → fixed step size

    def _recover_algebraic(self, spec, x, y_prev, p):
        """
        Explicit-path algebraic recovery.

        Runs a cheap inner Newton on the edge sub-system to find y such that
        g(x_new, y, p) = 0 at the new node state.
        Returns an empty array when there are no edge unknowns.
        """
        from .system_specs import ODESystemSpec
        if isinstance(spec, ODESystemSpec) or not spec.unknowns.edge_fields:
            return np.zeros(0)
        node_u, _ = spec.unpack_unknowns(x)
        inner      = NewtonSolver(SolverConfig(
            method   = "newton",
            tol      = self.config.tol * 0.1,
            max_iter = self.config.max_iter,
        ))
        packed_conv, _ = inner._integrate_step(
            spec, 0.0,
            spec.pack_unknowns(node_overrides=node_u),
            1.0, p, None,
        )
        _, edge_u = spec.unpack_unknowns(packed_conv)
        return np.concatenate([edge_u[fn] for fn in spec.unknowns.edge_fields])


class LinearDirectSolver(DAESolver):
    """
    One-shot direct sparse/dense solve for linear DAE systems: A x = b.

    Requires spec.matrix_evaluator and spec.rhs_evaluator to be set.
    method="linear_direct".
    """

    @property
    def order(self) -> int:
        return 1

    def _integrate_step(self, spec, t, x, h, p, prev_fields=None):
        packed = spec.pack_unknowns()
        mat    = spec._eval_matrix(packed, prev_fields, h)
        rhs    = spec._eval_rhs(packed, prev_fields, h)
        if issparse(mat):
            if self.config.prefer_sparse:
                result = np.asarray(
                    spsolve(mat.tocsr(), rhs), dtype=np.float64
                ).reshape(-1)
            else:
                result = np.linalg.solve(mat.toarray(), rhs)
        else:
            result = np.linalg.solve(
                np.asarray(mat, dtype=np.float64), rhs
            ).reshape(-1)
        return result, np.zeros_like(result)

    def _recover_algebraic(self, spec, x, y_prev, p):
        return np.zeros(0)


class ScipyRootSolver(DAESolver):
    """
    scipy.optimize.root wrappers for quasi-static DAE.

    method="scipy_krylov"   — Newton-GMRES (matrix-free, best for large N)
    method="scipy_anderson" — Anderson acceleration (weakly coupled systems)
    method="scipy_hybr"     — MINPACK hybrd trust-region Newton
    """

    @property
    def order(self) -> int:
        return 1

    def _integrate_step(self, spec, t, x, h, p, prev_fields=None):
        from scipy.optimize import root as scipy_root
        scipy_method = self.config.method[len("scipy_"):]
        packed0      = spec.pack_unknowns()

        jac_fn = None
        if scipy_method == "hybr" and (
            spec.jacobian_evaluator is not None
            or spec.matrix_evaluator is not None
        ):
            jac_fn = lambda xv: np.asarray(
                spec.jacobian(xv, prev_fields, h), dtype=np.float64
            )

        options = (
            {"maxfev": 200 * (1 + packed0.size)}
            if scipy_method == "hybr"
            else {"maxiter": self.config.max_iter}
        )
        result = scipy_root(
            fun    = lambda xv: spec.residual(xv, prev_fields, h),
            x0     = packed0,
            method = scipy_method,
            jac    = jac_fn,
            tol    = self.config.tol,
            options= options,
        )
        if not result.success:
            actual = np.linalg.norm(
                spec.residual(result.x, prev_fields, h), ord=np.inf
            )
            if actual >= self.config.tol * 1e3:
                raise AssertionError(
                    f"scipy.optimize.root ({scipy_method!r}) failed: "
                    f"{result.message}"
                )
        packed = np.asarray(result.x, dtype=np.float64)
        return packed, np.zeros_like(packed)

    def _recover_algebraic(self, spec, x, y_prev, p):
        return np.zeros(0)


class ScipyIVPSolver(DAESolver):
    """
    Full time integration via scipy.integrate.solve_ivp — a DAESolver
    that also owns a proper adaptive time loop.

    Inherits solve() from DAESolver (which overrides ODESolver.solve() to
    thread prev_fields and call _recover_algebraic).

    _integrate_step
    ───────────────
    Delegates one step [t, t+h] to solve_ivp (BDF or Radau).

    For ODESystemSpec: calls f(x, p) directly.
    For GraphDAESpec:  constructs the ODE rhs by recovering edge algebraics
                       at each evaluation (explicit DAE path):
                         ẏ_node = −R_spatial(y_node, q)  with M = I assumed.

    _recover_algebraic
    ──────────────────
    For GraphDAESpec with edge unknowns: runs an inner Newton solve on the
    edge algebraic sub-system after each accepted node step.
    For ODESystemSpec or node-only specs: returns empty array (no-op).

    method="scipy_ivp_bdf"   — stiff BDF, recommended for FSPM.
    method="scipy_ivp_radau" — stiff Radau, higher order.
    """

    @property
    def order(self) -> int:
        return 4   # effective order for step-size control heuristic

    def _integrate_step(self, spec, t, x, h, p, prev_fields=None):
        from scipy.integrate import solve_ivp
        from .system_specs   import ODESystemSpec

        ivp_method = "BDF" if "bdf" in self.config.method.lower() else "Radau"

        if isinstance(spec, ODESystemSpec):
            result = solve_ivp(
                fun    = lambda _t, _x: spec.eval_f(_x, p),
                t_span = (t, t + h),
                y0     = x,
                method = ivp_method,
                rtol   = self.config.rtol,
                atol   = self.config.atol,
                dense_output = True,
            )
            x_new   = result.y[:, -1]
            x_half  = result.sol(t + h / 2)
            err_est = np.abs(x_new - x_half) * 0.1
            return x_new, err_est

        # GraphDAESpec: node-only ODE form with inner algebraic recovery
        n_node_dof   = spec.graph.n_nodes * len(spec.unknowns.node_fields)
        inner_newton = NewtonSolver(
            SolverConfig(method="newton",
                         tol=self.config.tol * 0.1,
                         max_iter=self.config.max_iter)
        )

        def rhs(_t, y_node):
            node_overrides, cursor = {}, 0
            for fn in spec.unknowns.node_fields:
                w = spec.graph.n_nodes
                node_overrides[fn] = y_node[cursor : cursor + w]
                cursor += w
            if spec.unknowns.edge_fields:
                try:
                    packed_conv = inner_newton._integrate_step(
                        spec, _t, spec.pack_unknowns(), 1.0, p, None
                    )[0]
                    _, edge_u = spec.unpack_unknowns(packed_conv)
                    packed = spec.pack_unknowns(node_overrides=node_overrides,
                                                edge_overrides=edge_u)
                except AssertionError:
                    packed = spec.pack_unknowns(node_overrides=node_overrides)
            else:
                packed = spec.pack_unknowns(node_overrides=node_overrides)
            R = spec.residual(packed, prev_fields, None)
            return -R[:n_node_dof]

        sparsity = spec.jac_sparsity_matrix()
        sp_slice = sparsity[:n_node_dof, :n_node_dof]

        result = solve_ivp(
            fun          = rhs,
            t_span       = (t, t + h),
            y0           = x[:n_node_dof],
            method       = ivp_method,
            rtol         = self.config.rtol,
            atol         = self.config.atol,
            jac_sparsity = sp_slice if sp_slice.nnz > 0 else None,
            dense_output = True,
        )
        x_new   = result.y[:, -1]
        x_half  = result.sol(t + h / 2)
        err_est = np.abs(x_new - x_half) * 0.1
        return x_new, err_est

    def _recover_algebraic(self, spec, x, y_prev, p):
        """
        Explicit-path algebraic recovery: Newton on edge sub-system.

        For node-only specs or ODESystemSpec, returns empty array.
        For GraphDAESpec with edge unknowns, runs a quick Newton to find
        y such that g(x, y, p) = 0.
        """
        from .system_specs import ODESystemSpec
        if isinstance(spec, ODESystemSpec) or not spec.unknowns.edge_fields:
            return np.zeros(0)

        # Recover edge unknowns by Newton on the algebraic residual
        n_node_dof = spec.graph.n_nodes * len(spec.unknowns.node_fields)
        node_u, _ = spec.unpack_unknowns(x)

        inner = NewtonSolver(
            SolverConfig(method="newton",
                         tol=self.config.tol * 0.1,
                         max_iter=self.config.max_iter)
        )
        packed_conv = inner._integrate_step(
            spec, 0.0, spec.pack_unknowns(node_overrides=node_u), 1.0, p, None
        )[0]
        _, edge_u = spec.unpack_unknowns(packed_conv)
        return np.concatenate([edge_u[fn] for fn in spec.unknowns.edge_fields])


# ═══════════════════════════════════════════════════════════════════════════════
# Registry and factory
# ═══════════════════════════════════════════════════════════════════════════════

SOLVER_REGISTRY: dict[str, type] = {
    "explicit_euler"           : ExplicitEulerSolver,
    "newton"                   : NewtonSolver,
    "newton_fd"                : NewtonSolver,
    "newton_optional_jacobian" : NewtonSolver,
    "implicit_euler"           : ImplicitEulerSolver,
    "linear_direct"            : LinearDirectSolver,
    "scipy_krylov"             : ScipyRootSolver,
    "scipy_anderson"           : ScipyRootSolver,
    "scipy_hybr"               : ScipyRootSolver,
    "scipy_ivp_bdf"            : ScipyIVPSolver,
    "scipy_ivp_radau"          : ScipyIVPSolver,
}


def make_solver(method: str, config=None) -> AbstractSolver:
    """
    Factory: method string → concrete solver instance.

    Parameters
    ----------
    method : str     one of SOLVER_REGISTRY keys.
    config : SolverConfig | None
    """
    cls = SOLVER_REGISTRY.get(method)
    if cls is None:
        raise ValueError(
            f"Unknown solver method {method!r}. "
            f"Valid: {sorted(SOLVER_REGISTRY)}."
        )
    cfg = (SolverConfig(method=method)      if config is None else
           _dc_replace(config, method=method) if isinstance(config, SolverConfig) else
           SolverConfig(method=method))
    return cls(cfg)
