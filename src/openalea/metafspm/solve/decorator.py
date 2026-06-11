"""
decorator.py
────────────
Public API for metafspm graph-based equation systems.

Replaces graph_system.py + graph_system_decorators.py and builds on:
  data_structure.py   →  GraphView, BoundaryPort, DataStructure hierarchy
  system_specs.py     →  GraphDAESpec, EquationBlock, EquationContext, …
  solver.py           →  make_solver, SOLVER_REGISTRY, SolverConfig

Decorators (unchanged API)
──────────────────────────
  @graph_system(node_unknowns, edge_unknowns, method, …)
      Inner-class descriptor that wires a Choregrapher Functor.

  @node_balance(field, filters=None, explicit=False)
      Tag a method as a node residual block.

  @edge_law(field="flux", filters=None, explicit=False, integrate=False)
      Tag a method as an edge residual block.

  @boundary_condition(location, kind, field=None, filters=None, explicit=False)
      Tag a method as a Dirichlet or Neumann boundary condition.

  @graph_jacobian
      Tag a method as the optional analytic Jacobian.

  @graph_output(name)
      Tag a method as a post-solve output hook.

Re-exports for backward compatibility with code that previously imported
from graph_system or graph_system_decorators:
  GraphView, BoundaryPort, FieldState, UnknownLayout,
  EquationBlock, OutputBlock, EquationContext,
  GraphSystem, SolverSpec, weighted_laplacian
"""

from __future__ import annotations

import sys
import types
import inspect
from collections import defaultdict
from dataclasses import fields as dc_fields
from typing      import Optional

import numpy as np

# ── New module hierarchy ──────────────────────────────────────────────────────
from openalea.metafspm.data_structure.data_api import GraphView, BoundaryPort
from openalea.metafspm.solve.system_specs   import (
    BoundaryConditions,
    FieldState, UnknownLayout,
    EquationBlock, OutputBlock, EquationContext,
    GraphDAESpec, GraphSystem,
    weighted_laplacian,
    SolverResult,
)
from openalea.metafspm.solve.solver         import (
    SolverConfig, SolverSpec, make_solver, SOLVER_REGISTRY, ImplicitEulerSolver,
)

# Canonical method string for each concrete solver class (first key in SOLVER_REGISTRY wins).
_CLASS_TO_METHOD: dict = {}
for _k, _v in SOLVER_REGISTRY.items():
    if _v not in _CLASS_TO_METHOD:
        _CLASS_TO_METHOD[_v] = _k

# Map MPG integer scale constants → generic "node"/"edge" for snapshot routing.
from openalea.metafspm.data_structure.configs import ScalesConfig as _ScalesConfig
_SCALE_INT_TO_LOC: dict = {
    _ScalesConfig.Compartment: "node",
    _ScalesConfig.Connection: "edge",
}

# ── Choregrapher (unchanged) ──────────────────────────────────────────────────
from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.solve.legacy_functor import Functor


# ═══════════════════════════════════════════════════════════════════════════════
# Method-level decorators for Euler steps (public API)
# ═══════════════════════════════════════════════════════════════════════════════

def _step(name: str, *, total: bool = False, iterating: bool = False):
    """Return a decorator that registers func as a Choregrapher step."""
    def decorator(func):
        func.__step_tag__ = {"name": name, "total": total, "iterating": iterating}
        Choregrapher().add_process(Functor(func, total=total, iteraring=iterating), name=name)
        return func
    return decorator

priorbalance     = _step("priorbalance", iterating=True)
selfbalance      = _step("selfbalance", iterating=True)
stepinit         = _step("stepinit", iterating=True)
state            = _step("state")
rate             = _step("rate")
totalrate        = _step("totalrate", total=True)
deficit          = _step("deficit")
totalstate       = _step("totalstate", total=True)
axial            = _step("axial")
potential        = _step("potential")
allocation       = _step("allocation")
actual           = _step("actual")
segmentation     = _step("segmentation")
postsegmentation = _step("postsegmentation")


# ═══════════════════════════════════════════════════════════════════════════════
# Method-level decorators for graph systems  (public API)
# ═══════════════════════════════════════════════════════════════════════════════

def node_balance(field=None, filters=None, explicit=False):
    """
    Tag a method as a node residual block for *field*.

    Parameters
    ----------
    field    : str   node-unknown field this block contributes to.
    filters  : dict  node-property filter, e.g. {"tissue_type": ["cortex"]}.
    explicit : bool  when True the method returns the target value;
                     the framework generates R = unknown − value automatically.
    """
    def decorator(func):
        func.__graph_tag__ = {
            "kind": "node_balance", "field": field,
            "filters": filters, "explicit": explicit,
        }
        return func
    return decorator


def edge_law(func=None, *, field=None, filters=None,
             explicit=False, integrate=False):
    """
    Tag a method as an edge residual block.

    Parameters
    ----------
    field    : str   edge-unknown field this block contributes to.
    filters  : dict  edge-property filter.
    explicit : bool  method returns the value; R = unknown − value generated.
    integrate: bool  write converged flux as props["{field}_mean"] after solve.
    """
    def _decorate(f):
        f.__graph_tag__ = {
            "kind": "edge_law", "field": field,
            "filters": filters, "explicit": explicit, "integrate": integrate,
        }
        return f
    if func is not None:
        return _decorate(func)
    return _decorate


def boundary_condition(location, kind, field=None, filters=None, explicit=False):
    """
    Tag a method as a boundary condition that superimposes on the field's
    node_balance.

    Parameters
    ----------
    location : "node" | "edge"
    kind     : "dirichlet" | "neumann"
    field    : str   unknown field this BC applies to.
    filters  : dict  entity-property filter selecting BC-active entities.
    explicit : bool  accepted for API symmetry; BCs always return values.
    """
    def decorator(func):
        func.__graph_tag__ = {
            "kind": "boundary_condition",
            "location": location, "bc_kind": kind,
            "field": field, "filters": filters, "explicit": explicit,
        }
        return func
    return decorator


def graph_jacobian(func):
    """Tag a method as the optional analytic Jacobian evaluator."""
    func.__graph_tag__ = {"kind": "graph_jacobian"}
    return func


def graph_output(name):
    """Tag a method as a named post-solve output hook."""
    def decorator(func):
        func.__graph_tag__ = {"kind": "graph_output", "name": name}
        return func
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# Snapshot helpers  (unchanged internals)
# ═══════════════════════════════════════════════════════════════════════════════

def _declared_locations(instance):
    """Return {field_name: "node"|"edge"} from dataclass metadata.

    Reads the ``scale`` metadata key (set by declare/state_variable/
    input_variable/parameter).  Integer MPG scale constants (Compartment=9,
    Connection=10) are mapped to "node"/"edge".  Falls back to the legacy
    ``location`` key for backward compatibility.
    """
    locs = {}
    try:
        for f in dc_fields(type(instance)):
            loc = f.metadata.get("scale") or f.metadata.get("location")
            if loc is None:
                continue
            if isinstance(loc, int):
                loc = _SCALE_INT_TO_LOC.get(loc)
            if loc is not None:
                locs[f.name] = loc
    except TypeError:
        pass
    return locs


def _prop_location(name, declared_locs, props, node_set, edge_set):
    if name in declared_locs:
        return declared_locs[name]
    pdict = props.get(name, {})
    if not pdict:
        return "node"
    n_node = sum(1 for k in pdict if k in node_set)
    n_edge = sum(1 for k in pdict if k in edge_set)
    return "node" if n_node >= n_edge else "edge"


def _snapshot(instance, required_names, node_vids_int, edge_vids_int,
              node_unknowns, edge_unknowns, declared_locs):
    """Pull required props into float64 arrays before the Newton loop."""
    props    = instance.props
    node_set = set(node_vids_int)
    edge_set = set(edge_vids_int)
    node_snap: dict = {}
    edge_snap: dict = {}

    for name in required_names:
        loc    = _prop_location(name, declared_locs, props, node_set, edge_set)
        vids   = node_vids_int if loc == "node" else edge_vids_int
        target = node_snap    if loc == "node" else edge_snap
        pdict  = props.get(name, {})
        try:
            target[name] = np.asarray(
                [float(pdict.get(v, 0.0)) for v in vids], dtype=np.float64
            )
        except (TypeError, ValueError):
            target[name] = [pdict.get(v, None) for v in vids]

    return node_snap, edge_snap


def _type_mask(type_filter, snap, size):
    """Boolean mask over *size* entities from {prop: [allowed]} filter."""
    mask = np.ones(size, dtype=bool)
    for prop_name, allowed in type_filter.items():
        vals = snap.get(prop_name)
        if vals is None:
            continue
        allowed_set = set(allowed)
        mask &= np.asarray([v in allowed_set for v in vals])
    return mask


# ═══════════════════════════════════════════════════════════════════════════════
# GraphSystemBuilder
# Replaces the inline assembly logic of _invoke_graph_system with a class
# ═══════════════════════════════════════════════════════════════════════════════

class GraphSystemBuilder:
    """
    Builds a GraphDAESpec from a model instance and a graph_system spec dict,
    drives one solve step, and writes results back.

    Replacing the monolithic _invoke_graph_system function with a class makes
    each concern independently testable and separates build from solve from
    inject_result.
    """

    def __init__(self, instance, spec_def: dict):
        self._instance = instance
        self._spec_def = spec_def

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self) -> tuple[GraphDAESpec, dict, dict]:
        """
        Assemble a GraphDAESpec plus the snapshotted prop arrays.

        Returns
        -------
        (GraphDAESpec, node_snap, edge_snap)
            spec       — ready to pass to a solver.
            node_snap  — snapshotted float64 arrays for node props.
            edge_snap  — snapshotted float64 arrays for edge props.
        """
        instance        = self._instance
        spec_def        = self._spec_def
        gv              = instance._graph_view
        node_unknowns   = spec_def["node_unknowns"]
        edge_unknowns   = spec_def["edge_unknowns"]
        inner_cls       = spec_def["inner_class"]

        n = gv.node_ids.size
        m = gv.edge_ids.size
        node_vids_int = [int(v) for v in gv.node_ids]
        edge_vids_int = [int(v) for v in gv.edge_ids]

        declared_locs = _declared_locations(instance)

        # ── Collect tagged methods via inner class MRO ────────────────────────
        seen:              set[str] = set()
        node_balance_items = []   # (field, types, attr_name, bound, raw, explicit)
        edge_law_items     = []   # (field, types, attr_name, bound, raw, explicit, integrate)
        bc_items           = []   # (field, types, bc_kind, attr_name, bound, raw, explicit)
        jacobian_raw       = None
        output_items       = []   # (out_name, bound, raw)

        for cls_ in inner_cls.__mro__:
            for attr_name, obj in cls_.__dict__.items():
                if attr_name in seen:
                    continue
                seen.add(attr_name)
                tag = getattr(obj, "__graph_tag__", None)
                if tag is None:
                    continue
                bound = obj.__get__(instance, type(instance))
                kind  = tag["kind"]
                if kind == "node_balance":
                    node_balance_items.append((
                        tag["field"], tag.get("filters"), attr_name, bound, obj,
                        tag.get("explicit", False)
                    ))
                elif kind == "edge_law":
                    edge_law_items.append((
                        tag.get("field"), tag.get("filters"), attr_name, bound, obj,
                        tag.get("explicit", False), tag.get("integrate", False)
                    ))
                elif kind == "boundary_condition":
                    bc_items.append((
                        tag.get("field"), tag.get("filters"), tag.get("bc_kind"),
                        attr_name, bound, obj, tag.get("explicit", False)
                    ))
                elif kind == "graph_jacobian":
                    jacobian_raw = (bound, obj)
                elif kind == "graph_output":
                    output_items.append((tag["name"], bound, obj))

        # Sort node blocks to match node_unknowns order
        field_order = {f: i for i, f in enumerate(node_unknowns)}
        node_balance_items.sort(key=lambda x: field_order.get(x[0], len(node_unknowns)))

        for f, _, an, _, _, _, _ in edge_law_items:
            if f is None:
                raise ValueError(
                    f"@edge_law '{an}': field= must be set explicitly. "
                    f"Declared edge_unknowns: {list(edge_unknowns)}."
                )

        # ── integrate=True: extend edge unknowns with {field}_amount ──────────
        # For each edge law with integrate=True, a new DAE unknown Q_e is added:
        #   (Q_new − Q_old) / dt − q_e = 0  →  Q_new = Q_old + q_e · dt
        # Q_old is snapshotted from props before the Newton loop; Q_new is
        # solved together with concentration and flux in the same Newton step.
        integrate_fields = sorted({fn for fn, _, _, _, _, _, intg in edge_law_items if intg})
        dt_inst = float(getattr(instance, "time_step", None) or 1.0)

        amount_olds: dict[str, np.ndarray] = {}
        for fn in integrate_fields:
            pdict = instance.props.get(f"{fn}_amount", {})
            amount_olds[fn] = np.array(
                [float(pdict.get(v, 0.0)) for v in edge_vids_int], dtype=np.float64
            )

        all_edge_unknowns = list(edge_unknowns) + [f"{fn}_amount" for fn in integrate_fields]

        # ── Collect required prop names by signature inspection ───────────────
        all_raws = (
            [r for _, _, _, _, r, _ in node_balance_items]
            + [r for _, _, _, _, r, _, _ in edge_law_items]
            + [r for _, _, _, _, _, r, _ in bc_items]
            + ([jacobian_raw[1]] if jacobian_raw else [])
            + [r for _, _, r in output_items]
        )
        required: set[str] = set()
        for raw in all_raws:
            for aname in inspect.getfullargspec(raw)[0][1:]:
                if aname not in node_unknowns and aname not in all_edge_unknowns:
                    required.add(aname)
        for _, tf, _, _, _, _ in node_balance_items:
            if tf:
                required.update(tf.keys())
        for _, tf, _, _, _, _, _ in edge_law_items:
            if tf:
                required.update(tf.keys())
        for _, tf, _, _, _, _, _ in bc_items:
            if tf:
                required.update(tf.keys())

        node_snap, edge_snap = _snapshot(
            instance, required, node_vids_int, edge_vids_int,
            node_unknowns, edge_unknowns, declared_locs,
        )

        # ── Initial-guess FieldStates ─────────────────────────────────────────
        props = instance.props
        node_fields_gs = {
            fn: FieldState(fn, "node", np.asarray(
                [float(props[fn].get(v, 0.0)) for v in node_vids_int], dtype=np.float64
            ))
            for fn in node_unknowns
        }
        edge_fields_gs = {
            fn: FieldState(fn, "edge", np.asarray(
                [float(props[fn].get(v, 0.0)) for v in edge_vids_int], dtype=np.float64
            ))
            for fn in edge_unknowns
        }
        for fn in integrate_fields:
            edge_fields_gs[f"{fn}_amount"] = FieldState(
                f"{fn}_amount", "edge", amount_olds[fn].copy()
            )

        # ── Evaluator factory ─────────────────────────────────────────────────

        def make_evaluator(raw_func, bound_method, type_filter, entity):
            arg_names   = inspect.getfullargspec(raw_func)[0][1:]
            entity_size = n if entity == "node" else m
            mask_snap   = node_snap if entity == "node" else edge_snap

            def evaluator(ctx):
                args = []
                for aname in arg_names:
                    if aname in node_unknowns:
                        args.append(ctx.node_unknowns[aname])
                    elif aname in edge_unknowns:
                        args.append(ctx.edge_unknowns[aname])
                    elif aname in node_snap:
                        args.append(node_snap[aname])
                    elif aname in edge_snap:
                        args.append(edge_snap[aname])
                    else:
                        raise KeyError(
                            f"Method '{raw_func.__name__}': arg '{aname}' not "
                            f"found in props.  Node: {list(node_snap)}  "
                            f"Edge: {list(edge_snap)}"
                        )
                if type_filter:
                    mask = _type_mask(type_filter, mask_snap, entity_size)
                    sub  = [a[mask] if isinstance(a, np.ndarray)
                            and a.shape[0] == entity_size else a
                            for a in args]
                    result = np.asarray(bound_method(*sub), dtype=np.float64)
                    full   = np.zeros(entity_size, dtype=np.float64)
                    np.add.at(full, np.where(mask)[0], result)
                    return full
                return np.asarray(bound_method(*args), dtype=np.float64)

            return evaluator

        # ── BC factory ────────────────────────────────────────────────────────

        def make_bc_eval(raw_func, bound_method, type_filter,
                          bc_kind="dirichlet", field=None, explicit=False):
            arg_names = inspect.getfullargspec(raw_func)[0][1:]

            def bc_eval(ctx):
                args = []
                for aname in arg_names:
                    if aname in node_unknowns:
                        args.append(ctx.node_unknowns[aname])
                    elif aname in edge_unknowns:
                        args.append(ctx.edge_unknowns[aname])
                    elif aname in node_snap:
                        args.append(node_snap[aname])
                    elif aname in edge_snap:
                        args.append(edge_snap[aname])
                    else:
                        raise KeyError(
                            f"BC '{raw_func.__name__}': arg '{aname}' not found."
                        )
                if type_filter:
                    mask = _type_mask(type_filter, node_snap, n)
                    idx  = np.where(mask)[0]
                    sub  = [a[mask] if isinstance(a, np.ndarray)
                            and a.shape[0] == n else a for a in args]
                    vals = np.asarray(bound_method(*sub), dtype=np.float64)
                else:
                    idx  = np.arange(n)
                    vals = np.asarray(bound_method(*args), dtype=np.float64)

                if explicit and bc_kind == "dirichlet" and field is not None:
                    vals = ctx.node_unknowns[field][idx] - vals
                return idx, vals

            return bc_eval

        # ── Assemble equation blocks ──────────────────────────────────────────

        def make_combined_node_ev(bulk_evals, bc_specs, neumann_scale=1.0):
            def evaluator(ctx):
                result = np.zeros(n, dtype=np.float64)
                for w in bulk_evals:
                    result += w(ctx)
                for bkind, bc_ev in bc_specs:
                    idx, vals = bc_ev(ctx)
                    if bkind == "dirichlet":
                        result[idx] = vals
                    else:
                        result[idx] += vals * neumann_scale
                return result
            return evaluator

        node_groups: dict[str, list] = defaultdict(list)
        for field_name, tf, _, bound, raw, ex in node_balance_items:
            node_groups[field_name].append((tf, bound, raw, ex))

        bc_groups: dict[str, list] = defaultdict(list)
        for field_name, tf, bc_kind, _, bound, raw, ex in bc_items:
            bc_groups[field_name].append((tf, bc_kind, bound, raw, ex))

        equation_blocks = []
        for field_name in node_unknowns:
            grp    = node_groups.get(field_name, [])
            bc_grp = bc_groups.get(field_name, [])
            if not grp and not bc_grp:
                continue
            bulk_wrapped = []
            any_explicit = any(ex for _, _, _, ex in grp)
            for tf, b, r, ex in grp:
                inner = make_evaluator(r, b, tf, "node")
                if ex:
                    bulk_wrapped.append(
                        lambda ctx, _f=field_name, _e=inner:
                            ctx.node_unknowns[_f] - _e(ctx)
                    )
                else:
                    bulk_wrapped.append(inner)
            bc_wrapped = [
                (bk, make_bc_eval(r, b, tf, bc_kind=bk, field=field_name, explicit=ex))
                for tf, bk, b, r, ex in bc_grp
            ]
            neumann_scale = dt_inst if any_explicit else 1.0
            equation_blocks.append(EquationBlock(
                name      = f"node_balance_{field_name}",
                evaluator = make_combined_node_ev(bulk_wrapped, bc_wrapped, neumann_scale),
            ))

        edge_groups: dict[str, list] = defaultdict(list)
        for field_name, tf, _, bound, raw, ex, intg in edge_law_items:
            edge_groups[field_name].append((tf, bound, raw, ex, intg))

        for field_name in edge_unknowns:
            grp = edge_groups.get(field_name, [])
            if not grp:
                continue
            wrapped = []
            for tf, b, r, ex, intg in grp:
                inner = make_evaluator(r, b, tf, "edge")
                if ex:
                    wrapped.append(
                        lambda ctx, _f=field_name, _e=inner:
                            ctx.edge_unknowns[_f] - _e(ctx)
                    )
                else:
                    wrapped.append(inner)
            ev = wrapped[0] if len(wrapped) == 1 else (
                lambda ctx, _w=wrapped: sum(w(ctx) for w in _w)
            )
            equation_blocks.append(EquationBlock(
                name=f"edge_law_{field_name}", evaluator=ev
            ))

        # ── Amount integration equations: (Q_new − Q_old)/dt − q = 0 ─────────
        def _make_amount_ev(flux_field, q_old, dt):
            def evaluator(ctx):
                Q_new = ctx.edge_unknowns[f"{flux_field}_amount"]
                q     = ctx.edge_unknowns[flux_field]
                return (Q_new - q_old) / dt - q
            return evaluator

        for fn in integrate_fields:
            equation_blocks.append(EquationBlock(
                name      = f"edge_amount_{fn}",
                evaluator = _make_amount_ev(fn, amount_olds[fn], dt_inst),
            ))

        output_blocks = tuple(
            OutputBlock(
                name      = oname,
                evaluator = make_evaluator(raw, bound, None, "node"),
            )
            for oname, bound, raw in output_items
        )

        jac_evaluator = (
            make_evaluator(jacobian_raw[1], jacobian_raw[0], None, "node")
            if jacobian_raw else None
        )

        boundary_ports = tuple(getattr(instance, "_boundary_ports", None) or ())

        spec = GraphDAESpec(
            graph             = gv,
            node_fields       = node_fields_gs,
            edge_fields       = edge_fields_gs,
            boundary_ports    = boundary_ports,
            unknowns          = UnknownLayout(
                node_fields   = tuple(node_unknowns),
                edge_fields   = tuple(all_edge_unknowns),
            ),
            equation_blocks   = tuple(equation_blocks),
            output_blocks     = output_blocks,
            jacobian_evaluator= jac_evaluator,
            parameters        = {},
        )
        return spec, node_snap, edge_snap

    # ── Solve + inject ────────────────────────────────────────────────────────

    def build_and_step(self, previous_node_fields=None,
                       dt=None) -> tuple[np.ndarray, dict]:
        """
        Build the spec, run one solve step, return (packed, outputs).
        Called once per Choregrapher tick from _invoke_graph_system.
        """
        spec, _, _ = self.build()
        solver_cls = self._spec_def.get("solver_cls")
        if solver_cls is not None:
            solver = solver_cls(self._spec_cfg())
        else:
            solver = make_solver(self._spec_def["method"], self._spec_cfg())
        packed  = solver.step_once(spec, previous_node_fields, dt)
        outputs = solver.derive_outputs(spec, packed, previous_node_fields, dt)
        return packed, outputs

    def _spec_cfg(self) -> SolverConfig:
        sd = self._spec_def
        return SolverConfig(
            method       = sd["method"],
            max_iter     = sd["max_iter"],
            tol          = sd["tol"],
            fd_eps       = sd["fd_eps"],
            prefer_sparse= sd["prefer_sparse"],
            linesearch   = sd["linesearch"],
        )

    def inject_result(self, packed: np.ndarray, spec: GraphDAESpec) -> None:
        """Write converged solution back to self._instance.props.

        Writes all node unknowns and all edge unknowns (including any
        {field}_amount integration unknowns added by integrate=True).
        """
        instance      = self._instance
        node_unknowns = self._spec_def["node_unknowns"]
        gv            = instance._graph_view
        node_vids_int = [int(v) for v in gv.node_ids]
        edge_vids_int = [int(v) for v in gv.edge_ids]

        node_u, edge_u = spec.unpack_unknowns(packed)

        for fn in node_unknowns:
            target = instance.props.setdefault(fn, {})
            for i, vid in enumerate(node_vids_int):
                target[vid] = float(node_u[fn][i])

        for fn in spec.unknowns.edge_fields:
            target = instance.props.setdefault(fn, {})
            for i, vid in enumerate(edge_vids_int):
                target[vid] = float(edge_u[fn][i])


# ═══════════════════════════════════════════════════════════════════════════════
# Core lifecycle — called by the Choregrapher trampoline each timestep
# ═══════════════════════════════════════════════════════════════════════════════

def _invoke_graph_system(self, method_name: str) -> None:
    """
    Build, solve, and write back one named graph-system.

    This is the refactored version of the old monolithic _invoke_graph_system.
    It now delegates each concern to GraphSystemBuilder:
      build()            → GraphDAESpec construction
      build_and_step()   → spec + solver.step_once() + derive_outputs()
      inject_result()    → write-back to self.props
    """
    spec_def = type(self)._graph_system_specs[method_name]

    # ── Advance previous-field bookkeeping ────────────────────────────────────
    _saved_key = f"_gsol_{method_name}"
    _prev_key  = f"_gprev_{method_name}"
    _saved     = getattr(self, _saved_key, None)
    if _saved is not None:
        setattr(self, _prev_key, _saved)
    previous_fields = getattr(self, _prev_key, None)
    dt = getattr(self, "time_step", None)

    # First implicit_euler call: use current field values as u_prev
    _solver_cls = spec_def.get("solver_cls")
    _is_implicit = (
        spec_def.get("method") == "implicit_euler" or
        (_solver_cls is not None and issubclass(_solver_cls, ImplicitEulerSolver))
    )
    if previous_fields is None and _is_implicit:
        gv   = self._graph_view
        vids = [int(v) for v in gv.node_ids]
        previous_fields = {
            fn: np.asarray(
                [float(self.props[fn].get(v, 0.0)) for v in vids],
                dtype=np.float64,
            )
            for fn in spec_def["node_unknowns"]
        }

    # ── Build + solve ─────────────────────────────────────────────────────────
    builder = GraphSystemBuilder(self, spec_def)
    packed, outputs = builder.build_and_step(
        previous_node_fields=previous_fields,
        dt=dt,
    )

    # ── Store for next tick's previous-field and test introspection ───────────
    spec, _, _ = builder.build()    # re-build for unpack (cheap)
    node_u, _  = spec.unpack_unknowns(packed)

    setattr(self, _saved_key, {fn: node_u[fn].copy()
                                for fn in spec_def["node_unknowns"]})
    self._last_graph_solution = packed
    self._graph_solution_fields = {fn: node_u[fn].copy()
                                    for fn in spec_def["node_unknowns"]}

    # ── Inject results ────────────────────────────────────────────────────────
    builder.inject_result(packed, spec)

    # ── Write biological-scale fields back to the MTG ─────────────────────────
    if hasattr(self, "write_back_to_mtg"):
        self.write_back_to_mtg()

    # ── Write output-block results ─────────────────────────────────────────────
    gv            = self._graph_view
    node_vids_int = [int(v) for v in gv.node_ids]
    edge_vids_int = [int(v) for v in gv.edge_ids]
    n, m = gv.n_nodes, gv.n_edges

    for oname, arr in outputs.items():
        target = self.props.setdefault(oname, {})
        arr    = np.asarray(arr, dtype=np.float64).reshape(-1)
        write_vids = node_vids_int if arr.size == n else edge_vids_int
        for i, vid in enumerate(write_vids):
            target[vid] = float(arr[i])

    # Attach GraphSystem for test introspection (backward compat hook)
    self._last_graph_system = _make_compat_graph_system(self, spec_def, spec)


def _make_compat_graph_system(instance, spec_def, spec: GraphDAESpec):
    """Return a minimal GraphSystem for test introspection."""
    return GraphSystem(
        graph             = spec.graph,
        node_fields       = spec.node_fields,
        edge_fields       = spec.edge_fields,
        boundary_ports    = spec.boundary_ports,
        unknowns          = spec.unknowns,
        solver            = SolverConfig(
            method    = spec_def["method"],
            max_iter  = spec_def["max_iter"],
            tol       = spec_def["tol"],
            fd_eps    = spec_def["fd_eps"],
            prefer_sparse = spec_def["prefer_sparse"],
            linesearch    = spec_def["linesearch"],
        ),
        equation_blocks   = spec.equation_blocks,
        output_blocks     = spec.output_blocks,
        jacobian_evaluator= spec.jacobian_evaluator,
        parameters        = spec.parameters,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Method descriptor  (unchanged protocol)
# ═══════════════════════════════════════════════════════════════════════════════

class _GraphSystemDescriptor:
    """
    Returned by @graph_system.  Registers with the Choregrapher via
    __set_name__ and proxies _invoke_graph_system when called on an instance.
    """

    def __init__(self, cls, spec):
        self._inner_cls = cls
        self._spec      = spec
        self._attr_name = None

    def __set_name__(self, owner, name):
        self._attr_name = name

        mod_globals = (vars(sys.modules[owner.__module__])
                       if owner.__module__ in sys.modules else {})

        def _trampoline(self):
            self._invoke_graph_system(name)

        _trampoline = types.FunctionType(
            _trampoline.__code__, mod_globals, name,
            _trampoline.__defaults__, _trampoline.__closure__,
        )
        _trampoline.__qualname__ = f"{owner.__name__}.{name}"
        Choregrapher().add_process(Functor(_trampoline),
                                   name=self._spec["schedule_as"])

        spec_with_cls = {**self._spec, "inner_class": self._inner_cls, "trampoline": _trampoline}
        owner._graph_system_specs = {
            **getattr(owner, "_graph_system_specs", {}),
            name: spec_with_cls,
        }

        if "_invoke_graph_system" not in owner.__dict__:
            owner._invoke_graph_system = _invoke_graph_system

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return lambda: obj._invoke_graph_system(self._attr_name)


# ═══════════════════════════════════════════════════════════════════════════════
# Public class decorator
# ═══════════════════════════════════════════════════════════════════════════════

def graph_system(
    node_unknowns,
    edge_unknowns  = (),
    solver         = "newton",
    method         = None,
    max_iter       = 15,
    tol            = 1e-10,
    fd_eps         = 1e-8,
    prefer_sparse  = True,
    linesearch     = False,
    schedule_as    = "axial",
):
    """
    Inner-class decorator that wires a GraphSystem solve into the Choregrapher.

    Usage::

        from openalea.metafspm.solve.solver import NewtonSolver

        @dataclass
        class MyModel(FunctionalComponent):

            @graph_system(node_unknowns=["pressure"], solver=NewtonSolver,
                          schedule_as="axial")
            class _pressure_solve:

                @node_balance(field="pressure")
                def _balance(self, pressure, K, soil_pressure): ...

                @edge_law
                def _darcy(self, pressure, K): ...

    Parameters
    ----------
    node_unknowns  : list[str]
        Ordered node-unknown field names.
    edge_unknowns  : list[str]
        Ordered edge-unknown field names.
    solver         : str | type
        Either a solver-registry key (e.g. ``"newton"``, ``"newton_fd"``,
        ``"implicit_euler"``) for backward compatibility, or an uninstantiated
        ``DAESolver`` subclass (e.g. ``NewtonSolver``, ``ImplicitEulerSolver``).
        Passing the class directly is preferred: it is transparent and
        inspectable without registry look-up.
    method         : str | None
        Deprecated alias for *solver*.  If supplied alongside *solver*, raises
        ``TypeError``.
    max_iter       : int     Newton / nonlinear iteration cap.
    tol            : float   convergence criterion (||R||_inf).
    fd_eps         : float   finite-difference step for FD Jacobian.
    prefer_sparse  : bool    use sparse linear solves when available.
    linesearch     : bool    Armijo backtracking in Newton loop.
    schedule_as    : str     Choregrapher step name.
    """
    # Backward-compat: honour deprecated method= kwarg.
    if method is not None:
        if solver != "newton":
            raise TypeError(
                "graph_system() received both 'solver' and 'method'.  "
                "Use 'solver' only; 'method' is a deprecated alias."
            )
        solver = method

    # Resolve to (solver_cls, method_str) pair.
    if isinstance(solver, str):
        solver_cls = SOLVER_REGISTRY.get(solver)
        if solver_cls is None:
            raise ValueError(
                f"Unknown solver {solver!r}.  "
                f"Valid keys: {sorted(SOLVER_REGISTRY)}.  "
                "Alternatively, pass an uninstantiated DAESolver subclass."
            )
        method_str = solver
    elif isinstance(solver, type):
        solver_cls = solver
        method_str = _CLASS_TO_METHOD.get(solver_cls, solver_cls.__name__.lower())
    else:
        raise TypeError(
            f"graph_system() solver must be a str or an uninstantiated "
            f"DAESolver subclass, got {type(solver).__name__!r}."
        )

    spec = {
        "node_unknowns" : list(node_unknowns),
        "edge_unknowns" : list(edge_unknowns),
        "solver_cls"    : solver_cls,
        "method"        : method_str,
        "max_iter"      : max_iter,
        "tol"           : tol,
        "fd_eps"        : fd_eps,
        "prefer_sparse" : prefer_sparse,
        "linesearch"    : linesearch,
        "schedule_as"   : schedule_as,
    }

    def decorator(cls):
        return _GraphSystemDescriptor(cls, spec)

    return decorator
