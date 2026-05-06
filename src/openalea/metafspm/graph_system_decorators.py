"""
Decorator API for graph-based equation systems in metafspm.

``@graph_system`` is now a **method decorator** placed *inside* the class body.
Python's ``__set_name__`` protocol gives it the class reference when the class
body finishes, so it can register a Choregrapher Functor under the decorated
class's name — exactly like ``@actual`` / ``@rate`` do for scalar methods, but
triggering a whole-graph solve instead of per-vertex dispatch.

Residual and output methods use **named arguments** rather than a context
object.  The framework classifies each argument at assembly time:

- Name matches ``node_unknowns`` → current Newton-iterate slice (node-sized).
- Name matches ``edge_unknowns`` → current Newton-iterate slice (edge-sized).
- Anything else → snapshotted once from ``self.props`` before the Newton loop
  (treated as a static parameter for the duration of the solve).

Scalar model parameters (``self.reflection_xylem``, ``self.time_step``, …)
are accessed directly via ``self``.  Graph topology (incidence matrix,
boundary ports) lives on ``self._graph_view`` and ``self._boundary_ports``.

``@node_balance`` and ``@edge_law`` accept an optional ``types`` dict that
filters the active node/edge subset:

    @node_balance(field="xylem_pressure_in", types={"tissue_type": ["cortex"]})
    def _xylem_cortex(self, xylem_pressure_in, kr_cortex): ...

Multiple blocks for the same field are **summed** (additive accumulation),
so you can split physics cleanly across tissue types or transport pathways.

Public API
----------
``@graph_system(node_unknowns, edge_unknowns, method, …)``
    Method decorator.  Registers a Choregrapher Functor and installs
    ``_invoke_graph_system`` on the class.

``@node_balance(field, types=None)``
    Tags a method as a node residual block for *field*.

``@edge_law(field=None, types=None)``
    Tags a method as an edge residual block; *field* defaults to
    ``edge_unknowns[0]``.

``@graph_jacobian``
    Tags a method as the optional analytic Jacobian (full 2D array).

``@graph_output(name)``
    Tags a method as a post-solve output hook.
"""

import sys
import types
import inspect
import numpy as np
from collections import defaultdict
from dataclasses import fields as dc_fields

from .graph_system import EquationBlock, OutputBlock, GraphSystem, UnknownLayout, SolverSpec, FieldState
from .component_factory import Choregrapher, Functor


# ── Method-level decorators ───────────────────────────────────────────────────

def node_balance(field=None, types=None, explicit=False):
    """Tag a method as a node residual block for *field*.

    Parameters
    ----------
    field : str
        Node-unknown field this block contributes to.
    types : dict[str, list], optional
        Node-property filter, e.g. ``{"tissue_type": ["cortex", "endodermis"]}``.
        The method receives sub-arrays for matching nodes; its returned residual
        is scattered additively back into the full n-node vector.
    explicit : bool
        When ``True`` the method returns the **value** the field should take
        (assignment form) rather than a residual.  The framework generates
        ``R = field_unknown − method(…)`` automatically.  The field name must
        not appear in the method's argument list.
    """
    def decorator(func):
        func.__graph_tag__ = {
            "kind": "node_balance", "field": field,
            "types": types, "explicit": explicit,
        }
        return func
    return decorator


def edge_law(func=None, *, field=None, types=None, explicit=False, integrate=False):
    """Tag a method as an edge residual block.

    Can be used bare (``@edge_law``) or with keyword arguments
    (``@edge_law(field="axial_flux", types={"e_type": ["symplastic"]})``).

    Parameters
    ----------
    field : str, optional
        Edge-unknown field this block contributes to.  Defaults to
        ``edge_unknowns[0]`` at assembly time.
    types : dict[str, list], optional
        Edge-property filter; semantics mirror ``node_balance``.
    explicit : bool
        When ``True`` the method returns the **value** the edge field should
        take (assignment form) rather than a residual.  The framework generates
        ``R = edge_unknown − method(…)`` automatically.  The edge field name
        must not appear in the method's argument list.
    integrate : bool
        When ``True`` the framework writes the converged flux as
        ``props["{field}_mean"]`` after every solve.  This value represents
        the mean flux over the timestep (= converged flux for quasi-static and
        implicit Euler; will be the true trajectory mean in Phase 2 solve_ivp).
        Use ``{field}_mean`` in coupled modules for conservative mass exchange:
        total amount exchanged = ``{field}_mean * dt``.
    """
    def _decorate(f):
        f.__graph_tag__ = {
            "kind": "edge_law", "field": field,
            "types": types, "explicit": explicit, "integrate": integrate,
        }
        return f
    if func is not None:
        return _decorate(func)
    return _decorate


def boundary_condition(location, kind, field=None, types=None, explicit=False):
    """Tag a method as a boundary condition that superimposes on the field's node_balance.

    Parameters
    ----------
    location : str
        ``"node"`` or ``"edge"``.
    kind : str
        ``"dirichlet"`` — overwrites the residual at BC nodes (``result[mask] = bc``).
        ``"neumann"``   — adds a flux contribution at BC nodes (``result[mask] += bc``).
    field : str, optional
        Unknown field this BC applies to.
    types : dict[str, list], optional
        Node/edge-property filter selecting BC-active entities.
    explicit : bool
        Accepted for API symmetry with ``node_balance`` / ``edge_law``.
        Boundary conditions always return values (not residuals), so this flag
        has no behavioral effect.
    """
    def decorator(func):
        func.__graph_tag__ = {
            "kind": "boundary_condition",
            "location": location,
            "bc_kind": kind,
            "field": field,
            "types": types,
            "explicit": explicit,
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


# ── Method descriptor ─────────────────────────────────────────────────────────

class _GraphSystemDescriptor:
    """
    Returned by ``@graph_system``.  Registers with the Choregrapher via
    ``__set_name__`` (called once when the class body finishes) and proxies
    ``_invoke_graph_system`` when accessed on an instance.
    """

    def __init__(self, cls, spec):
        self._inner_cls = cls
        self._spec = spec
        self._attr_name = None

    def __set_name__(self, owner, name):
        self._attr_name = name

        # Build a trampoline carrying the *owner's* module globals so that the
        # Choregrapher's `inheriting` lookup resolves in the right module.
        mod_globals = vars(sys.modules[owner.__module__]) if owner.__module__ in sys.modules else {}

        def _trampoline(self):
            self._invoke_graph_system(name)

        _trampoline = types.FunctionType(
            _trampoline.__code__, mod_globals, name,
            _trampoline.__defaults__, _trampoline.__closure__,
        )
        _trampoline.__qualname__ = f"{owner.__name__}.{name}"
        Choregrapher().add_process(Functor(_trampoline), name=self._spec["schedule_as"])

        # Accumulate specs without mutating an inherited dict.
        # Store the inner class so _invoke_graph_system can find tagged methods.
        spec_with_cls = {**self._spec, "inner_class": self._inner_cls}
        owner._graph_system_specs = {**getattr(owner, "_graph_system_specs", {}), name: spec_with_cls}

        # Install generic lifecycle on owner only if not already defined there.
        if "_invoke_graph_system" not in owner.__dict__:
            owner._invoke_graph_system = _invoke_graph_system

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return lambda: obj._invoke_graph_system(self._attr_name)


# ── Snapshot helpers ──────────────────────────────────────────────────────────

def _declared_locations(instance):
    """Return {field_name: "node" | "edge"} from dataclass metadata."""
    locs = {}
    try:
        for f in dc_fields(type(instance)):
            if "location" in f.metadata:
                locs[f.name] = f.metadata["location"]
    except TypeError:
        pass
    return locs


def _prop_location(name, declared_locs, props, node_set, edge_set):
    """
    Determine whether *name* is a node or edge prop.

    Uses ``declare`` metadata when available; falls back to counting how many
    of the prop dict's keys land in node_set vs edge_set.
    """
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
    """
    Pull only the *required* props into float64 arrays (or lists for string
    props used as type filters).  Node and edge arrays are returned separately.
    """
    props = instance.props
    node_set = set(node_vids_int)
    edge_set = set(edge_vids_int)
    node_snap: dict = {}
    edge_snap: dict = {}

    for name in required_names:
        loc = _prop_location(name, declared_locs, props, node_set, edge_set)
        vids = node_vids_int if loc == "node" else edge_vids_int
        target = node_snap if loc == "node" else edge_snap
        pdict = props.get(name, {})
        try:
            target[name] = np.asarray(
                [float(pdict.get(v, 0.0)) for v in vids], dtype=np.float64
            )
        except (TypeError, ValueError):
            target[name] = [pdict.get(v, None) for v in vids]

    return node_snap, edge_snap


def _type_mask(type_filter, snap, size):
    """Boolean mask over *size* entities from ``{prop: [allowed]}`` filter."""
    mask = np.ones(size, dtype=bool)
    for prop_name, allowed in type_filter.items():
        vals = snap.get(prop_name)
        if vals is None:
            continue
        allowed_set = set(allowed)
        mask &= np.asarray([v in allowed_set for v in vals])
    return mask


# ── Core lifecycle ────────────────────────────────────────────────────────────

def _invoke_graph_system(self, method_name):
    """
    Build, solve, and write back one named graph-system.

    Called by the Choregrapher trampoline each timestep.  Steps:

    1. Collect all ``@node_balance`` / ``@edge_law`` / ``@graph_jacobian`` /
       ``@graph_output`` tagged methods via MRO scan.
    2. Inspect their signatures to discover which props are required.
    3. Snapshot those props once (before the Newton loop).
    4. Wrap each method so the solver can call ``evaluator(ctx)`` while the
       method itself receives named arrays.
    5. Build ``GraphSystem``, solve, and write the solution + outputs back to
       ``self.props``.

    After the call:
    - ``self._last_graph_system`` — the assembled ``GraphSystem`` (test hook).
    - ``self._last_graph_solution`` — the packed solution vector.
    """
    spec = type(self)._graph_system_specs[method_name]
    gv = self._graph_view
    node_unknowns: list[str] = spec["node_unknowns"]
    edge_unknowns: list[str] = spec["edge_unknowns"]

    node_vids = gv.node_ids
    edge_vids = gv.edge_ids
    n = node_vids.size
    m = edge_vids.size
    node_vids_int = [int(v) for v in node_vids]
    edge_vids_int = [int(v) for v in edge_vids]

    declared_locs = _declared_locations(self)

    # ── Collect tagged methods via inner class MRO ────────────────────────────
    inner_cls = spec["inner_class"]
    seen: set[str] = set()
    node_balance_items = []   # (field, types, attr_name, bound, raw, explicit)
    edge_law_items    = []    # (field, types, attr_name, bound, raw)
    bc_items          = []    # (attr_name, bound, raw)
    jacobian_raw      = None  # (bound, raw)
    output_items      = []    # (out_name, bound, raw)

    for cls_ in inner_cls.__mro__:
        for attr_name, obj in cls_.__dict__.items():
            if attr_name in seen:
                continue
            seen.add(attr_name)
            tag = getattr(obj, "__graph_tag__", None)
            if tag is None:
                continue
            bound = obj.__get__(self, type(self))
            kind = tag["kind"]
            if kind == "node_balance":
                node_balance_items.append((tag["field"], tag.get("types"), attr_name, bound, obj, tag.get("explicit", False)))
            elif kind == "edge_law":
                edge_law_items.append((tag.get("field"), tag.get("types"), attr_name, bound, obj, tag.get("explicit", False), tag.get("integrate", False)))
            elif kind == "boundary_condition":
                bc_items.append((tag.get("field"), tag.get("types"), tag.get("bc_kind"), attr_name, bound, obj, tag.get("explicit", False)))
            elif kind == "graph_jacobian":
                jacobian_raw = (bound, obj)
            elif kind == "graph_output":
                output_items.append((tag["name"], bound, obj))

    # Sort node blocks to match node_unknowns order; resolve edge field defaults.
    field_order = {f: i for i, f in enumerate(node_unknowns)}
    node_balance_items.sort(key=lambda x: field_order.get(x[0], len(node_unknowns)))

    default_edge_field = edge_unknowns[0] if edge_unknowns else None
    edge_law_items = [
        (f if f is not None else default_edge_field, tf, an, b, r, ex, intg)
        for f, tf, an, b, r, ex, intg in edge_law_items
    ]
    edge_law_items.sort(key=lambda x: x[2])  # stable by attr_name

    # ── Inspect-first: collect required prop names ────────────────────────────
    all_raws = (
        [r for _, _, _, _, r, _ in node_balance_items]
        + [r for _, _, _, _, r, _, _ in edge_law_items]
        + [r for _, _, _, _, _, r, _ in bc_items]
        + ([jacobian_raw[1]] if jacobian_raw else [])
        + [r for _, _, r in output_items]
    )
    required: set[str] = set()
    for raw in all_raws:
        for aname in inspect.getfullargspec(raw)[0][1:]:  # skip self
            if aname not in node_unknowns and aname not in edge_unknowns:
                required.add(aname)

    # Props used only in type filters (not as method args) must also be snapshotted.
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
        self, required, node_vids_int, edge_vids_int,
        node_unknowns, edge_unknowns, declared_locs,
    )

    # ── Initial-guess FieldStates (unknowns only) ─────────────────────────────
    props = self.props
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

    # ── Evaluator factory ─────────────────────────────────────────────────────

    def make_evaluator(raw_func, bound_method, type_filter, entity):
        """
        Wrap *bound_method* so the Newton loop can call ``evaluator(ctx)``
        while *bound_method* receives named arrays.

        entity = "node"  →  output size n, mask from node_snap
        entity = "edge"  →  output size m, mask from edge_snap
        """
        arg_names = inspect.getfullargspec(raw_func)[0][1:]
        entity_size = n if entity == "node" else m
        mask_snap   = node_snap if entity == "node" else edge_snap

        def evaluator(ctx):
            args = []
            for aname in arg_names:
                if aname in node_unknowns:
                    val = ctx.node_unknowns[aname]
                elif aname in edge_unknowns:
                    val = ctx.edge_unknowns[aname]
                elif aname in node_snap:
                    val = node_snap[aname]
                elif aname in edge_snap:
                    val = edge_snap[aname]
                else:
                    raise KeyError(
                        f"Method '{raw_func.__name__}': arg '{aname}' not found "
                        f"in props.  Node props: {list(node_snap)}  "
                        f"Edge props: {list(edge_snap)}"
                    )
                args.append(val)

            if type_filter:
                mask = _type_mask(type_filter, mask_snap, entity_size)
                sub = [
                    a[mask] if isinstance(a, np.ndarray) and a.shape[0] == entity_size else a
                    for a in args
                ]
                result = np.asarray(bound_method(*sub), dtype=np.float64)
                full   = np.zeros(entity_size, dtype=np.float64)
                np.add.at(full, np.where(mask)[0], result)
                return full
            else:
                return np.asarray(bound_method(*args), dtype=np.float64)

        return evaluator

    # ── Assemble equation_blocks (additive per field, BCs folded in) ──────────
    #
    # Dirichlet BCs overwrite the residual at their nodes; Neumann BCs add to
    # it.  Both are folded into the same per-field EquationBlock so that the
    # residual vector stays n-node-unknowns tall (avoids overdetermination).

    equation_blocks: list[EquationBlock] = []

    def _make_bc_eval(raw_func, bound_method, type_filter, bc_kind="dirichlet", field=None, explicit=False):
        """Return ``(ctx) -> (idx_array, vals_array)`` for a BC method.

        When ``explicit=True`` and ``bc_kind == "dirichlet"``, the method returns
        the **target value** (e.g. ``P_collar``); the framework computes the
        Dirichlet residual as ``unknown[idx] − target``.  For Neumann BCs,
        ``explicit`` has no effect (the method already returns the flux value).
        """
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
                        f"BC method '{raw_func.__name__}': arg '{aname}' not found "
                        f"in props.  Node props: {list(node_snap)}  "
                        f"Edge props: {list(edge_snap)}"
                    )
            if type_filter:
                mask = _type_mask(type_filter, node_snap, n)
                idx = np.where(mask)[0]
                sub = [a[mask] if isinstance(a, np.ndarray) and a.shape[0] == n else a
                       for a in args]
                vals = np.asarray(bound_method(*sub), dtype=np.float64)
            else:
                idx = np.arange(n)
                vals = np.asarray(bound_method(*args), dtype=np.float64)

            if explicit and bc_kind == "dirichlet" and field is not None:
                # Method returned the target; generate residual = unknown − target
                vals = ctx.node_unknowns[field][idx] - vals

            return idx, vals

        return bc_eval

    def _make_combined_node_ev(bulk_evals, bc_specs):
        """Combine bulk evaluators with BC overrides into a single evaluator."""
        def evaluator(ctx):
            result = np.zeros(n, dtype=np.float64)
            for w in bulk_evals:
                result += w(ctx)
            for bkind, bc_ev in bc_specs:
                idx, vals = bc_ev(ctx)
                if bkind == "dirichlet":
                    result[idx] = vals
                else:
                    result[idx] += vals
            return result
        return evaluator

    node_groups: dict[str, list] = defaultdict(list)
    for field, tf, _, bound, raw, ex in node_balance_items:
        node_groups[field].append((tf, bound, raw, ex))

    bc_groups: dict[str, list] = defaultdict(list)
    for field, tf, bc_kind, _, bound, raw, ex in bc_items:
        bc_groups[field].append((tf, bc_kind, bound, raw, ex))

    for field in node_unknowns:
        grp = node_groups.get(field, [])
        bc_grp = bc_groups.get(field, [])
        if not grp and not bc_grp:
            continue
        bulk_wrapped = []
        for tf, b, r, ex in grp:
            inner = make_evaluator(r, b, tf, "node")
            if ex:
                # Explicit form: R = node_unknown − formula(…)
                bulk_wrapped.append(
                    lambda ctx, _f=field, _e=inner: ctx.node_unknowns[_f] - _e(ctx)
                )
            else:
                bulk_wrapped.append(inner)
        bc_wrapped = [
            (bk, _make_bc_eval(r, b, tf, bc_kind=bk, field=field, explicit=ex))
            for tf, bk, b, r, ex in bc_grp
        ]
        equation_blocks.append(EquationBlock(
            name=f"node_balance_{field}",
            evaluator=_make_combined_node_ev(bulk_wrapped, bc_wrapped),
        ))

    edge_groups: dict[str, list] = defaultdict(list)
    for field, tf, _, bound, raw, ex, intg in edge_law_items:
        edge_groups[field].append((tf, bound, raw, ex, intg))

    integrate_edge_fields: set[str] = set()

    for field in edge_unknowns:
        grp = edge_groups.get(field, [])
        if not grp:
            continue
        wrapped = []
        for tf, b, r, ex, intg in grp:
            if intg:
                integrate_edge_fields.add(field)
            inner = make_evaluator(r, b, tf, "edge")
            if ex:
                # Explicit form: R = edge_unknown − formula(…)
                wrapped.append(
                    lambda ctx, _f=field, _e=inner: ctx.edge_unknowns[_f] - _e(ctx)
                )
            else:
                wrapped.append(inner)
        ev = wrapped[0] if len(wrapped) == 1 else (lambda ctx, _w=wrapped: sum(w(ctx) for w in _w))
        equation_blocks.append(EquationBlock(name=f"edge_law_{field}", evaluator=ev))

    output_blocks = tuple(
        OutputBlock(name=oname, evaluator=make_evaluator(raw, bound, None, "node"))
        for oname, bound, raw in output_items
    )

    jac_evaluator = None
    if jacobian_raw is not None:
        jac_evaluator = make_evaluator(jacobian_raw[1], jacobian_raw[0], None, "node")

    # ── Build and solve ────────────────────────────────────────────────────────
    boundary_ports = tuple(getattr(self, "_boundary_ports", None) or ())
    # Advance u_prev: the solution from the last timestep becomes this timestep's
    # starting point.  We do this at the START of the call so that during and
    # after this solve _previous_fields still holds the value from BEFORE this
    # step (the test residual check relies on this).
    _saved_key = f"_gsol_{method_name}"
    _prev_key = f"_gprev_{method_name}"
    _saved = getattr(self, _saved_key, None)
    if _saved is not None:
        setattr(self, _prev_key, _saved)
    previous_fields = getattr(self, _prev_key, None)
    dt = getattr(self, "time_step", None)

    system = GraphSystem(
        graph=gv,
        node_fields=node_fields_gs,
        edge_fields=edge_fields_gs,
        boundary_ports=boundary_ports,
        unknowns=UnknownLayout(
            node_fields=tuple(node_unknowns),
            edge_fields=tuple(edge_unknowns),
        ),
        solver=SolverSpec(
            method=spec["method"],
            max_iter=spec["max_iter"],
            tol=spec["tol"],
            fd_eps=spec["fd_eps"],
            prefer_sparse=spec["prefer_sparse"],
            linesearch=spec["linesearch"],
        ),
        equation_blocks=tuple(equation_blocks),
        output_blocks=output_blocks,
        jacobian_evaluator=jac_evaluator,
        parameters={},
    )

    # Store for test introspection.
    self._last_graph_system = system

    packed = system.solve(previous_node_fields=previous_fields, dt=dt)
    self._last_graph_solution = packed
    outputs = system.derive_outputs(packed, previous_node_fields=previous_fields, dt=dt)

    # Write back unknowns and graph outputs to self.props.
    node_u, edge_u = system.unpack_unknowns(packed)
    # Save the converged solution so the NEXT call can promote it to _previous_fields.
    # We must not overwrite _previous_fields here: the balance evaluators still read
    # self._previous_fields during the remainder of this timestep (e.g. test checks).
    setattr(self, _saved_key, {fn: node_u[fn].copy() for fn in node_unknowns})
    self._graph_solution_fields = {fn: node_u[fn].copy() for fn in node_unknowns}

    for fn in node_unknowns:
        target = self.props.setdefault(fn, {})
        for i, vid in enumerate(node_vids_int):
            target[vid] = float(node_u[fn][i])
    for fn in edge_unknowns:
        target = self.props.setdefault(fn, {})
        for i, vid in enumerate(edge_vids_int):
            target[vid] = float(edge_u[fn][i])

    # integrate=True: write {field}_mean = converged flux (= mean for quasi-static
    # and implicit Euler; Phase 2 solve_ivp will replace this with the trajectory mean).
    # Use {field}_mean in coupled modules: total exchanged = {field}_mean * dt.
    for fn in integrate_edge_fields:
        target = self.props.setdefault(f"{fn}_mean", {})
        for i, vid in enumerate(edge_vids_int):
            target[vid] = float(edge_u[fn][i])

    for oname, arr in outputs.items():
        target = self.props.setdefault(oname, {})
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        if arr.size == n:
            write_vids = node_vids_int
        elif arr.size == m:
            write_vids = edge_vids_int
        else:
            continue
        for i, vid in enumerate(write_vids):
            target[vid] = float(arr[i])


# ── Public class decorator ────────────────────────────────────────────────────

def graph_system(
    node_unknowns,
    edge_unknowns=(),
    method="newton",
    max_iter=15,
    tol=1e-10,
    fd_eps=1e-8,
    prefer_sparse=True,
    linesearch=False,
    schedule_as="axial",
):
    """
    Class decorator that wires an inner class into a ``GraphSystem`` solve.

    Place **inside** the outer class body above the inner class::

        @dataclass
        class MyModel(Model):

            @graph_system(node_unknowns=["pressure"], method="newton",
                          schedule_as="axial")
            class _pressure_solve:
                @node_balance(field="pressure")
                def _balance(self, pressure, K, soil_pressure): ...

    The inner class body holds all ``@node_balance``, ``@edge_law``,
    ``@graph_jacobian``, and ``@graph_output`` methods.  The ``self``
    argument of those methods is bound to the **outer model instance** at
    call time, so ``self.props``, ``self._graph_view``, etc. are all
    directly accessible.

    Parameters
    ----------
    node_unknowns : list[str]
        Ordered node-unknown field names.
    edge_unknowns : list[str]
        Ordered edge-unknown field names.
    method : str
        ``"newton"`` (analytic or FD Jacobian, quasi-static — transport
        equilibrates fully each timestep),
        ``"newton_fd"`` (always FD Jacobian),
        ``"linear_direct"`` (matrix / rhs evaluators),
        ``"implicit_euler"`` (backward-Euler transient: spatial residual
        augmented with ``(u − u_prev) / dt``; requires ``self.time_step``
        to be set; falls back to quasi-static on the first timestep),
        ``"scipy_krylov"`` (Jacobian-free Newton-Krylov),
        ``"scipy_anderson"`` (Anderson acceleration),
        ``"scipy_hybr"`` (MINPACK hybrd trust-region Newton).
    linesearch : bool
        Enable Armijo backtracking in the Newton loop.  Ignored for
        ``scipy_*`` and ``implicit_euler`` methods.
    max_iter, tol, fd_eps, prefer_sparse
        Forwarded to ``SolverSpec``.
    schedule_as : str
        Choregrapher step to register under (e.g. ``"axial"``, ``"actual"``).
    """
    spec = {
        "node_unknowns": list(node_unknowns),
        "edge_unknowns": list(edge_unknowns),
        "method": method,
        "max_iter": max_iter,
        "tol": tol,
        "fd_eps": fd_eps,
        "prefer_sparse": prefer_sparse,
        "linesearch": linesearch,
        "schedule_as": schedule_as,
    }

    def decorator(cls):
        return _GraphSystemDescriptor(cls, spec)

    return decorator
