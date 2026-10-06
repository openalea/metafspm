"""
Steps and graph systems: the decorators placing a component's methods in the schedule, and the declaration of
coupled equations over a DataStructure's graph.

::

    @rate, @state, @totalrate, @totalstate, @stepinit, @deficit, @axial,      a step, by its row in the schedule
    @potential, @allocation, @actual, @segmentation, @postsegmentation, ...

    @graph_system(node_unknowns, edge_unknowns=(), solver="newton", ...)    an inner class of equations, solved
                                                                            at its schedule_as row
        @node_balance(field)                     a node residual (time term written; Newton family)
        @node_rate(field)                        a node du/dt (time term by the framework; any solver)
        @edge_law(field=, integrate=False)       an edge residual (or an integrated edge flux)
        @pool_balance(field)                     the residual of a pool unknown (pool_unknowns=)
        name = boundary_set(filters=, kind=, value=, weight=)  a boundary condition on a set of nodes
        @boundary_condition(location, kind)      a boundary condition written as a method
        @graph_jacobian                          an optional analytic Jacobian
        @graph_output(name, location=None)       a variable computed after the solve
"""

from __future__ import annotations

import sys
import types
import inspect
from collections import defaultdict
import warnings
from dataclasses import fields as dc_fields
from typing      import Optional

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

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
    SolverConfig, SolverSpec, make_solver, SOLVER_REGISTRY, ImplicitEulerSolver, ExplicitEulerSolver, ScipyIVPSolver, NewtonSolver,
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
from openalea.metafspm.solve.functor import Functor


# ═══════════════════════════════════════════════════════════════════════════════
# Method-level decorators for Euler steps (public API)
# ═══════════════════════════════════════════════════════════════════════════════

def _step(name: str, *, total: bool = False, iterating: bool = False):
    """
    Return a decorator that registers func as a Choregrapher step.

    Usable bare (``@rate``) or with options (``@rate(vectorized=False)``). On DataStructure-backed components,
    step functions receive whole arrays; ``vectorized=False`` opts in to one call per element
    for functions written with scalar logic.
    """
    def decorator(func=None, *, vectorized: bool = True, location: str = None, locations: dict = None,
                  filters=None, include_inactive: bool = False):
        """
        location:  of the step's output when it is not a declared field ("node", "edge", "scalar", "cell", or a
                   coarse scale name); locations: {output name: location} for the supplementary outputs.
        filters:   the entities the step computes on ({variable: condition}, a mask name or a callable, see
                   Filters); the others keep their values.
        include_inactive:
                   the step also computes on the entities outside the DataStructure's "active" mask (emergence,
                   dead tissues), which it skips by default.
        """
        if func is None:
            return lambda f: decorator(f, vectorized=vectorized, location=location, locations=locations,
                                       filters=filters, include_inactive=include_inactive)
        check_filters(filters, f"@{name} {func.__name__}")
        func.__filters__ = filters
        func.__include_inactive__ = include_inactive
        func.__step_tag__ = {"name": name, "total": total, "iterating": iterating, "vectorized": vectorized}
        func.__vectorized__ = vectorized
        func.__output_locations__ = dict(locations or {})
        if location is not None:
            func.__output_locations__[func.__name__[1:]] = location
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
    filters  : the nodes the block applies on ({variable: condition}, a mask name or a callable, see Filters), e.g.
               {"tissue": CORTEX}; the method's node arguments are sliced to them, and it contributes 0 elsewhere.
               Values the method reads through self (self.previous(), self.data_structure) are not sliced.
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


def node_rate(field=None, filters=None):
    """
    Tag a method as the rate form of a node unknown's balance: it returns du/dt (sources minus the divergence of the
    fluxes, divided by the capacities), with no time term. The framework writes the time term for the solver
    chosen: (u - previous(u)) / dt - rate with the Newton family (backward Euler, also with integrate="substeps" or
    "adaptive"), u += dt * rate with explicit_euler, du/dt = rate with the scipy IVP solvers. The node unknowns of
    one graph system all use one form: @node_rate, or @node_balance (the residual form, Newton family only).

    Parameters
    ----------
    field   : node unknown the rate is of.
    filters : the nodes, as for node_balance: the rate applies on the selected nodes (zero elsewhere).
    """
    def decorator(func):
        func.__graph_tag__ = {"kind": "node_rate", "field": field, "filters": filters}
        return func
    return decorator


def pool_balance(field):
    """
    Tag a method as the residual of pool unknown *field*: it returns one value per pool entity (e.g. per plant),
    and takes the pool's values as an argument named after it, like any unknown.
    """
    def decorator(func):
        func.__graph_tag__ = {"kind": "pool_balance", "field": field}
        return func
    return decorator


def edge_law(func=None, *, field=None, filters=None,
             explicit=False, integrate=False):
    """
    Tag a method as an edge residual block.

    Parameters
    ----------
    field    : str   edge-unknown field this block contributes to.
    filters  : the edges the law applies on ({variable: condition} on edge variables, a mask name or a callable,
               see Filters).
    explicit : bool  method returns the value; R = unknown − value generated.
    integrate: bool  add the integrated amount "{field}_amount" (Q_new = Q_old + q dt) as an edge unknown.
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
    Tag a method as a boundary condition on a set of nodes, for conditions given by an equation: the method takes
    its arguments by name, like node_balance (unknowns, and DataStructure variables, e.g. coupled ones, read at
    each solve), node-located ones sliced to the selected nodes. For a condition given by a variable or a constant,
    a boundary_set is enough::

        @boundary_condition("node", "neumann", field="concentration", filters={"is_collar": ">0"})
        def _collar_uptake(self, concentration, soil_concentration, uptake_rate):
            return uptake_rate * (soil_concentration - concentration)          # an inflow

    Parameters
    ----------
    location : "node"
    kind     : "dirichlet": the selected nodes' residual rows become the method's values (a residual, e.g.
               ``p - collar_pressure``; with explicit=True, the prescribed value itself);
               "neumann": the method's values are an inflow, subtracted from the residual, as boundary_set's.
    field    : the node unknown it applies to.
    filters  : the nodes ({variable: condition}, a mask name or a callable, see Filters). Default: every node.
    explicit : for dirichlet, the method returns the prescribed value instead of a residual.
    """
    if location == "edge":
        raise NotImplementedError("@boundary_condition(location='edge') is not supported yet: conditions are applied "
                                  "on nodes (use a boundary_set on the nodes)")
    if location != "node":
        raise ValueError(f"@boundary_condition: location must be 'node', got '{location}'")
    if kind not in ("dirichlet", "neumann"):
        raise ValueError(f"@boundary_condition: kind must be 'dirichlet' or 'neumann', got '{kind}'")
    check_filters(filters, "@boundary_condition")

    def decorator(func):
        func.__graph_tag__ = {
            "kind": "boundary_condition",
            "location": location, "bc_kind": kind,
            "field": field, "filters": filters, "explicit": explicit,
        }
        return func
    return decorator


def check_filters(filters, owner: str) -> None:
    """Refuse a filters= that is not a {variable: condition} dict, a mask name or a callable."""
    if filters is None or callable(filters) or isinstance(filters, str) or (isinstance(filters, dict) and filters):
        return
    raise TypeError(f"{owner}: filters must be a {{variable: condition}} dict, a mask name or a callable, "
                    f"got {filters!r}")


class Filters:
    """
    The elements a decorator operates on, from its filters= argument::

        {variable: condition, ...}   every pair holding: a value, a list of values, or a comparison (">0", "<=0.03");
                                     label names are resolved, and a variable of a coarser scale is read at each
                                     element's entity of that scale
        "mask name"                  a mask of the DataStructure (ds.define_mask), e.g. shared with a mapping
        callable                     ds -> boolean array, e.g. a geometric selection

    Resolved to a mask of the DataStructure (at the elements' location), recomputed when its variables are written
    or the topology changes.
    """

    def __init__(self, rule, key: str, location: str = "node"):
        check_filters(rule, key)
        self.rule, self.key, self.location = rule, key, location

    def mask_name(self, ds) -> str:
        if isinstance(self.rule, str):
            if not ds.has_mask(self.rule):
                raise KeyError(f"{self.key}: filters='{self.rule}' names no mask of the DataStructure "
                               f"(masks: {ds.masks()}); give {{variable: condition}} for a selection by variables")
            return self.rule
        name, location = f"__filters:{self.key}", _entity_location(ds, self.location)
        spec = ds.__dict__.get("_masks", {}).get(name)
        if spec is None or spec["rule"] is not self.rule or spec["location"] != location:
            ds.define_mask(name, self.rule, location=location)
        return name

    def mask(self, ds, take=None) -> np.ndarray:
        values = np.asarray(ds.mask(self.mask_name(ds)), dtype=bool).reshape(-1)
        return values if take is None else values[take]

    def members(self, ds, take=None) -> np.ndarray:
        return np.flatnonzero(self.mask(ds, take))


class boundary_set:
    """
    Boundary condition on a set of nodes, declared in a graph-system class and assembled by the framework::

        leaves = boundary_set(filters={"label": "LeafElement"}, kind="robin", value="air_water_potential",
                              weight="leaf_conductance")

    Arguments::

        filters {variable: condition} | a mask name | a callable ds -> boolean mask (see Filters); membership
                follows the selecting variables and topology changes.
        kind    "robin":     + w * (x - v) in the field's residual (an outflow towards the external value v);
                "dirichlet": the residual row becomes x - v;
                "neumann":   - v in the residual (v is an inflow);
                None:        a selection only (no term), e.g. the nodes exchanging with a pool unknown.
        kinds   instead of kind, a node variable name: each node's kind is read at each solve from that variable,
                as boundary_set.CODES (1 dirichlet, 2 neumann, 3 robin, anything else no condition), e.g. a collar
                switching between a pressure and a flux.
        value, weight
                node variables of the DataStructure (read at each solve) or constants.
        field   the node unknown it applies to; default: the only node unknown.
    """

    KINDS = ("robin", "dirichlet", "neumann")
    # Codes of a per-node kind variable; other values (e.g. 0) apply no condition
    CODES = {"dirichlet": 1, "neumann": 2, "robin": 3}

    def __init__(self, filters, kind=None, value=0., weight=1.0, field=None, kinds: str = None):
        if kind is not None and kind not in self.KINDS:
            raise ValueError(f"boundary_set: kind must be one of {self.KINDS}, got '{kind}'")
        if kinds is not None:
            if kind is not None:
                raise ValueError("boundary_set: give kind= (one kind) or kinds= (a node variable), not both")
            kind = "per_node"
        self.kind_variable = kinds
        if filters is None:
            raise TypeError("boundary_set: filters= is required (the nodes of the set)")
        check_filters(filters, "boundary_set")
        self.filters, self.kind, self.value, self.weight, self.field = filters, kind, value, weight, field
        self.name = None
        self.__graph_tag__ = {"kind": "boundary_set"}

    def __set_name__(self, owner, name):
        self.name = name

    def variables(self) -> list:
        """DataStructure variables read by the solve (value and weight given by name)."""
        weight = self.weight if self.kind in ("robin", "per_node") else None
        return [x for x in (self.value, weight, self.kind_variable) if isinstance(x, str)]

    def members(self, instance, ds, size, take) -> np.ndarray:
        """Indices, in the solved graph, of the nodes of the set."""
        return Filters(self.filters, f"{type(instance).__name__}.{self.name}").members(ds, take)


def graph_jacobian(func):
    """Tag a method as the optional analytic Jacobian evaluator."""
    func.__graph_tag__ = {"kind": "graph_jacobian"}
    return func


def graph_output(name, location: str = None, filters=None):
    """
    Tag a method as a named post-solve output hook. *location* ("node" or "edge") is required when *name* is not a
    declared field and its size does not identify a single location.

    filters: the nodes the output is computed on ({variable: condition}, a mask name or a callable, see Filters);
    the method's node arguments are sliced to them, and the other nodes get 0. Node outputs only.
    """
    if location not in (None, "node", "edge"):
        raise ValueError(f"@graph_output('{name}'): location must be 'node' or 'edge', got '{location}'")
    if filters is not None and location == "edge":
        raise ValueError(f"@graph_output('{name}'): filters= chooses nodes, not edges")
    check_filters(filters, f"@graph_output('{name}')")

    def decorator(func):
        func.__graph_tag__ = {"kind": "graph_output", "name": name, "location": location, "filters": filters}
        return func
    return decorator


def infer_output_location(owner: str, name: str, shape: tuple, shapes: dict) -> str:
    """
    Location of an undeclared output that gives none, from its shape: accepted, with a DeprecationWarning, only
    when exactly one of *shapes* ({location: shape}) matches; ambiguous or unmatched shapes raise.
    """
    matches = [location for location, location_shape in shapes.items() if tuple(location_shape) == tuple(shape)]
    if len(matches) == 1:
        warnings.warn(f"{owner}: output '{name}' has no declared location, '{matches[0]}' was inferred from its "
                      "shape; declare the field or give location=", DeprecationWarning, stacklevel=3)
        return matches[0]
    if not matches:
        raise ValueError(f"{owner}: output '{name}' of shape {tuple(shape)} matches no location {shapes}")
    raise ValueError(f"{owner}: output '{name}' of shape {tuple(shape)} matches several locations {matches}: "
                     "declare the field or give location=")


# ═══════════════════════════════════════════════════════════════════════════════
# Snapshot helpers  (unchanged internals)
# ═══════════════════════════════════════════════════════════════════════════════

def _declared_locations(instance):
    """{field name: location} of the instance's declared DataStructure variables (resolve_declaration)."""
    specs = getattr(instance, "_variable_specs", None)
    if specs is None:
        ds = getattr(instance, "data_structure", None)
        if ds is None:
            return {}
        from openalea.metafspm.coupling.declaration import declared_specs
        specs = declared_specs(instance, ds)
    return {name: spec.location for name, spec in specs.items()}


def _live_ds(instance):
    """The instance's DataStructure (graph systems read and write it live)."""
    ds = getattr(instance, "data_structure", None)
    if ds is None or not (hasattr(ds, "register") and hasattr(ds, "get")):
        raise TypeError(f"{type(instance).__name__}: graph systems need a DataStructure with a variable store "
                        "(props-based components were removed, see docs/migration.md)")
    return ds


def _read_array(ds, name, location, size, owner=None, take=None, read_only=False):
    """
    Copy of variable *name* as a per-*location* array: scalars are broadcast; a missing name raises.
    *take*: indices of the entities of an active subgraph (filters=), the others being left out.
    *read_only*: a read-only view instead of a copy when possible (parameters and inputs).
    """
    if not ds.has(name):
        raise KeyError(f"{owner + ': ' if owner else ''}'{name}' is used by a graph system but is not registered on "
                       f"the DataStructure (declare it on the component, or register it). Registered: "
                       f"{sorted(ds.available_vars())}")
    values = ds.get(name)
    if values.dtype == object:
        raise TypeError(f"{owner + ': ' if owner else ''}'{name}' holds objects, which graph systems cannot use")
    if values.ndim == 0:
        return np.full(size, float(values))
    values = np.asarray(values).reshape(-1) if values.ndim > 1 else values   # grid cells, in flat C order
    if take is not None:
        values = values[take]
    if read_only and values.dtype == np.float64:
        view = values.view()
        view.flags.writeable = False
        return view
    return np.array(values, dtype=np.float64)


def _entity_location(ds, location):
    """The DataStructure location of the graph's nodes or edges: grid cells play the nodes."""
    return "cell" if location == "node" and "node" not in ds._var_stores() and "cell" in ds._var_stores() else location


def _in_equation(instance, method, args):
    """Call an equation with the instance flagged, so that self.<parameter> is refused inside it."""
    previous = instance.__dict__.get("_in_equation", False)
    instance.__dict__["_in_equation"] = True
    try:
        return method(*args)
    finally:
        instance.__dict__["_in_equation"] = previous


def _take(instance, location):
    """Indices of the active subgraph's entities at *location* during a filtered solve, else None."""
    restriction = instance.__dict__.get("_restriction")
    if restriction is None:
        return None
    return restriction.node_idx if location == "node" else restriction.edge_idx


class _Restriction:
    """
    Active subgraph of a graph system solved with filters=, or one
    connected piece of it (split="components"): the selected nodes, the edges with both ends selected, and the
    corresponding GraphView. A piece leaves the other edges alone (drops_edges=False).
    """

    def __init__(self, view, node_idx, edge_idx, n, m, drops_edges: bool = True):
        self.view, self.node_idx, self.edge_idx = view, node_idx, edge_idx
        self.n, self.m = n, m
        self.drops_edges = drops_edges
        self._dropped = None

    @property
    def dropped_edges(self) -> np.ndarray:
        if self._dropped is None:
            kept = np.zeros(self.m, dtype=bool)
            kept[self.edge_idx] = True
            self._dropped = np.flatnonzero(~kept) if self.drops_edges else np.empty(0, dtype=np.int64)
        return self._dropped

    def scatter(self, ds, name, values, location, dropped=None):
        """
        Write *values* of the subgraph's entities into the full variable, in place (O(subgraph)); dropped edges get
        *dropped* if given.
        """
        full = ds.get(name)
        in_place = full.dtype == np.float64 and full.flags.writeable and full.flags.c_contiguous
        flat = full.reshape(-1) if in_place else np.array(full, dtype=np.float64).reshape(-1)
        if location in ("node", "cell"):
            flat[self.node_idx] = values
        else:
            flat[self.edge_idx] = values
            if dropped is not None and self.dropped_edges.size:
                flat[self.dropped_edges] = dropped
        if in_place:
            ds.mark_written(name)
        else:
            ds.set(name, flat.reshape(full.shape))


def _pieces_of(instance, base) -> list:
    """
    The connected pieces of *base* (a _Restriction, or None for the whole graph) as piece restrictions, computed once
    per topology and subgraph; each with its local GraphView built from its own edges.
    """
    from scipy.sparse.csgraph import connected_components
    ds = _live_ds(instance)
    full = instance._graph_view
    key = (ds.topology_version, id(base))
    cache = instance.__dict__.get("_pieces_cache")
    if cache is not None and cache[0] == key:
        return cache[1]
    view = base.view if base is not None else full
    nodes = base.node_idx if base is not None else np.arange(full.n_nodes)
    edges = base.edge_idx if base is not None else np.arange(full.n_edges)
    n = view.n_nodes
    adjacency = coo_matrix((np.ones(view.n_edges), (view.tail, view.head)), shape=(n, n))
    count, labels = connected_components(adjacency, directed=False)
    node_order = np.argsort(labels, kind="stable")
    node_bounds = np.searchsorted(labels[node_order], np.arange(count + 1))
    edge_labels = labels[view.tail]
    edge_order = np.argsort(edge_labels, kind="stable")
    edge_bounds = np.searchsorted(edge_labels[edge_order], np.arange(count + 1))
    local = np.empty(n, dtype=np.int64)
    pieces = []
    for piece in range(count):
        members = node_order[node_bounds[piece]:node_bounds[piece + 1]]
        links = edge_order[edge_bounds[piece]:edge_bounds[piece + 1]]
        local[members] = np.arange(members.size)
        tail, head, k = local[view.tail[links]], local[view.head[links]], links.size
        incidence = csc_matrix((np.r_[np.ones(k), -np.ones(k)], (np.r_[tail, head], np.r_[np.arange(k), np.arange(k)])),
                               shape=(members.size, k))
        piece_view = GraphView(node_ids=view.node_ids[members], edge_ids=view.edge_ids[links], tail=tail, head=head,
                               incidence=incidence,
                               boundary_incidence=csc_matrix((members.size, 0), dtype=np.float64), boundary_names=(),
                               difference=_sliced_difference(view, links, members))
        pieces.append(_Restriction(piece_view, nodes[members], edges[links], full.n_nodes, full.n_edges,
                                   drops_edges=False))
    instance.__dict__["_pieces_cache"] = (key, pieces)
    return pieces


def _sliced_difference(view, edges, nodes):
    """The edge difference operator of a sub-view (its edges and nodes), None when the view has the default one."""
    if view.difference is None:
        return None
    return view.difference.tocsr()[edges][:, nodes].tocsr()


def _restriction_for(instance, where):
    """The _Restriction of mask *where*, rebuilt when the topology or the mask's values changed."""
    ds = _live_ds(instance)
    if not hasattr(ds, "has_mask") or not ds.has_mask(where):
        raise KeyError(f"{type(instance).__name__}: graph system filtered by mask '{where}' but the DataStructure "
                       "defines no such mask")
    if ds.__dict__["_masks"][where]["location"] not in ("node", "cell"):
        raise ValueError(f"{type(instance).__name__}: the filters of a graph system select nodes (or grid cells)")
    key = (ds.topology_version, ds.mask_version(where))
    cache = instance.__dict__.setdefault("_restriction_cache", {})
    if where in cache and cache[where][0] == key:
        return cache[where][1]
    full = instance._graph_view
    mask = np.asarray(ds.mask(where), dtype=bool).reshape(-1)
    node_idx = np.flatnonzero(mask)
    edge_idx = np.flatnonzero(mask[full.tail] & mask[full.head])
    local = np.full(full.n_nodes, -1, dtype=np.int64)
    local[node_idx] = np.arange(node_idx.size)
    view = GraphView(
        node_ids=full.node_ids[node_idx], edge_ids=full.edge_ids[edge_idx],
        tail=local[full.tail[edge_idx]], head=local[full.head[edge_idx]],
        incidence=full.incidence[node_idx][:, edge_idx].tocsc(),
        boundary_incidence=csc_matrix((node_idx.size, 0), dtype=np.float64), boundary_names=(),
        difference=_sliced_difference(full, edge_idx, node_idx),
    )
    restriction = _Restriction(view, node_idx, edge_idx, full.n_nodes, full.n_edges)
    cache[where] = (key, restriction)
    return restriction


def _check_well_posed(instance, method_name, view, anchored):
    """
    Every connected piece of a steady subgraph needs an anchor (a Dirichlet node; Robin boundaries come with
    boundary sets), otherwise its solution is defined up to a constant.
    """
    from scipy.sparse.csgraph import connected_components
    n = view.n_nodes
    adjacency = coo_matrix((np.ones(view.n_edges), (view.tail, view.head)), shape=(n, n))
    count, labels = connected_components(adjacency, directed=False)
    for piece in range(count):
        members = np.flatnonzero(labels == piece)
        if not anchored[members].any():
            raise ValueError(f"{type(instance).__name__}.{method_name}: piece of {members.size} nodes "
                             f"{view.node_ids[members][:10].tolist()} has no Dirichlet "
                             "or positive-weight Robin anchor in a steady system (declare transient=True if its balance has "
                             "a time derivative)")


def _snapshot(instance, required_names, node_vids_int, edge_vids_int,
              node_unknowns, edge_unknowns, declared_locs):
    """Pull required variables into float64 arrays before the Newton loop (copies, read at each solve)."""
    ds = _live_ds(instance)
    node_snap, edge_snap = {}, {}
    specs = getattr(instance, "_variable_specs", {})
    node_location = "cell" if "cell" in ds._var_stores() else "node"
    for name in required_names:
        spec = specs.get(name)
        if (spec is not None and spec.variable_type == "parameter" and ds.has(name)
                and ds.location(name) not in ("node", "edge", "cell")):
            # A parameter stored per plant (or as a scalar) is seen by node equations per node and by edge laws per
            # edge, each edge taking its child's plant
            on_nodes = np.asarray(ds.parameter_view(name, node_location)).reshape(-1)
            on_edges = np.asarray(ds.parameter_view(name, "edge")).reshape(-1)
            take_nodes, take_edges = _take(instance, "node"), _take(instance, "edge")
            node_snap[name] = on_nodes if take_nodes is None else on_nodes[take_nodes]
            edge_snap[name] = on_edges if take_edges is None else on_edges[take_edges]
            continue
        if name in declared_locs:
            loc = declared_locs[name]
        elif ds.has(name):
            loc = ds.location(name)
        else:
            loc = "node"
        if loc in ("scalar", "cell"):
            loc = "node"   # scalars are broadcast over the entities; grid cells are the graph's nodes
        elif loc not in ("node", "edge"):
            raise ValueError(f"{type(instance).__name__}: '{name}' is stored at {loc}, graph equations take node "
                             f"or edge arrays: declare it with location='node' and mapping='broadcast'")
        size = len(node_vids_int) if loc == "node" else len(edge_vids_int)
        # Parameters and inputs: read-only views, so that equations cannot overwrite them
        (node_snap if loc == "node" else edge_snap)[name] = _read_array(ds, name, loc, size, type(instance).__name__,
                                                                        take=_take(instance, loc), read_only=True)
    return node_snap, edge_snap


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
            node_snap  — snapshotted float64 arrays for node variables.
            edge_snap  — snapshotted float64 arrays for edge variables.
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
        output_items       = []   # (out_name, bound, raw, selection or None)
        boundary_sets      = []   # boundary_set objects
        pool_items         = []   # (field, bound, raw): pool balances
        pool_specs         = spec_def.get("pool_unknowns") or {}
        pool_names         = list(pool_specs)
        self.output_locations = {}   # out_name -> location given by @graph_output (or None)

        for cls_ in inner_cls.__mro__:
            for attr_name, obj in cls_.__dict__.items():
                if attr_name in seen:
                    continue
                seen.add(attr_name)
                tag = getattr(obj, "__graph_tag__", None)
                if tag is None:
                    continue
                if tag["kind"] == "boundary_set":
                    boundary_sets.append(obj)          # declared data, not a method
                    continue
                bound = obj.__get__(instance, type(instance))
                kind  = tag["kind"]
                key   = f"{type(instance).__name__}.{inner_cls.__name__}.{attr_name}"
                selection = None
                if tag.get("filters") is not None:
                    selection = Filters(tag["filters"], key, "edge" if kind == "edge_law" else "node")
                if kind in ("node_balance", "node_rate"):
                    node_balance_items.append((
                        tag["field"], selection, attr_name, bound, obj,
                        "rate" if kind == "node_rate" else tag.get("explicit", False)
                    ))
                elif kind == "edge_law":
                    edge_law_items.append((
                        tag.get("field"), selection, attr_name, bound, obj,
                        tag.get("explicit", False), tag.get("integrate", False)
                    ))
                elif kind == "boundary_condition":
                    bc_items.append((
                        tag.get("field"), selection, tag.get("bc_kind"),
                        attr_name, bound, obj, tag.get("explicit", False)
                    ))
                elif kind == "pool_balance":
                    pool_items.append((tag["field"], bound, obj))
                elif kind == "graph_jacobian":
                    jacobian_raw = (bound, obj)
                elif kind == "graph_output":
                    output_items.append((tag["name"], bound, obj, selection))
                    self.output_locations[tag["name"]] = tag.get("location")

        # Sort node blocks to match node_unknowns order
        field_order = {f: i for i, f in enumerate(node_unknowns)}
        node_balance_items.sort(key=lambda x: field_order.get(x[0], len(node_unknowns)))

        # One form of node balance per graph system; explicit and IVP solvers need the rate form
        system_name = f"{type(instance).__name__}.{inner_cls.__name__}"
        rate_form = any(ex == "rate" for *_, ex in node_balance_items)
        if rate_form and any(ex != "rate" for *_, ex in node_balance_items):
            raise ValueError(f"{system_name}: its node unknowns mix @node_rate and @node_balance; use one form")
        rate_solver = spec_def.get("rate_solver", False)
        if rate_solver and node_balance_items and not rate_form:
            raise ValueError(f"{system_name}: {spec_def['method']} integrates du/dt, which a residual cannot give: "
                             "write the balance as @node_rate (du/dt, without time term)")
        if rate_form and jacobian_raw is not None:
            raise ValueError(f"{system_name}: @graph_jacobian is the Jacobian of residuals, not of @node_rate")
        if rate_solver:
            dirichlet = [an for _, _, bk, an, _, _, _ in bc_items if bk == "dirichlet"]
            dirichlet += [b.name for b in boundary_sets if b.kind in ("dirichlet", "per_node")]
            if dirichlet:
                raise ValueError(f"{system_name}: Dirichlet conditions ({dirichlet}) fix a value, which {spec_def['method']} "
                                 "cannot integrate: use a Newton solver, or a Robin condition")

        for f, _, an, _, _, _, _ in edge_law_items:
            if f is None:
                raise ValueError(
                    f"@edge_law '{an}': field= must be set explicitly. "
                    f"Declared edge_unknowns: {list(edge_unknowns)}."
                )

        # ── integrate=True: extend edge unknowns with {field}_amount ──────────
        # For each edge law with integrate=True, a new DAE unknown Q_e is added:
        #   (Q_new − Q_old) / dt − q_e = 0  →  Q_new = Q_old + q_e · dt
        # Q_old is read from the DataStructure before the Newton loop; Q_new is
        # solved together with concentration and flux in the same Newton step.
        integrate_fields = sorted({fn for fn, _, _, _, _, _, intg in edge_law_items if intg})
        dt_inst = float(instance.__dict__.get("_current_dt", getattr(instance, "time_step", None)) or 1.0)

        ds = _live_ds(instance)
        # Framework-managed previous state: the unknowns at the start of this solve
        instance._previous_state = {fn: _read_array(ds, fn, location, n if location == "node" else m,
                                                    take=_take(instance, location))
                                    for location, names in (("node", node_unknowns), ("edge", edge_unknowns))
                                    for fn in names if ds.has(fn)}
        for fn in integrate_fields:
            # The integrated amount starts at zero, registered explicitly rather than read as a missing variable
            if not ds.has(f"{fn}_amount"):
                ds.register(f"{fn}_amount", location="edge", default=0.)
        amount_olds = {fn: _read_array(ds, f"{fn}_amount", "edge", m, take=_take(instance, "edge"))
                       for fn in integrate_fields}

        all_edge_unknowns = list(edge_unknowns) + [f"{fn}_amount" for fn in integrate_fields]

        # ── Collect required prop names by signature inspection ───────────────
        all_raws = (
            [r for _, _, _, _, r, _ in node_balance_items]
            + [r for _, _, _, _, r, _, _ in edge_law_items]
            + [r for _, _, _, _, _, r, _ in bc_items]
            + ([jacobian_raw[1]] if jacobian_raw else [])
            + [r for _, _, r, _ in output_items]
            + [r for _, _, r in pool_items]
        )
        required: set[str] = set()
        for raw in all_raws:
            for aname in inspect.getfullargspec(raw)[0][1:]:
                if aname not in node_unknowns and aname not in all_edge_unknowns and aname not in pool_names:
                    required.add(aname)
        for bset in boundary_sets:
            required.update(bset.variables())

        node_snap, edge_snap = _snapshot(
            instance, required, node_vids_int, edge_vids_int,
            node_unknowns, edge_unknowns, declared_locs,
        )

        # Boundary sets: members on the solved graph, values and weights read now
        set_terms = defaultdict(list)    # field -> [(kind, idx, value, weight)]
        for bset in boundary_sets:
            if bset.kind is None:
                continue                      # a selection only (e.g. a pool's exchange set)
            if bset.field is not None:
                field = bset.field
            elif len(node_unknowns) == 1:
                field = node_unknowns[0]
            else:
                raise ValueError(f"boundary_set '{bset.name}': give field=, the system has several node unknowns")
            idx = bset.members(instance, ds, n, _take(instance, "node"))

            def read(x, idx=idx):
                return node_snap[x][idx] if isinstance(x, str) else np.full(idx.size, float(x))
            if bset.kind == "per_node":
                codes = np.asarray(read(bset.kind_variable)).round().astype(np.int64)
                value, weight = read(bset.value), read(bset.weight)
                for kind, code in boundary_set.CODES.items():
                    chosen = codes == code
                    set_terms[field].append((kind, idx[chosen], value[chosen],
                                             weight[chosen] if kind == "robin" else None))
                continue
            set_terms[field].append((bset.kind, idx, read(bset.value), read(bset.weight) if bset.kind == "robin"
                                     else None))

        if instance.__dict__.get("_restriction") is not None and not (spec_def.get("transient", False) or rate_form):
            anchored = np.zeros(n, dtype=bool)
            for _, tf, bc_kind, _, _, _, _ in bc_items:
                if bc_kind == "dirichlet":
                    anchored |= tf.mask(ds, _take(instance, "node")) if tf is not None else True
            for terms in set_terms.values():
                for kind, idx, _, weight in terms:
                    if kind == "dirichlet":
                        anchored[idx] = True
                    elif kind == "robin":
                        anchored[idx[weight > 0]] = True
            _check_well_posed(instance, spec_def["inner_class"].__name__, gv, anchored)

        # ── Pool unknowns: the pools of the solved nodes' entities, coupled to their exchange set ──
        pool_fields_gs, pool_coupling, pool_targets = {}, {}, {}
        sets_by_name = {bset.name: bset for bset in boundary_sets}
        for pname, pspec in pool_specs.items():
            if not ds.has(pname):
                ds.register(pname, location=pspec["location"])
            location = ds.location(pname)
            owner = np.asarray(ds.owner(location), dtype=np.int64)
            take_nodes = _take(instance, "node")
            owner = owner if take_nodes is None else owner[take_nodes]
            entities = np.unique(owner)
            local = np.searchsorted(entities, owner)
            if pspec["exchange"] is None:
                members = np.arange(n)
            elif pspec["exchange"] in sets_by_name:
                members = sets_by_name[pspec["exchange"]].members(instance, ds, n, take_nodes)
            else:
                raise KeyError(f"{type(instance).__name__}: pool '{pname}' exchanges with boundary set "
                               f"'{pspec['exchange']}', which the graph system does not declare")
            pool_coupling[pname] = csr_matrix((np.ones(members.size), (members, local[members])),
                                              shape=(n, entities.size))
            values = np.array(np.asarray(ds.get(pname), dtype=np.float64).reshape(-1)[entities])
            pool_fields_gs[pname] = FieldState(pname, "pool", values)
            pool_targets[pname] = entities
            instance._previous_state[pname] = values.copy()
        instance.__dict__["_pool_exchange"] = pool_coupling
        self.pool_targets = pool_targets
        if pool_names and jacobian_raw is not None:
            raise NotImplementedError(f"{type(instance).__name__}: an analytic @graph_jacobian with pool unknowns")
        missing_balances = set(pool_names) - {field for field, _, _ in pool_items}
        if missing_balances:
            raise ValueError(f"{type(instance).__name__}: pool unknowns {sorted(missing_balances)} have no "
                             "@pool_balance")

        # ── Initial-guess FieldStates ─────────────────────────────────────────
        # Copies: the implicit solvers also use them as u_prev, they must not follow later writes
        node_fields_gs = {fn: FieldState(fn, "node", _read_array(ds, fn, "node", n, take=_take(instance, "node")))
                          for fn in node_unknowns}
        edge_fields_gs = {fn: FieldState(fn, "edge", _read_array(ds, fn, "edge", m, take=_take(instance, "edge")))
                          for fn in edge_unknowns}
        for fn in integrate_fields:
            edge_fields_gs[f"{fn}_amount"] = FieldState(
                f"{fn}_amount", "edge", amount_olds[fn].copy()
            )

        # ── Evaluator factory ─────────────────────────────────────────────────

        def arg_location(aname):
            """Entity ("node" or "edge") of an equation argument: filters slice only those of the filtered entity."""
            if aname in node_unknowns or aname in node_snap:
                return "node"
            if aname in all_edge_unknowns or aname in edge_snap:
                return "edge"
            return None

        def make_evaluator(raw_func, bound_method, type_filter, entity):
            arg_names   = inspect.getfullargspec(raw_func)[0][1:]
            entity_size = n if entity == "node" else m
            own_unknowns = node_unknowns if entity == "node" else all_edge_unknowns
            own = node_snap if entity == "node" else edge_snap
            sliced      = [aname in own_unknowns or (aname not in node_unknowns and aname not in all_edge_unknowns
                                                     and aname in own) for aname in arg_names]

            own_snap, other_snap = (edge_snap, node_snap) if entity == "edge" else (node_snap, edge_snap)

            def evaluator(ctx):
                args = []
                for aname in arg_names:
                    if aname in node_unknowns:
                        args.append(ctx.node_unknowns[aname])
                    elif aname in edge_unknowns:
                        args.append(ctx.edge_unknowns[aname])
                    elif aname in ctx.pool_unknowns:
                        args.append(ctx.pool_unknowns[aname])
                    elif aname in own_snap:
                        args.append(own_snap[aname])
                    elif aname in other_snap:
                        args.append(other_snap[aname])
                    else:
                        raise KeyError(
                            f"Method '{raw_func.__name__}': arg '{aname}' not "
                            f"found in the DataStructure.  Node: {list(node_snap)}  "
                            f"Edge: {list(edge_snap)}"
                        )
                if type_filter is not None:
                    mask = type_filter.mask(ds, _take(instance, entity))
                    sub  = [a[mask] if cut else a for a, cut in zip(args, sliced)]
                    result = np.asarray(_in_equation(instance, bound_method, sub), dtype=np.float64)
                    full   = np.zeros(entity_size, dtype=np.float64)
                    np.add.at(full, np.where(mask)[0], result)
                    return full
                return np.asarray(_in_equation(instance, bound_method, args), dtype=np.float64)

            return evaluator

        # ── BC factory ────────────────────────────────────────────────────────

        def make_bc_eval(raw_func, bound_method, type_filter,
                          bc_kind="dirichlet", field=None, explicit=False):
            arg_names = inspect.getfullargspec(raw_func)[0][1:]
            sliced    = [arg_location(aname) == "node" for aname in arg_names]

            def bc_eval(ctx):
                args = []
                for aname in arg_names:
                    if aname in node_unknowns:
                        args.append(ctx.node_unknowns[aname])
                    elif aname in edge_unknowns:
                        args.append(ctx.edge_unknowns[aname])
                    elif aname in ctx.pool_unknowns:
                        args.append(ctx.pool_unknowns[aname])
                    elif aname in node_snap:
                        args.append(node_snap[aname])
                    elif aname in edge_snap:
                        args.append(edge_snap[aname])
                    else:
                        raise KeyError(
                            f"BC '{raw_func.__name__}': arg '{aname}' not found."
                        )
                if type_filter is not None:
                    mask = type_filter.mask(ds, _take(instance, "node"))
                    idx  = np.where(mask)[0]
                    sub  = [a[mask] if cut else a for a, cut in zip(args, sliced)]
                    vals = np.asarray(_in_equation(instance, bound_method, sub), dtype=np.float64)
                else:
                    idx  = np.arange(n)
                    vals = np.asarray(_in_equation(instance, bound_method, args), dtype=np.float64)

                if explicit and bc_kind == "dirichlet" and field is not None:
                    vals = ctx.node_unknowns[field][idx] - vals
                return idx, vals

            return bc_eval

        # ── Assemble equation blocks ──────────────────────────────────────────

        def make_combined_node_ev(bulk_evals, bc_specs, neumann_scale=1.0, field=None):
            terms = set_terms.get(field, [])

            def evaluator(ctx):
                result = np.zeros(n, dtype=np.float64)
                for w in bulk_evals:
                    result += w(ctx)
                x = ctx.node_unknowns[field] if terms else None
                for kind, idx, value, weight in terms:
                    if kind == "robin":
                        result[idx] += weight * (x[idx] - value)
                    elif kind == "neumann":
                        result[idx] -= value * neumann_scale
                for bkind, bc_ev in bc_specs:
                    idx, vals = bc_ev(ctx)
                    if bkind == "dirichlet":
                        result[idx] = vals
                    else:
                        result[idx] -= vals * neumann_scale         # an inflow, as a boundary_set's value
                for kind, idx, value, _ in terms:
                    if kind == "dirichlet":
                        result[idx] = x[idx] - value
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
            if not grp and not bc_grp and not set_terms.get(field_name):
                continue
            bulk_wrapped = []
            any_explicit = any(ex is True for _, _, _, ex in grp)
            for tf, b, r, ex in grp:
                inner = make_evaluator(r, b, tf, "node")
                if ex == "rate" and rate_solver:
                    # explicit and IVP solvers read du/dt as -R
                    bulk_wrapped.append(lambda ctx, _e=inner: -_e(ctx))
                elif ex == "rate":
                    # backward Euler: (u - previous(u)) / dt - du/dt
                    bulk_wrapped.append(
                        lambda ctx, _f=field_name, _e=inner:
                            (ctx.node_unknowns[_f] - instance.previous(_f)) / instance.dt - _e(ctx)
                    )
                elif ex:
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
                evaluator = make_combined_node_ev(bulk_wrapped, bc_wrapped, neumann_scale, field=field_name),
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

        for pname in pool_names:
            balances = [make_evaluator(raw, bound, None, "pool") for field, bound, raw in pool_items if field == pname]
            equation_blocks.append(EquationBlock(
                name=f"pool_balance_{pname}",
                evaluator=balances[0] if len(balances) == 1 else (lambda ctx, _b=balances: sum(b(ctx) for b in _b)),
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
                evaluator = make_evaluator(raw, bound, selection, "node"),
            )
            for oname, bound, raw, selection in output_items
        )

        jac_evaluator = (
            make_evaluator(jacobian_raw[1], jacobian_raw[0], None, "node")
            if jacobian_raw else None
        )
        if jac_evaluator is not None and any(set_terms.values()):
            # The user's Jacobian covers the equations; the framework adds the boundary sets' terms
            user_jacobian = jac_evaluator

            def jac_evaluator(ctx, _user=user_jacobian):
                from scipy.sparse import issparse
                J = _user(ctx)
                sparse = issparse(J)
                J = J.tolil() if sparse else np.array(J, dtype=np.float64)
                for field, terms in set_terms.items():
                    offset = list(node_unknowns).index(field) * n
                    for kind, idx, _, weight in terms:
                        rows = offset + idx
                        if kind == "robin":
                            J[rows, rows] = np.asarray(J[rows, rows]).reshape(-1) + weight
                        elif kind == "dirichlet":
                            J[rows, :] = 0.
                            J[rows, rows] = 1.
                return J.tocsr() if sparse else J

        boundary_ports = ()

        spec = GraphDAESpec(
            graph             = gv,
            node_fields       = node_fields_gs,
            edge_fields       = edge_fields_gs,
            boundary_ports    = boundary_ports,
            unknowns          = UnknownLayout(
                node_fields   = tuple(node_unknowns),
                edge_fields   = tuple(all_edge_unknowns),
                pool_fields   = tuple(pool_names),
            ),
            equation_blocks   = tuple(equation_blocks),
            output_blocks     = output_blocks,
            jacobian_evaluator= jac_evaluator,
            parameters        = {},
            pool_fields       = pool_fields_gs,
            pool_coupling     = pool_coupling,
        )
        # IVP solvers report their evaluation time, at which equations read forcings
        spec.parameters["time_hook"] = lambda t, _i=instance: _i.__dict__.__setitem__("_ivp_time", t)
        return spec, node_snap, edge_snap

    # ── Solve + inject ────────────────────────────────────────────────────────

    def build_and_step(self, previous_node_fields=None,
                       dt=None) -> tuple[np.ndarray, dict]:
        """
        Build the spec, run one solve step, return (packed, outputs).
        Called once per Choregrapher tick from _invoke_graph_system.
        """
        spec, _, _ = self.build()
        self.last_spec = spec
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
        """Write the converged unknowns (including {field}_amount) to the DataStructure, in place."""
        node_u, edge_u = spec.unpack_unknowns(packed)
        ds = _live_ds(self._instance)
        for location, values_by_name in (("node", {fn: node_u[fn] for fn in self._spec_def["node_unknowns"]}),
                                         ("edge", {fn: edge_u[fn] for fn in spec.unknowns.edge_fields})):
            restriction = self._instance.__dict__.get("_restriction")
            for fn, values in values_by_name.items():
                if not ds.has(fn):
                    ds.register(fn, location=_entity_location(ds, location))
                if restriction is None:
                    ds.set(fn, values)
                else:
                    # Inactive nodes stay frozen; dropped edges carry no flux, their integrated amounts are kept
                    dropped = None if location == "node" or fn.endswith("_amount") else 0.
                    restriction.scatter(ds, fn, values, location, dropped=dropped)
        for pname, values in spec.unpack_pools(packed).items():      # pools at their own scale
            stored = ds.get(pname)
            full = np.array(stored, dtype=np.float64).reshape(-1)
            full[self.pool_targets[pname]] = values
            ds.set(pname, full.reshape(stored.shape))


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
      inject_result()    → write-back to the DataStructure
    """
    # Refresh biological-scale fields from MTG before snapshotting.
    if hasattr(self, "_refresh_from_bio_scale"):
        self._refresh_from_bio_scale()
    spec_def = type(self)._graph_system_specs[method_name]
    filters = spec_def.get("filters")
    # The active subgraph: the DataStructure mask of the system's filters
    where = (None if filters is None
             else Filters(filters, f"{type(self).__name__}.{method_name}").mask_name(_live_ds(self)))
    split = spec_def.get("split", "whole") == "components"
    if where is None and not split:
        _solve_graph_system(self, method_name, spec_def)
        return
    restriction = None
    if where is not None:
        # Active subgraph: the equations see its GraphView through self._graph_view
        restriction = _restriction_for(self, where)
        if restriction.node_idx.size == 0:
            ds = _live_ds(self)
            for fn in spec_def["edge_unknowns"]:
                if ds.has(fn):
                    ds.set(fn, 0.)            # no active edge: no flux
            return
    if not split:
        _solve_restricted(self, method_name, spec_def, restriction)
        return
    # One solve per connected piece; edges outside the active subgraph carry no flux, set once
    if restriction is not None and restriction.dropped_edges.size:
        ds = _live_ds(self)
        for fn in spec_def["edge_unknowns"]:
            if ds.has(fn):
                values = np.array(ds.get(fn), dtype=np.float64).reshape(-1)
                values[restriction.dropped_edges] = 0.
                ds.set(fn, values)
    # The pieces share one dict of previous / solved node fields, copied once and updated at each piece's nodes, so
    # that the bookkeeping stays proportional to each piece
    saved_key, previous_key = f"_gsol_{method_name}", f"_gprev_{method_name}"
    saved = getattr(self, saved_key, None)
    if saved is not None:
        saved = {fn: np.array(values, dtype=np.float64) for fn, values in saved.items()}
        setattr(self, saved_key, saved)
        setattr(self, previous_key, saved)
    self.__dict__["_in_pieces"] = True
    try:
        for piece in _pieces_of(self, restriction):
            _solve_restricted(self, method_name, spec_def, piece)
    finally:
        self.__dict__.pop("_in_pieces", None)


def _solve_restricted(self, method_name, spec_def, restriction) -> None:
    self.__dict__["_restriction"], self.__dict__["_solve_view"] = restriction, restriction.view
    try:
        _solve_graph_system(self, method_name, spec_def)
    finally:
        self.__dict__.pop("_restriction", None)
        self.__dict__.pop("_solve_view", None)


def _unknown_names(ds, spec_def) -> list:
    names = list(spec_def["node_unknowns"]) + list(spec_def["edge_unknowns"])
    names += [f"{fn}_amount" for fn in spec_def["edge_unknowns"] if ds.has(f"{fn}_amount")]
    return names + [pool for pool in (spec_def.get("pool_unknowns") or {}) if ds.has(pool)]


def _capture(ds, names, instance=None) -> dict:
    """Copies of *names*: of the current subgraph's entities only during a restricted solve, as (location, values)."""
    restriction = instance.__dict__.get("_restriction") if instance is not None else None
    if restriction is None:
        return {name: np.array(ds.get(name), dtype=np.float64) for name in names if ds.has(name)}
    captured = {}
    for name in names:
        if ds.has(name):
            stored = ds.location(name)
            if stored not in ("node", "cell", "edge"):          # pools at a coarse scale: small, kept whole
                captured[name] = (stored, np.array(ds.get(name), dtype=np.float64))
                continue
            location = "edge" if stored == "edge" else "node"
            idx = restriction.node_idx if location == "node" else restriction.edge_idx
            captured[name] = (location, np.array(np.asarray(ds.get(name)).reshape(-1)[idx], dtype=np.float64))
    return captured


def _captured(state, name) -> np.ndarray:
    values = state[name]
    return values[1] if isinstance(values, tuple) else values


def _restore(ds, state, instance=None) -> None:
    restriction = instance.__dict__.get("_restriction") if instance is not None else None
    for name, values in state.items():
        if restriction is None:
            ds.set(name, values)
        else:
            location, values = values
            if location in ("node", "edge"):
                restriction.scatter(ds, name, values, location)
            else:
                ds.set(name, values)


def _saved_fields(self, key, ds, names) -> None:
    """Keep the node unknowns on every node (attribute *key*), updating only the subgraph's during a restricted solve."""
    restriction = self.__dict__.get("_restriction")
    saved = getattr(self, key, None)
    if (restriction is None or saved is None
            or any(fn not in saved or np.size(saved[fn]) != restriction.n for fn in names)):
        setattr(self, key, {fn: np.array(ds.get(fn), dtype=np.float64).reshape(-1) for fn in names})
        return
    if self.__dict__.get("_in_pieces"):
        for fn in names:
            saved[fn][restriction.node_idx] = np.asarray(ds.get(fn)).reshape(-1)[restriction.node_idx]
        return
    # A new dict of the same arrays: _gprev_ (the previous solve's) may hold the old dict, whose arrays it must keep
    updated = {}
    for fn in names:
        values = saved[fn] if not _shared_with_previous(self, key, saved[fn]) else np.array(saved[fn])
        values[restriction.node_idx] = np.asarray(ds.get(fn)).reshape(-1)[restriction.node_idx]
        updated[fn] = values
    setattr(self, key, updated)


def _shared_with_previous(self, key, array) -> bool:
    previous = getattr(self, key.replace("_gsol_", "_gprev_"), None) if key.startswith("_gsol_") else None
    return previous is not None and any(array is other for other in previous.values())


def _solve_graph_system(self, method_name: str, spec_def: dict) -> None:
    """
    Integrate one graph system over the component's time step:
    "step" one solve of time_step; "substeps" n solves of time_step / n; "adaptive" step doubling. During each
    (sub-)step, self.dt is its length and self.previous(fn) the state at its start.
    """
    ds = _live_ds(self)
    time_step = float(getattr(self, "time_step", None) or 1.0)
    integrate = spec_def.get("integrate", "step")
    names = _unknown_names(ds, spec_def)
    take = {"node": _take(self, "node"), "edge": _take(self, "edge")}
    # previous(fn, at="solve"): the unknowns at the start of this call's solve
    self.__dict__["_solve_start_state"] = {
        fn: _read_array(ds, fn, location, 0, take=take[location])
        for location, unknowns in (("node", spec_def["node_unknowns"]), ("edge", spec_def["edge_unknowns"]))
        for fn in unknowns if ds.has(fn)}
    try:
        self.__dict__["_solve_offset"] = 0.         # start of the current (sub-)step within the call (forcings)
        if integrate == "step":
            self.__dict__["_current_dt"] = time_step
            _solve_once(self, method_name, spec_def)
        elif integrate == "substeps":
            n = int(spec_def["n_substeps"])
            for k in range(n):
                self.__dict__["_solve_offset"] = k * time_step / n
                _solve_substep(self, method_name, spec_def, time_step / n)
            self._last_integration = {"steps": n, "rejected": 0}
        else:
            _integrate_adaptive(self, method_name, spec_def, time_step, names)
    finally:
        for key in ("_current_dt", "_solve_offset", "_ivp_time"):
            self.__dict__.pop(key, None)


def _solve_substep(self, method_name, spec_def, h) -> None:
    """One sub-step of length h starting from the current DataStructure state."""
    ds = _live_ds(self)
    # The solver's previous fields are the state at the start of the sub-step
    _saved_fields(self, f"_gsol_{method_name}", ds, spec_def["node_unknowns"])
    self.__dict__["_current_dt"] = h
    _solve_once(self, method_name, spec_def)


def _integrate_adaptive(self, method_name, spec_def, time_step, names) -> None:
    """
    Step doubling: a step of h is compared with two steps of h/2; accepted when the difference is within
    rtol * |x| + atol on the node unknowns, h being adapted after each trial. Raises when h falls below min_step.
    """
    ds = _live_ds(self)
    rtol, atol = spec_def["rtol"], spec_def["atol"]
    max_step = min(spec_def.get("max_step") or time_step, time_step)
    min_step = spec_def.get("min_step") or time_step * 1e-6
    t, h, steps, rejected = 0., max_step, 0, 0
    while t < time_step * (1. - 1e-12):
        h = min(h, time_step - t)
        start = _capture(ds, names, self)
        self.__dict__["_solve_offset"] = t
        _solve_substep(self, method_name, spec_def, h)
        coarse = _capture(ds, spec_def["node_unknowns"], self)
        _restore(ds, start, self)
        _solve_substep(self, method_name, spec_def, h / 2.)
        self.__dict__["_solve_offset"] = t + h / 2.
        _solve_substep(self, method_name, spec_def, h / 2.)
        fine = _capture(ds, spec_def["node_unknowns"], self)
        error = max((np.max(np.abs(_captured(fine, fn) - _captured(coarse, fn))
                            / (atol + rtol * np.abs(_captured(fine, fn)))) for fn in fine), default=0.)
        if error <= 1.:
            t, steps = t + h, steps + 1
            h *= min(2., max(0.2, 0.9 / np.sqrt(max(error, 1e-12))))
        else:
            _restore(ds, start, self)
            rejected += 1
            h *= max(0.2, 0.9 / np.sqrt(error))
            if h < min_step:
                raise RuntimeError(f"{type(self).__name__}.{method_name}: adaptive step below min_step={min_step:g} "
                                   f"at t={t:g} of {time_step:g}")
    self._last_integration = {"steps": steps, "rejected": rejected}


def _solve_once(self, method_name: str, spec_def: dict) -> None:
    """Build, solve and write back one graph system, on the whole graph or on the current active subgraph."""

    # ── Advance previous-field bookkeeping ────────────────────────────────────
    _saved_key = f"_gsol_{method_name}"
    _prev_key  = f"_gprev_{method_name}"
    _saved     = getattr(self, _saved_key, None)
    if _saved is not None:
        setattr(self, _prev_key, _saved)
    previous_fields = getattr(self, _prev_key, None)
    dt = self.__dict__.get("_current_dt", getattr(self, "time_step", None))

    # First implicit_euler call: use current field values as u_prev
    _solver_cls = spec_def.get("solver_cls")
    _is_implicit = (
        spec_def.get("method") == "implicit_euler" or
        (_solver_cls is not None and issubclass(_solver_cls, ImplicitEulerSolver))
    )
    if previous_fields is None and _is_implicit:
        previous_fields = {fn: np.array(_live_ds(self).get(fn), dtype=np.float64).reshape(-1)
                           for fn in spec_def["node_unknowns"]}
    take = _take(self, "node")
    if previous_fields is not None and take is not None:
        previous_fields = {fn: np.asarray(values)[take] for fn, values in previous_fields.items()}

    # ── Build + solve ─────────────────────────────────────────────────────────
    builder = GraphSystemBuilder(self, spec_def)
    packed, outputs = builder.build_and_step(
        previous_node_fields=previous_fields,
        dt=dt,
    )

    # ── Store for next tick's previous-field and test introspection ───────────
    spec       = builder.last_spec  # the spec that was solved
    node_u, _  = spec.unpack_unknowns(packed)

    self._graph_solution_fields = {fn: node_u[fn].copy()
                                    for fn in spec_def["node_unknowns"]}

    # ── Inject results ────────────────────────────────────────────────────────
    builder.inject_result(packed, spec)
    # Kept on every node, so that a later solve on another active subgraph (or piece) finds its previous values
    _saved_fields(self, _saved_key, _live_ds(self), spec_def["node_unknowns"])

    # ── Write output-block results ─────────────────────────────────────────────
    n  = self._graph_view.n_nodes
    ds = _live_ds(self)
    restriction = self.__dict__.get("_restriction")
    for oname, arr in outputs.items():
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        location = builder.output_locations.get(oname)
        if ds.has(oname):
            if location is not None and ds.location(oname) != _entity_location(ds, location):
                raise ValueError(f"{type(self).__name__}: @graph_output('{oname}', location='{location}') but "
                                 f"'{oname}' is registered at {ds.location(oname)}")
        else:
            if location is None:
                location = infer_output_location(type(self).__name__, oname, arr.shape,
                                                  {"node": (n,), "edge": (self._graph_view.n_edges,)})
            ds.register(oname, location=_entity_location(ds, location))
        if restriction is None:
            ds.set(oname, arr)
        else:
            location = ds.location(oname)
            restriction.scatter(ds, oname, arr, location, dropped=None if location == "node" else 0.)


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
        _trampoline.__qualname__ = f"{owner.__qualname__}.{name}"
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
    filters        = None,
    transient      = None,
    integrate      = "step",
    n_substeps     = 1,
    rtol           = 1e-4,
    atol           = 1e-8,
    min_step       = None,
    max_step       = None,
    split          = "whole",
    pool_unknowns  = None,
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
    filters        : the nodes of the active subgraph the system is solved on ({variable: condition}, a mask name or
        a callable, see Filters): the nodes it selects and the edges between them; the other nodes are frozen, and
        dropped edges carry no flux. None: the whole graph.
    transient      : bool | None
        Whether the balance has a time derivative. Steady systems on an active subgraph need a Dirichlet anchor in
        every connected piece, which is checked. Default: True for the time-stepping solvers (explicit and implicit
        Euler, IVP), False otherwise.
    integrate      : "step" | "substeps" | "adaptive"
        "step": one solve of the component's time_step (default); "substeps": n_substeps solves of
        time_step / n_substeps; "adaptive": step doubling with rtol / atol, min_step / max_step. Equations must
        write their time terms with self.dt and self.previous().
    pool_unknowns  : {name: location} | {name: {"location": ..., "exchange": boundary set name}} | None
        Unknowns at a coarse scale, one per entity (e.g. the shoot phloem pool of each plant, location "Plant"),
        solved with the node and edge unknowns. Their residual is a @pool_balance method; equations exchange
        with them through self.pool_exchange(name), the sparse map between the nodes of the exchange set (default:
        every node) and the pool of their entity. Newton solvers only.
    split          : "whole" | "components"
        "components": each connected piece of the graph (of the active subgraph with filters=), e.g. each plant of a
        population, is solved on its own, with its own Newton convergence and integration steps. "whole": one
        system (default).
    """
    # Backward-compat: honour deprecated method= kwarg.
    if method is not None:
        if solver != "newton":
            raise TypeError(
                "graph_system() received both 'solver' and 'method'.  "
                "Use 'solver' only; 'method' is a deprecated alias."
            )
        solver = method

    check_filters(filters, "graph_system")
    pools = {}
    for pool_name, pool_spec in (pool_unknowns or {}).items():
        pool_spec = {"location": pool_spec} if isinstance(pool_spec, str) else dict(pool_spec)
        if "location" not in pool_spec:
            raise ValueError(f"graph_system: pool '{pool_name}' needs a location (e.g. 'Plant')")
        pools[pool_name] = {"location": pool_spec["location"], "exchange": pool_spec.get("exchange")}
    if split not in ("whole", "components"):
        raise ValueError(f"graph_system: split must be 'whole' or 'components', got '{split}'")
    if integrate not in ("step", "substeps", "adaptive"):
        raise ValueError(f"graph_system: integrate must be 'step', 'substeps' or 'adaptive', got '{integrate}'")
    if integrate == "substeps" and int(n_substeps) < 1:
        raise ValueError("graph_system: n_substeps must be at least 1")

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

    if issubclass(solver_cls, ImplicitEulerSolver):
        warnings.warn("solver='implicit_euler' is deprecated: it is solver='newton' with transient=True, the "
                      "equations writing their time terms (or given as @node_rate)", DeprecationWarning, stacklevel=2)
        solver_cls, method_str = NewtonSolver, "newton"
        transient = True if transient is None else transient
    rate_solver = issubclass(solver_cls, (ExplicitEulerSolver, ScipyIVPSolver))

    if pools and rate_solver:
        raise ValueError("graph_system: pool unknowns need a Newton solver (write their time terms in @pool_balance)")

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
        "filters"       : filters,
        "integrate"     : integrate,
        "n_substeps"    : n_substeps,
        "rtol"          : rtol,
        "atol"          : atol,
        "min_step"      : min_step,
        "max_step"      : max_step,
        "split"         : split,
        "pool_unknowns" : pools,
        "transient"     : rate_solver if transient is None else bool(transient),
        "rate_solver"   : rate_solver,
    }

    def decorator(cls):
        return _GraphSystemDescriptor(cls, spec)

    return decorator
