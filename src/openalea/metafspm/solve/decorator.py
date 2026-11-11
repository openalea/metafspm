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
        name = boundary_set(where=, condition=, value=, weight=)  a boundary condition on a set of nodes
        @boundary_condition(location, condition) a boundary condition written as a method
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
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, issparse

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
    SolverConfig, SolverConvergenceError, _SOLVER_REGISTRY, ExplicitEulerSolver, ScipyIVPSolver,
)

# Map MPG integer scale constants → generic "node"/"edge" for snapshot routing.
from openalea.metafspm.data_structure.configs import ScalesConfig as _ScalesConfig
_SCALE_INT_TO_LOC: dict = {
    _ScalesConfig.Compartment: "node",
    _ScalesConfig.Connection: "edge",
}

# ── Choregrapher (unchanged) ──────────────────────────────────────────────────
from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.solve.functor import Functor
from openalea.metafspm.solve import conservation
from openalea.metafspm.solve.conservation import check_conserved


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
                  where=None, include_inactive: bool = False, memo: bool = False):
        """
        location:  of the step's output when it is not a declared field ("node", "edge", "scalar", "cell", or a
                   coarse scale name); locations: {output name: location} for the supplementary outputs.
        where:     the entities the step computes on ({variable: condition}, a mask name or a callable, see
                   _EntityFilter); the others keep their values.
        include_inactive:
                   the step also computes on the entities outside the DataStructure's "active" mask (planting times,
                   dead tissues), which it skips by default.
        memo:      the step's outputs are computed again only when its arguments' values, its component's
                   parameters or settings, the active mask or the structure changed (an expensive step whose inputs
                   rarely change: a network's conductances); else its last outputs are written again.
        """
        if func is None:
            return lambda f: decorator(f, vectorized=vectorized, location=location, locations=locations,
                                       where=where, include_inactive=include_inactive, memo=memo)
        _check_where(where, f"@{name} {func.__name__}")
        func.__where__ = where
        func.__include_inactive__ = include_inactive
        func.__memo__ = memo
        func.__step_tag__ = {"name": name, "total": total, "iterating": iterating, "vectorized": vectorized}
        func.__vectorized__ = vectorized
        func.__output_locations__ = dict(locations or {})
        if location is not None:
            func.__output_locations__[func.__name__[1:]] = location
        Choregrapher().add_process(Functor(func, total=total, iteraring=iterating), name=name)
        return func
    return decorator

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

def node_balance(field=None, where=None):
    """
    Tag a method as a node residual block for *field*.

    Parameters
    ----------
    field    : str   node-unknown field this block contributes to.
    where    : the nodes the block applies on ({variable: condition}, a mask name or a callable: conventions.md), e.g.
               {"tissue": CORTEX}; the method's node arguments are sliced to them, and it contributes 0 elsewhere.
               Values the method reads through self (self.previous(), self.data_structure) are not sliced.
    The method returns the residual (its time term written with self.previous() and self.dt); @node_rate is the
    rate form.
    """
    def decorator(func):
        func.__graph_tag__ = {
            "term": "node_balance", "field": field,
            "where": where, "explicit": False,
        }
        return func
    return decorator


def node_rate(field=None, where=None):
    """
    Tag a method as the rate form of a node unknown's balance: it returns du/dt (sources minus the divergence of the
    fluxes, divided by the capacities), with no time term. The framework writes the time term for the solver
    chosen: (u - previous(u)) / dt - rate with the Newton family (backward Euler, also with integrate="substeps" or
    "adaptive"), u += dt * rate with explicit_euler, du/dt = rate with the scipy IVP solvers. The node unknowns of
    one graph system all use one form: @node_rate, or @node_balance (the residual form, Newton family only).

    Parameters
    ----------
    field   : node unknown the rate is of.
    where : the nodes, as for node_balance: the rate applies on the selected nodes (zero elsewhere).
    """
    def decorator(func):
        func.__graph_tag__ = {"term": "node_rate", "field": field, "where": where}
        return func
    return decorator


def pool_balance(field):
    """
    Tag a method as the residual of pool unknown *field*: it returns one value per pool entity (e.g. per plant),
    and takes the pool's values as an argument named after it, like any unknown.
    """
    def decorator(func):
        func.__graph_tag__ = {"term": "pool_balance", "field": field}
        return func
    return decorator


def edge_law(func=None, *, field=None, where=None,
             explicit=False, integrate=False):
    """
    Tag a method as an edge residual block.

    Parameters
    ----------
    field    : str   edge-unknown field this block contributes to.
    where    : the edges the law applies on ({variable: condition} on edge variables, a mask name or a callable,
               conventions.md).
    explicit : bool  method returns the value; R = unknown − value generated.
    integrate: bool  add the integrated amount "{field}_amount" (Q_new = Q_old + q dt) as an edge unknown.
    """
    def _decorate(f):
        f.__graph_tag__ = {
            "term": "edge_law", "field": field,
            "where": where, "explicit": explicit, "integrate": integrate,
        }
        return f
    if func is not None:
        return _decorate(func)
    return _decorate


def boundary_condition(location, condition, field=None, where=None, explicit=False):
    """
    Tag a method as a boundary condition on a set of nodes, for conditions given by an equation: the method takes
    its arguments by name, like node_balance (unknowns, and DataStructure variables, e.g. coupled ones, read at
    each solve), node-located ones sliced to the selected nodes. For a condition given by a variable or a constant,
    a boundary_set is enough::

        @boundary_condition("node", "neumann", field="concentration", where={"is_collar": ">0"})
        def _collar_uptake(self, concentration, soil_concentration, uptake_rate):
            return uptake_rate * (soil_concentration - concentration)          # an inflow

    Parameters
    ----------
    location : "node"
    condition: "dirichlet": the selected nodes' residual rows become the method's values (a residual, e.g.
               ``p - collar_pressure``; with explicit=True, the prescribed value itself);
               "neumann": the method's values are an inflow, subtracted from the residual, as boundary_set's.
    field    : the node unknown it applies to.
    where    : the nodes ({variable: condition}, a mask name or a callable: conventions.md). Default: every node.
    explicit : for dirichlet, the method returns the prescribed value instead of a residual.
    """
    if location == "edge":
        raise NotImplementedError("@boundary_condition(location='edge') is not supported yet: conditions are applied "
                                  "on nodes (use a boundary_set on the nodes)")
    if location != "node":
        raise ValueError(f"@boundary_condition: location must be 'node', got '{location}'")
    if condition not in ("dirichlet", "neumann"):
        raise ValueError(f"@boundary_condition: condition must be 'dirichlet' or 'neumann', got '{condition}'")
    _check_where(where, "@boundary_condition")

    def decorator(func):
        func.__graph_tag__ = {
            "term": "boundary_condition",
            "location": location, "condition": condition,
            "field": field, "where": where, "explicit": explicit,
        }
        return func
    return decorator


def _check_where(where, owner: str) -> None:
    """Refuse a where= that is not a {variable: condition} dict, a mask name or a callable."""
    if where is None or callable(where) or isinstance(where, str) or (isinstance(where, dict) and where):
        return
    raise TypeError(f"{owner}: where must be a {{variable: condition}} dict, a mask name or a callable, "
                    f"got {where!r}")


class _EntityFilter:
    """
    The elements a decorator operates on, from its where= argument::

        {variable: condition, ...}   every pair holding: a value, a list of values, or a comparison (">0", "<=0.03");
                                     label names are resolved, and a variable of a coarser scale is read at each
                                     element's entity of that scale
        "mask name"                  a mask of the DataStructure (ds.define_mask), e.g. shared with a mapping
        callable                     ds -> boolean array, e.g. a geometric selection

    Resolved to a mask of the DataStructure (at the elements' location), recomputed when its variables are written
    or the topology changes.
    """

    def __init__(self, rule, key: str, location: str = "node"):
        _check_where(rule, key)
        self.rule, self.key, self.location = rule, key, location

    def mask_name(self, ds) -> str:
        if isinstance(self.rule, str):
            if not ds.has_mask(self.rule):
                raise KeyError(f"{self.key}: where='{self.rule}' names no mask of the DataStructure "
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

        leaves = boundary_set(where={"label": "LeafElement"}, condition="robin", value="air_water_potential",
                              weight="leaf_conductance")

    Arguments::

        where   {variable: condition} | a mask name | a callable ds -> boolean mask (conventions.md); membership
                follows the selecting variables and topology changes.
        condition "robin":     + w * (x - v) in the field's residual (an outflow towards the external value v);
                "dirichlet": the residual row becomes x - v;
                "neumann":   - v in the residual (v is an inflow);
                None:        a selection only (no term), e.g. the nodes exchanging with a pool unknown.
        condition_variable
                instead of condition, a node variable name: each node's condition is read at each solve from that
                variable,
                as boundary_set.CODES (1 dirichlet, 2 neumann, 3 robin, anything else no condition), e.g. a collar
                switching between a pressure and a flux.
        value, weight
                node variables of the DataStructure (read at each solve) or constants.
        field   the node unknown it applies to; default: the only node unknown.
    """

    CONDITIONS = ("robin", "dirichlet", "neumann")
    # Codes of a per-node condition variable; other values (e.g. 0) apply no condition
    CODES = {"dirichlet": 1, "neumann": 2, "robin": 3}

    def __init__(self, where, condition=None, value=0., weight=1.0, field=None, condition_variable: str = None):
        if condition is not None and condition not in self.CONDITIONS:
            raise ValueError(f"boundary_set: condition must be one of {self.CONDITIONS}, got '{condition}'")
        if condition_variable is not None:
            if condition is not None:
                raise ValueError("boundary_set: give condition= (one condition) or condition_variable= (a node variable), "
                                 "not both")
            condition = "per_node"
        self.condition_variable = condition_variable
        if where is None:
            raise TypeError("boundary_set: where= is required (the nodes of the set)")
        _check_where(where, "boundary_set")
        self.where, self.condition, self.value, self.weight, self.field = where, condition, value, weight, field
        self.name = None
        self.__graph_tag__ = {"term": "boundary_set"}

    def __set_name__(self, owner, name):
        self.name = name

    def variables(self) -> list:
        """DataStructure variables read by the solve (value and weight given by name)."""
        weight = self.weight if self.condition in ("robin", "per_node") else None
        return [x for x in (self.value, weight, self.condition_variable) if isinstance(x, str)]

    def members(self, instance, ds, size, take) -> np.ndarray:
        """Indices, in the solved graph, of the nodes of the set."""
        return _EntityFilter(self.where, f"{type(instance).__name__}.{self.name}").members(ds, take)


def graph_jacobian(func):
    """Tag a method as the optional analytic Jacobian evaluator."""
    func.__graph_tag__ = {"term": "graph_jacobian"}
    return func


def explicit_term(field=None, jacobian=None):
    """
    Tag a method as a term of *field*'s node balance kept explicit by a reduction (solve.reduction): a nonlinear
    term, never eliminated with a group's interior, its nodes kept. The method returns its residual contribution;
    *jacobian*: the name of the method giving its Jacobian (the system's @graph_jacobian then covers the other terms
    only). Without a reduction it is one more term of the balance.
    """
    def decorator(func):
        func.__graph_tag__ = {"term": "node_balance", "field": field, "where": None, "explicit": False,
                              "explicit_term": True, "term_jacobian": jacobian}
        return func
    return decorator


def input_jacobian(name):
    """
    Tag a method as the derivative of the system's node balance with respect to the DataStructure node variable
    *name* (a sparse matrix, nodes × nodes), for a reduction's per-group inputs (solve.reduction): a group's input
    changes act through it, carried to the kept nodes.
    """
    def decorator(func):
        func.__graph_tag__ = {"term": "input_jacobian", "name": name}
        return func
    return decorator


def archetype_derived(name: str):
    """
    Tag a method of a reduced graph system's component as a value its reduction's archetypes derive (solve.reduction):
    method(self, archetype, full_template, kron, nodes) -> value, run once per archetype of the systems the component declares,
    on the reduction's library (self its library component, self.data_structure the library's DataStructure),
    while the archetype's interior factorisation is live (before kron.release()); its value is stored in
    archetype.outputs[name]. *archetype*: the Archetype (D, e, kept_rows, interior_rows: its full template's Compartment rows),
    *full_template*: the full SectionTemplate, *kron*: its KronReduction (kron.shares, kron.recover: no
    refactorisation), *nodes*: the full group's node indices in the library's DataStructure, in the kron's local
    order (its kept nodes nodes[kron.kept_nodes]; with merged nodes, Reduction(merge=), a merged group's
    representative only: archetype.contract(values per full template Compartment) sums them into the kron's order). E.g.
    the kept nodes' share of the interior's volumes.
    """
    def decorator(func):
        func.__archetype_derived__ = name
        return func
    return decorator


def _archetype_derived_methods(component) -> dict:
    """{name: method} of *component*'s @archetype_derived methods (its class's and its bases', the most derived
    first)."""
    out = {}
    for cls in type(component).__mro__:
        for attribute in vars(cls).values():
            name = getattr(attribute, "__archetype_derived__", None)
            if name is not None and name not in out:
                out[name] = attribute.__get__(component, type(component))
    return out


# Reading an archetype's operator (solve.reduction): the system's equations alone, without its boundary sets, explicit
# terms and reduced terms
_EXTRACTING = []
# The graph systems not solved while a reduction's library runs (their operators are read, not their solutions)
_SKIPPED = set()


def declared_system(instance, inner_name: str, systems) -> str:
    """The name among *systems* ("Component._system") of *instance*'s graph system *inner_name*: its class's or a
    base class's (a subclass of the declared component, e.g. with a parameter more, runs the same system), or
    None."""
    for cls in type(instance).__mro__:
        name = f"{cls.__name__}.{inner_name}"
        if name in systems:
            return name
    return None


def linear_system(instance, name: str):
    """*instance*'s graph system *name* as its linear algebra sees it (solve.linear.LinearSystem): its caches' key,
    its qualified names "Component._system" (its class's, then its bases' declaring it: Compute(linear_solver=)'s
    mapping may name either) and the properties it declares (SYSTEM_PROPERTIES: symmetric=True)."""
    from openalea.metafspm.solve.linear import LinearSystem
    spec_def = type(instance)._graph_system_specs[name]
    names = tuple(f"{cls.__name__}.{name}" for cls in type(instance).__mro__
                  if name in getattr(cls, "_graph_system_specs", {}))
    return LinearSystem(key=(type(instance).__name__, name, id(instance)), names=names,
                        properties=frozenset({"symmetric"} if spec_def.get("symmetric") else ()))


def linear_systems(components) -> dict:
    """{qualified name: LinearSystem} of the graph systems of *components* (linear_system: each of its names)."""
    found = {}
    for component in components:
        for name in getattr(type(component), "_graph_system_specs", {}):
            system = linear_system(component, name)
            for qualified in system.names:
                found.setdefault(qualified, system)
    return found


def graph_output(name, location: str = None, where=None):
    """
    Tag a method as a named post-solve output hook. *location* ("node" or "edge") is required when *name* is not a
    declared field and its size does not identify a single location.

    where: the nodes the output is computed on ({variable: condition}, a mask name or a callable: conventions.md);
    the method's node arguments are sliced to them, and the other nodes get 0. Node outputs only.
    """
    if location not in (None, "node", "edge"):
        raise ValueError(f"@graph_output('{name}'): location must be 'node' or 'edge', got '{location}'")
    if where is not None and location == "edge":
        raise ValueError(f"@graph_output('{name}'): where= chooses nodes, not edges")
    _check_where(where, f"@graph_output('{name}')")

    def decorator(func):
        func.__graph_tag__ = {"term": "graph_output", "name": name, "location": location, "where": where}
        return func
    return decorator


def _undeclared_output(owner: str, name: str):
    """An output that is neither a registered variable nor given a location: an error asking to declare it."""
    raise ValueError(f"{owner}: output '{name}' has no declared location: declare it as a field (state_variable("
                     "location=...)) or give the step's decorator location= / locations={...} (@graph_output("
                     "location=) for a graph system)")


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
    *take*: indices of the entities of an active subgraph (where=), the others being left out.
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
    Active subgraph of a graph system solved with where=, or one
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
    cache = instance.caches().get("graph_pieces")
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
    instance.caches()["graph_pieces"] = (key, pieces)
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
        raise ValueError(f"{type(instance).__name__}: the filter of a graph system (where=) selects nodes (or grid cells)")
    key = (ds.topology_version, ds.mask_version(where))
    cache = instance.caches().setdefault("graph_restrictions", {})
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
        if (spec is not None and spec.role == "parameter" and ds.has(name)
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
                             f"or edge arrays: declare it with location='node' and aggregation='broadcast'")
        size = len(node_vids_int) if loc == "node" else len(edge_vids_int)
        # Parameters and inputs: read-only views, so that equations cannot overwrite them
        (node_snap if loc == "node" else edge_snap)[name] = _read_array(ds, name, loc, size, type(instance).__name__,
                                                                        take=_take(instance, loc), read_only=True)
    return node_snap, edge_snap


# ═══════════════════════════════════════════════════════════════════════════════
# _GraphSystemBuilder
# Replaces the inline assembly logic of _invoke_graph_system with a class
# ═══════════════════════════════════════════════════════════════════════════════

class _GraphSystemBuilder:
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
        bc_items           = []   # (field, types, condition, attr_name, bound, raw, explicit)
        jacobian_raw       = None
        explicit_items     = []   # (field, bound method, its Jacobian's method name): @explicit_term
        self.input_jacobians = {} # {variable: (bound, raw)}: @input_jacobian
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
                if tag["term"] == "boundary_set":
                    boundary_sets.append(obj)          # declared data, not a method
                    continue
                bound = obj.__get__(instance, type(instance))
                term = tag["term"]
                key   = f"{type(instance).__name__}.{inner_cls.__name__}.{attr_name}"
                selection = None
                if tag.get("where") is not None:
                    selection = _EntityFilter(tag["where"], key, "edge" if term == "edge_law" else "node")
                if term in ("node_balance", "node_rate") and tag.get("explicit_term"):
                    explicit_items.append((tag["field"], bound, tag.get("term_jacobian")))
                    if _EXTRACTING:
                        continue                       # the archetype's operator: the equations alone
                    node_balance_items.append((tag["field"], selection, attr_name, bound, obj, False))
                elif term in ("node_balance", "node_rate"):
                    node_balance_items.append((
                        tag["field"], selection, attr_name, bound, obj,
                        "rate" if term == "node_rate" else tag.get("explicit", False)
                    ))
                elif term == "input_jacobian":
                    self.input_jacobians[tag["name"]] = (bound, obj)
                elif term == "edge_law":
                    edge_law_items.append((
                        tag.get("field"), selection, attr_name, bound, obj,
                        tag.get("explicit", False), tag.get("integrate", False)
                    ))
                elif term == "boundary_condition":
                    bc_items.append((
                        tag.get("field"), selection, tag.get("condition"),
                        attr_name, bound, obj, tag.get("explicit", False)
                    ))
                elif term == "pool_balance":
                    pool_items.append((tag["field"], bound, obj))
                elif term == "graph_jacobian":
                    jacobian_raw = (bound, obj)
                elif term == "graph_output":
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
            dirichlet += [b.name for b in boundary_sets if b.condition in ("dirichlet", "per_node")]
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
            + [getattr(inner_cls, name) for _, _, name in explicit_items if name is not None]
            + [raw for _, raw in self.input_jacobians.values()]
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
        set_terms = defaultdict(list)    # field -> [(condition, idx, value, weight)]
        if _EXTRACTING:                  # an archetype's operator: the equations alone (the groups' conditions stay live)
            boundary_sets, bc_items = [], []
        for bset in boundary_sets:
            if bset.condition is None:
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
            if bset.condition == "per_node":
                codes = np.asarray(read(bset.condition_variable)).round().astype(np.int64)
                value, weight = read(bset.value), read(bset.weight)
                for condition, code in boundary_set.CODES.items():
                    chosen = codes == code
                    set_terms[field].append((condition, idx[chosen], value[chosen],
                                             weight[chosen] if condition == "robin" else None))
                continue
            set_terms[field].append((bset.condition, idx, read(bset.value), read(bset.weight) if bset.condition == "robin"
                                     else None))

        if instance.__dict__.get("_restriction") is not None and not (spec_def.get("transient", False) or rate_form):
            anchored = np.zeros(n, dtype=bool)
            for _, tf, condition, _, _, _, _ in bc_items:
                if condition == "dirichlet":
                    anchored |= tf.mask(ds, _take(instance, "node")) if tf is not None else True
            for terms in set_terms.values():
                for condition, idx, _, weight in terms:
                    if condition == "dirichlet":
                        anchored[idx] = True
                    elif condition == "robin":
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
            """Entity ("node" or "edge") of an equation argument: a filter slices only those of the filtered entity."""
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
                result = _in_equation(instance, bound_method, args)
                if issparse(result):                           # a sparse analytic Jacobian stays sparse
                    return result.tocsr().astype(np.float64)
                return np.asarray(result, dtype=np.float64)

            return evaluator

        # ── BC factory ────────────────────────────────────────────────────────

        def make_bc_eval(raw_func, bound_method, type_filter,
                          condition="dirichlet", field=None, explicit=False):
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

                if explicit and condition == "dirichlet" and field is not None:
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
                for condition, idx, value, weight in terms:
                    if condition == "robin":
                        result[idx] += weight * (x[idx] - value)
                    elif condition == "neumann":
                        result[idx] -= value * neumann_scale
                for condition, bc_ev in bc_specs:
                    idx, vals = bc_ev(ctx)
                    if condition == "dirichlet":
                        result[idx] = vals
                    else:
                        result[idx] -= vals * neumann_scale         # an inflow, as a boundary_set's value
                for condition, idx, value, _ in terms:
                    if condition == "dirichlet":
                        result[idx] = x[idx] - value
                return result
            return evaluator

        node_groups: dict[str, list] = defaultdict(list)
        for field_name, tf, _, bound, raw, ex in node_balance_items:
            node_groups[field_name].append((tf, bound, raw, ex))

        bc_groups: dict[str, list] = defaultdict(list)
        for field_name, tf, condition, _, bound, raw, ex in bc_items:
            bc_groups[field_name].append((tf, condition, bound, raw, ex))

        reduced = None if _EXTRACTING else _reduced_terms(instance, spec_def, ds, gv)
        if reduced is not None:
            if len(node_unknowns) != 1 or edge_unknowns:
                raise ValueError(f"{system_name}: a reduction needs one node unknown and no edge unknowns")
            node_groups[node_unknowns[0]].append(
                (None, None, None, ("reduced", reduced)))

        equation_blocks = []
        for field_name in node_unknowns:
            grp    = node_groups.get(field_name, [])
            bc_grp = bc_groups.get(field_name, [])
            if not grp and not bc_grp and not set_terms.get(field_name):
                continue
            bulk_wrapped = []
            any_explicit = any(ex is True for _, _, _, ex in grp)
            for tf, b, r, ex in grp:
                if isinstance(ex, tuple) and ex[0] == "reduced":   # the reduced groups' interiors (solve.reduction)
                    D, e = ex[1]
                    bulk_wrapped.append(lambda ctx, _f=field_name, _D=D, _e=e:
                                        np.asarray(_D @ ctx.node_unknowns[_f]).reshape(-1) + _e)
                    continue
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
                (bk, make_bc_eval(r, b, tf, condition=bk, field=field_name, explicit=ex))
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
        self.input_jacobian_evaluators = {name: make_evaluator(raw, bound, None, "node")
                                          for name, (bound, raw) in self.input_jacobians.items()}
        added_jacobians = []             # the explicit terms' own, and the reduced terms'
        if not _EXTRACTING:
            for _, _, name in explicit_items:
                if name is not None:
                    method = getattr(inner_cls, name)
                    added_jacobians.append(make_evaluator(method, method.__get__(instance, type(instance)), None,
                                                          "node"))
        if reduced is not None:
            added_jacobians.append(lambda ctx, _D=reduced[0]: _D)
        if jac_evaluator is not None and added_jacobians:
            equations_jacobian = jac_evaluator

            def jac_evaluator(ctx, _base=equations_jacobian, _added=tuple(added_jacobians)):
                J = _base(ctx)
                for add in _added:
                    J = J + add(ctx)
                return J.tocsr() if issparse(J) else J
        if jac_evaluator is not None and any(set_terms.values()):
            # The user's Jacobian covers the equations; the framework adds the boundary sets' terms
            user_jacobian = jac_evaluator

            def jac_evaluator(ctx, _user=user_jacobian):
                from scipy.sparse import diags, issparse
                J = _user(ctx)
                if issparse(J):
                    # robin: + w on the diagonal; dirichlet: the row becomes the identity's (whatever else it holds)
                    J = J.tocsr()
                    added, fixed = np.zeros(J.shape[0]), np.zeros(J.shape[0], dtype=bool)
                    for field, terms in set_terms.items():
                        offset = list(node_unknowns).index(field) * n
                        for condition, idx, _, weight in terms:
                            rows = offset + idx
                            if condition == "robin":
                                np.add.at(added, rows, np.broadcast_to(weight, rows.shape))
                            elif condition == "dirichlet":
                                fixed[rows] = True
                    J = J + diags(added)
                    if fixed.any():
                        J = diags((~fixed).astype(np.float64)) @ J + diags(fixed.astype(np.float64))
                    return J.tocsr()
                J = np.array(J, dtype=np.float64)
                sparse = False
                for field, terms in set_terms.items():
                    offset = list(node_unknowns).index(field) * n
                    for condition, idx, _, weight in terms:
                        rows = offset + idx
                        if condition == "robin":
                            J[rows, rows] = np.asarray(J[rows, rows]).reshape(-1) + weight
                        elif condition == "dirichlet":
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
        condense = spec_def.get("condense")
        location = condense(instance) if callable(condense) else condense
        spec.parameters["linear_system"] = linear_system(instance, spec_def["name"])
        if location is not None:
            owner = np.asarray(ds.owner(location), dtype=np.int64)
            take = _take(instance, "node")
            kept = np.zeros(n, dtype=bool)
            for _, tf, condition, _, _, _, _ in bc_items:
                if condition == "dirichlet":
                    kept |= tf.mask(ds, take) if tf is not None else True
            for terms in set_terms.values():
                for condition, idx, _, weight in terms:
                    if condition in ("dirichlet", "robin"):
                        kept[idx] = True
            cache = instance.__dict__.setdefault("_condensation_archetypes", {}).setdefault(spec_def["inner_class"].__name__,
                                                                                      {})
            spec.parameters["condensation"] = (owner if take is None else owner[take], kept, cache)
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
        solver = self._spec_def["solver_cls"](self._spec_cfg())
        packed  = solver.step_once(spec, previous_node_fields, dt)
        outputs = solver.derive_outputs(spec, packed, previous_node_fields, dt)
        return packed, outputs

    def _spec_cfg(self) -> SolverConfig:
        sd = self._spec_def
        return SolverConfig(
            method       = sd["method"],
            max_iterations         = sd["max_iterations"],
            absolute_tolerance     = sd["absolute_tolerance"],
            relative_tolerance     = sd.get("relative_tolerance", 0.),
            step_tolerance         = sd["step_tolerance"],
            finite_difference_step = sd["finite_difference_step"],
            prefer_sparse          = sd["prefer_sparse"],
            linesearch             = sd["linesearch"],
            # per unknown: step doubling only
            integration_relative_tolerance = _tolerance(sd["integration_relative_tolerance"], None, 1e-6),
            integration_absolute_tolerance = _tolerance(sd["integration_absolute_tolerance"], None, 1e-9),
            max_step               = np.inf if sd["max_step"] is None else sd["max_step"],
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

def _reduced_terms(instance, spec_def, ds, gv):
    """(D, e) of the reduced groups among the solved nodes (solve.reduction: their interiors' terms, a sparse
    matrix and a vector on the solved nodes), or None."""
    reduction = getattr(getattr(ds, "structure", None), "reduction", None)
    if reduction is None:
        return None
    name = declared_system(instance, spec_def["inner_class"].__name__, reduction.systems)
    if name is None:
        return None
    return reduction.terms(instance, name, ds, np.asarray(gv.node_ids, dtype=np.int64))


def _invoke_graph_system(self, method_name: str) -> None:
    """One call of a graph system (_solve_call), its conserved quantities' balance kept (solve.conservation)."""
    conserved = type(self)._graph_system_specs[method_name].get("conserved")
    if not conserved:
        _solve_call(self, method_name)
        return
    conservation.begin_call(self, _live_ds(self), method_name, conserved)
    _solve_call(self, method_name)
    conservation.end_call(self, _live_ds(self), method_name, conserved)


def _solve_call(self, method_name: str) -> None:
    """
    Build, solve, and write back one named graph-system.

    This is the refactored version of the old monolithic _invoke_graph_system.
    It now delegates each concern to _GraphSystemBuilder:
      build()            → GraphDAESpec construction
      build_and_step()   → spec + solver.step_once() + derive_outputs()
      inject_result()    → write-back to the DataStructure
    """
    # Refresh biological-scale fields from MTG before snapshotting.
    if hasattr(self, "_refresh_from_bio_scale"):
        self._refresh_from_bio_scale()
    spec_def = type(self)._graph_system_specs[method_name]
    if _SKIPPED and declared_system(self, spec_def["inner_class"].__name__, _SKIPPED) is not None:
        return
    where = spec_def.get("where")
    # The active subgraph: the DataStructure mask of the system's filter (where=)
    mask_name = (None if where is None
                 else _EntityFilter(where, f"{type(self).__name__}.{method_name}").mask_name(_live_ds(self)))
    split = spec_def.get("split", "whole") == "components"
    if mask_name is None and not split:
        _solve_graph_system(self, method_name, spec_def)
        return
    restriction = None
    if mask_name is not None:
        # Active subgraph: the equations see its GraphView through self._graph_view
        restriction = _restriction_for(self, mask_name)
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
    saved_key, previous_key = ("solution", method_name), ("previous_solution", method_name)
    caches = self.caches()
    saved = caches.get(saved_key)
    if saved is not None:
        saved = {fn: np.array(values, dtype=np.float64) for fn, values in saved.items()}
        caches[saved_key] = saved
        caches[previous_key] = saved
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
    saved = self.caches().get(key)
    if (restriction is None or saved is None
            or any(fn not in saved or np.size(saved[fn]) != restriction.n for fn in names)):
        self.caches()[key] = {fn: np.array(ds.get(fn), dtype=np.float64).reshape(-1) for fn in names}
        return
    if self.__dict__.get("_in_pieces"):
        for fn in names:
            saved[fn][restriction.node_idx] = np.asarray(ds.get(fn)).reshape(-1)[restriction.node_idx]
        return
    # A new dict of the same arrays: the previous solve's may hold the old dict, whose arrays it must keep
    updated = {}
    for fn in names:
        values = saved[fn] if not _shared_with_previous(self, key, saved[fn]) else np.array(saved[fn])
        values[restriction.node_idx] = np.asarray(ds.get(fn)).reshape(-1)[restriction.node_idx]
        updated[fn] = values
    self.caches()[key] = updated


def _shared_with_previous(self, key, array) -> bool:
    previous = self.caches().get(("previous_solution", key[1])) if key[0] == "solution" else None
    return previous is not None and any(array is other for other in previous.values())


def _solve_graph_system(self, method_name: str, spec_def: dict) -> None:
    """
    Integrate one graph system over the component's time step:
    "once" one solve of time_step; "substeps" n solves of time_step / n; "adaptive" step doubling. During each
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
        if integrate == "once":
            self.__dict__["_current_dt"] = time_step
            _solve_once(self, method_name, spec_def)
        elif integrate == "substeps":
            n = int(spec_def["substeps"])
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
    _saved_fields(self, ("solution", method_name), ds, spec_def["node_unknowns"])
    self.__dict__["_current_dt"] = h
    _solve_once(self, method_name, spec_def)


def _tolerance(value, name, default) -> float:
    """An error-control tolerance of unknown *name*: a number for all unknowns, or {unknown: value} (*default* for
    the unknowns it does not name)."""
    return float(value.get(name, default)) if isinstance(value, dict) else float(value)


def _integrate_adaptive(self, method_name, spec_def, time_step, names) -> None:
    """
    Step doubling: a step of h is compared with two steps of h/2; accepted when the difference is within
    integration_relative_tolerance * |x| + integration_absolute_tolerance on the node unknowns (per unknown when given as dicts), h being adapted after each trial. A
    failed solve (SolverConvergenceError: no convergence, a residual that is not finite...) or a state that is not
    finite rejects the trial too, h halved. Raises when h falls below min_step.
    """
    ds = _live_ds(self)
    relative, absolute = spec_def["integration_relative_tolerance"], spec_def["integration_absolute_tolerance"]
    max_step = min(spec_def.get("max_step") or time_step, time_step)
    min_step = spec_def.get("min_step") or time_step * 1e-6
    t, h, steps, rejected = 0., max_step, 0, 0
    while t < time_step * (1. - 1e-12):
        h = min(h, time_step - t)
        start = _capture(ds, names, self)
        failure = None
        try:
            self.__dict__["_solve_offset"] = t
            trial = conservation.mark(self)
            _solve_substep(self, method_name, spec_def, h)
            coarse = _capture(ds, spec_def["node_unknowns"], self)
            _restore(ds, start, self)
            conservation.reject(self, trial)              # the coarse step only measures the error
            _solve_substep(self, method_name, spec_def, h / 2.)
            self.__dict__["_solve_offset"] = t + h / 2.
            _solve_substep(self, method_name, spec_def, h / 2.)
            fine = _capture(ds, spec_def["node_unknowns"], self)
            error = max((np.max(np.abs(_captured(fine, fn) - _captured(coarse, fn))
                                / (_tolerance(absolute, fn, 1e-9) + _tolerance(relative, fn, 1e-6)
                                   * np.abs(_captured(fine, fn)))) for fn in fine), default=0.)
            if not np.isfinite(error):
                failure = SolverConvergenceError("the solved state is not finite")
        except SolverConvergenceError as error_raised:
            failure = error_raised
        if failure is None and error <= 1.:
            t, steps = t + h, steps + 1
            h *= min(2., max(0.2, 0.9 / np.sqrt(max(error, 1e-12))))
            continue
        _restore(ds, start, self)
        conservation.reject(self, trial)
        rejected += 1
        h *= 0.5 if failure is not None else max(0.2, 0.9 / np.sqrt(error))
        if h < min_step:
            scene_time = self.__dict__.get("_clock", 0.) + t
            if failure is not None:
                raise SolverConvergenceError(
                    f"{type(self).__name__}.{method_name}: the solve failed down to the minimum sub-step "
                    f"min_step={min_step:g} s, at t={t:g} s of the {time_step:g} s step (scene time "
                    f"{scene_time:g} s): {failure}") from failure
            raise RuntimeError(f"{type(self).__name__}.{method_name}: adaptive step below min_step={min_step:g} "
                               f"at t={t:g} of {time_step:g} (scene time {scene_time:g} s)")
    self._last_integration = {"steps": steps, "rejected": rejected}


def _solve_once(self, method_name: str, spec_def: dict) -> None:
    """Build, solve and write back one graph system, on the whole graph or on the current active subgraph."""

    # ── Advance previous-field bookkeeping ────────────────────────────────────
    _saved_key = ("solution", method_name)            # the component's caches (Component.caches)
    _prev_key  = ("previous_solution", method_name)
    caches     = self.caches()
    _saved     = caches.get(_saved_key)
    if _saved is not None:
        caches[_prev_key] = _saved
    previous_fields = caches.get(_prev_key)
    dt = self.__dict__.get("_current_dt", getattr(self, "time_step", None))

    take = _take(self, "node")
    if previous_fields is not None and take is not None:
        previous_fields = {fn: np.asarray(values)[take] for fn, values in previous_fields.items()}

    # ── Build + solve ─────────────────────────────────────────────────────────
    builder = _GraphSystemBuilder(self, spec_def)
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
                _undeclared_output(type(self).__name__, oname)
            ds.register(oname, location=_entity_location(ds, location))
        if restriction is None:
            ds.set(oname, arr)
        else:
            location = ds.location(oname)
            restriction.scatter(ds, oname, arr, location, dropped=None if location == "node" else 0.)
    if spec_def.get("conserved"):                          # its flows, pending until the sub-step is accepted
        conservation.record_substep(self, ds, spec_def["conserved"], float(dt))


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

    @property
    def equations(self):
        """The class holding the graph system's equations (its @node_balance, @edge_law, boundary methods), e.g. to
        call or re-solve them outside the component (Component._flow.equations)."""
        return self._inner_cls

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

        spec_with_cls = {**self._spec, "inner_class": self._inner_cls, "trampoline": _trampoline, "name": name}
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
    edge_unknowns          = (),
    solver                 = "newton",
    max_iterations         = 15,
    absolute_tolerance     = 1e-10,
    relative_tolerance     = 0.,
    step_tolerance         = 1e-10,
    finite_difference_step = 1e-8,
    prefer_sparse          = True,
    linesearch             = False,
    schedule_as            = "axial",
    where                  = None,
    transient              = None,
    integrate              = "once",
    substeps               = 1,
    integration_relative_tolerance = 1e-6,
    integration_absolute_tolerance = 1e-9,
    min_step               = None,
    max_step               = None,
    split                  = "whole",
    pool_unknowns          = None,
    condense               = None,
    conserved              = None,
    symmetric              = False,
):
    """
    Inner-class decorator that wires a graph-system solve into the schedule::

        @dataclass
        class MyModel(FunctionalComponent):

            @graph_system(node_unknowns=["pressure"], solver="newton", schedule_as="axial")
            class _pressure_solve:

                @node_balance(field="pressure")
                def _balance(self, pressure, K, soil_pressure): ...

                @edge_law(field="flux")
                def _darcy(self, pressure, K): ...

    node_unknowns, edge_unknowns: the ordered names of the node (field= of the equations) and edge unknowns.

    solver: a key, "newton" (default), "newton_fd" (finite-difference Jacobian), "explicit_euler",
        "scipy_krylov", "scipy_anderson", "scipy_hybr", "scipy_ivp_bdf" or "scipy_ivp_radau" (a backward-Euler
        step is "newton" with transient=True, the equations writing their time terms or given as @node_rate), or
        a key registered by solve.solver.register_solver. The nonlinear method only: the linear algebra's backend
        is the run's (Compute(linear_solver=), by the system's qualified name or its declared properties).

    max_iterations, absolute_tolerance (||R||_inf), relative_tolerance (converged when ||R||_inf < absolute_tolerance +
        relative_tolerance ||R_0||_inf, R_0 the first iterate's residual; 0: absolute only), step_tolerance
        (converged too when a step ||dx||_inf <= step_tolerance max(1, ||x||_inf): below the unknowns' round-off,
        whatever the residual's floor eps ||J|| ||x||, e.g. a linear system at its second iteration; 1e-10 by
        default, 0: never), finite_difference_step (of the finite-difference Jacobian), prefer_sparse, linesearch
        (Armijo backtracking): the Newton iterations. A failed solve raises solve.solver.SolverConvergenceError,
        its message the residuals' history and the floor.

    schedule_as: the schedule row the solve runs in ("axial", "state", ...).

    where: the nodes of the active subgraph the system is solved on ({variable: condition}, a mask name or a
        callable, conventions.md, Filters): the nodes it selects and the edges between them; the other nodes are frozen, and
        dropped edges carry no flux. None: the whole graph.

    transient: whether the balance has a time term; it only decides the anchoring check (a steady system on an
        active subgraph needs a Dirichlet anchor in every connected piece). Inferred: True with @node_rate or a
        time-stepping solver (explicit_euler, the IVP solvers); give transient=True for a @node_balance writing its
        own time term (self.previous(...)).

    integrate: "once": one solve of the component's time_step (default); "substeps": *substeps* solves of
        time_step / substeps; "adaptive": step doubling with integration_relative_tolerance /
        integration_absolute_tolerance, min_step / max_step, a failed solve (no
        convergence, a residual that is not finite) rejecting the sub-step and halving it down to min_step. Equations
        write their time terms with self.dt and self.previous().

    integration_relative_tolerance, integration_absolute_tolerance, max_step: the error control of "adaptive" and
        of the IVP solvers (scipy_ivp_bdf, scipy_ivp_radau). With "adaptive", both tolerances may be given per node
        unknown, {unknown: value} (1e-6 and 1e-9 for the
        unknowns not named).

    pool_unknowns: unknowns at a coarse scale, one per entity ({name: location} or {name: {"location": ...,
        "exchange": boundary set name}}, e.g. the shoot phloem pool of each plant, location "Plant"), solved with the
        node and edge unknowns. Their residual is a @pool_balance method; equations exchange with them through
        self.pool_exchange(name), the sparse map between the nodes of the exchange set (default: every node) and the
        pool of their entity. Newton solvers only.

    split: "components": each connected piece of the graph (of the active subgraph with where=), e.g. each plant
        of a population, is solved on its own, with its own Newton convergence and integration steps. "whole": one
        system (default).

    condense: the linear solves by static condensation (solve.network.condensed_solve) on the nodes' entities at a
        coarse location (e.g. "SubOrgan": each section's interior eliminated, one factorisation per section archetype),
        or a function of the component giving that location or None (e.g. a setting choosing the solver). The nodes
        of a Robin or Dirichlet condition are kept. Newton solvers, one node unknown, no edge or pool unknowns.

    conserved: the quantities the system conserves, {quantity: solve.conservation.Conservation(amount=, inflow=,
        outflow=, unit=)}: over each call, the inflow and outflow integrated over the accepted sub-steps and the
        change of the amount held, residual = change - (inflow - outflow) (Component.conservation(); the
        SceneRecorder's conservation.csv after every step).

    symmetric: its linear steps' matrix is symmetric (a conductance network's Jacobian, its Dirichlet nodes
        eliminated or held by a Robin term): checked once per pattern (solve.linear.SymmetryError when it is not).
        A property, not a library: the run's Compute(linear_solver=) maps it to a backend, e.g.
        {"symmetric": "cholmod"} (CHOLMOD's Cholesky), or names the system itself ("Component._system").
    """

    _check_where(where, "graph_system")
    pools = {}
    for pool_name, pool_spec in (pool_unknowns or {}).items():
        pool_spec = {"location": pool_spec} if isinstance(pool_spec, str) else dict(pool_spec)
        if "location" not in pool_spec:
            raise ValueError(f"graph_system: pool '{pool_name}' needs a location (e.g. 'Plant')")
        pools[pool_name] = {"location": pool_spec["location"], "exchange": pool_spec.get("exchange")}
    if split not in ("whole", "components"):
        raise ValueError(f"graph_system: split must be 'whole' or 'components', got '{split}'")
    if integrate not in ("once", "substeps", "adaptive"):
        raise ValueError(f"graph_system: integrate must be 'once', 'substeps' or 'adaptive', got '{integrate}'")
    if integrate == "substeps" and int(substeps) < 1:
        raise ValueError("graph_system: substeps must be at least 1")
    for tolerance_name, value in (("integration_relative_tolerance", integration_relative_tolerance),
                                  ("integration_absolute_tolerance", integration_absolute_tolerance)):
        if isinstance(value, dict):
            if integrate != "adaptive":
                raise ValueError(f"graph_system: {tolerance_name} per unknown ({{unknown: value}}) needs "
                                 "integrate='adaptive'")
            unknown = sorted(set(value) - set(node_unknowns))
            if unknown:
                raise ValueError(f"graph_system: {tolerance_name} names {unknown}, not node unknowns")

    if not isinstance(solver, str):
        raise TypeError(f"graph_system(solver=...): a solver key, one of {sorted(_SOLVER_REGISTRY)}, got "
                        f"{solver!r}")
    solver_cls = _SOLVER_REGISTRY.get(solver)
    if solver_cls is None:
        raise ValueError(f"Unknown solver {solver!r}. Valid keys: {sorted(_SOLVER_REGISTRY)}.")
    method_str = solver
    rate_solver = issubclass(solver_cls, (ExplicitEulerSolver, ScipyIVPSolver))
    if condense is not None and (len(list(node_unknowns)) != 1 or list(edge_unknowns) or pools or rate_solver):
        raise ValueError("graph_system: condense= needs a Newton solver, one node unknown and no edge or pool unknowns")

    if pools and rate_solver:
        raise ValueError("graph_system: pool unknowns need a Newton solver (write their time terms in @pool_balance)")

    spec = {
        "node_unknowns" : list(node_unknowns),
        "edge_unknowns" : list(edge_unknowns),
        "solver_cls"    : solver_cls,
        "method"        : method_str,
        "max_iterations": max_iterations,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": float(relative_tolerance),
        "step_tolerance": float(step_tolerance),
        "finite_difference_step": finite_difference_step,
        "prefer_sparse" : prefer_sparse,
        "linesearch"    : linesearch,
        "schedule_as"   : schedule_as,
        "where"       : where,
        "integrate"     : integrate,
        "substeps"      : substeps,
        "integration_relative_tolerance": integration_relative_tolerance,
        "integration_absolute_tolerance": integration_absolute_tolerance,
        "min_step"      : min_step,
        "max_step"      : max_step,
        "split"         : split,
        "pool_unknowns" : pools,
        "transient"     : rate_solver if transient is None else bool(transient),
        "rate_solver"   : rate_solver,
        "condense"      : condense,
        "conserved"     : check_conserved(conserved),
        "symmetric"     : bool(symmetric),
    }

    def decorator(cls):
        return _GraphSystemDescriptor(cls, spec)

    return decorator
