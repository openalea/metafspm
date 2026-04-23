"""
Decorator API for graph-based equation systems in metafspm.

Decorators let a modeler annotate ordinary Python methods with their
mathematical role (node balance, edge constitutive law, Jacobian, …) and
then call ``build_graph_system`` to get a ready-to-solve ``GraphSystem``
without writing any boilerplate assembly code.

Usage
-----
::

    from dataclasses import dataclass
    from openalea.metafspm.component import Model, declare
    from openalea.metafspm.graph_system_decorators import (
        graph_system, node_balance, edge_law, graph_jacobian, graph_output,
        boundary_condition,
    )

    @graph_system(node_unknowns=["pressure"], edge_unknowns=[], method="newton")
    @dataclass
    class MyHydraulics(Model):
        pressure: float = declare(...)

        @node_balance(field="pressure")
        def _balance(self, ctx): ...

        @graph_jacobian
        def _jacobian(self, ctx): ...

        @graph_output(name="edge_flux")
        def _flux(self, ctx): ...

    model = MyHydraulics()
    system = model.build_graph_system(graph, node_fields, edge_fields,
                                      boundary_ports, parameters)
    solution = system.solve()

Decorator summary
-----------------
``@node_balance(field)``
    Tags a method as contributing the residual for the node conservation
    equation of the named unknown field.  Blocks are sorted to match the
    ``node_unknowns`` order given to ``@graph_system``.

``@edge_law``
    Tags a method as an edge constitutive residual.  Blocks follow the
    ``edge_unknowns`` order.

``@boundary_condition(location, kind, field=None)``
    Tags a method as an extra boundary residual block appended after node
    and edge blocks.  Row-substitution semantics are deferred to a future
    version; the modeler is responsible for the equation count.

``@graph_jacobian``
    Tags a method as the optional analytic Jacobian (dense ``np.ndarray``).
    If absent, ``GraphSystem`` uses finite-difference.

``@graph_output(name)``
    Tags a method as a post-solve output hook.

``@graph_system(node_unknowns, edge_unknowns, method, …)``
    Class decorator.  Attaches ``_graph_spec`` and ``build_graph_system``
    to the decorated class.
"""

import sys
import types

from .graph_system import EquationBlock, OutputBlock, GraphSystem, UnknownLayout, SolverSpec
from .component_factory import Choregrapher, Functor


# ── Method-level decorators ────────────────────────────────────────────────────

def node_balance(field=None):
    """Tag a method as the residual block for the named node unknown."""
    def decorator(func):
        func.__graph_tag__ = {"kind": "node_balance", "field": field}
        return func
    return decorator


def edge_law(func):
    """Tag a method as an edge constitutive residual block."""
    func.__graph_tag__ = {"kind": "edge_law"}
    return func


def boundary_condition(location, kind, field=None):
    """Tag a method as a boundary condition residual block."""
    def decorator(func):
        func.__graph_tag__ = {
            "kind": "boundary_condition",
            "location": location,
            "bc_kind": kind,
            "field": field,
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


# ── Class-level decorator ──────────────────────────────────────────────────────

def graph_system(
    node_unknowns,
    edge_unknowns=(),
    method="newton",
    max_iter=15,
    tol=1e-10,
    fd_eps=1e-8,
    prefer_sparse=True,
    schedule_as="axial",
):
    """
    Class decorator that wires tagged methods into a ``GraphSystem``.

    Parameters
    ----------
    node_unknowns : list[str]
        Ordered list of node unknown field names.  The order determines the
        row layout of the assembled residual (node_balance blocks must match).
    edge_unknowns : list[str]
        Ordered list of edge unknown field names.
    method : str
        Solver method passed to ``SolverSpec``.
        ``"newton_fd"`` — Newton with finite-difference Jacobian.
        ``"newton"`` — Newton; uses ``@graph_jacobian`` if present, else FD.
        ``"linear_direct"`` — direct sparse solve (requires matrix/rhs blocks).
    max_iter, tol, fd_eps, prefer_sparse
        Forwarded to ``SolverSpec``.
    schedule_as : str
        Scheduling hint for future Choregrapher integration (not yet used).
    """

    def decorator(cls):
        cls._graph_spec = {
            "node_unknowns": list(node_unknowns),
            "edge_unknowns": list(edge_unknowns),
            "method": method,
            "max_iter": max_iter,
            "tol": tol,
            "fd_eps": fd_eps,
            "prefer_sparse": prefer_sparse,
            "schedule_as": schedule_as,
        }

        def build_graph_system(
            self,
            graph,
            node_fields,
            edge_fields,
            boundary_ports=(),
            parameters=None,
        ):
            """
            Assemble and return a ``GraphSystem`` from this model's tagged methods.

            Parameters
            ----------
            graph : GraphView
                Solver-facing graph (nodes, edges, incidence matrices).
            node_fields : dict[str, FieldState]
                All node fields, including initial-guess values for unknowns.
            edge_fields : dict[str, FieldState]
                All edge fields, including initial-guess values for unknowns.
            boundary_ports : tuple[BoundaryPort, ...]
                Boundary ports already embedded in ``graph.boundary_incidence``.
            parameters : dict, optional
                Scalar / array parameters not expressed as graph fields.
            """
            spec = type(self)._graph_spec
            if parameters is None:
                parameters = {}

            # Collect tagged methods via MRO (subclass methods take priority)
            seen: set[str] = set()
            node_balance_items: list[tuple[str | None, str, object]] = []
            edge_law_items: list[tuple[str, object]] = []
            bc_items: list[tuple[str, object]] = []
            jacobian_evaluator = None
            output_items: list[tuple[str, object]] = []

            for cls_ in type(self).__mro__:
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
                        node_balance_items.append((tag["field"], attr_name, bound))
                    elif kind == "edge_law":
                        edge_law_items.append((attr_name, bound))
                    elif kind == "boundary_condition":
                        bc_items.append((attr_name, bound))
                    elif kind == "graph_jacobian":
                        jacobian_evaluator = bound
                    elif kind == "graph_output":
                        output_items.append((tag["name"], bound))

            # Sort node_balance blocks to match node_unknowns order
            field_order = {f: i for i, f in enumerate(spec["node_unknowns"])}
            node_balance_items.sort(
                key=lambda x: field_order.get(x[0], len(spec["node_unknowns"]))
            )

            # edge_law blocks sorted by method name for stability
            edge_law_items.sort(key=lambda x: x[0])

            equation_blocks = tuple(
                [
                    EquationBlock(name=f"node_balance_{f}", evaluator=m)
                    for f, _, m in node_balance_items
                ]
                + [
                    EquationBlock(name=f"edge_law_{n}", evaluator=m)
                    for n, m in edge_law_items
                ]
                + [
                    EquationBlock(name=f"bc_{n}", evaluator=m)
                    for n, m in bc_items
                ]
            )

            output_blocks = tuple(
                OutputBlock(name=n, evaluator=m) for n, m in output_items
            )

            unknowns = UnknownLayout(
                node_fields=tuple(spec["node_unknowns"]),
                edge_fields=tuple(spec["edge_unknowns"]),
            )
            solver = SolverSpec(
                method=spec["method"],
                max_iter=spec["max_iter"],
                tol=spec["tol"],
                fd_eps=spec["fd_eps"],
                prefer_sparse=spec["prefer_sparse"],
            )

            return GraphSystem(
                graph=graph,
                node_fields=node_fields,
                edge_fields=edge_fields,
                boundary_ports=tuple(boundary_ports),
                unknowns=unknowns,
                solver=solver,
                equation_blocks=equation_blocks,
                output_blocks=output_blocks,
                jacobian_evaluator=jacobian_evaluator,
                parameters=parameters,
            )

        cls.build_graph_system = build_graph_system

        # GAP 1 fix: wire _run_graph_system into the Choregrapher under cls.__name__.
        # We create _graph_solve with the decorated class's module globals so that the
        # Choregrapher's `inheriting` lookup finds any parent-functor transfer declarations
        # in the calling module rather than in this decorator module.
        if hasattr(cls, "_run_graph_system"):
            mod_globals = vars(sys.modules[cls.__module__]) if cls.__module__ in sys.modules else {}

            def _gs(self):
                self._run_graph_system()

            _gs = types.FunctionType(_gs.__code__, mod_globals, "_graph_solve")
            _gs.__qualname__ = f"{cls.__name__}._graph_solve"
            Choregrapher().add_process(Functor(_gs), name=schedule_as)

        return cls

    return decorator
