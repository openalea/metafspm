"""
Prototype answer to two questions:

1. How could the generic equation-block formulation be used to reimplement
   `axial_transport_N_arrays` from Root-CyNAPS?
2. How could the same system be declared with decorators in a way that extends
   MetaFSPM's current "signature-driven per-node process" logic to edge-aware
   systems?

This file deliberately stays in the test suite because it is an architecture
prototype, not a production implementation.

The strategy is:

- first write a Root-CyNAPS-like axial transport problem directly with generic
  residual callbacks
- then write the *same* problem using decorators that attach metadata to
  methods, while still resolving method arguments from their signature names
  like MetaFSPM already does for node-local processes

The biological mapping to `axial_transport_N_arrays` is the following:

- node unknown      : solute amount in each conducting element
- edge unknown      : axial solute flux
- node fields       : previous amount, conductive volume, radial source,
                      boundary source
- edge fields       : water flow and axial diffusivity
- node equation     : storage + divergence(axial_flux) - local_sources = 0
- edge equation     : axial_flux - edge_law(concentration gradient, advection) = 0

The important idea is that the *system class* does not know nitrogen transport.
It only sees residual blocks. The model-specific logic lives in callbacks or in
decorated methods that are turned into callbacks.
"""

from dataclasses import dataclass
import inspect as ins

import numpy as np

from test_mtg_to_generic_equation_system import (
    EquationBlock,
    EquationContext,
    GenericEquationSystem,
    GenericSolverSpec,
)
from test_mtg_to_network import FieldState, UnknownLayout, build_cell_symplast_open_graph


def build_root_cynaps_like_transport_fields():
    """
    Create one tiny axial transport problem on the 3-node / 2-edge chain built
    from the MTG in the other tests.

    This is not a literal copy of Root-CyNAPS parameters; it is a small system
    that mirrors the *decomposition* of `axial_transport_N_arrays`.
    """
    _, graph, boundary_fluxes = build_cell_symplast_open_graph()

    node_fields = {
        # Unknown updated by the implicit transport solve.
        "solute_amount": FieldState(
            name="solute_amount",
            location="node",
            values=np.asarray([0.4, 0.25, 0.1], dtype=np.float64),
        ),
        # Previous time step state, needed for backward Euler.
        "previous_amount": FieldState(
            name="previous_amount",
            location="node",
            values=np.asarray([0.4, 0.25, 0.1], dtype=np.float64),
        ),
        # Equivalent to the "conductive element volume" used in Root-CyNAPS.
        "volume": FieldState(
            name="volume",
            location="node",
            values=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        ),
        # Aggregate of radial exchange terms, analogous to `R_others`.
        "radial_source": FieldState(
            name="radial_source",
            location="node",
            values=np.asarray([0.08, 0.03, 0.01], dtype=np.float64),
        ),
        # Generic boundary contribution already distributed on nodes.
        # This mirrors the idea of `boundary_inflow` / `boundary_outflow`
        # arrays in `axial_transport_N_arrays`.
        "boundary_source": FieldState(
            name="boundary_source",
            location="node",
            values=np.asarray([0.05, 0.0, 0.0], dtype=np.float64),
        ),
    }

    edge_fields = {
        # Unknown updated by the solve.
        "axial_flux": FieldState(
            name="axial_flux",
            location="edge",
            values=np.asarray([0.12, 0.08], dtype=np.float64),
        ),
        # Analogous to the axial water flow used for advection.
        "water_flow": FieldState(
            name="water_flow",
            location="edge",
            values=np.asarray([0.25, 0.18], dtype=np.float64),
        ),
        # Analogous to D in the Root-CyNAPS advection-diffusion operator.
        "axial_diffusivity": FieldState(
            name="axial_diffusivity",
            location="edge",
            values=np.asarray([0.07, 0.05], dtype=np.float64),
        ),
    }

    return graph, boundary_fluxes, node_fields, edge_fields


def root_cynaps_like_node_balance(ctx: EquationContext) -> np.ndarray:
    """
    Root-CyNAPS-like node transport equation written as one residual block.

    This is the direct conceptual equivalent of the implicit amount balance that
    `axial_transport_N_arrays` solves after assembling diffusion, advection, and
    radial terms:

        n - n_prev + dt * (B q - R_radial - R_boundary) = 0

    Here:
    - n         is the updated solute amount
    - n_prev    is the previous amount
    - B q       is the axial divergence term
    - R_radial  is the aggregate radial source/sink
    - R_boundary is the collar/shoot contribution already projected on nodes
    """
    if ctx.dt is None:
        raise AssertionError("This Root-CyNAPS-like prototype is meant for transient BE use")

    amount = ctx.node_unknowns["solute_amount"]
    previous_amount = ctx.node_fields["previous_amount"].values
    radial_source = ctx.node_fields["radial_source"].values
    boundary_source = ctx.node_fields["boundary_source"].values
    axial_flux = ctx.edge_unknowns["axial_flux"]

    divergence = ctx.graph.incidence @ axial_flux
    return amount - previous_amount + float(ctx.dt) * (divergence - radial_source - boundary_source)


def root_cynaps_like_edge_transport(ctx: EquationContext) -> np.ndarray:
    """
    Root-CyNAPS-like edge equation.

    This mirrors the advection-diffusion structure inside
    `axial_transport_N_arrays`, but written in mixed form with explicit edge
    flux unknowns.

    For each edge e = (tail -> head), we use:

        q_e = u_e * c_tail - D_e * (c_head - c_tail)

    where:
    - u_e is the edge water flow
    - D_e is the edge axial diffusivity
    - c = amount / volume is the node concentration

    This keeps the same decomposition as the Root-CyNAPS matrix assembly:
    the node equation sees a divergence of edge fluxes, and the edge equation
    defines those fluxes from node states and edge states.
    """
    amount = ctx.node_unknowns["solute_amount"]
    volume = ctx.node_fields["volume"].values
    axial_flux = ctx.edge_unknowns["axial_flux"]
    water_flow = ctx.edge_fields["water_flow"].values
    axial_diffusivity = ctx.edge_fields["axial_diffusivity"].values

    concentration = amount / volume
    c_tail = concentration[ctx.graph.tail]
    c_head = concentration[ctx.graph.head]

    target_flux = water_flow * c_tail - axial_diffusivity * (c_head - c_tail)
    return axial_flux - target_flux


def build_root_cynaps_like_generic_system():
    """
    Assemble the Root-CyNAPS-like transport problem explicitly from callbacks.

    This is the "manual generic" formulation: it already reimplements the
    monolithic axial method as two residual blocks.
    """
    graph, boundary_fluxes, node_fields, edge_fields = build_root_cynaps_like_transport_fields()

    return GenericEquationSystem(
        graph=graph,
        node_fields=node_fields,
        edge_fields=edge_fields,
        boundary_fluxes=boundary_fluxes,
        unknowns=UnknownLayout(
            node_fields=("solute_amount",),
            edge_fields=("axial_flux",),
        ),
        equation_blocks=(
            EquationBlock(name="node_balance", evaluator=root_cynaps_like_node_balance),
            EquationBlock(name="edge_transport", evaluator=root_cynaps_like_edge_transport),
        ),
        solver=GenericSolverSpec(method="newton_fd"),
    )


def test_generic_system_can_reimplement_root_cynaps_axial_transport_arrays_logic():
    """
    This is the first answer requested by the user.

    It shows that a Root-CyNAPS-like axial transport system can be expressed as:
    - one node balance equation block
    - one edge transport equation block
    - one unknown layout
    - one generic solver

    instead of one monolithic `axial_transport_N_arrays` method.
    """
    system = build_root_cynaps_like_generic_system()
    solution = system.solve(dt=0.5)
    node_unknowns, edge_unknowns = system.unpack_unknowns(solution)

    np.testing.assert_allclose(
        system.residual(solution, dt=0.5),
        np.zeros_like(system.residual(solution, dt=0.5)),
        atol=1e-10,
    )
    assert np.all(node_unknowns["solute_amount"] > 0.0)
    assert np.all(edge_unknowns["axial_flux"] > 0.0)


# -----------------------------------------------------------------------------
# Prototype decorators extending MetaFSPM's signature-driven logic
# -----------------------------------------------------------------------------


def axial_node_equation(func):
    """
    Prototype decorator for node residual blocks.

    In production MetaFSPM, this could extend the decorator family alongside
    `@rate`, `@state`, and `@axial`.
    """
    func._axial_equation_role = "node"
    return func


def axial_edge_equation(func):
    """Prototype decorator for edge residual blocks."""
    func._axial_equation_role = "edge"
    return func


def axial_unknown_layout(func):
    """Prototype decorator returning the unknown layout of the system."""
    func._axial_equation_role = "unknowns"
    return func


def axial_solver(func):
    """Prototype decorator returning the solver configuration."""
    func._axial_equation_role = "solver"
    return func


@dataclass
class DecoratedRootCyNAPSLikeComponent:
    """
    Prototype of a component declared with decorators.

    The important point is that method signatures now describe dependencies on:
    - node unknown arrays
    - edge unknown arrays
    - fixed node fields
    - fixed edge fields
    - graph helper arrays such as `incidence`, `tail_index`, `head_index`
    - time step `dt`

    This mirrors MetaFSPM's current "signature -> data dependency" logic, but
    extends it from per-node scalar computations to graph-wide vector equations.
    """

    @axial_unknown_layout
    def _unknown_layout(self):
        return UnknownLayout(
            node_fields=("solute_amount",),
            edge_fields=("axial_flux",),
        )

    @axial_solver
    def _solver(self):
        return GenericSolverSpec(method="newton_fd")

    @axial_node_equation
    def _node_balance(
        self,
        solute_amount,
        previous_amount,
        radial_source,
        boundary_source,
        incidence,
        axial_flux,
        dt,
    ):
        """
        Same equation as `root_cynaps_like_node_balance`, but written in the
        "MetaFSPM style": inputs are named dependencies resolved from the method
        signature.
        """
        return solute_amount - previous_amount + float(dt) * (
            incidence @ axial_flux - radial_source - boundary_source
        )

    @axial_edge_equation
    def _edge_transport(
        self,
        solute_amount,
        volume,
        water_flow,
        axial_diffusivity,
        tail_index,
        head_index,
        axial_flux,
    ):
        """
        Same edge law as `root_cynaps_like_edge_transport`, but now the graph
        helper arrays are injected through the method signature.
        """
        concentration = solute_amount / volume
        c_tail = concentration[tail_index]
        c_head = concentration[head_index]
        target_flux = water_flow * c_tail - axial_diffusivity * (c_head - c_tail)
        return axial_flux - target_flux


def make_decorated_equation_block(component, method):
    """
    Turn one decorated method into a generic `EquationBlock`.

    This is the exact bridge to MetaFSPM's current decorator system:
    the method signature is inspected, and each parameter name is resolved from
    the current graph / fields / unknowns environment.
    """
    method_name = method.__name__
    arg_names = [p.name for p in ins.signature(method).parameters.values()]

    def evaluator(ctx: EquationContext):
        # Environment made available to decorated methods.
        env = {
            "incidence": ctx.graph.incidence,
            "tail_index": ctx.graph.tail,
            "head_index": ctx.graph.head,
            "dt": ctx.dt,
        }

        # Unknowns have priority over fixed fields with the same name because the
        # whole point is to inject the current iterate into the equations.
        env.update(ctx.node_fields)
        env.update(ctx.edge_fields)
        env.update(ctx.node_unknowns)
        env.update(ctx.edge_unknowns)

        # FieldState objects should resolve to their `.values`.
        resolved = []
        for name in arg_names:
            value = env[name]
            if isinstance(value, FieldState):
                resolved.append(value.values)
            else:
                resolved.append(value)

        return np.asarray(method(*resolved), dtype=np.float64)

    return EquationBlock(name=method_name, evaluator=evaluator)


def build_system_from_decorated_component(component, graph, boundary_fluxes, node_fields, edge_fields):
    """
    Collect decorated methods from the component and turn them into a
    `GenericEquationSystem`.

    This is the second answer requested by the user:
    it shows how MetaFSPM could evolve from per-node Functors to graph equation
    blocks while preserving the idea that decorators and method signatures
    declare dependencies.
    """
    equation_blocks = []
    unknown_layout = None
    solver = None

    # Iterate on class __dict__ to preserve declaration order.
    for _, value in component.__class__.__dict__.items():
        if not callable(value):
            continue
        role = getattr(value, "_axial_equation_role", None)
        if role == "unknowns":
            unknown_layout = value(component)
        elif role == "solver":
            solver = value(component)
        elif role in ("node", "edge"):
            bound_method = value.__get__(component, component.__class__)
            equation_blocks.append(make_decorated_equation_block(component, bound_method))

    return GenericEquationSystem(
        graph=graph,
        node_fields=node_fields,
        edge_fields=edge_fields,
        boundary_fluxes=boundary_fluxes,
        unknowns=unknown_layout,
        equation_blocks=tuple(equation_blocks),
        solver=solver,
    )


def test_decorators_can_define_same_axial_transport_system_as_manual_generic_blocks():
    """
    This is the second answer requested by the user.

    It shows how a component written with decorators can declare the *same*
    system as the manual generic formulation above.

    The comparison is important:
    - same graph
    - same fields
    - same unknowns
    - same equations
    - same solution

    Only the declaration style changes.
    """
    graph, boundary_fluxes, node_fields, edge_fields = build_root_cynaps_like_transport_fields()

    manual_system = build_root_cynaps_like_generic_system()
    decorated_component = DecoratedRootCyNAPSLikeComponent()
    decorated_system = build_system_from_decorated_component(
        decorated_component,
        graph=graph,
        boundary_fluxes=boundary_fluxes,
        node_fields=node_fields,
        edge_fields=edge_fields,
    )

    manual_solution = manual_system.solve(dt=0.5)
    decorated_solution = decorated_system.solve(dt=0.5)

    np.testing.assert_allclose(manual_solution, decorated_solution, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        decorated_system.residual(decorated_solution, dt=0.5),
        np.zeros_like(decorated_system.residual(decorated_solution, dt=0.5)),
        atol=1e-10,
    )


if __name__ == "__main__":
    test_generic_system_can_reimplement_root_cynaps_axial_transport_arrays_logic()
    test_decorators_can_define_same_axial_transport_system_as_manual_generic_blocks()
