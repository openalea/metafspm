"""
This file is a prototype, not just a unit test.

The goal is to explore how an MTG can be converted into a generic "open graph"
representation on which we can define:

1. node states
2. edge states
3. boundary conditions
4. residual equations
5. a solver / unknown layout

The concrete example chosen here is intentionally very small:

- nodes carry a scalar "potential"
- edges carry a scalar "flux"
- edges also have a fixed "conductance"
- one node has a fixed Dirichlet value
- one boundary injects/extracts a prescribed flux

Mathematically, the steady problem is:

    B q + B_b q_b = 0
    q - K (B^T p) = 0

where:

- p   : node potentials
- q   : edge fluxes
- K   : edge conductances
- B   : node-edge incidence matrix
- B_b : node-boundary incidence matrix
- q_b : prescribed boundary fluxes

The transient variant simply adds a storage term on the nodes:

    C (p - p_old) / dt + B q + B_b q_b = 0

This is enough to prototype the API shape for more complex transport systems.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, diags, eye, lil_matrix
from scipy.sparse.linalg import spsolve

from generate_mtg import (
    build_seedling_mtg,
    e_type,
    get_representative_segment_id,
    n_type,
    scales,
)


@dataclass(frozen=True)
class FieldState:
    """Description of one field living either on nodes or on edges."""

    name: str
    location: str
    values: np.ndarray


@dataclass(frozen=True)
class DirichletNodeValue:
    """
    Strong boundary condition on a node field.

    In this prototype we only use it for the node potential, but the structure
    is meant to be generic enough for any node field.
    """

    field: str
    node_id: int
    value: float


@dataclass(frozen=True)
class BoundaryFlux:
    """
    Boundary port attached to a node.

    `orientation` is kept explicit because, in a generic formulation, boundary
    ports should behave like signed columns of an incidence matrix.
    """

    name: str
    node_id: int
    flux: float
    orientation: float = 1.0


@dataclass(frozen=True)
class UnknownLayout:
    """
    Ordered list of fields that are part of the nonlinear / linear solve.

    The important point is that a field may exist in the model but not be an
    unknown of the current resolution strategy.
    """

    node_fields: tuple[str, ...]
    edge_fields: tuple[str, ...]


@dataclass(frozen=True)
class SolverSpec:
    """Minimal placeholder for the chosen solver strategy."""

    method: str


@dataclass(frozen=True)
class OpenGraph:
    """
    Generic graph extracted from the MTG subset we want to solve on.

    This is the real bridge between MTG structure and solver structure.

    It stores:
    - the chosen node ids
    - the chosen edge ids
    - local tail/head indexing for the edges
    - the standard incidence matrix
    - an additional "boundary incidence" for external ports

    In a richer implementation, this object would likely also cache:
    - masks by scale / type
    - grouped adjacency
    - sparse matrix patterns
    - invalidation flags for growing MTGs
    """

    node_ids: np.ndarray
    edge_ids: np.ndarray
    tail: np.ndarray
    head: np.ndarray
    incidence: csc_matrix
    boundary_incidence: csc_matrix
    boundary_names: tuple[str, ...]

    @classmethod
    def from_mtg_subset(
        cls,
        g,
        node_scale: int,
        node_ids: np.ndarray,
        edge_scale: int,
        edge_ids: np.ndarray,
        boundary_fluxes: tuple[BoundaryFlux, ...] = (),
    ):
        """
        Build an "open graph" from a subset of MTG nodes and edges.

        The MTG is still the source of truth, but solvers usually need a compact
        local indexing:

        - node ids become local rows
        - edge ids become local columns
        - `tail` and `head` define the orientation of each edge

        The orientation convention used here is:
        - +1 on the tail
        - -1 on the head

        so that `B @ q` gives the node balance induced by edge fluxes.
        """
        node_ids = np.sort(np.asarray(node_ids, dtype=np.int64))
        edge_ids = np.sort(np.asarray(edge_ids, dtype=np.int64))

        # Read the whole MTG arrays first. We then slice them to the subnetwork
        # we want to solve on.
        all_node_ids = np.asarray(g.array_at_scale("vertex_id", scale=node_scale), dtype=np.int64)
        all_edge_ids = np.asarray(g.array_at_scale("vertex_id", scale=edge_scale), dtype=np.int64)
        all_n_id_a = np.asarray(g.array_at_scale("n_id_a", scale=edge_scale), dtype=np.int64)
        all_n_id_b = np.asarray(g.array_at_scale("n_id_b", scale=edge_scale), dtype=np.int64)

        # Keep only the requested MTG edges.
        edge_mask = np.isin(all_edge_ids, edge_ids)
        selected_n_id_a = all_n_id_a[edge_mask]
        selected_n_id_b = all_n_id_b[edge_mask]

        # Convert MTG node ids into compact local indices [0..n_nodes-1].
        tail = np.searchsorted(node_ids, selected_n_id_a)
        head = np.searchsorted(node_ids, selected_n_id_b)
        edge_cols = np.arange(edge_ids.size, dtype=np.int64)

        # Build the classical incidence matrix B.
        #
        # One column per edge:
        # - +1 at the start node
        # - -1 at the end node
        #
        # This is the core object for most graph-based formulations.
        incidence = coo_matrix(
            (
                np.r_[np.ones(edge_ids.size), -np.ones(edge_ids.size)],
                (np.r_[tail, head], np.r_[edge_cols, edge_cols]),
            ),
            shape=(node_ids.size, edge_ids.size),
        ).tocsc()

        if boundary_fluxes:
            # Boundary ports are encoded in exactly the same spirit as edges:
            # one column per external boundary action.
            boundary_rows = np.asarray(
                [np.searchsorted(node_ids, bc.node_id) for bc in boundary_fluxes],
                dtype=np.int64,
            )
            boundary_cols = np.arange(len(boundary_fluxes), dtype=np.int64)
            boundary_data = np.asarray([bc.orientation for bc in boundary_fluxes], dtype=np.float64)
            boundary_incidence = coo_matrix(
                (boundary_data, (boundary_rows, boundary_cols)),
                shape=(node_ids.size, len(boundary_fluxes)),
            ).tocsc()
            boundary_names = tuple(bc.name for bc in boundary_fluxes)
        else:
            boundary_incidence = csc_matrix((node_ids.size, 0), dtype=np.float64)
            boundary_names = ()

        return cls(
            node_ids=node_ids,
            edge_ids=edge_ids,
            tail=tail,
            head=head,
            incidence=incidence,
            boundary_incidence=boundary_incidence,
            boundary_names=boundary_names,
        )

    @property
    def n_nodes(self):
        return self.node_ids.size

    @property
    def n_edges(self):
        return self.edge_ids.size

    def node_local_index(self, node_id: int) -> int:
        return int(np.searchsorted(self.node_ids, int(node_id)))


@dataclass
class MixedTransportSystem:
    """
    Small residual-based system living on an OpenGraph.

    This is the main prototype for a generic API:
    - the graph gives topology
    - node_fields / edge_fields give data
    - unknowns selects what is solved for
    - residual defines the equation system
    - jacobian optionally provides derivatives
    - solve applies one numerical strategy
    """

    graph: OpenGraph
    node_fields: dict[str, FieldState]
    edge_fields: dict[str, FieldState]
    boundary_fluxes: tuple[BoundaryFlux, ...]
    dirichlet: tuple[DirichletNodeValue, ...]
    unknowns: UnknownLayout
    solver: SolverSpec

    def pack_unknowns(self, node_overrides=None, edge_overrides=None):
        """
        Convert the selected unknown fields into one solver vector.

        This is the usual "layout" step of equation-based solvers:
        all unknown blocks are concatenated in a deterministic order.
        """
        node_overrides = {} if node_overrides is None else node_overrides
        edge_overrides = {} if edge_overrides is None else edge_overrides
        blocks = []

        for field_name in self.unknowns.node_fields:
            blocks.append(
                np.asarray(
                    node_overrides.get(field_name, self.node_fields[field_name].values),
                    dtype=np.float64,
                )
            )
        for field_name in self.unknowns.edge_fields:
            blocks.append(
                np.asarray(
                    edge_overrides.get(field_name, self.edge_fields[field_name].values),
                    dtype=np.float64,
                )
            )

        return np.concatenate(blocks)

    def unpack_unknowns(self, packed):
        """
        Reverse operation of `pack_unknowns`.

        Solvers operate on a flat vector, but model equations are easier to read
        when we recover named node and edge blocks.
        """
        packed = np.asarray(packed, dtype=np.float64)
        cursor = 0
        node_unknowns = {}
        edge_unknowns = {}

        for field_name in self.unknowns.node_fields:
            width = self.graph.n_nodes
            node_unknowns[field_name] = packed[cursor:cursor + width]
            cursor += width

        for field_name in self.unknowns.edge_fields:
            width = self.graph.n_edges
            edge_unknowns[field_name] = packed[cursor:cursor + width]
            cursor += width

        return node_unknowns, edge_unknowns

    def boundary_flux_vector(self):
        """Extract prescribed boundary flux values in boundary-column order."""
        return np.asarray([bc.flux for bc in self.boundary_fluxes], dtype=np.float64)

    def residual(self, packed_unknowns, previous_potential=None, dt=None):
        """
        Residual form of the mixed node/edge system.

        Unknowns in this prototype:
        - node potential `p`
        - edge flux `q`

        Equations:
        - node balance: B q + B_b q_b = 0
        - constitutive edge law: q - K (B^T p) = 0

        If `dt` is provided we add a backward-Euler storage term:
        - C (p - p_old) / dt + B q + B_b q_b = 0

        Dirichlet conditions are enforced by overwriting the corresponding node
        residual row with:
            p_i - p_i_prescribed = 0
        """
        node_unknowns, edge_unknowns = self.unpack_unknowns(packed_unknowns)
        potential = node_unknowns["potential"]
        flux = edge_unknowns["flux"]
        conductance = self.edge_fields["conductance"].values

        # Node balance induced by internal edge fluxes.
        node_residual = self.graph.incidence @ flux
        if self.boundary_fluxes:
            # Add external boundary ports to the node balance.
            node_residual = node_residual + self.graph.boundary_incidence @ self.boundary_flux_vector()

        if dt is not None:
            # Backward-Euler accumulation on the nodes.
            previous_potential = (
                self.node_fields["potential"].values
                if previous_potential is None
                else np.asarray(previous_potential, dtype=np.float64)
            )
            capacity = self.node_fields["capacity"].values
            node_residual = node_residual + capacity * (potential - previous_potential) / float(dt)

        # Constitutive law on edges:
        #   q = K * delta_p
        # Here delta_p is obtained by applying the transpose of the incidence.
        edge_residual = flux - conductance * (self.graph.incidence.T @ potential)
        residual = np.concatenate([node_residual, edge_residual])

        for bc in self.dirichlet:
            # Strong Dirichlet imposition by replacing the natural balance row.
            local_idx = self.graph.node_local_index(bc.node_id)
            residual[local_idx] = potential[local_idx] - bc.value

        return residual

    def analytic_jacobian(self, dt=None):
        """
        Analytic Jacobian of the residual.

        This is written explicitly here because the equations are simple.
        In a more generic framework, this could be:
        - provided by the modeler
        - derived by automatic differentiation
        - approximated by finite differences
        """
        if dt is None:
            node_mass = np.zeros(self.graph.n_nodes, dtype=np.float64)
        else:
            node_mass = self.node_fields["capacity"].values / float(dt)

        conductance = self.edge_fields["conductance"].values
        jacobian = lil_matrix(
            (
                self.graph.n_nodes + self.graph.n_edges,
                self.graph.n_nodes + self.graph.n_edges,
            ),
            dtype=np.float64,
        )

        # Block structure of the mixed Jacobian:
        #
        # [ dR_node/dp   dR_node/dq ]
        # [ dR_edge/dp   dR_edge/dq ]
        jacobian[:self.graph.n_nodes, :self.graph.n_nodes] = diags(node_mass)
        jacobian[:self.graph.n_nodes, self.graph.n_nodes:] = self.graph.incidence
        jacobian[self.graph.n_nodes:, :self.graph.n_nodes] = -diags(conductance) @ self.graph.incidence.T
        jacobian[self.graph.n_nodes:, self.graph.n_nodes:] = eye(self.graph.n_edges, format="csc")

        for bc in self.dirichlet:
            # If a node residual row is replaced by a Dirichlet equation, then
            # its Jacobian row must also be replaced by [0 ... 1 ... 0].
            row = self.graph.node_local_index(bc.node_id)
            jacobian.rows[row] = [row]
            jacobian.data[row] = [1.0]

        return jacobian.tocsc()

    def finite_difference_jacobian(self, packed_unknowns, previous_potential=None, dt=None, eps=1e-8):
        """
        Numerical fallback Jacobian.

        This is useful for prototyping because it shows that the residual API is
        already enough to drive a solver, even before analytic Jacobians or AD
        are introduced.
        """
        packed_unknowns = np.asarray(packed_unknowns, dtype=np.float64)
        base_residual = self.residual(packed_unknowns, previous_potential=previous_potential, dt=dt)
        jacobian = np.zeros((base_residual.size, packed_unknowns.size), dtype=np.float64)

        for k in range(packed_unknowns.size):
            delta = np.zeros_like(packed_unknowns)
            delta[k] = eps
            forward = self.residual(
                packed_unknowns + delta,
                previous_potential=previous_potential,
                dt=dt,
            )
            backward = self.residual(
                packed_unknowns - delta,
                previous_potential=previous_potential,
                dt=dt,
            )
            jacobian[:, k] = (forward - backward) / (2 * eps)

        return jacobian

    def solve(self, previous_potential=None, dt=None):
        """
        One linear Newton-like correction step.

        Since the prototype equations are linear, one solve is enough to land
        exactly on the solution:

            J delta = -R
            x_new = x_old + delta
        """
        if self.solver.method != "linear_newton":
            raise ValueError(f"Unsupported prototype solver: {self.solver.method}")

        packed = self.pack_unknowns()
        residual = self.residual(packed, previous_potential=previous_potential, dt=dt)
        jacobian = self.analytic_jacobian(dt=dt)
        correction = spsolve(jacobian, -residual)
        return packed + correction


def build_cell_symplast_open_graph():
    """
    Extract a very small subnetwork from one segment inside the seedling MTG.

    We only keep:
    - the cell-center nodes
    - the symplastic edges connecting those cell centers

    This still gives us a tiny 3-node / 2-edge chain, but now that chain comes
    from a realistic seedling-scale MTG instead of from an isolated single
    segment built from scratch.
    """
    g = build_seedling_mtg()
    segment_id = get_representative_segment_id(g)

    node_ids = np.asarray(g.component_roots_at_scale(segment_id, scale=scales["node"]), dtype=np.int64)
    edge_ids = np.asarray(g.component_roots_at_scale(segment_id, scale=scales["edge"]), dtype=np.int64)
    node_type_map = {int(vid): int(g.node(vid).n_type) for vid in node_ids}
    edge_type_map = {int(vid): int(g.node(vid).e_type) for vid in edge_ids}

    # Keep only the "cell" nodes as the coarse unknown locations.
    cell_nodes = np.asarray([vid for vid in node_ids if node_type_map[int(vid)] == n_type["cell"]], dtype=np.int64)
    # Keep only the symplastic edges between these cell nodes.
    symplastic_edges = np.asarray([vid for vid in edge_ids if edge_type_map[int(vid)] == e_type["symplastic"]], dtype=np.int64)

    # Add one external sink at the right-most cell.
    boundary_fluxes = (
        BoundaryFlux(
            name="right_sink",
            node_id=int(cell_nodes.max()),
            flux=0.5,
            orientation=1.0,
        ),
    )

    graph = OpenGraph.from_mtg_subset(
        g=g,
        node_scale=scales["node"],
        node_ids=cell_nodes,
        edge_scale=scales["edge"],
        edge_ids=symplastic_edges,
        boundary_fluxes=boundary_fluxes,
    )
    return g, graph, boundary_fluxes


def build_mixed_transport_prototype():
    """
    Instantiate one complete mixed node/edge problem on the extracted open graph.

    Choices made here:
    - unknown node field: potential
    - unknown edge field: flux
    - parameter node field: capacity
    - parameter edge field: conductance
    - left boundary: fixed potential = 1
    - right boundary: imposed sink flux = 0.5
    """
    _, graph, boundary_fluxes = build_cell_symplast_open_graph()

    system = MixedTransportSystem(
        graph=graph,
        node_fields={
            "potential": FieldState(
                name="potential",
                location="node",
                values=np.zeros(graph.n_nodes, dtype=np.float64),
            ),
            "capacity": FieldState(
                name="capacity",
                location="node",
                values=np.ones(graph.n_nodes, dtype=np.float64),
            ),
        },
        edge_fields={
            "conductance": FieldState(
                name="conductance",
                location="edge",
                values=np.asarray([2.0, 1.0], dtype=np.float64),
            ),
            "flux": FieldState(
                name="flux",
                location="edge",
                values=np.zeros(graph.n_edges, dtype=np.float64),
            ),
        },
        boundary_fluxes=boundary_fluxes,
        dirichlet=(
            DirichletNodeValue(
                field="potential",
                node_id=int(graph.node_ids.min()),
                value=1.0,
            ),
        ),
        unknowns=UnknownLayout(node_fields=("potential",), edge_fields=("flux",)),
        solver=SolverSpec(method="linear_newton"),
    )

    return system


def test_mtg_to_network():
    """
    First sanity check: the fine MTG can be turned into a symmetric graph matrix.

    This is the simplest "network view" of the MTG:
    - take node ids
    - take edge endpoints
    - assemble a Laplacian-like sparse matrix

    The point of this test is not physics yet; it is just to validate the idea
    that scale-based MTG data can be vectorized into graph operators.
    """
    g = build_seedling_mtg()

    nids = np.asarray(g.array_at_scale("vertex_id", scale=scales["node"]), dtype=np.int64)
    n = len(nids)
    n_id_a = np.asarray(g.array_at_scale("n_id_a", scale=scales["edge"]), dtype=np.int64)
    n_id_b = np.asarray(g.array_at_scale("n_id_b", scale=scales["edge"]), dtype=np.int64)
    length = np.asarray(g.array_at_scale("length", scale=scales["edge"]), dtype=np.float64)

    nid_to_index = {int(vid): idx for idx, vid in enumerate(nids)}
    idx_a = np.asarray([nid_to_index[int(vid)] for vid in n_id_a], dtype=np.int64)
    idx_b = np.asarray([nid_to_index[int(vid)] for vid in n_id_b], dtype=np.int64)

    # Standard symmetric edge contribution:
    #  +w on both diagonals, -w on both off-diagonal positions.
    rows = np.r_[idx_a, idx_b, idx_a, idx_b]
    cols = np.r_[idx_a, idx_b, idx_b, idx_a]
    data = np.r_[length, length, -length, -length]

    adjacency_laplacian = coo_matrix((data, (rows, cols)), shape=(n, n))

    assert adjacency_laplacian.shape == (n, n)
    np.testing.assert_allclose(adjacency_laplacian.toarray(), adjacency_laplacian.toarray().T)


def test_open_graph_subset_from_mtg_supports_boundary_edges():
    """
    Second check: extract a coarse 3-node chain and verify its open-graph data.

    Expected structure:

        node_0 --edge--> node_1 --edge--> node_2
                                       |
                                       boundary sink

    The incidence matrix should therefore be:

        [[ 1,  0],
         [-1,  1],
         [ 0, -1]]
    """
    _, graph, _ = build_cell_symplast_open_graph()

    assert graph.n_nodes == 3
    assert graph.n_edges == 2
    np.testing.assert_array_equal(graph.tail, np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(graph.head, np.asarray([1, 2], dtype=np.int64))
    np.testing.assert_allclose(
        graph.incidence.toarray(),
        np.asarray(
            [
                [1.0, 0.0],
                [-1.0, 1.0],
                [0.0, -1.0],
            ]
        ),
    )
    np.testing.assert_allclose(
        graph.boundary_incidence.toarray(),
        np.asarray(
            [
                [0.0],
                [0.0],
                [1.0],
            ]
        ),
    )


def test_generic_mixed_system_solves_node_states_edge_states_and_boundaries():
    """
    Third check: solve the steady mixed problem.

    Unknowns:
    - node potentials p
    - edge fluxes q

    Data:
    - conductances [2, 1]
    - left node fixed at p=1
    - right boundary removes flux 0.5

    We verify both:
    - the solved values
    - the fact that the final residual is zero
    """
    system = build_mixed_transport_prototype()
    solution = system.solve(dt=None)
    node_unknowns, edge_unknowns = system.unpack_unknowns(solution)

    np.testing.assert_allclose(node_unknowns["potential"], np.asarray([1.0, 0.75, 0.25]))
    np.testing.assert_allclose(edge_unknowns["flux"], np.asarray([0.5, 0.5]))
    np.testing.assert_allclose(
        system.residual(solution),
        np.zeros(system.graph.n_nodes + system.graph.n_edges),
        atol=1e-12,
    )


def test_same_residual_api_supports_backward_euler_and_fd_jacobians():
    """
    Fourth check: the exact same residual API also supports:
    - a transient backward-Euler term
    - a finite-difference Jacobian fallback

    This is important for the larger design question:
    the model description can stay the same while the chosen resolution strategy
    changes.
    """
    system = build_mixed_transport_prototype()
    previous_potential = np.zeros(system.graph.n_nodes, dtype=np.float64)

    transient_solution = system.solve(previous_potential=previous_potential, dt=0.5)
    node_unknowns, edge_unknowns = system.unpack_unknowns(transient_solution)

    # The fixed left boundary stays at 1, and the profile decreases to the right.
    np.testing.assert_allclose(node_unknowns["potential"][0], 1.0)
    assert node_unknowns["potential"][0] > node_unknowns["potential"][1] > node_unknowns["potential"][2]
    assert np.all(edge_unknowns["flux"] >= 0.0)
    np.testing.assert_allclose(
        system.residual(transient_solution, previous_potential=previous_potential, dt=0.5),
        np.zeros(system.graph.n_nodes + system.graph.n_edges),
        atol=1e-12,
    )

    # Compare analytic and finite-difference Jacobians on an arbitrary probe
    # state. This demonstrates how a generic framework could accept:
    # - explicit Jacobians
    # - AD Jacobians later
    # - or finite-difference fallback during prototyping.
    probe = system.pack_unknowns(
        node_overrides={"potential": np.asarray([1.0, 0.7, 0.4])},
        edge_overrides={"flux": np.asarray([0.6, 0.3])},
    )
    analytic = system.analytic_jacobian(dt=0.5).toarray()
    finite_difference = system.finite_difference_jacobian(
        probe,
        previous_potential=previous_potential,
        dt=0.5,
    )
    np.testing.assert_allclose(finite_difference, analytic, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    test_mtg_to_network()
    test_open_graph_subset_from_mtg_supports_boundary_edges()
    test_generic_mixed_system_solves_node_states_edge_states_and_boundaries()
    test_same_residual_api_supports_backward_euler_and_fd_jacobians()
