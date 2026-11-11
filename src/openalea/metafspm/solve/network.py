"""
Linear algebra on a DataStructure's graph, for the upscaled variables a model declares (state_variable(upscaling=)):

  conductance, inflow = equivalent_conductance(data_structure.to_graph_view(), edge_conductance, sources=soil_interface,
                                               sinks=collar, source_conductance=soil_conductance)

No physics: the model assembles its network's conductances and chooses the protocol (which nodes are held at which
potential); the operator solves it once.

Every factorisation and solve here is the run's linear solver's (Compute(linear_solver=), solve.linear's registry:
LinearSolver.solve_once, factorize), resolved for the graph system the work belongs to (*system*, a
solve.linear.LinearSystem: a graph system's static condensation, a reduction's archetypes), else under "default".
"""
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components


def _solver(system):
    """The current Compute's LinearSolver for *system* (None: its "default")."""
    from openalea.metafspm.compute import Compute
    return Compute.current().linear(system)


def _symmetric(system) -> bool:
    return system is not None and system.symmetric


def _mask(nodes, n: int, name: str) -> np.ndarray:
    nodes = np.asarray(nodes)
    if nodes.dtype == bool:
        if nodes.shape != (n,):
            raise ValueError(f"equivalent_conductance: {name} is a mask of {nodes.shape}, the graph has {n} nodes")
        return nodes.copy()
    mask = np.zeros(n, dtype=bool)
    mask[nodes.astype(np.int64)] = True
    return mask


def equivalent_conductance(view, edge_conductance, sources, sinks, source_conductance=None, groups=None,
                           cache: dict = None, extra=None, system=None) -> tuple:
    """
    The equivalent conductance of a network between two sets of nodes: (conductance, inflow).

    *view*: a GraphView (its incidence B, nodes × edges); *edge_conductance*: one per edge (q_e = k_e (Bᵀψ)_e);
    *sources*, *sinks*: the nodes (a boolean mask or local indices) at potential 1 and 0. With
    *source_conductance* (one per node, or a number), each source node is joined to a reservoir at potential 1
    through it (a Robin condition, e.g. the soil's conductance at a root's surface), instead of being held at 1.

    One sparse solve of the steady network: *conductance* is the total flow from the sources into the sinks for a
    unit potential difference; *inflow* (one per node, 0 off the sources) each source node's share of it, so that
    inflow / conductance are the standard uptake fractions when the sources are a root's surface and the sinks its
    collar. Nodes in a piece of the graph that touches no source or sink carry no flow.

    *groups* (an integer per node, e.g. each node's segment; *cache* kept between calls): the solve by static
    condensation on them (condensed_solve), the sources and sinks kept. *extra*: a matrix (nodes × nodes) added to
    the network's (e.g. a reduction's groups' interiors, solve.reduction). *system*: the
    solve.linear.LinearSystem the solve is resolved for (Compute(linear_solver=)); None: "default".
    """
    incidence = csr_matrix(view.incidence)
    n = incidence.shape[0]
    k = np.broadcast_to(np.asarray(edge_conductance, dtype=float), (incidence.shape[1],))
    sources, sinks = _mask(sources, n, "sources"), _mask(sinks, n, "sinks")
    if np.any(sources & sinks):
        raise ValueError("equivalent_conductance: a node is both a source and a sink")
    if not sources.any() or not sinks.any():
        raise ValueError("equivalent_conductance: give at least one source and one sink")
    laplacian = (incidence @ diags(k) @ incidence.T).tocsr()
    if extra is not None:
        laplacian = (laplacian + csr_matrix(extra)).tocsr()
    robin = None
    if source_conductance is not None:
        robin = np.where(sources, np.broadcast_to(np.asarray(source_conductance, dtype=float), (n,)), 0.)
    fixed = sinks | (sources if robin is None else np.zeros(n, dtype=bool))
    value = np.where(sources & fixed, 1., 0.)
    # the pieces touching a fixed node or a Robin source carry flow; the others float
    _, piece = connected_components(abs(laplacian) > 0, directed=False)
    anchored = np.unique(piece[fixed | (robin > 0 if robin is not None else False)])
    free = ~fixed & np.isin(piece, anchored)
    matrix = laplacian[free][:, free]
    rhs = -laplacian[free][:, fixed] @ value[fixed]
    if robin is not None:
        matrix = matrix + diags(robin[free])
        rhs = rhs + robin[free]
    potential = value.copy()
    if free.any() and groups is not None:
        held = sources[free] if robin is not None else np.zeros(int(free.sum()), dtype=bool)
        potential[free] = condensed_solve(matrix, np.asarray(rhs).reshape(-1), np.asarray(groups)[free], held,
                                          cache=cache, system=system)
    elif free.any():
        potential[free] = _solver(system).solve_once(matrix.tocsc(), rhs, _symmetric(system))
    if robin is not None:
        inflow = np.where(sources, robin * (1. - potential), 0.)
    else:
        inflow = np.where(sources, np.asarray(laplacian @ potential).reshape(-1), 0.)
    return float(inflow.sum()), inflow


# ── Static condensation: a sparse system solved through its groups' interiors, factorised once per archetype ──

def condensed_solve(matrix, rhs, groups, kept=None, cache: dict = None, system=None,
                    _local: bool = True) -> np.ndarray:
    """
    The solution of *matrix* x = *rhs* by static condensation on node *groups* (an integer per node, e.g. each
    node's segment; < 0: in no group).

    A node of a group is interior unless it is *kept* (a mask, e.g. the nodes a boundary condition holds or joins to
    a reservoir) or coupled to a node outside its group. Each group's interior is eliminated: one sparse
    factorisation per archetype, the groups whose interior block (and its coupling to their kept nodes) is the
    same, up to a positive factor, sharing it; the system on the kept and ungrouped nodes is then solved, and the
    interiors recovered from it. Exact up to round-off (an inexact solve within Newton iterations is corrected by the
    next one).

    The kept nodes of a group coupled to nothing outside it (e.g. a section's nodes joined to its soil, their terms
    the group's own) are then eliminated within their group, at each call (no factorisation kept: their terms may
    differ from group to group and from call to call); the system left holds the nodes coupled across groups and
    those in none.

    *cache*: a dict kept between calls (e.g. by the graph system): the partition of the matrix's pattern (computed
    once per pattern) and the factorisations of the archetypes, reused while their blocks are unchanged (each
    block is compared with its archetype's at every call).

    *system*: the solve.linear.LinearSystem whose backend factorises the archetypes and solves the systems left
    (Compute(linear_solver=)); None: "default".
    """
    from scipy.sparse import coo_matrix
    A = csr_matrix(matrix)
    A.sort_indices()
    b = np.asarray(rhs, dtype=np.float64).reshape(-1)
    n = A.shape[0]
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    kept = np.zeros(n, dtype=bool) if kept is None else np.asarray(kept, dtype=bool).reshape(-1)
    cache = {} if cache is None else cache
    solver, symmetric = _solver(system), _symmetric(system)
    plan = _condensation_plan(A, groups, kept, cache.setdefault("plans", {}))
    if plan is None:
        return solver.solve_once(A.tocsc(), b, symmetric)
    archetypes = cache.setdefault("archetypes", {})
    interior, kept_nodes, position = plan["interior"], plan["kept_nodes"], plan["position"]
    data = A.data
    reduced_values = []
    b_K = b[kept_nodes].copy()
    pending = {}                                          # archetype -> [(group, factor)]
    for group, (block_s, coupling_s, back_s) in enumerate(plan["structures"]):
        block, coupling, back = (csr_matrix((data[where], indices, indptr), shape=shape)
                                 for where, indices, indptr, shape in (block_s, coupling_s, back_s))
        factor = float(block.data[0]) if block.nnz and block.data[0] != 0 else 1.
        pending.setdefault(_condensation_archetype(block, coupling, back, factor, archetypes, solver, symmetric),
                           []).append((group, factor))
    starts, ends, columns = plan["starts"], plan["ends"], plan["columns"]
    solutions = {}
    reduction_order = []
    for archetype, members in pending.items():
        entry = archetypes[archetype]
        Z = entry["lu"].solve(np.stack([b[interior[starts[group]:ends[group]]] for group, _ in members], axis=1))
        for j, (group, factor) in enumerate(members):
            reduction_order.append(group)
            reduced_values.append(-(factor * entry["S"]).ravel())    # minus A_KI A_II⁻¹ A_IK, scaled
            b_K[position[kept_nodes[columns[group]]]] -= entry["back"] @ Z[:, j]   # (c A_KI_ref) (A_ref⁻¹ b_I / c)
        solutions[archetype] = (members, Z)
    rows = np.concatenate([plan["reduced_rows"][group] for group in reduction_order])
    cols = np.concatenate([plan["reduced_cols"][group] for group in reduction_order])
    where_KK, indices_KK, indptr_KK, shape_KK = plan["A_KK"]
    A_KK = csr_matrix((data[where_KK], indices_KK, indptr_KK), shape=shape_KK)
    reduced = (A_KK + coo_matrix((np.concatenate(reduced_values), (rows, cols)), shape=shape_KK).tocsr()).tocsr()
    shared = plan["shared"]
    if _local and not shared[kept_nodes].all():           # the groups' own kept nodes, eliminated within each
        local = cache.setdefault("local", {})
        local["archetypes"] = {}                               # their terms change between calls: nothing kept
        x_K = condensed_solve(reduced, b_K, groups[kept_nodes], shared[kept_nodes], cache=local, system=system,
                              _local=False)
    else:
        x_K = solver.solve_once(reduced.tocsc(), b_K, symmetric).reshape(-1)
    x = np.empty(n, dtype=np.float64)
    x[kept_nodes] = x_K
    for archetype, (members, Z) in solutions.items():
        X = archetypes[archetype]["X"]
        for j, (group, factor) in enumerate(members):
            x[interior[starts[group]:ends[group]]] = Z[:, j] / factor - X @ x_K[position[kept_nodes[columns[group]]]]
    return x


def _condensation_plan(A, groups, kept, plans: dict):
    """The partition of *A*'s pattern by *groups* (computed once per pattern, groups and kept nodes): the interior
    nodes by group, the kept ones, and each group's blocks as (where in A.data, indices, indptr, shape); None when
    no node is interior."""
    import hashlib
    digest = hashlib.blake2b(digest_size=16)
    for array in (A.indptr, A.indices, groups, kept):
        digest.update(np.ascontiguousarray(array).tobytes())
    key = (A.shape, digest.hexdigest())
    if key in plans:
        return plans[key]
    n = A.shape[0]
    keep = kept.copy()
    rows = np.repeat(np.arange(n), np.diff(A.indptr))
    outside = groups[rows] != groups[A.indices]
    shared = groups < 0                                    # the nodes of the final system: in no group, or coupled
    shared[rows[outside]] = True                           # across groups
    shared[A.indices[outside]] = True
    keep |= shared
    interior = np.flatnonzero(~keep)
    if interior.size == 0:
        plans[key] = None
        return None
    interior = interior[np.lexsort((interior, groups[interior]))]       # by group, then by node
    kept_nodes = np.flatnonzero(keep)
    position = np.full(n, -1, dtype=np.int64)
    position[kept_nodes] = np.arange(kept_nodes.size)
    where = csr_matrix((np.arange(1, A.nnz + 1, dtype=np.float64), A.indices, A.indptr), shape=A.shape)

    def structure(m):
        m = m.tocsr()
        m.sort_indices()
        return (m.data.astype(np.int64) - 1, m.indices.copy(), m.indptr.copy(), m.shape)

    W_II = where[interior][:, interior].tocsr()
    W_IK = where[interior][:, kept_nodes].tocsr()
    W_KI = where[kept_nodes][:, interior].tocsc()
    owner = groups[interior]
    _, starts = np.unique(owner, return_index=True)
    ends = np.r_[starts[1:], interior.size]
    structures, columns, reduced_rows, reduced_cols = [], [], [], []
    for start, end in zip(starts.tolist(), ends.tolist()):
        coupling = W_IK[start:end]                         # the group's interior to its kept nodes
        here = np.unique(coupling.indices)
        structures.append((structure(W_II[start:end, start:end]), structure(coupling[:, here]),
                           structure(W_KI[:, start:end].tocsr()[here])))
        columns.append(here)
        grid_r, grid_c = np.meshgrid(here, here, indexing="ij")
        reduced_rows.append(grid_r.ravel())
        reduced_cols.append(grid_c.ravel())
    plan = {"interior": interior, "kept_nodes": kept_nodes, "position": position, "shared": shared,
            "starts": starts.tolist(), "ends": ends.tolist(), "columns": columns, "structures": structures,
            "reduced_rows": reduced_rows, "reduced_cols": reduced_cols,
            "A_KK": structure(where[kept_nodes][:, kept_nodes])}
    if len(plans) > 8:
        plans.clear()
    plans[key] = plan
    return plan


def _condensation_archetype(block, coupling, back, factor, archetypes: dict, solver, symmetric: bool):
    """The archetype of a group (its key in *archetypes*): its blocks divided by *factor*, matched with an archetype's within
    round-off; a new archetype is factorised by *solver* (its interior's factorisation, X = A_II⁻¹ A_IK and S = A_KI X,
    for factor 1)."""
    scaled = [m.multiply(1. / factor).tocsr() for m in (block, coupling, back)]
    structure = tuple(bytes(m.indptr.tobytes()) + bytes(m.indices.tobytes()) + bytes(np.int64(m.shape[1]).tobytes())
                      for m in scaled)
    signature = hash(structure + tuple(np.round(m.data, 9).tobytes() for m in scaled))
    slot = 0
    while (signature, slot) in archetypes:                     # the archetypes of that signature (a collision: the next slot)
        entry = archetypes[(signature, slot)]
        if all(np.allclose(m.data, r.data, rtol=1e-11, atol=0.) for m, r in zip(scaled, entry["blocks"])):
            return signature, slot
        slot += 1
    key = (signature, slot)
    block, coupling, back = scaled
    lu = solver.factorize(block.tocsc(), symmetric)
    X = lu.solve(coupling.toarray()) if coupling.shape[1] else np.zeros((block.shape[0], 0))
    archetypes[key] = {"blocks": scaled, "lu": lu, "X": X, "S": np.asarray(back @ X), "back": back}
    return key


# ── Kron reduction: a network's equivalent network on chosen nodes ──

class KronReduction:
    """
    A network's matrix *A* (nodes × nodes, e.g. a conductance Laplacian with its diagonal terms) reduced onto its
    *kept* nodes: the Schur complement S = A_KK − A_KI A_II⁻¹ A_IK, exact for every potential of the kept nodes.

      reduced           S (kept × kept, dense)
      pairs()           (i, j, conductance) of the equivalent network: −S_ij for each pair of kept nodes i < j
      shunts()          each kept node's diagonal term left (S_ii + Σ_j S_ij: 0 for a network of conductances alone)
      inject(b_I)       the interior sources *b_I* as sources at the kept nodes (−A_KI A_II⁻¹ b_I)
      injection()       inject's matrix (kept × interior): −Xᵀ for a symmetric A (X = A_II⁻¹ A_IK), no solve
      shares(T, v_I)    the interior values *v_I* shared among the kept nodes *T* by the recovery weights
      recover(x_K, b_I) the interior's potentials (A_II⁻¹ (b_I − A_IK x_K)), from the kept nodes' and the sources
      release()         the interior's factorisation, X and S freed: a reduction kept for recovery, from rarely

    Kept and interior nodes are local indices of *A* (kept_nodes, interior_nodes). The interior is factorised once,
    and X = A_II⁻¹ A_IK (dense, interior × kept: S's) solved once, both kept until release(): what a reduction needs
    of them (its injection, its shares) is computed before. After release, the reduction keeps the sparse blocks of
    *A* only.

    The interior's factorisation and its solves are the current Compute's linear solver's for *system* (a
    solve.linear.LinearSystem, the graph system reduced; None: "default"): LinearSolver.factorize, by default the
    device's (backend.HostBackend's SuperLU on the cpu; on "cuda", SuperLU's factors copied to the GPU and X's many
    right-hand sides solved there, A_IK sent sparse and densified on the device, X copied back once); CHOLMOD's for
    a system declared symmetric mapped to "cholmod".
    """

    def __init__(self, matrix, kept, system=None):
        A = csr_matrix(matrix)
        n = A.shape[0]
        kept = _mask(kept, n, "kept")
        self.kept_nodes, self.interior_nodes = np.flatnonzero(kept), np.flatnonzero(~kept)
        K, I = self.kept_nodes, self.interior_nodes
        self._A_KI = A[K][:, I].tocsr()
        self._A_IK = A[I][:, K].tocsr()
        self._A_II = A[I][:, I].tocsc()
        difference = abs(A - A.T)
        self.symmetric = bool(difference.nnz == 0 or difference.max() <= 1e-14 * abs(A).max())
        self._lu, self._keep, self._system = None, True, system
        from openalea.metafspm.compute import Compute
        with Compute.current().timed("solve.kron_reduction"):          # run.json's timing (an archetype's reduction)
            self._X = (self._factorisation().solve(self._A_IK) if I.size and K.size
                       else np.zeros((I.size, K.size)))
        self.reduced = np.asarray(A[K][:, K].toarray() - self._A_KI @ self._X)

    def _factorisation(self):
        """The interior's factorisation (the current Compute's linear solver's for the system: solve(rhs, trans)): the
        kept one, else a new one (kept unless released)."""
        if self._lu is not None:
            return self._lu
        lu = _solver(self._system).factorize(self._A_II, _symmetric(self._system))
        if self._keep:
            self._lu = lu
        return lu

    def release(self) -> None:
        """Free the interior's factorisation, X and the reduced matrix S, and keep none from now on: inject,
        injection, shares and recover factorise the interior again at each call (a reduction kept for recovery,
        recovered from rarely: what was derived from S was taken before, e.g. an archetype's D); pairs and shunts need S,
        so are not available after."""
        self._lu, self._X, self._keep, self.reduced = None, None, False, None

    def pairs(self) -> tuple:
        """(i, j, conductance) of the equivalent network's pairs i < j (local indices among the kept nodes)."""
        i, j = np.triu_indices(self.kept_nodes.size, k=1)
        return i, j, -0.5 * (self.reduced[i, j] + self.reduced[j, i])

    def shunts(self) -> np.ndarray:
        """Each kept node's diagonal term left once its pairs are counted (its Robin terms, carried through)."""
        return self.reduced.sum(axis=1)

    def inject(self, interior_sources) -> np.ndarray:
        """The interior's sources (one per interior node, or a matrix of columns) as sources at the kept nodes."""
        b = np.asarray(interior_sources, dtype=float)
        if not self.interior_nodes.size:
            return np.zeros((self.kept_nodes.size,) + b.shape[1:])
        return -(self._A_KI @ self._factorisation().solve(b))

    def injection(self) -> np.ndarray:
        """inject's matrix (kept × interior, dense): −A_KI A_II⁻¹. For a symmetric A (A_KI = A_IKᵀ), −Xᵀ while X is
        kept (no solve); otherwise one transposed solve with the kept nodes' columns (not one per interior
        source)."""
        if not self.interior_nodes.size:
            return np.zeros((self.kept_nodes.size, 0))
        if self.symmetric and self._X is not None:
            return -self._X.T
        return -np.asarray(self._factorisation().solve(self._A_KI.T.tocsr(), trans="T")).T

    def shares(self, targets, interior_values) -> np.ndarray:
        """
        The interior's values *interior_values* (one per interior node, e.g. its volumes) shared among the kept
        nodes *targets* (a mask over the kept nodes, or their indices among them): each interior node's value split
        by its recovery weights over the targets (its row of W = −A_II⁻¹ A_IK, the potential it owes to each kept
        node, restricted to the targets and normalised to 1; a node owing nothing to them, its value shared evenly).
        One value per kept node, 0 off the targets. From X while it is kept, else from two single solves (the
        weights' row sums, then the transposed solve of the normalised values): no dense interior × kept matrix.
        """
        K = self.kept_nodes.size
        targets = np.asarray(targets)
        mask = targets.astype(bool) if targets.dtype == bool else np.isin(np.arange(K), targets)
        if mask.shape != (K,):
            raise ValueError(f"KronReduction.shares: targets is a mask of the {K} kept nodes, got {mask.shape}")
        values = np.asarray(interior_values, dtype=float).reshape(-1)
        if values.size != self.interior_nodes.size:
            raise ValueError(f"KronReduction.shares: {values.size} interior values, {self.interior_nodes.size} "
                             "interior nodes")
        out = np.zeros(K)
        if not values.size or not mask.any():
            return out
        if self._X is not None:
            total = -self._X[:, mask].sum(axis=1)
        else:
            total = self._factorisation().solve(-(self._A_IK @ mask.astype(float)))
        bound = total > 0
        scaled = np.where(bound, values / np.where(bound, total, 1.), 0.)
        if self._X is not None:
            weighted = -(self._X.T @ scaled)
        else:
            weighted = -(self._A_IK.T @ self._factorisation().solve(scaled, trans="T"))
        out[mask] = weighted[mask] + values[~bound].sum() / mask.sum()
        return out

    def recover(self, kept_values, interior_sources=None) -> np.ndarray:
        """The interior's values for the kept nodes' *kept_values* (a vector, or a matrix of columns) and the
        interior's sources, if any."""
        x = np.asarray(kept_values, dtype=float)
        if not self.interior_nodes.size:
            return np.zeros((0,) + x.shape[1:])
        b = -(self._A_IK @ x)
        if interior_sources is not None:
            b = b + np.asarray(interior_sources, dtype=float)
        return np.asarray(self._factorisation().solve(b))


def kron_reduction(matrix, kept, system=None) -> KronReduction:
    """The Kron reduction of *matrix* onto its *kept* nodes (a mask or local indices), factorised by the linear
    solver of *system* (a solve.linear.LinearSystem; None: "default"): KronReduction."""
    return KronReduction(matrix, kept, system)
