"""
Linear algebra on a DataStructure's graph, for the upscaled variables a model declares (state_variable(upscaling=)):

  conductance, inflow = equivalent_conductance(ds.to_graph_view(), edge_conductance, sources=soil_interface,
                                               sinks=collar, source_conductance=soil_conductance)

No physics: the model assembles its network's conductances and chooses the protocol (which nodes are held at which
potential); the operator solves it once.
"""
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve


def _mask(nodes, n: int, name: str) -> np.ndarray:
    nodes = np.asarray(nodes)
    if nodes.dtype == bool:
        if nodes.shape != (n,):
            raise ValueError(f"equivalent_conductance: {name} is a mask of {nodes.shape}, the graph has {n} nodes")
        return nodes.copy()
    mask = np.zeros(n, dtype=bool)
    mask[nodes.astype(np.int64)] = True
    return mask


def equivalent_conductance(view, edge_conductance, sources, sinks, source_conductance=None) -> tuple:
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
    if free.any():
        potential[free] = spsolve(matrix.tocsc(), rhs)
    if robin is not None:
        inflow = np.where(sources, robin * (1. - potential), 0.)
    else:
        inflow = np.where(sources, np.asarray(laplacian @ potential).reshape(-1), 0.)
    return float(inflow.sum()), inflow
