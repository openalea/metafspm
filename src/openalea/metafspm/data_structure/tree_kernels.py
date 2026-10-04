"""
Tree kernels: computations along a plant's topology, vectorised over every plant of a DataStructure (design note
docs/design/population_and_performance.md §3, §9, §12; plan P3).

They work on local node indices: a parent array (-1 at roots) and, for chains, an ordering of the nodes into
chains (axes, or ranks within a group). Models call them through MPGDataStructure (chain_scan, accumulate,
path_window, chain_shift, chain_write, path_compose), never with indices of their own.

Exactness: chain scans and path windows add their terms in the same sequential order as the loops they replace
(e.g. rhizodep's distance_from_tip and supply for elongation), so their results are identical bit for bit.
"""
import numpy as np
from numba import njit, prange


# ── Chains ─────────────────────────────────────────────────────────────────────

def chains_from_predecessors(pred: np.ndarray) -> dict:
    """
    Chains from a predecessor array (pred[v] = the previous node of v on its chain, -1 at a chain start).
    Returns {"order": nodes chain by chain, from start to end; "offsets": chain boundaries in order (n_chains + 1);
    "chain": chain index of each node; "position": rank of each node on its chain}.
    """
    n = pred.size
    successor = np.full(n, -1, dtype=np.int64)
    has_pred = pred >= 0
    targets = pred[has_pred]
    if np.unique(targets).size < targets.size:
        raise ValueError("a node has several successors on a chain: chains must be linear")
    successor[targets] = np.flatnonzero(has_pred)
    starts = np.flatnonzero(~has_pred)
    order, offsets = _follow_chains(starts, successor, n)
    if order.size != n:
        raise ValueError("chains contain a cycle")
    chain = np.empty(n, dtype=np.int64)
    position = np.empty(n, dtype=np.int64)
    _chain_positions(order, offsets, chain, position)
    return {"order": order, "offsets": offsets, "chain": chain, "position": position}


def chains_from_groups(group: np.ndarray, rank: np.ndarray) -> dict:
    """Chains as the nodes of each *group* value, ordered by *rank* (e.g. metamer ranks of an axis)."""
    order = np.lexsort((rank, group)).astype(np.int64)
    sorted_groups = group[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1], True])
    offsets = boundaries.astype(np.int64)
    chain = np.empty(order.size, dtype=np.int64)
    position = np.empty(order.size, dtype=np.int64)
    _chain_positions(order, offsets, chain, position)
    return {"order": order, "offsets": offsets, "chain": chain, "position": position}


@njit(cache=True)
def _follow_chains(starts, successor, n):
    order = np.empty(n, dtype=np.int64)
    offsets = np.empty(starts.size + 1, dtype=np.int64)
    k = 0
    for c in range(starts.size):
        offsets[c] = k
        v = starts[c]
        while v >= 0 and k < n:
            order[k] = v
            k += 1
            v = successor[v]
    offsets[starts.size] = k
    return order[:k], offsets


@njit(cache=True)
def _chain_positions(order, offsets, chain, position):
    for c in range(offsets.size - 1):
        for i in range(offsets[c], offsets[c + 1]):
            chain[order[i]] = c
            position[order[i]] = i - offsets[c]


@njit(cache=True, parallel=True)
def _segmented_scan(values, order, offsets, op_max, reverse, exclusive):
    """Inclusive or exclusive scan (sum or max) of values (n, k) along each chain, sequentially, in parallel over chains."""
    out = np.zeros_like(values)
    for c in prange(offsets.size - 1):
        start, end = offsets[c], offsets[c + 1]
        for j in range(values.shape[1]):
            acc = 0.
            first = True
            for t in range(end - start):
                i = order[end - 1 - t] if reverse else order[start + t]
                if exclusive:
                    out[i, j] = acc if not first else (-np.inf if op_max else 0.)
                if first:
                    acc = values[i, j]
                    first = False
                elif op_max:
                    acc = max(acc, values[i, j])
                else:
                    acc = acc + values[i, j]
                if not exclusive:
                    out[i, j] = acc
    return out


def chain_scan(values, chains: dict, op: str = "sum", reverse: bool = False, exclusive: bool = False) -> np.ndarray:
    """
    Cumulative sum or max of *values* along each chain, from its start (reverse=False) or from its end
    (reverse=True); exclusive=True leaves out the node's own value (0, or -inf for max, at the first node).
    """
    if op not in ("sum", "max"):
        raise ValueError(f"chain_scan: op must be 'sum' or 'max', got '{op}'")
    array, vector = _as_columns(values)
    out = _segmented_scan(array, chains["order"], chains["offsets"], op == "max", reverse, exclusive)
    return out if vector else out[:, 0]


def chain_shift(values, chains: dict, k: int = 1, fill=np.nan) -> np.ndarray:
    """The value of the node k positions earlier on the same chain (k < 0: later), *fill* beyond the chain's ends."""
    array, vector = _as_columns(values)
    out = np.full_like(array, fill, dtype=np.float64)
    order, chain, position = chains["order"], chains["chain"], chains["position"]
    source_rank = np.empty_like(order)
    source_rank[order] = np.arange(order.size)            # rank of each node in `order`
    rank = source_rank - k
    valid = (rank >= 0) & (rank < order.size)
    candidates = np.where(valid, order[np.clip(rank, 0, order.size - 1)], -1)
    same_chain = valid & (chain[np.maximum(candidates, 0)] == chain)
    out[same_chain] = array[candidates[same_chain]]
    return out if vector else out[:, 0]


def chain_write(event, values, chains: dict, k: int = 1, base=None) -> np.ndarray:
    """
    Forward writes along chains: where *event* holds at a node, the node k positions later on its chain gets that
    node's *values*; every other node keeps *base* (default: nan). E.g. leaf n's emergence setting values of n + 1.
    """
    array, vector = _as_columns(values)
    out = (np.full_like(array, np.nan) if base is None else _as_columns(base)[0].astype(np.float64).copy())
    targets = chain_shift(np.arange(array.shape[0], dtype=np.float64), chains, k=-k, fill=-1.).astype(np.int64)
    sources = np.flatnonzero(np.asarray(event, dtype=bool) & (targets >= 0))
    out[targets[sources]] = array[sources]
    return out if vector else out[:, 0]


# ── Levels and accumulations ───────────────────────────────────────────────────

def depth(parents: np.ndarray) -> np.ndarray:
    """Number of edges from each node to its root (pointer doubling, synchronous updates)."""
    parents = np.asarray(parents, dtype=np.int64)
    dist = (parents >= 0).astype(np.int64)
    jump = parents.copy()
    while True:
        idx = np.flatnonzero(jump >= 0)
        if idx.size == 0:
            return dist
        above = jump[idx]
        dist_above, jump_above = dist[above].copy(), jump[above].copy()
        dist[idx] += dist_above
        jump[idx] = jump_above


def levels(parents: np.ndarray) -> list:
    """Node indices grouped by depth, from the roots."""
    d = depth(parents)
    order = np.argsort(d, kind="stable")
    bounds = np.flatnonzero(np.r_[True, np.diff(d[order]) != 0, True])
    return [order[bounds[i]:bounds[i + 1]] for i in range(bounds.size - 1)]


def accumulate(values, parents: np.ndarray, direction: str = "up", op: str = "sum") -> np.ndarray:
    """
    direction="up":   each node gets its value combined with its descendants' (subtree sum or max), level by level
                      from the deepest;
    direction="down": each node gets its value combined with its ancestors' (path sum or max from the root).
    """
    if direction not in ("up", "down") or op not in ("sum", "max"):
        raise ValueError("accumulate: direction must be 'up' or 'down', op 'sum' or 'max'")
    array, vector = _as_columns(values)
    out = np.array(array, dtype=np.float64)
    combine = np.add if op == "sum" else np.maximum
    groups = levels(parents)
    if direction == "up":
        for nodes in reversed(groups[1:]):
            combine.at(out, parents[nodes], out[nodes])
    else:
        for nodes in groups[1:]:
            out[nodes] = combine(out[nodes], out[parents[nodes]])
    return out if vector else out[:, 0]


def path_compose(transforms: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Composed transform of each node: its ancestors' transforms from the root, then its own (T_root @ ... @ T_v)."""
    out = np.array(transforms, dtype=np.float64)
    for nodes in levels(parents)[1:]:
        out[nodes] = np.matmul(out[parents[nodes]], out[nodes])
    return out


# ── Windows towards the base ───────────────────────────────────────────────────

@njit(cache=True, parallel=True)
def _path_window(targets, parents, budget, extent, values, include):
    sums = np.zeros((targets.size, values.shape[1]))
    for t in prange(targets.size):
        node = targets[t]
        remaining = budget[node]
        current = node
        while remaining > 0:
            if remaining > extent[current]:
                if include[current]:
                    for j in range(values.shape[1]):
                        sums[t, j] += values[current, j]
                    remaining = remaining - extent[current]
                current = parents[current]
                if current < 0:
                    break
            else:
                # value * remaining / extent, in this order (rhizodep's), for identical rounding
                for j in range(values.shape[1]):
                    sums[t, j] += values[current, j] * remaining / extent[current]
                remaining = 0.
    return sums


def path_window(parents, budget, extent, values, where=None, include=None) -> np.ndarray:
    """
    For each node of *where* (default: all), the sum of *values* over the node and its ancestors, walking towards
    the base until the cumulated *extent* reaches the node's *budget*; the last element contributes the fraction of
    its extent needed. Nodes where *include* is False are walked through without contributing (rhizodep skips
    elements of zero length). Nodes outside *where* get 0. Same summation order as rhizodep's
    calculating_supply_for_elongation.
    """
    array, vector = _as_columns(values)
    n = array.shape[0]
    targets = np.arange(n, dtype=np.int64) if where is None else np.flatnonzero(np.asarray(where, dtype=bool))
    include = np.ones(n, dtype=np.bool_) if include is None else np.asarray(include, dtype=np.bool_)
    sums = _path_window(targets, np.asarray(parents, dtype=np.int64), np.broadcast_to(np.asarray(budget, float), (n,)).copy(),
                        np.asarray(extent, dtype=np.float64), np.ascontiguousarray(array), include)
    out = np.zeros_like(array)
    out[targets] = sums
    return out if vector else out[:, 0]


def _as_columns(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return np.ascontiguousarray(array[:, None]), False
    return np.ascontiguousarray(array), True
