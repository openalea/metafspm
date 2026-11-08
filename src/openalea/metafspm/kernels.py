"""
metafspm's numba kernels, in one place (conventions.md, "Performance rules"): the loops that stay sequential (tree
scans, chains, path windows, segment pieces, the turtle, a reduction's groups' terms), each @njit(cache=True),
nogil=True for those a step runs on the whole scene, parallel=True only where the iterations are independent and the
work is large (numba's threads Compute.parallel's: the Compute's threads, at most one per chunk and the section's
ceiling), over chunks made from the work only (work_chunks). Called through their modules' functions
(data_structure.tree_kernels, coupling.geometry, coupling.cross, solve.reduction), which time them (run.json).
warm_up() runs every one on tiny inputs, which fills numba's on-disk cache (NUMBA_CACHE_DIR for a read-only install):
compute.warm_up runs it with every port's.
"""
from math import cos, radians, sin, sqrt

import numpy as np
from numba import njit, prange


# ── Tree kernels (data_structure.tree_kernels' chains, scans and path windows) ──


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


@njit(cache=True)
def _count_window(targets, parents, budget, extent, include):
    counts = np.zeros(targets.size, dtype=np.int64)
    for t in range(targets.size):
        remaining, current, count = budget[targets[t]], targets[t], 0
        while remaining > 0:
            if remaining > extent[current]:
                if include[current]:
                    count += 1
                    remaining = remaining - extent[current]
                current = parents[current]
                if current < 0:
                    break
            else:
                count += 1
                remaining = 0.
        counts[t] = count
    return counts


@njit(cache=True)
def _window_contributions(targets, parents, budget, extent, values, include, offsets, supplier, contribution):
    for t in prange(targets.size):
        remaining, current, k = budget[targets[t]], targets[t], offsets[t]
        while remaining > 0:
            if remaining > extent[current]:
                if include[current]:
                    supplier[k] = current
                    contribution[k] = values[current]
                    k += 1
                    remaining = remaining - extent[current]
                current = parents[current]
                if current < 0:
                    break
            else:
                supplier[k] = current
                contribution[k] = values[current] * remaining / extent[current]
                k += 1
                remaining = 0.


@njit(cache=True, nogil=True)
def depth_first_orders(indptr, indices, roots, n):
    """
    (pre, post): the depth-first pre and post orders of a forest of *n* nodes given by its children's CSR (*indptr*,
    *indices*: the children of node v are indices[indptr[v]:indptr[v + 1]], in that order), from each of *roots* in
    turn: a node, then each child's subtree in its order (pre); each child's subtree, then the node (post). One
    explicit stack (a node and its next child), as a recursive visit; the nodes not reached are left out.
    """
    pre = np.empty(n, dtype=np.int64)
    post = np.empty(n, dtype=np.int64)
    stack_node = np.empty(n, dtype=np.int64)
    stack_next = np.empty(n, dtype=np.int64)
    visited = 0
    finished = 0
    for r in range(roots.size):
        root = roots[r]
        top = 0
        stack_node[0] = root
        stack_next[0] = indptr[root]
        pre[visited] = root
        visited += 1
        while top >= 0:
            node = stack_node[top]
            k = stack_next[top]
            if k < indptr[node + 1]:
                stack_next[top] = k + 1
                child = indices[k]
                if visited >= n or top + 1 >= n:
                    raise ValueError("depth_first_orders: the graph is not a forest (a node reached twice)")
                top += 1
                stack_node[top] = child
                stack_next[top] = indptr[child]
                pre[visited] = child
                visited += 1
            else:
                post[finished] = node
                finished += 1
                top -= 1
    return pre[:visited], post[:finished]


@njit(cache=True, nogil=True)
def tree_depths(parents):
    """The number of edges from each node to its root in the forest given by *parents* (-1 at a root): each node's
    chain walked up to the first node of known depth, the chain's depths written down from it."""
    n = parents.size
    depth = np.full(n, -1, dtype=np.int64)
    chain = np.empty(n, dtype=np.int64)
    for v in range(n):
        if depth[v] >= 0:
            continue
        length = 0
        node = v
        while node >= 0 and depth[node] < 0:
            if length >= n:
                raise ValueError("tree_depths: the parents hold a cycle")
            chain[length] = node
            length += 1
            node = parents[node]
        known = -1 if node < 0 else depth[node]
        for i in range(length - 1, -1, -1):
            known += 1
            depth[chain[i]] = known
    return depth


@njit(cache=True, nogil=True)
def tree_levels(parents):
    """(order, offsets): the nodes grouped by depth from the roots (tree_depths), ascending within each depth (a
    stable counting sort): level d is order[offsets[d]:offsets[d + 1]]."""
    depth = tree_depths(parents)
    n = parents.size
    deepest = -1
    for v in range(n):
        if depth[v] > deepest:
            deepest = depth[v]
    offsets = np.zeros(deepest + 2, dtype=np.int64)
    for v in range(n):
        offsets[depth[v] + 1] += 1
    for d in range(deepest + 1):
        offsets[d + 1] += offsets[d]
    cursor = offsets[:-1].copy()
    order = np.empty(n, dtype=np.int64)
    for v in range(n):
        order[cursor[depth[v]]] = v
        cursor[depth[v]] += 1
    return order, offsets


# ── The turtle's frame operations (PlantGL's PglTurtle), on heading and up arrays (3,) changed in place ──


@njit(cache=True)
def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


@njit(cache=True)
def _rotate(vector, axis, angle):
    """*vector* rotated by *angle* (radians) about the unit *axis* (Rodrigues, right-handed)."""
    c, s = cos(angle), sin(angle)
    along = (axis[0] * vector[0] + axis[1] * vector[1] + axis[2] * vector[2]) * (1. - c)
    across = _cross(axis, vector)
    return vector * c + across * s + axis * along


@njit(cache=True)
def left_of(heading, up):
    """The frame's left: U × H."""
    return _cross(up, heading)


@njit(cache=True)
def roll_right(heading, up, angle):
    """PglTurtle.rollR(angle): the frame turned about its heading, its up towards its left."""
    up[:] = _rotate(up, heading, -radians(angle))


@njit(cache=True)
def down(heading, up, angle):
    """PglTurtle.down(angle): the frame turned about its left, its heading towards its up's opposite."""
    left = left_of(heading, up)
    heading[:] = _rotate(heading, left, radians(angle))
    up[:] = _rotate(up, left, radians(angle))


@njit(cache=True)
def turn_left(heading, up, angle):
    """PglTurtle.left(angle): the frame turned about its up, its heading towards its left."""
    heading[:] = _rotate(heading, up, radians(angle))


@njit(cache=True)
def roll_to_vertical(heading, up):
    """PglTurtle.rollToVert(): the frame rolled about its heading so that its left is horizontal (L along z × H;
    (0, -1, 0) when the heading is vertical, as PlantGL)."""
    left = np.array([-heading[1], heading[0], 0.])
    norm = sqrt(left[0] ** 2 + left[1] ** 2)
    left = left / norm if norm > 1e-12 else np.array([0., -1., 0.])
    up[:] = _cross(heading, left)


@njit(cache=True)
def follow_parents(order, parent, base_position, base_heading, base_up, pre_roll, down_angle, roll, elasticity, length,
                   tropism):
    """
    Elements placed from their parent's end, as PlantGL's turtle draws a tree of segments: per element, from its
    parent's end frame (from its *base_* position, heading and up when it has no parent, *parent* -1):

    1. rollL(*pre_roll*) (°);
    2. down(*down_angle*) (°);
    3. rollL(*roll*) (°);
    4. the tropism: the frame turned about H × *tropism* by *elasticity* times the angle between H and *tropism*
       (PglTurtle.F's, whatever the length; none for a zero length, as F(0));
    5. forward by *length*.

    *order*: a pre-order of the elements (a parent before its children, e.g. ds.order("pre")); *parent*: each one's
    parent's index, -1 for none; per element arrays, base_* (n, 3). Returns (position, heading, up, end), each (n, 3):
    the element's start, its frame after the tropism (the one it is drawn in, and its children start from), its end.
    """
    n = parent.size
    position, heading, up, end = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
    norm_tropism = sqrt(tropism[0] ** 2 + tropism[1] ** 2 + tropism[2] ** 2)
    target = tropism / norm_tropism if norm_tropism > 0. else tropism
    for i in order:
        k = parent[i]
        if k < 0:
            p, h, u = base_position[i].copy(), base_heading[i].copy(), base_up[i].copy()
        else:
            p, h, u = end[k].copy(), heading[k].copy(), up[k].copy()
        if pre_roll[i] != 0.:
            roll_right(h, u, -pre_roll[i])
        if down_angle[i] != 0.:
            down(h, u, down_angle[i])
        if roll[i] != 0.:
            roll_right(h, u, -roll[i])
        if elasticity[i] != 0. and norm_tropism > 0. and length[i] != 0.:
            axis = _cross(h, target)
            sine = sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
            if sine > 1e-12:
                cosine = min(1., max(-1., h[0] * target[0] + h[1] * target[1] + h[2] * target[2]))
                angle = elasticity[i] * np.arccos(cosine)
                axis = axis / sine
                h[:] = _rotate(h, axis, angle)
                u[:] = _rotate(u, axis, angle)
        position[i], heading[i], up[i] = p, h, u
        end[i] = p + h * length[i]
    return position, heading, up, end


# ── Segment pieces in cells (coupling.cross: length overlap) ──


@njit(cache=True)
def _count_pieces(p1, p2, origin, dx):
    n, d = p1.shape
    counts = np.empty(n, np.int64)
    for i in range(n):
        c = 1
        for a in range(d):
            lo, hi = min(p1[i, a], p2[i, a]), max(p1[i, a], p2[i, a])
            c += int(np.floor((hi - origin[a]) / dx[a]) - np.floor((lo - origin[a]) / dx[a]))
        counts[i] = c
    return counts


@njit(cache=True)
def _pieces(p1, p2, origin, dx, counts, offsets, middles, fractions):
    """Pieces of each segment between the cell faces it crosses: middle point and length fraction."""
    n, d = p1.shape
    for i in range(n):
        m = counts[i]
        ts = np.empty(m + 1)
        ts[0], j = 0., 1
        for a in range(d):
            delta = p2[i, a] - p1[i, a]
            if delta == 0.:
                continue
            lo, hi = min(p1[i, a], p2[i, a]), max(p1[i, a], p2[i, a])
            first = np.floor((lo - origin[a]) / dx[a]) + 1.
            last = np.floor((hi - origin[a]) / dx[a])
            k = first
            while k <= last:
                ts[j] = (origin[a] + k * dx[a] - p1[i, a]) / delta
                j += 1
                k += 1.
        ts[j] = 1.
        ts = np.sort(ts[:j + 1])
        for q in range(m):
            t = 0.5 * (ts[q] + ts[q + 1])
            for a in range(d):
                middles[offsets[i] + q, a] = p1[i, a] + t * (p2[i, a] - p1[i, a])
            fractions[offsets[i] + q] = ts[q + 1] - ts[q]


# ── A reduction's groups' terms (solve.reduction: each group's dense kept × kept block in the scene's CSR) ──


@njit(cache=True, nogil=True)
def group_pattern(local, offsets, n):
    """
    The CSR pattern (indptr, indices, sorted per row: scipy's canonical one) of dense blocks local[offsets[g]:
    offsets[g + 1]] × the same, one per group, on *n* rows; slots: each group's (row, column) entry's position in
    the CSR data, row by row (concatenated, k² per group of k nodes). A node is in one group at most (ValueError).
    """
    groups = offsets.size - 1
    counts = np.zeros(n, dtype=np.int64)
    size = 0
    for g in range(groups):
        k = offsets[g + 1] - offsets[g]
        size += k * k
        for i in range(offsets[g], offsets[g + 1]):
            if counts[local[i]] != 0:
                raise ValueError("a node in two reduced groups")
            counts[local[i]] = k
    indptr = np.zeros(n + 1, dtype=np.int64)
    for r in range(n):
        indptr[r + 1] = indptr[r] + counts[r]
    indices = np.empty(indptr[n], dtype=np.int32)
    slots = np.empty(size, dtype=np.int32)
    at = 0
    for g in range(groups):
        nodes = local[offsets[g]:offsets[g + 1]]
        k = nodes.size
        order = np.argsort(nodes)
        rank = np.empty(k, dtype=np.int64)
        for q in range(k):
            rank[order[q]] = q
        for i in range(k):
            start = indptr[nodes[i]]
            for q in range(k):
                indices[start + q] = nodes[order[q]]
            for j in range(k):
                slots[at] = start + rank[j]
                at += 1
    return indptr, indices, slots


@njit(cache=True, nogil=True, parallel=True)
def scatter_weighted_blocks(data, slots, kinds, weights, blocks, chunks):
    """
    data[slots[g, i, j]] = the weighted sum of the groups' kinds' blocks: for each group g of one template (k × k
    blocks), from 0, corner by corner (kinds[g, c] >= 0: blocks[kinds[g, c]], weighted by weights[g, c]), as
    numpy's corner by corner accumulation gives it. Parallel over *chunks* (group boundaries, made from the work
    only): each entry written once, the same bits for any thread count.
    """
    k = slots.shape[1]
    for chunk in prange(chunks.size - 1):
        total = np.empty((k, k))                 # one buffer a chunk: no allocation per group (threads contend)
        for g in range(chunks[chunk], chunks[chunk + 1]):
            total[:] = 0.
            for c in range(kinds.shape[1]):
                kind = kinds[g, c]
                if kind < 0:
                    continue
                block, weight = blocks[kind], weights[g, c]
                for i in range(k):
                    for j in range(k):
                        total[i, j] += weight * block[i, j]
            for i in range(k):
                for j in range(k):
                    data[slots[g, i, j]] = total[i, j]


# ── Gathers by owner (DataStructure.local_gather) ──


@njit(cache=True, nogil=True)
def gather_local(owner, owner_key, table_offsets, flat, fill):
    """
    Per entity e: the value at its local row of its owner's table, flat[table_offsets[k] + row] with k =
    owner_key[owner[e]], its local row its rank among its owner's entities (ascending e); *fill* where its owner or
    the owner's key is -1. A row beyond its table raises (the owner's entities and its table differ in size).
    """
    n = owner.size
    out = np.empty(n)
    rows = np.zeros(owner_key.size, dtype=np.int64)
    for e in range(n):
        o = owner[e]
        if o < 0:
            out[e] = fill
            continue
        row = rows[o]
        rows[o] = row + 1
        k = owner_key[o]
        if k < 0:
            out[e] = fill
            continue
        if row >= table_offsets[k + 1] - table_offsets[k]:
            raise ValueError("gather_local: an owner has more entities than its table has rows")
        out[e] = flat[table_offsets[k] + row]
    return out


def work_chunks(count: int, work_each: int, target: int = 1 << 16) -> np.ndarray:
    """Boundaries of chunks of *count* items of *work_each* work (e.g. nonzeros) each, about *target* work a chunk:
    from the work only, never the thread count (determinism, conventions.md)."""
    step = max(1, target // max(work_each, 1))
    return np.append(np.arange(0, count, step, dtype=np.int64), np.int64(count))


# ── Warm-up ──

def warm_up() -> None:
    """Every kernel compiled (or loaded from numba's cache) on tiny inputs of the types its callers pass: a cold
    start's compilation done once, at install or in CI (compute.warm_up runs it with every port's)."""
    from openalea.metafspm.data_structure import tree_kernels
    parents = np.array([-1, 0, 1, 1], dtype=np.int64)
    chains = tree_kernels.chains_from_predecessors(np.array([-1, 0, 1, -1], dtype=np.int64))
    tree_kernels.chains_from_groups(np.array([0, 0, 1, 1]), np.array([1, 0, 0, 1]))
    values = np.arange(4, dtype=np.float64)
    for op in ("sum", "max"):
        tree_kernels.chain_scan(values, chains, op=op)
        tree_kernels.chain_scan(np.column_stack([values, values]), chains, op=op, reverse=True, exclusive=True)
    extent = np.ones(4)
    tree_kernels.path_window(parents, 2., extent, values)
    tree_kernels.levels(parents), tree_kernels.depth(parents)
    tree_kernels.traversal(parents)
    tree_kernels.path_contributions(parents, 2., extent, values, np.arange(4, dtype=np.int64))
    base = np.zeros((2, 3))
    heading, up = np.tile([0., 0., 1.], (2, 1)), np.tile([1., 0., 0.], (2, 1))
    angles = np.array([0., 10.])
    follow_parents(np.array([0, 1], dtype=np.int64), np.array([-1, 0], dtype=np.int64), base, heading, up, angles,
                   angles, angles, np.array([0., 0.1]), np.ones(2), np.array([0., 0., -1.]))
    h, u = np.array([0., 0., 1.]), np.array([1., 0., 0.])
    roll_right(h, u, 5.), down(h, u, 5.), turn_left(h, u, 5.), roll_to_vertical(h, u), left_of(h, u)
    _rotate(h, u, 0.1), _cross(h, u)
    p1, p2 = np.zeros((2, 3)), np.array([[0.3, 0.2, 0.1], [0.05, 0.05, 0.05]])
    origin, dx = np.zeros(3), np.full(3, 0.1)
    counts = _count_pieces(p1, p2, origin, dx)
    offsets = np.zeros(2, dtype=np.int64)
    np.cumsum(counts[:-1], out=offsets[1:])
    _pieces(p1, p2, origin, dx, counts, offsets, np.empty((int(counts.sum()), 3)), np.empty(int(counts.sum())))
    from numba.typed import List
    indptr, indices, slots = group_pattern(np.array([2, 0, 3], dtype=np.int64), np.array([0, 2, 3], dtype=np.int64),
                                           4)
    blocks = List([np.ones((2, 2)), np.full((2, 2), 2.)])
    scatter_weighted_blocks(np.empty(indices.size), slots[:4].reshape(1, 2, 2), np.array([[0, 1]], dtype=np.int64),
                            np.array([[0.5, 0.5]]), blocks, work_chunks(1, 4))
    gather_local(np.array([0, 1, 0, -1], dtype=np.int64), np.array([0, 1], dtype=np.int64),
                 np.array([0, 2, 3], dtype=np.int64), np.array([1., 2., 3.]), np.nan)

