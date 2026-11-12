"""
Tree kernels, each checked against a plain loop
written after the rule it replaces (rhizodep's distance from tip and supply for elongation, cnwgrass's prefix
maximum of ligule heights and forward writes, adel's frames). Scans and windows are compared bit for bit.
"""
import numpy as np
import pytest

from openalea.metafspm.data_structure.data_api import MPGDataStructure
from plants import branched_root_system


@pytest.fixture
def plant():
    mtg = branched_root_system(n_axes=6, n_segments=30, laterals_every=4)
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
    rng = np.random.default_rng(7)
    return mtg, data_structure, rng


def _by_vid(data_structure, values):
    return dict(zip(data_structure.entity_ids("node").tolist(), np.asarray(values).tolist()))


def _successor(mtg, vid):
    return next((c for c in mtg.children(vid) if mtg.view("edge_type").get(c) == '<'), None)


def test_distance_from_tip_is_identical_to_rhizodeps_loop(plant):
    mtg, data_structure, rng = plant
    length = rng.random(data_structure.n_nodes()) * 1e-3
    lengths = _by_vid(data_structure, length)
    reference = {}

    def distance(vid):                         # rhizodep: d = length at an apex, else d(successor) + length
        if vid not in reference:
            successor = _successor(mtg, vid)
            reference[vid] = lengths[vid] if successor is None else distance(successor) + lengths[vid]
        return reference[vid]

    for vid in lengths:
        distance(vid)
    assert _by_vid(data_structure, data_structure.chain_scan(length, reverse=True)) == reference          # bit for bit


def test_supply_window_is_identical_to_rhizodeps_walk(plant):
    mtg, data_structure, rng = plant
    n = data_structure.n_nodes()
    volume, hexose, mass = rng.random(n) * 1e-9, rng.random(n) * 1e-3, rng.random(n) * 1e-4
    length = np.where(rng.random(n) < 0.1, 0., rng.random(n))                   # some elements of zero length
    budget = rng.random(n) * 6e-9
    apices = np.isin(data_structure.entity_ids("node"), [v for v in data_structure.entity_ids("node") if _successor(mtg, int(v)) is None])
    index = {int(v): i for i, v in enumerate(data_structure.entity_ids("node"))}

    def walk(i):                                 # rhizodep's calculating_supply_for_elongation
        sugar = contributing = 0.
        remaining, current = budget[i], i
        while remaining > 0:
            if remaining > volume[current]:
                if length[current] > 0.:
                    sugar += hexose[current] * mass[current]
                    contributing += mass[current]
                    remaining = remaining - volume[current]
                parent = mtg.parent(int(data_structure.entity_ids("node")[current]))
                if parent is None:
                    break
                current = index[parent]
            else:
                sugar += hexose[current] * mass[current] * remaining / volume[current]
                contributing += mass[current] * remaining / volume[current]
                remaining = 0.
        return sugar, contributing

    sums = data_structure.path_window(budget, volume, np.stack([hexose * mass, mass], axis=1), where=apices, include=length > 0)
    for i in np.flatnonzero(apices):
        assert tuple(sums[i]) == walk(i)                                          # bit for bit
    assert np.all(sums[~apices] == 0.)


def test_accumulations_up_and_down(plant):
    mtg, data_structure, rng = plant
    x = rng.random(data_structure.n_nodes())
    parents = data_structure.parents()
    children = {i: np.flatnonzero(parents == i) for i in range(data_structure.n_nodes())}

    def subtree(i, op):
        values = [x[i]] + [subtree(c, op) for c in children[i]]
        return sum(values) if op == "sum" else max(values)

    np.testing.assert_allclose(data_structure.accumulate(x), [subtree(i, "sum") for i in range(data_structure.n_nodes())], rtol=1e-12)
    np.testing.assert_array_equal(data_structure.accumulate(x, op="max"), [subtree(i, "max") for i in range(data_structure.n_nodes())])

    displacement = rng.random((data_structure.n_nodes(), 3))                                   # vector values: positions
    positions = data_structure.accumulate(displacement, direction="down")
    for i in range(data_structure.n_nodes()):
        path, j = np.zeros(3), i
        while j >= 0:
            path += displacement[j]
            j = parents[j]
        np.testing.assert_allclose(positions[i], path, rtol=1e-12)


def test_rank_chains_prefix_max_and_exclusive_sums(plant):
    _, data_structure, rng = plant
    n = data_structure.n_nodes()
    data_structure.register("axis_id", rng.integers(0, 4, n).astype(float), location="node")
    data_structure.register("rank", rng.permutation(n).astype(float), location="node")
    data_structure.define_chain("phytomers", group="axis_id", rank="rank")
    height, internode = rng.random(n), rng.random(n)
    pseudostem = data_structure.chain_scan(height, chain="phytomers", op="max", exclusive=True)   # cnwgrass: max over lower ranks
    below = data_structure.chain_scan(internode, chain="phytomers", exclusive=True)
    axis, rank = data_structure.get("axis_id"), data_structure.get("rank")
    for i in range(n):
        lower = (axis == axis[i]) & (rank < rank[i])
        assert pseudostem[i] == (height[lower].max() if lower.any() else -np.inf)
        assert below[i] == pytest.approx(internode[lower][np.argsort(rank[lower])].sum(), rel=1e-12)


def test_chain_shift_and_forward_writes(plant):
    mtg, data_structure, rng = plant
    x = rng.random(data_structure.n_nodes())
    previous = data_structure.chain_shift(x, k=1)
    for i, vid in enumerate(data_structure.entity_ids("node")):
        successor = _successor(mtg, int(vid))
        if successor is not None:
            assert previous[data_structure.index_of(successor)] == x[i]
    event = rng.random(data_structure.n_nodes()) < 0.3
    written = data_structure.chain_write(event, x, k=1, base=np.zeros(data_structure.n_nodes()))
    for i, vid in enumerate(data_structure.entity_ids("node")):
        successor = _successor(mtg, int(vid))
        if event[i] and successor is not None:
            assert written[data_structure.index_of(successor)] == x[i]


def test_path_compose_is_the_product_from_the_root(plant):
    _, data_structure, rng = plant
    n = data_structure.n_nodes()
    angles = rng.random(n)
    transforms = np.tile(np.eye(4), (n, 1, 1))
    transforms[:, 0, 0] = transforms[:, 1, 1] = np.cos(angles)
    transforms[:, 0, 1], transforms[:, 1, 0] = -np.sin(angles), np.sin(angles)
    transforms[:, 2, 3] = rng.random(n)
    frames = data_structure.path_compose(transforms)
    parents = data_structure.parents()
    for i in range(0, n, 7):
        path, j = [], i
        while j >= 0:
            path.append(j)
            j = parents[j]
        expected = np.eye(4)
        for k in reversed(path):
            expected = expected @ transforms[k]
        np.testing.assert_allclose(frames[i], expected, rtol=1e-12, atol=1e-12)


def test_kernels_run_on_every_plant_of_a_population_at_once():
    from plants import population
    mtg, roots = population(3)
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
    length = np.arange(1., data_structure.n_nodes() + 1)
    distance = _by_vid(data_structure, data_structure.chain_scan(length, reverse=True))
    totals = _by_vid(data_structure, data_structure.accumulate(length))
    for root in roots:                                   # each plant: a chain of 5 segments from its root
        chain = [root]
        while (s := next((c for c in mtg.children(chain[-1]) if mtg.scale(c) == mtg.scales.SubOrgan), None)) is not None:
            chain.append(s)
        lengths = [dict(zip(data_structure.entity_ids("node").tolist(), length))[v] for v in chain]
        assert distance[root] == sum(reversed(lengths)) and totals[root] == pytest.approx(sum(lengths))


# ---------------------------------------------------------------- consumption shared in rhizodep's order

@pytest.mark.parametrize("overlap", ["partial", "maximal"])
def test_consumption_is_shared_bit_for_bit_as_rhizodeps_loop(plant, overlap):
    from openalea.mtg.traversal import post_order2
    mtg, data_structure, rng = plant
    n = data_structure.n_nodes()
    volume, hexose, mass = rng.random(n) * 1e-9, rng.random(n) * 1e-3, rng.random(n) * 1e-4
    length = np.where(rng.random(n) < 0.1, 0., rng.random(n))
    budget = rng.random(n) * (6e-9 if overlap == "partial" else 1.)          # maximal: every window to the base
    consumption = rng.random(n) * 1e-6
    vids = data_structure.entity_ids("node").tolist()
    index = {v: i for i, v in enumerate(vids)}
    apices = np.isin(vids, [v for v in vids if _successor(mtg, v) is None])

    def walk(i):                                 # rhizodep's calculating_supply_for_elongation, the lists it keeps
        suppliers, provided, total = [], [], 0.
        remaining, current = budget[i], i
        while remaining > 0:
            if remaining > volume[current]:
                if length[current] > 0.:
                    contribution = hexose[current] * mass[current]
                    total += contribution
                    suppliers.append(current)
                    provided.append(contribution)
                    remaining = remaining - volume[current]
                parent = mtg.parent(vids[current])
                if parent is None:
                    break
                current = index[parent]
            else:
                contribution = hexose[current] * mass[current] * remaining / volume[current]
                total += contribution
                suppliers.append(current)
                provided.append(contribution)
                remaining = 0.
        return suppliers, provided, total

    reference = np.zeros(n)                      # actual_growth_and_corresponding_respiration, apices in post order
    for root in data_structure.roots().tolist():
        for vid in post_order2(mtg, vids[root]):
            if vid in index and apices[index[vid]]:
                suppliers, provided, total = walk(index[vid])
                for supplier, contribution in zip(suppliers, provided):
                    reference[supplier] += consumption[index[vid]] * contribution / total

    contributions = data_structure.path_contributions(budget, volume, hexose * mass, where=apices, include=length > 0)
    totals = data_structure.path_window(budget, volume, hexose * mass, where=apices, include=length > 0)
    shared = data_structure.scatter_contributions(contributions, consumption, totals)
    np.testing.assert_array_equal(shared, reference)                         # bit for bit
    owners, suppliers, _ = contributions
    if overlap == "maximal":
        assert np.bincount(suppliers).max() > 5                              # segments shared by many apices


def test_the_openalea_post_order_is_post_order2(plant):
    from openalea.mtg.traversal import post_order2
    mtg, data_structure, _ = plant
    vids = data_structure.entity_ids("node").tolist()
    expected = [v for root in data_structure.roots().tolist() for v in post_order2(mtg, vids[root])]
    assert [vids[i] for i in data_structure.order("post", traversal="openalea")] == expected


def test_chains_at_a_coarse_location(plant):
    """define_chain(location=...): the entities of a coarse location chained by group (an owner at a coarser
    location, the Axis) and rank (a variable at that location); the chain kernels take and give one value per
    entity there, chain_gather's source by default each entity's own chain (a phytomer offset ranks below its own
    on its axis: use case 1's nodal root sites)."""
    _, data_structure, rng = plant
    location = "Organ"
    count = data_structure.entity_ids(location).size
    axis = np.asarray(data_structure.owner("Axis", of=location))
    rank = np.zeros(count)
    for group in np.unique(axis):                                 # 1, 2, ... along each axis, shuffled
        members = np.flatnonzero(axis == group)
        rank[members] = rng.permutation(members.size) + 1.
    data_structure.register("organ_rank", rank, location=location)
    data_structure.define_chain("organs", location=location, group="Axis", rank="organ_rank")
    values = rng.random(count)
    below = data_structure.chain_shift(values, chain="organs")
    total = data_structure.chain_scan(values, chain="organs")
    site = data_structure.chain_gather(np.arange(count, dtype=float), rank - 2, chain="organs", fill=-1.)
    for i in range(count):
        same = axis == axis[i]
        lower = same & (rank == rank[i] - 1)
        assert (below[i] == values[lower][0]) if lower.any() else np.isnan(below[i])
        assert total[i] == pytest.approx(values[same & (rank <= rank[i])].sum(), rel=1e-12)
        wanted = np.flatnonzero(same & (rank == rank[i] - 2))
        assert site[i] == (wanted[0] if wanted.size else -1.)
    with pytest.raises(ValueError, match="group and rank"):
        data_structure.define_chain("bad", edge_type="<", location=location)


def _python_orders(indptr, indices, roots):
    """The depth-first pre and post orders with an explicit stack of (node, done), the loop the kernel replaced."""
    pre, post = [], []
    for root in roots:
        stack = [(int(root), False)]
        while stack:
            node, done = stack.pop()
            if done:
                post.append(node)
                continue
            pre.append(node)
            stack.append((node, True))
            for child in indices[indptr[node]:indptr[node + 1]][::-1]:
                stack.append((int(child), False))
    return np.array(pre, dtype=np.int64), np.array(post, dtype=np.int64)


def _python_levels(parents):
    """Levels by pointer doubling of the depths and a stable sort, the code the kernel replaced."""
    dist, jump = (parents >= 0).astype(np.int64), parents.copy()
    while True:
        idx = np.flatnonzero(jump >= 0)
        if idx.size == 0:
            break
        above = jump[idx]
        dist_above, jump_above = dist[above].copy(), jump[above].copy()
        dist[idx] += dist_above
        jump[idx] = jump_above
    order = np.argsort(dist, kind="stable")
    bounds = np.flatnonzero(np.r_[True, np.diff(dist[order]) != 0, True])
    return dist, [order[bounds[i]:bounds[i + 1]] for i in range(bounds.size - 1)]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_traversal_and_levels_kernels_are_the_loops_they_replace(seed):
    """kernels.depth_first_orders (tree_kernels.traversal), tree_depths and tree_levels on random forests (children
    in a given order, several roots, a single node, none) equal the Python loops, element for element."""
    from openalea.metafspm.data_structure import tree_kernels
    rng = np.random.default_rng(seed)
    for n in (0, 1, 2, 50, 400):
        parents = np.array([-1 if (v == 0 or rng.random() < 0.05) else int(rng.integers(0, v)) for v in range(n)],
                           dtype=np.int64)
        relabel = rng.permutation(n)                       # parents not before children, children in any order
        shuffled = np.full(n, -1, dtype=np.int64)
        shuffled[relabel] = np.where(parents >= 0, relabel[np.maximum(parents, 0)], -1)
        for parent in (parents, shuffled):
            result = tree_kernels.traversal(parent)
            indptr, indices = result["children"]
            pre, post = _python_orders(indptr, indices, np.flatnonzero(parent < 0))
            np.testing.assert_array_equal(result["pre"], pre)
            np.testing.assert_array_equal(result["post"], post)
            reversed_children = np.concatenate([indices[indptr[v]:indptr[v + 1]][::-1] for v in range(n)]) \
                if n else indices
            given = tree_kernels.traversal(parent, children=(indptr, reversed_children))
            expected = _python_orders(indptr, reversed_children, np.flatnonzero(parent < 0))
            np.testing.assert_array_equal(given["pre"], expected[0])
            np.testing.assert_array_equal(given["post"], expected[1])
            depth, levels = _python_levels(parent)
            np.testing.assert_array_equal(tree_kernels.depth(parent), depth)
            got = tree_kernels.levels(parent)
            assert len(got) == len(levels) and all(np.array_equal(a, b) for a, b in zip(got, levels))
    with pytest.raises(ValueError):
        tree_kernels.depth(np.array([1, 0], dtype=np.int64))      # a cycle


def test_a_scan_cut_at_some_nodes_is_the_loop_stopping_there(plant):
    """chain_scan(cut=): the distance from tip restarting at the cut nodes (ArchiSimple's apices and nodules: d =
    length at a cut node or a tip, else the successor's d + length), bit for bit, and the tips' ids carried down."""
    mtg, data_structure, rng = plant
    length = rng.random(data_structure.n_nodes()) * 1e-3
    cut = rng.random(data_structure.n_nodes()) < 0.1
    vids = data_structure.entity_ids("node").tolist()
    index = {v: i for i, v in enumerate(vids)}
    reference, tips = {}, {}

    def distance(vid):
        if vid not in reference:
            successor = _successor(mtg, vid)
            if successor is None or cut[index[vid]]:
                reference[vid], tips[vid] = length[index[vid]], vid
            else:
                reference[vid], tips[vid] = distance(successor) + length[index[vid]], tips[distance_tip(successor)]
        return reference[vid]

    def distance_tip(vid):
        distance(vid)
        return vid

    for vid in vids:
        distance(vid)
    after = data_structure.chain_scan(np.ones(len(vids)), reverse=True, exclusive=True, cut=cut)
    scanned = data_structure.chain_scan(np.column_stack([length, np.where(after == 0., np.asarray(vids, float), 0.)]),
                            reverse=True, cut=cut)
    assert _by_vid(data_structure, scanned[:, 0]) == reference                                  # bit for bit
    assert _by_vid(data_structure, scanned[:, 1].astype(np.int64)) == tips


def test_the_preorder_is_the_walk_branches_first_with_deferred_vertices(plant):
    """data_structure.preorder: from each base node in the nodes' order, depth first over the MPG's children (in its order),
    branches before successors, deferred vertices visited as the last branches of others: an explicit-stack walk."""
    mtg, data_structure, rng = plant
    vids = data_structure.entity_ids("node").tolist()
    position = {v: i for i, v in enumerate(vids)}
    edge = mtg.view("edge_type")
    branches = [v for v in vids if edge.get(v) == "+" and mtg.parent(v) in position]
    pairs = [(position[v], position[vids[int(rng.integers(0, len(vids)))]]) for v in branches[:3]]
    pairs = [(v, after) for v, after in pairs if after != v and not _below(mtg, vids[after], vids[v])]
    for defer in ([], pairs):
        moved = {vids[v] for v, _ in defer}
        later = {}
        for v, after in defer:
            later.setdefault(vids[after], []).append(vids[v])
        order = []
        for root in [v for v in vids if mtg.parent(v) not in position]:
            stack = [root]
            while stack:
                vid = stack.pop()
                order.append(position[vid])
                children = [c for c in mtg.children(vid) if c in position and c not in moved]
                stack.extend(reversed([c for c in children if edge.get(c) != "<"] + later.get(vid, [])
                                      + [c for c in children if edge.get(c) == "<"]))
        np.testing.assert_array_equal(data_structure.preorder(defer=defer), order)


def _below(mtg, vid, ancestor) -> bool:
    while vid is not None:
        if vid == ancestor:
            return True
        vid = mtg.parent(vid)
    return False


def test_a_local_gather_is_the_loop_over_owners(plant):
    """data_structure.local_gather: per entity, its owner's table at its rank among the owner's entities, as a loop writing each
    owner's table at its entities (ascending); the fill where an owner or its key is none."""
    _, data_structure, rng = plant
    n, owners = 500, 40
    owner = np.where(rng.random(n) < 0.1, -1, rng.integers(0, owners, n))
    counts = np.bincount(owner[owner >= 0], minlength=owners)
    tables = [rng.random(size) for size in sorted(set(counts.tolist()))]
    key = np.array([sorted(set(counts.tolist())).index(c) for c in counts], dtype=np.int64)
    key[::7] = -1
    expected = np.full(n, -2.)
    for o in range(owners):
        if key[o] >= 0:
            expected[np.flatnonzero(owner == o)] = tables[key[o]]
    got = data_structure.local_gather(owner, key, tables, fill=-2.)
    assert got.tobytes() == expected.tobytes()
    with pytest.raises(ValueError):
        data_structure.local_gather(np.array([0, 0]), np.array([0]), [np.ones(1)])               # more entities than rows
