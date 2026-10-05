"""
Tree kernels, each checked against a plain loop
written after the rule it replaces (rhizodep's distance from tip and supply for elongation, cnwgrass's prefix
maximum of ligule heights and forward writes, adel's frames). Scans and windows are compared bit for bit.
"""
import numpy as np
import pytest

from openalea.metafspm.data_structure.data_api import MPGDataStructure
from test_topology_arrays import branched_root_system


@pytest.fixture
def plant():
    g = branched_root_system(n_axes=6, n_segments=30, laterals_every=4)
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    rng = np.random.default_rng(7)
    return g, ds, rng


def _by_vid(ds, values):
    return dict(zip(ds.entity_ids("node").tolist(), np.asarray(values).tolist()))


def _successor(g, vid):
    return next((c for c in g.children(vid) if g.property("edge_type").get(c) == '<'), None)


def test_distance_from_tip_is_identical_to_rhizodeps_loop(plant):
    g, ds, rng = plant
    length = rng.random(ds.n_nodes()) * 1e-3
    lengths = _by_vid(ds, length)
    reference = {}

    def distance(vid):                         # rhizodep: d = length at an apex, else d(successor) + length
        if vid not in reference:
            successor = _successor(g, vid)
            reference[vid] = lengths[vid] if successor is None else distance(successor) + lengths[vid]
        return reference[vid]

    for vid in lengths:
        distance(vid)
    assert _by_vid(ds, ds.chain_scan(length, reverse=True)) == reference          # bit for bit


def test_supply_window_is_identical_to_rhizodeps_walk(plant):
    g, ds, rng = plant
    n = ds.n_nodes()
    volume, hexose, mass = rng.random(n) * 1e-9, rng.random(n) * 1e-3, rng.random(n) * 1e-4
    length = np.where(rng.random(n) < 0.1, 0., rng.random(n))                   # some elements of zero length
    budget = rng.random(n) * 6e-9
    apices = np.isin(ds.entity_ids("node"), [v for v in ds.entity_ids("node") if _successor(g, int(v)) is None])
    index = {int(v): i for i, v in enumerate(ds.entity_ids("node"))}

    def walk(i):                                 # rhizodep's calculating_supply_for_elongation
        sugar = contributing = 0.
        remaining, current = budget[i], i
        while remaining > 0:
            if remaining > volume[current]:
                if length[current] > 0.:
                    sugar += hexose[current] * mass[current]
                    contributing += mass[current]
                    remaining = remaining - volume[current]
                parent = g.parent(int(ds.entity_ids("node")[current]))
                if parent is None:
                    break
                current = index[parent]
            else:
                sugar += hexose[current] * mass[current] * remaining / volume[current]
                contributing += mass[current] * remaining / volume[current]
                remaining = 0.
        return sugar, contributing

    sums = ds.path_window(budget, volume, np.stack([hexose * mass, mass], axis=1), where=apices, include=length > 0)
    for i in np.flatnonzero(apices):
        assert tuple(sums[i]) == walk(i)                                          # bit for bit
    assert np.all(sums[~apices] == 0.)


def test_accumulations_up_and_down(plant):
    g, ds, rng = plant
    x = rng.random(ds.n_nodes())
    parents = ds.parents()
    children = {i: np.flatnonzero(parents == i) for i in range(ds.n_nodes())}

    def subtree(i, op):
        values = [x[i]] + [subtree(c, op) for c in children[i]]
        return sum(values) if op == "sum" else max(values)

    np.testing.assert_allclose(ds.accumulate(x), [subtree(i, "sum") for i in range(ds.n_nodes())], rtol=1e-12)
    np.testing.assert_array_equal(ds.accumulate(x, op="max"), [subtree(i, "max") for i in range(ds.n_nodes())])

    displacement = rng.random((ds.n_nodes(), 3))                                   # vector values: positions
    positions = ds.accumulate(displacement, direction="down")
    for i in range(ds.n_nodes()):
        path, j = np.zeros(3), i
        while j >= 0:
            path += displacement[j]
            j = parents[j]
        np.testing.assert_allclose(positions[i], path, rtol=1e-12)


def test_rank_chains_prefix_max_and_exclusive_sums(plant):
    _, ds, rng = plant
    n = ds.n_nodes()
    ds.register("axis_id", rng.integers(0, 4, n).astype(float), location="node")
    ds.register("rank", rng.permutation(n).astype(float), location="node")
    ds.define_chain("phytomers", group="axis_id", rank="rank")
    height, internode = rng.random(n), rng.random(n)
    pseudostem = ds.chain_scan(height, chain="phytomers", op="max", exclusive=True)   # cnwgrass: max over lower ranks
    below = ds.chain_scan(internode, chain="phytomers", exclusive=True)
    axis, rank = ds.get("axis_id"), ds.get("rank")
    for i in range(n):
        lower = (axis == axis[i]) & (rank < rank[i])
        assert pseudostem[i] == (height[lower].max() if lower.any() else -np.inf)
        assert below[i] == pytest.approx(internode[lower][np.argsort(rank[lower])].sum(), rel=1e-12)


def test_chain_shift_and_forward_writes(plant):
    g, ds, rng = plant
    x = rng.random(ds.n_nodes())
    previous = ds.chain_shift(x, k=1)
    for i, vid in enumerate(ds.entity_ids("node")):
        successor = _successor(g, int(vid))
        if successor is not None:
            assert previous[ds.index_of(successor)] == x[i]
    event = rng.random(ds.n_nodes()) < 0.3
    written = ds.chain_write(event, x, k=1, base=np.zeros(ds.n_nodes()))
    for i, vid in enumerate(ds.entity_ids("node")):
        successor = _successor(g, int(vid))
        if event[i] and successor is not None:
            assert written[ds.index_of(successor)] == x[i]


def test_path_compose_is_the_product_from_the_root(plant):
    _, ds, rng = plant
    n = ds.n_nodes()
    angles = rng.random(n)
    transforms = np.tile(np.eye(4), (n, 1, 1))
    transforms[:, 0, 0] = transforms[:, 1, 1] = np.cos(angles)
    transforms[:, 0, 1], transforms[:, 1, 0] = -np.sin(angles), np.sin(angles)
    transforms[:, 2, 3] = rng.random(n)
    frames = ds.path_compose(transforms)
    parents = ds.parents()
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
    from test_population_graph import _population
    g, roots = _population(3)
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    length = np.arange(1., ds.n_nodes() + 1)
    distance = _by_vid(ds, ds.chain_scan(length, reverse=True))
    totals = _by_vid(ds, ds.accumulate(length))
    for root in roots:                                   # each plant: a chain of 5 segments from its root
        chain = [root]
        while (s := next((c for c in g.children(chain[-1]) if g.scale(c) == g.scales.SubOrgan), None)) is not None:
            chain.append(s)
        lengths = [dict(zip(ds.entity_ids("node").tolist(), length))[v] for v in chain]
        assert distance[root] == sum(reversed(lengths)) and totals[root] == pytest.approx(sum(lengths))


# ---------------------------------------------------------------- consumption shared in rhizodep's order

@pytest.mark.parametrize("overlap", ["partial", "maximal"])
def test_consumption_is_shared_bit_for_bit_as_rhizodeps_loop(plant, overlap):
    from openalea.mtg.traversal import post_order2
    g, ds, rng = plant
    n = ds.n_nodes()
    volume, hexose, mass = rng.random(n) * 1e-9, rng.random(n) * 1e-3, rng.random(n) * 1e-4
    length = np.where(rng.random(n) < 0.1, 0., rng.random(n))
    budget = rng.random(n) * (6e-9 if overlap == "partial" else 1.)          # maximal: every window to the base
    consumption = rng.random(n) * 1e-6
    vids = ds.entity_ids("node").tolist()
    index = {v: i for i, v in enumerate(vids)}
    apices = np.isin(vids, [v for v in vids if _successor(g, v) is None])

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
                parent = g.parent(vids[current])
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
    for root in ds.roots().tolist():
        for vid in post_order2(g, vids[root]):
            if vid in index and apices[index[vid]]:
                suppliers, provided, total = walk(index[vid])
                for supplier, contribution in zip(suppliers, provided):
                    reference[supplier] += consumption[index[vid]] * contribution / total

    contributions = ds.path_contributions(budget, volume, hexose * mass, where=apices, include=length > 0)
    totals = ds.path_window(budget, volume, hexose * mass, where=apices, include=length > 0)
    shared = ds.scatter_contributions(contributions, consumption, totals)
    np.testing.assert_array_equal(shared, reference)                         # bit for bit
    owners, suppliers, _ = contributions
    if overlap == "maximal":
        assert np.bincount(suppliers).max() > 5                              # segments shared by many apices


def test_the_openalea_post_order_is_post_order2(plant):
    from openalea.mtg.traversal import post_order2
    g, ds, _ = plant
    vids = ds.entity_ids("node").tolist()
    expected = [v for root in ds.roots().tolist() for v in post_order2(g, vids[root])]
    assert [vids[i] for i in ds.order("post", convention="openalea")] == expected
