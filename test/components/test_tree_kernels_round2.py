"""
Tree kernels, round 2: folds with a custom function, gathers and recurrences along chains, each
checked against a loop written after the rule it replaces (rhizodep's pipe model and death propagation in
potential_growth, Root-CyNAPS' barrier reopening, elongwheat's tiller cohort reads, a turtle-like frame), exactly.
"""
from math import pi, sqrt

import numpy as np
import pytest
from openalea.mtg.traversal import post_order2, pre_order2

from openalea.metafspm.data_structure.data_api import MPGDataStructure
from plants import branched_root_system

DEAD, JUST_DEAD, NODULE, ALIVE = 4, 3, 5, 1


@pytest.fixture
def plant():
    g = branched_root_system(n_axes=6, n_segments=30, laterals_every=4)
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    vids = ds.entity_ids("node").tolist()
    return g, ds, vids, {v: i for i, v in enumerate(vids)}, np.random.default_rng(3)


def test_the_pipe_model_is_rhizodeps(plant):
    g, ds, vids, index, rng = plant
    n, SGC = ds.n_nodes(), 0.3
    initial = rng.random(n) * 1e-4 + 1e-4
    length = np.where(rng.random(n) < 0.2, 0., rng.random(n) * 1e-2)
    kind = np.where(rng.random(n) < 0.1, NODULE, ALIVE)
    edge = g.property("edge_type")

    reference = initial.copy()                 # potential_growth: theoretical_radius, children first (post order)
    for root in ds.roots().tolist():
        for vid in post_order2(g, vids[root]):
            i = index.get(vid)
            children = g.children(vid) if i is not None else []
            if not children:
                continue
            son_section, sum_of_lateral_sections = 0., 0.
            for child in children:
                c = index[child]
                if edge.get(child) == '<':
                    son_section = reference[c] ** 2 * pi
                elif edge.get(child) == '+' and length[c] > 0. and kind[c] != NODULE:
                    sum_of_lateral_sections += reference[c] ** 2 * pi
            radius = sqrt(son_section / pi + SGC * sum_of_lateral_sections / pi)
            reference[i] = initial[i] if radius - initial[i] <= 0.001 * initial[i] else radius

    def pipe(level, radius):
        son = level.children(radius ** 2 * pi, "sum", edge="<")
        laterals = level.children(radius ** 2 * pi, "sum", edge="+", where=(length > 0.) & (kind != NODULE))
        new = np.sqrt(son / pi + SGC * laterals / pi)
        own = initial[level.nodes]
        new = np.where(new - own <= 0.001 * own, own, new)
        return np.where(level.children(radius, "count") > 0, new, radius[level.nodes])

    np.testing.assert_array_equal(ds.fold(pipe, initial), reference)


def test_death_propagates_up_with_the_minimum_time_since_death(plant):
    g, ds, vids, index, rng = plant
    n = ds.n_nodes()
    tips = ds.tips()
    state = np.full(n, ALIVE)
    state[tips[rng.random(tips.size) < 0.7]] = JUST_DEAD
    time_since_death = np.where(state == JUST_DEAD, rng.random(n) * 10., 0.)

    ref_state, ref_time = state.copy(), time_since_death.copy()
    for root in ds.roots().tolist():
        for vid in post_order2(g, vids[root]):
            if vid not in index:                 # MTG vertices that are not graph nodes
                continue
            i = index[vid]
            children = [index[c] for c in g.children(vid)]
            if not children:
                continue
            dead = [c for c in children if ref_state[c] in (JUST_DEAD, DEAD)]
            if len(dead) == len(children):
                ref_state[i] = DEAD if ref_state[i] in (JUST_DEAD, DEAD) else JUST_DEAD
                ref_time[i] = min(ref_time[c] for c in dead)

    def death(level, values):
        st, t = values[:, 0], values[:, 1]
        dead = np.isin(st, (JUST_DEAD, DEAD))
        has = level.children(st, "count") > 0
        all_dead = has & level.children(dead, "all")
        own = st[level.nodes]
        new_state = np.where(all_dead, np.where(np.isin(own, (JUST_DEAD, DEAD)), DEAD, JUST_DEAD), own)
        new_time = np.where(all_dead, level.children(t, "min", where=dead), t[level.nodes])
        return np.stack([new_state, new_time], axis=1)

    out = ds.fold(death, np.stack([state.astype(float), time_since_death], axis=1))
    np.testing.assert_array_equal(out[:, 0], ref_state)
    np.testing.assert_array_equal(out[:, 1], ref_time)
    assert (out[:, 0] != ALIVE).sum() > (state != ALIVE).sum()            # death reached some parents


def test_barrier_reopening_is_a_filtered_max_over_laterals(plant):
    g, ds, vids, index, rng = plant
    n = ds.n_nodes()
    length, radius = rng.random(n) * 1e-3, rng.random(n) * 1e-3
    edge = g.property("edge_type")
    parents = ds.parents()
    reference = np.zeros(n)                    # Root-CyNAPS: max length over '+' children longer than the radius
    for vid in vids:
        i = index[vid]
        eligible = [length[index[c]] for c in g.children(vid)
                    if edge.get(c) == '+' and length[index[c]] >= radius[i]]
        reference[i] = max(eligible) if eligible else 0.
    long_enough = length >= np.where(parents >= 0, radius[np.maximum(parents, 0)], np.inf)
    out = ds.fold(lambda level, x: level.children(length, "max", edge="+", where=long_enough, fill=0.),
                  np.zeros(n))
    np.testing.assert_array_equal(out, reference)
    assert (out > 0).any()


def test_tillers_read_the_main_stem_at_their_cohort_rank(plant):
    g, ds, vids, index, rng = plant
    n = ds.n_nodes()
    axis_of = np.repeat(np.arange(6), int(np.ceil(n / 6)))[:n].astype(float)       # 6 "axes" of metamers
    rank = np.zeros(n)
    for a in range(6):
        members = np.flatnonzero(axis_of == a)
        rank[members] = np.arange(1, members.size + 1)
    ds.register("axis_id", axis_of, location="node")
    ds.register("metamer_rank", rank, location="node")
    ds.define_chain("metamers", group="axis_id", rank="metamer_rank")
    lmax = rng.random(n)
    cohort = np.where(axis_of == 0, 1., axis_of + 2.)                                  # tiller cohort
    asked = cohort + rank - 1.
    reference = np.full(n, np.nan)             # elongwheat: tiller rank n copies main-stem rank cohort + n - 1
    main = {rank[i]: lmax[i] for i in np.flatnonzero(axis_of == 0)}
    for i in range(n):
        if asked[i] in main:
            reference[i] = main[asked[i]]
    out = ds.chain_gather(lmax, asked, chain="metamers", source=0.)
    np.testing.assert_array_equal(out, reference)
    assert np.isnan(out).any() and (~np.isnan(out)).any()


def test_a_clamped_recurrence_along_axes(plant):
    g, ds, vids, index, rng = plant
    n = ds.n_nodes()
    delta = rng.normal(0., 0.4, n)
    chains = ds.chain("axis")
    reference = np.zeros(n)                    # adel-like whorl height: a clamped, non-associative recurrence
    for c in range(chains["offsets"].size - 1):
        previous = 0.
        for i in chains["order"][chains["offsets"][c]:chains["offsets"][c + 1]]:
            previous = 0.9 * min(max(previous + delta[i], -1.), 1.)
            reference[i] = previous

    def whorl(k, nodes, previous, out):
        before = np.where(previous >= 0, out[np.maximum(previous, 0)], 0.)
        return 0.9 * np.clip(before + delta[nodes], -1., 1.)

    np.testing.assert_array_equal(ds.chain_recurrence(whorl, np.zeros(n)), reference)


def test_a_downward_fold_carries_frames_like_a_turtle(plant):
    g, ds, vids, index, rng = plant
    n = ds.n_nodes()
    bend = rng.normal(0., 0.05, n)
    edge = g.property("edge_type")
    reference = np.zeros(n)                    # heading: a branch turns by 0.8 rad, a successor bends, roots start at 0
    for root in ds.roots().tolist():
        for vid in pre_order2(g, vids[root]):
            if vid not in index:
                continue
            i = index[vid]
            parent = g.parent(vid)
            if parent is None or parent not in index:
                reference[i] = bend[i]
                continue
            turn = 0.8 if edge.get(vid) == '+' else 0.
            reference[i] = (reference[index[parent]] + turn + bend[i]) % (2 * pi)

    def heading(level, out):
        turn = np.where(level.edge == "+", 0.8, 0.)
        is_root = ds.parents()[level.nodes] < 0
        return np.where(is_root, bend[level.nodes], (level.parent(out) + turn + bend[level.nodes]) % (2 * pi))

    np.testing.assert_array_equal(ds.fold(heading, np.zeros(n), direction="down"), reference)


def test_fold_checks_its_arguments(plant):
    _, ds, *_ = plant
    with pytest.raises(ValueError, match="direction"):
        ds.fold(lambda level, x: x[level.nodes], np.zeros(ds.n_nodes()), direction="sideways")
    with pytest.raises(ValueError, match="op must be"):
        ds.fold(lambda level, x: level.children(x, "median"), np.zeros(ds.n_nodes()))
