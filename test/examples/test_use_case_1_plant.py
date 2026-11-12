"""
The plant of examples/use_case_1 (component.WheatCrown, NodalRootEmergence), with Adel's shoot and ArchiSimple's roots
on one MPG: the seed at the sowing depth and the subcrown internode up to the crown built first, Adel's main stem '<'
on the crown's top (shoot_anchor), ArchiSimple's six seminal roots '+' on the seed (root_anchor), and the nodal roots
the rule asks for grown on the right phytomers at the right thermal times, the counts cumulative; each model sees its
own entities only. The plant model here is the test's (the use case's models.py assembles the whole plant).
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("openalea.adel")
pytest.importorskip("openalea.archisimple")

from openalea.adel.development import DAY, ScenarioShoot                              # noqa: E402
from openalea.archisimple.architecture import RootArchitecture                        # noqa: E402
from openalea.archisimple.parameter_sets import WHEAT_ON_A_SHOOT                      # noqa: E402
from openalea.metafspm import CompositeModel, Planting, planting, Scene
from openalea.metafspm.data_structure.data_api import EntityView                      # noqa: E402

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "use_case_1")
sys.path.insert(0, os.path.abspath(EXAMPLE))

from component import NodalRootEmergence, WheatCrown                                   # noqa: E402

CROWN_DEPTH, SUBCROWN_ELEMENTS = 0.02, 3
SOWING_DEPTHS = (0.04, 0.05)            # m: the first plant's from its planting, the second's from its scenario
PER_LEAF, OFFSET, DELAY = 2, 1, 20.     # the rule: two nodal roots per leaf, one rank below it, 20 °Cd later
N_DAYS = 16                             # at 15 °C: about three leaves on the main stem
SHOOT_ANCHOR = {"label": "StemElement", "last": True}
ROOT_LABELS = ("RootSegment", "RootApex")


class WheatPlant(CompositeModel):
    """The crown, Adel's shoot (ScenarioShoot, nested: its development and architecture grown together), the nodal
    root rule, then ArchiSimple's roots: models.WheatPlant's first parts, as it steps them."""
    initiators = (WheatCrown, ScenarioShoot, RootArchitecture)
    parts = {"crown": WheatCrown, "shoot": ScenarioShoot, "rule": NodalRootEmergence, "roots": RootArchitecture}


class WheatShoot(CompositeModel):
    """The crown, Adel's shoot and the rule, without roots (the rule's requests alone)."""
    initiators = (WheatCrown, ScenarioShoot)
    parts = {"crown": WheatCrown, "shoot": ScenarioShoot, "rule": NodalRootEmergence}


def _scenario(roots=True, rule=True, **values):
    """A plant's scenario: the crown's, Adel's (two sectors per blade), the rule's (rule=False: its defaults) and
    ArchiSimple's wheat on a shoot (roots=False: none), *values* last."""
    scenario = dict(shoot_anchor=SHOOT_ANCHOR, n_sect=2, crown_depth=CROWN_DEPTH, subcrown_elements=SUBCROWN_ELEMENTS)
    if rule:
        scenario.update(nodal_roots_per_leaf=PER_LEAF, nodal_root_rank_offset=OFFSET, nodal_root_delay=DELAY)
    if roots:
        scenario.update(WHEAT_ON_A_SHOOT, root_anchor={"label": "Seed"})
    return dict(scenario, **values)


def _scene(model=WheatPlant, time_step=DAY, scenarios=None, sowing_depths=SOWING_DEPTHS):
    plants = [Planting(0.05, 0.05, sowing_depths[0], 0.), Planting(0.15, 0.05, sowing_depths[1], 90.)]
    scenarios = scenarios or [_scenario(), _scenario()]
    return Scene(planting(plants[:len(scenarios)], model, plant_scenarios=scenarios), time_step=time_step,
                 forcings={"air_temperature": 15.})


def _vertices(mtg, label, scale="SubOrgan"):
    code = getattr(getattr(mtg.labels, scale), label)
    return sorted(v for v in mtg.vertices(scale=getattr(mtg.scales, scale)) if mtg.label(v) == code)


def _nodal_roots(mtg, bearer):
    """The root axes '+' on *bearer* (their first elements)."""
    roots = {getattr(mtg.labels.SubOrgan, label) for label in ROOT_LABELS}
    return sorted(v for v in mtg.children(bearer) if mtg.edge_type(v) == '+' and mtg.label(v) in roots)


def _phytomers(data_structure):
    """{(plant index, axis row, rank): phytomer id} of the shoot's phytomers."""
    ids = np.asarray(data_structure.entity_ids("Phytomer"))
    axis = np.asarray(data_structure.owner("Axis", of="Phytomer"))
    row = np.asarray(data_structure.get("axis_row"))[axis]
    plant = np.asarray(data_structure.owner("Plant", of="Phytomer"))
    rank = np.asarray(data_structure.get("phytomer_rank"))
    return {(int(plant[k]), int(row[k]), int(rank[k])): int(ids[k]) for k in np.flatnonzero(row >= 0)}


def _record(scene):
    """After each step: the plants' thermal time, and per shoot phytomer its leaf's appearance, the nodal roots asked
    for on it and those grown on its bearer."""
    data_structure = scene.populations[0].data_structure
    mtg = data_structure.structure
    phytomers = _phytomers(data_structure)
    index = data_structure.index_of(list(phytomers.values()), location="Phytomer")
    bearers = data_structure.bearer("Phytomer", list(phytomers.values()))
    values = {name: np.asarray(data_structure.get(name))[index] for name in ("leaf_appearance_time", "nodal_axes_requested",
                                                                   "nodal_axes_emerged")}
    return {"thermal_time": np.asarray(data_structure.get("thermal_time")).copy(),
            "phytomers": {key: dict(id=vid, bearer=int(bearer), roots=_nodal_roots(mtg, int(bearer)),
                                    **{name: float(values[name][k]) for name in values})
                          for k, ((key, vid), bearer) in enumerate(zip(phytomers.items(), bearers))}}


@pytest.fixture(scope="module")
def grown():
    scene = _scene()
    history = [_record(scene)]                                      # before the first step: as initiated
    for _ in range(N_DAYS):
        scene.run()
        history.append(_record(scene))
    return scene, history


def _crown(mtg, plant):
    """(the seed, the subcrown internode's StemElements from the seed up) of *plant*."""
    (seed,) = [v for v in _vertices(mtg, "Seed") if mtg.complex_at_scale(v, mtg.scales.Plant) == plant]
    elements = []
    while True:
        following = [v for v in mtg.children(seed if not elements else elements[-1])
                     if mtg.edge_type(v) == '<' and mtg.label(v) == mtg.labels.SubOrgan.StemElement
                     and mtg.complex_at_scale(v, mtg.scales.Organ) == mtg.complex_at_scale(seed, mtg.scales.Organ)]
        if not following:
            return seed, elements
        elements.append(following[0])


def test_the_seed_and_the_subcrown_internode(grown):
    scene, _ = grown
    data_structure = scene.populations[0].data_structure
    mtg = data_structure.structure
    plants = data_structure.entity_ids("Plant").tolist()
    for plant, sowing_depth in zip(plants, SOWING_DEPTHS):
        seed, elements = _crown(mtg, plant)
        assert len(elements) == SUBCROWN_ELEMENTS
        organ = mtg.complex_at_scale(seed, mtg.scales.Organ)
        assert mtg.label(organ) == mtg.labels.Organ.StemInternode and mtg.edge_type(elements[0]) == '<'
        axis = mtg.complex_at_scale(seed, mtg.scales.Axis)
        assert mtg.label(axis) == mtg.labels.Axis.Shoot
        assert np.asarray(data_structure.get("axis_row"))[data_structure.index_of([axis], location="Axis")][0] == -1   # not Adel's axis
        index = data_structure.index_of([seed] + elements)
        z1, z2 = (np.asarray(data_structure.get(name))[index] for name in ("z1", "z2"))
        np.testing.assert_allclose([z1[0], z2[0]], -sowing_depth, rtol=0., atol=1e-15)        # the seed: a point
        rise = (sowing_depth - CROWN_DEPTH) / SUBCROWN_ELEMENTS
        np.testing.assert_allclose(z1[1:], -sowing_depth + rise * np.arange(SUBCROWN_ELEMENTS), atol=1e-15)
        np.testing.assert_allclose(z2[1:], -sowing_depth + rise * np.arange(1, SUBCROWN_ELEMENTS + 1), atol=1e-15)
        np.testing.assert_allclose(np.asarray(data_structure.get("length"))[index],
                                   [WheatCrown.seed_length] + [rise] * SUBCROWN_ELEMENTS, atol=1e-15)
        np.testing.assert_allclose(np.asarray(data_structure.get("radius"))[index], [WheatCrown.seed_radius]
                                   + [WheatCrown.subcrown_radius] * SUBCROWN_ELEMENTS)
        x = mtg.property("x")[plant]
        np.testing.assert_allclose(np.asarray(data_structure.get("x1"))[index], x)
        np.testing.assert_allclose(np.asarray(data_structure.get("x2"))[index], x)
    np.testing.assert_allclose(data_structure.get("crown_depth"), CROWN_DEPTH)          # its parameters, per plant


def test_the_main_stem_on_the_crown_and_the_seminal_roots_on_the_seed(grown):
    scene, _ = grown
    data_structure = scene.populations[0].data_structure
    mtg = data_structure.structure
    roots = {getattr(mtg.labels.SubOrgan, label) for label in ROOT_LABELS}
    for plant in data_structure.entity_ids("Plant").tolist():
        seed, elements = _crown(mtg, plant)
        base = mtg.property("shoot_base")[plant]
        assert mtg.parent(base) == elements[-1] and mtg.edge_type(base) == '<'
        assert mtg.label(base) == mtg.labels.SubOrgan.HiddenElement                  # Adel's first internode's
        np.testing.assert_allclose(np.asarray(data_structure.get("z1"))[data_structure.index_of(base)], -CROWN_DEPTH, atol=1e-15)
        seminal = [v for v in mtg.children(seed) if mtg.edge_type(v) == '+']
        assert len(seminal) == 6 and all(mtg.label(v) in roots for v in seminal)
        start = [np.asarray(data_structure.get(name))[data_structure.index_of(seminal)] for name in ("x1", "y1", "z1")]
        end = [np.asarray(data_structure.get(name))[data_structure.index_of(seed)] for name in ("x2", "y2", "z2")]
        np.testing.assert_allclose(np.column_stack(start), np.tile(end, (6, 1)), atol=1e-15)   # from the seed
        assert all(mtg.complex_at_scale(v, mtg.scales.Axis) != mtg.complex_at_scale(seed, mtg.scales.Axis) for v in seminal)


def test_each_model_sees_its_own_entities(grown):
    scene, _ = grown
    instance = scene.populations[0].instance
    data_structure = scene.populations[0].data_structure
    shoot, rule, roots = (component.data_structure for component in (instance.shoot.architecture, instance.rule,
                                                                      instance.roots))
    assert all(isinstance(view, EntityView) for view in (shoot, rule, roots))
    assert set(np.asarray(roots.labels()).tolist()) == set(ROOT_LABELS)
    assert set(np.asarray(shoot.labels()).tolist()) == {"HiddenElement", "StemElement", "LeafElement"}
    assert np.array_equal(rule.entity_ids("node"), shoot.entity_ids("node"))      # the rule's: Adel's elements
    shoot_ids, root_ids = set(shoot.entity_ids("node").tolist()), set(roots.entity_ids("node").tolist())
    assert not shoot_ids & root_ids
    crown = {v for plant in data_structure.entity_ids("Plant").tolist()
             for v in [_crown(data_structure.structure, plant)[0], *_crown(data_structure.structure, plant)[1]]}
    assert shoot_ids | root_ids | crown == set(data_structure.entity_ids("node").tolist())   # everything is someone's
    assert not crown & (shoot_ids | root_ids)
    assert isinstance(instance.crown.data_structure, EntityView)                  # the crown's: its elements
    assert set(instance.crown.data_structure.entity_ids("node").tolist()) == crown
    assert (np.asarray(shoot.parents()) == -1).sum() == 2                          # each main stem on its crown
    assert (np.asarray(roots.parents()) == -1).sum() == 2 * 6 + sum(len(record["roots"]) for record
                                                                   in _record(scene)["phytomers"].values())


def test_the_leaves_appearance_times(grown):
    """The main stems' leaves appear one phyllochron apart from the first (Adel's scenario: its thermal time starts
    at the main stem's first leaf), interpolated within the steps; three leaves on each main stem."""
    scene, history = grown
    data_structure = scene.populations[0].data_structure
    phyllochron = float(np.asarray(data_structure.get("phyllochron"))[0])
    first = history[1]["thermal_time"]                                  # the first step's: the first leaf
    main = history[-1]["phytomers"]
    for plant in range(2):
        ranks = sorted(rank for (p, row, rank) in main if p == plant and row == 0)
        assert ranks[:3] == [1, 2, 3]
        appeared = [rank for rank in ranks if np.isfinite(main[(plant, 0, rank)]["leaf_appearance_time"])]
        assert len(appeared) == 3
        np.testing.assert_allclose([main[(plant, 0, rank)]["leaf_appearance_time"] for rank in appeared],
                                   first[plant] + phyllochron * np.arange(3), rtol=0., atol=1e-9)
    crown_and_roots = np.asarray(data_structure.get("axis_row"))[data_structure.owner("Axis", of="Phytomer")] < 0
    assert np.isnan(np.asarray(data_structure.get("leaf_appearance_time"))[crown_and_roots]).all()


def test_nodal_roots_on_the_right_phytomers_at_the_right_times(grown):
    """Each leaf's nodal roots are asked for on its axis' phytomer OFFSET ranks below, at the first step DELAY °Cd
    after its appearance, and ArchiSimple grows them there in the same step, each its own axis."""
    scene, history = grown
    data_structure = scene.populations[0].data_structure
    mtg = data_structure.structure
    final = history[-1]["phytomers"]
    asked = 0
    for plant, row, rank in final:
        leaf = final.get((plant, row, rank + OFFSET))
        due = None if leaf is None or np.isnan(leaf["leaf_appearance_time"]) else \
            leaf["leaf_appearance_time"] + DELAY
        for before, after in zip(history[:-1], history[1:]):
            expected = PER_LEAF if due is not None and after["thermal_time"][plant] >= due else 0
            now = after["phytomers"].get((plant, row, rank))
            if now is None:
                continue
            assert now["nodal_axes_requested"] == expected, (plant, row, rank)
            assert len(now["roots"]) == expected and now["nodal_axes_emerged"] == expected
            if expected and (plant, row, rank) in before["phytomers"]:
                assert set(before["phytomers"][(plant, row, rank)]["roots"]) <= set(now["roots"])   # kept
        asked += final[(plant, row, rank)]["nodal_axes_requested"]
    assert asked >= 2 * 2 * PER_LEAF                     # leaves 2 and 3 of each main stem at least, on ranks 1 and 2
    for plant in range(2):
        on_crown = final[(plant, 0, 1)]
        assert len(on_crown["roots"]) == PER_LEAF
        bearer = on_crown["bearer"]                  # the element at its node: its internode's top, on the crown
        assert mtg.label(bearer) == mtg.labels.SubOrgan.StemElement
        assert mtg.label(mtg.complex_at_scale(bearer, mtg.scales.Organ)) == mtg.labels.Organ.StemInternode
        assert mtg.parent(mtg.parent(bearer)) == _crown(mtg, int(data_structure.entity_ids("Plant")[plant]))[1][-1]
        axes = {mtg.complex_at_scale(v, mtg.scales.Axis) for v in on_crown["roots"]}
        assert len(axes) == PER_LEAF and all(mtg.label(a) == mtg.labels.Axis.Root for a in axes)
        np.testing.assert_allclose(np.asarray(data_structure.get("z1"))[data_structure.index_of(on_crown["roots"])], -CROWN_DEPTH,
                                   atol=1e-12)               # at the crown's depth (phytomer 1's node)


def test_the_nodal_roots_start_on_the_crown(grown):
    """Every nodal root starts at its phytomer's node (data_structure.position("Phytomer", at="end"), from the elements'
    topological ends): on the crown, the internodes below not elongated (the tillering zone), the crown's x and y;
    Adel draws the first element of a phytomer above the first on the coleoptile's top, out of the soil (its display
    ends), where the nodal roots would otherwise start. The plant's ends are topological: each element's base at its
    parent's end (data_structure.check_connectivity), a nodal root's at its bearer's end, its node."""
    scene, history = grown
    data_structure = scene.populations[0].data_structure
    mtg = data_structure.structure
    final = history[-1]["phytomers"]
    plants = data_structure.entity_ids("Plant").tolist()
    nodes = data_structure.position("Phytomer", at="end")
    grown_on = 0
    for (plant, row, rank), record in final.items():
        if not record["roots"]:
            continue
        grown_on += 1
        node = nodes[data_structure.index_of([record["id"]], location="Phytomer")][0]
        np.testing.assert_allclose(node, [mtg.property("x")[plants[plant]], mtg.property("y")[plants[plant]],
                                          -CROWN_DEPTH], atol=1e-12)
        index = data_structure.index_of(record["roots"])
        start = np.column_stack([np.asarray(data_structure.get(name))[index] for name in ("x1", "y1", "z1")])
        np.testing.assert_allclose(start, np.tile(node, (len(record["roots"]), 1)), atol=1e-12)
        if rank >= 2:                                                   # its first element drawn out of the soil
            first = min(int(v) for v in mtg.components_at_scale(record["id"], scale=mtg.scales.SubOrgan))
            assert np.asarray(data_structure.get("display_z2"))[data_structure.index_of(first)] > 0.
    assert grown_on >= 4                                                # ranks 1 and 2 of each main stem
    assert data_structure.check_connectivity(tolerance=1e-12).empty


def test_the_plant_is_drawn_connected(grown):
    """The plant's triangles (those of its SubOrgans, the frames' and the light's): the crown's elements too (the
    seed as the grain, the subcrown internode's cylinders from the seed up to the crown), so that the seminal roots,
    on the seed, and the shoot, on the crown, are drawn joined."""
    scene, _ = grown
    data_structure = scene.populations[0].data_structure
    mtg = data_structure.structure
    elements = data_structure.at("SubOrgan") if data_structure._anatomy else data_structure                # the plant model's here: its nodes
    triangles = elements.triangles()
    owner, vertices = np.asarray(triangles.owner), np.asarray(triangles.vertices)
    drawn = set(np.asarray(elements.entity_ids("node"))[np.unique(owner)].tolist())
    for plant, sowing_depth in zip(data_structure.entity_ids("Plant").tolist(), SOWING_DEPTHS):
        seed, subcrown = _crown(mtg, plant)
        assert {seed, *subcrown} <= drawn
        heights = vertices[np.isin(owner, elements.index_of(subcrown))][..., 2]
        np.testing.assert_allclose([heights.min(), heights.max()], [-sowing_depth, -CROWN_DEPTH], atol=1e-12)
        grain = vertices[owner == elements.index_of(seed)].reshape(-1, 3)
        np.testing.assert_allclose(np.ptp(grain[:, 0]), WheatCrown.seed_length, rtol=1e-12)    # along x
        assert grain[:, 2].min() < -sowing_depth < grain[:, 2].max()


def test_requests_are_cumulative_and_none_is_lost():
    """The same plant, the rule alone on Adel's shoot, at a step of a day and of six hours: the main stems' leaves
    appear at the same thermal times, and at the same final thermal time each phytomer has been asked for the same
    nodal roots, whatever the steps (a request counted once, when it falls due)."""
    scenarios = [_scenario(roots=False, nodal_root_rank_offset=0, nodal_root_delay=40.)]

    def grown(time_step):
        scene = _scene(WheatShoot, time_step, scenarios)
        data_structure = scene.populations[0].data_structure
        requested = []
        steps_per_day = int(round(DAY / time_step))
        for _ in range((N_DAYS - 1) * steps_per_day + 1):       # the same final thermal time (the first call adds none)
            scene.run()
            requested.append(dict(zip(_phytomers(data_structure).values(), np.asarray(data_structure.get("nodal_axes_requested"))[
                data_structure.index_of(list(_phytomers(data_structure).values()), location="Phytomer")])))
        return _phytomers(data_structure), data_structure, requested

    daily, daily_ds, daily_requested = grown(DAY)
    fine, fine_ds, fine_requested = grown(DAY / 4)
    np.testing.assert_allclose(daily_ds.get("thermal_time"), fine_ds.get("thermal_time"), rtol=0., atol=1e-9)
    assert set(daily) == set(fine)
    main = [key for key in daily if key[1] == 0]
    for name in ("leaf_appearance_time", "nodal_axes_requested"):
        a = np.asarray(daily_ds.get(name))[daily_ds.index_of([daily[key] for key in main], location="Phytomer")]
        b = np.asarray(fine_ds.get(name))[fine_ds.index_of([fine[key] for key in main], location="Phytomer")]
        np.testing.assert_allclose(a, b, rtol=0., atol=1e-9, err_msg=name)
    for steps in (daily_requested, fine_requested):
        for before, after in zip(steps[:-1], steps[1:]):
            assert all(after[vid] >= count for vid, count in before.items())       # never decreasing
    assert max(daily_requested[-1].values()) == PER_LEAF                          # once per leaf


def test_a_crown_below_the_seed_raises():
    with pytest.raises(ValueError, match="above the seed"):
        _scene(WheatShoot, scenarios=[_scenario(roots=False)], sowing_depths=(0.015, 0.015))


def test_the_rule_s_defaults_wait_for_the_fourth_leaf():
    """With its defaults (Klepper et al. 1984: the roots of node n with leaf n + 3), three leaves ask for nothing."""
    scenarios = [_scenario(roots=False, rule=False)]
    scene = _scene(WheatShoot, DAY, scenarios)
    for _ in range(N_DAYS):
        scene.run()
    data_structure = scene.populations[0].data_structure
    assert np.isfinite(np.asarray(data_structure.get("leaf_appearance_time"))).sum() >= 3
    assert not np.asarray(data_structure.get("nodal_axes_requested")).any()
    assert NodalRootEmergence.nodal_root_rank_offset == 3. and NodalRootEmergence.nodal_roots_per_leaf == 2.
