"""
A component's own entities (DataStructureComponent.entities, data_structure.EntityView): two structural components
growing one plant, the shoot (a seed and its stem elements) and the roots (a root chain '+' on the seed), each see
their entities only: their nodes, their bases where the parent is the other's, their steps' arrays of their size, one
array per variable name, growth with parents inside or outside the view, mappings to the plants over their entities;
in segment mode and in anatomy mode (the SubOrgans of a Compartment graph), and in a Scene.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import (FunctionalComponent, NewNodes, StructuralComponent, input_variable,
                                                   state_variable)
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import EntityView, LocatedView, MPGDataStructure
from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.scene.population import Planting, planting, population_data_structure
from openalea.metafspm.scene.scene import Scene
from openalea.metafspm.solve.decorator import postsegmentation, rate, segmentation

from anatomy import add_anatomy, wiring
from growth import DOC

DT = 3600.
STEM_LENGTH, SEED_LENGTH, ROOT_LENGTH = 2., 0.5, 1.


def _complexes(g, plant, axis, organ):
    """An Axis, GrowthUnit, Phytomer and Organ under *plant*; returns the Organ."""
    s, labels = g.scales, g.labels
    parent = plant
    for scale, label in ((s.Axis, getattr(labels.Axis, axis)), (s.GrowthUnit, getattr(labels.GrowthUnit, axis)),
                         (s.Phytomer, getattr(labels.Phytomer, axis)), (s.Organ, getattr(labels.Organ, organ))):
        parent = g.add_component(parent, **PropsConfig(scale=scale, edge_type='/', label=label))
    return parent


def _length(g, vid, value):
    g.property("length")[vid] = value
    g.property("radius")[vid] = 0.1


def _build_shoot(g, plant, n_stems=2):
    """The seed, then a chain of stem elements '<' on it; returns (seed, stem elements)."""
    s, labels = g.scales, g.labels
    seed = g.add_component(_complexes(g, plant, "Shoot", "StemInternode"),
                           **PropsConfig(scale=s.SubOrgan, edge_type='/', label=labels.SubOrgan.Seed))
    _length(g, seed, SEED_LENGTH)
    stems, parent = [], seed
    for _ in range(n_stems):
        parent = g.add_child(parent, **PropsConfig(scale=s.SubOrgan, edge_type='<', label=labels.SubOrgan.StemElement))
        _length(g, parent, STEM_LENGTH)
        stems.append(parent)
    return seed, stems


def _build_roots(g, plant, n_segments=2):
    """A root chain branched '+' on the plant's seed, its last element an apex; returns its elements."""
    s, labels = g.scales, g.labels
    seed = g.find_anchor(plant, "Seed")
    organ = _complexes(g, plant, "Root", "RootInternode")
    label = labels.SubOrgan.RootApex if n_segments == 1 else labels.SubOrgan.RootSegment
    roots = [g.add_component_with_topo(organ, seed, **PropsConfig(scale=s.SubOrgan, edge_type='+', label=label))]
    for k in range(1, n_segments):
        label = labels.SubOrgan.RootApex if k == n_segments - 1 else labels.SubOrgan.RootSegment
        roots.append(g.add_child(roots[-1], **PropsConfig(scale=s.SubOrgan, edge_type='<', label=label)))
    for vid in roots:
        _length(g, vid, ROOT_LENGTH)
    return roots


@dataclass
class Shoot(StructuralComponent):
    """The seed and the stem: a stem element grows on its top tip when `grow` is set."""
    entities = {"label": ("Seed", "StemElement")}
    length: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    radius: float = state_variable(**DOC, default=0.1, scale=scales.SubOrgan, kind="descriptor")
    shoot_mass: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")
    rank: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")

    @classmethod
    def initiate_plant(cls, g, plant, scenario):
        _build_shoot(g, plant)

    @rate
    def _shoot_mass(self, length):
        self.keep("sizes", len(length), saved=False)
        return 2. * length

    @segmentation
    def _new_element(self, length) -> dict:
        ds = self.data_structure
        tips = ds.tips() if self.kept("grow", False) else np.zeros(0, dtype=np.int64)
        return {"new_elements": NewNodes(parent=ds.entity_ids("node")[tips], edge="<",
                                         label=ds.structure.labels.SubOrgan.StemElement,
                                         values={"length": 0.25})}

    @postsegmentation
    def _ranks(self):
        """MPG-style: each element's rank from its seed, on the MTG."""
        ds = self.data_structure
        g = self.mtg
        rank = g.property("rank")
        for vid in ds.entity_ids("node")[ds.order("pre")].tolist():
            rank[vid] = 0. if g.label(vid) == g.labels.SubOrgan.Seed else rank[g.parent(vid)] + 1.


@dataclass
class Roots(StructuralComponent):
    """The root chain: an apex grows on each of its tips (inside the view), and a nodal root '+' on each stem element
    of `bearers` (outside it)."""
    entities = {"label": ["RootSegment", "RootApex"]}
    length: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    root_mass: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")

    @classmethod
    def initiate_plant(cls, g, plant, scenario):
        _build_roots(g, plant)

    @rate
    def _root_mass(self, length):
        self.keep("sizes", len(length), saved=False)
        return 3. * length

    @segmentation
    def _growth(self, length) -> dict:
        ds = self.data_structure
        s, labels = ds.structure.scales, ds.structure.labels
        tips = ds.entity_ids("node")[ds.tips()] if self.kept("grow", False) else np.zeros(0, dtype=np.int64)
        bearers = np.asarray(self.kept("bearers", []), dtype=np.int64)
        new_axis = ((s.Axis, labels.Axis.Root), (s.GrowthUnit, labels.GrowthUnit.Root),
                    (s.Phytomer, labels.Phytomer.Root), (s.Organ, labels.Organ.RootInternode))
        return {"new_roots": NewNodes(parent=np.r_[tips, bearers], edge=["<"] * tips.size + ["+"] * bearers.size,
                                      label=labels.SubOrgan.RootApex, values={"length": 0.1},
                                      new_complex=[False] * tips.size + [True] * bearers.size,
                                      complex_levels=new_axis,
                                      relabel=(tips, labels.SubOrgan.RootSegment) if tips.size else None)}


@dataclass
class Carbon(FunctionalComponent):
    """A component without entities: it sees every SubOrgan."""
    length: float = input_variable(**DOC, by="Shoot", default=0., scale=scales.SubOrgan)
    sugar: float = state_variable(**DOC, default=1., scale=scales.SubOrgan, kind="intensive")

    @rate
    def _sugar(self, sugar, length):
        return sugar + 0. * length


class Plant(CompositeModel):
    initiators = (Shoot, Roots)

    def __init__(self, data_structure, time_step):
        self.shoot = Shoot(data_structure=data_structure, time_step=time_step)
        self.roots = Roots(data_structure=data_structure, time_step=time_step)
        self.carbon = Carbon(data_structure=data_structure, time_step=time_step)
        self.couple(self.shoot, self.roots, self.carbon)

    def run(self):
        self.shoot()
        self.roots()
        self.carbon()


def _population(n_plants=2):
    return population_data_structure(planting([Planting(x=0.1 * k) for k in range(n_plants)], Plant, {}), Plant,
                                     time_step=DT)


def _labels(view):
    return view.labels().tolist()


def _parent_ids(view):
    """{node id: its parent's id in the view, None at a base}."""
    ids, parents = view.entity_ids("node"), view.parents()
    return {int(v): (int(ids[p]) if p >= 0 else None) for v, p in zip(ids.tolist(), parents.tolist())}


def _expected_parent_ids(view):
    """{node id: its MPG parent when it is in the view, else None}."""
    g, ids = view.structure, view.entity_ids("node").tolist()
    return {v: (g.parent(v) if g.parent(v) in set(ids) else None) for v in ids}


def _per_plant(view, values, reduce):
    owner = view.owner("Plant")
    return [reduce(np.asarray(values)[owner == k]) for k in range(len(view.entity_ids("Plant")))]


# ── segment mode ──

def test_each_component_sees_its_entities_and_its_bases():
    population = _population()
    model, ds = population.instance, population.data_structure
    g = ds.structure
    shoot, roots = model.shoot.data_structure, model.roots.data_structure
    assert isinstance(shoot, EntityView) and isinstance(roots, EntityView) and model.carbon.data_structure is ds
    assert set(_labels(shoot)) == {"Seed", "StemElement"} and set(_labels(roots)) == {"RootSegment", "RootApex"}
    assert shoot.n_nodes() + roots.n_nodes() == ds.n_nodes() == 10
    # in the base's order
    assert shoot.entity_ids("node").tolist() == [v for v in ds.entity_ids("node").tolist()
                                                 if ds.labels()[ds.index_of(v)] in ("Seed", "StemElement")]
    # each plant: a seed and its two stem elements; a root chain whose base is on the seed (outside the view)
    for view in (shoot, roots):
        assert _parent_ids(view) == _expected_parent_ids(view)
    assert [_labels(shoot)[k] for k in shoot.roots()] == ["Seed"] * 2
    bases = roots.entity_ids("node")[roots.roots()]
    assert [_labels(roots)[k] for k in roots.roots()] == ["RootSegment"] * 2
    assert {g.label(g.parent(int(v))) for v in bases} == {g.labels.SubOrgan.Seed}
    assert [_labels(roots)[k] for k in roots.tips()] == ["RootApex"] * 2
    assert roots.n_edges() == 2 and sorted(roots.edges()) == sorted(
        (parent, child) for child, parent in _parent_ids(roots).items() if parent is not None)
    position = np.argsort(roots.order("post"))
    assert all(position[k] < position[p] for k, p in enumerate(roots.parents()) if p >= 0)
    assert [sorted(_labels(roots)[k] for k in level) for level in roots.levels()] == [["RootSegment"] * 2,
                                                                                    ["RootApex"] * 2]
    is_segment = np.asarray(_labels(roots)) == "RootSegment"
    np.testing.assert_allclose(roots.accumulate(np.ones(4)), np.where(is_segment, 2., 1.))
    np.testing.assert_allclose(roots.chain_scan(np.ones(4), reverse=True), np.where(is_segment, 2., 1.))
    graph = roots.to_graph_view()
    assert graph.n_nodes == 4 and graph.n_edges == 2
    owner = roots.owner("Plant")
    assert sorted(owner.tolist()) == [0, 0, 1, 1] and all(owner[k] == owner[p] for k, p in
                                                          enumerate(roots.parents()) if p >= 0)


def test_steps_work_on_arrays_of_their_entities_and_one_variable_per_name():
    population = _population()
    model, ds = population.instance, population.data_structure
    model.run()
    assert model.shoot.kept("sizes") == 6 and model.roots.kept("sizes") == 4
    # length: one array of the base, each component's MTG values on its entities
    shoot, roots = model.shoot.data_structure, model.roots.data_structure
    assert ds.location("length") == "node" and ds.get("length").size == 10
    shoot_lengths = np.where(np.asarray(_labels(shoot)) == "Seed", SEED_LENGTH, STEM_LENGTH)
    np.testing.assert_allclose(shoot.get("length"), shoot_lengths)
    np.testing.assert_allclose(roots.get("length"), ROOT_LENGTH)
    # a component's own variable: the default on the other entities
    np.testing.assert_allclose(ds.get("shoot_mass")[np.isin(ds.labels(), ["RootSegment", "RootApex"])], 0.)
    np.testing.assert_allclose(shoot.get("shoot_mass"), 2. * shoot_lengths)
    # an MPG-style step: the MTG's values read back at the view's entities
    np.testing.assert_allclose(sorted(shoot.get("rank")), [0., 0., 1., 1., 2., 2.])
    np.testing.assert_allclose(ds.get("rank")[ds.index_of(roots.entity_ids("node"))], 0.)
    # the MTG write-back: each component's states at its entities
    root_mass = ds.mtg.property("root_mass")
    assert [root_mass[v] for v in roots.entity_ids("node").tolist()] == pytest.approx(3. * roots.get("length"))
    # a view's values are a copy; set() writes into the base's array
    copy = roots.get("length")
    copy[:] = 7.
    np.testing.assert_allclose(roots.get("length"), ROOT_LENGTH)
    roots.set("length", 2.)
    np.testing.assert_allclose(ds.get("length")[ds.index_of(roots.entity_ids("node"))], 2.)
    np.testing.assert_allclose(shoot.get("length"), shoot_lengths)


def test_mappings_to_the_plants_are_over_the_view_s_entities():
    population = _population()
    model = population.instance
    shoot, roots = model.shoot.data_structure, model.roots.data_structure
    np.testing.assert_allclose(shoot.map_to(shoot.get("length"), "node", "Plant", "sum"),
                               [SEED_LENGTH + 2 * STEM_LENGTH] * 2)
    np.testing.assert_allclose(roots.map_to(roots.get("length"), "node", "Plant", "sum"), [2 * ROOT_LENGTH] * 2)
    np.testing.assert_allclose(shoot.map_to(shoot.get("length"), "node", "Plant", "max"), [STEM_LENGTH] * 2)
    values = np.array([1., 2., 3., 4.])
    np.testing.assert_allclose(roots.map_to(values, "node", "Plant", "mean"), _per_plant(roots, values, np.mean))
    owner = roots.owner("Plant")
    weights = np.array([1., 3., 1., 1.])
    totals = np.array([4., 6.])
    np.testing.assert_allclose(roots.map_to(totals, "Plant", "node", "split", weights=weights),
                               totals[owner] * weights / np.bincount(owner, weights=weights)[owner])
    np.testing.assert_allclose(roots.map_to(totals, "Plant", "node", "broadcast"), totals[owner])
    assert roots.labels(location="Organ").tolist() == ["RootInternode"] * 4


def test_growth_with_parents_inside_and_outside_the_view():
    population = _population()
    model, ds = population.instance, population.data_structure
    g = ds.structure
    shoot, roots = model.shoot.data_structure, model.roots.data_structure
    stems = shoot.entity_ids("node")[np.asarray(_labels(shoot)) == "StemElement"]
    tips = set(roots.entity_ids("node")[roots.tips()].tolist())
    before = set(roots.entity_ids("node").tolist())
    model.roots.keep("bearers", [int(stems[0])])         # a nodal root on a stem element of the first plant
    model.roots.keep("grow", True)
    model.shoot.keep("grow", True)
    model.run()
    assert shoot.n_nodes() == 8 and roots.n_nodes() == 4 + 2 + 1 and ds.n_nodes() == 15
    assert set(_labels(shoot)) == {"Seed", "StemElement"}
    new = sorted(set(roots.entity_ids("node").tolist()) - before)
    parents = _parent_ids(roots)
    assert parents == _expected_parent_ids(roots)
    # the nodal root: a base of the roots' view, its parent the stem element (outside it), in an axis of its own
    (nodal,) = [v for v in new if g.parent(v) == stems[0]]
    assert parents[nodal] is None and g.edge_type(nodal) == '+' and g.label(nodal) == g.labels.SubOrgan.RootApex
    assert g.label(g.complex(nodal)) == g.labels.Organ.RootInternode
    # the apices grown on the roots' tips (inside the view): the former tips relabelled segments
    assert {parents[v] for v in new if v != nodal} == tips
    assert all(g.label(v) == g.labels.SubOrgan.RootSegment for v in tips)
    assert _labels(roots).count("RootApex") == 3
    np.testing.assert_allclose(roots.get("length")[roots.index_of(new)], 0.1)
    np.testing.assert_allclose(shoot.get("length")[shoot.tips()], 0.25)
    np.testing.assert_allclose(roots.map_to(roots.get("length"), "node", "Plant", "sum"),
                               _per_plant(roots, roots.get("length"), np.sum))
    assert sorted(roots.map_to(roots.get("length"), "node", "Plant", "sum").tolist()) == pytest.approx(
        [2 * ROOT_LENGTH + 0.1, 2 * ROOT_LENGTH + 0.2])
    assert model.shoot.kept("sizes") == 8 and model.roots.kept("sizes") == 7     # the rates, after the growth rows
    assert ds.get("root_mass").size == 15
    np.testing.assert_allclose(roots.get("root_mass"), 3. * roots.get("length"))


def test_new_nodes_outside_the_component_s_entities_raise():
    population = _population(1)
    shoot = population.instance.shoot
    labels = shoot.data_structure.structure.labels
    with pytest.raises(ValueError, match="not among Shoot's entities"):
        shoot._create_nodes(NewNodes(parent=shoot.data_structure.entity_ids("node")[-1:], edge="<",
                                     label=labels.SubOrgan.LeafElement))


def test_a_component_without_entities_is_unchanged_and_views_are_shared():
    population = _population(1)
    ds = population.data_structure
    assert ds.entity_view({"label": ["RootApex", "RootSegment"]}) is ds.entity_view(
        {"label": ["RootApex", "RootSegment"]})
    assert population.instance.carbon.data_structure is ds
    assert ds.entity_view(lambda base: np.asarray(base.labels()) == "Seed").n_nodes() == 1
    with pytest.raises(TypeError, match="non-empty"):
        ds.entity_view({})


def test_the_base_s_triangles_are_each_component_s_on_its_entities():
    """Both components draw (default cylinders here): the plant's triangles are theirs together, in the base's node
    indices."""
    population = _population(1)
    ds = population.data_structure
    for name, value in (("x1", 0.), ("y1", 0.), ("z1", 0.), ("x2", 0.), ("y2", 0.), ("z2", -1.), ("radius", 0.1)):
        if ds.has(name):
            ds.set(name, value)
        else:
            ds.register(name, value)
    triangles = ds.triangles()
    assert sorted(set(triangles.owner.tolist())) == list(range(ds.n_nodes()))
    roots = population.instance.roots.data_structure
    assert sorted(set(roots.triangles().owner.tolist())) == list(range(roots.n_nodes()))


def test_both_run_in_a_scene():
    scene = Scene(planting([Planting(x=0.1), Planting(x=0.3)], Plant, {}), time_step=DT)
    model = scene.populations[0].instance
    model.roots.keep("grow", True)
    scene.simulate(2)
    roots, ds = model.roots.data_structure, scene.populations[0].data_structure
    assert roots.n_nodes() == 4 + 2 * 2 and _labels(roots).count("RootApex") == 2
    assert model.shoot.data_structure.n_nodes() == 6 and ds.n_nodes() == 14
    np.testing.assert_allclose(roots.get("root_mass"), 3. * roots.get("length"))
    np.testing.assert_allclose(ds.get("root_mass")[np.isin(ds.labels(), ["Seed", "StemElement"])], 0.)


# ── anatomy mode: the SubOrgans of a Compartment graph ──

def _anatomy_plant():
    """One plant: the seed and two stem elements without anatomies, a root chain '+' on the seed whose segments have
    one (test.helpers.anatomy)."""
    g = MPG()
    plant = g.add_component(g.root, **PropsConfig(scale=g.scales.Plant, edge_type='/'))
    _build_shoot(g, plant)
    roots = _build_roots(g, plant, n_segments=3)
    for vid in roots[:-1]:
        add_anatomy(g, vid)
    g.convert_properties_to_arraydict()
    return g, MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=wiring(g)), roots


def test_in_anatomy_mode_a_component_sees_its_suborgans():
    g, ds, root_vids = _anatomy_plant()
    shoot, roots = Shoot(data_structure=ds, time_step=DT), Roots(data_structure=ds, time_step=DT)
    assert isinstance(roots.data_structure, EntityView) and isinstance(roots.data_structure._inner, LocatedView)
    assert roots.data_structure.entity_ids("node").tolist() == root_vids
    assert roots.data_structure.parents().tolist() == [-1, 0, 1]
    assert shoot.data_structure.parents().tolist() == [-1, 0, 1]
    assert ds.location("length") == "SubOrgan" and ds.get("length").size == 6
    shoot(), roots()
    assert shoot.kept("sizes") == 3 and roots.kept("sizes") == 3
    np.testing.assert_allclose(roots.data_structure.map_to(roots.data_structure.get("length"), "node", "Plant", "sum"),
                               [3 * ROOT_LENGTH])
    np.testing.assert_allclose(ds.get("root_mass")[np.isin(ds.entity_ids("SubOrgan"), root_vids)], 3. * ROOT_LENGTH)
    # growth: an apex on the roots' tip (inside), a nodal root on a stem element (outside)
    stem = int(shoot.data_structure.entity_ids("node")[1])
    roots.keep("grow", True)
    roots.keep("bearers", [stem])
    roots()
    assert roots.data_structure.n_nodes() == 5 and shoot.data_structure.n_nodes() == 3
    assert roots.data_structure.parents().tolist() == [-1, 0, 1, 2, -1]
    assert g.parent(int(roots.data_structure.entity_ids("node")[-1])) == stem
    np.testing.assert_allclose(roots.data_structure.get("length")[-2:], 0.1)
    assert ds.get("length").size == 8


def test_in_anatomy_mode_a_compartment_component_cannot_declare_entities():
    _, ds, _ = _anatomy_plant()

    @dataclass
    class Cells(FunctionalComponent):
        entities = {"label": "RootSegment"}
        water: float = state_variable(**DOC, default=0., location="node")

    with pytest.raises(NotImplementedError, match="SubOrgan scale or coarser"):
        Cells(data_structure=ds, time_step=DT)
