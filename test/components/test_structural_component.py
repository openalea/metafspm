"""
StructuralComponent: MPG-style and array-style steps,
the synchronisation of the DataStructure around MPG-style steps, and topology updates only when the MPG changed.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher

from openalea.metafspm.coupling.component import state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales

from growth import DOC, SEGMENT_LENGTH, CarbonProbe, RootGrowthProbe, make_chain



@pytest.fixture
def plant():
    g, ds, vids = make_chain()
    growth = RootGrowthProbe(data_structure=ds)
    carbon = CarbonProbe(data_structure=ds)
    return g, ds, vids, growth, carbon


def _by_vid(ds, name):
    return dict(zip(ds.entity_ids("node").tolist(), ds.get(name).tolist()))


def test_structural_steps_run_in_the_growth_rows():
    Choregrapher().reset()
    _, ds, _ = make_chain()
    model = RootGrowthProbe(data_structure=ds)
    rows = [[f.func.name for f in group] for group in model.__dict__["_choregraphy"][1].values()]
    assert rows == [["potential_growth"], ["actual_growth", "radius"], ["segmentation"], ["distance_from_tip"]]


def test_the_mtg_is_the_data_structure_mtg(plant):
    g, _, _, growth, _ = plant
    assert growth.mtg is g


def test_inputs_are_flushed_to_the_mtg_before_an_mpg_style_step(plant):
    _, ds, vids, growth, _ = plant
    ds.set("C_hexose_root", 0.25)            # e.g. written by the carbon model, not yet on the MTG
    growth()
    assert growth.read_hexose == {vids[-1]: 0.25}


def test_structural_outputs_are_re_read_after_mpg_style_steps(plant):
    g, ds, vids, growth, _ = plant
    growth()
    apex = vids[-1]
    assert _by_vid(ds, "length")[apex] == pytest.approx(0.5 + 0.4 * 1.)       # elongated on the MTG
    assert _by_vid(ds, "struct_mass")[apex] == pytest.approx(2. * 0.9)
    assert _by_vid(ds, "length") == pytest.approx(dict(g.property("length")))


def test_array_style_outputs_reach_the_mtg_at_the_end_of_the_call(plant):
    g, ds, _, growth, _ = plant
    growth()
    np.testing.assert_allclose(ds.get("radius"), 1.1)
    np.testing.assert_allclose(list(dict(g.property("radius")).values()), 1.1)


def test_the_topology_is_updated_only_when_the_mtg_changed(plant):
    _, ds, vids, growth, carbon = plant
    growth()                                   # apex 0.9: no segmentation
    assert getattr(growth, "topology_updates", 0) == 0 and ds.n_nodes() == 3
    carbon()                                   # C_hexose_root 1 -> 0.5
    growth()                                   # apex 0.9 + 0.2 = 1.1: segmented
    assert growth.topology_updates == 1 and ds.n_nodes() == 4
    new_apex = int(ds.entity_ids("node")[ds.tips()[0]])
    assert new_apex not in vids
    lengths = _by_vid(ds, "length")
    assert lengths[vids[-1]] == pytest.approx(SEGMENT_LENGTH) and lengths[new_apex] == pytest.approx(0.1)


def test_functional_components_follow_the_new_topology(plant):
    _, ds, vids, growth, carbon = plant
    growth(); carbon(); growth()
    assert carbon._graph_view.n_nodes == 4
    new_apex = int(ds.entity_ids("node")[ds.tips()[0]])
    assert _by_vid(ds, "C_hexose_root")[new_apex] == pytest.approx(0.5)   # owner's on_grow="inherit"
    carbon()
    np.testing.assert_allclose(ds.get("C_hexose_root"), 0.25)


def test_post_segmentation_sees_the_new_structure(plant):
    _, ds, vids, growth, carbon = plant
    growth(); carbon(); growth()
    distance = _by_vid(ds, "distance_from_tip")
    assert distance[vids[0]] == pytest.approx(2. + 0.1)                 # two segments and the new apex below
    assert distance[int(ds.entity_ids("node")[ds.tips()[0]])] == 0.


def test_the_owner_of_a_variable_sets_its_growth_policy(plant):
    """RootGrowthProbe reads C_hexose_root first; CarbonProbe owns it with on_grow="inherit"."""
    _, ds, _, _, _ = plant
    meta = ds._variable_meta()["C_hexose_root"]
    assert (meta["variable_type"], meta["on_grow"], meta["kind"]) == ("state_variable", "inherit",
                                                                      "massic_concentration")


@dataclass
class _SegmentsProbe(RootGrowthProbe):
    x1: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")
    y1: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")
    z1: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")
    x2: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")
    y2: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")
    z2: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")


def test_default_triangles_are_the_segments_cylinders():
    Choregrapher().reset()
    _, ds, _ = make_chain()
    with pytest.raises(NotImplementedError):
        RootGrowthProbe(data_structure=ds).triangles()
    Choregrapher().reset()
    _, ds, _ = make_chain()
    model = _SegmentsProbe(data_structure=ds)
    ds.set("z1", np.array([0., -1., -2.]))
    ds.set("z2", np.array([-1., -2., -2.]))                     # the last one of zero length: no triangles
    triangles = model.triangles()
    assert triangles.owner.tolist() == [0] * 8 + [1] * 8
    assert triangles.vertices[:8, :, 2].max() == pytest.approx(0.)
    assert triangles.vertices[8:, :, 2].min() == pytest.approx(-2.)
    assert model.triangles(where=np.array([False, True, True])).owner.tolist() == [1] * 8


@dataclass
class _DrawnProbe(_SegmentsProbe):
    """Its own drawing: one flat triangle per element."""

    def triangles(self, where=None):
        from openalea.metafspm.coupling.geometry import Triangles
        nodes = np.arange(self.data_structure.n_nodes()) if where is None else np.flatnonzero(where)
        return Triangles(np.tile(np.eye(3), (nodes.size, 1, 1)), nodes)


def _segments(probe=None):
    Choregrapher().reset()
    _, ds, _ = make_chain()
    model = (probe or _SegmentsProbe)(data_structure=ds)
    ds.set("z1", np.array([0., -1., -2.]))
    ds.set("z2", np.array([-1., -2., -2.]))
    return ds, model


def test_a_data_structure_gives_its_structural_components_triangles():
    """ds.triangles(), for light models given DataStructures: its structural component's, the one drawing its
    elements itself before the segments' default, the one set explicitly first."""
    ds, model = _segments()
    assert ds.geometry_provider is model
    np.testing.assert_array_equal(ds.triangles().vertices, model.triangles().vertices)
    assert ds.triangles(where=np.array([False, True, True])).owner.tolist() == [1] * 8
    drawn = _DrawnProbe(data_structure=ds)
    assert ds.geometry_provider is drawn and ds.triangles().owner.tolist() == [0, 1, 2]
    _DrawnProbe(data_structure=ds)
    with pytest.raises(ValueError):                             # two drawings: which one is ambiguous
        ds.triangles()
    ds.geometry_provider = model
    assert len(ds.triangles()) == 16
    Choregrapher().reset()
    _, bare, _ = make_chain()
    with pytest.raises(ValueError):
        bare.triangles()


def test_a_union_gives_its_parts_triangles_with_its_node_indices():
    from openalea.metafspm.coupling.cross import UnionDataStructure
    first, _ = _segments()
    second, _ = _segments(_DrawnProbe)
    union = UnionDataStructure([first, second])
    triangles = union.triangles()
    assert triangles.owner.tolist() == [0] * 8 + [1] * 8 + [3, 4, 5]
    np.testing.assert_array_equal(triangles.vertices[:16], first.triangles().vertices)
    assert union.triangles(where=np.array([True, False, False, False, True, False])).owner.tolist() == [0] * 8 + [4]


def test_labels_and_triangles_of_a_mask():
    """ds.labels(): each node's MTG label name (or its complex's); a union's in its order. ds.triangles("name"):
    those of a mask, every node's when the DataStructure has no such mask."""
    from openalea.metafspm.coupling.cross import UnionDataStructure
    ds, _ = _segments()
    assert ds.labels().tolist() == ["RootSegment"] * 3
    assert len(ds.triangles("emerged")) == 16                          # no such mask: all
    ds.define_mask("deep", {"z1": "<0"})
    assert ds.triangles("deep").owner.tolist() == [1] * 8
    other, _ = _segments()
    union = UnionDataStructure([ds, other])
    assert union.labels().tolist() == ["RootSegment"] * 6
    assert union.triangles("deep").owner.tolist() == [1] * 8 + [3] * 8 + [4] * 8     # the other part: no mask


def test_labels_of_a_coarse_scale():
    """ds.labels(of=scale): each node's entity's label, or one per entity (per_node=False)."""
    from growth import RootGrowthProbe
    from scene_doubles import planting
    from openalea.metafspm.scene.scene import Scene
    Choregrapher().reset()

    class Roots:
        initiators = (RootGrowthProbe,)

        def __init__(self, data_structure, time_step):
            self.growth = RootGrowthProbe(data_structure=data_structure)
            self.components = [self.growth]

        def run(self):
            pass
    ds = Scene(planting([Roots]), time_step=3600.).populations[0].data_structure
    assert ds.labels(of="Organ").tolist() == ["RootInternode"] * ds.n_nodes()
    assert ds.labels(of="Organ", per_node=False).tolist() == ["RootInternode"]
    assert ds.labels(of="Axis", per_node=False).tolist() == ["Root"]
