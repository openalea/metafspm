"""
StructuralComponent: MPG-style and array-style steps,
the synchronisation of the DataStructure around MPG-style steps, and topology updates only when the MPG changed.
"""
from dataclasses import dataclass

import numpy as np
import pytest


from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.component import state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales

from growth import DOC, SEGMENT_LENGTH, CarbonProbe, RootGrowthProbe, make_chain



@pytest.fixture
def plant():
    mtg, data_structure, vids = make_chain()
    growth = RootGrowthProbe(data_structure=data_structure)
    carbon = CarbonProbe(data_structure=data_structure)
    return mtg, data_structure, vids, growth, carbon


def _by_vid(data_structure, name):
    return dict(zip(data_structure.entity_ids("node").tolist(), data_structure.get(name).tolist()))


def test_structural_steps_run_in_the_growth_rows():
    _, data_structure, _ = make_chain()
    model = RootGrowthProbe(data_structure=data_structure)
    rows = [[f.func.name for f in group] for group in model.__dict__["_choregraphy"][1].values()]
    assert rows == [["potential_growth"], ["actual_growth", "radius"], ["segmentation"], ["distance_from_tip"]]


def test_the_mtg_is_the_data_structure_mtg(plant):
    mtg, _, _, growth, _ = plant
    assert growth.mtg is mtg


def test_inputs_are_flushed_to_the_mtg_before_an_mpg_style_step(plant):
    _, data_structure, vids, growth, _ = plant
    data_structure.set("C_hexose_root", 0.25)            # e.g. written by the carbon model, not yet on the MTG
    growth()
    assert growth.read_hexose == {vids[-1]: 0.25}


def test_structural_outputs_are_re_read_after_mpg_style_steps(plant):
    mtg, data_structure, vids, growth, _ = plant
    growth()
    apex = vids[-1]
    assert _by_vid(data_structure, "length")[apex] == pytest.approx(0.5 + 0.4 * 1.)       # elongated on the MTG
    assert _by_vid(data_structure, "struct_mass")[apex] == pytest.approx(2. * 0.9)
    assert _by_vid(data_structure, "length") == pytest.approx(dict(mtg.property("length")))


def test_array_style_outputs_reach_the_mtg_at_the_end_of_the_call(plant):
    mtg, data_structure, _, growth, _ = plant
    growth()
    np.testing.assert_allclose(data_structure.get("radius"), 1.1)
    np.testing.assert_allclose(list(dict(mtg.property("radius")).values()), 1.1)


def test_the_topology_is_updated_only_when_the_mtg_changed(plant):
    _, data_structure, vids, growth, carbon = plant
    growth()                                   # apex 0.9: no segmentation
    assert getattr(growth, "topology_updates", 0) == 0 and data_structure.n_nodes() == 3
    carbon()                                   # C_hexose_root 1 -> 0.5
    growth()                                   # apex 0.9 + 0.2 = 1.1: segmented
    assert growth.topology_updates == 1 and data_structure.n_nodes() == 4
    new_apex = int(data_structure.entity_ids("node")[data_structure.tips()[0]])
    assert new_apex not in vids
    lengths = _by_vid(data_structure, "length")
    assert lengths[vids[-1]] == pytest.approx(SEGMENT_LENGTH) and lengths[new_apex] == pytest.approx(0.1)


def test_functional_components_follow_the_new_topology(plant):
    _, data_structure, vids, growth, carbon = plant
    growth(); carbon(); growth()
    assert carbon._graph_view.n_nodes == 4
    new_apex = int(data_structure.entity_ids("node")[data_structure.tips()[0]])
    assert _by_vid(data_structure, "C_hexose_root")[new_apex] == pytest.approx(0.5)   # owner's on_grow="inherit"
    carbon()
    np.testing.assert_allclose(data_structure.get("C_hexose_root"), 0.25)


def test_post_segmentation_sees_the_new_structure(plant):
    _, data_structure, vids, growth, carbon = plant
    growth(); carbon(); growth()
    distance = _by_vid(data_structure, "distance_from_tip")
    assert distance[vids[0]] == pytest.approx(2. + 0.1)                 # two segments and the new apex below
    assert distance[int(data_structure.entity_ids("node")[data_structure.tips()[0]])] == 0.


def test_the_owner_of_a_variable_sets_its_growth_policy(plant):
    """RootGrowthProbe reads C_hexose_root first; CarbonProbe owns it with on_grow="inherit"."""
    _, data_structure, _, _, _ = plant
    meta = data_structure._variable_meta()["C_hexose_root"]
    assert (meta["role"], meta["on_grow"], meta["kind"]) == ("state_variable", "inherit",
                                                                      "massic_concentration")


@dataclass
class _SegmentsProbe(RootGrowthProbe):
    x1: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    y1: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    z1: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    x2: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    y2: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    z2: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")


def test_default_triangles_are_the_segments_cylinders():
    _, data_structure, _ = make_chain()
    with pytest.raises(NotImplementedError):
        RootGrowthProbe(data_structure=data_structure).triangles()
    _, data_structure, _ = make_chain()
    model = _SegmentsProbe(data_structure=data_structure)
    data_structure.set("z1", np.array([0., -1., -2.]))
    data_structure.set("z2", np.array([-1., -2., -2.]))                     # the last one of zero length: no triangles
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
    _, data_structure, _ = make_chain()
    model = (probe or _SegmentsProbe)(data_structure=data_structure)
    data_structure.set("z1", np.array([0., -1., -2.]))
    data_structure.set("z2", np.array([-1., -2., -2.]))
    return data_structure, model


def test_a_data_structure_gives_its_structural_components_triangles():
    """data_structure.triangles(), for light models given DataStructures: its structural component's, the one drawing its
    elements itself before the segments' default, the one set explicitly first."""
    data_structure, model = _segments()
    assert data_structure.geometry_provider is model
    np.testing.assert_array_equal(data_structure.triangles().vertices, model.triangles().vertices)
    assert data_structure.triangles(where=np.array([False, True, True])).owner.tolist() == [1] * 8
    drawn = _DrawnProbe(data_structure=data_structure)
    assert data_structure.geometry_provider is drawn and data_structure.triangles().owner.tolist() == [0, 1, 2]
    _DrawnProbe(data_structure=data_structure)
    with pytest.raises(ValueError):                             # two drawings: which one is ambiguous
        data_structure.triangles()
    data_structure.geometry_provider = model
    assert len(data_structure.triangles()) == 16
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
    """data_structure.labels(): each node's MTG label name (or its complex's); a union's in its order. data_structure.triangles("name"):
    those of a mask, every node's when the DataStructure has no such mask."""
    from openalea.metafspm.coupling.cross import UnionDataStructure
    data_structure, _ = _segments()
    assert data_structure.labels().tolist() == ["RootSegment"] * 3
    assert len(data_structure.triangles("planted")) == 16                          # no such mask: all
    data_structure.define_mask("deep", {"z1": "<0"})
    assert data_structure.triangles("deep").owner.tolist() == [1] * 8
    other, _ = _segments()
    union = UnionDataStructure([data_structure, other])
    assert union.labels().tolist() == ["RootSegment"] * 6
    assert union.triangles("deep").owner.tolist() == [1] * 8 + [3] * 8 + [4] * 8     # the other part: no mask


def test_labels_of_a_coarse_scale():
    """data_structure.labels(location=scale): each node's entity's label, or one per entity (per_node=False)."""
    from growth import RootGrowthProbe
    from scene_doubles import doubles_planting
    from openalea.metafspm.scene.scene import Scene

    class Roots(CompositeModel):
        initiators = (RootGrowthProbe,)

        parts = {"growth": RootGrowthProbe}
        schedule = ()
    data_structure = Scene(doubles_planting([Roots]), time_step=3600.).populations[0].data_structure
    assert data_structure.labels(location="Organ").tolist() == ["RootInternode"] * data_structure.n_nodes()
    assert data_structure.labels(location="Organ", per_node=False).tolist() == ["RootInternode"]
    assert data_structure.labels(location="Axis", per_node=False).tolist() == ["Root"]


def test_a_component_whose_geometry_is_never_drawn_warns():
    """On an anatomy-mode DataStructure, the frames and the light models draw the view at its from_scale
    (data_structure.at("SubOrgan").triangles()): a StructuralComponent working at a coarser scale (its finest declaration at
    Organ) registers its geometry on data_structure.at("Organ"), never drawn, and warns, naming itself and that view; one working
    at the SubOrgans, or at the Compartments (an anatomy's: not drawn by design), does not."""
    import warnings
    from openalea.metafspm.coupling.component import StructuralComponent
    from openalea.metafspm.data_structure.data_api import MPGDataStructure
    from anatomy import make_rooted_anatomy, wiring

    @dataclass
    class AtOrgans(StructuralComponent):
        organ_length: float = state_variable(**DOC, default=0., scale=scales.Organ)

    @dataclass
    class AtElements(StructuralComponent):
        element_length: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)

    @dataclass
    class AtCompartments(StructuralComponent):
        water: float = state_variable(**DOC, default=0., scale=scales.Compartment, location="node")

    mtg, _, _ = make_rooted_anatomy(n_segments=3)
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan, nodes="Compartment", wiring=wiring(mtg))
    data_structure.simulation_time_step = 1
    with pytest.warns(UserWarning, match=r"AtOrgans works at the Organ scale .* data_structure\.at\('Organ'\), which the frames"):
        AtOrgans(data_structure=data_structure, time_step=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        AtElements(data_structure=data_structure, time_step=1)
        AtCompartments(data_structure=data_structure, time_step=1)


def test_the_geometry_of_a_data_structure_in_a_domain():
    """data_structure.geometry: each element's area fraction and length fraction above the soil surface (data_structure.soil_surface, the
    Scene's), the centre of its part there, and a geometric variable (derive_geometry) recomputed when the geometry
    changes (the segment ends written, or the provider called)."""
    from openalea.metafspm.coupling.geometry import CentreIn, FractionIn
    data_structure, model = _segments()                                      # z from 0 to -1, -1 to -2, the last of no length
    data_structure.set("z1", np.array([0.5, -1., -2.]))                      # the first one half above 0
    np.testing.assert_allclose(data_structure.geometry.fraction_in("air", by="length"), [1. / 3., 0., 0.])
    np.testing.assert_allclose(data_structure.geometry.fraction_in("soil", by="length"), [2. / 3., 1., 1.])
    by_area = data_structure.geometry.fraction_in("air")
    # (a slim cylinder: its sides and its caps, the cap above the soil)
    triangles = model.triangles()
    first = triangles.owner == 0
    expected = (triangles.areas()[first] * triangles.fraction_in(data_structure.geometry.domain("air"))[first]).sum() / \
        triangles.areas()[first].sum()
    assert 1. / 3. < by_area[0] == pytest.approx(expected, rel=1e-12) and by_area[1] == 0. and np.isnan(by_area[2])
    assert data_structure.geometry.fraction_in("air", empty=0.)[2] == 0.
    np.testing.assert_allclose(data_structure.geometry.centre_in("air", by="length")[0, 2], 0.25)
    assert 0.25 < data_structure.geometry.centre_in("air")[0, 2] < 0.5                  # the cap at the top
    assert np.isnan(data_structure.geometry.centre_in("air")[1]).all()
    data_structure.soil_surface = -1.5                                       # the soil lower: the second one half in the air
    np.testing.assert_allclose(data_structure.geometry.fraction_in("air", by="length"), [1., 0.5, 0.])
    data_structure.soil_surface = 0.
    with pytest.raises(KeyError):                                # the segments' cylinders carry no attribute
        data_structure.geometry.fraction_in("air", weight="green")
    values = data_structure.derive_geometry("emerged", FractionIn("air", by="length"))
    np.testing.assert_allclose(values, [1. / 3., 0., 0.])
    centre = data_structure.derive_geometry("height", CentreIn("air", by="length"))
    assert centre[0] == pytest.approx(0.25, rel=1e-9) and np.isnan(centre[1])
    data_structure.set("z1", np.array([1., -1., -2.]))                       # the ends written: recomputed when read
    np.testing.assert_allclose(data_structure.get("emerged"), [0.5, 0., 0.])
    assert data_structure.get("height")[0] == pytest.approx(0.5, rel=1e-9)
    with pytest.raises(ValueError):
        data_structure.set("emerged", 0.)                                    # derived: not written
