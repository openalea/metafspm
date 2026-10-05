"""
StructuralComponent (design note structure_and_boundaries §3, step 2b, DS19): MPG-style and array-style steps,
the synchronisation of the DataStructure around MPG-style steps, and topology updates only when the MPG changed.
"""
import numpy as np
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher

from growth import SEGMENT_LENGTH, CarbonProbe, RootGrowthProbe, make_chain


@pytest.fixture(autouse=True)
def _fresh_choregrapher():
    Choregrapher().reset()
    yield
    Choregrapher().reset()


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
