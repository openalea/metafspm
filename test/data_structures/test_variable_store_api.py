"""
Variable store and MPG services used directly: loading a checkpoint in place, pickling the labels, vector-valued
variables, mask versions, unregistering, MTG tracking and reading, incremental graph extension, and the ArrayDict
scatter and index helpers.
"""
import pickle

import numpy as np
import pytest

from openalea.metafspm.coupling.declaration import VariableSpec
from openalea.metafspm.data_structure.arraydict import ArrayDict
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from simple_seedling import generate_simple_mpg_seedling
from openalea.metafspm.data_structure.mpg import MPG
from plants import seedling_ds


def _seedling():
    g, seedling = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return g, seedling, MPGDataStructure(g, from_scale=g.scales.SubOrgan)


def _grow(g, seedling):
    """A new root segment below the deepest root tip; returns its vid."""
    return g.add_child(seedling.root_segment6, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<',
                                                             label=g.labels.SubOrgan.RootSegment))


def _compartment_of(g):
    """{from_scale vid: Compartment vid}."""
    vertex_id = g.property("vertex_id")
    return {int(vertex_id[c]): int(c) for c in g.vertices(scale=g.scales.Compartment) if c in vertex_id}


# ---------------------------------------------------------------- checkpoints loaded in place

def test_an_mpg_datastructure_loads_its_checkpoint_in_place(tmp_path):
    _, _, ds = _seedling()
    ds.register("hexose", np.random.default_rng(0).random(ds.n_nodes()), location="node")
    ds.register("total", 3., location="scalar")
    saved = {name: np.array(ds.get(name)) for name in ("hexose", "total")}
    ds.checkpoint(str(tmp_path / "plant"))
    ds.set("hexose", 0.)
    ds.set("total", 0.)
    reference = ds
    ds.load_checkpoint(str(tmp_path / "plant"))
    assert ds is reference
    for name, values in saved.items():
        np.testing.assert_array_equal(ds.get(name), values)


def test_a_grid_loads_its_checkpoint_in_place(tmp_path):
    grid = ArrayDataStructure(shape=(3, 2, 2), dx=0.1)
    grid.register("nitrate", np.arange(12.).reshape(3, 2, 2), location="cell")
    grid.checkpoint(str(tmp_path / "grid"))
    grid.set("nitrate", -1.)
    grid.load_checkpoint(str(tmp_path / "grid"))
    np.testing.assert_array_equal(grid.get("nitrate"), np.arange(12.).reshape(3, 2, 2))


def test_a_checkpoint_of_another_class_is_refused(tmp_path):
    grid = ArrayDataStructure(shape=(2,))
    grid.checkpoint(str(tmp_path / "grid"))
    _, _, ds = _seedling()
    with pytest.raises(TypeError, match="not a MPGDataStructure"):
        ds.load_checkpoint(str(tmp_path / "grid"))


# ---------------------------------------------------------------- labels

def test_the_labels_survive_pickling():
    g, _, _ = _seedling()
    labels = pickle.loads(pickle.dumps(g.labels))
    assert labels.translator == g.labels.translator
    assert labels.SubOrgan.RootSegment == g.labels.SubOrgan.RootSegment
    assert labels.Compartment.Symplastic == g.labels.Compartment.Symplastic


# ---------------------------------------------------------------- vector-valued variables

def test_a_vector_variable_has_one_row_per_entity():
    _, _, ds = _seedling()
    ds.register("pools", 0., location="node", shape=(3,))
    assert ds.get("pools").shape == (ds.n_nodes(), 3)
    values = np.arange(3. * ds.n_nodes()).reshape(-1, 3)
    ds.set("pools", values)
    np.testing.assert_array_equal(ds.get("pools"), values)


def test_vector_variables_are_carried_over_growth():
    g, seedling, ds = _seedling()
    ds.register("fresh", location="node", default=7., shape=(2,))   # new entities get the default
    ds.register("inherited", 0., location="node", shape=(2,), on_grow="inherit")
    rows = np.arange(2. * ds.n_nodes()).reshape(-1, 2)
    ds.set("fresh", rows)
    ds.set("inherited", rows)
    parent = int(seedling.root_segment6)
    parent_row = rows[list(ds.entity_ids("node")).index(parent)]
    new = _grow(g, seedling)
    ds.update_topology()
    index = {int(v): i for i, v in enumerate(ds.entity_ids("node"))}
    assert ds.get("fresh").shape == (ds.n_nodes(), 2)
    np.testing.assert_array_equal(ds.get("fresh")[index[new]], [7., 7.])
    np.testing.assert_array_equal(ds.get("inherited")[index[new]], parent_row)
    np.testing.assert_array_equal(ds.get("fresh")[index[parent]], parent_row)       # existing rows kept


def test_vector_variables_are_checkpointed(tmp_path):
    grid = ArrayDataStructure(shape=(2, 2, 1))
    grid.register("pools", np.arange(12.).reshape(2, 2, 1, 3), location="cell", shape=(3,))   # grid shape + (3,)
    grid.checkpoint(str(tmp_path / "grid"))
    restored = ArrayDataStructure.restore(str(tmp_path / "grid"))
    np.testing.assert_array_equal(restored.get("pools"), grid.get("pools"))


# ---------------------------------------------------------------- mask versions

def test_the_mask_version_changes_with_the_mask_values_only():
    g, seedling, ds = _seedling()
    ds.register("flag", 0., location="node")
    ds.define_mask("flagged", {"flag": ">0"})
    version = ds.mask_version("flagged")
    ds.set("flag", 0.)                                   # written, same selection
    assert ds.mask_version("flagged") == version
    ds.set("flag", np.r_[1., np.zeros(ds.n_nodes() - 1)])
    assert ds.mask_version("flagged") == version + 1
    _grow(g, seedling)
    ds.update_topology()                                  # a new entity: another selection array
    assert ds.mask("flagged").shape == (ds.n_nodes(),)
    assert ds.mask_version("flagged") == version + 2


# ---------------------------------------------------------------- unregister

def test_unregister_removes_a_variable_and_its_metadata():
    _, _, ds = _seedling()
    ds.register("a", 1., location="node")
    ds.register("b", 2., location="node")
    ds.derive("total", {"a": 1., "b": 1.})
    version = ds.version
    ds.unregister("total")
    assert not ds.has("total") and "total" not in ds.derived()
    assert ds.version > version
    ds.register("total", 0., location="node")            # the name is free again
    np.testing.assert_array_equal(ds.get("total"), 0.)


def test_an_alias_cannot_be_unregistered():
    _, _, ds = _seedling()
    ds.register("hexose", 1., location="node")
    ds.alias("sugar", "hexose")
    with pytest.raises(ValueError, match="alias of 'hexose'"):
        ds.unregister("sugar")


def test_links_left_on_an_unregistered_variable_are_reported():
    _, _, ds = _seedling()
    ds.register("a", 1., location="node")
    ds.register("b", 2., location="node")
    ds.alias("a_again", "a")
    ds.derive("total", {"a": 1., "b": 1.})
    ds.unregister("a")
    with pytest.raises(ValueError) as error:
        ds.validate_variables()
    assert "a_again" in str(error.value) and "total" in str(error.value)


# ---------------------------------------------------------------- MTG tracking and reading

def test_a_tracked_variable_reaches_the_mtg_and_is_read_back():
    g, _, ds = _seedling()
    spec = VariableSpec(name="water", location="node", scale=g.scales.SubOrgan)
    values = np.arange(1., ds.n_nodes() + 1.)
    ds.register("water", values, location="node")
    assert ds.read_mtg(spec) is None                      # the property does not exist yet
    ds.track_mtg(spec)
    ds.flush_mtg()
    water = g.property("water")
    assert [water[int(v)] for v in ds.entity_ids("node")] == values.tolist()
    np.testing.assert_array_equal(ds.read_mtg(spec), values)


def test_only_mtg_backed_variables_are_tracked():
    g, _, ds = _seedling()
    solver_only = VariableSpec(name="flux", location="node")             # no MTG scale
    vector = VariableSpec(name="pools", location="node", scale=g.scales.SubOrgan, shape=(3,))
    ds.register("flux", 1., location="node")
    ds.register("pools", 1., location="node", shape=(3,))
    for spec in (solver_only, vector):
        ds.track_mtg(spec)
        assert ds.read_mtg(spec) is None
    ds.flush_mtg()
    assert not dict(g.property("flux")) and not dict(g.property("pools"))


# ---------------------------------------------------------------- incremental graph extension

def test_extend_graph_adds_the_new_segments_only():
    g, seedling, _ = _seedling()
    before = _compartment_of(g)
    new = _grow(g, seedling)
    change = g.extend_graph(g.scales.SubOrgan)
    assert change["added"] == [new] and change["removed"] == []
    after = _compartment_of(g)
    assert set(after) == set(before) | {new}
    assert all(after[v] == c for v, c in before.items())                # existing Compartments kept
    n_id_b = g.property("n_id_b")
    assert new in {int(n_id_b[c]) for c in g.vertices(scale=g.scales.Connection) if c in n_id_b}
    assert g.extend_graph(g.scales.SubOrgan)["added"] == []              # nothing new the second time


# ---------------------------------------------------------------- ArrayDict helpers

def test_arraydict_scatter_writes_at_keys():
    values = ArrayDict({10: 1., 30: 3., 20: 2.})
    np.testing.assert_array_equal(values.indices_of([30, 10]), [2, 0])
    values.scatter([30, 10], [33., 11.])
    assert dict(values) == {10: 11., 20: 2., 30: 33.}
    with pytest.raises(KeyError):
        values.indices_of([40])


def test_arraydict_reindex_restores_the_sorted_order():
    values = ArrayDict({1: 1., 2: 2., 3: 3.})
    values.order[:3] = [3, 1, 2]                          # keys rewritten out of order, values aligned with them
    values.arr[:3] = [30., 10., 20.]
    assert not values.check_invariant()
    values.reindex_sorted_inplace()
    assert values.check_invariant()
    assert dict(values) == {1: 10., 2: 20., 3: 30.}


# ---------------------------------------------------------------- accessors

def test_get_set_are_live_and_in_place():
    _, _, ds = seedling_ds()
    n = ds.n_nodes()
    ds.register("c", np.arange(n, dtype=float), location="node")
    view = ds.get("c")

    ds.set("c", np.ones(n))
    assert view is ds.get("c") and (view == 1.).all()
    assert ds.location("c") == "node" and ds.has("c") and not ds.has("nope")


def test_set_checks_length_and_registration():
    _, _, ds = seedling_ds()
    ds.register("q", location="edge", default=3.)
    assert (ds.get("q") == 3.).all() and ds.get("q").shape == (ds.n_edges(),)
    with pytest.raises(ValueError, match="q"):
        ds.set("q", np.ones(ds.n_edges() + 1))
    with pytest.raises(KeyError, match="unknown"):
        ds.set("unknown", 1.)


def test_version_counts_registrations():
    _, _, ds = seedling_ds()
    v0 = ds.version
    ds.register("a", location="node")
    ds.set("a", 1.)
    assert ds.version == v0 + 1


def test_alias_resolves_to_the_source():
    _, _, ds = seedling_ds()
    ds.register("hexose", location="node", default=1.)
    ds.alias("sugar", "hexose")

    assert ds.get("sugar") is ds.get("hexose")
    ds.set("sugar", 5.)
    assert (ds.get("hexose") == 5.).all()
    assert ds.location("sugar") == "node"
    assert ds.aliases() == {"sugar": "hexose"}


def test_alias_rejects_cycles_and_shadowing():
    _, _, ds = seedling_ds()
    ds.register("a", location="node")
    ds.register("b", location="node", default=1.)
    ds.alias("x", "a")
    with pytest.raises(ValueError, match="cycle"):
        ds.alias("a", "x")
    with pytest.raises(ValueError, match="b"):
        ds.alias("b", "a")


def test_grid_accessors():
    grid = ArrayDataStructure(shape=(3, 2, 2), dx=0.1)
    grid.register("T", default=10.)
    view = grid.get("T")
    grid.set("T", np.full((3, 2, 2), 12.))
    assert view is grid.get("T") and (view == 12.).all()
    assert grid.location("T") == "cell"
    with pytest.raises(ValueError, match="T"):
        grid.set("T", np.ones(5))


# ---------------------------------------------------------------- labels per instance

def test_labels_are_per_instance():
    first, second = MPG(), MPG()
    assert len(first.labels.translator) > 0
    assert first.labels.translator == second.labels.translator
    assert second.labels.SubOrgan.RootSegment == first.labels.SubOrgan.RootSegment
    assert isinstance(second.labels.SubOrgan.RootSegment, int)


def test_compartment_and_connection_labels_are_distinct():
    labels = MPG().labels
    assert labels.Compartment.Apoplastic != labels.Connection.Apoplastic


def test_a_grid_gives_its_cell_sizes():
    grid = ArrayDataStructure(shape=(2, 3, 4), dx=(0.1, 0.2, 0.05))
    np.testing.assert_array_equal(grid.dx, [0.1, 0.2, 0.05])
    grid.dx[0] = 9.                                                       # a copy
    assert grid.dx[0] == 0.1


def test_a_scalar_is_written_from_a_one_value_array():
    grid = ArrayDataStructure(shape=(1,))
    grid.register("air_temperature", 20., location="scalar")
    grid.set("air_temperature", np.array([21.5]))                       # e.g. computed from one-value parameters
    assert grid.get("air_temperature").shape == () and float(grid.get("air_temperature")) == 21.5
