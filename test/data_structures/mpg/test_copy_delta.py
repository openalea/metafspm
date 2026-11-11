"""
A DataStructure's copy in another process (scene.runner's publication): patched from its owner's MPG delta
(MPG.copy_delta, MPG.apply_structure_delta), it is the owner, element for element, after growth, pruning, instances
added and removed and properties written; Compute(checks={"publication"})'s comparison finds a copy that is not; a
worker publishes in full when the delta cannot be used.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from openalea.metafspm.compute import Compute
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene import runner
from openalea.metafspm.scene.shared import SharedStore

from layered_anatomy import (grow_instanced_segment, grow_layered_segment, make_instanced_anatomy,
                             make_layered_anatomy, wiring)


def _data_structure(make):
    mtg, segments, sections = make(3)
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan, nodes="Compartment", wiring=wiring(mtg))
    data_structure.register("water", np.arange(data_structure.n_nodes(), dtype=float), location="node")
    return data_structure, segments


class _Copy:
    """An owner's DataStructure and its copy (built alike, then given the owner's publications), one folder."""

    def __init__(self, make, folder):
        self.owner, self.segments = _data_structure(make)
        self.copy, _ = _data_structure(make)
        self.folder = os.fspath(folder)
        self.writer = SharedStore(os.path.join(self.folder, "shared"))
        self.reader = SharedStore(os.path.join(self.folder, "shared"))
        self.marks = None

    def publish(self, mode, check=True):
        publication = runner.publish(self.owner, os.path.join(self.folder, "owner"), self.writer, mode,
                                     marks=self.marks, check=check)
        self.marks = publication.pop("marks")
        runner.load_publication(self.copy, os.path.join(self.folder, "owner"), self.reader, publication)
        return publication

    def assert_equal(self):
        owner, copy = self.owner, self.copy
        assert copy.n_nodes() == owner.n_nodes()
        np.testing.assert_array_equal(copy.get("water"), owner.get("water"))
        for location in ("node", "edge", "SubOrgan", "Cell"):
            np.testing.assert_array_equal(copy.entity_ids(location), owner.entity_ids(location))
        np.testing.assert_array_equal(copy.edge_endpoints(), owner.edge_endpoints())


@pytest.mark.parametrize("make, grow", [(make_instanced_anatomy, grow_instanced_segment),
                                        (make_layered_anatomy, grow_layered_segment)])
def test_a_copy_patched_after_growth_pruning_and_instances_is_the_owner(tmp_path, make, grow):
    """The copy's structure caches are its own, patched from its own log: checked against their builds too."""
    with Compute(checks={"topology"}).limits():
        _grow_prune_and_write(_Copy(make, tmp_path), grow)


def _grow_prune_and_write(pair, grow):
    pair.publish("full")
    mtg, copy_mtg = pair.owner.mtg, pair.copy.mtg
    for _ in range(2):                                      # growth: segments and their sections (instances or copies)
        segment, _ = grow(mtg, pair.segments[-1])
        pair.segments.append(segment)
        pair.owner.update_topology()
        pair.owner.set("water", np.arange(pair.owner.n_nodes(), dtype=float) * 2.)
        assert pair.publish("delta")["check"]               # checked against the MPG whole (check_publication)
        assert pair.copy.mtg is copy_mtg                    # patched in place, not replaced
        pair.assert_equal()
    mtg.remove_vertex(pair.segments.pop())                  # pruning: a segment, its section (an instance) removed
    pair.owner.update_topology()
    pair.publish("delta")
    pair.assert_equal()
    assert mtg.has_instances() == copy_mtg.has_instances()
    labels = mtg._column("label")                           # properties of kept vertices written in the owner's MPG
    labels[pair.segments[0]] = mtg.labels.SubOrgan.RootApex
    mtg.set_values("custom_length", pair.segments, [1., 2., 3., 4.])
    segment, _ = grow(mtg, pair.segments[-1])
    pair.owner.update_topology()
    pair.publish("delta")
    assert copy_mtg._column("label")[pair.segments[0]] == mtg.labels.SubOrgan.RootApex
    assert copy_mtg._column("custom_length")[pair.segments[1]] == 2.


def test_a_delta_leaves_out_the_numeric_columns_unchanged_since_the_marks():
    data_structure, segments = _data_structure(make_layered_anatomy)
    mtg = data_structure.mtg
    marks = mtg.copy_marks()
    delta = mtg.copy_delta(marks)
    numeric = {name for name, column in mtg._properties.items() if column.arr.dtype != object}
    assert numeric and not numeric & set(delta["columns"])
    mtg.set_values("vessel_index", [segments[0]], [5.])
    assert "vessel_index" in mtg.copy_delta(marks)["columns"]
    assert delta["names"] == list(mtg._properties)


def test_the_publication_check_finds_a_copy_that_differs(tmp_path):
    pair = _Copy(make_layered_anatomy, tmp_path)
    pair.owner.mtg.set_values("custom_length", pair.segments, [1., 2., 3.])
    pair.owner.mtg.convert_properties_to_arraydict()          # a numeric column: left out of a delta unwritten
    pair.publish("full")
    pair.copy.mtg.set_values("custom_length", [pair.segments[0]], [7.])   # the copy's own write: not the owner's
    grow_layered_segment(pair.owner.mtg, pair.segments[-1])
    pair.owner.update_topology()
    with pytest.raises(AssertionError, match="custom_length"):
        pair.publish("delta")


def _worker(folder, data_structure, forked_state):
    """A worker's publishing part (no process): its last publication's state that of a fork (*forked_state*: the copy
    is the worker's DataStructure) or of none."""
    worker = runner._Worker.__new__(runner._Worker)
    worker.scene, worker.folder = SimpleNamespace(compute=Compute()), os.fspath(folder)
    worker.store = SharedStore(os.path.join(worker.folder, "shared"))
    worker.published = {"roots": runner._Worker._published(data_structure,
                                                            runner._marks(data_structure) if forked_state else None)}
    return worker


def test_a_worker_publishes_in_full_when_the_delta_cannot_serve(tmp_path):
    """Without a copy's state, in full; a synchronisation in full; the values alone while the topology stays; a
    delta when it moved; in full again when the variables changed or the MPG was replaced."""
    data_structure, segments = _data_structure(make_layered_anatomy)
    worker = _worker(tmp_path, data_structure, forked_state=False)
    population = SimpleNamespace(name="roots", data_structure=data_structure)

    def mode(full=False):
        return worker.publish(population, full)["mode"]

    assert mode() == "full"
    assert mode() == "values"
    grow_layered_segment(data_structure.mtg, segments[-1])
    data_structure.update_topology()
    assert mode() == "delta"
    data_structure.register("sugar", 0., location="node")
    assert mode() == "full"
    assert mode(full=True) == "full"
    grow_layered_segment(data_structure.mtg, segments[-1])
    data_structure.update_topology()
    assert mode() == "delta"
    data_structure._mtg = make_layered_anatomy(3)[0]          # another MPG (e.g. a checkpoint loaded in the worker)
    assert mode() == "full"


def test_a_forked_worker_publishes_a_delta_from_the_fork_and_its_parts_by_id(tmp_path):
    """At the fork the copy is the worker's DataStructure: its first publication after growth is a delta; the shared
    parts both hold since the fork (the instances' templates) go by their id, no part written."""
    from openalea.metafspm.scene.shared import shared_objects
    data_structure, segments = _data_structure(make_instanced_anatomy)
    forked = shared_objects(data_structure.mtg)
    templates = list(data_structure.mtg.instances.templates.values())
    assert templates and all(forked.get(id(template)) is template for template in templates)
    worker = _worker(tmp_path, data_structure, forked_state=True)
    worker.store.forked = forked
    grow_instanced_segment(data_structure.mtg, segments[-1])
    data_structure.update_topology()
    population = SimpleNamespace(name="roots", data_structure=data_structure)
    assert worker.publish(population, False)["mode"] == "delta"
    shared = os.path.join(tmp_path, "shared")
    assert not os.path.isdir(shared) or not os.listdir(shared)                        # no part written
    reader = SharedStore(os.path.join(tmp_path, "shared"), forked=forked)
    with open(os.path.join(tmp_path, "roots", "state.pkl"), "rb") as file:
        content = reader.load(file)
    assert content["delta"]["attributes"]["_instances"].templates[templates[0].key] is templates[0]


def test_a_reader_kept_over_many_loads_lets_go_the_parts_no_longer_referred_to(tmp_path):
    """SharedStore.load_referring / retain: the parts a load refers to (and those they refer to) are known; the others
    are let go and their files removed; a writer whose part was removed writes it again."""
    from openalea.metafspm.scene.shared import Shared
    writer, reader = SharedStore(os.fspath(tmp_path)), SharedStore(os.fspath(tmp_path))
    inner, other = np.arange(5.), np.ones(3)
    outer = {"inner": Shared("inner", inner)}

    def load(value):
        path = os.path.join(tmp_path, "content.pkl")
        with open(path, "wb") as file:
            writer.dump(value, file)
        with open(path, "rb") as file:
            return reader.load_referring(file)

    content, referred = load({"outer": Shared("outer", outer), "other": Shared("other", other)})
    assert len(referred) == 3 and content["outer"]["inner"].tolist() == inner.tolist()
    _, again = load({"outer": Shared("outer", outer)})         # the outer part loaded already: its inner one counted
    assert len(again) == 2
    removed = reader.retain(again)
    assert [key.split("-")[0] for key in removed] == ["other"]
    assert not os.path.exists(os.path.join(tmp_path, removed[0] + ".pkl"))
    content, _ = load({"other": Shared("other", other)})       # written again by its writer
    assert content["other"].tolist() == other.tolist()
