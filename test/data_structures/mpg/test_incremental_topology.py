"""
Incremental topology (devplan_release_optimisation.md O-14): the caches patched after a structural edit are the arrays
a rebuild gives, element for element.
"""
import numpy as np

from openalea.metafspm.data_structure.instances import Instances, SectionTemplate


def _template(key, rng):
    size = int(rng.integers(3, 9))
    scale = rng.integers(3, 6, size=size)
    properties = {}
    for name in ("a", "b"):
        rows = np.flatnonzero(rng.random(size) < 0.6).astype(np.int64)
        if rows.size:
            properties[name] = (rows, rng.random(rows.size))
    return SectionTemplate(key=key, scale=scale, complex=np.full(size, -1), parent=np.full(size, -3),
                           properties=properties)


def _expected(instances, name, scale):
    """The tables as a loop over the instances computes them."""
    parts = []
    for i, key in enumerate(instances.template):
        template = instances.templates[key]
        if name is None:
            rows = np.flatnonzero(template.scale == scale)
        else:
            entry = template.properties.get(name)
            if entry is None:
                continue
            rows = entry[0] if scale is None else entry[0][template.scale[entry[0]] == scale]
        parts.append(instances.first[i] + rows)
    return np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, dtype=np.int64)


def _check(instances):
    for scale in (3, 4, 5):
        np.testing.assert_array_equal(instances.vids_at_scale(scale), _expected(instances, None, scale))
    for name in ("a", "b", "c"):
        for scale in (None, 3, 4):
            np.testing.assert_array_equal(instances.vids_having(name, scale), _expected(instances, name, scale))


def test_the_instances_tables_follow_appends_insertions_and_removals():
    rng = np.random.default_rng(3)
    templates = [_template(k, rng) for k in range(4)]
    instances, first = Instances(), 100
    for owner in range(30):
        template = templates[int(rng.integers(0, 4))]
        instances.add(owner, template, first)
        first += template.size + int(rng.integers(0, 3))
        _check(instances)                        # each table read after each append: extended, not rebuilt
    freed = instances.block(3)
    for owner in (3, 17, 29):
        instances.remove(owner)
        _check(instances)
    smallest = min(templates, key=lambda template: template.size)
    assert smallest.size <= freed[1]
    instances.add(1000, smallest, freed[0])      # an insertion below the last block (ids freed by a removal)
    _check(instances)


# ── The MPG's structural deltas and the caches patched from them (instanced and copied anatomies) ──

import pytest                                                               # noqa: E402

from openalea.metafspm.data_structure import mpg as mpg_module               # noqa: E402
from openalea.metafspm.data_structure.data_api import MPGDataStructure       # noqa: E402

from layered_anatomy import (grow_instanced_segment, grow_layered_segment, make_instanced_anatomy,  # noqa: E402
                             make_layered_anatomy, wiring)


@pytest.fixture
def checked(monkeypatch):
    """Every patched cache checked against its build (METAFSPM_CHECK_TOPOLOGY)."""
    monkeypatch.setattr(mpg_module, "_CHECK_TOPOLOGY", True)


def _topology(ds):
    """What a DataStructure derives from the structure, as plain values."""
    out = {"nodes": ds.entity_ids("node").tolist(), "edges": ds.entity_ids("edge").tolist(),
           "endpoints": ds.edge_endpoints().tolist()}
    for scale in ("Plant", "SubOrgan", "Layer", "Cell"):
        out[scale] = (ds.entity_ids(scale).tolist(), ds.owner(scale).tolist())
    return out


def _fresh(ds):
    """A DataStructure built anew on the same MPG (every array built, none patched)."""
    g = ds.mtg
    return MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=wiring(g))


@pytest.mark.parametrize("make, grow", [(make_instanced_anatomy, grow_instanced_segment),
                                        (make_layered_anatomy, grow_layered_segment)])
def test_the_topology_patched_after_growth_and_pruning_is_the_rebuilt_one(checked, make, grow):
    g, segments, sections = make(3)
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=wiring(g))
    ds.register("water", np.arange(ds.n_nodes(), dtype=float), location="node")
    _topology(ds)
    before = dict(g.__dict__.get("derived_counts", {"patched": 0}))
    for _ in range(3):                                          # growth: new sections (instances or copies)
        segment, section = grow(g, segments[-1])
        segments.append(segment)
        sections[segment] = section
        ds.update_topology()
        assert _topology(ds) == _topology(_fresh(ds))
    g.remove_vertex(segments.pop())                              # pruning: a section and its anatomy removed
    ds.update_topology()
    assert _topology(ds) == _topology(_fresh(ds))
    assert g.__dict__["derived_counts"]["patched"] > before.get("patched", 0)
    assert ds.n_nodes() == len(ds.get("water"))


def test_a_structure_delta_lists_the_vertices_removed_and_added():
    g, segments, sections = make_instanced_anatomy(2)
    since, last = g.structure_epoch, g.last_vid
    segment, _ = grow_instanced_segment(g, segments[-1])
    removed, added = g.structure_delta(since)
    assert removed.size == 0 and added.min() > last and segment in added.tolist()
    assert set(range(*np.add(g.instances.block(segment), (0, g.instances.block(segment)[0])))) <= set(added.tolist())
    since = g.structure_epoch
    block = g.instances.block(segment)
    g.remove_vertex(segment)
    removed, added = g.structure_delta(since)
    assert added.size == 0 and set(range(block[0], block[0] + block[1])) | {segment} <= set(removed.tolist())
    assert g.structure_delta(g.structure_epoch) is not None                       # no change: an empty delta


def test_a_moved_vertex_or_ids_given_again_give_no_delta():
    g, segments, sections = make_instanced_anatomy(3)
    since = g.structure_epoch
    g.insert_parent(segments[2], label=g.labels.SubOrgan.RootSegment, edge_type="<")   # a kept vertex re-parented
    assert g.structure_delta(since) is None
    g, segments, sections = make_instanced_anatomy(2)
    since = g.structure_epoch
    child = g.add_child(segments[-1], label=g.labels.SubOrgan.RootSegment, edge_type="<")
    g.structure_epoch
    g.remove_vertex(child)
    g.reuse_ids()                                                # the ids freed at the end given again
    assert g.structure_delta(since) is None
    assert g.structure_delta(-5) is None                         # older than the log
