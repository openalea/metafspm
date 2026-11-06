"""
solve.reduction vectorised (devplan_release_optimisation.md O-5): the solve's terms, the outputs and the kept
nodes' outputs batched by template and kind (Reduction.assignment, the sparse pattern cached per topology and node
set) equal, bit for bit, the group-by-group forms (each group's own kind or blend, its kept nodes' block); the
DataStructure's topology arrays are read-only, shared until the topology changes, then rebuilt.
"""
import numpy as np
import pytest
from scipy.sparse import coo_matrix

from openalea.metafspm.data_structure.configs import PropsConfig

from test_reduction_classes import CLASS, N_SEGMENTS, SYSTEM, _class, _population

LENGTHS = [_class(3) * (1 + 0.037 * i) for i in range(N_SEGMENTS)]      # between classes: blends, and one class
LENGTHS[2] = _class(4)


def _group_by_group(reduction, component, ds, node_ids):
    """The terms as the reduction gave them before O-5: per group, its kind's (or blend's) D and e on its kept
    nodes' positions in *node_ids* (a group with a kept node not among them left out)."""
    groups = reduction.reduced_groups(ds)
    kinds = reduction.group_kinds(ds, groups)
    position = {int(v): i for i, v in enumerate(np.asarray(node_ids).tolist())}
    rows, columns, values, e = [], [], [], np.zeros(len(node_ids))
    g = ds.structure
    for i, (owner, key) in groups.items():
        first, _ = g.instances.block(owner)
        kept = first + np.flatnonzero(reduction._reduced_scale(key) == g.scales.Compartment)
        local = np.array([position.get(int(v), -1) for v in kept.tolist()])
        if (local < 0).any():
            continue
        kind = kinds[i][SYSTEM]
        e[local] += kind.e
        r, c = np.meshgrid(local, local, indexing="ij")
        rows.append(r.ravel()), columns.append(c.ravel()), values.append(kind.D.ravel())
    n = len(node_ids)
    D = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(n, n)).tocsr()
    return D, e


@pytest.mark.parametrize("half", [False, True])
def test_the_terms_are_the_group_by_group_ones_bit_for_bit(half):
    population = _population(CLASS, LENGTHS)
    ds = population.data_structure
    reduction = ds.structure.reduction
    node_ids = ds.entity_ids("node")
    if half:                                         # a filtered solve: the groups of the first half of the nodes
        node_ids = node_ids[: node_ids.size // 2]
    D, e = reduction.terms(population.instance.hydraulics, SYSTEM, ds, node_ids)
    expected_D, expected_e = _group_by_group(reduction, population.instance.hydraulics, ds, node_ids)
    assert any(kind[SYSTEM].parts for kind in reduction.group_kinds(ds, reduction.reduced_groups(ds)).values())
    np.testing.assert_array_equal(D.indptr, expected_D.indptr)
    np.testing.assert_array_equal(D.indices, expected_D.indices)
    np.testing.assert_array_equal(D.data, expected_D.data)
    np.testing.assert_array_equal(e, expected_e)
    again = reduction.terms(population.instance.hydraulics, SYSTEM, ds, node_ids)     # the pattern reused
    np.testing.assert_array_equal(again[0].data, D.data)


def test_the_outputs_are_each_groups_kind_or_blend():
    """Reduction.outputs (a number per group) and kept_outputs (a value per kept node) of the kinds' outputs, their
    blends' weighted: as _blend gives a blended group's."""
    population = _population(CLASS, LENGTHS)
    ds = population.data_structure
    reduction = ds.structure.reduction
    reduction.terms(population.instance.hydraulics, SYSTEM, ds, ds.entity_ids("node"))
    for k, by_system in enumerate(reduction._kinds.values()):
        kind = by_system[SYSTEM]
        kind.outputs = {"number": 1. + 0.5 * k, "vector": np.arange(kind.D.shape[0]) * (2. + k)}
    reduction.__dict__["_blends"] = {}
    kinds = reduction.group_kinds(ds, reduction.reduced_groups(ds))
    outputs = reduction.outputs(ds, "number")
    for i, by_system in kinds.items():
        assert outputs[i] == by_system[SYSTEM].outputs["number"]
    owner = np.asarray(ds.owner("SubOrgan"))
    (kept, values), = reduction.kept_outputs(ds, "vector", SYSTEM)
    for nodes, row in zip(kept, values):
        np.testing.assert_array_equal(row, kinds[int(owner[nodes[0]])][SYSTEM].outputs["vector"])
        assert set(owner[nodes].tolist()) == {int(owner[nodes[0]])}


def test_the_topology_arrays_are_read_only_shared_and_rebuilt_after_growth():
    population = _population(CLASS, LENGTHS)
    ds = population.data_structure
    reduction = ds.structure.reduction
    nodes, owner = ds.entity_ids("node"), ds.owner("SubOrgan")
    assert ds.entity_ids("node") is nodes and ds.owner("SubOrgan") is owner          # no copy per call
    for array in (nodes, owner, ds.entity_ids("SubOrgan"), ds.entity_ids("edge")):
        with pytest.raises(ValueError):
            array[0] = -1
    table = reduction.group_table(ds)
    assert reduction.group_table(ds) is table and table.index.size == N_SEGMENTS
    g = ds.structure
    last = int(ds.entity_ids("SubOrgan")[-1])
    template = g.instances.template_of(last)
    child = g.add_child(last, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<', label=g.labels.SubOrgan.RootSegment))
    g.add_instance(child, template)
    ds.update_topology()
    grown = ds.entity_ids("node")
    assert grown is not nodes and grown.size == nodes.size + nodes.size // N_SEGMENTS
    assert reduction.group_table(ds) is not table and reduction.group_table(ds).index.size == N_SEGMENTS + 1
    D, e = reduction.terms(population.instance.hydraulics, SYSTEM, ds, grown)
    expected_D, expected_e = _group_by_group(reduction, population.instance.hydraulics, ds, grown)
    np.testing.assert_array_equal(D.toarray(), expected_D.toarray())
    np.testing.assert_array_equal(e, expected_e)
