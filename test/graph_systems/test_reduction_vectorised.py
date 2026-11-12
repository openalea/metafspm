"""
solve.reduction vectorised (devplan_release_optimisation.md O-5): the solve's terms, the outputs and the kept
nodes' outputs batched by template and archetype (Reduction.assignment, the sparse pattern cached per topology and node
set) equal, bit for bit, the group-by-group forms (each group's own archetype or blend, its kept nodes' block); the
DataStructure's topology arrays are read-only, shared until the topology changes, then rebuilt.
"""
import numpy as np
import pytest
from scipy.sparse import coo_matrix

from openalea.metafspm.data_structure.configs import PropsConfig

from test_reduction_classes import CLASS, N_SEGMENTS, SYSTEM, _class, _population

LENGTHS = [_class(3) * (1 + 0.037 * i) for i in range(N_SEGMENTS)]      # between classes: blends, and one class
LENGTHS[2] = _class(4)


def _group_by_group(reduction, component, data_structure, node_ids):
    """The terms as the reduction gave them before O-5: per group, its archetype's (or blend's) D and e on its kept
    nodes' positions in *node_ids* (a group with a kept node not among them left out)."""
    groups = reduction.reduced_groups(data_structure)
    archetypes = reduction.group_archetypes(data_structure, groups)
    position = {int(v): i for i, v in enumerate(np.asarray(node_ids).tolist())}
    rows, columns, values, e = [], [], [], np.zeros(len(node_ids))
    mtg = data_structure.structure
    for i, (owner, key) in groups.items():
        first, _ = mtg.instances.block(owner)
        kept = first + np.flatnonzero(reduction._reduced_scale(key) == mtg.scales.Compartment)
        local = np.array([position.get(int(v), -1) for v in kept.tolist()])
        if (local < 0).any():
            continue
        archetype = archetypes[i][SYSTEM]
        e[local] += archetype.e
        r, c = np.meshgrid(local, local, indexing="ij")
        rows.append(r.ravel()), columns.append(c.ravel()), values.append(archetype.D.ravel())
    n = len(node_ids)
    D = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(n, n)).tocsr()
    return D, e


@pytest.mark.parametrize("half", [False, True])
def test_the_terms_are_the_group_by_group_ones_bit_for_bit(half):
    population = _population(CLASS, LENGTHS)
    data_structure = population.data_structure
    reduction = data_structure.structure.reduction
    node_ids = data_structure.entity_ids("node")
    if half:                                         # a filtered solve: the groups of the first half of the nodes
        node_ids = node_ids[: node_ids.size // 2]
    D, e = reduction.terms(population.instance.hydraulics, SYSTEM, data_structure, node_ids)
    expected_D, expected_e = _group_by_group(reduction, population.instance.hydraulics, data_structure, node_ids)
    assert any(archetype[SYSTEM].parts for archetype in reduction.group_archetypes(data_structure, reduction.reduced_groups(data_structure)).values())
    np.testing.assert_array_equal(D.indptr, expected_D.indptr)
    np.testing.assert_array_equal(D.indices, expected_D.indices)
    np.testing.assert_array_equal(D.data, expected_D.data)
    np.testing.assert_array_equal(e, expected_e)
    again = reduction.terms(population.instance.hydraulics, SYSTEM, data_structure, node_ids)     # the pattern reused
    np.testing.assert_array_equal(again[0].data, D.data)


def test_the_outputs_are_each_groups_kind_or_blend():
    """Reduction.outputs (a number per group) and kept_outputs (a value per kept node) of the archetypes' outputs, their
    blends' weighted: as _blend gives a blended group's."""
    population = _population(CLASS, LENGTHS)
    data_structure = population.data_structure
    reduction = data_structure.structure.reduction
    reduction.terms(population.instance.hydraulics, SYSTEM, data_structure, data_structure.entity_ids("node"))
    for k, by_system in enumerate(reduction._archetypes.values()):
        archetype = by_system[SYSTEM]
        archetype.outputs = {"number": 1. + 0.5 * k, "vector": np.arange(archetype.D.shape[0]) * (2. + k)}
    reduction.__dict__["_blends"] = {}
    archetypes = reduction.group_archetypes(data_structure, reduction.reduced_groups(data_structure))
    outputs = reduction.outputs(data_structure, "number")
    for i, by_system in archetypes.items():
        assert outputs[i] == by_system[SYSTEM].outputs["number"]
    owner = np.asarray(data_structure.owner("SubOrgan"))
    (kept, values), = reduction.kept_outputs(data_structure, "vector", SYSTEM)
    for nodes, row in zip(kept, values):
        np.testing.assert_array_equal(row, archetypes[int(owner[nodes[0]])][SYSTEM].outputs["vector"])
        assert set(owner[nodes].tolist()) == {int(owner[nodes[0]])}


def test_the_topology_arrays_are_read_only_shared_and_rebuilt_after_growth():
    population = _population(CLASS, LENGTHS)
    data_structure = population.data_structure
    reduction = data_structure.structure.reduction
    nodes, owner = data_structure.entity_ids("node"), data_structure.owner("SubOrgan")
    assert data_structure.entity_ids("node") is nodes and data_structure.owner("SubOrgan") is owner          # no copy per call
    for array in (nodes, owner, data_structure.entity_ids("SubOrgan"), data_structure.entity_ids("edge")):
        with pytest.raises(ValueError):
            array[0] = -1
    table = reduction.group_table(data_structure)
    assert reduction.group_table(data_structure) is table and table.index.size == N_SEGMENTS
    mtg = data_structure.structure
    last = int(data_structure.entity_ids("SubOrgan")[-1])
    template = mtg.instances.template_of(last)
    child = mtg.add_child(last, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<', label=mtg.labels.SubOrgan.RootSegment))
    mtg.add_instance(child, template)
    data_structure.update_topology()
    grown = data_structure.entity_ids("node")
    assert grown is not nodes and grown.size == nodes.size + nodes.size // N_SEGMENTS
    assert reduction.group_table(data_structure) is not table and reduction.group_table(data_structure).index.size == N_SEGMENTS + 1
    D, e = reduction.terms(population.instance.hydraulics, SYSTEM, data_structure, grown)
    expected_D, expected_e = _group_by_group(reduction, population.instance.hydraulics, data_structure, grown)
    np.testing.assert_array_equal(D.toarray(), expected_D.toarray())
    np.testing.assert_array_equal(e, expected_e)


def test_the_assignment_is_kept_until_a_kind_variable_or_the_topology_changes():
    population = _population(CLASS, LENGTHS)
    data_structure = population.data_structure
    reduction = data_structure.structure.reduction
    first = reduction.assignment(data_structure)
    assert reduction.assignment(data_structure, release=True) is first                   # a step's solves and outputs: once
    data_structure.set("length", np.asarray(data_structure.get("length")) * 1.01)                    # its groups grown: assigned again
    second = reduction.assignment(data_structure)
    assert second is not first and not np.array_equal(second.weights, first.weights)


class Shapes:
    """A plant's parameter object of the default repr (Adel's LeafShapes): digested by its pickle."""

    def __init__(self):
        self.table = [1., 2.]


def test_a_signatures_kept_pickles_give_the_same_digest():
    from openalea.metafspm.solve.memo import content_key

    shapes, pickles = Shapes(), {}
    expected = content_key({"leaves": shapes, "a": 1.})
    assert content_key({"leaves": shapes, "a": 1.}, pickles=pickles) == expected
    assert content_key({"leaves": shapes, "a": 1.}, pickles=pickles) == expected and len(pickles) == 1
