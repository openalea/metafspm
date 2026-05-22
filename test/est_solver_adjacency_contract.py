from dataclasses import dataclass

import numpy as np

from openalea.metafspm.utils import ArrayDict


@dataclass(frozen=True)
class SolverAdjacency:
    focus_vids: np.ndarray
    focus_glob_idx: np.ndarray
    root_local_idx: int
    parent_vid: np.ndarray
    has_parent: np.ndarray
    parent_idx: np.ndarray
    children: np.ndarray
    parents: np.ndarray
    child_offsets: np.ndarray
    children_grouped: np.ndarray
    edge_parents: np.ndarray
    is_root: np.ndarray
    is_tip: np.ndarray

    def children_of(self, local_idx: int) -> np.ndarray:
        start = self.child_offsets[local_idx]
        stop = self.child_offsets[local_idx + 1]
        return self.children_grouped[start:stop]

    def child_sum(self, node_values: np.ndarray) -> np.ndarray:
        return np.bincount(
            self.parents,
            weights=np.asarray(node_values, dtype=np.float64)[self.children],
            minlength=self.focus_vids.size,
        )


def build_solver_adjacency(vertex_index: ArrayDict, parent_id: ArrayDict, focus_elements, root_vid: int = 1) -> SolverAdjacency:
    """Minimal contract shared by the vectorized axial transport solvers."""
    focus_vids = np.asarray(focus_elements, dtype=np.int64)
    focus_glob_idx = vertex_index.indices_of(list(map(int, focus_vids)))

    global2local = np.full(vertex_index.size, -1, dtype=np.int64)
    global2local[focus_glob_idx] = np.arange(focus_vids.size, dtype=np.int64)

    root_glob_idx = vertex_index.indices_of([root_vid])[0]
    root_local_idx = int(global2local[root_glob_idx])

    parent_vid = parent_id.values_array()[focus_glob_idx].astype(np.int64, copy=False)
    has_parent = parent_vid >= 0
    parent_idx = np.full(focus_vids.size, -1, dtype=np.int64)

    if np.any(has_parent):
        parent_glob_idx = vertex_index.indices_of(parent_vid[has_parent].tolist()).astype(np.int64, copy=False)
        parent_loc = global2local[parent_glob_idx]
        child_loc = np.flatnonzero(has_parent)
        valid = parent_loc >= 0
        parent_idx[child_loc[valid]] = parent_loc[valid]

    children = np.flatnonzero(parent_idx >= 0).astype(np.int64, copy=False)
    parents = parent_idx[children]

    child_counts = np.bincount(parents, minlength=focus_vids.size).astype(np.int64, copy=False)
    child_offsets = np.empty(focus_vids.size + 1, dtype=np.int64)
    child_offsets[0] = 0
    np.cumsum(child_counts, out=child_offsets[1:])

    order = np.argsort(parents, kind="stable")
    children_grouped = children[order]
    edge_parents = parents[order]

    is_root = focus_vids == root_vid
    is_tip = child_counts == 0

    return SolverAdjacency(
        focus_vids=focus_vids,
        focus_glob_idx=focus_glob_idx,
        root_local_idx=root_local_idx,
        parent_vid=parent_vid,
        has_parent=has_parent,
        parent_idx=parent_idx,
        children=children,
        parents=parents,
        child_offsets=child_offsets,
        children_grouped=children_grouped,
        edge_parents=edge_parents,
        is_root=is_root,
        is_tip=is_tip,
    )


def make_linearized_tree():
    # Tree:
    # 1
    # |- 2
    # |  |- 3
    # |  `- 4
    # `- 5
    #    `- 6
    vertex_index = ArrayDict({1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}, dtype=np.int64)
    parent_id = ArrayDict({1: -1, 2: 1, 3: 2, 4: 2, 5: 1, 6: 5}, dtype=np.int64)
    return vertex_index, parent_id


def test_solver_adjacency_is_independent_from_focus_order():
    vertex_index, parent_id = make_linearized_tree()

    adjacency = build_solver_adjacency(vertex_index, parent_id, focus_elements=[5, 1, 4, 2], root_vid=1)

    assert adjacency.root_local_idx == 1
    assert adjacency.parent_idx.tolist() == [1, -1, 3, 1]
    assert adjacency.focus_vids[adjacency.children_of(1)].tolist() == [5, 2]
    assert adjacency.focus_vids[adjacency.children_of(3)].tolist() == [4]
    assert adjacency.focus_vids[adjacency.children_of(0)].tolist() == []
    assert adjacency.is_tip.tolist() == [True, False, True, False]


def test_solver_adjacency_distinguishes_root_from_truncated_boundary():
    vertex_index, parent_id = make_linearized_tree()

    adjacency = build_solver_adjacency(vertex_index, parent_id, focus_elements=[1, 3, 4], root_vid=1)

    assert adjacency.parent_vid.tolist() == [-1, 2, 2]
    assert adjacency.has_parent.tolist() == [False, True, True]
    assert adjacency.parent_idx.tolist() == [-1, -1, -1]
    assert adjacency.is_root.tolist() == [True, False, False]

    detached_boundary = adjacency.has_parent & (adjacency.parent_idx < 0) & (~adjacency.is_root)
    assert detached_boundary.tolist() == [False, True, True]


def test_solver_adjacency_supports_parent_reductions_used_by_axial_solvers():
    vertex_index, parent_id = make_linearized_tree()

    adjacency = build_solver_adjacency(vertex_index, parent_id, focus_elements=[5, 1, 4, 2], root_vid=1)
    node_values = np.asarray([10.0, 20.0, 30.0, 40.0])

    reduced = adjacency.child_sum(node_values)
    manual = np.asarray([0.0, 50.0, 0.0, 30.0])

    np.testing.assert_allclose(reduced, manual)
