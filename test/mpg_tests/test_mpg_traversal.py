from simple_seedling import seedling
from openalea.mtg.traversal import pre_order_in_scale, pre_order2, post_order2


g = seedling.g

# Unpack all vertex IDs for readable assertions
axis          = seedling.axis
growth_unit   = seedling.growth_unit
phytomer      = seedling.phytomer
phytomer2     = seedling.phytomer2
internode     = seedling.internode
meristem      = seedling.meristem
leaf          = seedling.leaf
internode2    = seedling.internode2
meristem2     = seedling.meristem2
leaf2         = seedling.leaf2
leafelement1  = seedling.leafelement1
leafelement2  = seedling.leafelement2
leafelement3  = seedling.leafelement3
leafelement4  = seedling.leafelement4
leafelement5  = seedling.leafelement5
leafelement6  = seedling.leafelement6
root_axis          = seedling.root_axis
growth_unit_root   = seedling.growth_unit_root
phytomer_root      = seedling.phytomer_root
root_internode1    = seedling.root_internode1
root_internode2    = seedling.root_internode2
root_segment1      = seedling.root_segment1
root_segment2      = seedling.root_segment2
root_segment3      = seedling.root_segment3
root_segment4      = seedling.root_segment4
root_segment5      = seedling.root_segment5
root_segment6      = seedling.root_segment6
internodeelement   = seedling.internodeelement
internodeelement2  = seedling.internodeelement2

# All biological (non-anchor) vertices created in the seedling
_ALL_VERTICES = {
    axis, growth_unit,
    phytomer, phytomer2,
    internode, meristem, leaf,
    internode2, meristem2, leaf2,
    internodeelement, internodeelement2,
    leafelement1, leafelement2, leafelement3,
    leafelement4, leafelement5, leafelement6,
    root_axis, growth_unit_root, phytomer_root,
    root_internode1, root_internode2,
    root_segment1, root_segment2, root_segment3,
    root_segment4, root_segment5, root_segment6,
}

# Pairs that must be ORDERED in any correct traversal:
#   (complex_vertex, one_of_its_components) — scale axis
#   (topo_parent, topo_child)               — topo axis within one complex
#
# In pre-order  : first element must appear BEFORE the second.
# In post-order : first element must appear AFTER  the second.
_SCALE_PAIRS = [
    (axis,           growth_unit),     # GrowthUnit decomposes Axis
    (growth_unit,    phytomer),        # Phytomer decomposes GrowthUnit
    (phytomer,       internode),       # Organ decomposes Phytomer
    (leaf,           leafelement1),    # SubOrgan decomposes Organ
    (root_axis,      growth_unit_root),
    (root_internode1, root_segment1),
    (root_internode2, root_segment4),  # lateral root: complex is root_internode2
    (internode,  internodeelement),    # SubOrgan decomposes Organ
    (internode2, internodeelement2),
]

_TOPO_PAIRS = [
    (internode,    meristem),      # meristem/leaf are topo children of internode
    (internode,    leaf),          #   within phytomer's component set
    (leafelement1, leafelement2),
    (leafelement2, leafelement3),
    (phytomer,     phytomer2),     # phytomer2 is topo successor of phytomer
    (root_internode1, root_internode2),   # lateral root branches off root_internode1
    (root_segment1, root_segment2),
    (root_segment2, root_segment3),
    (root_segment4, root_segment5),
    (root_segment5, root_segment6),
]


def test_partial_traversal():
    """Show why existing single-axis traversals were insufficient for MPG."""
    label = g.property('label')
    scale = g.property('scale')

    print("\npre_order_in_scale")
    for vid in pre_order_in_scale(g, g.root):
        if vid != g.root:
            print(g.scales.translator[scale[vid]], g.labels.translator[label[vid]])

    print("\nvertices")
    for vid in g.vertices():
        if vid != g.root:
            print(vid, g.scales.translator[scale[vid]], g.labels.translator[label[vid]])

    print("\npre_order2 (finest scale from leafelement1)")
    for vid in pre_order2(g, leafelement1):
        print(vid, g.scales.translator[scale[vid]], g.labels.translator[label[vid]])

    print("\npost_order2 (finest scale from leafelement1)")
    for vid in post_order2(g, leafelement1):
        print(vid, g.scales.translator[scale[vid]], g.labels.translator[label[vid]])


def test_combined_traversal():
    """pre_order_mpg and post_order_mpg visit every vertex in the correct order.

    Two invariants are checked for both traversals:

    Scale axis  — a complex is always processed before (pre) or after (post)
                  each of its fine-scale components.

    Topo axis   — within a complex's component set, a topological parent is
                  always processed before (pre) or after (post) its children.

    The multiscale branching case is covered explicitly:
      root_segment4 belongs to root_internode2 (scale axis) but its same-scale
      topological parent is root_segment2 (in root_internode1).  The correct
      pre-order must put root_segment2 before root_segment4, and the correct
      post-order must put root_segment4 before root_segment2, because
      root_internode1 (complex of root_segment2) precedes root_internode2
      (complex of root_segment4) in the organ-scale topology.
    """
    pre  = [v for v in g.pre_order_mpg()  if v != g.root]
    post = [v for v in g.post_order_mpg() if v != g.root]

    # ── Completeness ────────────────────────────────────────────────────────────
    assert set(pre)  == _ALL_VERTICES, f"pre_order_mpg missing: {_ALL_VERTICES - set(pre)}"
    assert set(post) == _ALL_VERTICES, f"post_order_mpg missing: {_ALL_VERTICES - set(post)}"

    # ── Ordering helpers ────────────────────────────────────────────────────────
    def pre_before(a, b):
        assert pre.index(a) < pre.index(b), f"pre-order: expected {a} before {b}"

    def post_after(a, b):
        assert post.index(a) > post.index(b), f"post-order: expected {a} after {b}"

    # ── Scale-axis ordering ─────────────────────────────────────────────────────
    for complex_v, component in _SCALE_PAIRS:
        pre_before(complex_v, component)   # complex before component in pre-order
        post_after(complex_v, component)   # complex after  component in post-order

    # ── Topo-axis ordering ──────────────────────────────────────────────────────
    for parent_v, child_v in _TOPO_PAIRS:
        pre_before(parent_v, child_v)
        post_after(parent_v, child_v)

    # ── Multiscale branching: cross-complex topo parent ─────────────────────────
    # root_segment4 is a lateral-component of root_internode2 but its topo parent
    # is root_segment2 (in root_internode1).  Since root_internode1 precedes
    # root_internode2 in the organ-scale topology, all of root_internode1's
    # SubOrgan components must precede root_segment4 in pre-order.
    pre_before(root_segment2, root_segment4)
    post_after(root_segment2, root_segment4)


if __name__ == "__main__":
    test_partial_traversal()
    test_combined_traversal()
    print("All assertions passed.")
