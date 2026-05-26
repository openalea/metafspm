
from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.mtg.traversal import pre_order_in_scale, pre_order2, post_order2


# ── Tests ─────────────────────────────────────────────────────────────────────
# Note: LabelsConfig mutates class attributes on first MPG() construction, so
# a second MPG() in the same process gets an empty translator.  All tests share
# one instance created at module import time.

_g = MPG()
_axis         = _g.add_system_root_at_scale(_g.scales.Axis, label=_g.labels.Axis.Shoot)
_growth_unit  = _g.add_component(_axis,        **PropsConfig(scale=_g.scales.GrowthUnit, edge_type='/', label=_g.labels.GrowthUnit.Shoot))
_phytomer     = _g.add_component(_growth_unit, **PropsConfig(scale=_g.scales.Phytomer,   edge_type='/', label=_g.labels.Phytomer.Shoot))
_internode    = _g.add_component(_phytomer,    **PropsConfig(scale=_g.scales.Organ,      edge_type='/', label=_g.labels.Organ.StemInternode))
_meristem     = _g.add_child(_internode,       **PropsConfig(scale=_g.scales.Organ,      edge_type='/', label=_g.labels.Organ.Meristem))
_leaf         = _g.add_child(_internode,       **PropsConfig(scale=_g.scales.Organ,      edge_type='/', label=_g.labels.Organ.Leaf))
_leafelement1 = _g.add_component(_leaf,        **PropsConfig(scale=_g.scales.SubOrgan,   edge_type='/', label=_g.labels.SubOrgan.LeafElement))
_leafelement2 = _g.add_child(_leafelement1,    **PropsConfig(scale=_g.scales.SubOrgan,   edge_type='<', label=_g.labels.SubOrgan.LeafElement))
_leafelement3 = _g.add_child(_leafelement2,    **PropsConfig(scale=_g.scales.SubOrgan,   edge_type='<', label=_g.labels.SubOrgan.LeafElement))


def test_partial_traversal():
    g = _g
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
    for vid in pre_order2(g, _leafelement1):
        print(vid, g.scales.translator[scale[vid]], g.labels.translator[label[vid]])

    print("\npost_order2 (finest scale from leafelement1)")
    for vid in post_order2(g, _leafelement1):
        print(vid, g.scales.translator[scale[vid]], g.labels.translator[label[vid]])


def test_combined_traversal():
    g = _g
    label = g.property('label')
    scale = g.property('scale')

    print("\npre_order_mpg  (complex → components, parent → children within scale)")
    for vid in g.pre_order_mpg():
        if vid != g.root:
            print(vid, g.scales.translator[scale[vid]], g.labels.translator[label[vid]])

    print("\npost_order_mpg  (children → parents within scale, components → complex)")
    for vid in g.post_order_mpg():
        if vid != g.root:
            print(vid, g.scales.translator[scale[vid]], g.labels.translator[label[vid]])


if __name__ == "__main__":
    test_partial_traversal()
    test_combined_traversal()
