from types import SimpleNamespace
from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.data_structure.configs import PropsConfig


def generate_simple_mpg_seedling():
    g = MPG()
    axis = g.add_system_root_at_scale(g.scales.Axis, label=g.labels.Axis.Shoot)
    growth_unit = g.add_component(axis, **PropsConfig(scale=g.scales.GrowthUnit, edge_type='/', label=g.labels.GrowthUnit.Shoot))
    # Add two consecutive phytomers
    phytomer = g.add_component(growth_unit, **PropsConfig(scale=g.scales.Phytomer, edge_type='/', label=g.labels.Phytomer.Shoot))
    internode = g.add_component(phytomer, **PropsConfig(scale=g.scales.Organ, edge_type='/', label=g.labels.Organ.StemInternode))
    meristem = g.add_child(internode, **PropsConfig(scale=g.scales.Organ, edge_type='/', label=g.labels.Organ.Meristem))
    leaf = g.add_child(internode, **PropsConfig(scale=g.scales.Organ, edge_type='/', label=g.labels.Organ.Leaf))
    leafelement1 = g.add_component(leaf, **PropsConfig(scale=g.scales.SubOrgan, edge_type='/', label=g.labels.SubOrgan.LeafElement))
    leafelement2 = g.add_child(leafelement1, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<', label=g.labels.SubOrgan.LeafElement))
    leafelement3 = g.add_child(leafelement2, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<', label=g.labels.SubOrgan.LeafElement))
    # Second phytomer
    phytomer2 = g.add_child(phytomer, **PropsConfig(scale=g.scales.Phytomer, edge_type='<', label=g.labels.Phytomer.Shoot))
    internode2 = g.add_component(phytomer2, **PropsConfig(scale=g.scales.Organ, edge_type='/', label=g.labels.Organ.StemInternode))
    meristem2 = g.add_child(internode2, **PropsConfig(scale=g.scales.Organ, edge_type='/', label=g.labels.Organ.Meristem))
    leaf2 = g.add_child(internode2, **PropsConfig(scale=g.scales.Organ, edge_type='/', label=g.labels.Organ.Leaf))
    leafelement4 = g.add_component(leaf2, **PropsConfig(scale=g.scales.SubOrgan, edge_type='/', label=g.labels.SubOrgan.LeafElement))
    leafelement5 = g.add_child(leafelement4, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<', label=g.labels.SubOrgan.LeafElement))
    leafelement6 = g.add_child(leafelement5, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<', label=g.labels.SubOrgan.LeafElement))

    # Single ramified root axis
    root_axis = g.add_child(axis, **PropsConfig(scale=g.scales.Axis, edge_type="+", label=g.labels.Axis.Root))
    growth_unit_root = g.add_component(root_axis, **PropsConfig(scale=g.scales.GrowthUnit, edge_type='/', label=g.labels.GrowthUnit.Root))
    phytomer_root = g.add_component(growth_unit_root, **PropsConfig(scale=g.scales.Phytomer, edge_type='/', label=g.labels.Phytomer.Root))
    root_internode1 = g.add_component(phytomer_root, **PropsConfig(scale=g.scales.Organ, edge_type='/', label=g.labels.Organ.RootInternode))
    root_segment1 = g.add_component(root_internode1, **PropsConfig(scale=g.scales.SubOrgan, edge_type='/', label=g.labels.SubOrgan.RootSegment))
    root_segment2 = g.add_child(root_segment1, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<', label=g.labels.SubOrgan.RootSegment))
    root_segment3 = g.add_child(root_segment2, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<', label=g.labels.SubOrgan.RootSegment))
    root_internode2 = g.add_child(root_internode1, **PropsConfig(scale=g.scales.Organ, edge_type='+', label=g.labels.Organ.RootInternode))
    # WARNING: Special case where we don't want a root segment to inherit from its parent's complex
    # root_segment4 belongs to root_internode2 (complex) but physically emerges from root_segment2
    # (topological parent at SubOrgan scale) — multiscale branching.
    root_segment4 = g.add_lateral_component(root_internode2, root_segment2, **PropsConfig(scale=g.scales.SubOrgan, edge_type='+', label=g.labels.SubOrgan.RootSegment))
    root_segment5 = g.add_child(root_segment4, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<', label=g.labels.SubOrgan.RootSegment))
    root_segment6 = g.add_child(root_segment5, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<', label=g.labels.SubOrgan.RootSegment))

    assert g.parent(root_segment4) == root_segment2,  "topo parent must be root_segment2"
    assert g.complex(root_segment4) == root_internode2, "complex must be root_internode2"
    assert root_segment4 in list(g.components_iter(root_internode2)), "must be component of root_internode2"
    assert root_segment4 not in list(g.components_iter(root_internode1)), "must NOT be component of root_internode1"

    return SimpleNamespace(**locals())


# Module-level shared instance: import _s from this module to access g and all vertex IDs.
# Note: LabelsConfig mutates class attributes on the first MPG() construction; this call
# must happen before any other MPG() in the test session.  Place this import before
# test_mpg_traversal if both are used.
seedling = generate_simple_mpg_seedling()
