from types import SimpleNamespace
from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.data_structure.configs import PropsConfig


def generate_simple_mpg_seedling():
    mtg = MPG()
    axis = mtg._add_system_root_at_scale(mtg.scales.Axis, label=mtg.labels.Axis.Shoot)
    growth_unit = mtg.add_component(axis, **PropsConfig(scale=mtg.scales.GrowthUnit, edge_type='/', label=mtg.labels.GrowthUnit.Shoot))
    # Add two consecutive phytomers
    phytomer = mtg.add_component(growth_unit, **PropsConfig(scale=mtg.scales.Phytomer, edge_type='/', label=mtg.labels.Phytomer.Shoot))
    internode = mtg.add_component(phytomer, **PropsConfig(scale=mtg.scales.Organ, edge_type='/', label=mtg.labels.Organ.StemInternode))
    internodeelement = mtg.add_component(internode, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='/', label=mtg.labels.SubOrgan.StemElement))
    meristem = mtg.add_child(internode, **PropsConfig(scale=mtg.scales.Organ, edge_type='/', label=mtg.labels.Organ.Meristem))
    leaf = mtg.add_child(internode, **PropsConfig(scale=mtg.scales.Organ, edge_type='/', label=mtg.labels.Organ.Leaf))
    leafelement1 = mtg.add_component_with_topo(leaf, internodeelement, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='+', label=mtg.labels.SubOrgan.LeafElement))
    leafelement2 = mtg.add_child(leafelement1, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<', label=mtg.labels.SubOrgan.LeafElement))
    leafelement3 = mtg.add_child(leafelement2, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<', label=mtg.labels.SubOrgan.LeafElement))
    # Second phytomer
    phytomer2 = mtg.add_child(phytomer, **PropsConfig(scale=mtg.scales.Phytomer, edge_type='<', label=mtg.labels.Phytomer.Shoot))
    internode2 = mtg.add_component(phytomer2, **PropsConfig(scale=mtg.scales.Organ, edge_type='/', label=mtg.labels.Organ.StemInternode))
    internodeelement2 = mtg.add_component_with_topo(internode2, internodeelement, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<', label=mtg.labels.SubOrgan.StemElement))
    meristem2 = mtg.add_child(internode2, **PropsConfig(scale=mtg.scales.Organ, edge_type='/', label=mtg.labels.Organ.Meristem))
    leaf2 = mtg.add_child(internode2, **PropsConfig(scale=mtg.scales.Organ, edge_type='/', label=mtg.labels.Organ.Leaf))
    leafelement4 = mtg.add_component_with_topo(leaf2, internodeelement2, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='+', label=mtg.labels.SubOrgan.LeafElement))
    leafelement5 = mtg.add_child(leafelement4, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<', label=mtg.labels.SubOrgan.LeafElement))
    leafelement6 = mtg.add_child(leafelement5, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<', label=mtg.labels.SubOrgan.LeafElement))

    # Single ramified root axis
    root_axis = mtg.add_child(axis, **PropsConfig(scale=mtg.scales.Axis, edge_type="+", label=mtg.labels.Axis.Root))
    growth_unit_root = mtg.add_component(root_axis, **PropsConfig(scale=mtg.scales.GrowthUnit, edge_type='/', label=mtg.labels.GrowthUnit.Root))
    phytomer_root = mtg.add_component(growth_unit_root, **PropsConfig(scale=mtg.scales.Phytomer, edge_type='/', label=mtg.labels.Phytomer.Root))
    root_internode1 = mtg.add_component(phytomer_root, **PropsConfig(scale=mtg.scales.Organ, edge_type='/', label=mtg.labels.Organ.RootInternode))
    root_segment1 = mtg.add_component_with_topo(root_internode1, internodeelement, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='+', label=mtg.labels.SubOrgan.RootSegment))
    root_segment2 = mtg.add_child(root_segment1, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<', label=mtg.labels.SubOrgan.RootSegment))
    root_segment3 = mtg.add_child(root_segment2, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<', label=mtg.labels.SubOrgan.RootSegment))
    root_internode2 = mtg.add_child(root_internode1, **PropsConfig(scale=mtg.scales.Organ, edge_type='+', label=mtg.labels.Organ.RootInternode))
    # WARNING: Special case where we don't want a root segment to inherit from its parent's complex
    # root_segment4 belongs to root_internode2 (complex) but physically emerges from root_segment2
    # (topological parent at SubOrgan scale) — multiscale branching.
    root_segment4 = mtg.add_component_with_topo(root_internode2, root_segment2, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='+', label=mtg.labels.SubOrgan.RootSegment))
    root_segment5 = mtg.add_child(root_segment4, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<', label=mtg.labels.SubOrgan.RootSegment))
    root_segment6 = mtg.add_child(root_segment5, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<', label=mtg.labels.SubOrgan.RootSegment))

    assert mtg.parent(root_segment4) == root_segment2,  "topo parent must be root_segment2"
    assert mtg.complex(root_segment4) == root_internode2, "complex must be root_internode2"
    assert root_segment4 in list(mtg.components_iter(root_internode2)), "must be component of root_internode2"
    assert root_segment4 not in list(mtg.components_iter(root_internode1)), "must NOT be component of root_internode1"

    return mtg, SimpleNamespace(**locals())


# Module-level shared instance: import _s from this module to access g and all vertex IDs.
# Note: LabelsConfig mutates class attributes on the first MPG() construction; this call
# must happen before any other MPG() in the test session.  Place this import before
# test_mpg_traversal if both are used.
mtg, seedling = generate_simple_mpg_seedling()
