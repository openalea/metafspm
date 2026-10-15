"""
Plant structures shared by the tests: a branched root system and a population of root chains, as MPGs.
"""
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.data_structure.mpg import MPG
from simple_seedling import generate_simple_mpg_seedling


def branched_root_system(n_axes=5, n_segments=40, laterals_every=4):
    """A seminal axis bearing laterals, segments at SubOrgan scale under one Organ per axis."""
    g = MPG()
    s = g.scales
    axis = g.add_system_root_at_scale(s.Axis, label=g.labels.Axis.Root)
    gu = g.add_component(axis, **PropsConfig(scale=s.GrowthUnit, edge_type='/', label=g.labels.GrowthUnit.Root))
    phytomer = g.add_component(gu, **PropsConfig(scale=s.Phytomer, edge_type='/', label=g.labels.Phytomer.Root))
    organ = g.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/', label=g.labels.Organ.RootInternode))
    main = [g.add_component(organ, **PropsConfig(scale=s.SubOrgan, edge_type='/', label=g.labels.SubOrgan.RootSegment))]
    for _ in range(n_segments - 1):
        main.append(g.add_child(main[-1], **PropsConfig(scale=s.SubOrgan, edge_type='<',
                                                        label=g.labels.SubOrgan.RootSegment)))
    for k in range(1, n_axes):
        lateral_organ = g.add_child(organ, **PropsConfig(scale=s.Organ, edge_type='+', label=g.labels.Organ.RootInternode))
        segment = g.add_component_with_topo(lateral_organ, main[k * laterals_every], **PropsConfig(
            scale=s.SubOrgan, edge_type='+', label=g.labels.SubOrgan.RootSegment))
        for _ in range(n_segments // 2):
            segment = g.add_child(segment, **PropsConfig(scale=s.SubOrgan, edge_type='<',
                                                         label=g.labels.SubOrgan.RootSegment))
    return g


def population(n_plants=3):
    """*n_plants* plants of 5 segments each in one MPG (not populated); returns (g, first segment of each plant)."""
    g = MPG()
    s = g.scales
    roots = []
    for _ in range(n_plants):
        plant = g.add_component(g.root, **PropsConfig(scale=s.Plant, edge_type='/'))
        axis = g.add_component(plant, **PropsConfig(scale=s.Axis, edge_type='/'))
        gu = g.add_component(axis, **PropsConfig(scale=s.GrowthUnit, edge_type='/'))
        ph = g.add_component(gu, **PropsConfig(scale=s.Phytomer, edge_type='/'))
        organ = g.add_component(ph, **PropsConfig(scale=s.Organ, edge_type='/'))
        last = g.add_component(organ, **PropsConfig(scale=s.SubOrgan, edge_type='/'))
        roots.append(last)
        for _ in range(4):
            last = g.add_child(last, **PropsConfig(scale=s.SubOrgan, edge_type='<'))
    return g, roots


def seedling_ds():
    """The simple seedling, populated at SubOrgan scale: (g, seedling, MPGDataStructure)."""
    g, seedling = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return g, seedling, MPGDataStructure(g, from_scale=g.scales.SubOrgan)


def grow_root(g, seedling):
    """Append one root segment below the seedling's root_segment6; returns its vid."""
    return g.add_child(seedling.root_segment6,
                       **PropsConfig(scale=g.scales.SubOrgan, edge_type='<', label=g.labels.SubOrgan.RootSegment))
