"""
Plant structures shared by the tests: a branched root system and a population of root chains, as MPGs.
"""
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.data_structure.mpg import MPG
from simple_seedling import generate_simple_mpg_seedling


def branched_root_system(n_axes=5, n_segments=40, laterals_every=4):
    """A seminal axis bearing laterals, segments at SubOrgan scale under one Organ per axis."""
    mtg = MPG()
    s = mtg.scales
    axis = mtg._add_system_root_at_scale(s.Axis, label=mtg.labels.Axis.Root)
    gu = mtg.add_component(axis, **PropsConfig(scale=s.GrowthUnit, edge_type='/', label=mtg.labels.GrowthUnit.Root))
    phytomer = mtg.add_component(gu, **PropsConfig(scale=s.Phytomer, edge_type='/', label=mtg.labels.Phytomer.Root))
    organ = mtg.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/', label=mtg.labels.Organ.RootInternode))
    main = [mtg.add_component(organ, **PropsConfig(scale=s.SubOrgan, edge_type='/', label=mtg.labels.SubOrgan.RootSegment))]
    for _ in range(n_segments - 1):
        main.append(mtg.add_child(main[-1], **PropsConfig(scale=s.SubOrgan, edge_type='<',
                                                        label=mtg.labels.SubOrgan.RootSegment)))
    for k in range(1, n_axes):
        lateral_organ = mtg.add_child(organ, **PropsConfig(scale=s.Organ, edge_type='+', label=mtg.labels.Organ.RootInternode))
        segment = mtg.add_component_with_topo(lateral_organ, main[k * laterals_every], **PropsConfig(
            scale=s.SubOrgan, edge_type='+', label=mtg.labels.SubOrgan.RootSegment))
        for _ in range(n_segments // 2):
            segment = mtg.add_child(segment, **PropsConfig(scale=s.SubOrgan, edge_type='<',
                                                         label=mtg.labels.SubOrgan.RootSegment))
    return mtg


def population(n_plants=3):
    """*n_plants* plants of 5 segments each in one MPG (not populated); returns (mtg, first segment of each plant)."""
    mtg = MPG()
    s = mtg.scales
    roots = []
    for _ in range(n_plants):
        plant = mtg.add_component(mtg.root, **PropsConfig(scale=s.Plant, edge_type='/'))
        axis = mtg.add_component(plant, **PropsConfig(scale=s.Axis, edge_type='/'))
        gu = mtg.add_component(axis, **PropsConfig(scale=s.GrowthUnit, edge_type='/'))
        ph = mtg.add_component(gu, **PropsConfig(scale=s.Phytomer, edge_type='/'))
        organ = mtg.add_component(ph, **PropsConfig(scale=s.Organ, edge_type='/'))
        last = mtg.add_component(organ, **PropsConfig(scale=s.SubOrgan, edge_type='/'))
        roots.append(last)
        for _ in range(4):
            last = mtg.add_child(last, **PropsConfig(scale=s.SubOrgan, edge_type='<'))
    return mtg, roots


def seedling_ds():
    """The simple seedling, populated at SubOrgan scale: (mtg, seedling, MPGDataStructure)."""
    mtg, seedling = generate_simple_mpg_seedling()
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    return mtg, seedling, MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)


def grow_root(mtg, seedling):
    """Append one root segment below the seedling's root_segment6; returns its vid."""
    return mtg.add_child(seedling.root_segment6,
                       **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<', label=mtg.labels.SubOrgan.RootSegment))
