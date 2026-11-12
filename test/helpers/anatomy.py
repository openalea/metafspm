"""
A small synthetic anatomy per root segment, to test the anatomy mode of MPGDataStructure. Not a plant anatomy: four Compartments per segment and a star of anatomy
Connections around the cortex, enough to exercise the multiscale graph assembly.

Per SubOrgan segment:
    epidermis (Apoplastic) -- cortex (Symplastic) -- xylem vessel 0 (MetaXylem, vessel_index 0)
                                                   -- xylem vessel 1 (MetaXylem, vessel_index 1)
Junctions between a segment and its linked parent (MPG.wire_junctions):
    xylem to xylem of equal vessel_index, cortex to cortex.
"""
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.mpg import MPG


def wiring(mtg):
    return [
        dict(node_label=mtg.labels.Cell.MetaXylem, edge_label=mtg.labels.Connection.Apoplastic, ordering="vessel_index",
             match="equal"),
        dict(node_label=mtg.labels.Compartment.Symplastic, edge_label=mtg.labels.Connection.Symplastic),
    ]


def add_anatomy(mtg, segment):
    """The four Compartments of *segment* and their three anatomy Connections; returns the Compartment vids."""
    node_anchor = mtg.scales.anchors[mtg.scales.Compartment]
    edge_anchor = mtg.scales.anchors[mtg.scales.Connection]

    def compartment(label, **props):
        return mtg.add_component_with_topo(node_anchor, segment, **PropsConfig(scale=mtg.scales.Compartment, edge_type='/',
                                                                             label=label, **props))

    epidermis = compartment(mtg.labels.Compartment.Apoplastic)
    cortex = compartment(mtg.labels.Compartment.Symplastic)
    xylem = [compartment(mtg.labels.Cell.MetaXylem, vessel_index=float(i)) for i in range(2)]
    for a, b, label in ((cortex, epidermis, mtg.labels.Connection.Transmembrane),
                        (cortex, xylem[0], mtg.labels.Connection.Transmembrane),
                        (cortex, xylem[1], mtg.labels.Connection.Transmembrane)):
        mtg.add_component_with_topo(edge_anchor, segment, **PropsConfig(scale=mtg.scales.Connection, edge_type='/',
                                                                      label=label, n_id_a=a, n_id_b=b))
    return dict(epidermis=epidermis, cortex=cortex, xylem=xylem)


def make_rooted_anatomy(n_segments=3):
    """Chain of root segments with their anatomies, not yet wired; returns (mtg, segment vids, anatomies by segment)."""
    mtg = MPG()
    scale = mtg.scales.SubOrgan
    anchor = mtg.scales.anchors[scale]
    props = dict(label=mtg.labels.SubOrgan.RootSegment)
    segments = [mtg._add_system_root_at_scale(scale, **props)]
    for _ in range(n_segments - 1):
        segments.append(mtg.add_component_with_topo(anchor, segments[-1], **PropsConfig(scale=scale, edge_type='<',
                                                                                      **props)))
    anatomies = {segment: add_anatomy(mtg, segment) for segment in segments}
    mtg.convert_properties_to_arraydict()
    return mtg, segments, anatomies


def grow_segment(mtg, parent):
    """Add a segment below *parent*, with its anatomy (as a growth and an anatomy generator would)."""
    segment = mtg.add_child(parent, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<',
                                                label=mtg.labels.SubOrgan.RootSegment))
    return segment, add_anatomy(mtg, segment)
