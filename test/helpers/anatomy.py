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


def wiring(g):
    return [
        dict(node_label=g.labels.Cell.MetaXylem, edge_label=g.labels.Connection.Apoplastic, ordering="vessel_index",
             match="equal"),
        dict(node_label=g.labels.Compartment.Symplastic, edge_label=g.labels.Connection.Symplastic),
    ]


def add_anatomy(g, segment):
    """The four Compartments of *segment* and their three anatomy Connections; returns the Compartment vids."""
    node_anchor = g.scales.anchors[g.scales.Compartment]
    edge_anchor = g.scales.anchors[g.scales.Connection]

    def compartment(label, **props):
        return g.add_component_with_topo(node_anchor, segment, **PropsConfig(scale=g.scales.Compartment, edge_type='/',
                                                                             label=label, **props))

    epidermis = compartment(g.labels.Compartment.Apoplastic)
    cortex = compartment(g.labels.Compartment.Symplastic)
    xylem = [compartment(g.labels.Cell.MetaXylem, vessel_index=float(i)) for i in range(2)]
    for a, b, label in ((cortex, epidermis, g.labels.Connection.Transmembrane),
                        (cortex, xylem[0], g.labels.Connection.Transmembrane),
                        (cortex, xylem[1], g.labels.Connection.Transmembrane)):
        g.add_component_with_topo(edge_anchor, segment, **PropsConfig(scale=g.scales.Connection, edge_type='/',
                                                                      label=label, n_id_a=a, n_id_b=b))
    return dict(epidermis=epidermis, cortex=cortex, xylem=xylem)


def make_rooted_anatomy(n_segments=3):
    """Chain of root segments with their anatomies, not yet wired; returns (g, segment vids, anatomies by segment)."""
    g = MPG()
    scale = g.scales.SubOrgan
    anchor = g.scales.anchors[scale]
    props = dict(label=g.labels.SubOrgan.RootSegment)
    segments = [g.add_system_root_at_scale(scale, **props)]
    for _ in range(n_segments - 1):
        segments.append(g.add_component_with_topo(anchor, segments[-1], **PropsConfig(scale=scale, edge_type='<',
                                                                                      **props)))
    anatomies = {segment: add_anatomy(g, segment) for segment in segments}
    g.convert_properties_to_arraydict()
    return g, segments, anatomies


def grow_segment(g, parent):
    """Add a segment below *parent*, with its anatomy (as a growth and an anatomy generator would)."""
    segment = g.add_child(parent, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<',
                                                label=g.labels.SubOrgan.RootSegment))
    return segment, add_anatomy(g, segment)
