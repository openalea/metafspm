"""
A synthetic cross-section with the GRANAP structure, to test anatomy mode with the Layer and Cell scales: Layers and
Cells are true MTG components of their segment, each Cell's protoplast a component of the Cell, and the walls and the
junction (shared by several cells) Compartments anchored to the segment.

Per SubOrgan segment:
    Layer cortex (CorticalParenchyma): Cells c0, c1, c2, their protoplasts Symplastic
    Layer xylem  (MetaXylem):          Cells x0, x1 (vessel_index 0, 1), their protoplasts labelled MetaXylem
    protoplasts kind 0; walls w0..w3 (Apoplastic, kind 1), junction j (Apoplastic, kind 2), anchored to the segment
    membranes (Transmembrane): c0-w0, c1-w0, c1-w1, c2-w1, c2-w2, x0-w2, x0-w3, x1-w3
    plasmodesmata (Symplastic): c0-c1, c1-c2
    wall paths (Apoplastic):    w0-j, w1-j
Junctions between a segment and its linked parent: xylem protoplasts of equal vessel_index.
"""
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.mpg import MPG

PROTOPLAST, WALL, JUNCTION = 0., 1., 2.
MEMBRANES = (("c0", "w0"), ("c1", "w0"), ("c1", "w1"), ("c2", "w1"), ("c2", "w2"), ("x0", "w2"), ("x0", "w3"),
             ("x1", "w3"))
PLASMODESMATA = (("c0", "c1"), ("c1", "c2"))
WALL_PATHS = (("w0", "j"), ("w1", "j"))


def wiring(g):
    return [dict(node_label=g.labels.Cell.MetaXylem, edge_label=g.labels.Connection.Apoplastic,
                 ordering="vessel_index", match="equal")]


def add_section(g, segment, bulk=True):
    """The Layers, Cells, protoplasts, walls, junction and Connections of *segment*; returns {name: vid}."""
    s, L = g.scales, g.labels
    node_anchor, edge_anchor = s.anchors[s.Compartment], s.anchors[s.Connection]
    v = {}
    def props(scale, **values):
        out = dict(PropsConfig(scale=scale, edge_type='/'))
        out.update(values)
        return out

    if bulk:
        cortex, xylem = g.add_components_bulk(segment, 2, **props(s.Layer, label=[L.Layer.CorticalParenchyma,
                                                                                  L.Layer.MetaXylem]))
        cells = g.add_components_bulk(cortex, 3, **props(s.Cell, label=L.Cell.CorticalParenchyma))
        cells += g.add_components_bulk(xylem, 2, **props(s.Cell, label=L.Cell.MetaXylem, vessel_index=[0., 1.]))
        protoplasts = g.add_components_bulk(cells[:3], 3, **props(s.Compartment, label=L.Compartment.Symplastic,
                                                                  kind=PROTOPLAST))
        protoplasts += g.add_components_bulk(cells[3:], 2, **props(s.Compartment, label=L.Cell.MetaXylem,
                                                                   vessel_index=[0., 1.], kind=PROTOPLAST))
        apoplast = g.add_components_bulk(node_anchor, 5, topo_parents=[segment] * 5,
                                         **props(s.Compartment, label=L.Compartment.Apoplastic,
                                                 kind=[WALL] * 4 + [JUNCTION]))
    else:
        def component(complex_, **props):
            return g.add_component(complex_, **PropsConfig(edge_type='/', **props))
        cortex, xylem = (component(segment, scale=s.Layer, label=label)
                         for label in (L.Layer.CorticalParenchyma, L.Layer.MetaXylem))
        cells = [component(cortex, scale=s.Cell, label=L.Cell.CorticalParenchyma) for _ in range(3)]
        cells += [component(xylem, scale=s.Cell, label=L.Cell.MetaXylem, vessel_index=float(i)) for i in range(2)]
        protoplasts = [component(c, scale=s.Compartment, label=L.Compartment.Symplastic, kind=PROTOPLAST)
                       for c in cells[:3]]
        protoplasts += [component(c, scale=s.Compartment, label=L.Cell.MetaXylem, vessel_index=float(i),
                                  kind=PROTOPLAST) for i, c in enumerate(cells[3:])]
        apoplast = [g.add_component_with_topo(node_anchor, segment, **PropsConfig(
            scale=s.Compartment, edge_type='/', label=L.Compartment.Apoplastic, kind=kind))
            for kind in [WALL] * 4 + [JUNCTION]]
    v.update(cortex=cortex, xylem=xylem)
    for name, cell, protoplast in zip(("c0", "c1", "c2", "x0", "x1"), cells, protoplasts):
        v["cell_" + name], v[name] = cell, protoplast
    v.update(zip(("w0", "w1", "w2", "w3", "j"), apoplast))
    links = ([(a, b, L.Connection.Transmembrane) for a, b in MEMBRANES]
             + [(a, b, L.Connection.Symplastic) for a, b in PLASMODESMATA]
             + [(a, b, L.Connection.Apoplastic) for a, b in WALL_PATHS])
    if bulk:
        g.add_components_bulk(edge_anchor, len(links), topo_parents=[segment] * len(links),
                              **props(s.Connection, n_id_a=[v[a] for a, _, _ in links],
                                      n_id_b=[v[b] for _, b, _ in links], label=[label for _, _, label in links]))
    else:
        for a, b, label in links:
            g.add_component_with_topo(edge_anchor, segment, **PropsConfig(scale=s.Connection, edge_type='/',
                                                                          label=label, n_id_a=v[a], n_id_b=v[b]))
    return v


def make_layered_anatomy(n_segments=2, bulk=True):
    """A chain of root segments with their sections, not yet wired; returns (g, segment vids, sections by segment)."""
    g = MPG()
    scale = g.scales.SubOrgan
    props = dict(label=g.labels.SubOrgan.RootSegment)
    segments = [g.add_system_root_at_scale(scale, **props)]
    for _ in range(n_segments - 1):
        segments.append(g.add_component_with_topo(g.scales.anchors[scale], segments[-1],
                                                  **PropsConfig(scale=scale, edge_type='<', **props)))
    sections = {segment: add_section(g, segment, bulk) for segment in segments}
    g.convert_properties_to_arraydict()
    return g, segments, sections


def grow_layered_segment(g, parent):
    """A segment below *parent*, with its section (as a growth and an anatomy generator would)."""
    segment = g.add_child(parent, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<',
                                                label=g.labels.SubOrgan.RootSegment))
    return segment, add_section(g, segment)
