"""
A synthetic cross-section with the GRANAP structure, to test anatomy mode with the Layer and Cell scales: Layers and
Cells are true MTG components of their segment, each Cell's protoplast a component of the Cell, and the walls and the
junction (shared by several cells) Compartments anchored to the segment.

Per SubOrgan segment:
    Layer cortex (CorticalParenchyma): Cells c0, c1, c2, their protoplasts Symplastic
    Layer xylem  (MetaXylem):          Cells x0, x1 (vessel_index 0, 1), their protoplasts labelled MetaXylem
    protoplasts compartment_type 0; walls w0..w3 (Apoplastic, compartment_type 1), junction j (Apoplastic, compartment_type 2), anchored to the segment
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


def wiring(mtg):
    return [dict(node_label=mtg.labels.Cell.MetaXylem, edge_label=mtg.labels.Connection.Apoplastic,
                 ordering="vessel_index", match="equal")]


def add_section(mtg, segment, bulk=True):
    """The Layers, Cells, protoplasts, walls, junction and Connections of *segment*; returns {name: vid}."""
    s, L = mtg.scales, mtg.labels
    node_anchor, edge_anchor = s.anchors[s.Compartment], s.anchors[s.Connection]
    v = {}
    def props(scale, **values):
        out = dict(PropsConfig(scale=scale, edge_type='/'))
        out.update(values)
        return out

    if bulk:
        cortex, xylem = mtg.add_components_bulk(segment, 2, **props(s.Layer, label=[L.Layer.CorticalParenchyma,
                                                                                  L.Layer.MetaXylem]))
        cells = mtg.add_components_bulk(cortex, 3, **props(s.Cell, label=L.Cell.CorticalParenchyma))
        cells += mtg.add_components_bulk(xylem, 2, **props(s.Cell, label=L.Cell.MetaXylem, vessel_index=[0., 1.]))
        protoplasts = mtg.add_components_bulk(cells[:3], 3, **props(s.Compartment, label=L.Compartment.Symplastic,
                                                                  compartment_type=PROTOPLAST))
        protoplasts += mtg.add_components_bulk(cells[3:], 2, **props(s.Compartment, label=L.Cell.MetaXylem,
                                                                   vessel_index=[0., 1.], compartment_type=PROTOPLAST))
        apoplast = mtg.add_components_bulk(node_anchor, 5, topo_parents=[segment] * 5,
                                         **props(s.Compartment, label=L.Compartment.Apoplastic,
                                                 compartment_type=[WALL] * 4 + [JUNCTION]))
    else:
        def component(complex_, **props):
            return mtg.add_component(complex_, **PropsConfig(edge_type='/', **props))
        cortex, xylem = (component(segment, scale=s.Layer, label=label)
                         for label in (L.Layer.CorticalParenchyma, L.Layer.MetaXylem))
        cells = [component(cortex, scale=s.Cell, label=L.Cell.CorticalParenchyma) for _ in range(3)]
        cells += [component(xylem, scale=s.Cell, label=L.Cell.MetaXylem, vessel_index=float(i)) for i in range(2)]
        protoplasts = [component(c, scale=s.Compartment, label=L.Compartment.Symplastic, compartment_type=PROTOPLAST)
                       for c in cells[:3]]
        protoplasts += [component(c, scale=s.Compartment, label=L.Cell.MetaXylem, vessel_index=float(i),
                                  compartment_type=PROTOPLAST) for i, c in enumerate(cells[3:])]
        apoplast = [mtg.add_component_with_topo(node_anchor, segment, **PropsConfig(
            scale=s.Compartment, edge_type='/', label=L.Compartment.Apoplastic, compartment_type=compartment_type))
            for compartment_type in [WALL] * 4 + [JUNCTION]]
    v.update(cortex=cortex, xylem=xylem)
    for name, cell, protoplast in zip(("c0", "c1", "c2", "x0", "x1"), cells, protoplasts):
        v["cell_" + name], v[name] = cell, protoplast
    v.update(zip(("w0", "w1", "w2", "w3", "j"), apoplast))
    links = ([(a, b, L.Connection.Transmembrane) for a, b in MEMBRANES]
             + [(a, b, L.Connection.Symplastic) for a, b in PLASMODESMATA]
             + [(a, b, L.Connection.Apoplastic) for a, b in WALL_PATHS])
    if bulk:
        mtg.add_components_bulk(edge_anchor, len(links), topo_parents=[segment] * len(links),
                              **props(s.Connection, n_id_a=[v[a] for a, _, _ in links],
                                      n_id_b=[v[b] for _, b, _ in links], label=[label for _, _, label in links]))
    else:
        for a, b, label in links:
            mtg.add_component_with_topo(edge_anchor, segment, **PropsConfig(scale=s.Connection, edge_type='/',
                                                                          label=label, n_id_a=v[a], n_id_b=v[b]))
    return v


def make_layered_anatomy(n_segments=2, bulk=True):
    """A chain of root segments with their sections, not yet wired; returns (mtg, segment vids, sections by segment)."""
    mtg = MPG()
    scale = mtg.scales.SubOrgan
    props = dict(label=mtg.labels.SubOrgan.RootSegment)
    segments = [mtg._add_system_root_at_scale(scale, **props)]
    for _ in range(n_segments - 1):
        segments.append(mtg.add_component_with_topo(mtg.scales.anchors[scale], segments[-1],
                                                  **PropsConfig(scale=scale, edge_type='<', **props)))
    sections = {segment: add_section(mtg, segment, bulk) for segment in segments}
    mtg.convert_properties_to_arraydict()
    return mtg, segments, sections


def grow_layered_segment(mtg, parent):
    """A segment below *parent*, with its section (as a growth and an anatomy generator would)."""
    segment = mtg.add_child(parent, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<',
                                                label=mtg.labels.SubOrgan.RootSegment))
    return segment, add_section(mtg, segment)


def section_template(key="toy section"):
    """The SectionTemplate of add_section's section, captured from one copy in a scratch MPG; and each named
    vertex's local index in it."""
    from openalea.metafspm.data_structure.instances import SectionTemplate
    scratch = MPG()
    segment = scratch._add_system_root_at_scale(scratch.scales.SubOrgan, label=scratch.labels.SubOrgan.RootSegment)
    first = scratch._id + 1
    names = add_section(scratch, segment)
    scratch.convert_properties_to_arraydict()
    template = SectionTemplate.capture(scratch, segment, range(first, scratch._id + 1), key=key)
    return template, {name: vid - first for name, vid in names.items()}


_TEMPLATE = None


def add_instanced_section(mtg, segment):
    """add_section as an instance of the section's template: the same vids, no MTG vertex; returns {name: vid}."""
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = section_template()
    template, local = _TEMPLATE
    vids = mtg.add_instance(segment, template)
    return {name: vids[index] for name, index in local.items()}


def make_instanced_anatomy(n_segments=2):
    """make_layered_anatomy with instanced sections."""
    mtg = MPG()
    scale = mtg.scales.SubOrgan
    props = dict(label=mtg.labels.SubOrgan.RootSegment)
    segments = [mtg._add_system_root_at_scale(scale, **props)]
    for _ in range(n_segments - 1):
        segments.append(mtg.add_component_with_topo(mtg.scales.anchors[scale], segments[-1],
                                                  **PropsConfig(scale=scale, edge_type='<', **props)))
    sections = {segment: add_instanced_section(mtg, segment) for segment in segments}
    mtg.convert_properties_to_arraydict()
    return mtg, segments, sections


def grow_instanced_segment(mtg, parent):
    segment = mtg.add_child(parent, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<',
                                                label=mtg.labels.SubOrgan.RootSegment))
    return segment, add_instanced_section(mtg, segment)
