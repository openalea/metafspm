"""
The seedling generator of the example: the architecture of a seedling at every scale (Plant, Axis, GrowthUnit,
Phytomer, Organ, SubOrgan), an anatomy per segment (root: epidermis, cortex, endodermis, xylem; stem: epidermis,
cortex, xylem; leaf: xylem, mesophyll, stomatal cavity) and the axial junctions between the xylems of linked segments,
with the codes the components read (tissue, organ, hydraulic type). It stands for the structural models (growth,
anatomy) a real simulation would couple; SeedlingStructure.initiate_plant calls it.
"""
import numpy as np

from openalea.metafspm.data_structure.configs import PropsConfig


# Codes written on the plant structure
ROOT, STEM, LEAF = 1, 2, 3                                          # organ of a Compartment
EPIDERMIS, CORTEX, ENDODERMIS, XYLEM, MESOPHYLL, STOMATAL_CAVITY = 1, 2, 3, 4, 5, 6   # tissue of a Compartment
ROOT_RADIAL, STEM_RADIAL, LEAF_RADIAL, AXIAL = 1, 2, 3, 4           # hydraulic type of a Connection

ANATOMY = {                                                         # tissues from the outside in (leaf: from xylem out)
    ROOT: (EPIDERMIS, CORTEX, ENDODERMIS, XYLEM),
    STEM: (EPIDERMIS, CORTEX, XYLEM),
    LEAF: (XYLEM, MESOPHYLL, STOMATAL_CAVITY),
}
RADIAL_TYPE = {ROOT: ROOT_RADIAL, STEM: STEM_RADIAL, LEAF: LEAF_RADIAL}


def _tissue_label(labels, tissue):
    return {EPIDERMIS: labels.Layer.Epidermis, CORTEX: labels.Layer.CorticalParenchyma,
            ENDODERMIS: labels.Layer.Endodermis, XYLEM: labels.Layer.MetaXylem, MESOPHYLL: labels.Layer.Mesophyll,
            STOMATAL_CAVITY: labels.Layer.StomatalCavity}[tissue]


def build_seedling(g, plant, parameters):
    """
    The seedling of *plant*: a shoot of n_phytomers (a stem element and a leaf of n_leaf_elements each) and
    n_root_axes first-order roots from the collar, each with a lateral: a nearly vertical pivot, and others bending
    from emergence_angle below the horizontal towards the vertical (gravitropism, per segment); every segment with
    its anatomy.
    """
    p = dict(n_phytomers=3, n_leaf_elements=3, n_root_axes=3, n_root_segments=6, n_lateral_segments=3,
             stem_length=0.02, leaf_length=0.03, leaf_width=0.005, root_length=0.025, lateral_length=0.015,
             emergence_angle=15., gravitropism=0.35, lateral_gravitropism=0.08, tortuosity=0.08,
             pivot_emergence_angle=80., pivot_gravitropism=0.8, n_pivot_segments=8)
    p.update({key: value for key, value in parameters.items() if key in p})
    s, labels = g.scales, g.labels
    x0, y0, z0 = (float(g.property(name)[plant]) for name in ("x", "y", "z"))
    rotation = float(g.property("rotation").get(plant, 0.))
    segments = {}                                                # vid -> (organ, length, start, end)

    def segment(vid, organ, length, start, direction):
        end = np.asarray(start) + length * np.asarray(direction) / np.linalg.norm(direction)
        segments[vid] = (organ, length, np.asarray(start, dtype=float), end)
        return end

    # shoot
    shoot = g.add_component(plant, **PropsConfig(scale=s.Axis, edge_type='/', label=labels.Axis.Shoot))
    unit = g.add_component(shoot, **PropsConfig(scale=s.GrowthUnit, edge_type='/', label=labels.GrowthUnit.Shoot))
    top, phytomer, stem_element, collar = (x0, y0, z0), None, None, None
    for rank in range(int(p["n_phytomers"])):
        if phytomer is None:
            phytomer = g.add_component(unit, **PropsConfig(scale=s.Phytomer, edge_type='/',
                                                           label=labels.Phytomer.Shoot))
        else:
            phytomer = g.add_child(phytomer, **PropsConfig(scale=s.Phytomer, edge_type='<',
                                                           label=labels.Phytomer.Shoot))
        internode = g.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/',
                                                             label=labels.Organ.StemInternode))
        props = PropsConfig(scale=s.SubOrgan, edge_type='<' if stem_element else '/',
                            label=labels.SubOrgan.StemElement)
        stem_element = (g.add_component_with_topo(internode, stem_element, **props) if stem_element
                        else g.add_component(internode, **props))
        collar = collar or stem_element
        top = segment(stem_element, STEM, p["stem_length"], top, (0., 0., 1.))
        leaf = g.add_child(internode, **PropsConfig(scale=s.Organ, edge_type='+', label=labels.Organ.Leaf))
        azimuth = rotation + rank * 2. * np.pi / 3.
        start, parent = top, stem_element
        for element in range(int(p["n_leaf_elements"])):
            props = PropsConfig(scale=s.SubOrgan, edge_type='+' if element == 0 else '<',
                                label=labels.SubOrgan.LeafElement)
            vid = (g.add_component_with_topo(leaf, parent, **props) if element == 0
                   else g.add_child(parent, **props))
            start = segment(vid, LEAF, p["leaf_length"], start,
                            (np.cos(azimuth), np.sin(azimuth), 0.6 - 0.4 * element))
            parent = vid

    # roots: three first-order roots from the collar, each with one lateral. The first is a pivot, nearly vertical
    # (pivot_emergence_angle, pivot_gravitropism, n_pivot_segments); the others start emergence_angle (degrees) below
    # the horizontal and bend towards the vertical segment after segment, their direction pulled down by the
    # gravitropism coefficient. All have a small random tortuosity (reproducible per plant).
    rng = np.random.default_rng(int(plant))

    def tropism(direction, coefficient):
        bent = np.asarray(direction) + coefficient * np.array([0., 0., -1.]) + rng.normal(0., p["tortuosity"], 3)
        return bent / np.linalg.norm(bent)

    for axis_rank in range(int(p["n_root_axes"])):
        axis = g.add_child(shoot, **PropsConfig(scale=s.Axis, edge_type='+', label=labels.Axis.Root))
        unit_r = g.add_component(axis, **PropsConfig(scale=s.GrowthUnit, edge_type='/',
                                                     label=labels.GrowthUnit.Root))
        phytomer_r = g.add_component(unit_r, **PropsConfig(scale=s.Phytomer, edge_type='/',
                                                           label=labels.Phytomer.Root))
        organ = g.add_component(phytomer_r, **PropsConfig(scale=s.Organ, edge_type='/',
                                                          label=labels.Organ.RootInternode))
        pivot = axis_rank == 0
        others = max(int(p["n_root_axes"]) - 1, 1)
        azimuth = rotation + (axis_rank - 1) * 2. * np.pi / others + rng.normal(0., 0.25)
        dip = np.radians(p["pivot_emergence_angle"] if pivot else p["emergence_angle"]) + rng.normal(0., 0.1)
        gravitropism = p["pivot_gravitropism"] if pivot else p["gravitropism"]
        direction = np.array([np.cos(azimuth) * np.cos(dip), np.sin(azimuth) * np.cos(dip), -np.sin(dip)])
        start, parent, axis_segments = (x0, y0, z0), collar, []
        for rank in range(int(p["n_pivot_segments"] if pivot else p["n_root_segments"])):
            props = PropsConfig(scale=s.SubOrgan, edge_type='+' if rank == 0 else '<',
                                label=labels.SubOrgan.RootSegment)
            vid = (g.add_component_with_topo(organ, parent, **props) if rank == 0 else g.add_child(parent, **props))
            start = segment(vid, ROOT, p["root_length"], start, direction)
            direction = tropism(direction, gravitropism)
            parent = vid
            axis_segments.append(vid)
        bearer = axis_segments[min(2, len(axis_segments) - 1)]
        lateral = g.add_child(organ, **PropsConfig(scale=s.Organ, edge_type='+', label=labels.Organ.RootInternode))
        side = azimuth + np.pi / 2. + rng.normal(0., 0.3)
        direction = np.array([np.cos(side), np.sin(side), -0.2])
        direction /= np.linalg.norm(direction)
        start, parent = segments[bearer][3], bearer
        for rank in range(int(p["n_lateral_segments"])):
            props = PropsConfig(scale=s.SubOrgan, edge_type='+' if rank == 0 else '<',
                                label=labels.SubOrgan.RootSegment)
            vid = (g.add_component_with_topo(lateral, parent, **props) if rank == 0
                   else g.add_child(parent, **props))
            start = segment(vid, ROOT, p["lateral_length"], start, direction)
            direction = tropism(direction, p["lateral_gravitropism"])
            parent = vid

    # segment properties, anatomies and axial junctions
    for name in ("length", "x1", "x2", "y1", "y2", "z1", "z2"):
        g.properties().setdefault(name, {})
    xylem = {}
    for vid, (organ, length, start, end) in segments.items():
        for name, value in (("length", length), ("x1", start[0]), ("x2", end[0]), ("y1", start[1]),
                            ("y2", end[1]), ("z1", start[2]), ("z2", end[2])):
            g.property(name)[vid] = float(value)
        xylem[vid] = _anatomy(g, vid, organ, length, p["leaf_width"])
    for vid in segments:
        parent = g.parent(vid)
        if parent in xylem:
            _connect(g, vid, xylem[parent], xylem[vid], AXIAL,
                         (segments[parent][1] + segments[vid][1]) / 2., junction=True)

def _anatomy(g, segment, organ, length, leaf_width):
    """The Compartments of *segment* and the radial Connections between them; returns its xylem."""
    node_anchor = g.scales.anchors[g.scales.Compartment]
    tissues = ANATOMY[organ]
    compartments = []
    for tissue in tissues:
        evaporating = organ == LEAF and tissue == STOMATAL_CAVITY
        contact = organ == ROOT and tissue == EPIDERMIS
        compartments.append(g.add_component_with_topo(node_anchor, segment, **PropsConfig(
            scale=g.scales.Compartment, edge_type='/', label=_tissue_label(g.labels, tissue),
            tissue=float(tissue), organ=float(organ), is_evaporating=float(evaporating),
            evaporating_area=leaf_width * length if evaporating else 0., is_soil_contact=float(contact),
            contact_length=length if contact else 0.)))
    for outer, inner in zip(compartments[:-1], compartments[1:]):
        _connect(g, segment, outer, inner, RADIAL_TYPE[organ], length)
    return compartments[tissues.index(XYLEM)]

def _connect(g, segment, a, b, hydraulic_type, conductance_length, junction=False):
    g.add_component_with_topo(g.scales.anchors[g.scales.Connection], segment, **PropsConfig(
        scale=g.scales.Connection, edge_type='/',
        label=g.labels.Connection.Apoplastic if junction else g.labels.Connection.Transmembrane,
        n_id_a=a, n_id_b=b, is_junction=int(junction), hydraulic_type=float(hydraulic_type),
        conductance_length=float(conductance_length)))
