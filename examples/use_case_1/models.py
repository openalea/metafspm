"""
The models of use case 1 (a wheat plant in its soil and its light): the plant population model (WheatPlant), its
environment models (the soil, component.RichardsSoilModel, and the light, Caribu's CaribuLightModel) and the
translators linking their components.

    WheatPlant          the plant, its parts: the seed and the subcrown internode (WheatCrown), Adel's shoot
                        (ScenarioShoot, nested) on the crown's top, ArchiSimple's roots on the seed and on the
                        phytomers the nodal root rule asks (NodalRootEmergence), GRANAP's anatomy on every section,
                        the roots' barriers (MECHA's Development), the elements' optics (Caribu's PlantOptics), their
                        gas exchange (CNW-Grass's GasExchange) and the water flow through the whole plant (MECHA's
                        MechaHydraulics), on one MPG whose nodes are the Compartments
    WHEAT_SCENARIO      the plant's scenario: the crown, Adel's blades (CN-Wheat's leaf shapes, LeafShapes.cnwheat:
                        CNW-Grass's stand's), ArchiSimple's wheat on a shoot and its anchor,
                        GRANAP's anatomies (a wheat root, the monocot stem, sheath and leaf), the barriers' zones,
                        MECHA's soil boundary on each root section's soil node, the sections reduced onto their
                        conduits
    PLANT_TRANSLATOR    within the plant: GRANAP's stems classed by their organ's diameter, the gas exchange's element
                        heights and green areas of their parts above the soil (CNW-Grass's emerged_links: from
                        Adel's triangles above the Scene's soil surface, its green area under a receiver-private
                        name, Adel's untouched), the elements' transpiration summed per organ into MECHA's
                        leaf transpiration, MECHA's water potential back to the gas exchange's leaf water potential
                        (Tuzet's stomata)
    SCENE_TRANSLATOR    between the plant and the soil (component.scene_translator: MECHA's root uptake into the
                        cells, the soil's water potential back to the roots, through the roots' soil nodes: the
                        links' mapping, the plant's mask root_surface, GRANAP's entity set soil_nodes); the light's
                        PARa and the optics are the Scene's default identity links
    WATER_TABLES        the water diagnostics of decision U-6 after each step (the SceneRecorder's declared tables,
                        Summary): the plants' root uptake and the largest change of MECHA's ψ since the step before
                        (plant_water), their organs' transpiration (leaf_water), the soil's water given to the roots
                        and the largest change of its water potential (soil_water); the balances themselves are
                        the solves' declared conservation (MECHA's plant water, the soil's water: conservation.csv)

Units: the plant's lengths in m (GRANAP's and MECHA's sections in mm and cm), MECHA's potentials in hPa and its flows
in cm3 d-1, the gas exchange's transpiration in mmol s-1, the soil's potentials in MPa and its flows in m3 s-1. A
link's factor is its units' conversion, checked by metafspm, except where the quantity changes: a diameter halved
into a radius (geometry), an amount of water into its volume (the water's molar volume).

The stomata are Tuzet's (decision U-5): each element's conductance limited by its leaf water potential, MECHA's
of the step before (the gas exchange runs first: a lagged feedback, decisions U-5 and U-6), the mean of MECHA's
water potential over the element's liquid (weighted by liquid_volume: CNW-Grass's mecha_links). An element without
liquid node (a section without anatomy) keeps the gas exchange's default (metafspm's missing="default").
"""

from openalea.adel.development import ScenarioShoot
from openalea.adel.leaf_shapes import LeafShapes
from openalea.archisimple.architecture import RootArchitecture
from openalea.archisimple.parameter_sets import WHEAT_ON_A_SHOOT
from openalea.caribu.optics import PlantOptics
from openalea.cnwgrass.metafspm.gas_exchange import GasExchange, emerged_links, mecha_links
from openalea.granap.metafspm.component import GranapAnatomy, junction_rules
from openalea.mecha.component import MechaHydraulics, scenario_reduction
from openalea.mecha.stand_ins import Development

from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.scene.recorder import Summary

from component import NodalRootEmergence, WheatCrown, scene_translator

WATER_MOLAR_VOLUME = 18e-3                   # cm3 mmol-1 (1 g of water is 1 cm3)
SECONDS_PER_DAY = 86400.
HPA_PER_MPA = 1e4

# The root sections' barriers (the segments' and the apices') by their distance from the tip (MECHA's Development,
# ArchiSimple's distance: from the section's base, an apex's its length): none at the tip (a section of distance 0:
# an apex without length, which has no anatomy), a Casparian strip up to 10 cm (a growing apex's: barrier 0 would
# make its xylem immature, MECHA's, cutting it off), a suberized endodermis beyond
BARRIERS = {"tip_zone": 0., "tip_barrier": 0., "young_zone": 0.1, "young_barrier": 1., "mature_barrier": 3.}

WHEAT_SCENARIO = {
    # the crown (WheatCrown) and the anchors of the shoot and the roots on it
    "crown_depth": 0.02, "subcrown_elements": 2,
    "shoot_anchor": {"label": "StemElement", "last": True},
    **WHEAT_ON_A_SHOOT, "root_anchor": {"label": "Seed"},
    # Adel's blades in two elements, their shapes CN-Wheat's (echap's Mercia profiles, Soissons' midribs), as
    # CNW-Grass's stand
    "n_sect": 2, "leaves": LeafShapes.cnwheat(),
    # GRANAP's anatomies: a wheat root, the seed and the stems the monocot stem's, the sheaths and blades the monocot
    # leaf's; each root section with its soil node
    "root_anatomy": {"preset": "for_wheat_root"}, "seed_anatomy": {}, "stem_anatomy": {}, "sheath_anatomy": {},
    "leaf_anatomy": {}, "soil_node": True,
    # MECHA: the soil's potential on each root section's soil node, no condition at the collar (the plant closed
    # between the soil and the air), the roots' barriers
    "soil_boundary": "node_dirichlet", "collar_condition": 0., **BARRIERS,
    # MECHA's sections reduced onto their conduits, their soil node and a blade's vapour path (scenario_reduction):
    # every section, the root segments and apices, the stem elements (the internodes' and the sheaths'), the blades
    # and the seed (a filter of the SubOrgans, Reduction(where=)); their kinds by length class of relative width 0.1
    # (a section between two classes their kinds' blend; each class evaluated once, shared by the plants: "exact" for
    # a kind per length, the reference)
    "reduced": True,
    "reduced_where": {"label": ("RootSegment", "RootApex", "StemElement", "LeafElement", "Seed")},
    "length_classes": 0.1,
    # the gas exchange: Tuzet's stomata, limited by MECHA's leaf water potential (decision U-5)
    "stomatal_model": "Tuzet",
}



PLANT_TRANSLATOR = (
    # the leaf water potential, MECHA's (Tuzet); the gas exchange's element heights and green areas of their parts
    # above the soil (emerged_links: Adel's shoot stands on the crown, below the soil, so that a blade Adel has
    # emerged from its sheath may still be in it, unlit, as Caribu cuts the light there, and without gas exchange)
    Translator(mecha_links().links + emerged_links().links)
    # GRANAP's stems (and the seed) classed by their radius: their organ's diameter halved (Adel's organs', the
    # subcrown internode's WheatCrown gives), broadcast to its elements
    .link("GranapAnatomy", "stem_radius", "ShootArchitecture", {"diameter": 0.5}, source_scale="Organ",
          scale="SubOrgan", aggregation="broadcast", unit_check=False)
    # each organ's transpiration (mmol s-1, the sum of its elements') leaving at its leaf surface (cm3 d-1)
    .link("MechaHydraulics", "leaf_transpiration", "GasExchange",
          {"transpiration": WATER_MOLAR_VOLUME * SECONDS_PER_DAY}, source_scale="SubOrgan", scale="Organ",
          aggregation="sum", unit_check=False)
)

SCENE_TRANSLATOR = scene_translator()


class WheatPlant(CompositeModel):
    """
    The plant population model: one MPG per population, its nodes the Compartments of GRANAP's sections (anatomy mode),
    each component on its own entities (Adel's elements, the rule's, ArchiSimple's roots) and at its own scale.

    Initiators, in order: WheatCrown (the seed and the subcrown internode), Adel's shoot (ScenarioShoot's
    ShootArchitecture: its main stem '<' on the crown's top, the scenario's shoot_anchor), ArchiSimple's
    RootArchitecture (its seminal roots '+' on the seed: root_anchor) and GranapAnatomy (every section's anatomy,
    joined by its junction rules, across the sections without one).

    Its parts, run in this order (the schedule): Adel's shoot (ScenarioShoot, nested: its development and
    architecture grown together), the nodal root rule, ArchiSimple's roots (the nodal roots on the phytomers the rule
    asks), GRANAP's anatomies of the new sections, the roots' barriers (MECHA's Development, by their distance from
    the tip), the new elements' optics, the gas exchange (its PARa the light's, its leaf water potential MECHA's of
    the step before) and MECHA's water flow (each organ's transpiration leaving at its leaf surface, the soil's
    potential on the roots' soil nodes); the crown has no step.

    reduction: MECHA's sections reduced onto their conduits and kept nodes when the scenario says so
    (scenario_reduction: "reduced", "reduced_where", "length_classes"), the whole network otherwise.
    """
    initiators = (WheatCrown, ScenarioShoot, RootArchitecture, GranapAnatomy)
    parts = {"crown": WheatCrown, "shoot": ScenarioShoot, "rule": NodalRootEmergence, "roots": RootArchitecture,
             "anatomy": GranapAnatomy, "barriers": Development, "optics": PlantOptics, "gas_exchange": GasExchange,
             "hydraulics": MechaHydraulics}
    from_scale, nodes, wiring = "SubOrgan", "Compartment", junction_rules
    translator = PLANT_TRANSLATOR
    reduction = staticmethod(scenario_reduction)

    def __init__(self, data_structure, time_step):
        super().__init__(data_structure, time_step)
        # the roots' soil nodes (GRANAP's entity set): the soil links' mapping (component.scene_translator)
        data_structure.define_mask("root_surface", entities="GranapAnatomy.soil_nodes")


# The water diagnostics after each step (decision U-6: the plant and the soil exchange once per step, the lag
# accepted): the plants' uptake (cm3 d-1) and the largest change of MECHA's ψ (hPa) since the step before, over the
# nodes there at both (NaN at the first row: ds.previous); their organs' transpiration (cm3 d-1); the soil's water
# given to the roots (m3 s-1) and the largest change of its water potential (MPa). Their balances are the solves'
# (graph_system(conserved=...): MECHA's PLANT_WATER, the soil's WATER), in conservation.csv: residual the plant's
# imbalance (cm3 per step), the soil's closure (m3)
WATER_TABLES = {"plant_water": Summary("node", sum=["root_uptake"], max_change=["psi"]),
                "leaf_water": Summary("Organ", sum=["leaf_transpiration"]),
                "soil_water": Summary("cell", sum=["plant_uptake"], max_change=["water_potential"],
                                      population="RichardsSoilModel")}
