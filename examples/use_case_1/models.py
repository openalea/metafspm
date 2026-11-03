"""
The models of use case 1 (a wheat plant in its soil and its light): the plant population model (WheatPlant), its
environment models (the soil, component.RichardsSoilModel, and the light, Caribu's CaribuLightModel) and the
translators linking their components.

    WheatPlant          the plant: the seed and the subcrown internode (WheatCrown), Adel's shoot on the crown's
                        top, ArchiSimple's roots on the seed and on the phytomers the nodal root rule asks
                        (NodalRootEmergence), GRANAP's anatomy on every section, the roots' barriers (MECHA's
                        Development), the elements' optics (Caribu's PlantOptics), their gas exchange (CNW-Grass's
                        GasExchange) and the water flow through the whole plant (MECHA's MechaHydraulics), on one MPG
                        whose nodes are the Compartments
    WHEAT_SCENARIO      the plant's scenario: the crown, Adel's blades, ArchiSimple's wheat on a shoot and its anchor,
                        GRANAP's anatomies (a wheat root, the monocot stem, sheath and leaf), the barriers' zones,
                        MECHA's soil boundary on each root section's soil node, the sections reduced onto their
                        conduits
    PLANT_TRANSLATOR    within the plant: GRANAP's stems classed by their organ's diameter, the gas exchange's element
                        heights from Adel's element ends, the elements' transpiration summed per organ into MECHA's
                        leaf transpiration, MECHA's water potential back to the gas exchange's leaf water potential
                        (Tuzet's stomata)
    SCENE_TRANSLATOR    between the plant and the soil (component.scene_translator: MECHA's root uptake into the
                        cells, the soil's water potential back to the roots); the light's PARa and the optics are
                        the Scene's default identity links
    root_surface_mappings
                        the roots' soil nodes onto the soil grid (the Scene's mappings=)
    WaterBalance        the diagnostics of decision U-6 after each step (a SceneRecorder table): the plant's balance
                        (transpiration, uptake), the soil's (its water stored, exchanged, given to the roots), the
                        largest changes of the plant's and the soil's water potentials since the previous step

Units: the plant's lengths in m (GRANAP's and MECHA's sections in mm and cm), MECHA's potentials in hPa and its flows
in cm3 d-1, the gas exchange's transpiration in mmol s-1, the soil's potentials in MPa and its flows in m3 s-1. A
link's factor is its units' conversion, checked by metafspm, except where the quantity changes: a diameter halved
into a radius (geometry), an amount of water into its volume (the water's molar volume).

The stomata are Tuzet's (decision U-5): each element's conductance limited by its leaf water potential, MECHA's
of the step before (the gas exchange runs first: a lagged feedback, decisions U-5 and U-6), the mean of MECHA's
water potential over the element's liquid (weighted by liquid_volume: CNW-Grass's mecha_links). An element without
liquid node (a section without anatomy) keeps the gas exchange's default (metafspm's missing="default").
"""
import numpy as np

from openalea.adel.architecture import ShootArchitecture
from openalea.adel.development import ScenarioDevelopment, ScenarioShoot
from openalea.archisimple.architecture import RootArchitecture
from openalea.archisimple.parameter_sets import WHEAT_ON_A_SHOOT
from openalea.caribu.optics import PlantOptics
from openalea.cnwgrass.metafspm.gas_exchange import GasExchange, adel_links, mecha_links
from openalea.granap.metafspm.build import SOIL_NODE
from openalea.granap.metafspm.component import GranapAnatomy, junction_rules
from openalea.mecha.component import MechaHydraulics, scenario_reduction
from openalea.mecha.stand_ins import Development

from openalea.metafspm.coupling.cross import CrossMapping
from openalea.metafspm.coupling.translator import Translator

from component import NodalRootEmergence, RichardsSoilModel, WheatCrown, scene_translator

WATER_MOLAR_VOLUME = 18e-3                   # cm3 mmol-1 (1 g of water is 1 cm3)
SECONDS_PER_DAY = 86400.
HPA_PER_MPA = 1e4
CM3_PER_M3 = 1e6

# The root segments' barriers by their distance from the tip (MECHA's Development): none on the tip segment, a
# Casparian strip up to 10 cm, a suberized endodermis beyond
BARRIERS = {"tip_zone": 0., "tip_barrier": 0., "young_zone": 0.1, "young_barrier": 1., "mature_barrier": 3.}

WHEAT_SCENARIO = {
    # the crown (WheatCrown) and the anchors of the shoot and the roots on it
    "crown_depth": 0.02, "subcrown_elements": 2,
    "shoot_anchor": {"label": "StemElement", "last": True},
    **WHEAT_ON_A_SHOOT, "root_anchor": {"label": "Seed"},
    # Adel's blades in two elements
    "n_sect": 2,
    # GRANAP's anatomies: a wheat root, the seed and the stems the monocot stem's, the sheaths and blades the monocot
    # leaf's; each root section with its soil node
    "root_anatomy": {"preset": "for_wheat_root"}, "seed_anatomy": {}, "stem_anatomy": {}, "sheath_anatomy": {},
    "leaf_anatomy": {}, "soil_node": True,
    # MECHA: the soil's potential on each root section's soil node, no condition at the collar (the plant closed
    # between the soil and the air), the roots' barriers
    "soil_boundary": "node_dirichlet", "collar_condition": 0., **BARRIERS,
    # MECHA's sections reduced onto their conduits, their soil node and a blade's vapour path (scenario_reduction)
    "reduced": True, "reduced_labels": ("RootSegment", "StemElement", "LeafElement", "Seed"),
    # the gas exchange: Tuzet's stomata, limited by MECHA's leaf water potential (decision U-5)
    "stomatal_model": "Tuzet",
}

PLANT_TRANSLATOR = (
    Translator(adel_links(provider="ShootArchitecture").links        # the gas exchange's element heights
               + mecha_links().links)                                # its leaf water potential, MECHA's (Tuzet)
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


class WheatPlant(ScenarioShoot):
    """
    The plant population model: one MPG per population, its nodes the Compartments of GRANAP's sections (anatomy mode),
    each component on its own entities (Adel's elements, the rule's, ArchiSimple's roots) and at its own scale.

    Initiators, in order: WheatCrown (the seed and the subcrown internode), Adel's ShootArchitecture (its main stem
    '<' on the crown's top: the scenario's shoot_anchor), ArchiSimple's RootArchitecture (its seminal roots '+' on
    the seed: root_anchor) and GranapAnatomy (every section's anatomy, joined by its junction rules, across the
    sections without one).

    A step: Adel's shoot (its development and architecture together, ScenarioShoot.run_shoot), the nodal root rule,
    ArchiSimple's roots (the nodal roots on the phytomers the rule asks), GRANAP's anatomies of the new sections, the
    roots' barriers (MECHA's Development, by their distance from the tip), the new elements' optics, the gas exchange
    (its PARa the light's, its leaf water potential MECHA's of the step before) and MECHA's water flow (each organ's transpiration leaving at its leaf surface, the soil's
    potential on the roots' soil nodes).

    reduction: MECHA's sections reduced onto their conduits and kept nodes when the scenario says so
    (scenario_reduction: "reduced", "reduced_labels", "length_classes"), the whole network otherwise.
    """
    initiators = (WheatCrown, ShootArchitecture, RootArchitecture, GranapAnatomy)
    from_scale, nodes, wiring = "SubOrgan", "Compartment", junction_rules
    translator = PLANT_TRANSLATOR
    reduction = staticmethod(scenario_reduction)

    def __init__(self, data_structure, time_step):
        ds = data_structure
        self.crown = WheatCrown(data_structure=ds, time_step=time_step)
        self.development = ScenarioDevelopment(data_structure=ds, time_step=time_step)
        self.architecture = ShootArchitecture(data_structure=ds, time_step=time_step)
        self.rule = NodalRootEmergence(data_structure=ds, time_step=time_step)
        self.roots = RootArchitecture(data_structure=ds, time_step=time_step)
        self.anatomy = GranapAnatomy(data_structure=ds, time_step=time_step)
        self.barriers = Development(data_structure=ds, time_step=time_step)
        self.optics = PlantOptics(data_structure=ds, time_step=time_step)
        self.gas_exchange = GasExchange(data_structure=ds, time_step=time_step)
        self.hydraulics = MechaHydraulics(data_structure=ds, time_step=time_step)
        self.couple(self.crown, self.development, self.architecture, self.rule, self.roots, self.anatomy,
                    self.barriers, self.optics, self.gas_exchange, self.hydraulics)
        ds.define_mask("root_surface", {"kind": float(SOIL_NODE)})      # the roots' soil nodes: the soil mapping's

    def run(self):
        self.run_shoot()
        self.rule()
        self.roots()
        self.anatomy()
        self.barriers()
        self.optics()
        self.gas_exchange()
        self.hydraulics()


def root_surface_mappings(scene):
    """The roots' soil nodes (the root_surface mask) of each population onto the soil grid."""
    soil = next(model for model in scene.environment if isinstance(model, RichardsSoilModel))
    return [CrossMapping(population.data_structure, soil.grid, method="overlap", mask="root_surface")
            for population in scene.populations]


class WaterBalance:
    """
    The water balance diagnostics after each step (decision U-6: the plant and the soil exchange once per step, the
    lag accepted), a SceneRecorder table (tables={"water_balance": WaterBalance()}), one row per step:

        transpiration, root_uptake      the plants' (cm3 d-1: MECHA's leaf transpiration and root uptake), equal for
                                        a steady plant
        plant_balance                   their difference relative to the transpiration
        soil_uptake                     what the soil's cells gave the roots (cm3 d-1, the soil's plant_uptake)
        soil_stored, soil_change, soil_exchanged
                                        the soil's water (m3), its change since the first run and what crossed its
                                        boundaries or went to the roots (RichardsSoilModel.water_balance): equal
        plant_potential_change, soil_potential_change
                                        the largest change of MECHA's ψ (MPa) and of the soil's water potential (MPa)
                                        since the previous step (NaN at the first, or when the nodes changed)
    """

    def __init__(self):
        self.previous = {}

    def _change(self, key, values) -> float:
        values = np.asarray(values, dtype=float).reshape(-1)
        before = self.previous.get(key)
        self.previous[key] = values.copy()
        if before is None or before.shape != values.shape:
            return np.nan
        return float(np.max(np.abs(values - before))) if values.size else 0.

    def __call__(self, scene) -> dict:
        row = {"time": scene.time}
        plants = [population.data_structure for population in scene.populations]
        transpiration = sum(float(np.sum(ds.get("leaf_transpiration"))) for ds in plants)
        uptake = sum(float(np.sum(ds.get("root_uptake"))) for ds in plants)
        row.update(transpiration=transpiration, root_uptake=uptake,
                   plant_balance=(transpiration - uptake) / transpiration if transpiration else np.nan)
        row["plant_potential_change"] = max(
            (self._change(("plant", k), np.asarray(ds.get("psi")) / HPA_PER_MPA) for k, ds in enumerate(plants)),
            default=np.nan)
        for model in scene.environment:
            if isinstance(model, RichardsSoilModel):
                balance = model.water_balance()
                row.update(soil_uptake=float(np.sum(model.grid.get("plant_uptake"))) * CM3_PER_M3 * SECONDS_PER_DAY,
                           soil_stored=balance["stored"], soil_change=balance["change"],
                           soil_exchanged=balance["exchanged"],
                           soil_potential_change=self._change("soil", model.grid.get("water_potential")))
        return row
