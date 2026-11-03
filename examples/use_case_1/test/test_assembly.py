"""
The assembly of examples/use_case_1 (models.py): WheatPlant (WheatCrown, Adel's shoot, the nodal root rule,
ArchiSimple's roots, GRANAP's anatomies, MECHA's barriers and water flow, Caribu's optics, CNW-Grass's gas exchange,
on one anatomy-mode MPG), the soil (RichardsSoilModel) and the light (CaribuLightModel) in one Scene: it builds with
every link, each model works on its own entities, a few hourly steps run, and the links carry their values in their
units (the transpiration to MECHA, the uptake to the soil, the soil's potential to the roots, the light to the gas
exchange); the plant is steady and the soil's water balance closes with the roots' uptake.

A young plant (its shoot started 100 °Cd after its first leaf, its roots just emerging), one plant in a small soil
column, three steps at midday: about 30 s.
"""
import os
import sys

import numpy as np
import pytest

for package in ("openalea.adel", "openalea.archisimple", "openalea.granap", "openalea.mecha", "openalea.cnwgrass",
                "openalea.caribu"):
    pytest.importorskip(package)

from openalea.caribu.component import CaribuLightModel                                # noqa: E402
from openalea.metafspm.coupling.units import conversion_factor                       # noqa: E402
from openalea.metafspm.data_structure.data_api import EntityView, LocatedView        # noqa: E402
from openalea.metafspm.scene.population import Planting, planting                   # noqa: E402
from openalea.metafspm.scene.recorder import SceneRecorder                           # noqa: E402
from openalea.metafspm.scene.scene import Scene                                      # noqa: E402

EXAMPLE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(EXAMPLE))

from component import RichardsSoilModel                                               # noqa: E402
from models import (PLANT_TRANSLATOR, SCENE_TRANSLATOR, WHEAT_SCENARIO, WATER_MOLAR_VOLUME, WaterBalance,  # noqa: E402
                    WheatPlant, root_surface_mappings)

HOUR = 3600.
STEPS = 3
SOIL_NODE = 4.
# The metafspm and port gaps the assembly meets (examples/use_case_1/README.md, "The assembly: open gaps"); each test
# is expected to fail until they are fixed (strict: a test passing again says its mark is to go)
GAPS = ("the use case's gaps: an entity view on an anatomy-mode DataStructure (its filters on coarser variables, its "
        "scale, its base), the reduction's library running Adel's development, GRANAP's anatomies of sections without "
        "length and of the sections made at a step, the light's and the soil's exchanges with an anatomy-mode plant")
open_gap = pytest.mark.xfail(strict=True, reason=GAPS)


def _scene():
    """One young plant (its shoot 100 °Cd after its first leaf) in a 20 cm column of loam closed at its bottom
    (no flux: the soil's water changes by the roots' uptake only), at midday, under a constant weather."""
    table = planting([Planting(x=0.05, y=0.05, z=-0.03)], WheatPlant, dict(WHEAT_SCENARIO, start=100.),
                     xrange=0.1, yrange=0.1)
    forcings = dict(PARi=1500., air_temperature=20., relative_humidity=0.6, ambient_CO2=410., wind_speed=2.,
                    soil_temperature=15., rainfall=0., potential_evaporation=0.)
    return Scene(table, environment=[RichardsSoilModel, CaribuLightModel],
                 environment_scenarios=[dict(depth=0.2, voxel=0.05, adaptive=False, initial_pressure_head=-1.,
                                             bottom_boundary="no_flux"),
                                        dict(sun=True, interval=HOUR)],
                 translator=SCENE_TRANSLATOR, time_step=HOUR, forcings=forcings, mappings=root_surface_mappings,
                 recorder=SceneRecorder(tables={"water_balance": WaterBalance()}), start=(130, 11.))


def _soil(scene):
    return next(model for model in scene.environment if isinstance(model, RichardsSoilModel))


def _light(scene):
    return next(model for model in scene.environment if isinstance(model, CaribuLightModel))


@pytest.fixture(scope="module")
def stepped():
    """The scene after each of STEPS steps: the plant's uptake and the soil's (the uptake it received at the step's
    start: the plant's previous one, decision U-6's lag)."""
    scene = _scene()
    ds = scene.populations[0].data_structure
    soil = _soil(scene)
    history = []
    for _ in range(STEPS):
        scene.run()
        history.append({"root_uptake": float(np.sum(ds.get("root_uptake"))),
                        "soil_uptake": float(np.sum(soil.grid.get("plant_uptake")))})
    scene.finish()
    return scene, history


def test_the_links_factors_are_their_units_conversions():
    """The plant's links, the soil's and the stems' class input: each factor its units' conversion (the
    transpiration's amount of water into its volume by its molar volume; a diameter into a radius)."""
    links = {(link.receiver, link.variable): link for link in PLANT_TRANSLATOR.links + SCENE_TRANSLATOR.links}
    transpiration = links[("MechaHydraulics", "leaf_transpiration")]
    assert transpiration.provider == "GasExchange" and transpiration.aggregation == "sum"
    assert transpiration.sources["transpiration"] == pytest.approx(
        WATER_MOLAR_VOLUME * conversion_factor("cm3 d-1", "cm3 s-1"), rel=1e-12)
    assert transpiration.sources["transpiration"] == pytest.approx(1555.2, rel=1e-12)       # 18e-3 × 86400
    assert links[("GranapAnatomy", "stem_radius")].sources == {"diameter": 0.5}
    potential = links[("MechaHydraulics", "soil_psi")]
    assert potential.sources["water_potential"] == pytest.approx(conversion_factor("hPa", "MPa"), rel=1e-12)
    uptake = links[("RichardsSoil", "plant_uptake")]
    assert uptake.sources["root_uptake"] == pytest.approx(conversion_factor("m3 s-1", "cm3 d-1"), rel=1e-12)
    assert links[("GasExchange", "height")].formula is not None                      # Adel's element ends


@open_gap
def test_the_assembly_builds_with_every_link(stepped):
    scene, _ = stepped
    instance = scene.populations[0].instance
    assert WheatPlant.initiators[0].__name__ == "WheatCrown" and WheatPlant.initiators[-1].__name__ == "GranapAnatomy"
    couplings = instance.couplings().set_index(["receiver", "variable"])
    expected = {("GranapAnatomy", "stem_radius"): ("ShootArchitecture", "derived"),
                ("MechaHydraulics", "leaf_transpiration"): ("GasExchange", "derived"),
                ("GasExchange", "height"): ("ShootArchitecture", "derived"),
                ("GasExchange", "PARa"): ("CaribuLight", "scene"),
                ("MechaHydraulics", "soil_psi"): ("RichardsSoil", "scene"),
                ("RootArchitecture", "nodal_axes_requested"): ("NodalRootEmergence", "shared"),
                ("MechaHydraulics", "barrier"): ("Development", "shared"),
                ("GranapAnatomy", "radius"): ("RootArchitecture", "shared")}
    for key, (provider, coupling) in expected.items():
        assert tuple(couplings.loc[key, ["provider", "coupling"]]) == (provider, coupling), key


@open_gap
def test_each_model_works_on_its_own_entities(stepped):
    """Adel's and the rule's views hold the shoot's elements, ArchiSimple's the roots', none of them the crown's;
    the optics and the gas exchange work on every SubOrgan, GRANAP and MECHA on the Compartments."""
    scene, _ = stepped
    instance = scene.populations[0].instance
    ds = scene.populations[0].data_structure
    shoot, rule, roots = (component.data_structure for component in (instance.architecture, instance.rule,
                                                                      instance.roots))
    assert all(isinstance(view, EntityView) for view in (shoot, rule, roots))
    assert set(np.asarray(shoot.labels()).tolist()) <= {"HiddenElement", "StemElement", "LeafElement"}
    assert set(np.asarray(roots.labels()).tolist()) <= {"RootSegment", "RootApex"}
    assert np.array_equal(rule.entity_ids("node"), shoot.entity_ids("node"))
    assert not set(shoot.entity_ids("node").tolist()) & set(roots.entity_ids("node").tolist())
    crown = set(np.asarray(ds.entity_ids("SubOrgan"))[np.isin(ds.at("SubOrgan").labels(), ["Seed"])].tolist())
    assert crown and not crown & (set(shoot.entity_ids("node").tolist()) | set(roots.entity_ids("node").tolist()))
    for component in (instance.optics, instance.gas_exchange):
        assert isinstance(component.data_structure, LocatedView)
        assert np.array_equal(component.data_structure.entity_ids("node"), ds.entity_ids("SubOrgan"))
    assert instance.anatomy.data_structure is ds and instance.hydraulics.data_structure is ds


@open_gap
def test_the_transpiration_reaches_mecha_in_its_units(stepped):
    """Each organ's leaf transpiration (cm3 d-1) is its elements' (mmol s-1) times 18e-3 × 86400, and leaves
    MECHA's plant at its leaf surface."""
    scene, _ = stepped
    ds = scene.populations[0].data_structure
    per_organ = ds.map_to(np.asarray(ds.get("transpiration")), "SubOrgan", "Organ", "sum")
    np.testing.assert_allclose(ds.get("leaf_transpiration"), 1555.2 * per_organ, rtol=1e-12, atol=0.)
    assert float(np.sum(ds.get("leaf_transpiration"))) > 0.
    np.testing.assert_allclose(np.sum(ds.get("evaporation")), np.sum(ds.get("leaf_transpiration")), rtol=1e-9)


@open_gap
def test_the_plant_is_steady(stepped):
    """What the roots take up is what the leaves transpire (MECHA's steady plant), at every step."""
    scene, _ = stepped
    rows = scene.recorder.table("water_balance")
    rows = rows[rows["time"] > 0.]                                     # the steps' (not the initial state's)
    assert len(rows) == STEPS
    np.testing.assert_allclose(rows["root_uptake"], rows["transpiration"], rtol=1e-8)
    assert (rows["transpiration"] > 0.).all()


@open_gap
def test_the_uptake_reaches_the_soil_in_its_units(stepped):
    """The soil's cells give the roots, in m3 s-1, the plant's uptake of the step before (cm3 d-1, times 1e-6 /
    86400): the exchange of decision U-6, lagged by a step."""
    _, history = stepped
    for before, now in zip(history[:-1], history[1:]):
        assert now["soil_uptake"] == pytest.approx(before["root_uptake"] * 1e-6 / 86400., rel=1e-9)
        assert now["soil_uptake"] > 0.


@open_gap
def test_the_soil_potential_reaches_the_roots(stepped):
    """Each root section's soil node holds the soil's water potential (MPa, times 1e4: hPa) of the cells it lies
    in, and no other Compartment receives it."""
    scene, _ = stepped
    ds = scene.populations[0].data_structure
    soil = _soil(scene)
    kind = np.asarray(ds.get("kind"))
    soil_psi = np.asarray(ds.get("soil_psi"))
    potentials = np.asarray(soil.grid.get("water_potential")) * 1e4
    assert (kind == SOIL_NODE).any()
    on_roots = soil_psi[kind == SOIL_NODE]
    assert (on_roots < 0.).all()
    assert on_roots.min() >= potentials.min() - 1e-9 and on_roots.max() <= potentials.max() + 1e-9
    np.testing.assert_array_equal(soil_psi[kind != SOIL_NODE], 0.)


@open_gap
def test_the_light_reaches_the_gas_exchange(stepped):
    """Each element's PARa (µmol m-2 s-1) is the light's, and the lit blades assimilate."""
    scene, _ = stepped
    ds = scene.populations[0].data_structure
    elements = ds.at("SubOrgan")
    light = _light(scene).light.data_structure
    np.testing.assert_allclose(elements.get("PARa"), light.get("PARa"), rtol=1e-12)
    lit = np.asarray(elements.get("PARa")) > 0.
    assert lit.any() and (np.asarray(elements.get("An"))[lit] > 0.).any()


@open_gap
def test_the_soil_water_balance_closes_with_the_uptake(stepped):
    """The soil closed at its bottom, without rain or evaporation: its water changes by what it gave the roots, and
    its own balance (stored against exchanged) closes."""
    scene, history = stepped
    balance = _soil(scene).water_balance()
    assert balance["change"] == pytest.approx(balance["exchanged"], rel=1e-9, abs=1e-15)
    given = sum(step["soil_uptake"] for step in history) * HOUR
    assert given > 0.
    assert balance["change"] == pytest.approx(-given, rel=1e-6)
