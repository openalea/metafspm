"""
The assembly of examples/use_case_1 (models.py): WheatPlant (WheatCrown, Adel's shoot, the nodal root rule,
ArchiSimple's roots, GRANAP's anatomies, MECHA's barriers and water flow, Caribu's optics, CNW-Grass's gas exchange,
on one anatomy-mode MPG), the soil (RichardsSoilModel) and the light (CaribuLightModel) in one Scene: it builds with
every link, each model works on its own entities, a few hourly steps run, and the links carry their values in their
units (the transpiration to MECHA, the uptake to the soil, the soil's potential to the roots, the light to the gas
exchange); the plant is steady and the soil's water balance closes with the roots' uptake.

One plant in a small soil column, grown from its sowing to its first blade's emergence by daily steps, resumed from
that checkpoint by hourly steps (a scene's time step is its construction's), three of them at midday: about 90 s.
"""
import os
import sys

import numpy as np
import pytest

for package in ("openalea.adel", "openalea.archisimple", "openalea.granap", "openalea.mecha", "openalea.cnwgrass",
                "openalea.caribu"):
    pytest.importorskip(package)

from openalea.caribu.component import CaribuLightModel                                # noqa: E402
from openalea.cnwgrass.metafspm.gas_exchange import GasExchange                      # noqa: E402
from openalea.metafspm.coupling.geometry import CentreIn, FractionIn                 # noqa: E402
from openalea.metafspm.coupling.units import conversion_factor                       # noqa: E402
from openalea.metafspm.data_structure.data_api import EntityView, LocatedView        # noqa: E402
from openalea.metafspm import Planting, planting, Scene, SceneRecorder

EXAMPLE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(EXAMPLE))

from component import RichardsSoilModel                                               # noqa: E402
from models import (FRAME_VARIABLES, PLANT_TRANSLATOR, SCENE_TRANSLATOR, WATER_TABLES, WHEAT_SCENARIO,  # noqa: E402
                    WheatPlant)

HOUR, DAY = 3600., 86400.
EMERGENCE_DAYS = 5                 # at 20 °C (100 °Cd), the first blade has emerged and is lit
STEPS = 3


def _scene(time_step=HOUR):
    """One plant sown 3 cm deep in a 20 cm column of loam closed at its bottom (no flux: the soil's water changes by
    the roots' uptake only), at midday, under a constant weather, by steps of *time_step* (s)."""
    table = planting([Planting(x=0.05, y=0.05, depth=0.03)], WheatPlant, dict(WHEAT_SCENARIO), x_range=0.1, y_range=0.1)
    forcings = dict(PARi=1500., air_temperature=20., relative_humidity=0.6, ambient_CO2=410., wind_speed=2.,
                    soil_temperature=15., rainfall=0., potential_evaporation=0.)
    return Scene(table, environment=[RichardsSoilModel, CaribuLightModel],
                 environment_scenarios=[dict(depth=0.2, voxel=0.05, adaptive=False, initial_pressure_head=-1.,
                                             bottom_boundary="no_flux"),
                                        dict(sun=True, cadence={"light": HOUR})],
                 translator=SCENE_TRANSLATOR, time_step=time_step, forcings=forcings,
                 recorder=SceneRecorder(tables=WATER_TABLES), start="2026-05-10 11:00")


def _soil(scene):
    return scene.model(RichardsSoilModel)


def _light(scene):
    return scene.model(CaribuLightModel)


@pytest.fixture(scope="module")
def stepped(tmp_path_factory):
    """The plant grown from its sowing by EMERGENCE_DAYS daily steps (Adel and ArchiSimple grow it whatever the step;
    the light, the gas exchange, MECHA and the soil run at each), checkpointed, resumed by an hourly scene
    (Scene.load_checkpoint); then the scene after each of STEPS hours: the plant's uptake and the soil's (the uptake
    it received at the step's start: the plant's previous one, decision U-6's lag)."""
    daily = _scene(DAY)
    for _ in range(EMERGENCE_DAYS):
        daily.run()
    daily.finish()
    checkpoint = str(tmp_path_factory.mktemp("emerged") / "checkpoint")
    daily.checkpoint(checkpoint)
    scene = _scene(HOUR)
    scene.load_checkpoint(checkpoint)
    data_structure = scene.populations[0].data_structure
    soil = _soil(scene)
    history = [{}]
    for _ in range(STEPS):
        history[-1]["compartments"] = np.asarray(data_structure.entity_ids("node")).copy()   # before the next step
        scene.run()
        history.append({"root_uptake": float(np.sum(data_structure.get("root_uptake"))),
                        "soil_uptake": float(np.sum(soil.grid.get("plant_uptake")))})
    scene.finish()
    return scene, history


def test_the_links_factors_are_their_units_conversions():
    """The plant's links, the soil's and the stems' class input: each factor its units' conversion, checked; where
    the quantity changes a named equivalence, its factor computed from the units (the transpiration's amount of water
    into its volume by the water's molar volume; a diameter into a radius)."""
    from openalea.adel.architecture import ShootArchitecture
    from openalea.granap.metafspm.component import GranapAnatomy
    from openalea.mecha.component import MechaHydraulics
    from openalea.metafspm.coupling.declaration import check_link
    from openalea.metafspm.coupling.units import RADIUS_OF_DIAMETER, WATER_MOLAR_VOLUME
    links = {(link.receiver, link.variable): link for link in PLANT_TRANSLATOR.links + SCENE_TRANSLATOR.links}
    transpiration = links[("MechaHydraulics", "leaf_transpiration")]
    assert transpiration.provider == "GasExchange" and transpiration.aggregation == "sum"
    assert transpiration.sources["transpiration"] is WATER_MOLAR_VOLUME and transpiration.unit_check
    resolved = check_link(transpiration, MechaHydraulics.__new__(MechaHydraulics), GasExchange.__new__(GasExchange))
    assert resolved.sources["transpiration"] == pytest.approx(
        WATER_MOLAR_VOLUME.value * conversion_factor("cm3 d-1", "cm3 s-1"), rel=1e-12)
    assert resolved.sources["transpiration"] == 18e-3 * 86400.                              # as written before
    radius = links[("GranapAnatomy", "stem_radius")]
    assert radius.sources == {"diameter": RADIUS_OF_DIAMETER}
    assert check_link(radius, GranapAnatomy.__new__(GranapAnatomy),
                      ShootArchitecture.__new__(ShootArchitecture)).sources == {"diameter": 0.5}
    potential = links[("MechaHydraulics", "soil_psi")]
    assert potential.sources["water_potential"] == pytest.approx(conversion_factor("hPa", "MPa"), rel=1e-12)
    uptake = links[("RichardsSoil", "plant_uptake")]
    assert uptake.sources["root_uptake"] == pytest.approx(conversion_factor("m3 s-1", "cm3 d-1"), rel=1e-12)
    height, area = links[("GasExchange", "height")], links[("GasExchange", "green_area")]      # above the soil
    assert isinstance(height.sources[0], CentreIn) and height.sources[0].domain == "air"
    assert area.sources[0] == "green_area" and isinstance(area.sources[1], FractionIn)
    assert area.names == {"green_area": "green_area_in_air"}                 # Adel's green_area untouched
    assert uptake.mapping == potential.mapping == {"mask": "root_surface", "method": "overlap"}   # the soil nodes


def test_the_gas_exchange_reads_the_green_area_above_the_soil(stepped):
    """The gas exchange's green_area is each element's green area above the soil (Adel's times its green triangles'
    share above the Scene's soil surface), under a private name: Adel's green_area is the whole element's. An
    element drawn wholly below the soil exchanges nothing (0, its height NaN), one wholly above over its whole green
    area; the light and the gas exchange cut the elements at the same surface."""
    scene, _ = stepped
    data_structure = scene.populations[0].data_structure
    elements = data_structure.at("SubOrgan")
    gas_exchange = scene.populations[0].instance.gas_exchange
    assert data_structure.soil_surface == 0. and _light(scene).light.soil_surface == 0.
    whole, read = np.asarray(elements.get("green_area")), gas_exchange.read("green_area")
    np.testing.assert_array_equal(read, data_structure.get("green_area_in_air"))
    triangles = elements.triangles()
    drawn = np.bincount(triangles.owner, minlength=elements.n_nodes()) > 0
    lowest, highest = np.full(elements.n_nodes(), np.inf), np.full(elements.n_nodes(), -np.inf)
    np.minimum.at(lowest, triangles.owner, triangles.vertices[..., 2].min(axis=1))
    np.maximum.at(highest, triangles.owner, triangles.vertices[..., 2].max(axis=1))
    above, below = drawn & (lowest >= 0.), drawn & (highest < 0.)
    assert (read <= whole * (1. + 1e-12)).all() and above.any() and below.any()
    np.testing.assert_allclose(read[above], whole[above], rtol=1e-12)
    assert (read[below] == 0.).all() and np.isnan(gas_exchange.read("height")[below]).all()


def test_the_assembly_builds_with_every_link(stepped):
    scene, _ = stepped
    instance = scene.populations[0].instance
    assert WheatPlant.initiators[0].__name__ == "WheatCrown" and WheatPlant.initiators[-1].__name__ == "GranapAnatomy"
    couplings = instance.couplings().set_index(["receiver", "variable"])
    expected = {("GranapAnatomy", "stem_radius"): ("ShootArchitecture", "derived"),
                ("MechaHydraulics", "leaf_transpiration"): ("GasExchange", "derived"),
                ("GasExchange", "height"): ("ShootArchitecture", "derived"),
                ("GasExchange", "green_area"): ("ShootArchitecture", "derived"),
                ("GasExchange", "leaf_water_potential"): ("MechaHydraulics", "derived"),
                ("GasExchange", "PARa"): ("CaribuLightModel.CaribuLight", "scene"),     # the Scene's identity link
                ("MechaHydraulics", "soil_psi"): ("RichardsSoil", "scene"),
                ("RootArchitecture", "nodal_axes_requested"): ("NodalRootEmergence", "shared"),
                ("MechaHydraulics", "barrier"): ("Development", "shared"),
                ("GranapAnatomy", "radius"): ("RootArchitecture", "shared")}
    for key, (provider, coupling) in expected.items():
        assert tuple(couplings.loc[key, ["provider", "coupling"]]) == (provider, coupling), key


def test_each_model_works_on_its_own_entities(stepped):
    """Adel's and the rule's views hold the shoot's elements, ArchiSimple's the roots', none of them the crown's;
    the optics and the gas exchange work on every SubOrgan, GRANAP and MECHA on the Compartments."""
    scene, _ = stepped
    instance = scene.populations[0].instance
    data_structure = scene.populations[0].data_structure
    shoot, rule, roots = (component.data_structure for component in (instance.shoot.architecture, instance.rule,
                                                                      instance.roots))
    assert all(isinstance(view, EntityView) for view in (shoot, rule, roots))
    assert set(np.asarray(shoot.labels()).tolist()) <= {"HiddenElement", "StemElement", "LeafElement"}
    assert set(np.asarray(roots.labels()).tolist()) <= {"RootSegment", "RootApex"}
    assert np.array_equal(rule.entity_ids("node"), shoot.entity_ids("node"))
    assert not set(shoot.entity_ids("node").tolist()) & set(roots.entity_ids("node").tolist())
    crown = set(np.asarray(data_structure.entity_ids("SubOrgan"))[np.isin(data_structure.at("SubOrgan").labels(), ["Seed"])].tolist())
    assert crown and not crown & (set(shoot.entity_ids("node").tolist()) | set(roots.entity_ids("node").tolist()))
    own = instance.crown.data_structure                                 # the crown's: the seed, the subcrown internode
    assert isinstance(own, EntityView) and crown <= set(own.entity_ids("node").tolist())
    assert set(np.asarray(own.labels()).tolist()) == {"Seed", "StemElement"}
    elements = data_structure.at("SubOrgan")                                        # drawn with the shoot and the roots
    drawn = set(np.asarray(elements.entity_ids("node"))[np.unique(elements.triangles().owner)].tolist())
    assert set(own.entity_ids("node").tolist()) <= drawn
    for component in (instance.optics, instance.gas_exchange):
        assert isinstance(component.data_structure, LocatedView)
        assert np.array_equal(component.data_structure.entity_ids("node"), data_structure.entity_ids("SubOrgan"))
    assert instance.anatomy.data_structure is data_structure and instance.hydraulics.data_structure is data_structure


def test_the_transpiration_reaches_mecha_in_its_units(stepped):
    """Each organ's leaf transpiration (cm3 d-1) is its elements' (mmol s-1) times 18e-3 × 86400, and leaves
    MECHA's plant at its leaf surface."""
    scene, _ = stepped
    data_structure = scene.populations[0].data_structure
    per_organ = data_structure.map_to(np.asarray(data_structure.get("transpiration")), "SubOrgan", "Organ", "sum")
    np.testing.assert_allclose(data_structure.get("leaf_transpiration"), 1555.2 * per_organ, rtol=1e-12, atol=0.)
    assert float(np.sum(data_structure.get("leaf_transpiration"))) > 0.
    np.testing.assert_allclose(np.sum(data_structure.get("evaporation")), np.sum(data_structure.get("leaf_transpiration")), rtol=1e-9)


def test_the_stomata_read_mechas_leaf_water_potential(stepped):
    """Tuzet's stomata: each element's leaf water potential (MPa) is the mean of MECHA's water potential (hPa) over
    its liquid (weighted by liquid_volume), held at the elements; an element without liquid keeps the gas exchange's
    default. Derived when read: the gas exchange, running before MECHA, reads the step before's (the lag of
    decision U-6); read now, MECHA's last."""
    scene, _ = stepped
    instance = scene.populations[0].instance
    data_structure = scene.populations[0].data_structure
    assert instance.gas_exchange.stomatal_model == "Tuzet"
    assert data_structure.location("leaf_water_potential") == "SubOrgan"
    owner = np.asarray(data_structure.owner("SubOrgan"))
    water, liquid = np.asarray(data_structure.get("water_potential")), np.asarray(data_structure.get("liquid_volume"))
    total = np.bincount(owner[owner >= 0], weights=liquid[owner >= 0], minlength=data_structure.entity_ids("SubOrgan").size)
    expected = np.bincount(owner[owner >= 0], weights=(water * liquid)[owner >= 0],
                           minlength=total.size) / np.where(total > 0., total, 1.) * 1e-4
    values = np.asarray(data_structure.get("leaf_water_potential"))
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values[total > 0.], expected[total > 0.], rtol=1e-12)
    np.testing.assert_array_equal(values[total == 0.], GasExchange.leaf_water_potential)      # its default
    leaves = np.asarray(data_structure.at("SubOrgan").labels()) == "LeafElement"
    assert (values[leaves & (total > 0.)] < 0.).all()


def test_the_blades_leaf_water_potential_is_mechas(stepped):
    """On the reduced plant (WHEAT_SCENARIO's reduced), each grown blade element's kept conduits carry their liquid
    volume: its leaf water potential is MECHA's, not the gas exchange's default (a blade element of no length yet has
    no section)."""
    scene, _ = stepped
    data_structure = scene.populations[0].data_structure
    owner = np.asarray(data_structure.owner("SubOrgan"))
    liquid = np.asarray(data_structure.get("liquid_volume"))
    elements = data_structure.at("SubOrgan")
    blades = np.flatnonzero((np.asarray(elements.labels()) == "LeafElement") & (np.asarray(elements.get("length")) > 0.))
    assert blades.size
    assert all(liquid[owner == blade].sum() > 0. for blade in blades)
    assert (np.asarray(data_structure.get("leaf_water_potential"))[blades] != GasExchange.leaf_water_potential).all()


def test_the_plant_is_steady(stepped):
    """What the roots take up is what the leaves transpire (MECHA's steady plant), at every step: the plant's
    declared conservation (MechaHydraulics' PLANT_WATER, conservation.csv) closes, its uptake (plant_water) is its
    organs' transpiration (leaf_water); the largest change of its ψ since the step before (data_structure.previous)."""
    scene, _ = stepped
    balances = scene.recorder.table("conservation")
    plant = balances[balances["component"] == "MechaHydraulics"]
    assert len(plant) == STEPS and set(plant["unit"]) == {"cm3"} and (plant["change"] == 0.).all()
    assert (plant["outflow"] > 0.).all()
    np.testing.assert_allclose(plant["residual"], 0., atol=1e-8 * plant["outflow"].max())
    uptake, transpiration = (scene.recorder.table(name) for name in ("plant_water", "leaf_water"))
    uptake, transpiration = uptake[uptake["time"] > 0.], transpiration[transpiration["time"] > 0.]
    assert len(uptake) == STEPS
    np.testing.assert_allclose(uptake["root_uptake"], transpiration["leaf_transpiration"], rtol=1e-8)
    assert (uptake["psi_max_change"] > 0.).all()


def test_the_uptake_reaches_the_soil_in_its_units(stepped):
    """The soil's cells give the roots, in m3 s-1, the plant's uptake of the step before (cm3 d-1, times 1e-6 /
    86400): the exchange of decision U-6, lagged by a step."""
    _, history = stepped
    for before, now in zip(history[1:-1], history[2:]):
        assert now["soil_uptake"] == pytest.approx(before["root_uptake"] * 1e-6 / 86400., rel=1e-9)
        assert now["soil_uptake"] > 0.


def test_the_soil_potential_reaches_the_roots(stepped):
    """Each root section's soil node holds the soil's water potential (MPa, times 1e4: hPa) of the cells it lies
    in, and no other Compartment receives it. The exchange precedes the plant's step (decision U-6): the soil nodes
    of the sections grown at the last step hold the default (0) until the next."""
    scene, history = stepped
    data_structure = scene.populations[0].data_structure
    soil = _soil(scene)
    soil_nodes = data_structure.mask("root_surface")                    # GRANAP's entity set soil_nodes
    soil_psi = np.asarray(data_structure.get("soil_psi"))
    potentials = np.asarray(soil.grid.get("water_potential")) * 1e4
    exchanged = np.isin(np.asarray(data_structure.entity_ids("node")), history[-2]["compartments"])
    assert soil_nodes.any()
    on_roots = soil_psi[soil_nodes & exchanged]
    assert (on_roots < 0.).all()
    assert on_roots.min() >= potentials.min() - 1e-9 and on_roots.max() <= potentials.max() + 1e-9
    np.testing.assert_array_equal(soil_psi[~soil_nodes], 0.)


def test_the_light_reaches_the_gas_exchange(stepped):
    """Each element's PARa (µmol m-2 s-1) is the light's, and the lit blades assimilate. The light runs before the
    plant's step: the elements grown at the last step are not lit yet (no PARa)."""
    scene, _ = stepped
    data_structure = scene.populations[0].data_structure
    elements = data_structure.at("SubOrgan")
    light = _light(scene).light.data_structure
    position = {int(v): i for i, v in enumerate(np.asarray(elements.entity_ids("node")).tolist())}
    lit_by_light = np.array([position[int(v)] for v in light.source_ids()])
    np.testing.assert_allclose(np.asarray(elements.get("PARa"))[lit_by_light], light.get("PARa"), rtol=1e-12)
    np.testing.assert_array_equal(np.delete(np.asarray(elements.get("PARa")), lit_by_light), 0.)
    lit = np.asarray(elements.get("PARa")) > 0.
    assert lit.any() and (np.asarray(elements.get("An"))[lit] > 0.).any()
    assert (np.asarray(elements.get("height"))[lit] > 0.).all()            # their parts above the soil


def test_the_soil_water_balance_closes_with_the_uptake(stepped):
    """The soil closed at its bottom, without rain or evaporation: over the hourly steps, its water changes by what it
    gave the roots, its declared conservation (RichardsSoil's WATER, conservation.csv) closing at each step."""
    scene, history = stepped
    balances = scene.recorder.table("conservation")
    soil = balances[balances["component"] == "RichardsSoil"]
    assert len(soil) == STEPS and set(soil["unit"]) == {"m3"} and (soil["inflow"] == 0.).all()
    given = sum(step["soil_uptake"] for step in history[1:]) * HOUR
    assert given > 0.
    assert soil["outflow"].sum() == pytest.approx(given, rel=1e-9)
    assert soil["change"].sum() == pytest.approx(-given, rel=1e-6)
    np.testing.assert_allclose(soil["residual"], 0., atol=1e-9 * given)


def test_the_frames_keep_what_the_default_views_show(stepped):
    """run.py's frames keep a few variables (models.FRAME_VARIABLES) and what the default views colour, select and
    size tubes by (the recorder adds them), the Compartments' potential and tissue, a valid encoding3d frame marked
    filtered. The reduced sections are instances of their templates (display="instances", the default): the
    sections' derived psi is the recovered one of display="full" (its interior rows), the kept nodes' their own; the
    views declare the instances layer."""
    from openalea import encoding3d
    from openalea.encoding3d.validate import validate_bytes
    from openalea.metafspm.io.encoding3d import DISPLAY_ONLY, default_views, filtered_view_columns, scene_to_frame
    from openalea.metafspm.io.encoding3d import with_view_columns
    scene, _ = stepped
    views = default_views(scene, scene_to_frame(scene))
    # the frames' filter names the data; the recorder adds what the views show (SceneRecorder.frame_variables)
    assert "default/WheatPlant shoot: WheatPlant/Organ/organ_rank" in filtered_view_columns(views, FRAME_VARIABLES)
    kept = with_view_columns(FRAME_VARIABLES, views, ["WheatPlant", "RichardsSoilModel"])
    assert filtered_view_columns(views, kept) == []
    layers = {layer["name"]: layer for layer in views["configurations"]["default"]["layers"]}
    assert layers["WheatPlant GranapAnatomy instances"]["color_by"] == {"entity_set": "Compartment", "variable": "psi"}
    frame = scene_to_frame(scene, variables=kept)
    plant = frame.data_structure("WheatPlant")
    compartments = plant.entity_set("Compartment")
    assert set(compartments.columns) == {"id", "label", "complex", "parent", "edge_type", "psi", "water_potential",
                                         "tissue"}
    sections = plant.instance_set("GranapAnatomy")
    assert sections.owners == "SubOrgan" and [field.name for field in sections.derived] == ["psi"]
    assert {"An", "PARa", "radius", "length"} <= set(plant.entity_set("SubOrgan").columns)
    assert plant.extras["metafspm"]["filtered"]["variables"]["Compartment"] == ["psi", "tissue", "water_potential"]
    assert validate_bytes(encoding3d.to_bytes(frame, embed_parts=True)).ok
    full = scene_to_frame(scene, variables=kept, display="full").data_structure("WheatPlant")
    rows = full.entity_set("Compartment").columns
    shown = np.asarray(rows[DISPLAY_ONLY], dtype=bool)
    assert shown.any() and not full.instance_sets                  # the interiors as display rows
    recovered = dict(zip(rows["id"][shown].tolist(), rows["psi"][shown].tolist()))
    evaluated = encoding3d.evaluate(frame, "psi")
    from openalea.metafspm.solve.reduction import display_ids
    for row in range(0, sections.count, max(1, sections.count // 7)):      # a few sections
        owner = int(sections.columns["owner"][row])
        template = frame.part(sections.parts[int(sections.columns["template"][row])])
        template_rows = template.entity_set("Compartment").columns["id"]
        ids = display_ids(owner, template_rows)
        here = np.array([identifier in recovered for identifier in ids.tolist()])
        assert here.any()
        np.testing.assert_allclose(evaluated[row][here], [recovered[i] for i in ids[here].tolist()], rtol=1e-9,
                                   atol=1e-12)
