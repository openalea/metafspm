# Use case 1: a wheat plant in its soil and its light

A soil–plant–atmosphere assembly that tests metafspm before the release: ArchiSimple's roots, Adel's shoot, GRANAP's
anatomies, MECHA's water flow, CNW-Grass's gas exchange, Caribu's light, and the soil of this folder. The
environment is `environment.yaml` (`mamba env create -f environment.yaml`, the repositories cloned side by side).

## The soil (`component.py`)

`RichardsSoil` is the transient water flow of Richards' equation on a soil grid: a regular `ArrayDataStructure` or
an `AdaptiveGridDataStructure`.

- **Equation.** The mixed form (Celia et al. 1990), the pressure head h (m) as each cell's unknown:
  (θ(h) − θ(h_prev)) V + dt (Σ q + uptake − boundary inflows) = 0, written per cell volume.
- **Fluxes.** On each face, q = K_f A (Δh/d + g), with g = 1 on the z faces (z points down from the surface).
  K_f is upstream-weighted by default; the arithmetic mean is a setting (`face_conductivity`).
- **Soil.** van Genuchten–Mualem, with θr, θs, α, n, Ks and l per cell, set by layers (`set_soil_layers`). A loam
  by default (Carsel & Parrish 1988's classes in `SOILS`). Gardner's exponential soil is a setting
  (`retention_model="gardner"`).
- **Boundaries.**
  - The top: `rainfall` − `potential_evaporation` (inputs that a Scene's forcings of the same names write). Both
    are limited by what the soil can take or give: the rain it cannot take runs off.
  - The bottom: free drainage, a water table (`water_table_head`) or no flux (`bottom_boundary`).
  - The sides: the grid's (periodic over the stand). A root leaving the stand's pattern exchanges with the cells of
    its periodic image (metafspm's `CrossMapping`, `periodic=(True, True, False)`): it takes up water where a
    neighbour's root would come in. The frames draw it at its own position (the plant's architecture), out of the
    grid; a root below the grid exchanges with its bottom cells.
- **Solver.** Newton with the analytic Jacobian, and adaptive time steps (step doubling).
- **Water balance.** A second unknown, `exchanged_water`, integrates what crosses the boundaries and what the roots
  take, so Σ θ V − Σ θ₀ V = Σ exchanged_water V over any run.

`RichardsSoilModel` is the soil of a Scene: the grid over the stand's pattern, adaptive by default.

- After each run, it refines the grid where the roots take up water or where h changes steeply, and coarsens it
  elsewhere.
- **Carry-over.** When the grid changes, the water is carried over as a volume (θ V), not as h. h is then recomputed
  from θ by the inverse retention curve (`RichardsSoil.carry_over`). metafspm has no hook for such a derived
  carry-over, so the model does it after its refinement.
- **Links with the plants.** `scene_translator()` gives them:
  - MECHA's `root_uptake` (cm3 d-1) summed into the cells' `plant_uptake` (m3 s-1);
  - the soil's `water_potential` (MPa) sent back to MECHA's `soil_psi` (hPa).

Its tests are in metafspm's `test/examples/test_use_case_1_soil.py`:

- the soil curves and the Jacobian;
- the hydrostatic equilibrium;
- a steady drainage to a unit gradient;
- the water balance with a root sink and regridding;
- Srivastava & Yeh (1991)'s exact infiltration towards a water table.

## The plant's crown and its nodal roots (`component.py`)

Adel's shoot and ArchiSimple's roots grow one plant, on one MPG: one connected graph from the roots to the leaves.
Two components of this folder join them.

`WheatCrown` is the plant's first initiator: the seed and the subcrown internode, which the other two build on.

- **Structure.** Under its own Axis, GrowthUnit, Phytomer and Organ (a StemInternode):
  - the seed (a SubOrgan `Seed`), at the sowing depth;
  - then `subcrown_elements` StemElements `'<'`, vertical, up to the crown at `crown_depth` below the surface.
- **Geometry.** Each element gets its start and end (`x1` … `z2`), its length and its radius. The seed is a point,
  its length and radius the grain's (Gegas et al. 2010).
- **Drawing.** Its entities are its elements (`entities = {"label": ("Seed", "StemElement"), "axis_row": "<0"}`): it
  draws them as Adel and ArchiSimple draw theirs (the frames, the light), the subcrown internode's cylinders and the
  seed as the grain (along x). The seminal roots, on the seed, and the shoot, on the crown, are drawn joined.
- **Its organ's diameter** (`diameter`, twice `subcrown_radius`), as Adel's organs have theirs: GRANAP's stems and
  seed are classed by it (the assembly's `stem_radius` link).
- **Depths.** The seed's depth is the planting's (z: minus the sowing depth). The crown's depth does not follow the sowing depth (Kirby 1993): the subcrown internode spans the
  rest, and a crown below the seed raises.
- **Anchors**, in the plant's scenario:
  - Adel's main stem `'<'` on the crown's top, `shoot_anchor={"label": "StemElement", "last": True}`;
  - ArchiSimple's six seminal roots `'+'` on the seed, `root_anchor={"label": "Seed"}` (`WHEAT_ON_A_SHOOT`).

`NodalRootEmergence` is the nodal roots' rule: a FunctionalComponent between the shoot's development and the root
model's sites.

- **What it reads**, Adel's public state:
  - each phytomer's rank and axis (`phytomer_rank`, `axis_row`);
  - each axis' Haun stage (`haun_stage`) and each plant's thermal time (`thermal_time`, °Cd).
- **The rule.** Each leaf appeared asks for `nodal_roots_per_leaf` nodal roots on its axis' phytomer
  `nodal_root_rank_offset` ranks below its own, `nodal_root_delay` °Cd after its appearance.
  - By default: two roots, three ranks, no delay (Klepper et al. 1984: the roots of node n appear with leaf n + 3,
    in pairs at the first nodes). The delay has no source.
  - A leaf appears when its axis' Haun stage reaches its rank minus one. Its time is interpolated within the step.
  - A leaf whose site would lie below its axis' first phytomer asks for none.
- **What it writes:** `nodal_axes_requested` per phytomer, the cumulative count ArchiSimple reads. ArchiSimple
  grows the difference with the roots it grew, `'+'` on the phytomer's bearer (its first element). Each leaf's
  request is counted once, when it falls due: none is lost, whatever the steps.
- **Where the nodal roots start:** each phytomer's node (`site_x`, `site_y`, `site_z`, ArchiSimple's site position):
  the seed's position raised by the internodes on its path (the subcrown internode's, the shoot's, its own
  included), on the crown while they have not elongated (the tillering zone). Adel's turtle draws a phytomer's first
  element on the top of the visible parts below it (phytomer 2 on the coleoptile's top, out of the soil): ArchiSimple
  starting the roots at their bearer's end would put the nodal roots of phytomer 2 and above in the air.
- **Its entities:** Adel's elements (`entities = {"axis_row": ">=0"}`), its phytomers those they belong to.

Their tests are in `test/test_plant.py` (the example's own: `python -m pytest examples/use_case_1/test`, outside metafspm's suite, which needs no port), on a test plant model (WheatCrown, Adel,
the rule, ArchiSimple):

- the seed and the subcrown internode, at the planting's depth or the scenario's;
- Adel's main stem on the crown's top, the six seminal roots on the seed;
- each model's own entities;
- the leaves' appearance times;
- the nodal roots on the right phytomers at the right thermal times, kept, starting on the crown;
- the plant drawn connected (the grain and the subcrown internode drawn);
- the requests cumulative, the same at a step of a day and of six hours;
- the rule's defaults (nothing before the fourth leaf).

## The assembly (`models.py`, `run.py`)

`WheatPlant` (a `CompositeModel`, Adel's `ScenarioShoot` extended) is the plant: one MPG per population, its nodes the
Compartments of GRANAP's sections (anatomy mode).

- **Initiators**, in order: `WheatCrown`, Adel's `ShootArchitecture` (on the crown's top), ArchiSimple's
  `RootArchitecture` (on the seed) and `GranapAnatomy` (every section's anatomy, joined by its junction rules).
- **A step:** Adel's shoot (`run_shoot`), the nodal root rule, ArchiSimple's roots, GRANAP's anatomies of the new
  sections, the roots' barriers (MECHA's `Development`, by their distance from the tip: no root maturation model is
  ported), Caribu's `PlantOptics`, CNW-Grass's `GasExchange`, MECHA's `MechaHydraulics`.
- **Its scenario,** `WHEAT_SCENARIO`: the crown, Adel's leaf shapes CN-Wheat's (`LeafShapes.cnwheat()`: echap's
  Mercia profiles and Soissons' midribs, as CNW-Grass's stand; Adel's samples by default), `WHEAT_ON_A_SHOOT` and
  the anchors, GRANAP's wheat root
  (`for_wheat_root`) and monocot stem, sheath and leaf, a soil node per root section, MECHA's soil boundary on it
  (`node_dirichlet`) and no collar condition, MECHA's sections reduced onto their conduits (`reduction =
  staticmethod(scenario_reduction)`, the scenario's `reduced`, `reduced_labels`: GRANAP's anatomy kinds, the root
  segments and apices, the stems, the sheaths, the blades and the seed; a sheath's elements are Adel's StemElements,
  the kind `Sheath`). ArchiSimple's root apices (`RootApex`, an axis' tip beyond its last segment) are root sections
  with the root's anatomy once they have a length, their barriers by their distance from the tip as the segments'
  (a growing apex's distance is its length: a Casparian strip; at barrier 0 MECHA's xylem would be immature, the apex
  cut off).

The links, each in its units:

| link | from | to | factor |
|---|---|---|---|
| GRANAP's stems' class input `stem_radius` | the organ's `diameter` (m; Adel's, the crown's) | its elements (m) | 0.5, broadcast |
| the gas exchange's element `height` | Adel's `z1`, `z2` | the elements | `emerged_height`: the mean of the part above the soil, NaN wholly below |
| MECHA's `leaf_transpiration` | `GasExchange.transpiration` (mmol s⁻¹, per element) | per Organ (cm³ d⁻¹) | 18e-3 × 86400, summed |
| the soil's `plant_uptake` | MECHA's `root_uptake` (cm³ d⁻¹) | the cells (m³ s⁻¹) | 1e-6 / 86400 |
| MECHA's `soil_psi` | the soil's `water_potential` (MPa) | the roots' soil nodes (hPa) | 1e4 |
| `GasExchange.PARa`, the optics | Caribu's light, `PlantOptics` | both ways | the Scene's identity links |
| the gas exchange's `leaf_water_potential` | MECHA's `water_potential` (hPa, at the Compartments) | the elements (MPa) | 1e-4, the mean weighted by `liquid_volume` |

The soil links are `component.scene_translator()`, through `root_surface_mappings` (the roots' soil nodes onto the
grid). The stomata are Tuzet's (`"stomatal_model": "Tuzet"` in the scenario, decision U-5): the leaf water potential
is CNW-Grass's `mecha_links()`, MECHA's water potential averaged over each element's liquid (its walls and
protoplasts, not its air spaces and pores), MECHA's of the step before (the gas exchange runs first: the lag of
decision U-6). An element without liquid node (a section without anatomy yet) keeps the gas exchange's default,
-0.1 MPa (metafspm's `missing="default"`).

Adel's shoot stands on the crown, 2 cm below the soil: a blade Adel has brought out of its sheath may still be in
the soil. Caribu lights only the parts above the soil (its `soil_level`), and the gas exchange's element `height` is
the mean of its part above the soil (`models.emerged_height`; NaN for an element wholly below it: no gas exchange).
CNW-Grass's `adel_links` height, the mean of the element's ends, put the seedling's canopy below the soil: the gas
exchange's canopy height was 0 and no element exchanged (gs, An and the transpiration 0 under a full sun).

`run.py` runs two plants over two weeks of May, hour by hour, from their sowing (`python run.py [folder] [days]
[plants]`, by default `outputs/from_sowing`, 14 days, 2 plants): a clear day's PAR, the air's temperature and
humidity in a daily course, a rain on the second day, in a loam 50 cm deep (`SOIL_SCENARIO`: ten layers of 5 cm
cells, refined once where the roots take up water). Every step is an hour from the sowing: an older plant is a
resumed run (a checkpoint), not a later start of one model. The first blade emerges after about 100 °Cd (a week);
before, the plant has no light and transpires nothing. After each hour it prints, and writes to the folder's
`run.log`, `WaterBalance`'s diagnostics (decision U-6: the plant and the soil exchange once per step) and the
leaves' gas exchange (`gas_exchange`):

- the plant's transpiration and uptake (equal for a steady plant), and what the soil gave the roots;
- the soil's water change and what crossed its boundaries (equal);
- the largest changes of the plant's and the soil's water potentials since the previous hour;
- the blades' green area, PARa, An, gs and leaf water potential, the roots' water potential.

The SceneRecorder writes the summaries, `water_balance.csv` and `gas_exchange.csv` (appended to after every hour),
`run.json` (the run's progress, after every hour), a frame every 12 hours with its views (`frame_views="default"`),
the sections' drawings in the frames of every 24 hours and the last (`display_period`), and a checkpoint every day
(the last two kept): `Scene.load_checkpoint` resumes the run from it, bit for bit.

The frames keep a few variables, the run's explicit choice (`run.FRAME_VARIABLES`, the recorder's
`frame_options={"variables": ...}`; `scene(frame_variables=None)` keeps them all). Every entity set keeps its
structure (ids, labels, complex and parent links, the segment ends, the display rows' marker), and:

| where | variables |
|---|---|
| `WheatPlant` Compartment (a reduced section's interiors too) | `psi` (MECHA's, hPa), `water_potential` (MPa, the solved nodes'), `tissue` |
| Cell, Layer | `tissue` |
| SubOrgan | `length`, `radius`, `type` (the roots' default colour), `anatomy_class`, `anatomy_template`, `anatomy_kind`, `PARa`, `An`, `gs`, `transpiration` |
| Organ | `organ_rank` (the shoot's default colour) |
| the other entity sets, the Connections | none |
| `RichardsSoilModel` cell | `pressure_head` (the soil's default colour), `water_potential`, `water_content` (its faces and scalars unfiltered) |

The default views colour by kept variables (the recorder raises at the first frame otherwise). A frame drawing the
sections is 56, 70 and 89 MB at days 1 to 3 (243, 306 and 389 MB with every variable); the sections' drawing (the
GranapAnatomy mesh, 55 MB at day 3) is most of what is left.

Its tests are in `test/test_assembly.py` (the example's own, as the plant's), on one plant grown from its sowing to
its first blade's emergence by daily steps (Adel and ArchiSimple grow it whatever the step; the light, the gas
exchange, MECHA and the soil run at each), checkpointed and resumed by an hourly scene for three hours: the links'
factors, the assembly's couplings, each model's own entities, the transpiration into MECHA, MECHA's leaf
water potential into the stomata, the steady plant,
the uptake into the soil, the soil's potential on the roots, the light on the gas exchange, the soil's balance with
the uptake.

### The assembly: open gaps

What `run.py`'s run from the sowing shows (2026-11-04), the ports' gaps first:

- **Adel and ArchiSimple, a plant's stage.** The scenario's `start` (°Cd) advances Adel's shoot only: a shoot started
  at a few leaves has the roots of a seed; an older plant is a run resumed from a checkpoint.
