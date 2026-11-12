# Use case 1: a wheat plant in its soil and its light

A soil–plant–atmosphere assembly that tests metafspm before the release: ArchiSimple's roots, Adel's shoot, GRANAP's
anatomies, MECHA's water flow, CNW-Grass's gas exchange, Caribu's light (the direct light and its scattering in the
infinite canopy), and the soil of this folder. The
environment is `environment.yaml` (`mamba env create -f environment.yaml`, the repositories cloned side by side).

## The soil (`component.py`)

`RichardsSoil` is the transient water flow of Richards' equation on a soil grid: a regular `ArrayDataStructure` or
an `AdaptiveGridDataStructure`.

- **Equation.** The mixed form (Celia et al. 1990), the pressure head h (m) as each cell's unknown:
  (θ(h) − θ(h_prev)) V + dt (Σ q + uptake − boundary inflows) = 0, written per cell volume.
- **Fluxes.** On each face, q = K_f A (Δh/d + mtg), with g = 1 on the z faces (z points down from the surface).
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
- **Water balance.** The water is declared conserved by the solve (`graph_system(conserved={"water": WATER})`,
  metafspm's `Conservation`: held θ V, in through the surface, out through the bottom and to the roots): metafspm
  integrates what crosses over the accepted sub-steps and checks the balance at every step
  (`RichardsSoil.conservation()`, the recorder's `conservation.csv`: its residual is the solver's tolerance).

`RichardsSoilModel` is the soil of a Scene: the grid over the stand's pattern, adaptive by default.

- After each run, it refines the grid where the roots take up water or where h changes steeply, and coarsens it
  elsewhere.
- **Carry-over.** When the grid changes, the water is carried over as a volume (θ V), not as h. h is then recomputed
  from θ by the inverse retention curve (`RichardsSoil.carry_over`, the grid's carry-over hook: metafspm's
  `@after_regrid`, called by the grid after each refinement or coarsening).
- **Geometry.** The cells' volumes, horizontal areas and heights, the surface and bottom layers and the z faces are
  the grid's own descriptors (metafspm's `cell_volume`, `cell_section_z`, `cell_size_z`, `boundary_low_z`,
  `boundary_high_z`, `face_normal_z`), which the soil reads as inputs: no geometry of its own to set again.
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
  - each phytomer's rank (`phytomer_rank`), its phytomers chained by rank along each axis (metafspm's
    `define_chain("phytomers", location="Phytomer", group="Axis", rank="phytomer_rank")`: a leaf's site is
    `chain_gather` along its axis);
  - each axis' Haun stage (`haun_stage`) and each plant's thermal time (`thermal_time`, °Cd), and both at its last
    call (metafspm's `last_call`, `kept_from_last_call`: no state of its own to copy them into).
- **The rule.** Each leaf appeared asks for `nodal_roots_per_leaf` nodal roots on its axis' phytomer
  `nodal_root_rank_offset` ranks below its own, `nodal_root_delay` °Cd after its appearance.
  - By default: two roots, three ranks, no delay (Klepper et al. 1984: the roots of node n appear with leaf n + 3,
    in pairs at the first nodes). The delay has no source.
  - A leaf appears when its axis' Haun stage reaches its rank minus one. Its time is interpolated within the step.
  - A leaf whose site would lie below its axis' first phytomer asks for none.
- **What it writes:** `nodal_axes_requested` per phytomer, the cumulative count ArchiSimple reads. ArchiSimple
  grows the difference with the roots it grew, `'+'` on the phytomer's bearer (metafspm's `data_structure.bearer`: the element
  at its node, its internode's top). Each leaf's request is counted once, when it falls due: none is lost, whatever
  the steps.
- **Where the nodal roots start:** each phytomer's node, their bearer's end (metafspm's `data_structure.position("Phytomer",
  at="end")`, from the elements' topological ends: each element's base at its parent's end, whatever the axes'
  inclination): on the crown while the internodes below have not elongated (the tillering zone). Adel's turtle draws
  a phytomer's first element on the top of the visible parts below it (phytomer 2 on the coleoptile's top, out of
  the soil: its display ends), where the nodal roots would otherwise start in the air.
- **Its entities:** Adel's elements (`entities = {"axis_row": ">=0"}`), its phytomers those they belong to.

Their tests are in metafspm's `test/examples/test_use_case_1_plant.py` (skipped without Adel and ArchiSimple), on a test plant model (WheatCrown, Adel,
the rule, ArchiSimple):

- the seed and the subcrown internode, at the planting's depth or the scenario's;
- Adel's main stem on the crown's top, the six seminal roots on the seed;
- each model's own entities;
- the leaves' appearance times;
- the nodal roots on the right phytomers at the right thermal times, kept, starting on the crown;
- the plant drawn connected (the grain and the subcrown internode drawn);
- the requests cumulative, the same at a step of a day and of six hours;
- the rule's defaults (nothing before the fourth leaf).

## The assembly (`models.py`)

`WheatPlant` (a `CompositeModel` declaring its parts, Adel's `ScenarioShoot` nested among them) is the plant: one MPG
per population, its nodes the Compartments of GRANAP's sections (anatomy mode).

- **Initiators**, in order: `WheatCrown`, Adel's `ShootArchitecture` (`ScenarioShoot`'s, on the crown's top),
  ArchiSimple's `RootArchitecture` (on the seed) and `GranapAnatomy` (every section's anatomy, joined by its junction
  rules).
- **A step**, its parts in order (`WheatPlant.steps()`, each step's reads and writes): Adel's shoot (`ScenarioShoot`'s
  `grow`), the nodal root rule, ArchiSimple's roots, GRANAP's anatomies of the new
  sections, the roots' barriers (MECHA's `Development`, by their distance from the tip: no root maturation model is
  ported), Caribu's `PlantOptics`, CNW-Grass's `GasExchange`, MECHA's `MechaHydraulics`.
- **Its scenario,** `WHEAT_SCENARIO`: the crown, Adel's leaf shapes CN-Wheat's (`LeafShapes.cnwheat()`: echap's
  Mercia profiles and Soissons' midribs, as CNW-Grass's stand; Adel's samples by default), `WHEAT_ON_A_SHOOT` and
  the anchors, GRANAP's wheat root
  (`for_wheat_root`) and monocot stem, sheath and leaf, a soil node per root section, MECHA's soil boundary on it
  (`node_dirichlet`) and no collar condition, MECHA's sections reduced onto their conduits (`reduction =
  staticmethod(scenario_reduction)`, the scenario's `reduced`, `reduced_where`: a filter of the SubOrgans, every
  section, the root segments and apices, the stem elements (the internodes' and the sheaths'), the blades and the
  seed), their archetypes by length class (`length_classes`: 0.1, a section between two classes their archetypes'
  blend; each class evaluated once and shared by the plants, checkpointed: `"exact"` for an archetype per length). ArchiSimple's root apices (`RootApex`, an axis' tip beyond its last segment) are root sections
  with the root's anatomy once they have a length, their barriers by their distance from the tip as the segments'
  (a growing apex's distance is its length: a Casparian strip; at barrier 0 MECHA's xylem would be immature, the apex
  cut off).

The links, each in its units:

| link | from | to | factor |
|---|---|---|---|
| GRANAP's stems' class input `stem_radius` | the organ's `diameter` (m; Adel's, the crown's) | its elements (m) | `RADIUS_OF_DIAMETER` (0.5), broadcast |
| the gas exchange's element `height` | Adel's triangles above the soil | the elements | CNW-Grass's `emerged_links`: the centre of the part above the soil (`CentreIn("air")`), NaN wholly below |
| the gas exchange's `green_area` (a private name, `green_area_in_air`) | Adel's `green_area`, its green triangles' share above the soil | the elements | `emerged_links`: their product (`FractionIn("air", weight="green")`), Adel's `green_area` untouched |
| MECHA's `leaf_transpiration` | `GasExchange.transpiration` (mmol s⁻¹, per element) | per Organ (cm³ d⁻¹) | `WATER_MOLAR_VOLUME` (18e-3 cm³ mmol⁻¹, × 86400 from the units), summed |
| the soil's `plant_uptake` | MECHA's `root_uptake` (cm³ d⁻¹) | the cells (m³ s⁻¹) | 1e-6 / 86400 |
| MECHA's `soil_psi` | the soil's `water_potential` (MPa) | the roots' soil nodes (hPa) | 1e4 |
| `GasExchange.PARa`, the optics | Caribu's light, `PlantOptics` | both ways | the Scene's identity links |
| the gas exchange's `leaf_water_potential` | MECHA's `water_potential` (hPa, at the Compartments) | the elements (MPa) | 1e-4, the mean weighted by `liquid_volume` |

Where the quantity changes (a diameter into a radius, an amount of water into its volume), the factor is a named
equivalence of metafspm's (`coupling.units`), its number computed from the declared units and checked: no link of
the assembly turns the unit check off.

The soil links are `component.scene_translator()`, through the roots' soil nodes: the links' mapping
(`mapping={"mask": "root_surface", "method": "overlap"}`, from which the Scene infers its CrossMapping), the plant
model's mask `root_surface` defined by GRANAP's entity set (`define_mask("root_surface",
entities="GranapAnatomy.soil_nodes")`): no soil node code in the assembly. The stomata are Tuzet's (`"stomatal_model": "Tuzet"` in the scenario, decision U-5): the leaf water potential
is CNW-Grass's `mecha_links()`, MECHA's water potential averaged over each element's liquid (its walls and
protoplasts, not its air spaces and pores), MECHA's of the step before (the gas exchange runs first: the lag of
decision U-6). An element without liquid node (a section without anatomy yet) keeps the gas exchange's default,
-0.1 MPa (metafspm's `missing="default"`).

Adel's shoot stands on the crown, 2 cm below the soil: a blade Adel has brought out of its sheath may still be in
the soil. The soil's surface is defined once, the Scene's `soil_surface` (0 by default): the soil grid's top, the
crown's depth below it, Caribu's cut (it lights only the triangles above it) and the gas exchange's, which reads the
height and the green area of each element's part above it (CNW-Grass's `emerged_links`: from Adel's triangles above
the soil, weighted by their green share, Adel's senescence going down from the top; NaN height and no area for an
element wholly below it: no gas exchange). A blade's tip, the first part out of the soil, is narrow: its area above
the soil is a smaller share of the element's than its length (the former chord clip, by length, gave it more).
CNW-Grass's `adel_links` height, the mean of the element's ends, put the seedling's canopy below the soil: the gas
exchange's canopy height was 0 and no element exchanged (gs, An and the transpiration 0 under a full sun).

## The run (`run.py`)

`run.py` is the script to read first: two plants over two weeks of May, hour by hour, from their sowing (`python
run.py [days] [plants] [folder]`, by default 14 days, 2 plants, `outputs/from_sowing`). It reads top to bottom: the
settings (days, plants, the time step, the sowing's date), the stand (one row, 5 cm apart, 3 cm deep), the Scene
(the plants, the soil and the light, the weather, the run configuration `Compute(threads="auto")`), its
recorder (the tables, frames every 12 hours with their views, the first plant's sections once a day, a checkpoint a
day, the last one kept), `scene.simulate(...)`, then `scene.recorder.pack()`, the frames in one archive, checked,
then their folder removed. What it assembles lives in
`models.py`: the plant model and its scenario, the soil's and the light's scenarios (`SOIL_SCENARIO`: a loam 50 cm
deep, ten layers of 5 cm cells refined once where the roots take up water; `LIGHT_SCENARIO`: Caribu's nested
radiosity, `mode="nested"`, the direct light and its scattering in the infinite canopy, radiosity between the
triangles within a sphere of 0.5 m around each, their periodic images included, and a turbid medium of 10 layers
beyond, the sun at Paris' latitude, lit every hour), the weather
(`weather(start)`: a clear day's PAR, the air's temperature and humidity in a daily course, a rain of 5 mm on the
second day, `forcings.pulse`: its total reaches the soil whatever the step), the tables (`TABLES`) and the frames'
variables (`FRAME_VARIABLES`).

Every step is an hour from the sowing: an older plant is a resumed run (a checkpoint), not a later start of one
model. The first blade emerges after about 100 °Cd (a week); before, the plant has no light and transpires nothing.
Every model reads the weather over its step (the forcings' means over the hour, the rain's total): the results do
not depend on the time step through the forcings. The run prints nothing: what it gives is in its folder, after
every hour (the tables can be read while it runs):

- `summaries/Plant/WheatPlant/summaries.csv` and `summaries/Soil/RichardsSoilModel/summaries.csv`: every
  component's state variables per plant (those of the components on the SubOrgans, Adel's, ArchiSimple's, the gas
  exchange's, the light's, too: the lengths summed, the potentials, conductances and light averaged by their
  declared weights) and for the soil;
- `summaries/hand_declared/`, the tables declared in `models.TABLES`:
  - `leaves.csv` (`models.LEAVES`): the blades' green area, PARa, An, gs and leaf water potential (over the blade
    elements of some green area, An per area times the area summed, the means weighted by the green area);
  - `roots.csv` (`models.ROOTS`): MECHA's psi averaged over the root sections by its declared weight, the liquid
    volume (the soil nodes and the air spaces have none), and their uptake;
  - `plant_water.csv`, `leaf_water.csv`, `soil_water.csv` (`models.WATER_TABLES`, decision U-6: the plant and the
    soil exchange once per step): the plant's uptake and transpiration, what the soil gave the roots, the largest
    changes of the plant's and the soil's water potentials since the hour before (`Summary(max_change=)`, from the
    values before the step the Scene keeps: `data_structure.previous`);
- `summaries/conservation.csv`: the plant's and the soil's water balances, each conserved quantity's inflow, outflow,
  change and residual per step (their solves' declared conservation: MECHA's `PLANT_WATER`, the soil's `WATER`; the
  residuals are round-off);
- `run.json`: the run's progress after every hour, then its end, with the wall time per model and per component
  (`timing`), the calls per component, the threads chosen and each kernel's and solver's time and threads
  (`compute`) and the frames' archive;
- `frames/`, while the run goes: a frame every 12 hours with its views (`frame_views="default"`), the sections of the
  first plant (`display_plants=["p0"]`) drawn in the frames of every 24 hours and the last (`display_period`);
- `frames.fspm.zip`, at the end (`scene.recorder.pack()`): the frames folder in one archive, each file an unchanged
  member. The archive is checked once written (every file of the folder a member, of the same size and CRC-32 read
  back, and the archive valid by encoding3d's validator), then the folder is removed (`pack(keep_folder=True)`
  keeps it; when the check fails, the folder stays and the error is raised). ParaView, `encoding3d.read_index` and
  `read` open the archive as the folder; `encoding3d unpack frames.fspm.zip frames` gives it back;
- `checkpoints/`: one a day, the last one kept (`keep_checkpoints=1`: each removed once the next is written; the
  anatomy templates and the reduced sections saved once, in `checkpoints/shared`, those no kept checkpoint uses
  removed): `Scene.load_checkpoint` resumes the run from it, bit for bit, the frames unpacked from `frames.fspm.zip`
  into `frames/` again, the next ones added, all packed again at the end.

**Its speed and memory.** `run.py` uses metafspm's defaults (`Compute(threads="auto")`, one process): about 40 s
for 6 plants' first 12 hours and under 4 s per hour for 8 plants at day 12 on a 14-thread laptop, under 4 GB; the frames
are written while the next hours run. `Compute(processes=n)` runs the plants in n processes, the same bits (it pays
on machines of equal cores and enough memory). The measurements, and the benchmark that makes them, are in
[test/benchmarks/use_case_1](../../test/benchmarks/use_case_1/README.md).

The frames keep a few variables, the run's explicit choice (`models.FRAME_VARIABLES`, the recorder's
`frame_options={"variables": ...}`; without it, every variable), and what the frames' views show, which the recorder
adds itself (in italics below). Every entity set keeps its structure (ids, labels, complex and parent links, the
segment ends), and:

| where | variables |
|---|---|
| `WheatPlant` Compartment (and the sections' templates) | *`psi`* (MECHA's, hPa: the sections' colour), `water_potential` (MPa, the solved nodes'), `tissue` |
| Cell, Layer | `tissue` |
| SubOrgan | `length`, *`radius`* (the roots' tubes), *`type`* (the roots' colour), `anatomy_class`, `anatomy_template`, `anatomy_kind`, `PARa`, `An`, `gs`, `transpiration` |
| Organ | *`organ_rank`* (the shoot's colour) |
| the other entity sets, the Connections | none |
| `RichardsSoilModel` cell | *`pressure_head`* (the soil's colour), `water_potential`, `water_content` (its faces and scalars unfiltered) |

**The sections in the frames.** MECHA solves each section reduced (its conduits, soil node and vapour path kept),
so the frames draw the first plant's sections as *instances* (metafspm's `display="instances"`, the recorder's
default; `SceneRecorder(display="full")` for the former drawing): per section its transform and its kept nodes' `psi`; once
per template and per archetype, in `frames/parts/` (listed in `scene.fspm.json`), the template's shape and nodes (their
`tissue`) and the archetype's recovery operator; `psi` on every cell is a derived field, rebuilt for the sections you
select. In ParaView (encoding3d's plugin): show the hidden layer "WheatPlant GranapAnatomy instances", pick a
section, apply "encoding3d Select at scale" (scale SubOrgan): the section is drawn with `psi` on each cell (or tick
the reader's "Evaluate Derived Fields" for every section). `run.py`'s last line, `scene.recorder.pack()`, writes
`frames.fspm.zip` (deflate, each frame an unchanged member: 16.3 MB for the 45.0 MB folder of a 2-plant 3-day run,
which it then removes), which ParaView opens as the folder (`encoding3d.read_index` and `read` too; `encoding3d
unpack` gives the folder back). `pvbatch encoding3d/paraview/check_in_paraview.py --instances shots
outputs/<run>/frames.fspm.zip` checks it by hand (it passed on a 2-day run: three sections selected, their 30,065 and 3,420 cells' `psi` equal to
`encoding3d.evaluate`'s).

Its tests are in metafspm's `test/examples/test_use_case_1_assembly.py` (skipped without the ports), on one plant grown from its sowing to
its first blade's emergence by daily steps (Adel and ArchiSimple grow it whatever the step; the light, the gas
exchange, MECHA and the soil run at each), checkpointed and resumed by an hourly scene for three hours: the links'
factors, the assembly's couplings, each model's own entities, the transpiration into MECHA, MECHA's leaf
water potential into the stomata, the steady plant,
the uptake into the soil, the soil's potential on the roots, the light on the gas exchange, the soil's balance with
the uptake.

### The assembly: open gaps

What `run.py`'s run from the sowing shows (2026-11-04), the ports' gaps first:

- **Growth is prescribed.** Adel grows the shoot on thermal time only. ArchiSimple grows the roots at their potential:
  on the soil temperature (a constant 12 °C forcing) and an offered structural mass (`struct_mass_input`) that no
  model provides, infinite by default. So growth responds neither to the soil's water nor to the light. No carbon is
  allocated: the leaves' assimilation `An` is recorded but feeds no other model.

- **Adel and ArchiSimple, a plant's stage.** The scenario's `start` (°Cd) advances Adel's shoot only: a shoot started
  at a few leaves has the roots of a seed; an older plant is a run resumed from a checkpoint.
