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
  grows the difference with the roots it grew, `'+'` on the phytomer's bearer (metafspm's `ds.bearer`: the element
  at its node, its internode's top). Each leaf's request is counted once, when it falls due: none is lost, whatever
  the steps.
- **Where the nodal roots start:** each phytomer's node, their bearer's end (metafspm's `ds.position("Phytomer",
  at="end")`, from the elements' topological ends: each element's base at its parent's end, whatever the axes'
  inclination): on the crown while the internodes below have not elongated (the tillering zone). Adel's turtle draws
  a phytomer's first element on the top of the visible parts below it (phytomer 2 on the coleoptile's top, out of
  the soil: its display ends), where the nodal roots would otherwise start in the air.
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
  seed), their kinds by length class (`length_classes`: 0.1, a section between two classes their kinds'
  blend; each class evaluated once and shared by the plants, checkpointed: `"exact"` for a kind per length). ArchiSimple's root apices (`RootApex`, an axis' tip beyond its last segment) are root sections
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
(the plants, the soil and the light, the weather, the run configuration `Compute(threads="auto", workers="auto")`), its
recorder (the tables, frames every 12 hours with their views, the first plant's sections once a day, a checkpoint a
day), `scene.simulate(...)`, then `scene.recorder.pack()`, the frames in one archive. What it assembles lives in
`models.py`: the plant model and its scenario, the soil's and the light's scenarios (`SOIL_SCENARIO`: a loam 50 cm
deep, ten layers of 5 cm cells refined once where the roots take up water; `LIGHT_SCENARIO`), the weather
(`weather(start)`: a clear day's PAR, the air's temperature and humidity in a daily course, a rain of 5 mm on the
second day, `forcings.pulse`: its total reaches the soil whatever the step), the tables (`TABLES`) and the frames'
variables (`FRAME_VARIABLES`).

Every step is an hour from the sowing: an older plant is a resumed run (a checkpoint), not a later start of one
model. The first blade emerges after about 100 °Cd (a week); before, the plant has no light and transpires nothing.
Every model reads the weather over its step (the forcings' means over the hour, the rain's total): the results do
not depend on the time step through the forcings. The run prints nothing: what it gives is in its folder, after
every hour (the tables can be read while it runs):

- `Plant/WheatPlant/summaries.csv` and `Soil/RichardsSoilModel/summaries.csv`: every component's state variables per
  plant (those of the components on the SubOrgans, Adel's, ArchiSimple's, the gas exchange's, the light's, too: the
  lengths summed, the potentials, conductances and light averaged by their declared weights) and for the soil;
- `leaves.csv` (`models.LEAVES`): the blades' green area, PARa, An, gs and leaf water potential (over the blade
  elements of some green area, An per area times the area summed, the means weighted by the green area);
- `roots.csv` (`models.ROOTS`): MECHA's psi averaged over the root sections by its declared weight, the liquid volume
  (the soil nodes and the air spaces have none), and their uptake;
- `plant_water.csv`, `leaf_water.csv`, `soil_water.csv` (`models.WATER_TABLES`, decision U-6: the plant and the soil
  exchange once per step): the plant's uptake and transpiration, what the soil gave the roots, the largest changes
  of the plant's and the soil's water potentials since the hour before (`Summary(max_change=)`, from the values
  before the step the Scene keeps: `ds.previous`);
- `conservation.csv`: the plant's and the soil's water balances, each conserved quantity's inflow, outflow, change and
  residual per step (their solves' declared conservation: MECHA's `PLANT_WATER`, the soil's `WATER`; the residuals
  are round-off);
- `run.json`: the run's progress after every hour, then its end, with the wall time per model and per component
  (`timing`), the calls per component, the threads chosen and each kernel's and solver's time and threads
  (`compute`) and the frames' archive;
- `frames/`: a frame every 12 hours with its views (`frame_views="default"`), the sections of the first plant
  (`display_plants=["p0"]`) drawn in the frames of every 24 hours and the last (`display_period`), packed at the end
  into `frames.fspm.zip`;
- `checkpoints/`: one a day (the last two kept; the anatomy templates and the reduced sections saved once, in
  `checkpoints/shared`): `Scene.load_checkpoint` resumes the run from it, bit for bit.

The run's configuration, `Compute(threads="auto", workers="auto")` (metafspm's default, shown in `run.py`): the
processors for numba's kernels, the plant system's blocks and the reduction's kinds (each kind's Kron reduction in
a thread), each part taking those that pay (the plant system's blocks in the count found fastest on its own solves:
4 on this laptop; BLAS stays at one thread at any `threads`: the results are the same bit for bit at any number;
8 plants, day 13: 4.35 s per hour at 1 thread, 4.08 with "auto", measured 2026-11-08 after the generic kernels'
GK-1 to GK-5, against 4.93 and 4.75 the same day before them), the memory freed given
back after each hour (`malloc_trim`), and the environment's `MALLOC_ARENA_MAX=2` (`environment.yaml`'s
`variables:`; outside the conda environment, `export MALLOC_ARENA_MAX=2` before running). Its `workers="auto"` starts a pool of processes for the
reduction's kinds when a batch is worth it (the first hours, the days the organs emerge): measured on a 14-thread
laptop (2026-11-06), the same results bit for bit with 1, "auto" (6) or 4 processes; 2 plants to day 8 in 235 s
with 1, 212 s with "auto" (last day 1.97 → 1.83 s per hour, first hour 6.3 → 4.1 s), 208 s with 4 (1.68 s), the
memory's peak, the processes included, 3.4, 4.3 and 5.0 GB; 8 plants to day 4, last day 1.08 → 0.97 s; 16 plants,
1.78 → 1.52 s. After the generic kernels (2026-11-08), 8 plants to day 8 take 275 s with 4 processes and 298 s
without the pool at `threads="auto"` (the kinds' threads), the scope's peak 3.8 and 3.2 GB: the pool stays the
faster where batches are large, at about 0.6–1.2 GB more. `benchmark.py --workers 1|auto|n --linear-solver superlu|superlu_reuse|cholmod|pardiso|mumps` compares them
(`--threads auto|n`: the budget of numba's kernels and the plant system's blocks, "auto" by default; BLAS stays at
one thread; the benchmark's entry records the threads chosen and each kernel's and solver's time, per step).

**Scaling** (measured 2026-11-07 on the same laptop, metafspm f3a2148, GRANAP c626201, after the incremental
topology updates; `benchmark.py --variability 1`: each plant its own root draws, the plants sharing their kinds;
under an 8 GB bound, the memory the scope's peak, the pool's processes included; each row the day's mean step, and
before them, at metafspm d2e708e, every step's results identical):

| plants, day | workers | s per step (before) | kinds evaluated that day (s per step) | nodes | memory peak |
|---|---|---|---|---|---|
| 2, day 8 (identical plants) | auto (7) | 1.44 (1.49) | 139 | 13,924 | 3.6 GB |
| 8, day 8 (the day-13 run) | 1 | 2.65 (2.85) | 115 (0.60) | 56,598 | |
| 16, day 8 | 2 | 4.06 (4.59) | 122 (0.46) | 113,438 | 5.2 GB |
| 8, day 12 (the day-13 run) | 1 | 4.86 (5.68) | 91 (0.50) | 193,672 | |
| 8, day 13 | 1 | 6.05 (7.21) | 83 (0.44) | 252,478 | 6.2 GB |
| 8, day 14 (metafspm db12072; stopped at h334 by the run's 6.5 GB guard) | 1 | 7.8 at h313–329, 8.5 at h333 (9.8, 10.2: final pass, GRANAP 9464738) | 60 | 326,034 | 7.0 GB |

The kinds saturate (a few hundred a day in the first week, 50 to 90 a day from day 9: the organs' new classes, a
few tenths of a second per step); the step grows with the plants' nodes (the roots' sections and their kept nodes:
about 25 µs per node per step from day 10 on, MECHA's plant solve first, then GRANAP's anatomies and the topology
updates), and so does the memory (about 10 kB per node beyond 1.5 GB). Extrapolated, 40 plants at day 14 would hold
about 1.7 million nodes, take about 40 to 45 s per step and need 13 to 18 GB: beyond this laptop (the run is a
workstation's, or fewer plants, or a coarser root anatomy). On a large scene the pool's forked processes add 1 to
2.5 GB (the pages of the main process they touch are copied): beyond about 150,000 nodes, `--workers 1` or `2` keeps the memory down at little cost (the
kinds take a few tenths of a second per step there). Recorded (`benchmark.py --frames`: a frame a day, p0's sections
drawn as instances), 8 plants to day 8 take 357 s (334 s unrecorded), the frames 1.1 MB at the sowing to 25 MB at day
8 (about 0.45 kB per node), the parts 81 MB (77 files), the folder 176 MB.
On a machine with an NVIDIA GPU and CuPy, `benchmark.py --device cuda` runs the kinds' Kron reductions and the
linear solves on the device (metafspm's docs/gpu.md, "Running on a GPU": what is offloaded, how to compare it with
the CPU run, the transfers in the entry's `device`); not measured yet (no GPU on the development laptop).

`benchmark.py` times it (its `build_scene`: `run.py`'s scene, in memory, with the benchmark's options; `python
benchmark.py [--plants 2] [--days 3]`, under a memory bound): each hour's time by
model, the kinds of the reduction evaluated, the nodes, the memory (after each hour, after `malloc_trim` and the
peak), written to `outputs/benchmarks/<commit>.json`, one entry per mode; `--checkpoint-at 168` checkpoints the
scene after those hours and `--resume <checkpoint>` runs on from one (its kinds loaded, none evaluated again).
`--scenario KEY=VALUE ...` updates the plants' scenario, e.g. MECHA's lumped shoot nodes, `--scenario
leaf_ports=bundles vapour_path=air_spaces` (`plant_reduction`'s options: the shoot's conduits per vascular bundle,
a leaf's evaporating walls per air space); each step records the reduced groups' dense blocks (their nonzeros), the
end the kinds held and their pickled size. `--memory` adds what the scene holds at the end by owner
(`memory_probe.report`: the DataStructure's columns, the MPG's properties and topology, the Instances, the GRANAP
templates, the kinds' parts, the group tables, the libraries, the solver's caches, the pool's processes, and the
process before and after `malloc_trim`), `--tracemalloc 8` the allocations alive at the end by line and by stack
(about 8 times slower), `--thread-scaling` the last step's large linear solves, its parallel kernels (a
reduction's terms among them) and the memory bandwidth timed at 1 to 12 threads, `--threads` the run's
(`Compute(threads=)`); `memory_probe.profile_split` splits a `--profile-hours` profile into Python overhead and numeric
work by call site (devplan_scene_parallel.md, P0).

The use case keeps MECHA's defaults (each conduit and evaporating wall its own node). Measured with the options (2
plants, day 8, against the defaults; 2026-11-06): `leaf_ports="bundles"` keeps 12,060 nodes (13,924), 1.09M dense
nonzeros (1.96M), the plant solve 19 s over the run (30 s; the step 1.66 s on the last day, 1.77 s), the kinds 494 MB
(613 MB), the transpiration, uptake, An and gs within 0.04 %, the leaves' water potential 0.45 % above (0.8 % of
their drop below the roots'); with `vapour_path="air_spaces"` too, 10,250 nodes, 0.30M nonzeros, the solve 12 s (the
step 1.55 s), the kinds 337 MB, the fluxes within 0.2 %, the leaves' water potential 3 % above. Both are options
only, not recommendations, at any scene size. `leaf_ports="bundles"` gives up each vessel's own axial path and its
own junction to its successor section (a bundle's vessels are held at one potential and joined bundle to bundle),
and the leaves' water potential, which Tuzet's stomata read, comes out biased (beyond the 0.5 % bound as the plants
grow). At 16 plants (their own root draws), day 8:
`leaf_ports="bundles"` keeps 98,526 nodes (113,438), 8.8M dense nonzeros (15.7M), MECHA 1.44 s per step (2.18;
the step 3.28 s, 4.66 measured before GRANAP's faster junctions, 5f5b34b), the scope's peak 4.7 GB (5.3), the transpiration and uptake unchanged, gs within 0.04 %, the
leaves' water potential 0.36 % above on average, 0.54 % at most (0.8 % of their drop below the roots').

**Running a model less often** (a schedule step's cadence, metafspm's `Step(cadence=)` or the model's scenario
`{"cadence": {step: duration}}`; measured 2026-11-07, 2 plants, 3 days from the sowing, `benchmark.py --cadence
WheatPlant.anatomy="3 h" ...`, against the hourly run; the steps' time without the build). The use case keeps every
model hourly:

| | calls | steps (hourly 56 s) | the model's own time | what changes |
|---|---|---|---|---|
| GRANAP's anatomies every 3 h (`WheatPlant.anatomy="3 h"`) | 24 | 52 s | 10.0 → 8.8 s; MECHA 34 → 31 s | fluxes within 1e-7; the roots' mean ψ up to 74 hPa off (its range −131 to −50 hPa): the sections wait for their anatomy |
| GRANAP's anatomies every day | 3 | 36 s | 10.0 → 7.6 s; MECHA 34 → 17 s (fewer sections), the kinds 27 → 11 s | fluxes within 1e-6; the roots' ψ up to 98 hPa off (−4 instead of −102 hPa at the first day's end) |
| Caribu's whole light (lighting and PARa) every 3 h, from the interval's start | 24 | 50 s | 2.8 → 1.0 s | PARa held at its 3-hour mean: up to 36 % off at an hour; over the run PARa +1.5 %, An +6 %, transpiration +0.5 % |
| Caribu's light every day, from its start | 3 | 52 s | 2.8 → 0.2 s | PARa held at the day's mean, night included: transpiration ×1.9, An −14 % |
| the former light model's `interval` of 3 h or 1 d | 72 | 52 s | unchanged | identical: the light is relit whenever elements appear (its geometry_input), every hour here |

A call of a cadenced component costs what it builds (the sections new since the last call), so GRANAP gains little
until the day; the whole light's cadence holds PARa (its output) between calls. Caribu's light model now schedules
its lighting (`CaribuLightModel.light`, its rate stage, `CaribuLightModel.light="3 h"`) apart from its PARa (its
state stage, every step), the lighting relit early when elements appear (its geometry_input), its call then
covering to the interval's end. Measured the same way (`CaribuLightModel.light="3 h"`): the light's time is
unchanged (2.8 s: elements appear every hour while the plants grow, so it is relit every hour), and its relative
light, the mean over the rest of the interval rather than over the hour, moves PARa up to 38 % at an hour (An up to
41 %, transpiration 26 %). A sparser lighting would need the relight rule relaxed.

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
per template and per kind, in `frames/parts/` (listed in `scene.fspm.json`), the template's shape and nodes (their
`tissue`) and the kind's recovery operator; `psi` on every cell is a derived field, rebuilt for the sections you
select. In ParaView (encoding3d's plugin): show the hidden layer "WheatPlant GranapAnatomy instances", pick a
section, apply "encoding3d Select at scale" (scale SubOrgan): the section is drawn with `psi` on each cell (or tick
the reader's "Evaluate Derived Fields" for every section). `run.py`'s last line, `scene.recorder.pack()`, writes
`frames.fspm.zip` (deflate, each frame an unchanged member: 16 MB for the 47 MB folder of a 3-day run), which
ParaView opens as the folder (`encoding3d.read_index` and `read` too; `encoding3d unpack` gives the folder back).
`pvbatch encoding3d/paraview/check_in_paraview.py --instances shots outputs/<run>/frames/scene.fspm.json` checks it
by hand (it passed on a 2-day run: three sections selected, their 30,065 and 3,420 cells' `psi` equal to
`encoding3d.evaluate`'s).

Measured with `benchmark.py --outputs` (2026-11-06; the display frames draw p0's sections, the use case's frame
variables; "+parts": the parts the frame added to `frames/parts`, the second display frame one hour later):

| | data frame | display frame, instances (+parts) | display frame, full | 2nd frame, instances (+parts) | 2nd, full | 2 frames' folder: instances / packed / full |
|---|---|---|---|---|---|---|
| 1 plant, day 3 | 0.7 MB | 0.8 MB, 1.2 s (+33 MB) | 33.0 MB, 3.1 s | 0.8 MB, 0.3 s (+2.6 MB) | 33.6 MB, 3.1 s | 37 / 14 / 67 MB |
| 2 plants, day 3 | 1.4 MB | 1.4 MB, 1.3 s (+33 MB) | 33.7 MB, 3.1 s | 1.4 MB, 0.4 s (+2.6 MB) | 34.3 MB, 3.1 s | 39 / 14 / 68 MB |
| 1 plant, day 8 | 2.9 MB | 3.0 MB, 2.3 s (+52 MB) | 143.5 MB, 11.1 s | 3.1 MB, 0.9 s (+5.0 MB) | 145.3 MB, 11.5 s | 63 / 24 / 289 MB |
| 2 plants, day 8 | 5.8 MB | 6.0 MB, 2.9 s (+52 MB) | 146.5 MB, 11.5 s | 6.0 MB, 1.4 s (+5.0 MB) | 148.3 MB, 11.3 s | 69 / 25 / 295 MB |

A display frame costs about 0.1 MB more than the data frame; the parts are written once (the templates, then a
kind's operator, 0.6 to 5.7 MB as its sparse system, when a new kind is drawn: they saturate with the kinds), so a
run's frames grow by the kinds drawn, not by the frames. A 2-day run (1 plant, 5 frames, 3 drawing the sections):
frames 0.3 to 0.6 MB, parts 34 MB, the folder 37 MB, packed 14 MB. A checkpoint is 8 MB at day 3 (1 plant), 36 MB at
day 8 (2 plants), the anatomy templates and the kinds (0.4 to 0.9 GB) written once in `checkpoints/shared`.

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
