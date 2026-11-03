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
  - The sides: the grid's (periodic over the stand).
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
  grows the difference with the roots it grew, `'+'` on the phytomer's bearer (its first element: at the crown for
  the first phytomers). Each leaf's request is counted once, when it falls due: none is lost, whatever the steps.
- **Its entities:** Adel's elements (`entities = {"axis_row": ">=0"}`), its phytomers those they belong to.

Their tests are in `test/test_plant.py` (the example's own: `python -m pytest examples/use_case_1/test`, outside metafspm's suite, which needs no port), on a test plant model (WheatCrown, Adel,
the rule, ArchiSimple):

- the seed and the subcrown internode, at the planting's depth or the scenario's;
- Adel's main stem on the crown's top, the six seminal roots on the seed;
- each model's own entities;
- the leaves' appearance times;
- the nodal roots on the right phytomers at the right thermal times, kept;
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
- **Its scenario,** `WHEAT_SCENARIO`: the crown, `WHEAT_ON_A_SHOOT` and the anchors, GRANAP's wheat root
  (`for_wheat_root`) and monocot stem, sheath and leaf, a soil node per root section, MECHA's soil boundary on it
  (`node_dirichlet`) and no collar condition, MECHA's sections reduced onto their conduits (`reduction =
  staticmethod(scenario_reduction)`, the scenario's `reduced`, `reduced_labels`).

The links, each in its units:

| link | from | to | factor |
|---|---|---|---|
| GRANAP's stems' class input `stem_radius` | the organ's `diameter` (m; Adel's, the crown's) | its elements (m) | 0.5, broadcast |
| the gas exchange's element `height` | Adel's `z1`, `z2` | the elements | `adel_links`' formula |
| MECHA's `leaf_transpiration` | `GasExchange.transpiration` (mmol s⁻¹, per element) | per Organ (cm³ d⁻¹) | 18e-3 × 86400, summed |
| the soil's `plant_uptake` | MECHA's `root_uptake` (cm³ d⁻¹) | the cells (m³ s⁻¹) | 1e-6 / 86400 |
| MECHA's `soil_psi` | the soil's `water_potential` (MPa) | the roots' soil nodes (hPa) | 1e4 |
| `GasExchange.PARa`, the optics | Caribu's light, `PlantOptics` | both ways | the Scene's identity links |

The soil links are `component.scene_translator()`, through `root_surface_mappings` (the roots' soil nodes onto the
grid). Not linked yet: MECHA's leaf water potential to the gas exchange's (Tuzet's stomata, lagged by a step): both
are named `water_potential` (hPa at the Compartments, MPa at the elements), one variable on one DataStructure. The
gas exchange keeps Ball-Woodrow-Berry's stomata.

`run.py` runs a few plants over a few days of May, hour by hour, from their sowing (`python run.py [folder] [days]
[plants]`): a clear day's PAR, the air's temperature and humidity in a daily course, a rain on the second day. After
each hour it prints `WaterBalance`'s diagnostics (decision U-6: the plant and the soil exchange once per step):

- the plant's transpiration and uptake (equal for a steady plant), and what the soil gave the roots;
- the soil's water change and what crossed its boundaries (equal);
- the largest changes of the plant's and the soil's water potentials since the previous hour.

With a folder, the SceneRecorder writes the summaries, `water_balance.csv`, and a frame every 6 hours with its
views (`frame_views="default"`).

Its tests are in `test/test_assembly.py` (the example's own, as the plant's), on one young plant over three hours: the
links' factors, the assembly's couplings, each model's own entities, the transpiration into MECHA, the steady plant,
the uptake into the soil, the soil's potential on the roots, the light on the gas exchange, the soil's balance with
the uptake.

### The assembly: open gaps

The tests but the links' factors are marked `xfail` (strict) until these are fixed (metafspm or the ports):

- an entity view on an anatomy-mode DataStructure: its filters on a coarser variable (Adel's `axis_row`) evaluated
  at the Compartments; the rule's view at its finest scale (Phytomer) rather than the SubOrgans; its `base` the
  Compartments' DataStructure (ArchiSimple's bearers);
- the reduction's library runs every functional component of the plant model (Adel's development, without its
  tables);
- GRANAP's anatomies: a section without length (Adel's internodes and sheaths before they grow) gets one (MECHA
  divides by its length); Adel's lengths are not written to the MPG before GRANAP's step; the anatomies GRANAP makes
  at a step for Adel's new sections lose their values (leaf surfaces, network paths);
- the exchanges with an anatomy-mode plant: the light's PARa does not reach the gas exchange, the soil mapping
  reads the roots' ends at the SubOrgans while it maps the Compartments;
- a frame of an instanced MPG with Adel's per-plant tables;
- the leaf water potential's name (above).
