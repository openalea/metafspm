# Plan: a scene of plant populations in one process

Your questions of 2026-10-04 (1–3) are answered in §1. Questions for you are in §4, with answer lines; edit freely. This plan continues step 5 of `devplan_datastructures.md`, and replaces sub-steps 5c to 5e of `docs/design/population_and_performance.md`. 5a is done (`bb36bd4`); 5b (tree kernels) is unchanged and becomes P3 below.

## 1. Answers to your questions

**1) Environment models as ordinary components, with the exchanges handled by metafspm.**

**Yes. Most of it is already framework code.** Since WD.5:
- `coupling/coupler.py` holds the `VoxelLocator` (segment barycentres to cells, periodic in x and y) and the `Coupler` (`update_map`, `push` sums plant fluxes into cells, `pull` gathers cell states back to segments);
- `Coupler.from_translator` builds them from the translator links between the soil and the plant components;
- `Transport` exists only to carry the same exchange over shared memory between processes.

The downstream soil composites still own the mapping (`compute_mtg_voxel_neighbors_fast`, `apply_to_voxel_fast`, `get_from_voxel_fast` in RhizoSoil; `DSFakeSoil` builds Couplers per plant). In one process, that ownership can move to the scene:
- **Environment models become FunctionalComponents** like the others: the soil on an `ArrayDataStructure` grid; an atmosphere or energy-balance model on a voxel grid, like RATP; the light model on its own structure (§2, P5). They solve their own equations and know nothing about plants.
- **The exchanges between DataStructures are translator links whose two ends are on different DataStructures.** metafspm turns each into a **cross-DataStructure mapping**: a sparse incidence matrix `M` (cells × plant elements), computed from the elements' positions, and recomputed only when one side's topology changes (growth) or its geometry is updated.
  - **Plant → grid:** extensive variables are summed (`M @ x`, today's `push`), intensive ones are volume-weighted (`push` does not average them today).
  - **Grid → plant:** cell values are gathered (`Mᵀ`-indexing, today's `pull`).
  - The D9 defaults from `state_variable_type` apply, as within one DataStructure (step 3a).
- **The neighbourhood is inspired by RhizoSoil, without processes:** the barycentre of each segment locates its cell (today's `VoxelLocator`). An option, `overlap`, splits a segment that crosses several cells in proportion to its length in each, which is more accurate for long segments (Q2).
- **The light model is the special case:** its exchange is geometric (triangles, ray casting), not a grid incidence. It stays a component that reads the scene geometry (`x1 … z2`, radii, areas) of every population and writes per-element interception back. The light component is then a component with access to several DataStructures (Q3).

**2) The scene as a CompositeModel.**

**Yes.** A `Scene(CompositeModel)` would hold:
- the **environment components** on their grids;
- one or more **populations**: one MPG per plant model (sub-population), with that model's components on its `MPGDataStructure`;
- the **translator**: links inside each population, as today, and the cross-DataStructure links of point 1.

`Scene.__call__` runs **the environment components first, then each population's components**, sequentially, within one iteration (your S6 answer). Exchanges happen at the links' scheduled points (Q1).

**Planting:** today's `stand_initialization` (rows, density, depth, scene ranges, model frequency) gives a planting table: position, depth, rotation, model and per-plant parameters. The scene builds each population's MPG from it:
- one Plant-scale vertex per plant;
- its initial structure from the growth component's initialisation, placed at the plant's position and rotation;
- Plant-scale variables `x`, `y`, `z` and `rotation`.

**Intercropping:** two populations (two MPGs), each with its components, run one after the other. They interact only through the environment, at the next step, as you describe. If both populations use the **same component classes** (e.g. two cultivars of one model), the Choregrapher's binding by class name (DS13) must be fixed first (Q8).

**3) Heterogeneity between plants.**

The step 1 contract already gives both options, with no new mechanism:
- **Homogeneous:** a parameter declared without a scale stays a scalar attribute of the component (`self.k`), or is declared `location="scalar"`. It is the same for all plants.
- **Heterogeneous:** a parameter declared at **Plant scale**, `parameter(..., scale=scales.Plant)`, is stored with one value per plant (location `"Plant"`). With `location="node", mapping="broadcast"`, the equations see one value per segment, its plant's.
- A component's equations **solve on the whole population at once** either way: vectorised steps on all segments, graph systems per connected piece, i.e. per plant (S2).
- **Where the per-plant values come from:** the planting table, either from a scenario that draws them from distributions (mean, spread, law, seed per plant), or from a user table (Q6).

**Consequence for coupling-scale reasoning.** Per-plant outputs (e.g. a plant's total uptake) are aggregations at the `"Plant"` location (`sum` up, D9). The Logger can summarise per plant with `summarize` / `to_dataframe(location="Plant")`.

## 2. Steps

Each step has its design detail in a short note before code (complex steps), its tests, and a commit. UC1–UC5 and the contract tests stay unchanged.

| Step | Content | Notes |
|---|---|---|
| **P1** ✓ | Scene robustness (F1, agreed): the main loop stops on a worker's non-zero exit, then terminates blocked workers after a grace period; tested with a plant whose constructor raises | independent, small |
| **P2** ✓ | Population at scale (F2, agreed): incremental `update_topology()` in segment mode (only new segments get Compartments and Connections; vids and edge values kept), and bulk vertex creation in the MPG (one batched `ArrayDict` assignment per property) | measured on 2·10⁶ segments |
| **P3** ✓ | Tree kernels (5b, agreed): `chain_scan`, `accumulate`, `path_window`, `chain_shift` / `chain_write`, vector-valued values, minimal `path_compose`; reference loops from rhizodep, cnwgrass, adel and GRANAP rules | needed by growth at population scale |
| **P4** ✓ (emergence moved to P6) | Population builder and planting: a planting table from `stand_initialization`; one MPG per sub-population (Plant vertices; initial structures placed by position and rotation; Plant-scale `x, y, z, rotation`); per-plant parameters from the table or from distributions | Q4–Q6 |
| **P5** ✓ | Cross-DataStructure links: `CrossMapping` (incidence matrix from a locator: barycentre, or length overlap), recomputed on topology or geometry changes; translator links between DataStructures become mapped exchanges with D9 defaults; the light model as a component reading several DataStructures | generalises `Coupler`; Q1–Q3 |
| **P6** ✓ (with Q5 emergence) | `Scene(CompositeModel)`: environment components and populations, `__call__` order (environment, then each population), one or several populations (intercropping), the Logger per population and per plant | Q7–Q9 |
| **P7** ✓ | Benchmarks and decision: time per step for 1 to 1000 plants of about 2 000 segments, with and without anatomies, split by phase, against today's one-plant-per-process scene; then decide on `play_Orchestra` (Q10) | |
| **P8** | DS13 (per-instance scheduling) if Q8 needs it; DS15 (persistence, agreed: npz + JSON) | |

Order: P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8. P1 can go first because it is independent.

## 3. What stays as it is

- Plant components: FunctionalComponents and StructuralComponents on the population's `MPGDataStructure`, unchanged except that they see many plants.
- The translator format: Python or YAML, nested; the cross-DataStructure links are ordinary links.
- `play_Orchestra` and `Transport` keep working for one plant per process until P7 decides (Q10).

## 4. Questions

- **Q1, when exchanges happen.** Cross-DataStructure values could be exchanged at **fixed points**:
  - environment outputs pushed to the plants after the environment components run;
  - plant outputs pushed to the grids after the plants run.

  That is today's protocol, and it keeps each side's values constant during its own step. Or exchanges could be **lazy**, like derived variables within one DataStructure (D10): recomputed whenever read after a source changed. **Recommendation:** fixed points, so the scheduling is explicit and reproducible, with the lazy mode kept for derived variables inside one DataStructure.
  → answer: yes keep fixed points only
- **Q2, the plant ↔ grid incidence.** Should it use the segment **barycentre** (today's `VoxelLocator`, RhizoSoil's logic), or the **length overlap** of each segment with each cell (a segment crossing two cells shares its flux between them)? **Recommendation:** barycentre by default (identical to today), with `overlap` as an option.
  → answer: Agree with recommendation
- **Q3, the light model.** Is it acceptable that the light component reads the geometry of every population directly (positions, radii, areas, labels) and writes interception per element, with no grid in between? The alternative is to pass through a voxel grid, like RATP. CARIBU-like models need triangles, so the direct route is the general one. **Recommendation:** direct, with the geometry variables declared as the light component's inputs from each population.
  → answer: Like any other component, the light model can work either on an MPG DataStructure directly as caribu's example, or on a grid like RATP would. So both types of light models could be possibly declared, and it is only the datastructure coupling that gives the per-element and per-cell correspondance when there is a need to couple to a grid.
- **Q4, the plants' initial structure.** Is a population built by replicating **one template plant** (the growth component's initial MTG), translated and rotated per position? Or does each plant initialise itself (e.g. different seed reserves or initial axes, drawn per plant)? **Recommendation:** the growth component's initialisation is called per plant with that plant's parameters, so heterogeneity in the initial state is possible, and the result is one MPG.
  → answer: yes to recommendation
- **Q5, germination and emergence dates.** Should plants be able to start at different dates (staggered sowing or emergence)? With an `"active"` mask at Plant scale (step 2c/2d), dormant plants would be skipped by vectorised steps and graph systems.
  → answer: yes not priority task but still implement.
- **Q6, heterogeneous parameters.** How should they be specified: per parameter, a law (normal, lognormal, uniform), a mean, a spread and a seed, in the scenario; and/or a user table (one row per plant)? Which parameters are typically heterogeneous?
  → answer: Should be possible for any parameter so should not be specialized at this stage. Currently I would imagine passing a list a scenarios instead of just one when this mode is not activated. That way, I can implement the way I want the statistical repartition later.
- **Q7, logging.** Per-segment outputs of 1000 plants are large (2·10⁶ rows per logged step). Should the Logger write per-plant summaries every step and the full per-segment state only every `heavy_log_period`, as today, and only for selected plants (`log_only_one` generalised to a list)?
  → answer: Only for selected plants as today with an option.
- **Q8, two populations of the same model.** Two cultivars of one model can be:
  - one population with a `cultivar` label and per-plant parameters (no DS13 needed);
  - two populations with the same component classes (DS13 needed: per-instance scheduling).

  Which do you expect to use?
  → answer: First option, one population, only if these are completely different models 
- **Q9, time steps.** Do environment components and populations share the scene's time step, with sub-stepping inside components (DS10)? Or should some environment components run less often (e.g. light every 24 steps), with their outputs held in between?
  → answer: First option
- **Q10, `play_Orchestra`.** After P7: keep it for runs that do not fit one process (splitting the population over a few processes by blocks of plants, with the same exchanges through `Transport`), or retire it? **Recommendation:** decide after P7's measurements; keep it working until then.
  → answer: remove it after P7

## 5. Decisions from your answers (2026-10-05)

- **Q1:** exchanges between DataStructures happen at fixed points only: environment outputs are pushed to the plants after the environment components run, and plant outputs to the environment after the plants run.
- **Q2:** barycentre by default; `overlap` as an option.
- **Q3:** a light model is a component like any other, on an MPG (CARIBU-like) or on a grid (RATP-like). Only when it is on a grid does the cross-DataStructure mapping give the element ↔ cell correspondence; on an MPG it reads the geometry directly.
- **Q4:** the growth component's initialisation is called per plant, with that plant's parameters, into one MPG.
- **Q5:** staggered emergence through a Plant-scale `"active"` mask. To be implemented, but not a priority (end of P4 or P6).
- **Q6:** any parameter may differ between plants, with no specialisation now. The scene receives **a list of scenarios, one per plant**, instead of one scenario; the statistical repartition is built by you later, upstream.
- **Q7:** the Logger writes the selected plants only, as today, with an option.
- **Q8:** cultivars of one model are **one population** (labels and per-plant scenarios). Two populations only for completely different models, so their component classes differ and DS13 is not needed for that. P8 keeps only DS13's hazard fix (same-named classes from different modules, several instances of one class in tests) and DS15.
- **Q9:** one time step for the scene; sub-stepping only inside components (DS10).
- **Q10:** `play_Orchestra` and `Transport` are **removed after P7**.

## 6. Follow-up question (P4 only)

- **QH1, how a per-plant parameter reaches the equations.** Today a parameter declared without a scale is a scalar attribute (`self.k`), and equations use it as such (`return self.k * hexose`). With one scenario per plant, `k` can differ between plants, and an equation then needs, for each entity, its plant's value. Options:
  - **(a) Automatic:** a parameter whose scenario values differ between plants becomes a Plant-scale variable, and `self.k` becomes a per-entity array during vectorised steps: one value per segment, its plant's. This works in `@rate` / `@state` steps on nodes. It is ambiguous in graph-system edge laws, where an edge needs a per-edge value: for an edge, the plant of its child.
  - **(b) Declared:** a parameter that may vary between plants declares where it is used, `parameter(..., scale=scales.Plant, location="node", mapping="broadcast")`, and the equations take it as an argument (`def _rate(self, hexose, k)`). A scenario list making an undeclared scalar parameter differ between plants raises, naming the parameter and the declaration to add.

  **Recommendation:** (b). It keeps every equation's inputs explicit and works the same in steps and graph systems. Any parameter can be made heterogeneous by its declaration, with no specialisation of the mechanism. (a) can come later as a convenience for `@rate` steps if wanted.
  → answer: I am rather for a systematic storage of parameters at plant scale, even if they don't vary, to keep the equation writing the same everywhere and only if the input scenario varies it is mapped as heterogeneous for each plant. But how to differentiate parameters from variables in method's arguments in this case, while still ensuring methods are still vectorizable / numba compatible?

## 7. Parameters stored at Plant scale (your QH1 answer, 2026-10-06)

You chose: **every parameter is stored at Plant scale**, even when it does not vary, so that equations are written the same way everywhere. Only when the input scenarios differ between plants do the values differ. Your question: how are parameters told apart from variables in a method's arguments, while methods stay vectorisable and numba-compatible?

**Answer: by their declaration, not by the method's signature.** Nothing in the signature needs to change.
- The framework already knows each argument's declaration. When a step or an equation is called, it looks up every argument name in the component's resolved declarations (`VariableSpec`, step 1a): `variable_type` says parameter, input or state variable, and the declaration also gives the location and mapping. The method stays `def _rate(self, hexose, k)`.
- **What each argument receives:** always an array at the equation's entity, so the types are uniform and numba compiles one specialisation:
  - a state variable or input: its node (or edge) array, as today;
  - a parameter: its Plant-scale values **broadcast to the equation's entity**, so each segment gets its plant's value:
    - **heterogeneous:** an array gathered through the owner map (step 1 `broadcast`, step 2a `owner("Plant")`), cached as a derived variable and recomputed only when the parameter or the topology changes (D10);
    - **homogeneous** (all plants equal): a **zero-stride view**, `np.broadcast_to(value, (n,))`. It costs no memory, numba reads it like any array, and it is read-only, so an equation cannot overwrite it (4a).
- **For graph systems,** the entity is known from the decorator: node balances get node arrays, and edge laws get edge arrays, with each edge taking its child segment's plant. An edge never joins two plants (P2 keeps plants disconnected), so this is unambiguous.
- **numba:** a step compiled with `@njit` receives float arrays only, never Python scalars, so the same signature serves the homogeneous and the heterogeneous cases. Read-only and zero-stride arrays are supported by numba (they compile as `readonly`, layout `A`).

**Where the values come from:**
- one scenario per plant, as Q6 decided;
- the scene fills each Plant-scale parameter with the values of the plants' scenarios, or with the declared default;
- a single scenario for all plants gives the homogeneous case, with no special code.

**What changes for model writers:**
- equations take their parameters as arguments, like their variables;
- `self.k` remains readable when the population is homogeneous, and raises when the parameter differs between plants, naming the arguments to use instead (QH2);
- grids and other DataStructures without a Plant scale keep parameters at the `"scalar"` location, also passed as zero-stride arrays, so environment components are written the same way.

### Questions

- **QH2, `self.k`.** Keep `self.k` readable (one value) when the parameter is homogeneous, and raise when it is not? Or forbid it always, so that equations are written with arguments only, and a model that is correct in a homogeneous population cannot fail later in a heterogeneous one? **Recommendation:** forbid it inside steps and equations (a clear error naming the argument to add), and keep it readable outside them (initialisation, scenario setup) as the population value when homogeneous.
  → answer: follow recommendation
- **QH3, the default.** Should `parameter(...)` **without a scale** now default to the Plant scale on plant DataStructures (and `"scalar"` on grids)? That way every existing parameter declaration becomes population-ready with no change. A parameter that must stay a plain Python value (e.g. a configuration flag or a file path) would be declared `dtype="object"` or kept out of `declare`. **Recommendation:** yes, for numeric parameters.
  → answer: yes

## 8. P4 design: population builder, planting and Plant-scale parameters (draft, 2026-10-06)

QH2 and QH3 are agreed: `self.k` is forbidden inside steps and equations, and numeric parameters default to the Plant scale.

**Parameters**
- **Storage.**
  - On a plant DataStructure, a numeric `parameter(...)` declared without a scale is stored at the `"Plant"` location: one value per plant, filled from each plant's scenario, or from the declared default.
  - On grids, and on DataStructures without a Plant scale, it is stored at `"scalar"`.
  - Parameters declared with a scale keep today's behaviour. Object or non-numeric parameters (`dtype="object"`, paths, flags) stay plain attributes.
- **In steps and equations:** every argument is resolved by its declaration. A parameter is broadcast to the equation's entity:
  - heterogeneous: owner-map gather, cached and recomputed on change;
  - homogeneous: a zero-stride read-only view.

  Steps (Functor) and graph systems (snapshot) both do this.
- **`self.k`:** for each numeric parameter, the class gets a data descriptor (installed once per class, at its first instantiation):
  - **inside** a step or an equation (a flag set by the Functor and the builder while they evaluate), reading `self.k` raises `AttributeError("... declare k as an argument of _rate")`;
  - **outside,** reading gives the population value when homogeneous, and raises when heterogeneous;
  - **writing** `model.k = 0.2` (scenario setup, tests) sets every plant's value. Writes before the DataStructure is bound (the dataclass `__init__`) are kept and applied at registration.
- **Migration in this repo:** the test models reading declared parameters through `self.` inside equations (about 20 reads: UC1, UC3/UC4, the doubles, the growth helper) take them as arguments. `self.time_step`, `self.dt` and `self.previous()` are not declared parameters and stay.

**Population and planting**
- **Planting table:** from `stand_initialization` (positions, rotations, model per position, depth) plus one scenario per plant, as a `pandas.DataFrame` with one row per plant. For P4 a helper builds it from the existing arguments (`sowing_density`, `row_spacing`, …); P6 wires it into the Scene.
- **Plant vertices:** `build_population(g, table)` adds one Plant-scale vertex per row to a new MPG, with Plant-scale variables `x`, `y`, `z`, `rotation` and `model`.
- **Initial structures (Q4):** the population's StructuralComponent provides `initiate_plant(g, plant_vid, parameters)`, called once per plant. It builds that plant's initial structure (seed, first axes) under its Plant vertex, from its own scenario's parameters, positioned at the plant's coordinates. This is a new contract for structural components; rhizodep's `initiate_mtg` is the model of it.
- **Then:** `populate_graph`, one `MPGDataStructure` for the population, and every component constructed once on it.
- **Staggered emergence (Q5, not a priority):** an `emergence_time` Plant variable and a Plant-scale `"active"` mask, broadcast to segments, so that dormant plants are skipped (steps 2c/2d).

**Validation**
- 1 plant and 100 plants built from one scenario give identical per-plant results for the growth helper and a transport system, and the 100-plant run is computed in one call.
- Two plant groups with different scenarios give each group its own values, compared against separate one-plant runs.
- `self.k` inside a step raises with the argument to add; `model.k = …` sets every plant's value.
- Zero-stride parameters are accepted by a numba-compiled step.

### Questions

- **QP4a, `initiate_plant`.** Is this contract right for the structural components you will port (rhizodep's seed and first axes; later cnwgrass/adel tillers and phytomers; GRANAP anatomies)? The alternative is a separate "seed" component outside the structural components. **Recommendation:** the structural component, since it owns the topology rules.
  → answer: yes, the structural component will hold the initialization topology rules, probably several structuralcomponents will need to operate successively to initiate the plant before first execution / initialization of FunctionalComponents in __init__
- **QP4b, positions.** Should the segments' coordinates (`x1 … z2`) be computed by the structural component from the plant's `x, y, z, rotation`, as rhizodep's turtle does today from the plant's origin? Or should the scene apply each plant's translation and rotation to coordinates the component computed in a local frame? **Recommendation:** by the component, from Plant-scale inputs; the scene only provides the plant variables.
  → answer: This can be the job of the structural component / the component creating the datastructure. The datastructure translator should then operate from initialized variables to find neighbors and pass variables.

## 9. P5 design: links between DataStructures (draft, 2026-10-06)

**What exists.** `Coupler` maps each plant node to the cell holding its segment barycentre (`VoxelLocator`):
- `push()` scatter-adds plant fluxes into soil cells;
- `pull()` gathers the soil state of each node's cell.

It is plant ↔ soil only. The map is rebuilt by hand (`update_map()`), the soil inputs must be zeroed before several plants push, and the D9 defaults are not applied. `CompositeModel._couple_on_data_structures` refuses links across DataStructures.

**Proposal.**

1. **`CrossMapping(source_ds, target_ds, locator="barycentre" | "overlap", ...)`:** a sparse incidence (rows = source entities, columns = target cells, weights).
   - *Barycentre*: one cell per segment, weight 1. This is today's map, keeping the `flip_z` and periodic options.
   - *Overlap*: each segment is cut by the cell faces (a 3-D grid traversal), weight = its length fraction in each cell.
   - *Recompute*: automatic, at the next exchange after the source's `topology_version` or a coordinate variable's write count changed. `update_map()` is no longer needed.
2. **Exchange links.** A translator link whose receiver and provider are on different DataStructures becomes an exchange through the mapping of that DataStructure pair. Its default follows D9 from the variables' kinds:

   | Direction | Extensive (fluxes, amounts) | Intensive / massic (concentrations, temperature) |
   |---|---|---|
   | plant → cell | sum, split by weight | weighted mean (`weight=` required, e.g. `length`) |
   | cell → plant | split by weight (`weight=` required) | weight-averaged gather |

   - Both sides are checked with `kinds_agree`, and an explicit `aggregation=` overrides the default, as within one DataStructure.
   - A formula link is evaluated on the source DataStructure, then mapped.
   - Several sources feeding one target variable (several populations into one soil) are **summed in one exchange**, written by one `set()`. This removes `zero_soil_inputs()` and its ordering hazard.
3. **Fixed points (Q1).** The links are grouped per receiving DataStructure, and `exchange(into=ds)` runs them all. `couple_components` builds these exchanges; P6's Scene calls `exchange(into=population)` after the environment components, and `exchange(into=environment)` after the populations. Until P6 the exchanges are called by hand, as `push` / `pull` are today.
4. **`Coupler`** is reimplemented on `CrossMapping` (same results, tested bit for bit against today's map), so `Transport` keeps working until P7 removes both (Q10).
5. **Out of scope:** grid ↔ grid mappings at different resolutions (soil ↔ atmosphere); masks of dormant plants (Q5, P6).

**Sub-steps:**
- 5.1: `CrossMapping` (barycentre, overlap, automatic recompute), with tests against `VoxelLocator` and hand-computed overlaps.
- 5.2: exchange links from the translator, D9 defaults, several sources summed, `exchange(into=)`.
- 5.3: `Coupler` on `CrossMapping`.
- 5.4: the light cases of QP5b, as tests.

### Questions

- **QP5a, intensive plant → cell.** When a plant variable that is intensive is sent to a cell (e.g. root surface temperature, read by a soil model), is a weighted mean with a required `weight=` right? The alternative is to refuse such links, the soil model receiving extensive variables only. **Recommendation:** weighted mean with a required `weight=`, as D9 does upward within one DataStructure.
  → answer: In practice, only extensive will be passed, but it is better to plan for any type of variable yes.
- **QP5b, light over several populations.** A CARIBU-like light component on an MPG sees only its own DataStructure. With two populations of different models (Q8: intercropping), it would not see the shading of the other population. Options:
  - **(a)** a component declared on several DataStructures (reads and writes each);
  - **(b)** a **union mapping**: the light component runs on its own flat DataStructure of scene elements, i.e. the concatenation of the populations' elements. It reads geometry and writes interception through one-to-one exchanges (rows = elements of each population, in order), so components keep one DataStructure each.
  - **(c)** for now, one light component per population, with no shading between populations.

  **Recommendation:** (b). It needs no change to the component base and fits Q3 ("only the DataStructure coupling gives the correspondence"). With one population, the light component simply runs on the population's MPG.
  → answer: go for (b)
- **QP5c, cell → plant for extensive variables.** For example, a soil supply given per cell and shared among the segments in it. Is splitting by a required `weight=` (e.g. root length or surface in the cell) the right default? Or should the cell's amount be shared in proportion to the plants' demand, which is a model process and not a mapping? **Recommendation:** split by `weight=`, with demand-based sharing left to the models (computed as an intensive rate per cell, then gathered).
  → answer:ok for recommandation, even if in practice only intensive variables will be passed, but better to plan for any variable type.

## 10. P6 design: `Scene(CompositeModel)` (draft, 2026-10-06)

**Today:** `play_Orchestra` gives each plant its own process. The plant model is a `CompositeModel` built per plant with `(name, time_step, coordinates, rotation, queues…, **scenario)`. Soil and light run in their own workers, and data goes through `Transport` buffers.

**Proposal.**

```python
scene = Scene(planting=planting_table(...),            # or a user table, one scenario per plant (Q6)
              environment=[SoilModel, LightModel], environment_scenarios=[...],
              translator_path="scene_translator.py", time_step=3600, mapping_method="barycentre",
              log_plants=["plant_0"], heavy_log_period=24)
scene.run(n_iterations=2500)
```

1. **Populations.** The planting table is grouped by model: one population per model (Q8). For each population, the Scene:
   - builds the MPG with `build_population(table, Model.initiators)`;
   - constructs the model once, `Model(data_structure=ds, time_step=…, **shared_scenario)`. The model creates its components on `ds` and couples them within the DataStructure, as today;
   - fills the per-plant parameters with `apply_plant_scenarios`.
2. **Environment.** Each environment model is built with `(populations=[ds, …], scene_xrange, scene_yrange, time_step, **scenario)` and creates what it needs:
   - a grid (soil, RATP-like light);
   - a `UnionDataStructure` of the populations (CARIBU-like light, QP5b);
   - or it works directly on the single population's MPG.
3. **Coupling.** One `Exchanges` over all components, from the scene translator:
   - one `CrossMapping(population, grid, method=mapping_method)` for each population and grid pair that a link joins;
   - one `UnionMapping` per union.
4. **Step (Q1, Q9: one time step):**
   1. the environment models, in list order, each preceded by `exchange(into=its DataStructures)`;
   2. `exchange(into=population)` for each population;
   3. each population's `run()`, which grows its MPG;
   4. the time is advanced.

   An initial `exchange(into=environment)` at construction lets the environment see the plants at t = 0.
5. **Input tables** (meteo, …) apply to the environment models, as today (`apply_input_tables`).
6. **Logging (Q7):** at each step, per-plant summaries (sums and means at the `"Plant"` location) for every plant. Every `heavy_log_period` steps, the per-segment state of the `log_plants` only. One folder per population.
7. **Staggered emergence (Q5):** an `emergence_time` column of the planting table becomes a Plant-scale variable, and a Plant-scale `"active"` mask (time ≥ emergence) is broadcast to the nodes.
8. **`play_Orchestra` / `Transport` / `Coupler`** stay until P7 (Q10).

**Sub-steps:**
- 6.1: populations and the model contract (QP6a).
- 6.2: environment models and automatic mappings (QP6b).
- 6.3: the step order and the initial exchange.
- 6.4: logging (QP6d).
- 6.5: emergence (QP6c).

All of these are tested on in-repo doubles: the growth helper as a plant model, a toy soil on a grid, and the toy light models of P5.

### Questions

- **QP6a, plant model contract.** A plant model becomes a population model: `Model(data_structure, time_step, **scenario)` with a class attribute `initiators` (the StructuralComponent classes that build each plant). It is built once for all its plants; its parameters per plant come from the planting table. The queue, coordinate and rotation arguments go away. Is this the contract for your ports (rhizodep, Root-CyNAPS, cnwgrass, GRANAP)? **Recommendation:** yes. The in-repo UC tests follow it, and the external models are ported outside this repo.
  → answer: yes to the recommendation
- **QP6b, who builds the environment's DataStructures.** The proposal lets each environment model receive the populations and build its own grid or union. The alternative is for the Scene to build them from options (`soil_grid=…`, `light="union" | "grid"`). **Recommendation:** the environment model builds them, as plant models build their MPG. The Scene only infers the mappings from the translator links and the DataStructure types.
  → answer: Follow the recommendation I agree, Some of the environment components generate the structure and expose it to the scene for coupling as root_growth would to it, while keeping the possibility for other environment components of the same compartment to operate on it.
- **QP6c, plants before emergence.** Options:
  - **frozen**: no step runs on their entities; they are excluded from the mappings (they neither push to nor receive from the environment), and their values stay at their initial state;
  - **seed-only processes**: some steps (e.g. seed reserve mobilisation) run before emergence, by components that do not honour the mask.

  **Recommendation:** frozen by default, with a component able to opt out of the mask (it runs on all plants).
  → answer: yes to recommendation.
- **QP6d, the Logger.** The fspm-utility `Logger` (outside this repo) reads `model_instance.data_structures` and components per plant. Options:
  - **(a)** a Scene-native recorder in metafspm (the summaries and selected plants of point 6, written as one table per population, e.g. parquet or netCDF), with a `logger_class` hook kept so that you can adapt fspm-utility's Logger later;
  - **(b)** a per-plant view of the population, to plug today's Logger in unchanged.

  **Recommendation:** (a). (b) would need per-plant MTG views that the vectorised populations no longer have.
  → answer: (a)

## 11. P7 design: benchmarks, then the removal of `play_Orchestra` (draft, 2026-10-06)

**Benchmark** (a script under `test/benchmarks/`, not part of the suite, with results in `docs/design/population_and_performance.md` §14):
- **Population sizes:** 1, 10, 100 and 1000 plants of about 2 000 segments each, grown up to that size before timing.
- **Phases timed per step:**
  - vectorised steps (rate/state);
  - one graph system (axial diffusion on the segments, per plant connected piece);
  - growth (`update_topology`, repartition);
  - exchanges with a soil grid (barycentre and overlap);
  - the recorder.
- **Anatomy mode:** the same with Compartment nodes (2 symplastic + 1 apoplastic Compartment per segment).
- **Reference:** today's one-plant-per-process scene (`play_Orchestra` with `Transport` buffers), on the same toy plant, for 1, 10 and 100 plants. Per plant, the time is the slowest worker's step plus the queue round trips, measured in-process with the scene's barriers.
- **Models:** in-repo doubles only:
  - a rhizodep-like carbon model (vectorised rates, one graph system);
  - the growth helper rewritten with the P3 tree kernels, so that it does not measure Python loops;
  - the toy soil of P6.

**Removal (Q10), after the benchmark:**
- `scene/scene_wrapper.py`: `play_Orchestra`, the workers, the CPU-affinity helpers and the queues. `stand_initialization` moves to `scene/population.py`.
- `coupling/coupler.py`: `Coupler`, `Transport`, `BufferPlantView`, `VoxelLocator`.
- `CompositeModel` members tied to them: `soil_name`, `soil_inputs` / `soil_outputs`, `get_component_inputs_outputs`, and the soil-output registration in `_couple_on_data_structures`.
- Their tests and doubles: `test_scene_orchestration`, `test_scene_wrapper*`, `test_transport`, `test_coupler`, `test_ds_scene_contract`, and the multiprocessing part of the doubles. The equivalent cases are re-expressed on `Scene` where they are not already covered.
- `test/provide_usage_examples/scene_wrapper_example.py`, rewritten for `Scene`.
- Docs: the migration guide's per-process section is replaced by the `Scene` contract; the CHANGELOG gets a breaking entry.

### Questions

- **QP7a, the models benchmarked.** The real models (rhizodep, Root-CyNAPS, cnwgrass, GRANAP) are not yet ported to the population contract and live outside this repo, so the benchmark can only use in-repo doubles that reproduce their cost structure (vectorised rates, graph systems, growth with tree kernels). Is that enough for the decision, or do you want to port one model first (outside this repo, by you) and run the benchmark on it? **Recommendation:** the doubles now, to time the framework's own overhead (exchanges, growth bookkeeping, the solver per piece); you can rerun the same script on a ported model later.
  → answer: yes.
- **QP7b, removal regardless of the results.** Q10 says remove after P7. If the benchmark showed that one process is too slow for 1000 plants (e.g. graph systems limited to one core), should the removal still go ahead, with the parallelisation then done inside the Scene (blocks of plants or connected pieces over threads or processes, `split="components"`, §9 S2)? **Recommendation:** yes. Parallelism inside the Scene does not need the per-plant workers or `Transport`.
  → answer: yes, if runtime without vectorised step's optimization becomes an issue, we'll consider it then, but I consider the current play_Orchestra parallelisation is not right and does not fit to all the developments we just made to build populations and cleanly couple them to environment datastructures

### P7 results and follow-up questions (2026-10-06)

The benchmarks are in `docs/design/population_and_performance.md` §14.
- **Scale:** a step of 1000 plants of 2 000 segments takes 6.7 s in one process, and every phase scales linearly.
- **Overhead:** the Scene's per-step overhead is flat (about 0.3 ms), while `play_Orchestra`'s grew to 6.6 ms for 12 toy plants.
- **Removal:** `play_Orchestra`, `Transport`, `Coupler` and the soil members of `CompositeModel` are removed (QP7b).

Two framework costs dominate. They are not model costs, and fixing them changes no API except possibly F3's:

- **QF3, writing back to the MTG.** `write_back_to_mtg` runs after every component call and costs 97 % of the rates-and-states time. Options:
  - **(a)** write back lazily: only before an MPG-style step (`_run_mpg_step` already writes the variables it reads), before a structural change, and on export / `mtg` access;
  - **(b)** a declaration per variable (`mtg=True`) listing the variables kept in the MTG at each step.

  **Recommendation:** (a). The MTG is then a view brought up to date when someone reads it; the DataStructure is the reference.
  → answer:
- **QF4, the incremental graph extension.** `extend_graph` traverses the whole MTG (`components_at_scale`) at each growth event. **Recommendation:** restrict it to the vertices created since the last extension (their ids are above the last one seen) and their complexes, so that the cost follows the growth, not the population. This is internal, with no API change; I can do it without waiting, as part of P8 or before it.
  → answer:
