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
| **P2** | Population at scale (F2, agreed): incremental `update_topology()` in segment mode (only new segments get Compartments and Connections; vids and edge values kept), and bulk vertex creation in the MPG (one batched `ArrayDict` assignment per property) | measured on 2·10⁶ segments |
| **P3** | Tree kernels (5b, agreed): `chain_scan`, `accumulate`, `path_window`, `chain_shift` / `chain_write`, vector-valued values, minimal `path_compose`; reference loops from rhizodep, cnwgrass, adel and GRANAP rules | needed by growth at population scale |
| **P4** | Population builder and planting: a planting table from `stand_initialization`; one MPG per sub-population (Plant vertices; initial structures placed by position and rotation; Plant-scale `x, y, z, rotation`); per-plant parameters from the table or from distributions | Q4–Q6 |
| **P5** | Cross-DataStructure links: `CrossMapping` (incidence matrix from a locator: barycentre, or length overlap), recomputed on topology or geometry changes; translator links between DataStructures become mapped exchanges with D9 defaults; the light model as a component reading several DataStructures | generalises `Coupler`; Q1–Q3 |
| **P6** | `Scene(CompositeModel)`: environment components and populations, `__call__` order (environment, then each population), one or several populations (intercropping), the Logger per population and per plant | Q7–Q9 |
| **P7** | Benchmarks and decision: time per step for 1 to 1000 plants of about 2 000 segments, with and without anatomies, split by phase, against today's one-plant-per-process scene; then decide on `play_Orchestra` (Q10) | |
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
  → answer:
