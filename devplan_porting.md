# Plan: what metafspm still needs before porting the existing models

Status: **agreed** (2026-10-06, answers in §4, decisions in §5). It comes from a read-only audit of the downstream code, nothing in those repositories was edited:
- **Roots:** rhizodep, Root-CyNAPS, Root_BRIDGES.
- **Shoot:** cnwgrass, WheatFspm (cnwheat, elongwheat, growthwheat, senescwheat, farquharwheat) and adel.
- **Environment and scene:** RhizoSoil, soiltemp, Wheat-BRIDGES (composite and Caribu light), and fspm-utility.
- **Anatomy:** GRANAP, and how MECHA consumes it.

Answer the questions inline (→ answer:), as in the other plans.

## 1. What already fits

Most of the per-vertex physiology and both transport solvers fit the current API:
- **Rates and states** are vectorised steps on arrays, with parameters per plant. A step with several outputs (rhizodep's `(value, "deficit_x", deficit)`) is already supported through `-> tuple[...]`.
- **Masks** cover the thresholds on green area, structural mass and "is over" states.
- **Transport as graph systems:**
  - Root-CyNAPS water (Newton, two unknowns per node) and nitrogen transport (implicit Euler, upwind advection and diffusion);
  - RhizoSoil's diffusion, which could be written the same way on the grid.
- **Tree kernels:**
  - rhizodep's distance from tip and `axis_apex_id`: `chain_scan`;
  - the elongation supply windows: `path_window`;
  - the consumption shared between supplying segments: `path_contributions` / `scatter_contributions`, bit for bit;
  - elongwheat's pseudostem: `chain_scan` max and sum;
  - its n−1 / n−2 reads and forward writes to n+1: `chain_shift` / `chain_write`.
- **Root ↔ soil:** barycentre `CrossMapping` and `Exchanges`, with the translator's unit factors. Zeroing, `np.add.at` and broadcast back as today, without queues or shared memory.
- **Scene:** the Scene's order (environment, then plants) is the effective order of today's workers. Per-plant initialisation is `initiate_plant`; input tables are `apply_input_tables`; outputs and checkpoints exist.
- **Covered, though the audit asked for them:**
  - weighted sums (cnwheat's `nb_replications`): a formula link (`x * nb_replications`) with a `sum` mapping;
  - per-vertex lists (vessel radii, voxel triplets): `dtype=object` variables, kept out of solvers;
  - vertices inserted in a chain or removed with their children re-linked: `extend_graph` handles both (relinking). This is to be confirmed with a test on adel's edits (PT3).

## 2. Gaps, by area

| # | Gap | Needed by | Importance |
|---|---|---|---|
| G1 | **Upward and downward folds with a custom function**, level by level: the nonlinear pipe-model radius `sqrt(son² + SGC·Σ lat²)` with a threshold; death when all children are dead, with the minimum time since death; filtered max over children (barrier reopening when `child.length ≥ parent.radius`) | rhizodep, Root-CyNAPS anatomy | critical |
| G2 | **`chain_gather`**: tiller rank n reads the main stem at rank `cohort + n − 1` | elongwheat, cnwgrass | critical |
| G3 | **Deterministic growth with random draws:** per-vertex random streams (seeded by plant seed and vid) instead of global re-seeding in visiting order; chained creation of segments within one step (`dist_to_ramif`, ages) | rhizodep, Root_BRIDGES, adel azimuths | critical for reproducibility |
| G4 | **Graph systems coupled to a node outside the MTG** (the shoot phloem pool of Root-CyNAPS), and **boundary kinds switched at run time** (collar pressure or flux, depending on whether the shoot gives a value; inflow or outflow by the sign of the water flux) | Root-CyNAPS, Root_BRIDGES | high |
| G5 | **Forcings interpolated in time inside a solve** (cnwheat's `solve_ivp` reads meteo between t and t + dt), regime switches and one-off resets inside the right-hand side | cnwheat, cnwgrass hydraulics | high |
| G6 | **Shoot geometry for light:** triangles per element (a ragged store, or a triangle DataStructure indexed by element), an optical class per element, the plant transform, and the turtle's state (`rollToVert`, the tillers' shared base: a sequential `chain_recurrence`) | adel, Caribu light, the visualisation | critical for light |
| G7 | **Mappings between the plant's population and environment scalars** (LAI, aboveground dry matter, total root length to a soil-temperature model), and **column ↔ grid** (the 1-D temperature profile: layer means of the grid's moisture, broadcast back) | soiltemp ↔ RhizoSoil | high |
| G8 | **Scheduling:** a component run every N steps or when a condition holds (Caribu every 4 h when there is light); spin-up hooks (MIMICS steady state, Campbell's 351 days); scenario events and stop conditions (fertilisation, drought and rehydration, plant death) | light, soil, cnwgrass scenarios | medium |
| G9 | **One forcing table shared by the scene** (meteo read once, for the environment and the populations; today it is read three times, from different files) | composite | medium |
| G10 | **State that is not variables:** external solver objects (cmf project, Campbell's state tuple), vector-valued cell variables (MIMICS' 15 pools per voxel), and their checkpointing | RhizoSoil, soiltemp | high for the soil |
| G11 | **Removing or disabling an inherited step in a subclass** (Root_BRIDGES overrides steps with an empty body); components templated per configuration (one transport system per solute) | Root_BRIDGES, Root-CyNAPS | medium |
| G12 | **Anatomy library (GRANAP):** an anatomy template per class, attached to segments, instantiated per segment in the DataStructure and scaled by segment length; repartition when Compartments merge or split (aerenchyma) or a segment changes class; junction wiring one-to-many (fused vessels) and by geometric matching between templates | GRANAP, MECHA | critical for anatomies (Q-A4) |
| G13 | **Vectorised fixed-point loops** (Farquhar's Ci/Ts, up to 30 iterations per element) | farquharwheat, cnwgrass gas exchange | low (a step can loop on masked arrays) |
| G14 | **Non-uniform layer thickness** along a grid axis | soil stratification (later) | low |
| G15 | **The Logger and the images** without the MTG (pyvista tubes from x1…z2, voxels, shoot tessellation) | fspm-utility (downstream) | low for the physics, large for the tooling |

## 3. Proposed steps

Each step gets a short design note, tests against a reference loop taken from the model it serves (written in test helpers), and a commit.

| Step | Content | Gaps |
|---|---|---|
| **PT1** ✓ | Tree kernels, round 2: `fold(values, direction, fn)`, a numba fold applied level by level for nonlinear and min/max/all reductions, with a children filter; `chain_gather`; `chain_recurrence` | G1, G2, G6 (turtle) |
| **PT2** ✓ | Reproducible random draws: `ds.rng(entities, seed)`, one stream per (plant seed, vid, step), vectorised; documented MPG-style patterns for chained creation | G3 |
| **PT3** ✓ (templating: QPg) | Structure edits: tests and fixes for adel-like edits (inserting elements in a chain, removing with relinking, rebuilding elements each step) and the repartition after them; disabling inherited steps (`steps_removed`), and templated components | G11, §1 |
| **PT4** ✓ | Graph systems: extra unknowns outside the graph, coupled to nodes (a pool with its own balance); boundary sets whose kind is chosen per call; `self.forcing(name, t)` interpolating input tables inside solves | G4, G5 |
| **PT5** ✓ | Scene services: one forcing table; `every=` / `when=` scheduling; spin-up hooks; events and stop conditions | G8, G9 |
| **PT6** ✓ | Mappings: population → environment scalars (a reduction over the plants of every population); column ↔ grid (layer mean and broadcast) | G7 |
| **PT7** ✓ | Non-variable state: component state saved by checkpoints (a `__checkpoint__` hook); vector-valued variables `(n, k)` | G10 |
| **PT8** ✓ (as reduced by QPd) | Shoot geometry: per-element triangles (store and transforms), optical classes, a light-component skeleton on a `UnionDataStructure`, tested with a toy radiosity, and a Caribu adapter kept downstream | G6 |
| **PT9** | Anatomy library, after your answers on GRANAP | G12 |

Not planned in metafspm: G13 (a masked loop in a step), G14 (until stratification is needed), G15 (downstream, with the guide's §6).

**Suggested order:**
1. PT1 → PT2 → PT3 → PT4: these unblock the root models (rhizodep, Root-CyNAPS, Root_BRIDGES).
2. PT6 → PT5 → PT7: the soil.
3. PT8: the shoot and its light.
4. PT9: when GRANAP is ready.

## 4. Questions

- **QPa, reproducibility.** rhizodep re-seeds the global generator with `random_choice · vid` and draws in visiting order, and nodule emergence is unseeded. Must the port be identical to the current runs, or are per-vertex streams with the same distributions acceptable (reproducible, but not the same draws)? **Recommendation:** per-vertex streams. Bitwise identity with the current runs would require keeping the global generator and the visiting order in an MPG-style step.
  → answer: agree with Recommendation
- **QPb, the shoot's data model.** cnwheat, elongwheat and growthwheat keep dicts keyed by `(plant, axis, metamer, organ, element)` tuples, hidden zones as dicts on metamers, and adel rebuilds its elements each step. Should the port map them to MTG scales, with hidden zones as Phytomer-scale variables (or vertices) and elements as SubOrgan vertices kept from one step to the next? Or keep adel's rebuilt elements? **Recommendation:** MTG scales, with stable elements, so that variables are carried by the DataStructure and not copied back and forth by facades.
  → answer: MTG scales yes so this is not really a gad as is.
- **QPc, the shoot phloem pool for the roots (G4).** Should it be an extra unknown of the root transport system (a node outside the MTG, as Root-CyNAPS does)? Or should it be a Plant-scale variable exchanged with the shoot component at fixed points? **Recommendation:** an extra unknown attached to the collar inside the solve (PT4), since the collar flux and the pool must be solved together for stability.
  → answer: yes to Recommendation, so it does not really change the API right?
- **QPd, light (G6).** Keep Caribu as the engine, with metafspm providing per-element triangles, optical classes and a light component on the union of populations (the adapter downstream)? Or should metafspm also own the triangulation from organ dimensions (adel's leaf-shape database)? **Recommendation:** metafspm stores and transforms triangles; adel's geometry stays a structural component downstream that writes them.
  → answer: No, for now the light model must work from MPG DataStructure and handle triangulation by itself.
- **QPe, soil water (G10).** Keep cmf as an opaque solver inside the soil component, with its state outside the DataStructure and saved by a checkpoint hook? Or reimplement Richards and solute transport as graph systems on the grid? **Recommendation:** keep cmf first (PT7); a graph-system version can follow and be compared against it.
  → answer: yes, reimplementation is for later.
- **QPf, which model first?** **Recommendation:** rhizodep (PT1–PT3 cover it), then Root-CyNAPS (PT4), then RhizoSoil, then the shoot.
  → answer: yes to Recommendation, but in practice we will begin with GRANAP before all the others.
- **GRANAP (G12, after Q-A4 in `devplan_datastructures.md`):**
  - **Q-A4:** is a template shared per class, or are Compartments copied per segment? Do state variables differ per segment within a class? (The audit finds that the solved graph needs per-segment Compartments in any case, since state variables differ; only the geometry would be shared.)
  - What defines a class: the diameter only, or diameter with age or distance from the tip?
  - When a segment changes class (aerenchyma, barrier maturation), is its anatomy replaced, or edited in place? How are cell contents carried over?
  - Which cells get axial junctions: only xylem and sieve tubes (as MECHA's axial K does), or also the apoplast and the symplast?
  - Are walls and wall junctions Compartments, or are only cells nodes, with walls as edges?
  - How many classes and segments do you expect? An anatomy is about 5–30 k graph nodes (1.4–5 k cells), so the full per-cell graph of a root system may be out of reach without a reduction per class.
  → answer: first I agree, we share only to avoid geometry re-generation but it should indeed be copied, the most effitient way possible, yet copied so that each segment gets its state variables. Second, for now, diameter, distance from tip and distance from shoot-root junction. Third, replaced and content like cell solute concentrations is only an input provided by coarser metabolic components as root_carbon. Fourth, For now, only xylem and phloem. Fifth, the code says it explicitly, all nodes are Compartments and edges between nodes Connections. Sixth: Ideally not more than a hundred, since now for wheat first order to lateral diameter is a fixed parameter. When we will proceed to the port of GRANAP, I have precise guidelines I will tell you after resolving the general gaps.

## 5. Decisions from your answers (2026-10-06)

- **QPa:** per-vertex random streams, the same distributions, reproducible; draws are not identical to today's runs.
- **QPb:** the shoot is ported onto MTG scales with stable elements, so adel's per-step rebuild is not a metafspm gap. PT3 keeps only insertions at emergence and removals with relinking, plus the inherited-step changes.
- **QPc:** the shoot phloem pool is an extra unknown attached to the collar inside the root transport solve. To your question: it adds an option to `@graph_system` (extra unknowns outside the graph, with their own balance), and changes nothing in the existing API.
- **QPd:** the light model works from an MPG DataStructure (or a union of populations) and triangulates by itself. metafspm stores no triangles, so G6 reduces to what the light component reads: organ dimensions, positions and frames (`path_compose`, `fold` downward for the turtle), and its per-element outputs. PT8 becomes a test light component that triangulates the MPG's elements itself, to check that the API is enough.
- **QPe:** cmf stays an opaque solver inside the soil component (PT7); a graph-system Richards version comes later.
- **QPf:** the order is rhizodep, Root-CyNAPS, RhizoSoil, then the shoot, but **GRANAP is ported first in practice**. Its precise guidelines will come after the general gaps are resolved.
- **GRANAP:**
  - templates are shared only to avoid regenerating geometry; each segment gets **its own copy** (its Compartments and state variables), made as efficiently as possible;
  - a class is defined by diameter, distance from the tip and distance from the root–shoot junction;
  - a segment changing class gets its anatomy **replaced**; cell contents (e.g. solute concentrations) are inputs from coarser metabolic components (root_carbon), so the repartition after a replacement only needs those inputs, broadcast to the new Compartments;
  - axial junctions: xylem and phloem only;
  - all nodes are Compartments, and all edges between them are Connections;
  - at most about a hundred classes (for wheat, the diameter from the first order to the laterals is a fixed parameter).

**Order:** the general gaps first (PT1–PT7, and PT8 as a check), then PT9 for GRANAP with your guidelines.

## 6. PT1 design: tree kernels, round 2

- **`ds.fold(update, values, direction="up" | "down")`:** a fold over the tree, level by level: from the deepest level for "up" (children before parents), from the roots for "down".
  - At each level, `update(level, out)` returns the new values of the level's nodes; it is any vectorised function.
  - `level` gives `nodes` (local indices) and `edge` (each node's edge type: `"<"` or `"+"`).
  - For "up", `level.children(values, op, edge=None, where=None, fill=…)` reduces over each node's children: op is sum, max, min, all, any or count, optionally only children with that edge type or where a mask holds.
  - For "down", `level.parent(values)` gives the parents' values.
  - It covers the pipe model (successor section plus SGC times the sum of the emerged lateral sections, then the threshold), death (all children dead, the minimum time since death), the barrier reopening (filtered max over `+` children), and the turtle's frames (down, with the frame state of the parent).
- **`ds.chain_gather(values, rank, chain=…, source=…)`:** each node reads the value of the node at rank `rank` on the chain given by `source`.
  - For chains defined by a group and a rank variable, `source` is a group value and `rank` a rank value (e.g. a tiller's metamer n reads the main stem's metamer `cohort + n − 1`).
  - For edge-type chains, `source` is any node vid of that chain and `rank` a 0-based position.
  - Missing ranks give `fill`.
- **`ds.chain_recurrence(update, values, chain=…)`:** a non-associative recurrence along chains, position by position, vectorised across chains: `update(position, nodes, previous, out)` returns the new values, `previous` being each node's predecessor on its chain (−1 at the start). For recurrences along rank chains that are not parent links (e.g. metamer ranks).
- **Tests:** reference loops written after rhizodep's `potential_growth` (pipe model and death, with `edge_type`, emerged laterals and nodules excluded), Root-CyNAPS' barrier reopening, elongwheat's tiller cohort reads, and a turtle frame recurrence, all compared exactly.

## 7. After PT1–PT3 (2026-10-06): PT3's open part and the PT4 design

**Done:**
- **PT1:** `fold`, `chain_gather` and `chain_recurrence`, exact against the models' loops.
- **PT2:** per-entity random streams, kept by checkpoints.
- **PT3:**
  - insertion mid-chain;
  - removal with relinking, after a fix: openalea's `replace_parent` made shared complexes their own parents;
  - a step returning `None` writes nothing;
  - `steps_removed`.

**PT3's open part: templated components (G11).** Root-CyNAPS applies one generic transport code to a dict of solutes, each with its own variable names and conversions (`solute_configs`); Root_BRIDGES edits that dict per instance. Steps and graph-system equations take their variables by argument name, so generic equations cannot serve several solutes as they are.

- **QPg, how to template.** Options:
  - **(a) `component_template(Base, name, rename={...})`:** a subclass whose declared variables, step arguments, outputs and graph-system fields are renamed (e.g. `solute → C_sucrose_root`, `flux → hexose_diffusion_from_phloem`). One class per solute, generated, with the equations written once.
  - **(b) Vector-valued unknowns** `(n, k)`: one system for k solutes as columns. It needs PT7's vector variables in graph systems too, and the solutes must share their equations' form.
  - **(c) One class per solute** written by hand (duplication).

  **Recommendation:** (a). It keeps one DataStructure name per solute, so the translator, logging and checkpoints are unchanged. Conversions per solute become parameters of the generated class.
  → answer: I would rather like duplication for now (c) to keep things explicitly declared.

**PT4 design: graph-system extensions (G4, G5).**
1. **Pool unknowns at a coarse scale (QPc agreed):**
   - `@graph_system(pool_unknowns={"sucrose_shoot": "Plant"})`: one unknown per entity of that scale (one per plant), solved together with the node and edge unknowns;
   - `@pool_balance(field=...)` writes its residual per pool entity;
   - equations exchange with the pool through `self.pool_exchange(name, boundary_set)`, a sparse map between the nodes of a boundary set (the collar) and the pool of their own plant, with sums one way and broadcasts the other;
   - pools are stored at their scale on the DataStructure (Plant variables), so the shoot reads them as usual.
2. **Boundary kinds chosen at run time:** a `boundary_set(kind=...)` may name a node variable holding each node's kind, as label codes ("dirichlet", "neumann", "robin"), read at each solve. The collar switches between a pressure and a flux when the shoot gives no value, or by the sign of the flux.
3. **Forcings inside solves:** `self.forcing(name)` in an equation returns the forcing at the end of the current (sub-)step (implicit Euler's time), linearly interpolated in the time series given to the component or, after PT5, the scene's shared table. With adaptive steps each trial sees its own time.

- **QPh, pools (1).** Pool unknowns at a coarse scale, as above? The alternative is virtual nodes appended to the graph, with an edge to the collar, as Root-CyNAPS does today. **Recommendation:** pools at a scale. One pool per plant comes from the population without bookkeeping, and the pool is an ordinary Plant variable outside the solve.
  → answer: follow Recommendation
- **QPi, boundary kinds (2).** A per-node kind variable, read at each solve? **Recommendation:** yes. It also lets one set mix kinds (some tips Dirichlet, others Neumann) without one boundary set per kind.
  → answer: yes to Recommendation
- **QPj, forcings (3).** Is the end of the (sub-)step the right time? cnwheat's `solve_ivp` reads forcings continuously, which implicit Euler cannot do; IVP solvers would read them at each evaluation time. **Recommendation:** the end of the (sub-)step for implicit solvers, and the evaluation time for the IVP solver.
  → answer: yes to Recommendation

## 8. After PT4 (2026-10-06): designs for PT6, PT5 and PT7 (the soil)

**Done:**
- **PT4:**
  - pool unknowns per plant, solved with the graph;
  - per-node boundary kinds read at each solve (`kinds=`);
  - forcings at the end of each (sub-)step, or at the IVP evaluation time.
- **PT3's templating:** dropped (QPg: duplication by hand).

**PT6, mappings for the environment (G7):**
1. **Population ↔ environment scalars.** A translator link between a population variable and an environment variable stored at `"scalar"`. Examples: total LAI or aboveground dry matter to Campbell; the air temperature to every node. It is exchanged by `Exchanges` over every plant of every population: extensive values summed, intensive values averaged (weighted when `weight=` is given), a scalar broadcast to the nodes.
2. **Column ↔ grid.** `LayerMapping(column, grid, axis="z")` between a 1-D grid (Campbell's layers) and a 3-D grid (the soil):
   - weights are the overlaps of the layer intervals along z, so the layer thicknesses may differ;
   - grid → column: layer means of intensive values (sums of extensive ones);
   - column → grid: broadcast over x and y.

   Given to the Scene explicitly (`Scene(mappings=[...])`), since grid-to-grid pairs cannot be inferred from links alone.

**PT5, scene services (G8, G9):**
1. **Shared forcings:** `Scene(forcings=table)`, a DataFrame indexed by time (s). A component's `forcing(name)` falls back on it, so meteo is read once by every model.
2. **Scheduling:**
   - a model may set `run_every = n` (steps) or `run_when(scene) -> bool` (e.g. Caribu every 4 h, when there is light);
   - on the skipped steps its exchanges are not run, and its outputs keep their last values (a model may rescale them itself, as the Caribu adapter does with Erel).
3. **Spin-up:** an environment model's optional `spin_up(scene)` runs once after the scene is built, before the first step (MIMICS' steady state, Campbell's 351 days).
4. **Events and stops:**
   - `Scene(events=[(time, action)])` runs `action(scene)` at the start of the first step at or after `time` (fertilisation, rehydration);
   - `Scene(stop_when=condition)` ends `simulate` when `condition(scene)` holds after a step (plant death).

**PT7, state outside variables (G10):**
1. **`Scene.checkpoint(path)` / `Scene.restore(path, ...)`:**
   - every DataStructure as today;
   - plus each model's and component's optional `checkpoint_state()` (pickled) and `restore_state(state)`, e.g. the cmf project and Campbell's state tuple;
   - plus the scene time, the iteration and the recorder's position.
2. **Vector-valued variables:** `register(name, shape=(k,))` / `state_variable(..., shape=(k,))` stores an `(n, k)` array (MIMICS' 15 pools per voxel):
   - steps receive `(n, k)` arrays;
   - mappings and exchanges apply per column;
   - checkpoints and the recorder keep them (one column per component);
   - they are not graph-system unknowns for now.

### Questions

- **QPk, scalars (PT6.1).** Should exchanges between a population and an environment scalar reduce over every plant of every population (sums for extensive values, means for intensive ones), with no mapping to declare? **Recommendation:** yes, by kind, as the other exchanges.
  → answer: yes
- **QPl, column ↔ grid (PT6.2).** Should the mapping be given explicitly to the Scene, weighted by layer overlaps? **Recommendation:** yes.
  → answer: yes
- **QPm, skipped steps (PT5.2).** On steps a model does not run, should its outputs keep their last values, and the exchanges into it be skipped? **Recommendation:** yes. Rescaling, as Caribu's Erel × PARi does, stays in the model.
  → answer: yes
- **QPn, checkpoint hooks (PT7.1).** Should models and components save their non-variable state through `checkpoint_state()` / `restore_state()` hooks? **Recommendation:** yes. The DataStructures already restore bit for bit, and only opaque external solvers need the hooks.
  → answer: yes
- **QPo, vector variables (PT7.2).** Are `(n, k)` variables enough for MIMICS' pools (steps, mappings, checkpoints, recorder; not graph-system unknowns)? Or should they also be graph-system unknowns (k coupled fields per node)? **Recommendation:** the former now; graph-system unknowns of shape `(n, k)` when a model needs them (e.g. a vectorised multi-solute transport).
  → answer: keep it the former way yes because these wrapping CMF and MIMICS-CN will all tend to be replaced by Component compliant models in the future to be considered a proper metafspm component.

## 9. Status after PT1–PT8 (2026-10-06): what is left before porting

**Done, each with tests against reference loops or hand computations:**
- **PT1 tree kernels:** `fold`, `chain_gather`, `chain_recurrence`.
- **PT2:** reproducible random streams.
- **PT3:** structure edits mid-chain and removals with relinking (an openalea `replace_parent` bug worked around); disabling inherited steps.
- **PT4:** pool unknowns per plant, per-node boundary kinds, forcings in solves.
- **PT5 scene services:** shared forcings, `run_every` / `run_when`, spin-up, events, stops.
- **PT6:** scalars over populations, column ↔ grid.
- **PT7:** scene checkpoints with state hooks, vector-valued variables.
- **PT8:** a light model on the union of populations that triangulates the MPG's elements itself and runs every 4 steps. It checks that a CARIBU-like component needs nothing more from the API.

**Left in metafspm:**
- **PT9, the anatomy library (GRANAP):** waits for your guidelines, as agreed (QPf). The answers to Q-A4 and the GRANAP questions are in §5.
- **Not planned, by decision:**
  - G13: fixed-point loops stay in steps;
  - G14: non-uniform layers, when stratification comes;
  - G15: the Logger and images (downstream);
  - templated components (QPg: duplication).

**Limits to know when porting:**
- pool unknowns need a Newton solver and finite-difference Jacobians;
- vector variables are not graph-system unknowns;
- `split="components"` is not parallel;
- an MPG-style step sees only the MTG properties its component declares.

**Port order (QPf):** GRANAP first, after its guidelines; then rhizodep, Root-CyNAPS, RhizoSoil (with cmf and MIMICS wrapped as opaque solvers), and the shoot (cnwgrass and WheatFspm on MTG scales). The light model triangulates by itself.

## 10. Before GRANAP: remaining items (2026-10-06, your question)

**Done now:**
- **The test suite's warnings:** all 10 came from SciPy's Anderson root method, whose history matrix becomes singular as the residual vanishes (`LinAlgWarning`), while the iterate has converged. `ScipyRootSolver` checks the residual after the call anyway, so that warning is silenced for Anderson only. The suite now runs without any warning, hidden categories included (`-W default`).

**Open, with questions:**
1. **`MultiGridDataStructure` is barely integrated.** It has no variable store, so it cannot host a component. It has no graph topology, so graph systems cannot run on it, and no checkpoint. Its restriction and prolongation are built on flat indices, which is only right in 1-D. It is tested only for construction, delegation to the fine level and the topology hook. What it should be depends on its use:
   - **(a)** several resolutions holding their own variables (e.g. microbes on a coarse grid, transport on the fine one), exchanged by restriction (volume average) and prolongation at fixed points. Each level would be an `ArrayDataStructure`, and `MultiGrid` the set of levels with N-D operators, i.e. grid ↔ grid mappings at different resolutions (also soil ↔ atmosphere);
   - **(b)** a geometric multigrid accelerating the solves of graph systems on the fine grid (a preconditioner), invisible to models;
   - **(c)** both.

   **QPp:** which use do you intend? **Recommendation:** (a) first. It is the modelling feature, it generalises the column ↔ grid mapping (a `GridMapping` by cell-volume overlaps in N-D), and it fits the existing exchanges. (b) is a solver optimisation, worth doing only if grid solves become the bottleneck.
   → answer: What is the best for adaptative discretization uppon solved processes / flows intensity?
2. **Compartments are topological children of their segments** (B-i, deferred in WD.P). `populate_graph` links each Compartment to its segment with a topological parent, so openalea's `g.children(segment)`, `Sons()` and `post_order2` return Compartments (scale 9) mixed with the child segments (scale 6). The DataStructure and the kernels are not affected (they use the Connections). But MPG-style code ported from rhizodep, which loops on `children()` and `post_order2`, would count Compartments as children: death counts, the pipe model, `len(apex.children())` for primordia. Options:
   - **(a)** the MPG's `children` / `children_iter` / `Sons` / `nb_children` return the same-scale children only (overrides; raw `_children` unchanged for the framework);
   - **(b)** Compartments become components of their segment instead of children (a change in `populate_graph`, `extend_graph` and the anatomy wiring);
   - **(c)** leave it, and document that ported code must filter by scale.

   **QPq:** which? **Recommendation:** (a). It makes openalea's traversals (which use `children_iter`) behave as on a plain MTG, with a small, testable change.
   → answer: yes to (a)
3. **Coverage cannot be measured:** `coverage` / `pytest-cov` are not installed in the metafspm environment, although `[tool.coverage]` is configured (B1).

   **QPr:** may I install `pytest-cov` in that conda environment, to report untested code before GRANAP? **Recommendation:** yes.
   → answer: yes you can
4. **Still deferred, by earlier decisions, listed so that nothing is forgotten:**
   - DS7's MPG ↔ MPG coupler (until a plant uses two DataStructures);
   - W6.1 (an integration marker for downstream CI);
   - parallel pieces (S2);
   - the anatomy repartition and Q-A4's implementation (PT9, with GRANAP's guidelines).

## 11. After your answers (2026-10-06)

- **QPq (a), done:** the MPG's `children`, `children_iter` and `nb_children` return the children at the vertex's own scale, so openalea's `Sons`, `post_order2` and `pre_order2` see segments only, as on a plain MTG. The framework reads the raw links in `_children`, and the graph is unchanged. A test covers it.
- **QPr, done:**
  - `pytest-cov` is installed in the metafspm environment;
  - coverage is 86 % overall, with `pytest --cov=src/openalea/metafspm`;
  - `tree_kernels.py` shows 62 % only because numba-compiled bodies are not traced;
  - `solver.py` and `system_specs.py` (76–78 %) are mostly older solver paths and introspection helpers no current component uses.
- **QPp, your question:** "what is the best for adaptive discretization upon solved processes / flows intensity?"

  A fixed multigrid hierarchy (QPp's options) is not adaptive: its levels cover the whole domain at fixed resolutions. For a resolution that follows where the processes are intense (around roots, along strong fluxes or gradients), the usual answer is **cell-based adaptive refinement (an octree in 3-D)**:
  - **Refinement:** a cell whose indicator is high (flux magnitude, gradient, root length density) is split into 8 children; children whose indicator is low are merged back.
  - **Graph:** neighbouring cells may differ in size, and the faces between them get their real area and centre distance. Cells are nodes and faces are edges, with `face_area` and `face_distance`, exactly the grid's graph contract (DS1). So graph systems, boundary sets, masks and steps work on it unchanged.
  - **Carried values:** refining or coarsening is a topology change between steps, like growth. Extensive values are split by volume or summed, and intensive values copied or volume-averaged: the repartition rules of the plants, with cell volume as the weight.
  - **Coupling:** `locate` searches the tree, so CrossMapping (barycentre, overlap) and the exchanges are unchanged.
  - **Why not the others:** block-structured refinement (fine patches over a coarse grid) needs flux corrections at patch borders. A multigrid hierarchy only accelerates solves. Neither adapts as directly.

  **Proposed step, PT10, `AdaptiveGridDataStructure`:**
  - an octree over a coarse base grid, with periodic axes and a maximum level;
  - the 2:1 balance between neighbours, so a face has at most four neighbours on its other side;
  - a refinement criterion given as a function of the DataStructure, applied at fixed points by `refine(criterion)` / `coarsen(criterion)`, then `update_topology()` with conservative carry-over;
  - tests: conservation through refine and coarsen, a diffusion solve against the uniform fine grid, CrossMapping on refined cells.

  The current `MultiGridDataStructure` (no variable store, 1-D-only operators, no users) would then be removed. Multi-resolution variables (microbes on a coarse grid) can be two grids linked by an overlap mapping if a model asks for it.

### Questions

- **QPs:** cell-based octree refinement as above (PT10)? **Recommendation:** yes. It keeps one graph contract for plants and soils, so the models do not change when the soil becomes adaptive.
  → answer: yes
- **QPt:** refinement at fixed points (between steps, like growth) from a user criterion, with a maximum level and the 2:1 balance? Or within a step (re-solving after refinement)? **Recommendation:** between steps. The flux of the step that just ended decides the next step's mesh, which keeps solves on a fixed graph.
  → answer: yes to Recommendation
- **QPu:** remove `MultiGridDataStructure` once PT10 exists? **Recommendation:** yes. It is unusable as is, and adaptive cells cover the need it was meant for.
  → answer: yes to Recommendation
- **Before or after GRANAP?** PT10 concerns the soil, which GRANAP does not need. **Recommendation:** after GRANAP, unless you want the soil first.
  → answer: now so overall DataStructure API st

## 12. PT10 done (2026-10-06): adaptive grids; MultiGrid removed

- **Done:**
  - `AdaptiveGridDataStructure` (octree, 2:1 balance, faces from the finest lattice, conservative carry-over, checkpoints, `CrossMapping`), tested against `ArrayDataStructure`, with uniform-refinement equivalence and conservation;
  - `MultiGridDataStructure` removed (QPu).
- **Limits:**
  - faces are found on the finest lattice (base cells × 2^(d·max_level)), which suits moderate depths (2–3 levels);
  - `LayerMapping` works on regular grids only;
  - `layer_mask` takes the boundary layers 0 and −1.
- **Found, outside the suite:** `test/data_api_tests/examples/example_mpg_data_structure.py` (a legacy-to-MPG migration demo) fails, already before this step. It builds `MPGDataStructure` from an unpopulated MTG, and once populated, mixes legacy and MPG node counts.

  **QPv:** rewrite it on the current API, or delete it (the migration guide covers the path)? **Recommendation:** delete it, with its image, since `LegacyMPGDataStructure` / `from_legacy` are only kept for that migration.
  → answer:


## 13. Documentation and test audit (2026-10-05): questions

**Done:**
- the docs (`docs/index.md`, `user.md`, `ref.md`, `conventions.md`, the migration guide) and the README describe the current API, with no step numbers;
- the docstrings build as RST with no warnings;
- the tests:
  - one Choregrapher reset fixture for the whole suite;
  - obsolete tests and helpers removed (`test_component_base`, `test_field_consensus`, `test_partial_traversal`, `generate_anatomy_in_mtg.py`, `component_api_changelog.md`, the duplicate `example_translator.yaml`);
  - stale docstrings rewritten;
  - scene doubles moved to `structure_tests/scene_doubles.py`;
  - `solver=` and `self.dt` used throughout;
  - new tests for the API that had none (72, in four files).
- **Fixed** (bugs found by the new tests, see the CHANGELOG):
  - tuple forcings;
  - long-form links in `to_nested`;
  - same-name links stating their scales;
  - anatomy-mode populations and their `wiring`;
  - pending events across scene checkpoints;
  - Python translators in `CompositeModel`.

**Found, to decide:**
- **Neumann signs.** `boundary_set(kind="neumann")` subtracts its value (an inflow), but `@boundary_condition(kind="neumann")` adds the method's values (`solve/decorator.py`, `make_combined_node_ev`).
- **Time terms.** Equations write their own time terms (`(u - self.previous(u)) / self.dt`), solved by the Newton family. But `implicit_euler` adds `(u − u_prev)/h` to every unknown itself, and `explicit_euler` / `scipy_ivp_*` take `−R` as du/dt. With these solvers, a balance written to the convention has its time term twice. UC1 uses them with edge time terms (`q(1 + 1/dt)`).

### Questions

- **QPw — legacy APIs.** Remove the following, or keep them one more release?
  - `LegacyMPGDataStructure` / `from_legacy`, with `test_legacy_mpg`, `example_legacy_mpg` and `example_mpg_data_structure` (QPv);
  - `MPG.graph()`, `integrate_at_scale`, `average_at_scale`;
  - `GraphView.from_mtg_subset` (a test asserts its current bug);
  - the setters `set_node_property`, `add_field`, `_get_field`;
  - in the Choregrapher: `__call__(module_family)`, `build_schedule`, `add_schedule`;
  - `CompositeModel.declare_data`, `Translator.inputs_outputs`, the `props` view, the `_last_graph_system` shim.

  **Recommendation:** remove them now, before the models are ported (nothing ported depends on them yet). Keep only the `props` view if the logger needs it.
  → answer: agree with Recommendation
- **QPx — the solver layer below graph systems.** Make the following internal (no longer public API, tests through components only), or keep them public?
  - the `GraphSystem` shim, `ODESystemSpec`, the `solve(t_span)` time loop and `SolverResult`;
  - `BoundaryConditions`, the linear-assembly mode with `LinearDirectSolver`.

  **Recommendation:** make them internal. Keep the solver unit tests, but drop them from the API reference.
  → answer: yes to Recommendation
- **QPy — time-term convention.** Make one convention: equations write their own time terms. Then `implicit_euler` stops adding its own, and becomes Newton with a transient check. `explicit_euler` and `scipy_ivp_*` would take an explicit `rate` form (du/dt given by the equations), declared separately from the residual form. **Recommendation:** yes. UC1's solver comparisons are rewritten to the convention.
  → answer: explain in more details.
- **QPz — `@boundary_condition`.** Deprecate it in favour of `boundary_set`? It has the opposite Neumann sign and less selection. **Recommendation:** yes. Port UC4 to `boundary_set`, and keep the decorator one release with a warning.
  → answer: yes to Recommendation
- **QPα — old-protocol files in `test/provide_usage_examples`:**
  - rewrite `composite_wrapper_example` and `rhizosoil_component_example` as current-API sketches;
  - move `rhizosoil_core_model` and `logger_api_reference` (which no longer runs) to `test/provide_usage_examples/legacy/`;
  - delete `light_component_example` (`test_light_component` covers it).

  **Recommendation:** as listed.
  → answer: as listed
- **QPβ — `dev/design/*`.** The development notes there are full of step numbers. Move them out of `docs/`, e.g. to `dev/design/`, keeping the migration guide in `docs/`? **Recommendation:** yes.
  → answer: yes
- **QPγ — the `data_api_tests/examples` scripts.** Turn them into pytest smoke tests writing their images to `tmp_path`, so they cannot rot again? **Recommendation:** yes, for the ones on the current API (array, MPG). The legacy ones follow QPw.
  → answer: yes
- **QPδ — test layout.** Reorganise the test files by feature (data structures, components, graph systems, coupling, scenes), e.g. split `test_datastructure_prerequisites` and merge the two `test_composite_*` files? **Recommendation:** yes, as one commit of file moves only, after QPw.
  → answer: yes
- **QPε — UC1, UC1-organ and UC3.** They still use the `props` view and solver internals. UC3's hand-built ports also duplicate `test_anatomy_mode`. Rewrite them on the public API (components, `boundary_set`, `previous`) after QPy? **Recommendation:** yes.
  → answer: yes
- **QPζ — `legacy_functor.py`.** It is the current step wrapper, not a legacy one: rename it `functor.py`? **Recommendation:** yes.
  → answer: yes

## 14. QPy in more detail: one time-term convention (2026-10-05)

**What a graph system's equations mean today, solver by solver.** Take a diffusion of `c` on nodes, with fluxes `q` on edges:

```python
@node_balance(field="c")
def _balance(self, c, q):
    return (c - self.previous("c")) / self.dt + B @ q / volume - source      # residual form, time term written

@edge_law(field="q")
def _fick(self, c, q, K):
    return q - K * B.T @ c
```

| solver | what it solves with these equations | correct? |
|---|---|---|
| `newton`, `newton_fd`, `scipy_krylov/anderson/hybr` | R(c, q) = 0 as written: backward Euler, since the time term is in R | yes |
| the same with `integrate="substeps"` / `"adaptive"` | the framework splits the step and moves `previous` / `dt` with each sub-step | yes |
| `implicit_euler` | R + (u − u_prev)/h = 0 for **every** unknown: the time term twice on c, and a spurious `(q − q_prev)/h` on the fluxes | no |
| `explicit_euler` | c_new = c + dt · (−R): it reads −R as dc/dt, and R already holds (c − c_prev)/dt | no |
| `scipy_ivp_bdf/radau` | dc/dt = −R(c, q) integrated by solve_ivp: same misreading | no |

Today `implicit_euler`, `explicit_euler` and the IVP solvers are only correct if the node balance leaves out its time term and is written as −dc/dt, i.e. `B @ q / volume - source`. UC1's ImplicitEuler tests rely on this, and they assert the spurious edge term `q(1 + 1/dt) = K Bᵀ c`. So the same equations mean different physics depending on the solver chosen. Switching solvers silently changes the model, and nothing warns about it.

**Proposal.** Two ways of writing a node balance, each with one meaning whatever the solver:

1. **Residual form**, `@node_balance(field)`, as today with the Newton family. The method returns the residual at the end of the (sub-)step, time term included. It suits implicit solves, algebraic constraints and steady systems.
   - Solved by the Newton family, with `integrate="step" | "substeps" | "adaptive"` for time-step control. `explicit=True` keeps its meaning: the method returns the new value, and R = u − value.
   - `implicit_euler` stops adding anything. It becomes an alias of `newton` with `transient=True`, deprecated.
   - The explicit and IVP solvers refuse residual-form systems, with an error naming the rate form.
2. **Rate form**, a new `@node_rate(field)`. The method returns dc/dt (for the example, `-(B @ q) / volume + source`), with no time term.
   - The framework adds the time term for an implicit solver: R = (c − previous(c))/dt − rate.
   - `explicit_euler` steps c += dt · rate, and the IVP solvers integrate dc/dt = rate.
   - One rate-form system can thus be solved by any solver. Comparing implicit, explicit and BDF solutions of the same model becomes a solver switch.
   - Capacities (volume, mass) are the method's business: it returns dc/dt, so it divides by them itself (no mass matrix).
3. **Edge unknowns** are always algebraic (`@edge_law`, residual form). No solver adds a time term to them, so the `(q − q_prev)/h` artefact disappears.
4. **In one graph system**, all node unknowns use the same form, and mixing them raises. Forcings are read at the evaluation time in IVP solves, as today, through `forcing_time()`.

**Effects:**
- UC1's `implicit_euler` and edge time-term tests are rewritten. Each becomes one rate-form model solved by `newton` (backward Euler), `explicit_euler` (small dt) and `scipy_ivp_bdf`, compared to each other and to the analytic solution (QPε).
- `test_active_subgraph` and `test_pools_and_kinds` use `implicit_euler` only as "a transient solver": they switch to `newton, transient=True`.
- The solver unit tests on specs stay. They test the internal layer (QPx), where `ImplicitEulerSolver` keeps its mathematical definition on an `ODESystemSpec`.
- Downstream models today write residual forms with Newton (rhizodep-style), so nothing changes for them. The rate form is new, for models wanting explicit or adaptive ODE integration (e.g. a soil or microbial pool model with scipy BDF).

**Alternative (not recommended):** keep only the residual form and make `explicit_euler` / IVP derive the rate as −(R − time term). The framework cannot separate the user's time term from the rest of R, so this relies on the user writing it in one exact way.

- **QPy (restated):** adopt the two forms above (residual form for the Newton family, `@node_rate` for every solver), with `implicit_euler` deprecated as an alias of `newton, transient=True`? **Recommendation:** yes.
  → answer:

**Meanwhile**, the decided items go in this order (QPε waits for QPy):
1. QPζ: rename `legacy_functor.py`.
2. QPβ: move `dev/design/*` to `dev/design/`.
3. QPα: the old example files.
4. QPz: deprecate `@boundary_condition`; port UC4.
5. QPw: remove the legacy APIs.
6. QPx: internal solver layer.
7. QPγ: smoke tests for the example scripts.
8. QPδ: test layout, as file moves only.
