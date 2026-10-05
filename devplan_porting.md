# Plan: what metafspm still needs before porting the existing models

Status: **draft for your answers** (2026-10-06). It comes from a read-only audit of the downstream code, nothing in those repositories was edited:
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
| **PT1** | Tree kernels, round 2: `fold(values, direction, fn)`, a numba fold applied level by level for nonlinear and min/max/all reductions, with a children filter; `chain_gather`; `chain_recurrence` | G1, G2, G6 (turtle) |
| **PT2** | Reproducible random draws: `ds.rng(entities, seed)`, one stream per (plant seed, vid, step), vectorised; documented MPG-style patterns for chained creation | G3 |
| **PT3** | Structure edits: tests and fixes for adel-like edits (inserting elements in a chain, removing with relinking, rebuilding elements each step) and the repartition after them; disabling inherited steps (`steps_removed`), and templated components | G11, §1 |
| **PT4** | Graph systems: extra unknowns outside the graph, coupled to nodes (a pool with its own balance); boundary sets whose kind is chosen per call; `self.forcing(name, t)` interpolating input tables inside solves | G4, G5 |
| **PT5** | Scene services: one forcing table; `every=` / `when=` scheduling; spin-up hooks; events and stop conditions | G8, G9 |
| **PT6** | Mappings: population → environment scalars (a reduction over the plants of every population); column ↔ grid (layer mean and broadcast) | G7 |
| **PT7** | Non-variable state: component state saved by checkpoints (a `__checkpoint__` hook); vector-valued variables `(n, k)` | G10 |
| **PT8** | Shoot geometry: per-element triangles (store and transforms), optical classes, a light-component skeleton on a `UnionDataStructure`, tested with a toy radiosity, and a Caribu adapter kept downstream | G6 |
| **PT9** | Anatomy library, after your answers on GRANAP | G12 |

Not planned in metafspm: G13 (a masked loop in a step), G14 (until stratification is needed), G15 (downstream, with the guide's §6).

**Suggested order:**
1. PT1 → PT2 → PT3 → PT4: these unblock the root models (rhizodep, Root-CyNAPS, Root_BRIDGES).
2. PT6 → PT5 → PT7: the soil.
3. PT8: the shoot and its light.
4. PT9: when GRANAP is ready.

## 4. Questions

- **QPa, reproducibility.** rhizodep re-seeds the global generator with `random_choice · vid` and draws in visiting order, and nodule emergence is unseeded. Must the port be identical to the current runs, or are per-vertex streams with the same distributions acceptable (reproducible, but not the same draws)? **Recommendation:** per-vertex streams. Bitwise identity with the current runs would require keeping the global generator and the visiting order in an MPG-style step.
  → answer:
- **QPb, the shoot's data model.** cnwheat, elongwheat and growthwheat keep dicts keyed by `(plant, axis, metamer, organ, element)` tuples, hidden zones as dicts on metamers, and adel rebuilds its elements each step. Should the port map them to MTG scales, with hidden zones as Phytomer-scale variables (or vertices) and elements as SubOrgan vertices kept from one step to the next? Or keep adel's rebuilt elements? **Recommendation:** MTG scales, with stable elements, so that variables are carried by the DataStructure and not copied back and forth by facades.
  → answer:
- **QPc, the shoot phloem pool for the roots (G4).** Should it be an extra unknown of the root transport system (a node outside the MTG, as Root-CyNAPS does)? Or should it be a Plant-scale variable exchanged with the shoot component at fixed points? **Recommendation:** an extra unknown attached to the collar inside the solve (PT4), since the collar flux and the pool must be solved together for stability.
  → answer:
- **QPd, light (G6).** Keep Caribu as the engine, with metafspm providing per-element triangles, optical classes and a light component on the union of populations (the adapter downstream)? Or should metafspm also own the triangulation from organ dimensions (adel's leaf-shape database)? **Recommendation:** metafspm stores and transforms triangles; adel's geometry stays a structural component downstream that writes them.
  → answer:
- **QPe, soil water (G10).** Keep cmf as an opaque solver inside the soil component, with its state outside the DataStructure and saved by a checkpoint hook? Or reimplement Richards and solute transport as graph systems on the grid? **Recommendation:** keep cmf first (PT7); a graph-system version can follow and be compared against it.
  → answer:
- **QPf, which model first?** **Recommendation:** rhizodep (PT1–PT3 cover it), then Root-CyNAPS (PT4), then RhizoSoil, then the shoot.
  → answer:
- **GRANAP (G12, after Q-A4 in `devplan_datastructures.md`):**
  - **Q-A4:** is a template shared per class, or are Compartments copied per segment? Do state variables differ per segment within a class? (The audit finds that the solved graph needs per-segment Compartments in any case, since state variables differ; only the geometry would be shared.)
  - What defines a class: the diameter only, or diameter with age or distance from the tip?
  - When a segment changes class (aerenchyma, barrier maturation), is its anatomy replaced, or edited in place? How are cell contents carried over?
  - Which cells get axial junctions: only xylem and sieve tubes (as MECHA's axial K does), or also the apoplast and the symplast?
  - Are walls and wall junctions Compartments, or are only cells nodes, with walls as edges?
  - How many classes and segments do you expect? An anatomy is about 5–30 k graph nodes (1.4–5 k cells), so the full per-cell graph of a root system may be out of reach without a reduction per class.
  → answer:
