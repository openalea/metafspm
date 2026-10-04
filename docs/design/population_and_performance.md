# Design note: a plant population in one MPG, vectorised traversals, scheduling and persistence (step 5)

Status: **agreed** (2026-10-02: S1–S11 answered in §8 and §11, with refinements in §9–§12). 5a done (`bb36bd4`); F1–F2 agreed (§13). The steps after 5a continue in `devplan_population_scene.md` (P1–P8). It covers step 5 of `devplan_datastructures.md` §7, reordered by your answers to PA1–PA2 in `devplan_scene_paralellization.md`:
- the population prototype first: DS14b (MPG array mirrors), DS14a (tree kernels on the DataStructure), and N plants in one MPG;
- then DS13 (scheduling, reconsidered) and DS15 (persistence).

Branch `data_structure_api`, written against `258609f` (steps 1 to 4 complete). Code starts only after the questions are answered.

## 1. Where we start from (code facts)

**Scene.**
- `play_Orchestra` runs each plant in its own process. They exchange with the soil and light processes through `SharedMemory` buffers and queues.
- Each worker is pinned to a core, with `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / … set to 1.
- One plant per process is also forced by the Choregrapher, which binds a class's steps to its **last** instance (DS13; seen again in 4b's tests).

**Traversals, in Python.**
- `MPG.pre_order_mpg`, `post_order_mpg`, `components_iter`, `complex_at_scale`, and `populate_graph`'s link search (`_tip_component`, `linked_parent`) loop over openalea.mtg's dicts (`_parent`, `_children`, `_complex`).
- Some go through openalea.mtg's recursive `pre_order`: `populate_graph` on a single 20 000-segment chain raises `RecursionError` (found in 4a).
- In the DataStructure, `_membership` (owners at a coarse scale) loops over every node.
- The traversal arrays of step 2a (`parents`, `children`, `order`) come from the Connections, so they are already arrays.

**rhizodep's whole-plant loops** (`root_growth.py`, read only):
- `update_distance_from_tip`: post-order over the MTG, `d(v) = d(successor) + length(v)`, and `d = length` at an apex. This is a **reverse cumulative sum along each axis**.
- `calculating_supply_for_elongation`, per apex: walk towards the base (the `<` parent, then the `+` parent at an axis base), adding `C_hexose_root · struct_mass` and `struct_mass` until the volume `growing_zone_factor · radius · πr²` is reached, with a fractional last segment. It reads values without depleting them.
- `actual_growth_and_corresponding_respiration`: each apex's demand is limited by its own window, and the consumption is **accumulated** (`+=`) onto the supplying segments. Since sums commute, the result does not depend on the visiting order, **up to floating-point rounding**.
- `segmentation_and_primordia_formation` and `post_growth_updating`: per-apex edits, then a pass over every vertex.

## 2. MPG array mirrors (DS14b), sub-step 5a

- **`g.topology_arrays()`** returns integer arrays indexed by vid: `parent` (−1 for none), `complex`, `scale`, `edge_type` (`<` / `+` / `/` as codes) and `is_anchor`.
  - They are built in one pass over the MTG's dicts.
  - They are cached by the MPG signature of step 2b (vertex count, last vertex id). Any edit through the MTG API invalidates them, and they are rebuilt on the next use.
  - Rebuilding is O(n) with `np.fromiter`; incremental updates are left for later, if profiling asks for them (S4).
- **Derived from them, vectorised:**
  - children (CSR, by an `argsort` of `parent`);
  - the components of each complex;
  - `depth`;
  - `complex_at_scale(vids, scale)` for many vids at once, by pointer jumping on `complex`;
  - the linked parent of `populate_graph` / `wire_junctions`: the within-scale parent, or the tip of the complex parent, found by a group-by on `complex`.
- **Rewritten on the arrays:**
  - `populate_graph`, producing **the same Compartments and Connections, in the same order**, as today (tested vertex for vertex on the seedling and on generated root systems);
  - `wire_junctions`;
  - `MPGDataStructure._membership` and the owner maps.

  `pre_order_mpg` / `post_order_mpg` become iterative, so there is no recursion limit.
- **Unchanged:** openalea.mtg stays the editing API.

## 3. Tree kernels on the DataStructure (DS14a), sub-step 5b

All in local indices, cached per topology version, and vectorised over every plant of the DataStructure at once:

| Kernel | Use | How |
|---|---|---|
| `ds.levels()`, `ds.depth()` | nodes grouped by depth | frontier from the roots |
| `ds.accumulate(values, direction="up" \| "down", op="sum" \| "max")` | totals below a segment, propagation from the collar | one `np.add.at` / `np.maximum.at` per level |
| `ds.axes()` | axis id and position of each node along it (chains of `<` edges) | from `edge_type` and the levels |
| `ds.axis_scan(values, reverse=True)` | `distance_from_tip`, `axis_apex_id` | a segmented cumulative sum along each axis |
| `ds.path_window(budget, extent, values)` | rhizodep's supply for elongation: the sums of `values` over the ancestors of each node, until `extent` reaches `budget`, with a fractional last element | a compiled loop per node (numba, `prange` over nodes), see S1 |

**Exactness (PA1, S1).**
- `axis_scan` adds the lengths from the tip to the base, exactly in rhizodep's order (`d(successor) + length`), so it is **bitwise identical**.
- `path_window` walks each apex's ancestors in rhizodep's order, so the summation order is the same and the results are **bitwise identical**. It is compiled and runs in parallel over the apices.
- Consumption accumulated onto the supplying segments with `np.add.at` follows the apex order of the arrays, not rhizodep's post-order. It can differ in the last bits (about 1e-16 relative) where windows overlap. S1 asks whether that is acceptable.
- **Validation:** a rhizodep-like growth helper in this repo implements these rules both ways, with Python loops copied from rhizodep's algorithms and with the kernels, on generated root systems. The test compares them bitwise for the scans and windows, and at 1e-14 relative for the accumulations.

## 4. A population in one MPG: prototype and benchmarks, sub-step 5c

- **Population.**
  - A generator (test helper) builds N root systems in one MPG, one Plant-scale vertex each, with rhizodep-like branching: seminal axes and laterals at an inter-primordium distance.
  - Each plant's coordinates and per-plant parameters are `"Plant"`-located variables, broadcast to segments where needed (step 1 mappings).
- **Components.** One instance of each component for the whole population, on one `MPGDataStructure`:
  - the growth helper (5b kernels, plus the decide-then-apply segmentation);
  - a transport graph system;
  - a carbon `@rate` model.

  One `Coupler` maps every plant to the soil grid. An in-process soil runs with them, with no multiprocessing.
- **Graph solves.** Independent plants make a **block-diagonal** system. Sparse direct solvers keep each block's fill-in, so one solve on the whole population costs about the sum of the plants' solves. Two options:
  - **one** system, the simplest;
  - **connected components solved separately**, in parallel threads or processes, if memory or time requires it (S2).

  The prototype measures the first and reports.
- **Benchmarks**, reported in the devlog and in this note:
  - time per step, for N = 1, 10, 100, 1000 plants of about 2 000 segments each (S2);
  - split into topology update, tree kernels, rates, graph solve and soil exchange;
  - against today's scene: one process per plant, with its fixed per-process and per-message costs.
- **Decision point after 5c:** whether the population mode replaces the one-plant-per-process scene for typical runs, and what remains of DS13.

## 5. Scheduling (DS13, reconsidered), sub-step 5d

- **If the population lives in one MPG,** one instance per component class is the norm, and the remaining need is several **different** components per plant. Those already work, since they are different classes.
- **The class-name binding still has two hazards:**
  - two instances of one class in a process (tests, or two populations with different parameters);
  - two classes with the same name in different modules.
- **Minimal fix:** steps are registered per class object (not per name), and bound per instance at construction. `Component.__call__` runs its own instance's schedule. This keeps the Choregrapher API for decorators.
- **Validation:** two instances of one class with different DataStructures, called in any order (the cases that 4b had to serialise).

## 6. Persistence (DS15), sub-step 5e

- **`ds.checkpoint(path)` / `MPGDataStructure.restore(path, mtg)`**:
  - arrays per location, with dtypes (int, object pickled);
  - variable metadata, aliases, derivation specs (formulas by reference), masks, `topology_version` and write counters;
  - the MPG itself pickled alongside, or rebuilt by the caller.
- **Format:** `npz` for numeric arrays and a JSON manifest, with a pickle only for object variables and formulas (S5).
- **Validation:** checkpoint, restart, and two more steps equal two uninterrupted steps, bitwise.

## 7. On parallelism (your question)

**Yes.** "Vectorised numpy is single-threaded" means only that a numpy expression such as `a * b + c` runs on one core. Several tools use more:
- **numba:** `@njit(parallel=True)` with `prange` runs compiled loops on all cores. `path_window` and any remaining sequential rule would be written that way, at C speed and multi-threaded.
- **Element-wise expressions:** `numexpr` evaluates long expressions multi-threaded. `numba.vectorize(target="parallel")` does the same for user functions.
- **Linear algebra:** dense BLAS (`np.dot`, `scipy.linalg`) is already multi-threaded. Sparse direct solvers are mostly single-threaded:
  - `pypardiso` (MKL PARDISO) or `scikit-umfpack` are multi-threaded drop-ins for `spsolve`;
  - block-diagonal plant systems can also be split by connected component and solved in parallel.
- **Other:** JAX or CuPy (GPU) are possible later for large element-wise work, at the cost of a heavier dependency.

**One practical consequence:** the scene workers set `OMP_NUM_THREADS = 1`, and similar, so that one-plant-per-process does not oversubscribe the cores. A population process would instead own several cores, and set these variables (and numba's thread count) accordingly.

## 8. Questions (answer inline)

- **S1, exactness.** Scans and supply windows can be bitwise identical to rhizodep's loops (same summation order). Accumulating consumption onto shared supplying segments in array order can differ from rhizodep's post-order in the last bits (about 1e-16 relative) where windows overlap. Is "identical up to rounding (1e-14 relative)" acceptable there? The alternative is a numba loop in rhizodep's exact visiting order, which is bitwise identical but sequential.
  → answer: yes it is acceptable only if we garanty it won't be a larger difference than that.
- **S2, target sizes.**
  - How many plants, and how many segments per plant, should the prototype be benchmarked for? I propose 1 to 1000 plants of about 2 000 segments.
  - Do anatomies come in later (DS8 anatomy mode, about 10 to 50 Compartments per segment)? If so, the graph-solve strategy matters more (one system, or a solve per connected component).
  → answer: yes good target for number of plants and SubOrgan segments, and for anatomies, yes they will come into place. Regarding the solver, I don't know what is better between one system but might struggle or a solve per connected graph, i.e. plants within the single MPG will be isolated from each other because they are only connected through the environment and so the graph_view will yield them disconnected.
- **S3, numba as a dependency.** Required, or optional with a slow pure-Python fallback for `path_window`? **Recommendation:** required, since it is on conda-forge and pip, and the population mode needs it.
  → answer: yes numba can be a dependancy, it is robust enough to be required by metafspm, I think only the GPU interfacing should be optional.
- **S4, array mirrors:** rebuilt from the MTG dicts after any edit (O(n), simple), with incremental updates later only if profiling shows the rebuild matters. **Recommendation:** yes.
  → answer: yes
- **S5, persistence format:** `npz` + a JSON manifest, with pickle only for object variables and formulas. Or one pickle of everything, simpler but tied to class versions. **Recommendation:** npz + JSON.
  → answer: I don't understand the question
- **S6, the prototype scene:** soil, light and the population in one process, run sequentially, with no multiprocessing. Or keep soil and light in their own processes, with the population in one process through the existing protocol. **Recommendation:** first all in one process, to measure the pure computation, then decide.
  → answer: all in one process sequentially, with environment models first and then plants within a single iteration.
- **S7, rhizodep rules in the prototype:** reimplemented in this repo's test helpers from rhizodep's algorithms (read only, without importing rhizodep), with the Python-loop version kept as the reference. **Recommendation:** yes, within the scope rule.
  → answer: yes, but I fear a bit that you adapt everything to rhizodep and don't anticipate well enough for other types of structural models (e.g.cnwgrass' morphogenesis and adelwheat models, or GRANAP, to give examples of components I will LATER want to reimplement with metafspm, just examples for now), which might bring other necessary-iteration-problems of this kind. 
- **S8, order:** 5a (mirrors) → 5b (kernels) → 5c (population prototype and benchmarks, then decision) → 5d (DS13) → 5e (DS15). **Recommendation:** yes.
  → answer: yes for the order

## 9. After your answers (2026-10-02)

**S1, the guarantee.**
- **By construction, bitwise.** `np.add.at` applies its additions in the order of its input arrays. The consumption contributions are emitted in rhizodep's visiting order: apices in the MPG post-order rebuilt from the array mirrors (5a), and supplying segments in walking order. The sequence of additions on each segment is then rhizodep's, so the result is identical bit for bit, with no tolerance needed.
- **The bound, if another order is ever used.** All contributions are positive, so the rounding error of a sum of k terms is at most `(k − 1)·ε·sum`, with ε = 1.1·10⁻¹⁶. Up to 90 apices sharing a segment stays below 10⁻¹⁴ relative.
- **Enforced:** the tests compare against the reference loops, bitwise for the post-order emission, with a test of maximal window overlap.

**S2, solves per connected component.**
- Plants of one MPG are linked only through the environment, so their graph is disconnected, as you say. Anatomies (DS8) add more nodes per plant, but still one connected piece per plant.
- **Proposal:** `@graph_system(split="components")`, the default when the graph has several pieces.
  - The pieces are computed once per topology version, with `scipy.sparse.csgraph.connected_components`, which step 2d already uses.
  - Each piece is solved separately, with its own Newton convergence, in a thread pool: SuperLU and scipy's sparse kernels release the GIL during the solve. `split="whole"` keeps one system.
- **Why:** memory stays per plant, a plant that converges slowly does not hold the others, and the work spreads over cores without processes. Each piece is a restriction of the step 2d kind (sliced views and scatter-back), so the machinery exists.
- 5c benchmarks both modes, with and without anatomies.

**S5, re-explained.** "Persistence" means saving a running simulation to disk and restarting it later from that point, as if it had never stopped: to resume long runs, to restart after a crash, or to fork scenarios from a common state. Saving the MPG alone is not enough: the DataStructure holds things the MTG does not, such as derived-variable definitions, masks, write counters, typed arrays and local orders. Two formats are possible:
- a **plain pickle of everything**: one call, but tied to the exact Python classes and versions; a renamed class makes old saves unreadable;
- **`npz` arrays plus a readable JSON manifest** (the recommendation): the arrays are stored as data, and the JSON lists variables, locations, dtypes, scales and links. Saves stay readable across code versions and inspectable outside Python. Pickle is used only for what has no data form: object variables, and formulas passed as Python functions.

**S6:** agreed, all in one process: environment models first, then the plants, within each iteration.

**S7, beyond rhizodep.** A read-only survey of cnwgrass (morphogenesis, growth, senescence), adel (AdelWheat dynamic) and GRANAP lists the iteration patterns these models use. The kernel library of 5b is designed for all of them, and rhizodep is only the first validation case:

| Pattern | Example | Kernel |
|---|---|---|
| map with broadcast from an ancestor scale | element ages from axis teq (cnwgrass); plant placement (adel) | step 1 mappings (already there) |
| segmented prefix scan along a chain, **sum or max**, inclusive or exclusive | distance from tip (rhizodep); SAM height, ligule heights; pseudostem = prefix-max of ligule heights − Σ internodes (cnwgrass) | `ds.chain_scan(values, chain=, op="sum" \| "max", reverse=, exclusive=)` |
| accumulate tips → roots, propagate roots → tips | totals below a segment; inherited values | `ds.accumulate(values, direction=, op=)` |
| lagged neighbours and forward writes along a chain | rank n reads n−1 and n−2; leaf n's emergence sets `Lmax`, `pseudo_age` of n+1 (cnwgrass) | `ds.chain_shift(values, k)` and `ds.chain_write(event_mask, values, k=1)` |
| group-by reduce and scatter | organ area = Σ elements (adel); hidden zone → its lamina and sheath (cnwgrass) | coarse locations and mappings (step 1), `ds.group_reduce` for ad hoc groups |
| indexed gather across chains | tiller rank n copies main-stem rank `cohort + n − 1` (cnwgrass) | `ds.chain_gather(values, from_chain=, rank=)` |
| windowed walk to ancestors with a budget | rhizodep's supply for elongation | `ds.path_window` (numba) |
| composition of transforms along paths, with branch restore | the turtle frame of adel's geometry | `ds.path_compose(transforms)`, an associative scan of 4×4 matrices by levels |
| non-associative clamped recurrences | adel's whorl height; GRANAP's layer buffering | `ds.chain_recurrence(fn)`: a numba loop along chain positions, vectorised across all chains |
| budgeted greedy depletion within groups | GRANAP's aerenchyma per sector | `ds.group_budget(priority, amount, budget)`: a sorted cumulative sum and threshold per group |
| ragged expand, and connected components | n cells per layer (GRANAP); merging cells | `np.repeat`-style expand, and `connected_components` |
| topology events | primordia, elements, metamers | StructuralComponent's MPG-style steps (step 2b), with bulk decide-then-apply where possible |

- **"Chain".** A chain is any ordered sequence: an axis (`<` successors), the metamer ranks of an axis, or the layers of a section. It is given by an edge type or by a rank variable, so the same kernels serve phytomer-based shoots and segment-based roots.
- **Validation in 5b:** one reference-loop case per row, taken from these models' rules and reimplemented in test helpers (scope rule), not only rhizodep.

## 10. Readability (your question)

**The aim is that a model reads like its equations, with every traversal behind a named kernel.** Three rules keep it so:
1. **Model code never handles indices, orders or traversals.** It calls named kernels (`chain_scan`, `accumulate`, `path_window`, …), and the framework owns their vectorised or numba implementation.
2. **Per-element logic stays per element where it is clearer.** `@rate(vectorized=False)` remains for rules full of branches, and numba compiles those steps when they are hot, without rewriting them.
3. **Declarations carry the documentation** (units, scale, kind, `dtype`, `on_grow`), as today.

**Example: rhizodep's `update_distance_from_tip`** today is a post-order loop of 14 lines through the MTG node API (`Successor`, `node`, label and type tests). As a step:

```python
@postsegmentation
def _distance_from_tip(self, length):
    return self.data_structure.chain_scan(length, chain="axis", reverse=True)   # d = Σ lengths from the tip
```

**Example: the supply for elongation** today is a 60-line `while` walk per apex, with lists and index bookkeeping. As a step:

```python
@potential
def _growing_zone_C_hexose_root(self, C_hexose_root, struct_mass, volume, radius, is_apex):
    budget = self.growing_zone_factor * radius * np.pi * radius ** 2
    hexose, mass = self.data_structure.path_window(budget, extent=volume, values=(C_hexose_root * struct_mass,
                                                                                 struct_mass), where=is_apex)
    return np.where(mass > 0, hexose / mass, 0.)
```

- **What gets easier:** the rule fits on a few lines that read like the paper's equations, and changing it (another budget, another weight) is a one-line edit. The whole population runs at once, and the tests compare it with the reference.
- **What costs some readability:**
  - conditionals become `np.where` or masks, unless the step opts into `vectorized=False`;
  - order-dependent rules must be restated explicitly (decide, then apply). That is more honest about the model, but it is a rewrite.

  The guideline is to prefer a named kernel to clever indexing. A model whose rule needs a new traversal pattern adds a kernel to the framework, documented and tested once, instead of a loop in the model.
- **Compared with the three examples:**
  - rhizodep's growth (MTG node loops) gains the most;
  - cnwgrass's dict-of-tuples loops become chain scans and event shifts on declared variables;
  - adel's turtle and whorl recurrences become `path_compose` and `chain_recurrence`, compiled;
  - GRANAP is mostly per-section geometry and generation, which stays StructuralComponent code (MPG-style), with only its few array rules on kernels.

## 11. Follow-up questions

- **S9, chains.** Should an axis chain follow `<` edges at SubOrgan scale (rhizodep), and a phytomer chain follow metamer ranks at Phytomer scale (cnwgrass, adel), each declared by name (`chain="axis"`, `chain="phytomer_rank"`), with the chain definition (edge type or rank variable) given by the structural component?
  → answer: yes if that fits with those model logic and unifies as a chain logic for metafspm's StructuralComponents
- **S10, the turtle (adel geometry).** Is plant geometry (turtle frames) meant to come into metafspm as a StructuralComponent output (coordinates as variables, as `x1 … z2` now)? If so, `path_compose` is in scope; if geometry stays in adel or PlantGL, it is not.
  → answer: This adds another question: x, y, z, 1 and 2, are needed to position elements, for example to compute the interception with a grid like in the soil, or to provide scenes to the light models, so I think yes it is in the scope. 
- **S11, the 5b scope.** Implement now only the kernels the prototype needs (chain_scan, accumulate, path_window, chain_shift / chain_write), and the others when the corresponding model is reimplemented. Or implement the whole table now, with the survey-based reference tests. **Recommendation:** the first, with the table kept as the agreed API, so that names and signatures are fixed now.
  → answer: the first yes, only include additional if this is decisive to stabilize the API as general enough for the different possible use cases.

## 12. Consequences of S9–S11

- **S9:** chains are declared by name on the StructuralComponent, with an edge type (`<` successors) or a rank variable defining each one, so that one chain logic serves root axes and phytomer ranks.
- **S10:** geometry is in scope. Segment coordinates (`x1 … z2`) position elements for the soil Coupler and for light scenes.
- **S11:** only the prototype's kernels are implemented in 5b: `chain_scan`, `accumulate`, `path_window`, `chain_shift` / `chain_write`.
  - **Added, because they decide the API's generality:** values may be **vector-valued** (shape `(n, k)`, e.g. 3-D coordinates propagated from the base: `x1 = parent's x2`, `x2 = x1 + length · direction`).
  - `path_compose` keeps its signature (one 4×4 transform per node), with a minimal implementation, so that matrix-valued scans are part of the API from the start.
  - The rest of the §9 table follows when each model is reimplemented.

## 13. Findings of 5a, and questions

- **Recursion.** The recursion came from openalea.mtg's `components_iter` (recursive `pre_order`), which every MPG traversal calls. An iterative override in `MPG`, with the same order (tested against openalea's on every complex, and through identical `populate_graph` results), removes it.
- **`complex()` is O(depth) per vertex** in openalea.mtg: only the roots of component trees store their complex, and the others walk up their parent chain. The topology arrays resolve every vertex in O(n log depth). The owner maps went from 131 to 53 ms on 20 000 segments.
- **`populate_graph`'s cost is vertex creation, not traversal.** On 20 000 segments it takes 2.26 s:
  - creating the 40 001 Compartment and Connection vertices through `add_component` takes 1.30 s, mostly 160 000 single-item `ArrayDict` inserts (1.09 s, each a `searchsorted`);
  - the traversal takes 0.36 s.

  Vectorising the link search would therefore gain little. At the 5c target (1000 plants of 2 000 segments, 2·10⁶ segments), a full repopulation would take minutes. In segment mode, `update_topology()` repopulates everything at every growth step (`repopulate_graph`).
- **F1, a scene robustness gap (not caused by 5a).** During 5a, a bug made the plant workers fail when constructing their DataStructure. `test_data_structure_scene[fork]` then hung until its 60 s watchdog killed pytest, instead of failing.
  - The cause is in `scene_wrapper`: an environment worker constructs its model outside its `try`, and that constructor waits on a queue for the plants' first messages.
  - The main loop watches only `stop_event`, not the workers' exit codes.

  Proposed fix: the main loop also sets `stop_event` when any worker has exited with a non-zero code. After a grace period, `finally` terminates the workers still blocked on queues (`p.join(timeout)`, then `p.terminate()`), and the scene returns `clean_exit = False`. Tested with a plant whose constructor raises.
  → answer: agree
- **F2, populating at population scale.** Two complementary options:
  - **incremental population in segment mode:** `update_topology()` creates Compartments and Connections only for new segments, and keeps the others with their vids (as anatomy mode already does for junctions, D12). This also keeps edge identities stable, and so edge values (P6 kept child-vid edge ids precisely because repopulation recreated them);
  - **bulk vertex creation:** an `MPG` method creating many vertices and their properties in one go, with one batched `ArrayDict` assignment per property instead of one insert per vertex per property.

  **Recommendation:** both, as the first part of 5c, measured before and after on the population, with `repopulate_graph` kept for explicit full rebuilds.
  → answer: agree
