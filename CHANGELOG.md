# Changelog

## Unreleased (release2026)

### Breaking: module layout

The flat modules used before the 2026 restructure have been removed. No compatibility aliases are provided, so downstream packages must import from the new paths:

| Old import | New import |
|---|---|
| `openalea.metafspm.utils.ArrayDict` | `openalea.metafspm.data_structure.arraydict.ArrayDict` |
| `openalea.metafspm.utils.mtg_to_arraydict` | removed. Store variables in a `DataStructure` (`MPGDataStructure` for plants, the 3-D grid for soil). `MPG.convert_properties_to_arraydict` covers the transition |
| `openalea.metafspm.composite_wrapper.CompositeModel` | `openalea.metafspm.coupling.composite_wrapper.CompositeModel` |
| `openalea.metafspm.scene_wrapper.play_Orchestra` | `openalea.metafspm.scene.scene_wrapper.play_Orchestra` |
| `openalea.metafspm.component_factory.Choregrapher` | `openalea.metafspm.coupling.choregrapher.Choregrapher` |
| `openalea.metafspm.component_factory` step decorators (`rate`, `state`, `actual`, `potential`, …) | `openalea.metafspm.solve.decorator` |
| `openalea.metafspm.component_factory.Functor` | `openalea.metafspm.solve.legacy_functor.Functor` |
| `openalea.metafspm.component.Model`, `declare` | `openalea.metafspm.coupling.component.Component` / `FunctionalComponent`, `declare` |
| `openalea.metafspm.specializer` | `openalea.metafspm.solve.specializer` |

### Reproducible random draws (PT2)

- **New `data_structure/random_streams.py`:** counter-based draws (uniform, normal, exponential, integers), a pure function of (seed, stream, step, entity id).
- **`ds.random(...)`** and the component helper **`self.random(stream, ...)`**: its step counters live on the DataStructure and are kept by checkpoints; `random_seed` chooses the seed.
- Draws do not depend on the visiting order or on the other entities (QPa: rhizodep's global re-seeding is replaced by per-vertex streams with the same distributions).

### Tree kernels, round 2 (PT1)

- **`ds.fold(update, values, direction="up" | "down")`:** a level-by-level fold with a custom vectorised function. Each `FoldLevel` gives its nodes, their edge types, `children(values, op, edge=, where=, fill=)` (sum, max, min, all, any, count) and `parent(values)`. It is for nonlinear pipe models, death propagation, filtered maxima over laterals and turtle-like frames; tested exactly against rhizodep's and Root-CyNAPS' loops.
- **`ds.chain_gather(values, rank, chain=, source=)`:** read the value at a rank of another chain (a tiller's metamer reading the main stem's at `cohort + n − 1`).
- **`ds.chain_recurrence(update, values, chain=)`:** non-associative recurrences along chains, vectorised across chains.
- `ds.levels()` is cached per topology.

### Growth bookkeeping proportional to the growth (F5)

- `MPG.topology_arrays()` is extended for the vertices created since the last read (rebuilt only after removals). The vid → index dict is built lazily, graph builders look vids up in bulk, and the carry-over matches ids once per location.
- 1000 plants of 2 000 segments: 3.3 → 2.0 s per step.

### Removed: `solve/specializer.py` (Q29)

- It was unused since the legacy path was removed. Numba is used by steps directly: a step calls an `@njit` function on the arrays it receives.

### Graph systems solved per connected piece (S2)

- `@graph_system(split="components")` solves each connected piece of the graph (of the active subgraph with `where=`), e.g. each plant of a population, on its own, with its own Newton convergence and adaptive steps. A piece gives the same values as its plant solved alone (to 1e-15 in the tests).
- The pieces are computed once per topology, each with a local incidence. The restricted-solve bookkeeping is now proportional to the subgraph: in-place scatters, previous fields updated at the subgraph's nodes, adaptive capture and restore of the subgraph only. This also helps `where=` solves.
- `split="whole"` stays the default: on one core the pieces' per-solve overhead makes `components` about 1.3–1.6× slower at 100 plants (see the design doc §16).

### Consumption shared in visiting order (S1)

- `tree_kernels.path_contributions` / `ds.path_contributions(budget, extent, values, where=, include=)` give each supply window of `path_window` element by element as (owner, supplier, contribution), emitted in visiting order. `tree_kernels.scatter_contributions` / `ds.scatter_contributions` then add `amount[owner] · contribution / total[owner]` onto the suppliers one after the other.
- Bit for bit equal to rhizodep's sharing of each apex's consumption between its supplying segments (tested with partial and maximal window overlap). Accumulating in array order differs in the last bits.
- `ds.order("post", convention="openalea")`: the post order of openalea's `post_order2` (the successor subtree first, then the branches in reverse insertion order), the visiting order of models written with it.

### Checkpoints (DS15, P8)

- **`ds.checkpoint(path, include_mtg=True)` / `MPGDataStructure.restore(path, mtg=None)` / `ArrayDataStructure.restore(path)`:** a folder holding:
  - `arrays.npz`: numeric variables, and the entity ids, checked at restore;
  - `manifest.json`: class, construction, variables with location and dtype, counters;
  - `state.pkl`: object variables, metadata, aliases, derivations, masks, write counters, the anatomy wiring state, and the MPG.

  A restored DataStructure continues bit for bit; components are built again on it and keep its values.
- Derivation formulas and mask rules are kept by reference: module-level functions or picklable objects. Lambdas are refused with that hint. The Scene's combined emergence mask is now a picklable `AllMasks`.
- **`LabelsConfig` pickles** (its per-instance label groups are rebuilt on load), so MPGs can be pickled.

### Per-instance scheduling (DS13 hazard fix, P8)

- Steps are registered per class, keyed by module and qualified name (`choregrapher.family_of`), and bound per instance at construction; `Component.__call__` runs its own instance's schedule. Several instances of one class run on their own DataStructures in any order, and same-named classes of different modules no longer collide.
- **Fix:** subclasses run their bases' steps, a redefined step replacing its base's. Before, a subclass without steps of its own silently ran nothing. The `inheriting` globals mechanism is removed.
- **Introspection:** `Choregrapher().schedule_of(cls)` gives a class's unbound schedule. `scheduled_groups` is keyed by `module:qualname`.

### Performance at population scale (F3, F4)

- **Breaking: lazy MTG synchronisation.** `mtg_sync = "lazy"` is the new default, and the DataStructure is the reference. A state variable changed since its last synchronisation is written to the MTG:
  - before an MPG-style step;
  - when the MTG is read through `ds.mtg` / `component.mtg`;
  - by `ds.flush_mtg()`.

  Code reading the MTG object itself calls `ds.flush_mtg()` first; `mtg_sync = "after_call"` restores the former behaviour. Rates and states no longer pay the MTG write after each call, which was 97 % of their time at 2·10⁵ segments.
- **An MPG-style step flushes the variables its component declares** (states, inputs, parameters): it sees what it declares. Numeric MTG properties written by the DataStructure are `ArrayDict`s, so writes are array assignments (a plain dict made by MPG-style code is converted once); integer properties stay dicts. `MPGDataStructure.edges()` is cached until the topology or the MTG changes.
- **Faster growth bookkeeping:** `extend_graph`, the variable carry-over of `update_topology` and `topology_arrays` work on property arrays instead of MTG traversals and per-vertex loops. Writes to MTG properties that are not keyed like the nodes are batched.
- **Fix:** in anatomy mode, `update_topology` pairs the stored edge values with the ids they were built for. It used to rely on new Connections having the largest vids.

### Breaking: the one-plant-per-process scene is removed (P7)

- **Removed:**
  - `openalea.metafspm.scene.scene_wrapper`: `play_Orchestra`, the plant, soil and light workers, the CPU-affinity helpers and the queues;
  - `openalea.metafspm.coupling.coupler`: `Coupler`, `Transport`, `BufferPlantView`, `VoxelLocator`.

  Use `openalea.metafspm.scene.scene.Scene`, which also takes the package-level import `from openalea.metafspm import Scene, planting_table`. The plant ↔ environment exchanges go through `coupling.cross` (`CrossMapping`, `Exchanges`).
- **`stand_initialization`** moved to `scene/population.py`.
- **`CompositeModel`:** `soil_name`, `soil_inputs`, `soil_outputs` and `get_component_inputs_outputs` are removed; the translator query is `Translator.inputs_outputs`. A plant composite no longer registers its soil inputs and resets them to 0 at coupling: they keep their declared `initialize` until the scene's first exchange.
- **Tests:** the multiprocessing tests and their doubles are removed, with the `slow` marker. The plant / soil contract of the DataStructure doubles now runs in a `Scene` (`test/wrappers_tests/test_scene_contract.py`).
- **Benchmarks:** `test/benchmarks/bench_population.py` (results in `docs/design/population_and_performance.md` §14). On toy plants, the Scene's per-step overhead is flat, while the per-process scene's grew with the plants (0.27 ms against 6.6 ms for 12 plants). 1000 plants of 2 000 segments take 6.7 s per step in one process; findings F3 and F4 there are about 4.5 s of it.

### The population scene (P6)

- **New `scene/scene.py`, `Scene(CompositeModel)`:** the populations and the environment models of a stand, in one process.
  - **Populations:** one per model in the planting table. Each is built once on an MPG holding all its plants, as `Model(data_structure, time_step, **scenario)`, with the class attributes `initiators`, `from_scale` and `nodes`. Numeric parameters come per plant from the table; other scenario entries must be shared.
  - **Environment models:** `Model(populations, scene_xrange, scene_yrange, time_step, **scenario)` builds its own grid or `UnionDataStructure`.
  - **Coupling:** mappings are inferred from the scene translator (a `CrossMapping` per population and grid pair, a `UnionMapping` per union) and run through one `Exchanges`.
  - **Step:** each environment model after the exchanges into it, then each population after the exchanges into it.
  - Component classes must differ between models.
- **Staggered emergence:** an `emergence_time` column (s) keeps a plant frozen until it emerges. Its nodes leave the `active` mask, combined with a model's own `active` rule, and the mappings (`CrossMapping(mask=)`); values of entities left out are kept.
  - MPG-style steps restrict themselves with the new `DataStructureComponent.active_ids()`.
- **`SceneRecorder`:** for each population, `summaries.csv` gets one row per plant per step (sums of the extensive and means of the intensive state variables). `segments.csv` gets the `log_plants`' segments every `heavy_log_period` steps. A `logger_class` hook remains for adapting an external Logger.
- **`planting_table`:** a new `emergence_times=` argument, and the stand size is kept in `table.attrs`.

### Links between DataStructures (P5)

- **New `coupling/cross.py`:**
  - `CrossMapping(plants, grid, method="barycentre" | "overlap")`: plant segments to grid cells. Barycentre is the reference soil map; overlap weights each cell by the segment's length fraction in it. The map rebuilds itself after growth or a coordinate write.
  - `Exchanges(translator, components, mappings)`: translator links between components on different DataStructures, run by `exchange(into=ds)` at the scene's fixed points.
- **Defaults** follow the variables' kinds:
  - extensive plant → cell: `sum`;
  - intensive plant → cell: `weighted_mean`, `weight=` required;
  - intensive cell → plant: `broadcast` (the overlap-weighted cell values);
  - extensive cell → plant: `split` by `weight=`.
- **Several populations** feeding one grid variable are pooled in one write: sums add up, means pool their weights, and splits share each cell among every receiving population. `zero_soil_inputs()` is no longer needed.
- **`UnionDataStructure` and `UnionMapping`:** the nodes of several populations, one after the other, so that one component (a CARIBU-like light model) sees every population. A component declared at the populations' node scale runs on a union unchanged. Values follow the parts' growth, kept by (part, entity id).
- **`Coupler`** now uses `CrossMapping` (same map); it and `Transport` stay until P7.

### Plant populations and per-plant parameters (P4)

- **Parameters per plant.** A numeric `parameter(...)` declared without a scale is stored per plant (`"Plant"` location) on plant DataStructures, and at `"scalar"` on grids. String, boolean and object parameters stay plain attributes.
- **Parameters are arguments.** Steps and graph-system equations take parameters as arguments (`def _rate(self, hexose, k)`), broadcast to the equation's entities: nodes, or edges through their child's plant. Uniform values come as zero-stride read-only views, varied ones as cached gathers, so numba steps get float arrays in both cases.
- **`self.<parameter>`:**
  - inside a step or an equation, reading it raises `AttributeError`, naming the argument to add;
  - outside, it reads the population value (or the per-plant values), and writing it sets every plant.

  MPG-style structural steps read per-vertex values with `self.parameter_values(name)`.
- **New `scene/population.py`:**
  - `planting_table(...)`: `stand_initialization`'s layout, with one scenario per plant possible;
  - `build_population(table, initiators)`: one Plant vertex per plant, each plant's initial structure built by `StructuralComponent.initiate_plant(g, plant, parameters)`, in order;
  - `apply_plant_scenarios(ds, components, table, plants)`: per-plant parameters from the scenarios.
- **Fix:** the declaration resolver no longer converts the defaults of fields that are not DataStructure variables, so string parameters without a scale raised before.

### Tree kernels (P3)

- **New `data_structure/tree_kernels.py`**, through `MPGDataStructure` methods, vectorised over every plant of the DataStructure:
  - `define_chain(name, edge_type="<" | group=, rank=)` / `chain(name)`: chains along '<' successors (`"axis"` by default) or by rank within a group;
  - `chain_scan(values, chain=, op="sum" | "max", reverse=, exclusive=)`;
  - `chain_shift(values, k)` and `chain_write(event, values, k)` (lagged neighbours and forward writes);
  - `accumulate(values, direction="up" | "down", op=)` (subtree or root-path sums or maxima, by levels);
  - `path_window(budget, extent, values, where=, include=)` (sums over ancestors up to a budget);
  - `path_compose(transforms)` (4×4 frames from the root);
  - `depth()`, `levels()`.
- **Values** may be vector-valued (`(n, k)`).
- **Implementation:** chain scans and path windows are numba loops, parallel over chains and over nodes. They add their terms in the order of the loops they replace, so rhizodep's distance from tip and supply for elongation are reproduced bit for bit (tested).

### Building the graph at population scale (P2)

- **Incremental growth updates.** `MPGDataStructure.update_topology()` (segment mode) calls the new `MPG.extend_graph(from_scale)`: only new vertices get Compartments and Connections, removed ones (pruned with `remove_tree`) lose theirs, and children whose linked parent changed are relinked. Every other Compartment and Connection keeps its vid, so edge values carry over by identity.
  - It falls back to a full rebuild only when a new vertex has no linked parent and its plant has others.
  - On 20 000 segments, an update after 10 new segments takes 0.42 s instead of 398 s: the full repopulation deleted the properties vertex by vertex, which is quadratic.
- **`MPG.add_components_bulk`** creates many components with one batched write per property. `populate_graph` uses it, with the same vids and properties (1.99 → 1.30 s on 20 000 segments).
- **Plants stay disconnected.** `populate_graph` chains orphan vertices only within one plant, so the plants of a population are no longer linked to each other.

### Scene: a failing worker stops the scene (P1)

- **Model construction** of the plant, soil and light workers now runs inside their `try`: a failure there stops the scene like a failure in a step. Before, a plant failing in its constructor left the soil blocked in its own constructor, waiting for that plant's first message, and the scene hung.
- **Exit codes.** `play_Orchestra` also stops the scene when a worker exits with a non-zero code.
- **`shutdown_timeout`** (default 30 s) bounds the wait for the workers after the scene stops. Workers still alive are terminated (then killed), the shared memory is released, and the scene returns `False`.

### MPG traversals without recursion, and topology arrays (step 5a)

- **`MPG.components_iter`** keeps openalea.mtg's order (component roots, then '+' children before '<' successors) without its recursive `pre_order`. `populate_graph` and the multiscale traversals work on long axes: a 20 000-segment chain raised `RecursionError` before.
- **`MPG.topology_arrays()`**: vid-indexed `parent`, `complex`, `scale`, `edge_type` and `is_anchor` arrays, cached until the MPG changes. `complex` is resolved for all vertices at once; openalea's `complex()` walks the parent chain of each vertex.
- **`MPG.complex_at_scale_array(vids, scale)`**, used by `MPGDataStructure`'s coarse-scale owner maps: 131 → 53 ms on 20 000 segments.

### Typed variables (step 4c)

- **`dtype=`** on `register` and declarations (`declare`, `state_variable`, `input_variable`, `parameter`):
  - `"int"`: labels, types and indices kept as integers, read from the MTG as integers; non-integral writes raise;
  - `"object"`: lists and records, one per entity, carried over by growth and written to and from the MTG by identity. Graph systems, derivations and soil transport reject them.
- **Label names.** Masks, boundary-set dict selects and graph-system filters accept label names, e.g. `{"label": ["RootSegment"]}`. They are resolved by `MPGDataStructure.label_code` / `resolve_codes` through the MTG's `LabelsConfig`: a label value (unique) first, then an attribute of the variable's own scale group, then any group. Ambiguous or unknown names raise.
- **`export()`** keeps each variable's dtype.

### Sub-stepping and adaptive integration of graph systems (step 4b)

- **`@graph_system(integrate=...)`**:
  - `"step"` (default, unchanged): one solve of `time_step`;
  - `"substeps"`: `n_substeps` solves of `time_step / n_substeps`, re-reading boundary sets between them;
  - `"adaptive"`: step doubling with `rtol` / `atol`, bounded by `min_step` / `max_step`. A step below `min_step` raises.
- **Time terms in equations.** Equations use `self.dt` (the current sub-step's length, equal to `time_step` with `"step"`) and `self.previous(fn)` (the state at the start of the current sub-step).
- **`previous(fn, at=...)`**: `at="solve"` gives the state at the start of the call's solve, and `at="step"` the state at the start of the component's call, for operator splitting between graph systems.
- **Fix: `ArrayDataStructure.laplacian()` on axes with a single cell.** It added a spurious `−1/h²` sink on such axes, e.g. a `(nx, 1, 1)` column lost mass. Such an axis now contributes nothing, like the face graph.

### MTG sync policy and read-only snapshots (step 4a)

- **`mtg_sync`.** The class attribute `DataStructureComponent.mtg_sync = "after_call"` (default) writes the state variables to the MTG after every call; `"never"` leaves the MTG untouched.
- **Parameter refresh.** MTG-backed parameters are re-read at the start of every call, as well as before each graph solve.
- **Read-only snapshots.** Graph-system snapshots of parameters and inputs are read-only views instead of copies: an equation writing into one raises `ValueError`. On 20 000 nodes, a snapshot of ten parameters takes 45 µs instead of 96 µs and copies nothing.

### Graph systems on grids (step 3d)

- **`@graph_system` solves on `ArrayDataStructure`s.**
  - Cell variables are flattened for the solve (C order) and written back on the cells, edge variables live on the faces, and graph outputs at "node" are cell variables.
  - Masks and boundary sets on cells (e.g. `boundary_set(select=lambda ds: ds.layer_mask(z=-1), kind="dirichlet", ...)`) work as on plant graphs.
  - `where=` takes a cell mask.
- **Flat writes.** `DataStructure.set()` accepts a flat array for a grid variable of the same size.

### Grid topology (step 3c)

- **`ArrayDataStructure` has a graph topology** (DS1, D1): cells are nodes (flat C order), faces between adjacent cells are edges, axis by axis, oriented towards increasing coordinates (`B[lower, e] = +1`).
  - `topology()` / `to_graph_view()`, `incidence_matrix()`, `edges()`, `n_nodes()`, `n_edges()`, `face_axis()`.
  - The DataStructure itself also gets `topology()` as an alias of `to_graph_view()`.
- **Periodic axes.** `ArrayDataStructure(..., periodic=(True, True, False))` adds wrap faces (not with 2 cells, where they would duplicate the internal face). `locate()` uses the grid's periodic axes by default.
- **Edge location on grids.** An `"edge"` location holds face variables. `face_area` and `face_distance` are registered at construction, so `B · diag(face_area / face_distance) · Bᵀ / cell_volume()` equals `−laplacian()`.
- **`layer_mask(x=, y=, z=)`** gives cell masks of given layers, for boundary sets.
- **Declarations on grids** accept `location="edge"`.

### Translator link scales and targets (step 3b)

- **`scale` / `source_scale` are checks.** When a link states them, the receiver's and the sources' declared locations must be those scales', otherwise the coupling raises. Declarations stay the reference (R1).
- **`target=` on links.** A link (Python `Translator().link(..., target=)`, or a nested spec with `"target"`) names a DataStructure mask. The mapped values go to its entities only, the others getting the receiver's default (R2). The mask must be defined before coupling.

### Cross-scale links: default mappings and kinds (step 3a)

- **Default mapping.** A translator link between two locations that gives no `aggregation` is mapped from its provider's `state_variable_type`:
  - extensive: `sum` up, and an error down;
  - intensive: `mean` up, `broadcast` down;
  - massic concentration: `weighted_mean` up (the link's `weight` is required), `broadcast` down;
  - missing or mixed kinds: an error naming the link.
- **Aliases across scales.** A single-source factor-1 link whose ends are at different locations becomes a mapped derived variable instead of an alias.
- **Kinds on inputs.** `input_variable(..., state_variable_type=)` is optional. When given, it must belong to the provider's family (extensive or intensive), otherwise the coupling raises. `couplability_problems(..., data_structure=)` reports kind conflicts and missing mappings.
- **Coarse ↔ coarse mappings** (e.g. Organ → Axis, Axis → Organ) in `MPGDataStructure._map`.

### Anatomy mode: multiscale graphs assembled in the MPG (step 2f)

- **`MPGDataStructure(g, from_scale=SubOrgan, nodes="Compartment", wiring=[...])`**:
  - nodes are the Compartments of the anatomies held below the SubOrgans, keyed by their own vid;
  - edges are every Connection (anatomy edges and junctions), keyed by their own vid;
  - `"SubOrgan"` is a coarse location (owner = the Compartment's MTG parent), so step 1's mappings and `owner()` work across scales;
  - `scale=Compartment` declarations read the Compartments' MTG properties.
- **`MPG.wire_junctions(from_scale, rules, children=None)`** creates junctions between the Compartments of linked vertices:
  - `match="all" | "nearest" | "equal"` on an ordering property, or a callable rule;
  - junctions get `is_junction = 1`.

  `populate_graph_custom_connections` delegates to it, with unchanged results. New helpers: `linked_parent`, `compartments_by_owner`, `junction_vids`, `remove_connections`.
- **Growth and differentiation** (`update_topology()` in anatomy mode): only the junctions of vertices that are new, re-linked, or whose anatomy (or their parent's) changed are rebuilt. Every other Connection keeps its vid and values. `on_grow="inherit"` takes the same-label Compartment upstream.
- **Derived variables with a target.** `derive(..., target=mask)` gives values on the mask's entities only, the others getting the default (e.g. a SubOrgan concentration broadcast to symplastic Compartments).
- **Not in anatomy mode yet:** traversal (the assembled graph has cycles) and the repartition after structural steps (`NotImplementedError`).

### Boundary sets (step 2e)

- **`boundary_set(select=, kind="robin" | "dirichlet" | "neumann", value=, weight=, field=)`**, declared in a graph-system class, is assembled by the framework:
  - Robin `+ w·(x − v)` in the field's residual, Dirichlet rows `x − v`, Neumann `− v` (an inflow);
  - `select` is a `{variable: values}` dict, a variable name (> 0) or a callable, and membership follows its variables and the topology;
  - `value` and `weight` are DataStructure variables, read at each solve, or constants.
- **Jacobian.** With a user `@graph_jacobian`, the framework adds the sets' Robin and Dirichlet terms.
- **Active subgraphs and anchors.** Sets follow active subgraphs. Dirichlet sets and positive-weight Robin sets anchor steady pieces in the well-posedness check.
- **Boundary ports set by hand** (`component._boundary_ports`) are deprecated (`DeprecationWarning`).
- **New use case UC5, leaf transpiration** (`test/graph_system_tests/test_uc5_leaf_transpiration.py`), checked against the direct linear solve.

### Graph systems on the active subgraph (step 2d)

- **`@graph_system(where="active")`** solves on the nodes of a mask and the edges between them. The equations see the subgraph's view through `self._graph_view`; inactive nodes keep their values, and dropped edges carry no flux (their integrated amounts are kept).
  - The subgraph is rebuilt only when the topology or the mask's values changed.
  - An empty mask skips the solve.
- **`transient=`** tells whether the balance has a time derivative. It defaults to `True` for explicit and implicit Euler and IVP, and `False` otherwise. On an active subgraph, a steady system raises when a connected piece has no Dirichlet anchor, naming its nodes.
- **Boundary ports set by hand** cannot be combined with `where=` (`NotImplementedError`); boundary sets come in step 2e.
- **Previous values across solves** are now kept on every node, so that they survive a change of active subgraph.

### Repartition and the active mask (step 2c)

- **Repartition.** `StructuralComponent.partition_weight` (a node variable or a callable) enables the repartition, after each MPG-style step, of the other components' node variables by `state_variable_type`:
  - massic concentrations: amount split by weight, or dilution when the weight changed;
  - extensive: split;
  - intensive: copied;
  - NonInertialExtensive: weighted copy;
  - descriptors: `on_grow`.

  The rules are those of rhizodep's `post_growth_updating`. New inactive entities copy concentrations and hold no amount, and they are split when they become active. The weight before each step is recorded by the framework, so no `previous_weight` bookkeeping is needed.
- **Named masks.** `DataStructure.define_mask(name, rule)`, `mask(name)`, `mask_version(name)`, `has_mask(name)`.
- **Masked steps.** `StructuralComponent.active` defines the `"active"` mask. Vectorised steps compute on its entities when it is defined; `@rate(where=None)` opts out, and `where="name"` chooses a mask.

### StructuralComponent (step 2b)

- **New shared base.** `StructuralComponent` and `FunctionalComponent` now share `DataStructureComponent`: the DataStructure binding, declaration resolution, registration and MTG sync.
- **`StructuralComponent`** edits the plant's structure through `self.mtg`, the DataStructure's MPG, with the MPG's own methods.
  - **MPG-style steps** (decorated steps without arguments, e.g. `@potential`, `@segmentation`) run with synchronisation around them:
    - before: the component's declared variables are written to the MPG;
    - after: `ds.update_topology()` runs if the MPG's vertex count or last vertex id changed, then the declared state variables are re-read from the MPG.
  - **Array-style steps** (with arguments) are vectorised on the DataStructure, like functional steps.
  - The growth rows (`potential`, `actual`, `segmentation`, `postsegmentation`) order the steps within the component.
- **Metadata precedence.** When several components declare a variable, the one that does not declare it as an input sets its default, `on_grow` and `state_variable_type`. A component reading it as an input only fills what is still unknown. Before, the first component to register it won.

### Traversal and entity identity on the DataStructure (step 2a)

- `MPGDataStructure` gains, in local indices derived from the Connections and cached per topology version:
  - `parents()`, `children()` (CSR), `roots()`, `tips()`;
  - `order("pre" | "post")`;
  - `owner(location)`.

  A graph with several parents per node, or with a cycle, raises.
- `index_of(ids, location="node")` (every DataStructure) inverts `entity_ids`, and unknown ids raise `KeyError`. Grids have `index_of` but no traversal (`NotImplementedError`).
- Tests no longer use the private `_idx_to_vid` / `_vid_to_idx`, except the tests of the legacy DataStructure.

### Variable declarations: `scale` / `location` / `mapping` (design note `datastructure_contract.md`, step 1a)

- `declare` and its wrappers (`state_variable`, `input_variable`, `parameter`) take three new keys:
  - `location`: where the DataStructure stores the variable;
  - `mapping` (with `weight`): how values go between the MTG `scale` and the `location` when they differ;
  - `weight`: the weight variable of `weighted_mean`.

  One interpreter, `coupling.declaration.resolve_declaration`, now serves registration and the solver snapshot.
- **Behaviour change (N1).** `scale=<a scale coarser than the graph's nodes>` without `location` is now **stored at that scale** (location `"Organ"`, …). Before, it was broadcast to the nodes. For the former behaviour, write `location="node", mapping="broadcast"`.
- Scale names are accepted as locations, and resolved against the graph built by the MPG traversal (N5): `"SubOrgan"` is `"node"` when the graph is built from SubOrgan.
- Without `mapping`, the mapping follows `state_variable_type` (D9):
  - `sum` up for extensive variables;
  - `mean` up and `broadcast` down for intensive variables;
  - `weighted_mean` up (a `weight` is required) and `broadcast` down for massic concentrations.

  Ambiguous cases raise `DeclarationError`.
- Edge mappings are renamed: `proximal` → `child`, `distal` → `parent` (N2). `edge_mapping=` and the old names stay accepted for one release, with a `DeprecationWarning`.
- Declarations that cannot be resolved raise `DeclarationError` when the component is constructed, naming the field. This covers:
  - an unknown location;
  - a mapping on a variable stored at its own scale;
  - an edge variable without a mapping;
  - an edge `mean` mapping on a state variable;
  - a scale finer than the nodes;
  - a non-numeric default.
- `location="scalar"` fields are now registered on graph DataStructures too.
- A graph equation receiving a variable stored at a coarse scale raises, and points to `location="node", mapping="broadcast"`.

### MTG write-back through the scale mapping (step 1b)

- **State variables are written to the MTG after every component call** (`Component.__call__`, N4). Before, they were written only at the end of a graph solve, so `@rate`-only components never reached the MTG.
- **Values are written at the vertices of the declared scale, through the inverse mapping** (`MPGDataStructure.write_mtg`):
  - stored at its scale → as is (an Organ-scale pool is written at Organ vertices, no longer at the node vids);
  - broadcast from a coarse scale → the (weighted) mean of its nodes;
  - averaged to a coarse scale → broadcast back to the nodes;
  - on edges → at the child or parent endpoint.

  `MPGDataStructure.read_mtg` is the matching reader, used at registration and for the parameter refresh before each graph solve. That refresh now covers parameters at coarse locations too.
- **New declaration errors**, for state variables that could not be written back:
  - summed to a coarser scale;
  - an edge state at a scale coarser than the nodes.

  Writing a `parent`-mapped edge state where several edges share a parent raises.
- The DataStructure scale operators accept `child` / `parent` next to `proximal` / `distal`.

### Derived variables resolved at read (step 1c)

- **Per-variable write counters.** `register`, `set` and topology changes bump them; `write_count(name)` reads them.
- **Recomputed when read.** A derived variable is recomputed at `get()` when one of its sources was written since its last computation, in dependency order and in place, so earlier views stay valid. Before, it was recomputed only by the receiver's `pull_available_inputs`, at the start of its call. `refresh()` still forces a recomputation, and `is_stale(name)` tells whether `get` would recompute.
- **Read-only (N3).** `set()`, or `set_node_property` / `set_edge_property`, on a derived variable raises `ValueError`.
- **Writes through views.** `ds.get(x)[...] = v` is not seen by the variables derived from `x`. Call `ds.mark_written(x)` after such a write. The Coupler's `push` now writes through `set`.
- **Observable change.** Reading a derived input after other components ran now gives its current value, not the value its receiver last used. The scene contract anchor (`test_ds_scene_contract.py`) is updated accordingly, and every computed value is unchanged.

### Output locations (step 1d)

- **Declared outputs** take their declared location.
- **Undeclared outputs give theirs:**
  - `@graph_output(name, location="node" | "edge")`;
  - `@rate(location=...)` and the other step decorators, with `locations={name: location}` for supplementary outputs. A scale name is resolved against the graph.
- **Total steps and 0-d values** stay scalar.
- **Without a location,** it is inferred from the shape only when exactly one of node / edge / cell matches, with a `DeprecationWarning`. When the shape is ambiguous (n == m) or unmatched, it raises.

  This replaces two rules:
  - the `size == n` guess for graph outputs;
  - "the location of the first input" for step outputs, which was wrong when that input was a scalar.
- **Filtered equations and boundary conditions** slice an argument by its location (node or edge), no longer by its length.
- A `@graph_output` whose location differs from that of the registered variable raises.

### Validation and failure modes (step 1e)

- **`DataStructure.validate(strict=False)`** (`MPGDataStructure`, `ArrayDataStructure`, through `validate_variables`) raises one `ValueError` listing every inconsistency:
  - an array whose shape is not its location's;
  - a dangling or cyclic alias;
  - a derived variable whose source or weight is missing or moved, or which is not stored at its declared location.

  `strict=True` also recomputes the up-to-date derived variables and detects writes made through views.
- **Where it runs:**
  - at the end of `CompositeModel.declare_data_and_couple_components`;
  - in `testing.couplability_problems` / `assert_component_couplable`, when given `data_structure=`. These also report declarations that do not resolve on it.
- **Missing variables raise instead of acting as zeros:**
  - a graph-system argument that is not registered raises `KeyError`, naming the component;
  - a missing filter variable raises (it used to select every entity);
  - integrated `{field}_amount` unknowns are registered explicitly at zero before the first solve.
- `@boundary_condition(location="edge")` raises `NotImplementedError`, since conditions were always applied on nodes; any other location raises `ValueError`.
- `MPGDataStructure(g)` infers `from_scale` from the populated graph.

### Conventions page (step 1f)

- **New `docs/conventions.md`:** graph incidence and order, grid axes, locations, `scale` / `location` / `mapping` with the default-mapping table, write-back, outputs, derived variables, `previous()`, `on_grow`, and failure modes.
- **Documentation test:** `test/data_api_tests/test_conventions_doc.py` checks its tables against the declaration resolver.
- **Updated for the new keys:** `downstream_migration.md` §2 and the checklist. The README points to both.

### Scene and coupling wrappers

- `CompositeModel.open_or_create_translator(translator_path)` now takes the **full path of the translator YAML file**. It no longer takes a directory to which `/coupling_translator.yaml` was appended.
- `play_Orchestra`:
  - New `sowing_depth` argument, which replaces the unused `max_depth`.
  - New `debug_runs` and `poll_interval` arguments.
  - The light model now receives `light_scenario` instead of `plant_scenarios[0]`. The meteo table is read from `light_scenario["meteo"]`, given as a csv path indexed by `t` or as a DataFrame. It replaces the hard-coded `inputs/meteo_Ljutovac2002.csv`.
  - The soil and light workers are now pinned to dedicated cores, like the plant workers, and one core is still left free.
  - On platforms without `cpu_affinity` support (macOS), pinning is skipped.
  - New `plant_model_frequency` argument: the probability of each plant model at each sowing position. It defaults to uniform, so it is implicit for a single model. Before, only `plant_models[0]` was ever used. A single `sowing_depth` value is shared by all plant models.
  - `clean_exit` is now False when a worker failed. Plant and soil workers exit with code 1 when their model raised.
- **Breaking, light model contract:** `light_model` is now built as `light_model(queues_light_to_plants=, queue_plants_to_light=, scene_xrange=, scene_yrange=, meteo=, **scenario)`. Its constructor must answer the plants' initialization messages, as the soil model does. Before this change, the light worker ran one step short of the plants: the first worker to finish stopped the scene, and every model ran `n_iterations - 1` steps.
- Soil and light workers no longer stop the scene when they finish normally. Only plants end a scene, and an environment worker sets `stop_event` only on failure. Before, the first environment worker to finish could make the other one skip its last step, so plants waited forever.
- The soil worker flushes its reply queues before `os._exit`. Its last replies could be lost, which made the scene hang under load.
- The plant shared buffers are zero-initialised (they used `np.empty`).
- New constants `scene_wrapper.HANDSHAKE_SHAPE` (the plant/soil buffer, still `(35, 20000)`) and `CPU_REGISTRY_FOLDER` (the `outputs/` folder is created when missing).
- `stand_initialization` picks plant models from cumulative frequencies (the pick was wrong with more than 2 models).
- The soil and plant workers run without a `logger_class`.
- `CompositeModel`:
  - A same-name link with a factor ≠ 1 is now applied across data structures (it was silently dropped) and rejected within one data structure.
  - Multi-source links of the soil keep all their sources.
  - `apply_input_tables` recomputes its variable selection when its targets change.
  - `documentation` / `inputs` work on any component.
  - `recursive_reload` is removed.

### DataStructures (coupling refactor prerequisites, devplan WD.P)

- **New variable API on `MPGDataStructure` and `ArrayDataStructure`:**
  - `get`, `set`, `register`, `location`, `has`, `alias`, `aliases`, `version`.
  - `set` and the legacy `set_node_property` / `set_edge_property` / `_set_field` now **write in place** and check the shape. They used to rebind the array and accepted any length.
  - Aliases are name-level: `get(alias) is get(source)`. Cycles and shadowing are rejected.
- **Breaking, incidence sign convention:** `incidence_matrix()` now uses the solver's `GraphView` convention, `B[parent, e] = +1`, `B[child, e] = -1`. It used to be the opposite of `to_graph_view().incidence`.
- **`GraphView.node_local_index`** no longer assumes sorted ids (every lookup was wrong on MPG views) and raises `KeyError` for unknown ids.
- **`update_topology` carries registered variables over to the new topology.** Nodes are matched by vid, edges by child vid. New entities take the declared default, or the parent's value with `on_grow="inherit"`. The variables used to be cleared.
- `declare`, `input_variable`, `state_variable` and `parameter` take `on_grow="default" | "inherit"`, and `FunctionalComponent` registers its fields with it.
- **Scale-aware MTG mapping:**
  - `_mtg_to_node_array` / `_mtg_to_edge_array` take `scale=`, which maps a coarser-scale property through each node's complex.
  - The fast path checks the keys, not only the size.
  - Partial coverage raises `ValueError`; it used to fall back silently to the default.
- `write_node_to_mtg` / `write_edge_to_mtg` create a missing property and raise on shape errors. Every exception used to be swallowed.
- **`ArrayDataStructure`:** named axes `("x", "y", "z")`, plus `cell_centers()`, `cell_volume()` and `locate(points, periodic=, clip=)`, which returns flat C-order cell indices.
- **`LabelsConfig`:**
  - Integers now live on per-instance copies, so every MPG has a complete translator. The class attributes are no longer mutated.
  - `Connection.Apoplastic` is `"ApoplasticEdge"`; it used to collide with `Compartment.Apoplastic`.
- **Links on DataStructures (WD.3):**
  - `derive(name, {source: factor} | sources + formula, location=, aggregation=, weight=)` and `refresh(name=None)` recompute derived variables in place. Dependencies are refreshed first, and derivation cycles are rejected.
  - `MPGDataStructure` gets new locations:
    - the coarser biological scales, named after `ScalesConfig` (`"Organ"`, `"Plant"`, …);
    - `"scalar"`, which holds plant-scale values.
  - Scale operators: node → coarse (`sum` / `mean` / `weighted_mean`), coarse → node (`broadcast`), node → edge (`proximal` / `distal` / `mean`), and to or from `"scalar"`. `entity_ids(location)` gives the entity order.
  - `ArrayDataStructure` gets a `"scalar"` location, with `cell` ↔ `scalar` aggregation and broadcast.
  - `update_topology` carries every location over growth.
- **Live reading (WD.2):** `FunctionalComponent` no longer copies its DataStructure into a props snapshot.
  - **Graph-system solves:**
    - read the variables from the DataStructure at each solve (per-solve copies);
    - write unknowns, integrated amounts and `@graph_output` results back in place, registering them on first write;
    - take the implicit-Euler first-tick previous fields from it;
    - write the MTG directly from the DataStructure arrays.
  - `self.props` is a **read-only** `DataStructurePropsView` for compatibility, kept for one release.
  - Choregrapher steps run on the DataStructure arrays and **receive whole arrays**. `@rate(vectorized=False)` (and the same option on every step decorator) opts in to one call per element for scalar logic.
  - New `self.previous(fn)`: an unknown's value at the start of the current solve, managed by the framework. The user-managed `_previous_fields` is deprecated.
- **Translators as objects (WD.0):**
  - New `openalea.metafspm.coupling.translator`, providing `Link`, `Translator` and `parse_factor`:
    - Python-first translators with live `ScalesConfig` references, `formula=` callables, and optional `scale` / `aggregation` / `weight` on each link;
    - `Translator.from_yaml`, `from_dict`, `from_module(path)`, `load`, `to_nested`, `inputs_outputs`.
  - The YAML short form is unchanged. A long form is available: `variable: {sources: {...}, scale: Organ, aggregation: sum}`.
  - String factors are parsed by a restricted arithmetic parser (numbers and `+ - * / **`). **`eval` is no longer used** in `CompositeModel`.
  - `open_or_create_translator` also accepts a `.py` translator module.
- **`CompositeModel` on DataStructures (WD.4):** when every component is DataStructure-backed, `couple_components` registers the translator links on the shared DataStructure:
  - identity: nothing;
  - alias: `ds.alias`, which replaces the receiver's own default;
  - factor, sum, formula or scale change: `ds.derive`, refreshed by the receiver's `pull_available_inputs` before its step. A same-name factor is rejected.

  Soil outputs are registered on the plant DataStructure and initialised to 0. `apply_input_tables` writes DataStructure variables. The props path is kept for legacy components until the doubles are retargeted (WD.6). `DataStructure.unregister(name)` is new.
- **Plant ↔ soil `Coupler` (WD.5a):** new `openalea.metafspm.coupling.coupler`.
  - `VoxelLocator` finds each segment's cell from its barycentre (x1..z2 plant variables, names configurable), with `flip_z` and periodic x and y.
  - `Coupler.update_map` / `zero_soil_inputs` / `push` / `pull` do an extensive scatter-add to the soil, with factors, and an intensive in-place gather to the plant. It gives the same sums as the reference soil model's `apply_to_voxel_fast`, with the `(y, z, x)` → `(x, y, z)` permutation.
  - `Coupler.from_translator` builds a Coupler from a Translator.
  - `DataStructure.topology_version` makes a stale map raise.
- **Plant ↔ soil transport (WD.5b):**
  - `coupler.Transport` sets the buffer layout from the translator: rows `_n_nodes`, `_node_id`, the segment coordinates, the plant variables read by the soil, and the soil states. This replaces the `vertex_index >= 1` convention.
  - It provides `write_plant` / `read_soil` for the plant side, and `plant_view` for the soil side: a `BufferPlantView` on which the same `Coupler` runs. Capacity overflow raises `OverflowError`.
  - `play_Orchestra(handshake_shape=...)` defaults to the legacy `(35, 20000)` (Q27).
  - `FunctionalComponent` also runs on `ArrayDataStructure`: fields declared with `scale="cell"` or `"scalar"` are registered automatically.
- **Soil component name and sub time step:**
  - `CompositeModel.soil_name` sets the soil component name, which is no longer hard-coded to `"SoilModel"`.
  - `FunctionalComponent` runs once per simulation step, or per its own `sub_time_step`. It used to register a sub time step of 1, so a 3600 s simulation step ran it 3600 times per call.
- **Export for loggers (WD.9):** `ds.export(names)`, `ds.to_dataframe(names, location, time)` and `ds.summarize(sums, means, scalars, where)` are available on `MPGDataStructure` and `ArrayDataStructure`.
  - `to_dataframe` is indexed by vid, edge child vid or voxel, plus `t`; cells also get x/y/z centre columns. `.to_xarray()` turns the result into a dataset.
  - `summarize` builds the plant-scale csv row.
- **`openalea.metafspm.testing` (WD.8):** `couplability_problems` / `assert_component_couplable(component_cls, translator, name=)`, for downstream test suites.
- `FunctionalComponent._graph_view` is rebuilt when the DataStructure topology changes, so components keep solving after growth.

### Solvers (backlog B5, B6)

- Edge (algebraic) unknowns are recovered by solving the edge rows with the node unknowns held fixed. `ExplicitEulerSolver` and `ScipyIVPSolver` used to run a Newton solve on the whole quasi-static system, which gave the wrong fluxes, or a singular matrix for transient problems. `DAESolver.solve` stores the recovered values in the accepted state.
- `ScipyIVPSolver` returns the full packed state (nodes and edges) for specs with edge unknowns. It used to return the node unknowns only.
- `ScipyIVPSolver` no longer fakes an error estimate from the midpoint state. `solve_ivp` controls the error within each step, and the outer loop accepts the step. The old estimate made the step size collapse.
- `NewtonSolver`, `ImplicitEulerSolver` and `ScipyRootSolver` start from the state they are given instead of the spec's initial guess.
- `make_solver(method, config)` accepts a dict of `SolverConfig` fields (it used to replace them silently with the defaults). Unknown keys and other types raise `TypeError`.

### Breaking: legacy props path removed (devplan Q28)

Components are coupled only through DataStructures; the props-based path of the former downstream models is gone. See `docs/design/downstream_migration.md`.

- **Components:**
  - `FunctionalComponent` requires a DataStructure with a variable store (`MPGDataStructure`, `ArrayDataStructure`).
  - `Component.link_self_to_mtg` and the props-based `pull_available_inputs` are removed.
  - `self.props` remains as a read-only view.
- **Functor and Choregrapher:**
  - Steps run only on DataStructures: the `dict`, `ArrayDict` and `ndarray` branches of the Functor are removed, and so are the Choregrapher's `"length"` type probe and `focus_elements` computation.
  - The ArrayDict-only numba specialisation branch (`use_njit`) is removed.
  - `add_time_and_data(instance, sub_time_step, data, compartment="graph")`.
- **Graph systems:** they read and write only DataStructures; the props fallbacks and the `_prop_location` heuristic are removed.
- **`CompositeModel`:**
  - It couples only DataStructure-backed components.
  - `couple_current_with_components_list` (soil-side props coupling) is removed; the scene uses `coupler.Transport` and `Coupler`. `soil_handshake_inputs`, `soil_handshake_prefix` and `plant_side_soil_inputs` are removed too.
  - `apply_input_tables` writes DataStructures only.
  - `translator_matrix_builder` skips fields without metadata; it used to crash on `FunctionalComponent`.
- **`play_Orchestra`:** `handshake_shape` is required, and `scene_wrapper.HANDSHAKE_SHAPE` is removed.
- `MPGDataStructure.from_legacy` copies the values by vertex. It used to copy them in the legacy sorted order into the MPG's post-order, which misaligned them, or a length-1 array from a legacy structure at another scale; that case now raises.

