# metafspm conventions

These are the conventions that every component, translator and DataStructure follows. The design is explained in `dev/design/datastructure_contract.md`. This page is the reference to keep at hand, and `test/data_structures/test_conventions_doc.py` checks it against the code.

## Graph

- **Nodes and edges are the entities of the graph built by the MPG traversal** (`MPG.populate_graph(from_scale)`): one node per vertex of `from_scale` (e.g. SubOrgan), one edge per parent–child link. Model declarations anchored on biological scales are resolved against that graph, never the reverse.
- **Incidence sign.** `B[parent, e] = +1` and `B[child, e] = -1`, so `(Bᵀ c)_e = c_parent − c_child`.
- **Local order.** Arrays follow the order in which the population's Compartments were created (by the traversal of the MPG, then appended as the plants grow). It is **not** sorted by vertex id, nor a pre- or post-order: `data_structure.order("pre" | "post")` gives those. Use `data_structure.entity_ids(location)` to know which entity a position holds, and `data_structure.index_of(ids, location)` for the reverse. Never compute a position by hand.
- **Traversal, in local indices** (derived from the Connections, cached per topology):
  - `data_structure.parents()`: −1 at a root;
  - `data_structure.children()`: CSR `(indptr, indices)`;
  - `data_structure.roots()`, `data_structure.tips()`;
  - `data_structure.order("pre")`: parents first; `data_structure.order("post")`: children first;
  - `data_structure.owner("Organ")`: each node's entity at a coarse location.
- **Edge identity.** Edge `e` is identified by its child vertex (`entity_ids("edge")`).

## Grids

- Axes are `(x, y, z)`: `shape=(nx, ny, nz)`, and `cell_centers()` / `locate(points)` use that order.
- **Topology.**
  - Cells are graph nodes (flat C order). Faces between adjacent cells are edges, x faces first, then y, then z, oriented towards increasing coordinates: a positive face flux goes from the lower to the upper cell.
  - Periodic axes (`periodic=`) add wrap faces from the last cell to the first.
  - Outer faces are not edges, so there is no flux unless a boundary set says otherwise.
- **Geometric factor.** `face_area` and `face_distance` are edge variables. A face flux reads `K · face_area / face_distance · (Bᵀc)`, and a cell balance divides by `cell_volume()`.
- **Graph systems on grids.** The same decorators apply. Cell arrays are seen flat (C order) by the equations and written back on the cells. Boundary layers are boundary sets on `layer_mask(...)`.

## Locations

Where a DataStructure stores a variable, i.e. the shape the equations receive:

| Location | Meaning |
|---|---|
| `node` | one value per graph node |
| `edge` | one value per graph edge |
| `scalar` | one value, e.g. a plant-scale quantity |
| `cell` | one value per grid cell |

A **scale name** is also a location, in declarations (they resolve against the DataStructure the component is built on; DataStructure methods take the resolved locations):
- the name of the graph's node scale (e.g. `"SubOrgan"`) means `node`;
- `"Connection"` means `edge`;
- a coarser scale (e.g. `"Organ"`) is its own location, with one value per vertex of that scale.

Prefer scale names for biological variables: they keep their meaning when the graph's nodes change (anatomies). Keep `node` / `edge` for solver-level variables, which follow the graph's entities whatever they are.

## Declarations: `scale`, `location`, `aggregation`

```python
pool:    float = state_variable(..., scale=scales.Organ)                    # stored at "Organ"
k_organ: float = parameter(..., scale=scales.Organ, location="node")        # Organ value, broadcast to the nodes
K_axial: float = parameter(..., scale=scales.SubOrgan, location="edge", aggregation="mean")
flux:    float = state_variable(..., location="edge")                       # solver-only, no MTG property
```

- **`scale`:** the MTG scale of the property. Values are read from it at registration, and state variables are written back to it.
- **`location`:** where it is stored. It defaults to the location of `scale`.
- **`aggregation`:** how values go between the two when they differ (the links' word too).

| Mapping (aggregation) | Direction |
|---|---|
| `broadcast` | down: each node gets its owner's value |
| `sum` | up: total over the nodes of each coarse entity |
| `mean` | up: average over the nodes of each coarse entity; on edges, the mean of both ends |
| `weighted_mean` | up: average weighted by `weight=` |
| `max` | up: the largest (e.g. a plant's height from its elements' tops; not across DataStructures) |
| `min` | up: the smallest (not across DataStructures) |
| `split` | down: each node gets its share of its owner's amount, in proportion to `weight=` (the total conserved) |
| `child` | to edges: the child's value |
| `parent` | to edges: the parent's value |

**Default aggregation when none is given,** from the variable's `kind`:

| `kind` | up | down |
|---|---|---|
| extensive, non_inertial_extensive | `sum` | `split` by `weight=` (without a weight: error) |
| intensive, non_inertial_intensive | `weighted_mean` by `weight=`, its capacity (without a weight: error; `aggregation="mean"` for an unweighted mean) | `broadcast` |
| massic_concentration | `weighted_mean` by `weight=`, a mass (without a weight: error) | `broadcast` |
| none | error | error |

An intensive variable is averaged with the weight of its capacity: a volume, a mass (a massic concentration), an area
(a flux density, e.g. a transpiration rate per leaf area), or a mask (e.g. the liquid nodes only). The unweighted mean
of a section's Compartments would be dominated by its walls and junctions, which outnumber its cells. An extensive
amount goes down by a split, which conserves it; copying it would multiply it.

**Coverage.** Nodes without an owner at the target scale (walls and junctions at the Cell scale) are left out of the
means; a sum that would drop their non-zero amounts raises.

Node → edge mappings have no default. Edges map up like their child-side nodes: an edge is counted at the owner of
its child node (an axial junction at its child segment), with the defaults above.

**One storage.** A variable held at its own scale (its location's entities are the MTG vertices of its `scale`, with no mapping) is not copied to the MTG: the MTG's property is a view of the DataStructure's column (`DataStructureColumn`), read with its current values at any time. Values written through the MTG (an MPG-style step's, a vertex the DataStructure does not hold yet) are kept apart until the DataStructure reads them back. New entities have no MTG entry until the next synchronisation, as when the MTG held copies.

**Write-back.** The other state variables with a `scale` are written to the MTG at the vertices of their scale, through the inverse mapping, when the MTG is read (lazily: `data_structure.mtg`, an MPG-style step, `data_structure.flush_mtg()`; `mtg_sync = "after_call"` writes them after every component call, `"never"` not at all):
- a broadcast variable is written as the (weighted) mean of its nodes;
- an averaged variable is broadcast back to its nodes;
- an edge variable is written at its child or parent end.

Declarations that cannot be written back raise when the component is created: a state variable summed to a coarser scale, an edge state at a scale coarser than the nodes, or an edge state with mapping `mean`.

**Links between scales.** A translator link between two locations without `aggregation` uses the same table, applied to the provider's `kind`. A receiving input may declare its `kind`, which must be in the provider's family: extensive with extensive, intensive or massic with intensive or massic. Mappings between two coarse scales (e.g. Organ ↔ Axis) go through each finer entity's owner.

**Translator links:**
- `scale=` / `source_scale=` on a link only check the declared locations;
- `target="mask"` restricts a mapped link to the mask's entities (`data_structure.derive(where="mask")` underneath), the others getting the receiver's default.

**Types.**
- Variables are floats by default.
- `dtype="int"` keeps labels, types and indices as integers.
- `dtype="object"` stores a list or record per entity, kept out of graph systems, derivations and transport.
- **Label names** in masks and filters are resolved through `LabelsConfig`: `"RootSegment"` or `"SymplasticNode"` (label values, unique), or `"Symplastic"` within the variable's own scale group.

## Outputs

- A step or graph output that is a declared field takes its declared location.
- An undeclared output gives its location:
  - `@graph_output(name, location="node")`;
  - `@rate(location="edge")`, or `@rate(locations={"extra_output": "Organ"})` for supplementary outputs.
- Total steps and single values are `scalar`.
- Any other undeclared output raises a `ValueError` naming the component and the output (no shape inference).

**MTG synchronisation.**
- **`mtg_sync = "lazy"`** (the default): the DataStructure is the reference, and the MTG a view of it. A state variable held at its own scale is read from its column (one storage). A mapped one changed since its last synchronisation is written to the MTG:
  - before an MPG-style step;
  - when the MTG is read through `data_structure.mtg` (or `component.mtg`);
  - by `data_structure.flush_mtg()`.

  An MPG-style step sees the variables its component declares (states, inputs, parameters); it must declare the properties it reads. Code holding the MTG object itself (e.g. a Logger given `g`) calls `data_structure.flush_mtg()` first for the mapped ones.
- **MTG properties are columns** (`mtg.properties()`: typed columns, an object column for any value), read and written in bulk:
  - `mtg.values_at(name, vids)`: (the sorted *vids* having a value, their values);
  - `mtg.set_values(name, vids, values)`: existing and new vertices, a value each or one for all;
  - `mtg.column(name)`: (values, present), two arrays over the vid index, to index with numpy (on an MPG with instances: each template's values gathered at its instances, the explicit entries over them, no vid located);
  - `mtg.compartments_csr(from_scale)`: the Compartments of each owner as one CSR (`owners`, `indptr`, `compartments`; `has`, `compartments_of`), kept until the next structural edit; `mtg.compartments_by_owner` is its dict, for code reading it owner by owner;
  - `mtg.view(name)`: a property read vertex by vertex (its column, with the instanced vertices' template values);
  - `mtg.property(name)` is openalea.mtg's read API: a read-only mapping built on demand (slow; a write raises).
- **Checkpoints:** `data_structure.checkpoint(path)` and `Class.restore(path)`. Derivation formulas and mask rules are module-level functions or picklable objects, not lambdas.
- **No process-wide state:** what a run builds belongs to its DataStructures or its components, never to a module or a class: objects shared by content in `data_structure.objects(name, saved=)` (saved with the DataStructure, freed with it), caches in `component.caches()` (`cached()`, `keep(saved=False)`: dropped when a checkpoint is loaded), the simulation step in `data_structure.simulation_time_step`. A module-level memo is only for constants (a preset's defaults).
- **`mtg_sync = "after_call"`** writes the state variables after every call (the former default); **`"never"`** leaves the MTG untouched.
- MTG-backed parameters are re-read at the start of every call.
- In graph systems, parameters and inputs are read-only.

## Tree kernels

Models never write traversals: they call the DataStructure's kernels, which run on every plant at once.

| Kernel | For |
|---|---|
| `data_structure.chain_scan(length, reverse=True)` | distance from tip (rhizodep) |
| `data_structure.chain_scan(length, reverse=True, cut=apex)` | the same, each chain cut after its apices (ArchiSimple's tips and nodules) |
| `data_structure.preorder(branches_first=True, defer=pairs)` | a depth-first visit in the MPG's children order, branches first, some vertices visited after others (Adel's turtle) |
| `data_structure.local_gather(owner, owner_key, tables)` | per entity, its owner's table at its local row (MECHA's per-node geometry of each section's template) |
| `self.graph_view.laplacian(w)` | B diag(w) Bᵀ in a graph system's Jacobian, into a pattern kept per view (scipy's product, bit for bit) |
| `data_structure.chain_scan(height, chain="phytomers", op="max", exclusive=True)` | the maximum over lower ranks (cnwgrass pseudostem) |
| `data_structure.accumulate(x)` | subtree totals |
| `data_structure.accumulate(dx, direction="down")` | positions from the base |
| `data_structure.path_window(budget, volume, values, where=apex)` | supply windows towards the base |
| `data_structure.chain_shift` / `data_structure.chain_write` | neighbours on a chain, and forward writes |
| `data_structure.path_compose` | frames |
| `data_structure.order("pre")`, `data_structure.levels()`, `data_structure.fold(update, values)` | traversal orders, depths and level-by-level folds (`kernels.depth_first_orders`, `tree_levels`) |

Chains are `"axis"` ('<' successors) by default, or declared with `define_chain(name, group=, rank=)`.

## Performance rules

For metafspm and every port: no Python loop over groups, nodes or entities in a step. A loop over owners or vertices
in a functor is one of the platform's primitives (the tree kernels above, `data_structure.local_gather`, `data_structure.preorder`,
`graph_view.laplacian`): the ports declare what they compute, metafspm owns the kernels.

1. **numpy first.** Per-group quantities by `np.unique(owner, return_index=True)`, `np.bincount`/`np.add.at`,
   `searchsorted`; work batched by archetype or template (all groups of an archetype at once: groups × members arrays), not
   group by group (`Reduction.assignment`, `kept_outputs`; MECHA's condensed volumes and kr).
2. **Topology-derived arrays cached per topology version** and read-only (`setflags(write=False)`): `entity_ids` of
   every location, `owner`, the index maps, `Reduction.group_table`, the sparse pattern of a reduction's terms
   (only its values filled at each solve). Copy one before writing into it.
   After a structural edit they are **patched, not rebuilt** (O-14): the MPG logs what each structural change
   removed and added (`MPG.structure_epoch`, `MPG.structure_delta(since)`, from its topology arrays), and the
   structure-derived arrays (`compartment_owners`, `owned_compartments`, `connection_vids`, `structural_links`,
   the nodes' owners at each coarse scale) drop the removed ids and append the added ones (larger ids: arrays
   sorted by id stay sorted). A port edits the MPG as before and calls `data_structure.update_topology()`, nothing else; a cache
   of its own derived from the structure can use `MPG._derived(name, build, patch)`. A change no delta describes (a
   kept vertex moved, ids given again) rebuilds. `METAFSPM_CHECK_TOPOLOGY=1` checks every patch against a build;
   `METAFSPM_CHECK_ARCHETYPES=1` checks every archetype of a reduction's batches against its evaluation alone.
   `update_topology` carries a location's float and int variables over as one block per dtype, each a row taken
   from its old array through the location's match in one gather, without copying the old arrays first.
   Read edges as arrays: `data_structure.edge_endpoints()` ((edges, 2), read-only), not `data_structure.edges()` (Python tuples).
   Read every vertex's linked parent at a scale at once, `mtg.linked_parents(scale)` (a dict, cached per structure),
   not `mtg.linked_parent(v, scale)` vertex by vertex; `mtg.components_at_scale` lists each scale in one loop, in
   openalea.mtg's order (an order a port's creation of vertices may depend on). A traversal whose order does not
   matter reads the topology arrays or the scale column instead (`mtg._vertices_at_scale(scale)`).
   Read the same vids at many variables of a location in bulk, through the location's shared arrays: a view's
   `at()` keeps the positions of read-only vids in the location's sorted ids for the next variable (the states an
   MPG-style step leaves are read back so, with the instanced vertices known before the step found once per
   location), `contains(vids)` tells which have an entry without reading them, and an object variable is read with
   one bulk read, never a lookup per vertex.
3. **numba where a loop stays sequential or a scatter is irregular** (tree scans, z-buffers, form factors, a turtle,
   a reduction's dense blocks into the scene's sparse matrix): numba is a hard dependency, no fallback. Every kernel
   is `@njit(cache=True)` in its package's `kernels.py` (one module to warm up, review and port), `nogil=True` when a
   step runs it on the whole scene, `parallel=True` only for independent iterations and large work, called within
   `Compute.parallel(name, work, section)` (numba's threads: the Compute's budget, at most one per chunk and the
   section's ceiling; its time in run.json). **Determinism:** a parallel kernel loops over chunks made from the work only, never the
   thread count (`kernels.work_chunks`), sums in a fixed order within a chunk and combines chunks in chunk order, so
   that its result is the same for any thread count. A kernel replacing Python is checked bit for bit against it at
   one chunk before the Python goes (no dual path).
4. **Cold start.** `python -m openalea.metafspm warm_up` (`compute.warm_up`) runs every kernel on tiny inputs:
   metafspm's (`kernels.warm_up`) and each port's `kernels.warm_up`, registered as an entry point of group
   `openalea.metafspm.warm_up` in its `pyproject.toml`; at install and in CI before the suites. `NUMBA_CACHE_DIR`
   for a read-only install. No ahead-of-time compilation.

## Couplings and derived variables

- **Identity links:** components on one DataStructure share variables by name.
- **Aliases:** another name for the same array. Writing an alias writes its target.
- **Derived variables:** translator links with a factor, a sum, a formula or a scale change.
  - They are **recomputed when read**, if a source was written since their last computation.
  - They are **read-only**: write their sources.
- **Write through the DataStructure:** use `data_structure.set(name, values)`. Writing into a view (`data_structure.get(name)[...] = v`) is invisible to the variables derived from `name`; call `data_structure.mark_written(name)` after such a write. `data_structure.validate(strict=True)` detects forgotten ones.

## Time and growth

- **`previous(name)`:** inside a graph-system solve, the value of the unknown at the start of the current (sub-)step. `previous(name, at="solve")` gives the start of the call's solve, and `previous(name, at="step")` the start of the component's call.
- **`self.step_duration`:** the length of the current (sub-)step. Time terms use `self.step_duration`, which equals `time_step` unless the graph system sub-steps (`integrate="substeps"` or `"adaptive"`).
- **Sub-stepping across components:** each component runs `simulation_time_step / time_step` times per simulation step (its DataStructure's `simulation_time_step`, the Scene's time step: the Choregrapher holds no run state; a whole number, checked at construction and when the Scene is built), its clock advancing by `time_step` each time. The solver's own sub-steps (`integrate=`) come within each of these.
- **`on_grow`:** the value of entities created by growth. `"default"` gives the declared default; `"inherit"` gives the nearest pre-existing ancestor's value. After structural steps, the other components' amounts are repartitioned by kind (`partition_weight`, `active`).

## Anatomy mode

- **Construction.** `MPGDataStructure(mtg, from_scale=scales.SubOrgan, nodes="Compartment", wiring=rules)` solves on the anatomies.
  - Nodes are the Compartments of each SubOrgan's anatomy, created in either of two ways:
    - under a Cell, as true MTG components: SubOrgan → Layer → Cell → Compartment, e.g. a cell's protoplast;
    - anchored to the SubOrgan (`add_component_with_topo(compartment_anchor, suborgan)`), e.g. a wall or wall junction shared by several cells, or every Compartment when there are no Layers or Cells.
  - A Compartment's owner is its Cell, or the SubOrgan it is anchored to. Its entity at a coarser scale is the one above its owner.
  - Edges are the anatomy Connections plus the junctions that `wiring` creates between linked SubOrgans.
  - **Wiring rules** are declared dicts, each evaluated once over every (linked parent, child) pair being wired, never per pair (`MPG.wire_junctions`): the platform's `{"node_label", "edge_label", "ordering", "match"}`, or `{"pairs": f, "edge_label": E, "properties": p}` with `f(mtg, section_pairs) -> (pair, tails, heads)` and `p(mtg, section_pairs, pair, tails, heads) -> {name: values}` (`wiring.SectionPairs`: the pairs and each side's Compartments as one CSR). Junctions are created pair by pair, then rule by rule, in the order each rule gives them.
  - Both are keyed by their own vids.
  - `add_components_bulk` creates many components in one call, taking one complex, or one per component (e.g. the protoplasts of many Cells).
- **Locations.** `"SubOrgan"` and coarser scales are coarse locations, and so are `"Layer"` and `"Cell"` when they own Compartments. `scale=scales.Compartment` declarations read the Compartments' properties, and `scale=scales.Cell` declarations the Cells'.
- **Inheritance at growth** (`on_grow="inherit"`): a new Compartment takes the value of the first same-label Compartment upstream, unless `data_structure.inheritance` gives another rule, a callable `(data_structure, new_vids, is_known) -> source vids` (−1: the default value).
- **Nodes without a Layer or a Cell** (walls, junctions) have owner −1 there (`data_structure.owner("Cell")`). Sums and means over the scale skip them, and broadcasts from it (parameters, masks) give them the variable's default.
- **Growth.** Only the junctions of changed SubOrgans are rewired. In segment mode too, growth only adds or removes the Compartments and Connections of new or pruned segments (prune with `remove_tree`, which also removes the segment's Compartment). A new Compartment inherits from the same-label Compartment upstream.

## Boundary sets

```python
leaves = boundary_set(where={"label": "LeafElement"}, condition="robin", value="air_water_potential",
                      weight="leaf_conductance")
```

- **Declared** in a graph-system class, and assembled by the framework in the field's residual:

  | condition | residual term |
  |---|---|
  | `robin` | `+ w·(x − v)` |
  | `dirichlet` | the row becomes `x − v` |
  | `neumann` | `− v` (`v` is an inflow) |

  With a user Jacobian, the framework adds the sets' terms.
- **`where`** is a `{variable: condition}` dict, a mask name, or a callable, as for every decorator. Membership follows its variables and the topology.
- **Live values.** `value` and `weight` are read at each solve.
- **Parameters of grown entities.** A parameter read from the MTG gets its value for new entities at the refresh before the next solve. Until then, masks built on it use its `on_grow` value.

## Stopping criteria

A Newton solve (`graph_system(solver="newton")`) has converged when either holds:

- **the residual**: `||R||∞ < absolute_tolerance + relative_tolerance · ||R₀||∞` (`R₀` the first iterate's residual;
  `absolute_tolerance` 1e-10 and `relative_tolerance` 0 by default);
- **the step**: `||δx||∞ ≤ step_tolerance · max(1, ||x||∞)` (`step_tolerance` 1e-10 by default, 0 to disable): the
  step is below the unknowns' round-off.

The residual of a converged iterate cannot fall below its round-off floor, about `ε ||J||∞ ||x||∞` (ε = 2.2e-16): a
system of large unknowns (potentials in hPa, values to 1e4) and large coefficients has a floor above an
`absolute_tolerance` of 1e-8, which the residual test alone never meets. The step test stops it at its second iteration when
it is linear. A solve that converges by neither within `max_iterations` raises `SolverConvergenceError`, its message
the residuals' and the steps' history and the floor estimate, so a tolerance set below the floor shows.

## Parameters

- **Storage.** Numeric parameters declared without a scale are stored per plant (`"Plant"`; `"scalar"` on grids). They are uniform unless the plants' scenarios differ.
- **In steps and equations,** parameters are **arguments**, broadcast to the equation's nodes or edges: uniform values as zero-stride views, varied ones per entity. `self.<parameter>` raises there.
- **Outside them,** `self.k` reads the value (or the per-plant values when they differ), and `model.k = v` sets every plant.
- **MPG-style structural steps** use `self.parameter_values("k")`; several parameters at once, `self.parameter_table(names, location)` ({name: values per entity}: each entity's owner found once per topology, one gather per parameter).

## Scenes

- A **plant model** is a population model: `Model(data_structure, time_step, <its own arguments>)` with `initiators`, and component classes of its own (the scene translator identifies components by class name). Steps are scheduled per instance, and subclasses run their bases' steps.
- An **environment model** receives the populations and builds its DataStructures.
- **Scenarios** are applied to the components by the Scene, not forwarded by the model. A model names in `__init__` only the scenario keys that are its own arguments; initiators read theirs from `parameters`.
- **MPG-style steps** loop over `self.active_ids()`.

## Grids

- **Regular grids:** `ArrayDataStructure`.
- **Adaptive grids:** `AdaptiveGridDataStructure` (an octree over a base grid), with the same contract: location `"cell"`, faces as edges with `face_area` and `face_distance`, `cell_volume()` and `cell_sizes()` per cell, `locate`, `layer_mask(axis=0 | -1)`. Mappings to plants follow its refinements. Between cells of different sizes, write edge laws with the view's `edge_difference` (`k * (self.graph_view.edge_difference @ u)`), consistent there; `incidence.T @ u` is the two-point difference, exact only between equal cells.
- **Refinement:** refine and coarsen between steps with a criterion of the grid; values are carried over conservatively.
- **Equations** written with `cell_volume()`, `face_area` and `face_distance` run on both.

## Random draws

- **Reproducible draws per entity:** `self.random(stream, distribution="uniform", ids=None, **parameters)` in a component, or `data_structure.random(...)` with an explicit `step`. A draw depends on (seed, stream, step, entity id) only, not on the visiting order.
- **Advancing:** each call of a stream is a new step. Step counters are kept by checkpoints. `random_seed` (a component attribute) chooses the seed.
- **MPG-style steps** that create vertices one after the other (a chain of segments, each from the one just created) are plain Python, and draw for the new vertices with `self.random(stream, ids=new_vids)`.

## Structural components

- A `StructuralComponent` edits the MPG through `self.mtg`. The MPG is the source of truth for structure.
- **MPG-style steps** (no arguments) are synchronised: declared variables are written to the MPG before them; after them, the topology is updated if vertices were added or removed, and the declared state variables are re-read.
- **Array-style steps** (with arguments) work on the DataStructure.
- The component that owns a variable (declares it other than as an input) sets its default, `on_grow` and `kind`.
- **Repartition.** A structural component with `partition_weight = "<node variable>"` shares the other components' node variables after each MPG-style step, by `kind`:
  - **massic concentrations:** an amount split by weight, or diluted when the weight grew;
  - **extensive:** split;
  - **intensive:** copied;
  - **non_inertial_extensive:** a weighted copy, the parent unchanged;
  - **descriptors:** `on_grow`.

  A new entity that is inactive (e.g. a zero-mass primordium) copies concentrations and holds no amount. It is split from its parent when it becomes active.

## Masks

- **Defining.** `data_structure.define_mask(name, rule)` takes a `{variable: condition}` rule (a comparison such as `">0"` or `"<=0.03"`, a value, a list of values, label names; a variable at a coarser scale is read at each entity's entity of that scale) or a callable; `data_structure.define_mask(name, entities="GranapAnatomy.soil_nodes")` a named entity set.
- **Named entity sets.** A component publishes the sets assemblies select, `entity_sets = {"soil_nodes": {"kind": 4.}}` (a class attribute, `{variable: condition}` rules); assemblies name them, `"Class.set"`: `define_mask(entities=)`, a `Reduction`'s `where=`, `ports=`, `keep=` (evaluated on a section template's properties too). A port's internal codes (a node kind, a label used as a code) never appear in an assembly, its tests or its examples. The decorators' `where=` take the same rules, or a mask name. `data_structure.mask(name)` is recomputed when its variables were written; `data_structure.mask_version(name)` changes with its values.
- **The active mask.** A structural component's `active = {...}` defines the mask `"active"`.
- **Vectorised steps** compute on the entities of the `"active"` mask when the DataStructure defines it: arguments at its location are restricted, and outputs are written on them only. `@rate(include_inactive=True)` lifts it; `where=` restricts a step further. MPG-style steps always see every entity.
- **Graph systems** solve on the whole graph unless `@graph_system(where=...)`, e.g. `where="active"`. With it, they solve on the selected nodes and the edges between them:
  - inactive nodes are frozen, and dropped edges carry no flux;
  - a steady system (`transient=False`, the default for Newton and root solvers) needs a Dirichlet anchor in every connected piece.

## Failure modes

These raise; none of them is silently ignored:
- a variable used by a graph system that is not registered;
- a missing filter variable;
- an ambiguous output location;
- an edge boundary condition (use a boundary set on the nodes);
- a declaration that does not resolve;
- a write to a derived variable.

`data_structure.validate()` lists every inconsistency of a DataStructure at once.
