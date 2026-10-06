# metafspm conventions

These are the conventions that every component, translator and DataStructure follows. The design is explained in `dev/design/datastructure_contract.md`. This page is the reference to keep at hand, and `test/data_structures/test_conventions_doc.py` checks it against the code.

## Graph

- **Nodes and edges are the entities of the graph built by the MPG traversal** (`MPG.populate_graph(from_scale)`): one node per vertex of `from_scale` (e.g. SubOrgan), one edge per parent–child link. Model declarations anchored on biological scales are resolved against that graph, never the reverse.
- **Incidence sign.** `B[parent, e] = +1` and `B[child, e] = -1`, so `(Bᵀ c)_e = c_parent − c_child`.
- **Local order.** Arrays follow the Compartment post-order produced by population. It is **not** sorted by vertex id. Use `ds.entity_ids(location)` to know which entity a position holds, and `ds.index_of(ids, location)` for the reverse. Never compute a position by hand.
- **Traversal, in local indices** (derived from the Connections, cached per topology):
  - `ds.parents()`: −1 at a root;
  - `ds.children()`: CSR `(indptr, indices)`;
  - `ds.roots()`, `ds.tips()`;
  - `ds.order("pre")`: parents first; `ds.order("post")`: children first;
  - `ds.owner("Organ")`: each node's entity at a coarse location.
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

A **scale name** is also a location:
- the name of the graph's node scale (e.g. `"SubOrgan"`) means `node`;
- `"Connection"` means `edge`;
- a coarser scale (e.g. `"Organ"`) is its own location, with one value per vertex of that scale.

Prefer scale names for biological variables: they keep their meaning when the graph's nodes change (anatomies). Keep `node` / `edge` for solver-level variables, which follow the graph's entities whatever they are.

## Declarations: `scale`, `location`, `mapping`

```python
pool:    float = state_variable(..., scale=scales.Organ)                    # stored at "Organ"
k_organ: float = parameter(..., scale=scales.Organ, location="node")        # Organ value, broadcast to the nodes
K_axial: float = parameter(..., scale=scales.SubOrgan, location="edge", mapping="mean")
flux:    float = state_variable(..., location="edge")                       # solver-only, no MTG property
```

- **`scale`:** the MTG scale of the property. Values are read from it at registration, and state variables are written back to it.
- **`location`:** where it is stored. It defaults to the location of `scale`.
- **`mapping`:** how values go between the two when they differ.

| Mapping | Direction |
|---|---|
| `broadcast` | down: each node gets its owner's value |
| `sum` | up: total over the nodes of each coarse entity |
| `mean` | up: average over the nodes of each coarse entity; on edges, the mean of both ends |
| `weighted_mean` | up: average weighted by `weight=` |
| `child` | to edges: the child's value (formerly `proximal`) |
| `parent` | to edges: the parent's value (formerly `distal`) |

**Default mapping when none is given,** from `state_variable_type`:

| `state_variable_type` | up | down |
|---|---|---|
| extensive, NonInertialExtensive | `sum` | error: give a mapping |
| intensive, NonInertialIntensive | `mean` | `broadcast` |
| massic_concentration | `weighted_mean` (give `weight=`) | `broadcast` |
| none | error | error |

Node → edge mappings have no default.

**Write-back.** State variables with a `scale` are written to the MTG after every component call, at the vertices of their scale, through the inverse mapping:
- a broadcast variable is written as the (weighted) mean of its nodes;
- an averaged variable is broadcast back to its nodes;
- an edge variable is written at its child or parent end.

Declarations that cannot be written back raise when the component is created: a state variable summed to a coarser scale, an edge state at a scale coarser than the nodes, or an edge state with mapping `mean`.

**Links between scales.** A translator link between two locations without `aggregation` uses the same table, applied to the provider's `state_variable_type`. A receiving input may declare its `state_variable_type`, which must be in the provider's family: extensive with extensive, intensive or massic with intensive or massic. Mappings between two coarse scales (e.g. Organ ↔ Axis) go through each finer entity's owner.

**Translator links:**
- `scale=` / `source_scale=` on a link only check the declared locations;
- `target="mask"` restricts a mapped link to the mask's entities, the others getting the receiver's default.

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
- Shape inference remains only when exactly one location matches, with a `DeprecationWarning`.

**MTG synchronisation.**
- **`mtg_sync = "lazy"`** (the default): the DataStructure is the reference, and the MTG a view kept up to date when read. A state variable changed since its last synchronisation is written to the MTG:
  - before an MPG-style step;
  - when the MTG is read through `ds.mtg` (or `component.mtg`);
  - by `ds.flush_mtg()`.

  An MPG-style step sees the variables its component declares (states, inputs, parameters); it must declare the properties it reads. Code holding the MTG object itself (e.g. a Logger given `g`) calls `ds.flush_mtg()` first.
- **Checkpoints:** `ds.checkpoint(path)` and `Class.restore(path)`. Derivation formulas and mask rules are module-level functions or picklable objects, not lambdas.
- **`mtg_sync = "after_call"`** writes the state variables after every call (the former default); **`"never"`** leaves the MTG untouched.
- MTG-backed parameters are re-read at the start of every call.
- In graph systems, parameters and inputs are read-only.

## Tree kernels

Models never write traversals: they call the DataStructure's kernels, which run on every plant at once.

| Kernel | For |
|---|---|
| `ds.chain_scan(length, reverse=True)` | distance from tip (rhizodep) |
| `ds.chain_scan(height, chain="phytomers", op="max", exclusive=True)` | the maximum over lower ranks (cnwgrass pseudostem) |
| `ds.accumulate(x)` | subtree totals |
| `ds.accumulate(dx, direction="down")` | positions from the base |
| `ds.path_window(budget, volume, values, where=apex)` | supply windows towards the base |
| `ds.chain_shift` / `ds.chain_write` | neighbours on a chain, and forward writes |
| `ds.path_compose` | frames |

Chains are `"axis"` ('<' successors) by default, or declared with `define_chain(name, group=, rank=)`.

## Couplings and derived variables

- **Identity links:** components on one DataStructure share variables by name.
- **Aliases:** another name for the same array. Writing an alias writes its target.
- **Derived variables:** translator links with a factor, a sum, a formula or a scale change.
  - They are **recomputed when read**, if a source was written since their last computation.
  - They are **read-only**: write their sources.
- **Write through the DataStructure:** use `ds.set(name, values)`. Writing into a view (`ds.get(name)[...] = v`) is invisible to the variables derived from `name`; call `ds.mark_written(name)` after such a write. `ds.validate(strict=True)` detects forgotten ones.

## Time and growth

- **`previous(name)`:** inside a graph-system solve, the value of the unknown at the start of the current (sub-)step. `previous(name, at="solve")` gives the start of the call's solve, and `previous(name, at="step")` the start of the component's call.
- **`self.dt`:** the length of the current (sub-)step. Time terms use `self.dt`, which equals `time_step` unless the graph system sub-steps (`integrate="substeps"` or `"adaptive"`).
- **Sub-stepping across components:** each component runs `simulation_time_step / time_step` times per simulation step (a whole number, checked at construction), its clock advancing by `time_step` each time. The solver's own sub-steps (`integrate=`) come within each of these.
- **`on_grow`:** the value of entities created by growth. `"default"` gives the declared default; `"inherit"` gives the nearest pre-existing ancestor's value. After structural steps, the other components' amounts are repartitioned by kind (`partition_weight`, `active`).

## Anatomy mode

- **Construction.** `MPGDataStructure(g, from_scale=scales.SubOrgan, nodes="Compartment", wiring=rules)` solves on the anatomies.
  - Nodes are the Compartments created under each SubOrgan (`add_component_with_topo(compartment_anchor, suborgan)`).
  - Edges are the anatomy Connections plus the junctions that `wiring` creates between linked SubOrgans.
  - Both are keyed by their own vids.
- **Locations.** `"SubOrgan"` and coarser scales are coarse locations; `scale=scales.Compartment` declarations read the Compartments' properties.
- **Growth.** Only the junctions of changed SubOrgans are rewired. In segment mode too, growth only adds or removes the Compartments and Connections of new or pruned segments (prune with `remove_tree`, which also removes the segment's Compartment). A new Compartment inherits from the same-label Compartment upstream.

## Boundary sets

```python
leaves = boundary_set(filters={"label": "LeafElement"}, kind="robin", value="air_water_potential",
                      weight="leaf_conductance")
```

- **Declared** in a graph-system class, and assembled by the framework in the field's residual:

  | kind | residual term |
  |---|---|
  | `robin` | `+ w·(x − v)` |
  | `dirichlet` | the row becomes `x − v` |
  | `neumann` | `− v` (`v` is an inflow) |

  With a user Jacobian, the framework adds the sets' terms.
- **`filters`** is a `{variable: condition}` dict, a mask name, or a callable, as for every decorator. Membership follows its variables and the topology.
- **Live values.** `value` and `weight` are read at each solve.
- **Parameters of grown entities.** A parameter read from the MTG gets its value for new entities at the refresh before the next solve. Until then, masks built on it use its `on_grow` value.
- **Hand-set boundary ports** (`_boundary_ports`) are deprecated.

## Parameters

- **Storage.** Numeric parameters declared without a scale are stored per plant (`"Plant"`; `"scalar"` on grids). They are uniform unless the plants' scenarios differ.
- **In steps and equations,** parameters are **arguments**, broadcast to the equation's nodes or edges: uniform values as zero-stride views, varied ones per entity. `self.<parameter>` raises there.
- **Outside them,** `self.k` reads the value (or the per-plant values when they differ), and `model.k = v` sets every plant.
- **MPG-style structural steps** use `self.parameter_values("k")`.

## Scenes

- A **plant model** is a population model: `Model(data_structure, time_step, **scenario)` with `initiators`, and component classes of its own (the scene translator identifies components by class name). Steps are scheduled per instance, and subclasses run their bases' steps.
- An **environment model** receives the populations and builds its DataStructures.
- **MPG-style steps** loop over `self.active_ids()`.

## Grids

- **Regular grids:** `ArrayDataStructure`.
- **Adaptive grids:** `AdaptiveGridDataStructure` (an octree over a base grid), with the same contract: location `"cell"`, faces as edges with `face_area` and `face_distance`, `cell_volume()` per cell, `locate`, `layer_mask(axis=0 | -1)`.
- **Refinement:** refine and coarsen between steps with a criterion of the grid; values are carried over conservatively.
- **Equations** written with `cell_volume()`, `face_area` and `face_distance` run on both.

## Random draws

- **Reproducible draws per entity:** `self.random(stream, distribution="uniform", ids=None, **parameters)` in a component, or `ds.random(...)` with an explicit `step`. A draw depends on (seed, stream, step, entity id) only, not on the visiting order.
- **Advancing:** each call of a stream is a new step. Step counters are kept by checkpoints. `random_seed` (a component attribute) chooses the seed.
- **MPG-style steps** that create vertices one after the other (a chain of segments, each from the one just created) are plain Python, and draw for the new vertices with `self.random(stream, ids=new_vids)`.

## Structural components

- A `StructuralComponent` edits the MPG through `self.mtg`. The MPG is the source of truth for structure.
- **MPG-style steps** (no arguments) are synchronised: declared variables are written to the MPG before them; after them, the topology is updated if vertices were added or removed, and the declared state variables are re-read.
- **Array-style steps** (with arguments) work on the DataStructure.
- The component that owns a variable (declares it other than as an input) sets its default, `on_grow` and `state_variable_type`.
- **Repartition.** A structural component with `partition_weight = "<node variable>"` shares the other components' node variables after each MPG-style step, by `state_variable_type`:
  - **massic concentrations:** an amount split by weight, or diluted when the weight grew;
  - **extensive:** split;
  - **intensive:** copied;
  - **NonInertialExtensive:** a weighted copy, the parent unchanged;
  - **descriptors:** `on_grow`.

  A new entity that is inactive (e.g. a zero-mass primordium) copies concentrations and holds no amount. It is split from its parent when it becomes active.

## Masks

- **Defining.** `ds.define_mask(name, rule)` takes a `{variable: condition}` rule (a comparison such as `">0"` or `"<=0.03"`, a value, a list of values, label names; a variable at a coarser scale is read at each entity's entity of that scale) or a callable. The decorators' `filters=` take the same rules, or a mask name. `ds.mask(name)` is recomputed when its variables were written; `ds.mask_version(name)` changes with its values.
- **The active mask.** A structural component's `active = {...}` defines the mask `"active"`.
- **Vectorised steps** compute on the entities of the `"active"` mask when the DataStructure defines it: arguments at its location are restricted, and outputs are written on them only. `@rate(include_inactive=True)` lifts it; `filters=` restricts a step further. MPG-style steps always see every entity.
- **Graph systems** solve on the whole graph unless `@graph_system(filters=...)`, e.g. `filters="active"`. With it, they solve on the selected nodes and the edges between them:
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

`ds.validate()` lists every inconsistency of a DataStructure at once.
