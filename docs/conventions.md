# metafspm conventions

These are the conventions that every component, translator and DataStructure follows. The design is explained in `design/datastructure_contract.md`. This page is the reference to keep at hand, and `test/data_api_tests/test_conventions_doc.py` checks it against the code.

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

Prefer scale names for biological variables: they keep their meaning when the graph's nodes change (anatomies, DS8). Keep `node` / `edge` for solver-level variables, which follow the graph's entities whatever they are.

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

**Default mapping when none is given,** from `state_variable_type` (decision D9):

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

## Outputs

- A step or graph output that is a declared field takes its declared location.
- An undeclared output gives its location:
  - `@graph_output(name, location="node")`;
  - `@rate(location="edge")`, or `@rate(locations={"extra_output": "Organ"})` for supplementary outputs.
- Total steps and single values are `scalar`.
- Shape inference remains only when exactly one location matches, with a `DeprecationWarning`.

## Couplings and derived variables

- **Identity links:** components on one DataStructure share variables by name.
- **Aliases:** another name for the same array. Writing an alias writes its target.
- **Derived variables:** translator links with a factor, a sum, a formula or a scale change.
  - They are **recomputed when read**, if a source was written since their last computation.
  - They are **read-only**: write their sources.
- **Write through the DataStructure:** use `ds.set(name, values)`. Writing into a view (`ds.get(name)[...] = v`) is invisible to the variables derived from `name`; call `ds.mark_written(name)` after such a write. `ds.validate(strict=True)` detects forgotten ones.

## Time and growth

- **`previous(name)`:** inside a graph-system solve, the value of the unknown at the start of that solve.
- **`on_grow`:** the value of entities created by growth. `"default"` gives the declared default; `"inherit"` gives the nearest pre-existing ancestor's value. The repartition of amounts at segmentation comes with DS20.

## Anatomy mode

- **Construction.** `MPGDataStructure(g, from_scale=scales.SubOrgan, nodes="Compartment", wiring=rules)` solves on the anatomies.
  - Nodes are the Compartments created under each SubOrgan (`add_component_with_topo(compartment_anchor, suborgan)`).
  - Edges are the anatomy Connections plus the junctions that `wiring` creates between linked SubOrgans.
  - Both are keyed by their own vids.
- **Locations.** `"SubOrgan"` and coarser scales are coarse locations; `scale=scales.Compartment` declarations read the Compartments' properties.
- **Growth.** Only the junctions of changed SubOrgans are rewired. A new Compartment inherits from the same-label Compartment upstream.

## Boundary sets

```python
leaves = boundary_set(select=is_leaf, kind="robin", value="air_water_potential", weight="leaf_conductance")
```

- **Declared** in a graph-system class, and assembled by the framework in the field's residual:

  | kind | residual term |
  |---|---|
  | `robin` | `+ w·(x − v)` |
  | `dirichlet` | the row becomes `x − v` |
  | `neumann` | `− v` (`v` is an inflow) |

  With a user Jacobian, the framework adds the sets' terms.
- **`select`** is a `{variable: values}` dict, a variable name (selects where it is > 0), or a callable. Membership follows its variables and the topology.
- **Live values.** `value` and `weight` are read at each solve.
- **Parameters of grown entities.** A parameter read from the MTG gets its value for new entities at the refresh before the next solve. Until then, masks built on it use its `on_grow` value.
- **Hand-set boundary ports** (`_boundary_ports`) are deprecated.

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

- **Defining.** `ds.define_mask(name, rule)` takes a `{variable: condition}` rule (`">0"`, a value, a list of values) or a callable. `ds.mask(name)` is recomputed when its variables were written; `ds.mask_version(name)` changes with its values.
- **The active mask.** A structural component's `active = {...}` defines the mask `"active"`.
- **Vectorised steps** compute on the entities of the `"active"` mask when the DataStructure defines it: arguments at its location are restricted, and outputs are written on them only. `@rate(where=None)` computes everywhere; `where="name"` chooses another mask. MPG-style steps always see every entity.
- **Graph systems** solve on the whole graph unless `@graph_system(where="active")`. With it, they solve on the active nodes and the edges between them:
  - inactive nodes are frozen, and dropped edges carry no flux;
  - a steady system (`transient=False`, the default for Newton and root solvers) needs a Dirichlet anchor in every connected piece.

## Failure modes

These raise; none of them is silently ignored:
- a variable used by a graph system that is not registered;
- a missing filter variable;
- an ambiguous output location;
- an edge boundary condition (until boundary sets, DS6);
- a declaration that does not resolve;
- a write to a derived variable.

`ds.validate()` lists every inconsistency of a DataStructure at once.
