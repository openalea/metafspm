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

