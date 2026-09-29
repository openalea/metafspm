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

