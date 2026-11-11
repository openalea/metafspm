# Migrating downstream packages to DataStructure coupling

This guide applies to rootbridges, rootcynaps, cnwgrass, rhizosoil and grassbridges.
The design is in `coupling_through_datastructures.md`.

Every step below has a **working, tested reference** in `test/helpers/doubles_ds.py`, whose contract tests reproduce the numbers of the props-based coupling:

| Downstream role | Reference double |
|---|---|
| plant components (RootCNUnified, RootWaterModel, …) | `PlantCarbon`, `PlantNitrogen` |
| plant composite (GrassBRIDGES) | `DSFakePlant` |
| soil component (rhizosoil `SoilModel`) | `GridSoil` |
| soil composite (RhizoSoil) | `DSFakeSoil` |
| light model (LightModel) | `doubles.FakeLight` |

**The legacy props path has been removed from metafspm.** Downstream packages keep working against their pinned metafspm, the `publish_WB` branch, until they are migrated to this API.

## 1. Imports

Replace the flat modules, following this table:

| Old | New |
|---|---|
| `metafspm.utils` | `data_structure.arraydict` |
| `composite_wrapper` | `coupling.composite_wrapper` |
| `component_factory` | `coupling.choregrapher`, `solve.decorator`, `solve.functor` |
| `component.Model` | `coupling.component.FunctionalComponent` |

## 2. Components → `FunctionalComponent`

**Before**, with the MTG props:

```python
@dataclass
class RootCNUnified(Model):
    hexose: float = declare(default=1e-3, ..., role="state_variable", ...)
    def __init__(self, g, time_step, **scenario):
        self.props = g.properties(); self.vertices = ...
        self.apply_scenario(**scenario); self.link_self_to_mtg()     # apply_scenario is removed: the Scene sets them
        self.choregrapher.add_time_and_data(instance=self, sub_time_step=time_step, data=self.props)
```

**After**, reading the DataStructure live:

```python
@dataclass
class RootCNUnified(FunctionalComponent):
    hexose: float = state_variable(..., default=1e-3, scale=scales.SubOrgan, on_grow="inherit")
    struct_mass: float = input_variable(..., scale=scales.SubOrgan)
    K_axial: float = parameter(..., scale=scales.SubOrgan, location="edge", aggregation="mean")
    total_cytokinins: float = state_variable(..., location="scalar")       # plant-scale value (was vertex 1)

    @rate
    def _hexose_exudation(self, hexose, struct_mass):        # receives whole arrays
        return self.k * hexose * struct_mass

    @rate(vectorized=False)                                  # opt-in: one call per element, for scalar logic
    def _something_with_ifs(self, hexose): ...

model = RootCNUnified(data_structure=MPGDataStructure(g, from_scale=g.scales.SubOrgan))
```

The rules:
- **Every coupled or solved variable declares where it lives** (`docs/conventions.md`), and is then registered on the DataStructure automatically:
  - `scale=`: its MTG scale (e.g. `scales.SubOrgan`), read at registration and written back (lazily, when the MTG is read);
  - `location=`: where it is stored, defaulting to the location of its scale. Use `"edge"` for edge variables, `"node"` / `"edge"` / `"scalar"` / `"cell"` for solver-only variables, or a scale name;
  - `aggregation=`: when the scale and the location differ. `broadcast` and `split` go down; `sum` / `mean` / `weighted_mean` go up; `child` / `parent` / `mean` go to edges (the former `edge_mapping="proximal"` becomes `location="edge", aggregation="child"`). Without it, the aggregation follows the variable's `kind=` (the former `state_variable_type=`) where unambiguous.
  - `default=`: the initial value of every declaration (the former `initialize=` of state variables and inputs).
  - **A variable at a scale coarser than the nodes** (e.g. `scales.Organ`) is stored at that scale. Add `location="node"` (broadcast) if the equations need one value per node.
- **Parameters are arguments**. A numeric parameter is stored per plant, and steps and equations take it as an argument: `def _hexose_exudation(self, hexose, struct_mass, exudation_rate)`, not `self.exudation_rate`. Reading `self.exudation_rate` inside a step raises with that hint. Outside steps (scenario setup), `model.exudation_rate = …` sets every plant.
- **Step functions receive arrays**. Make the bodies numpy-compatible, e.g. with `np.where` instead of `if`, or mark the step `vectorized=False`.
- **Previous state:** inside graph-system equations, use `self.previous("concentration")` instead of a user-managed `_previous_fields`.
- **Growth:** declare `on_grow="inherit"` for variables that new segments should take from their parent. The default is the declared value. The growth model may still overwrite new entities, e.g. from parent concentrations or split extensive quantities.
- **Growth models** become `StructuralComponent`s on the plant DataStructure (the reference is `test/helpers/growth.py`):
  - their steps without arguments edit `self.mtg` as before (`add_child`, property writes);
  - the framework writes the declared variables to the MPG before each such step, and calls `ds.update_topology()` after it when vertices were added or removed;
  - declared state variables (length, radius, `struct_mass`, …) are re-read from the MPG after each such step;
  - a `post_growth_updating` pass becomes a `@postsegmentation` step; the other components' variables are repartitioned after structural steps by kind (`partition_weight`, `active`).

  Registered variables are carried over, and FunctionalComponents rebuild their graph views.
- **Segment geometry:** the growth model registers `x1, x2, y1, y2, z1, z2` as node variables. The soil coupling needs them.
- **`self.props` is gone.** Replace reads with `self.data_structure.get(name)` (or take the variable as a step argument), and writes with `ds.set(name, values)`. Translator-derived inputs are read-only.
- **Boundary conditions on sets of nodes** (leaves, root surfaces, the collar) become `boundary_set(...)` in the graph system, with value and weight as DataStructure variables, rather than Robin terms assembled in the balance and the Jacobian by hand (`test_uc5_leaf_transpiration.py` is the reference).
- **Undeclared outputs** give their location: `@graph_output(name, location=...)`, `@rate(location=...)`.
- **Missing variables raise:** a graph-system argument or a filter variable must be registered (declared, or set before the solve). Missing values are no longer read as zeros.
- **Non-float variables.** Labels and types are declared with `dtype="int"`. Lists such as `xylem_vessel_radii` are declared with `dtype="object"`: they are stored per entity and carried over by growth, but graph systems, derivations and transport reject them. Filters and masks accept label names (`{"label": ["RootSegment"]}`).

## 3. Plant model (GrassBRIDGES): a population model

A plant model is built once for all the plants of that model in a scene:

```python
class GrassBRIDGES(CompositeModel):
    initiators = (RootGrowth, ShootGrowth)       # StructuralComponents: initiate_plant(g, plant, scenario) per plant
    from_scale = "SubOrgan"                      # the graph nodes; nodes = "Compartment" for anatomies
    parts = {"growth": RootGrowth, "carbon": RootCNUnified, ...}   # built on the MPG's DataStructure, coupled
    translator = TRANSLATOR                      # None: every input shared by its name
    # schedule = ("growth", "carbon", ...)       # the parts in order by default (coupling.schedule)
```

- **Plants:** each structural component builds a plant's initial structure in `initiate_plant(g, plant, scenario)`, from the Plant variables `x, y, z, rotation` (z the soil surface minus the planting's depth), its elements placed with `g.add_organ` and `g.add_element`, its parameters `cls.plant_parameters(scenario)`. Several initiators run in order.
- **Parameters:**
  - numeric parameters are stored per plant and come from each plant's scenario;
  - steps take them as arguments;
  - other scenario entries are shared by the plants of one model.
- **Coupling:**
  - the model's translator couples its own components within the MPG: identity links need nothing; aliases and conversions become DataStructure aliases and derived variables. Derived variables are recomputed when read after a source changed, and are read-only;
  - links with the environment are the scene translator's;
  - `mtg_to_arraydict` and the "convert before coupling" ordering constraint are gone.
- **Translator:** YAML files load unchanged. A Python translator (`translator = Translator().link(...)` in a `.py` module) adds live `scales.*` references, `aggregation=` / `weight=` for scale changes, and `formula=`. Keep identity links written explicitly for readability. String factors are parsed arithmetic; `eval` is gone.
- **Removed:** the queues, `name`, `coordinates`, `rotation`, `Transport` and `handshake_shape`. The scene does the exchanges.
- **Initial values:** the soil inputs are no longer reset to 0 at coupling; they keep their declared `initialize` until the first exchange.
- **MPG-style steps** loop over `self.active_ids()`, so that plants before their planting time are skipped.
- **MTG reads:** a state variable held at its own scale is read from its DataStructure column by the MTG; mapped ones reach the MTG lazily (`mtg_sync = "lazy"`). Code that reads the MTG object directly, such as a Logger or plotting, calls `ds.flush_mtg()` first, or reads `ds.mtg`.
- **MTG properties:** `g.property(name)` is read-only (a mapping built on demand). Write `g.set_values(name, vids, values)` instead of `g.property(name)[v] = x`; read in bulk with `g.values_at(name, vids)` or `g.column(name)`, vertex by vertex with `g.view(name)`.
- **Check:** in the package's own tests, add `openalea.metafspm.testing.assert_component_couplable(Component, translator)` for every component.

## 4. Environment models and the scene (RhizoSoil, the light model)

- **Soil component:** `SoilModel` becomes a `FunctionalComponent` on `ArrayDataStructure(shape=(nx, ny, nz), dx=side)`.
  - Axes are `(x, y, z)`. Legacy `(ny, nz, nx)` voxel arrays convert with `legacy.transpose(2, 0, 1)`.
  - Fields are declared with `location="cell"`, or `"scalar"` for uniform drivers such as rain.
  - The grid gives `cell_centers()`, `cell_volume()` and `locate(points, periodic=, clip=)`.
- **Environment model:** `Model(plant_data_structures, xrange, yrange, time_step, <its own arguments>)` builds its DataStructures and exposes `components` and `run()`, its `compartment` ("Soil" or "Atmosphere") naming its outputs' folder; its time-varying inputs are the scene's forcings. Its DataStructure can be:
  - a grid (soil, RATP-like light);
  - a `UnionDataStructure(populations)` for a model that must see every population (CARIBU-like light);
  - a population's MPG.
- **Scene translator:** the links between the plant and environment components. Defaults follow the variables' kinds:
  - extensive plant → cell: `sum`;
  - intensive plant → cell: `weighted_mean`, which needs `weight=`;
  - intensive cell → plant: `broadcast`;
  - extensive cell → plant: `split`, which needs `weight=`;
  - variables without a declared kind need an explicit `aggregation=`.
- **The scene** (`openalea.metafspm.scene.scene.Scene`, see `test/provide_usage_examples/scene_example.py`):
  - builds one population per plant model of the planting table (`planting(...)`, with `stand(...)` for a sown stand), then the environment models;
  - infers the mappings (barycentre by default, `mapping_method="overlap"` as an option) and runs one `Exchanges` before each model: each environment model after the exchanges into it, then each population after the exchanges into it;
  - several plants feed one soil in one write, without zeroing.
- **Replaced:** `compute_mtg_voxel_neighbors_fast`, `apply_to_voxel_fast` and `get_from_voxel_fast` by `CrossMapping`. `play_Orchestra`, `Transport`, `Coupler` and the queue protocol are removed.
- **Light model:** known bug to fix at the same time: the first `run` crashes when there is no light at t=0 (`previous_Erel` is None).

## 5. Outputs

- **The SceneRecorder replaces fspm-utility's Logger** (`Scene(..., recorder=SceneRecorder(folder, ...))`):
  - `summaries.csv` (sums and means per plant, and per environment grid in `Soil/` or `Atmosphere/`) replaces the
    summed CSV;
  - `segments.csv` for the `segment_plants` and the encoding3d frames replace the NetCDF raw logs and the MTG pickles;
  - `run.json` (status, timing) and `scenario.json` replace the simulation log and the saved scenario;
  - `checkpoints/` restart a run;
  - `tables={name: function(scene)}` adds a package's own summary.

  The `logger_class` / `log_settings` hook is removed.
- **Scenario files:** `read_scenarios` reads fspm-utility's sheets once laid out as one row per entry, a `model`
  column instead of `Dedicated_to`, and `type` (`parameter`, `forcing`) instead of `Input_type`.
  `MakeScenarios.from_table` and the Sobol sampler become `read_scenarios` and `variants(design="sampling")`.
- **Input tables:** `CompositeModel.apply_input_tables` is removed. Time series are the Scene's `forcings=`, which
  write the uncoupled inputs of that name at each step.

## 6. Checklist per package

1. The imports are migrated.
2. Each component is a `FunctionalComponent`, with `scale` on its coupled and solved fields, vectorised steps (or the explicit opt-in), `previous()` and `on_grow`.
3. `assert_component_couplable` passes for each component against the shipped translator, with `data_structure=` the plant (or soil) DataStructure, so that declarations are resolved and the DataStructure validated.
4. The plant model follows the population contract, and the environment models follow §4.
5. A short `Scene` run gives the expected outputs for a fixed seed and a few steps, the way `test/scenes/test_scene_contract.py` does for the doubles.

The legacy props path (the props branches of `CompositeModel`, the Functor and `FunctionalComponent`) and the per-process scene (`play_Orchestra`, `Transport`, `Coupler`) are already removed from metafspm. Migrated packages must target the current API.
