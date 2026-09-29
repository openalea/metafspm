# Migrating downstream packages to DataStructure coupling

This guide covers devplan WD.7. It applies to rootbridges, rootcynaps, cnwgrass, rhizosoil, grassbridges and the `openalea.fspm` Logger.
The design is in `coupling_through_datastructures.md`, and the full list of API changes is in `CHANGELOG.md`.

Every step below has a **working, tested reference** in `test/wrappers_tests/doubles_ds.py`, whose contract tests reproduce the numbers of the props-based coupling:

| Downstream role | Reference double |
|---|---|
| plant components (RootCNUnified, RootWaterModel, …) | `PlantCarbon`, `PlantNitrogen` |
| plant composite (GrassBRIDGES) | `DSFakePlant` |
| soil component (rhizosoil `SoilModel`) | `GridSoil` |
| soil composite (RhizoSoil) | `DSFakeSoil` |
| light model (LightModel) | `doubles.FakeLight` |

**The legacy props path has been removed from metafspm (Q28).** Downstream packages keep working against their pinned metafspm, the `publish_WB` branch, until they are migrated to this API.

## 1. Imports

Replace the flat modules, following the table in `CHANGELOG.md`:

| Old | New |
|---|---|
| `metafspm.utils` | `data_structure.arraydict` |
| `composite_wrapper` | `coupling.composite_wrapper` |
| `component_factory` | `coupling.choregrapher`, `solve.decorator`, `solve.legacy_functor` |
| `component.Model` | `coupling.component.FunctionalComponent` |

## 2. Components → `FunctionalComponent`

**Before**, with the MTG props:

```python
@dataclass
class RootCNUnified(Model):
    hexose: float = declare(default=1e-3, ..., variable_type="state_variable", ...)
    def __init__(self, g, time_step, **scenario):
        self.props = g.properties(); self.vertices = ...
        self.apply_scenario(**scenario); self.link_self_to_mtg()
        self.choregrapher.add_time_and_data(instance=self, sub_time_step=time_step, data=self.props)
```

**After**, reading the DataStructure live:

```python
@dataclass
class RootCNUnified(FunctionalComponent):
    hexose: float = state_variable(..., initialize=1e-3, scale=scales.SubOrgan, on_grow="inherit")
    struct_mass: float = input_variable(..., scale=scales.SubOrgan)
    total_cytokinins: float = state_variable(..., scale="scalar")          # plant-scale value (was vertex 1)

    @rate
    def _hexose_exudation(self, hexose, struct_mass):        # receives whole arrays
        return self.k * hexose * struct_mass

    @rate(vectorized=False)                                  # opt-in: one call per element, for scalar logic
    def _something_with_ifs(self, hexose): ...

model = RootCNUnified(data_structure=MPGDataStructure(g, from_scale=g.scales.SubOrgan))
```

The rules:
- **Every coupled or solved variable declares a `scale`.** It is then registered on the DataStructure automatically:
  - a bio-scale integer (e.g. `scales.SubOrgan`), with `edge_mapping` for edge variables;
  - `"node"`, `"edge"`, `"cell"` or `"scalar"`.
- **Step functions receive arrays** (Q20). Make the bodies numpy-compatible, e.g. with `np.where` instead of `if`, or mark the step `vectorized=False`.
- **Previous state:** inside graph-system equations, use `self.previous("concentration")` instead of a user-managed `_previous_fields` (Q21).
- **Growth:** declare `on_grow="inherit"` for variables that new segments should take from their parent. The default is the declared value. The growth model may still overwrite new entities, e.g. from parent concentrations or split extensive quantities (Q24).
- **Growth models** call `ds.update_topology()` after changing the MTG. Registered variables are carried over, and components rebuild their graph views.
- **Segment geometry:** the growth model registers `x1, x2, y1, y2, z1, z2` as node variables. The soil coupling needs them (Q25).
- **`self.props`** is a read-only view kept for one release. Replace reads with `self.data_structure.get(name)`.
- **Non-float variables** (lists such as `xylem_vessel_radii`) are not solver or transport variables. Keep them on the MTG or as instance attributes, and couple them by identity only (Q22).

## 3. Plant composite (GrassBRIDGES)

```python
class GrassBRIDGES(CompositeModel):
    soil_name = "SoilModel"                                # the soil component name used in the translator
    def __init__(self, queues_soil_to_plants, queue_plants_to_soil, queues_light_to_plants, queue_plants_to_light,
                 name, time_step, coordinates, rotation, translator_path, **scenario):
        Choregrapher().add_simulation_time_step(time_step)     # before building the components
        self.plant_ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
        self.components = (RootGrowth(data_structure=self.plant_ds), RootCNUnified(data_structure=self.plant_ds), ...)
        self.declare_data_and_couple_components(root=self.plant_ds, translator_path=translator_path,
                                                components=self.components)
        self.transport = Transport.from_translator(Translator.load(translator_path), soil=self.soil_name,
                                                   plant_components=[c.__class__.__name__ for c in self.components],
                                                   capacity=capacity_from_the_shared_memory_size)
```

- **Coupling:**
  - identity links need nothing;
  - aliases and conversions become DataStructure aliases and derived variables, refreshed before each receiver's step;
  - `mtg_to_arraydict` and the "convert before coupling" ordering constraint are gone.
- **Translator:** YAML files load unchanged. A Python translator (`translator = Translator().link(...)` in a `.py` module) adds live `scales.*` references, `aggregation=` / `weight=` for scale changes, and `formula=` (Q4b). Keep identity links written explicitly for readability (Q26). String factors are parsed arithmetic; `eval` is gone.
- **Soil exchange:**
  - each step, call `self.transport.write_plant(buffer, self.plant_ds)` before sending the status, and `self.transport.read_soil(buffer, self.plant_ds)` after the soil's `"finished"`;
  - send `{"plant_id", "model_name", "handshake": self.transport.rows, "capacity": ...}`, adding `"carried_components"` at init;
  - the `vertex_index >= 1` mask, `soil_handshake` and the fixed `(35, 20000)` shape are gone.
- **Scene:** call `play_Orchestra(..., handshake_shape=Transport.from_translator(...).shape)` (Q27).
- **Check:** in the package's own tests, add `openalea.metafspm.testing.assert_component_couplable(Component, translator)` for every component.

## 4. Soil (rhizosoil)

- **Component:** `SoilModel` becomes a `FunctionalComponent` on `ArrayDataStructure(shape=(nx, ny, nz), dx=side)`.
  - Axes are `(x, y, z)` (Q16b). Legacy `(ny, nz, nx)` voxel arrays convert with `legacy.transpose(2, 0, 1)`.
  - Fields are declared with `scale="cell"`, or `"scalar"` for uniform drivers such as rain.
  - The grid gives `cell_centers()`, `cell_volume()` and `locate(points, periodic=, clip=)`.
- **Composite (RhizoSoil):** see `DSFakeSoil`.
  - At init, for each plant message: `Transport.from_rows(message["handshake"], message["capacity"], to_soil=, to_plant=)`, taking the links from the soil's translator for the plant's `carried_components`.
  - Each step, for each plant: `Coupler(transport.plant_view(buffer), soil_ds, VoxelLocator(soil_ds))`, `update_map()`, then:
    1. one `zero_soil_inputs()`;
    2. `push()` for all plants;
    3. the soil step;
    4. `pull()` for all plants;
    5. reply `"finished"`.
  - This replaces `compute_mtg_voxel_neighbors_fast`, `apply_to_voxel_fast` and `get_from_voxel_fast`, with identical sums (tested).
- **Aggregation:** fluxes are always summed into voxels (extensive), and soil states are gathered (intensive), whatever `state_variable_type` says.

## 5. Light model

- The constructor receives the queues: `LightModel(queues_light_to_plants, queue_plants_to_light, scene_xrange, scene_yrange, meteo, **scenario)`.
- It must **answer the plants' initialization messages in `__init__`**, as the soil does (Q17). Without this, every model runs `n_iterations - 1` steps.
- `meteo` comes from `light_scenario["meteo"]`, a csv path or a DataFrame (Q8).
- Known bug to fix at the same time (B11): the first `run` crashes when there is no light at t=0 (`previous_Erel` is None).

## 6. Logger (`openalea.fspm`): the xarray and csv writers only (Q15)

- Accept any DataStructure in `model_instance.data_structures`, alongside the MTG / dict types.
- In `mtg_to_dataset` / `recording_raw_MTG_properties_in_xarray`, replace the props traversal with:

  ```python
  table = ds.to_dataframe(variables_at_that_location, location="node", time=self.simulation_time_in_hours)
  table = table[table["struct_mass"] > 0]                      # emerged segments, as before
  dataset = table.to_xarray()                                  # soil: location="cell", voxel index + x/y/z columns
  ```

- In `recording_summed_MTG_properties_to_csv`, replace it with:

  ```python
  row = ds.summarize(sums=summable, means=meanable, scalars=plant_scale_state, where="struct_mass")
  ```

  Plant-scale values come from the `"scalar"` store instead of vertex 1 (Q23).
- The Logger reads `fields(component)` metadata. Skip fields without `variable_type`, e.g. `FunctionalComponent.data_structure`, as `CompositeModel.get_documentation` already does.

## 7. Checklist per package

1. The imports are migrated (§1).
2. Each component is a `FunctionalComponent`, with `scale` on its coupled and solved fields, vectorised steps (or the explicit opt-in), `previous()` and `on_grow`.
3. `assert_component_couplable` passes for each component against the shipped translator.
4. The composite uses `Transport` and passes `handshake_shape`; the soil and light models follow §4–§5.
5. A short scene run gives the same outputs as before the migration, for a fixed seed and a few steps, the way `test_ds_scene_contract.py` does for the doubles.

The legacy props path (the props branches of `CompositeModel`, the Functor and `FunctionalComponent`, and the `HANDSHAKE_SHAPE` default) is already removed from metafspm. Migrated packages must target the current API.
