# Design note: coupling components through DataStructure variables

Status: **draft for review** (devplan WD.0 + WD.1). Branch `release2026`, written against `a3a9877`.
Code changes (WD.P, WD.2–WD.9) start only after this note is accepted. The open decisions are listed in §11.

## 1. Decisions already taken

| Ref | Decision |
|---|---|
| Q3 | Components are coupled dynamically, through the variables of `DataStructure` derivatives, and no longer through `g.properties()` dicts. |
| Q13 | Components read DataStructure arrays **live**. The `self.props = ds.to_props_dict()` snapshot is removed. |
| Q14 | The switch is total, with no dual MTG-props support. Plants use `MPGDataStructure` and the soil uses `ArrayDataStructure` (3-D). `MultiGridDataStructure` comes later (Q16). |
| Q16b | The canonical grid axis order is `(x, y, z)`. RhizoSoil's `(y, z, x)` indexing is converted at the migration boundary. |
| Q4/Q4b | The translator becomes Python-first, with live references such as `scales.SubOrgan` and free formulas. The existing YAML files stay loadable. |
| Q15 | The Logger scope is limited to its xarray and csv writers, fed by a DataStructure export API. |
| Q17/Q18/Q19 | Scene protocol decisions, already implemented (`062c61c`, `a3a9877`). |

**Regression gate.** The wrapper `[contract]` tests (`test/wrappers_tests/`) must pass **unchanged** after the doubles are retargeted to DataStructures. That holds in particular for the hand-derived two-cycle anchor `test_composite_contract.py::test_two_cycles_regression_anchor`.

## 2. Where we start from

This summary comes from three code surveys (2026-09-29); their key findings are recorded in the devlog.

**Three copies of every variable, which drift apart:**
1. **MTG/MPG properties** (`ArrayDict` or `dict`), keyed by bio VID.
2. **DataStructure arrays** (`MPGDataStructure._node_data` / `_edge_data`, `ArrayDataStructure._fields`), in local order.
3. **`FunctionalComponent.props`**, a dict-of-dicts copy built once in `__post_init__` (`component.py:292`). Nodes are keyed by VID, edges by the 0-based edge index, and the two share one integer key space.

After construction, every solver and Choregrapher write goes to copy 3 only. Copy 2 goes stale, and copy 1 is updated only by `write_back_to_mtg`, for state variables, after graph solves. No test asserts on the DataStructure arrays after a solve.

**Array-native already:**
- the spec layer (`system_specs.GraphDAESpec`, `EquationContext`, `FieldState`), which works only on local-index arrays;
- `make_evaluator`;
- `_auto_declare_on_ds`;
- `write_*_to_mtg` (`assign_at` fast paths).

**The hard part:**
- the legacy `Functor` per-element branch, which calls a scalar Python function once per vid;
- the props-coupled code in `CompositeModel` (identity aliases, `pullable_inputs` with per-vid loops, `apply_input_tables` writing vertex 1).

**Bugs that block the design (WD.P):**

| # | Bug | Where |
|---|---|---|
| B-a | `GraphView.node_local_index` uses `searchsorted` on unsorted ids. MPG local order is Compartment post-order, so every lookup is wrong. | `data_api.py:176` |
| B-b | The incidence signs are opposite: `incidence_matrix` is −1 at the parent, `to_graph_view` is +1. | `data_api.py:778` vs `:856` |
| B-c | The bio-scale mapping helpers have no scale argument. An Organ variable returns `None` and falls back silently to the default, and the fast path matches on *size* only. | `data_api.py:636-718`, `component.py:353-360` |
| B-d | `write_*_to_mtg` swallow every exception. | `data_api.py:741,773` |
| B-e | `update_topology` clears **all** arrays. Growth wipes the state unless the caller re-registers everything. | `data_api.py:842` |
| B-f | There is no common public accessor: graphs use `node_property`, fields use `_get_field`. `set_*` rebinds the array (which breaks aliasing) and does not check lengths. | `data_api.py:616-632, 1057-1063` |
| B-g | Grid axes have no names. `coordinates()` returns vertex positions, not cell centres. There is no point→voxel lookup and no voxel volume. | `data_api.py:988-1019` |
| B-h | `LabelsConfig` mutates its class attributes, so a second MPG gets an empty translator. `Compartment.Apoplastic == Connection.Apoplastic`. | `configs.py:181-201` |
| B-i | `populate_graph` adds Compartments to the SubOrgan `children()`. The SubOrgan anchor matches `scale == SubOrgan` filters. | `mpg.py:75,129` |

## 3. Requirements (from the downstream reference code)

The WheatBRIDGES translator (`test/inputs/wheatbridges_coupling_translator.yaml`) has **98 links**:

| Link kind | Count |
|---|---|
| identity | 66 |
| alias | 10 |
| numeric factor | 4, including `1e-6` |
| string expression | 18, including negative sign conventions |
| same-name factor | 0 |
| multi-source | 0 |

The new layer must:

| # | Requirement |
|---|---|
| R1 | Keep **identity links free**: components on one DataStructure share variables by name. |
| R2 | Keep **aliases valid** through in-place writes, re-registration and growth. Today this relies on object identity (`GrassBRIDGES` comments: "BEFORE THE COUPLING FOR ALIASES TO REMAIN UNBROKEN", "per-variable update otherwise dynamic links are broken"). |
| R3 | Evaluate **factor and weighted-sum links lazily**, at the receiver's step, in the order the composite runs its components. Negative factors must be allowed. |
| R4 | Accept factors written as **expressions** (`"0.000001 / 3600 / 12"`). |
| R5 | Store the soil's links **per plant model**, since different plant composites may send different variables to one soil. |
| R6 | Aggregate **plant → soil as an extensive sum** over all segments of all plants in a voxel (`np.add.at`). Do not trust `state_variable_type`, which is blank for 7 of the 16 soil inputs: the aggregation must be declared on the link or the coupler. |
| R7 | Transfer **soil → plant as an intensive gather**: each segment takes its voxel's value. |
| R8 | Handle **growth**: the vertex→voxel map is recomputed at every exchange, and receiver fields must hold values for new vertices before the next exchange. Today the growth model fills them (`soil_boundaries_to_infer`). |
| R9 | Give **plant-scale values** (collar flows, `total_*`, shoot↔root scalars, input tables) an explicit home. Today they live at "vertex 1" of the root props. |
| R10 | Support **non-float payloads** (`xylem_vessel_radii`, `adventitious_to_emerge`), or state explicitly that they are not couplable. |
| R11 | Keep the scene wire protocol (`[contract]` tests), with a handshake **sized from the translator**, which today exactly fills the fixed 35 rows. |
| R12 | Provide documentation from field metadata. The Logger requires `variable_type` on *every* field. |

## 4. Translator (WD.0)

### 4.1 Model

```python
from dataclasses import dataclass, field
from typing import Callable, Mapping, Union

Factor = Union[float, str]          # str: restricted arithmetic, parsed once, e.g. "-12 * 1e6 * 3600"

@dataclass(frozen=True)
class Link:
    receiver: str                   # component class name, e.g. "SoilModel"
    variable: str                   # receiver variable name
    provider: str                   # component class name
    sources: Mapping[str, Factor]   # {provider variable: factor}; several entries = weighted sum
    # Optional, all default to "same location, no transform":
    scale: "int | None" = None      # receiver-side scale, a live reference: scales.SubOrgan, scales.Plant, ...
    source_scale: "int | None" = None
    aggregation: "str | Callable | None" = None   # "sum" | "mean" | "weighted_mean" | "proximal" | "distal" | "broadcast" | callable
    weight: "str | None" = None     # variable used by "weighted_mean"
    formula: "Callable | None" = None             # free formula: (source arrays...) -> array, replaces Σ f·x

@dataclass
class Translator:
    links: list[Link] = field(default_factory=list)

    def link(self, receiver, variable, provider, sources=None, **options): ...   # builder, returns self
    def links_of(self, receiver, provider=None) -> list[Link]: ...
    def inputs_outputs(self, components, target) -> tuple[list[str], list[str]]: ...  # replaces get_component_inputs_outputs
    @classmethod
    def from_yaml(cls, path) -> "Translator": ...   # the existing format, unchanged
    @classmethod
    def from_module(cls, path_or_module) -> "Translator": ...  # a module exposing `translator = Translator(...)`
```

**Classification.** A link's kind is derived from its fields, never declared:
- **identity**: `sources == {variable: 1}` with no scale change;
- **alias**: `sources == {other: 1}`;
- **derived**: everything else, meaning a factor ≠ 1, several sources, a scale change or a formula.

A same-name factor on one DataStructure is rejected, as today (W2.5).

### 4.2 A Python translator, written against the real file

```python
# grassbridges/coupling_translator.py
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import ScalesConfig as scales

translator = (Translator()
    # identities need no entry at all when both components live on the same DataStructure (R1)
    .link("RootCNUnified", "soil_AA", "SoilModel", {"C_amino_acids_soil": 1})                 # alias
    .link("SoilModel", "hexose_exudation_massic", "RootCNUnified", {"hexose_exudation": 12 * 6},
          aggregation="sum")                                                                 # extensive, to voxels (R6)
    .link("CNW_Grass", "Unloading_Sucrose_phloem", "RootCNUnified",
          {"sucrose_root_to_shoot_phloem": -12 * 1e6 * 3600}, scale=scales.Plant)            # plant scale (R9)
    .link("RootCNUnified", "mean_soil_temperature", "SoilModel", {"soil_temperature": 1},
          scale=scales.Organ, aggregation="weighted_mean", weight="length")                 # scale change
)
```

- **Python gives:** real `scales.*` references, IDE refactoring, computed factors, and `formula=` callables for anything that isn't a linear combination.
- **The YAML loader** maps each `receiver: provider: variable: {source: factor}` entry to one `Link`. Scale and aggregation are optional keys:

  ```yaml
  variable: {sources: {...}, scale: SubOrgan, aggregation: sum}
  ```

  Scale names are resolved in `ScalesConfig` when the file is loaded. The short form stays valid, so the current files load unchanged.
- **Identity entries** in the YAML are kept for documentation and for the soil handshake, but they create no DS link.

### 4.3 Why not YAML-only, and why keep it

YAML can't hold live references or callables. Keeping the loader costs little and lets the current `.yaml` files and `translator_matrix_builder` keep working. `translator_matrix_builder` will write the short form.

## 5. Variables on a DataStructure

### 5.1 One accessor for graphs and grids

Add to `DataStructure`:

```python
def get(self, name) -> np.ndarray          # live view; resolves aliases; raises KeyError with the available names
def set(self, name, values) -> None        # writes IN PLACE when registered (length/shape checked), registers otherwise
def register(self, name, values=None, location=None, default=0.) -> None
def location(self, name) -> str            # "node" | "edge" | "cell" | "scalar"
def has(self, name) -> bool
```

`MPGDataStructure` routes node and edge names to `_node_data` / `_edge_data`. `ArrayDataStructure` routes to `_fields`, and adds a small **scalar store** for plant- or scene-scale values (§6.2). The existing `node_property`, `set_node_property`, `_get_field` and `_set_field` remain as thin wrappers.

**In-place rule (fixes B-f, serves R2).** `set` never rebinds a registered array, so a view taken earlier stays valid. Rebinding happens only in `register` and `update_topology`, and both bump `ds.version`.

### 5.2 Aliases

```python
ds.alias("sugar", "hexose")   # get("sugar") is get("hexose"); set("sugar", ...) writes hexose in place
```

- The alias table is name-level (`_aliases: dict[str, str]`), so it survives rebinds and `update_topology`.
- It rejects cycles, and aliasing a name that already has its own registered array unless the two are equal.
- `available_vars()` lists aliases separately.

### 5.3 Derived variables

```python
ds.derive("carbon_supply", {"hexose_exudation": 2.0})          # linear combination
ds.derive("x", formula=f, sources=("a", "b"))                   # free formula
ds.refresh("carbon_supply")                                     # get(x) = Σ f_i * get(s_i), written in place
```

- A derived variable is a registered array like any other, recomputed on `refresh`.
- `refresh` is called by the **receiver** right before its step (R3), replacing `pull_available_inputs`.
- The component knows its derived inputs from the links registered by the composite.

## 6. Scales within a plant

### 6.1 Addressing

Variables stay addressed by **name**, and each registered variable carries its location: `ds.location(name)`, plus `ds.scale(name)` for bio-scale anchoring. A link with `scale=` / `source_scale=` builds a derived variable through a **scale operator**, a sparse matrix built from the MPG and cached per `ds.version`:

| Aggregation | Operator (fine → coarse unless stated) |
|---|---|
| `sum` | 0/1 membership matrix `M` (coarse × fine), `y = M x` |
| `mean` | row-normalised `M` |
| `weighted_mean` | `diag(1/(M w)) M diag(w)` |
| `broadcast` | `Mᵀ`, coarse → fine |
| `proximal` / `distal` | node → edge selection, the existing `_bio_edge_b_idx` / `_a_idx` |

- These replace the Python loops of `integrate_at_scale` / `average_at_scale`, which remain for the MTG itself.
- The membership comes from `components_iter` / `complex`. `up_scale_parent` and `down_scale_children` are declared in `PropsConfig` but never filled, so either fill them in `populate_graph` or drop them.
- The `edge_mapping` naming is inverted from botanical usage (here "proximal" means the child owns the edge). Document it, don't rename it.

### 6.2 Plant scale (R9)

- Plant-scale values move from "vertex 1" to the **scalar store** (`location == "scalar"`), with `get` returning a 0-d array view.
- Links to or from them use `scale=scales.Plant`, with `sum` / `mean` / `broadcast` as needed.
- `apply_input_tables` writes the scalar store, or `fill`s grid cells, exactly as today.
- This changes the Logger CSV writer, which currently removes vid 1 (§9).

## 7. Coupler across DataStructures (plant graph ↔ soil grid)

```python
coupler = Coupler(plant_ds, soil_ds,
                  locate=VoxelLocator(origin=(0, 0, 0), dx=(dx, dy, dz), shape=(nx, ny, nz),
                                      periodic=(True, True, False), flip_z=True),
                  to_soil={"hexose_exudation_massic": LinkSpec({"hexose_exudation": 72}, "sum")},
                  to_plant={"C_hexose_soil": LinkSpec({"C_hexose_soil": 1}, "gather")})
coupler.update_map()      # vertex -> cell index from segment barycenters (x1..z2 on plant_ds); every exchange (R8)
coupler.push()            # np.add.at(soil[name], cells, Σ f * plant[src])  (R6; zeroing policy explicit)
coupler.pull()            # plant[name][:] = soil[name].ravel()[cells]      (R7, in place)
```

- **Axis order `(x, y, z)`** (Q16b):
  - `ArrayDataStructure(shape=(nx, ny, nz))` stores C-order `(x, y, z)`.
  - `VoxelLocator` returns flat cell indices, so no `(iy, iz, ix)` triples leak out.
  - The RhizoSoil adapter permutes once, at migration.
- **Cell geometry (fixes B-g):** add `ArrayDataStructure.cell_centers()`, `cell_volume()` and `locate(points)`.
- **Several plants → one soil:**
  - one Coupler per plant, and all of them share the soil DS;
  - `push` of all plants happens after one explicit `soil_ds.zero(inputs)`, which makes today's implicit zeroing in `SoilModel.__call__` explicit;
  - the init-time "sum on top of defaults" quirk (R12 of the survey) gets decided explicitly.
- **Across processes:** the Coupler owns its transport.
  - `Transport.plan(coupler)` derives rows from the Coupler's variables, which replaces `soil_handshake_inputs` + `HANDSHAKE_SHAPE`. It derives columns from `plant_ds.n_nodes()`, with capacity growth.
  - The plant side sends the barycenters and push variables; the soil side runs `update_map` / `push` / `pull` on its copy.
  - The queue protocol and the `"finished"` handshake stay as tested (WD.5).
- **Growth (R8, B-e):**
  - `update_topology` must **preserve** existing variables. Add a `on_grow(policy)` hook: new nodes get the declared default, or the parent's value (`inherit`), per variable.
  - It must not clear the arrays, which is today's behaviour.
  - Couplers and scale operators rebuild when `ds.version` changes.

## 8. Live reading in components and the solver (WD.2)

1. **Bind the DataStructure, not a dict.**
   - `FunctionalComponent.__post_init__` keeps `_auto_declare_on_ds` and `_graph_view`, and drops the `props` snapshot and `focus_elements` injection.
   - The Choregrapher binds `ds` itself: `add_time_and_data(self, 1, ds)`, where the DS identity is stable across topology changes.
2. **Functor gets a DataStructure branch**, dispatching on `isinstance(data, DataStructure)` instead of the `"length"` string probe:
   - inputs are `ds.get(arg)[mask]`, and the output is `ds.get(name)[mask] = fun(instance, *inputs)`, in place, routed by the declared location;
   - `mask` is a local-index mask replacing `focus_elements`, recomputed per `ds.version`;
   - supplementary outputs are kept;
   - `total` / plant scale go to the scalar store.
3. **Vectorisation contract** (Q20): step functions receive arrays. Scalar-only functions opt in to a per-element loop, via `@rate(vectorized=False)`, which keeps today's semantics and cost.
4. **Solver path** (`decorator.py`):
   - `_snapshot` becomes `ds.get(name)`;
   - the `_prop_location` heuristic is replaced by `ds.location`;
   - `_type_mask` becomes `np.isin`;
   - unknowns, initial guesses, `amount_olds` and the first-tick previous fields are **explicit copies** per solve, because the implicit solvers use them as `u_prev`;
   - `inject_result` and `@graph_output` write `ds.set` in place;
   - `{fn}_amount` and output names are registered on first write, with their declared location;
   - `build_and_step` returns its spec, which removes the second `build()`.
5. **Previous state:** a framework-managed `ds` field `"<fn>__prev"`, advanced after each accepted solve, exposed as `self.previous(fn)`. It replaces the user-managed `_previous_fields`, with a transition period (Q21).
6. **MTG sync:**
   - `write_back_to_mtg` becomes `ds.write_node_to_mtg(name, ds.get(name))`, and exceptions are no longer swallowed (B-d);
   - it runs at the end of every `Component.__call__`, so that components without a graph solve also sync;
   - `_refresh_from_bio_scale` keeps its parameters-only rule;
   - both use the scale operators (§6.1), which fixes B-c.
7. **Compatibility view:** a read-only `props` property (a lazy `{vid: value}` view) for one release, so that external scripts can migrate. Tests move to `ds.get`.

## 9. Logger export (WD.9, reduced scope per Q15)

The metafspm side adds the following API:

```python
ds.export(names=None) -> dict[str, np.ndarray]          # live copies, local order
ds.ids() -> np.ndarray                                   # node vids / flat cell ids
ds.coords() -> dict[str, np.ndarray]                     # per-location coordinates for xarray
ds.to_xarray(names=None) -> xarray.Dataset               # optional dependency
```

Only the Logger's `mtg_to_dataset` / `recording_raw_MTG_properties_in_xarray` and `recording_summed_MTG_properties_to_csv` change:
- they accept any `DataStructure` in `data_structures`, besides MTGs and dicts;
- they read `ds.export()` / `ds.ids()`;
- plant-scale values come from the scalar store instead of vid 1;
- `fields()` without `variable_type` metadata are skipped, as `get_documentation` already does.

Tests run on `MPGDataStructure` and `ArrayDataStructure`. The rest of the Logger is out of scope.

## 10. Migration sequence and test gates

| Step | Content | Gate |
|---|---|---|
| WD.P | Fix B-a, B-b, B-c (scale arg + key-matching fast path), B-d, B-e (preserve on grow), B-f (`get` / `set` / `register`, in place), B-g (grid geometry), B-h (LabelsConfig per instance); decide B-i | New unit tests; existing 398 stay green |
| WD.3 | Alias table, derived variables, scale operators, scalar store | Unit tests (chains, cycles, growth, missing sources) |
| WD.2 | Live reading in the solver path, then the Functor DS branch, then removal of the snapshot | UC1 suites green; new "ds holds the solution" tests |
| WD.0 impl | `Translator` / `Link` (Python + YAML loader, restricted parser) | Round-trip of the WheatBRIDGES YAML: 98 links, same classification counts as §3 |
| WD.4 | `CompositeModel` on Translator + DS links; documentation from metadata | W2 tests retargeted |
| WD.5 | Coupler + Transport, sized from the translator | W3 + W5 unchanged, including the regression anchor |
| WD.6 | Doubles on `MPGDataStructure` / `ArrayDataStructure (x, y, z)` | **All `[contract]` tests unchanged** |
| WD.7–WD.9 | Migration guide, `assert_component_couplable`, Logger writers | Downstream checklist |

## 11. Open decisions for review

| # | Question | Recommendation |
|---|---|---|
| Q20 | **Vectorisation contract:** must step functions (`@rate`, `@state`, …) accept arrays, with a per-element loop only as an opt-in (`vectorized=False`)? | Yes. It is the only way live DS reading avoids per-vid Python loops, and the opt-in keeps legacy bodies working. |
| Q21 | **Previous state:** should a framework-managed `self.previous(fn)` replace the user-managed `_previous_fields`? | Yes, with a one-release deprecation. |
| Q22 | **Non-float variables** (lists such as `xylem_vessel_radii`, `adventitious_to_emerge`): store them in an object store on the DS that is not solver- or transport-eligible, or keep them MTG-only? | Object store on the DS, couplable by identity/alias only. |
| Q23 | **Plant-scale values** move from "vertex 1" to a scalar store, which changes the Logger CSV (vid 1 is no longer special). Accept? | Yes. The Logger is adapted in WD.9. |
| Q24 | **Growth policy:** when topology grows, do new nodes take the declared default or the parent's value? Should the policy be declared per variable (e.g. `declare(..., on_grow="inherit")`)? | Per-variable, default `"default"`. The growth model can still overwrite. |
| Q25 | **Segment geometry** (`x1..z2`): confirm that the growth model registers them as node variables on the plant DS, under these names, as the Coupler's locator input. | Yes. Keep the names, and make the locator's variable names configurable. |
| Q26 | **Translator identity entries:** keep them in Python translators for documentation and handshake derivation, or derive the handshake from the Couplers only and allow identities to be omitted? | Derive from the Couplers; identities are optional in Python and kept in YAML. |
