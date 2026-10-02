# Design note: the DataStructure variable contract (step 1)

Status: **under review**. N1–N5 are agreed (2026-10-02). Implementation in progress (1a). It covers step 1 of `devplan_datastructures.md` §7: DS3 (locations and scale mapping), DS17 (derived variables resolved at read), DS5 (output locations), DS11 (validation) and DS16 (conventions).
Branch `data_structure_api`, written against `1bf8356`. Code starts only after this note is agreed. The open points are in §8.

Decisions this note builds on: D2 (split `scale=` into location / scale / mapping), D3 (MTG optional, write state variables after every call), D8 (identity through `entity_ids`), D9 option A (defaults from `state_variable_type`), D10 (derived variables lazy at read).

## 1. Where we start from (code facts)

- **Declaration.** `declare(..., scale=, edge_mapping=, on_grow=)` (`component.py:19`) stores one `scale` key. It holds either a location string (`"node"`, `"edge"`, `"cell"`, `"scalar"`) or an MTG scale int.
- **Two parallel interpreters of `scale`.**
  - `FunctionalComponent._auto_declare_on_ds` (`component.py:~310`) registers fields, but only grid fields at `"cell"` / `"scalar"`. Graph fields with `scale="scalar"` are skipped.
  - `_declared_locations` (`decorator.py:208`) classifies them again for the solver.

  Both map **every bio-scale int to `"node"`**, unless `edge_mapping` is set (then `"edge"`). A coarser scale (e.g. Organ when the nodes are SubOrgans) is therefore broadcast to the nodes and stored there: it never exists at its own scale.
- **Store.** `VariableStoreMixin` (`data_api.py:259`): `register` / `get` (live view) / `set` (in place) / `alias` / `derive` / `refresh`. Locations on `MPGDataStructure` are `node`, `edge`, `scalar`, and one per bio scale coarser than the nodes (`_coarse_scale_names`). `_map` knows node→edge (proximal / distal / mean), node→coarse (sum / mean / weighted_mean), coarse→node (broadcast), and any→scalar / scalar→any.
- **MTG sync.**
  - The MTG is read at registration (`_mtg_to_node_array` / `_mtg_to_edge_array`). Parameters are re-read before each graph solve (`_refresh_from_bio_scale`).
  - State variables are written back by `write_back_to_mtg` **only at the end of a graph solve** (`decorator.py:739`), so `@rate` results of a component without a graph system never reach the MTG.
  - Write-back goes to the node vids, whatever the declared scale (Q2 finding).
- **Derived variables** are recomputed only by `refresh()`, which `pull_available_inputs` calls at the start of the receiver's `__call__` (`component.py:150`). Between two refreshes they can be stale.
- **Output locations are guessed by size:**
  - graph outputs: `location="node" if arr.size == n else "edge"` (`decorator.py:750`);
  - undeclared step outputs take the location of the step's first input (`legacy_functor.py:~62`);
  - filtered evaluators slice every argument whose `shape[0] == entity_size` (`decorator.py:466`).

  All three are wrong when `n == m`, or when the first input is a scalar.
- **Silent zeros:**
  - `_read_array` returns zeros for a missing name (`decorator.py:256`), so a typo in an equation argument solves with zeros;
  - `_type_mask` ignores a missing filter variable (`decorator.py:288`), so the filter silently selects everything;
  - `@boundary_condition(location="edge")` is accepted, but always masked over nodes.

## 2. Declaration API (DS3)

```python
concentration: float = state_variable(..., scale=scales.SubOrgan)                       # stored at its scale (= the nodes)
carbon_pool:   float = state_variable(..., scale=scales.Organ)                          # stored at location "Organ"
K_axial:       float = parameter(..., scale=scales.SubOrgan, location="edge", mapping="mean")
k_organ:       float = parameter(..., scale=scales.Organ, location="node", mapping="broadcast")  # Organ value, used per node
axial_flux:    float = state_variable(..., location="edge")                             # solver-only, no MTG property
total_N:       float = state_variable(..., location="scalar")
```

**Three keys:**
- **`scale`** (MTG scale int, a live `ScalesConfig` reference, or None): where the MTG property lives. Values are read from, and written back to, the vertices of that scale. None means a solver-only variable without an MTG property.
- **`location`** (`"node"`, `"edge"`, `"scalar"`, `"cell"`, or a scale name): where the DataStructure stores it, i.e. the shape the equations receive. **Default: the location of `scale`.**
  - the node scale → `"node"`;
  - a coarser scale → its name, e.g. `"Organ"`;
  - `Connection` → `"edge"`.
- **`mapping`** (with `weight=` when needed): how values go between `scale` and `location`, when they differ.
  - **down** (coarse → node): `broadcast`;
  - **up** (node → coarse): `sum`, `mean`, `weighted_mean`;
  - **node → edge**: `child`, `parent`, `mean`.
  - **When `mapping` is omitted** and `scale` ≠ `location`, the D9 table applies:

    | Source type | Up | Down |
    |---|---|---|
    | extensive | `sum` | raise |
    | intensive | `mean` | `broadcast` |
    | massic_concentration | `weighted_mean`, with a weight to give | `broadcast` |
    | missing | raise | raise |

  - node → edge has no default: `mapping` is required.

**Location names.** The scale name of the nodes is accepted as a synonym of `"node"` (e.g. `"SubOrgan"` when `from_scale=SubOrgan`), and `"Connection"` as a synonym of `"edge"`. Declarations written with scale names then keep their meaning when the node scale changes (DS8 anatomy mode).

**Edge mapping names.** `proximal` / `distal` become `child` / `parent`, which say what they do: `proximal` currently takes the **child's** value, which DS16 was going to document. The old names stay accepted for one release, with a `DeprecationWarning`.

**Legacy forms**, accepted for one release (D2):
- `scale="node" | "edge" | "scalar" | "cell"` → `location=` the same, no MTG property;
- `scale=<node scale>` → unchanged;
- `scale=<int>, edge_mapping=m` → `location="edge"`, `mapping=m`.
- **One deliberate change:** `scale=<coarser int>` without `location` now means "stored at that scale" instead of "broadcast to the nodes" (§8, N1). Code that wants the old behaviour writes `location="node", mapping="broadcast"`.

**One interpreter.** A single function, `resolve_declaration(field, ds) -> VariableSpec(location, scale, mapping, weight, kind, on_grow, default)`, replaces `_auto_declare_on_ds`'s branches and `_declared_locations`. The registration, the snapshot, the write-back and `assert_component_couplable` all use it. `VariableSpec` is recorded in the store's metadata: `_var_meta[name]` gains `scale`, `mapping`, `weight` and `kind` (= `state_variable_type`). DS20's repartition will need `kind` later.

**Mapping errors at declaration.** The following raise when the component is constructed, naming the field:
- an unknown location;
- `mapping` given when `scale` = `location`;
- a missing weight;
- an extensive variable going down without a mapping;
- `mapping="mean"` on an edge **state variable**, which cannot be written back (no unique owner; it is silently skipped today).

## 3. MTG reading and write-back (DS3, D3)

- **Read**, at registration, and at `refresh_from_mtg()` for parameters: the MTG property at `scale` goes through `mapping` to `location`. One function replaces `_mtg_to_node_array`, `_mtg_to_edge_array` and `_keys_at_scale`:
  - `ds.read_mtg(name, spec)`;
  - its fast path is kept for ArrayDicts keyed by the node vids;
  - it raises on partial coverage, as today.
- **Write-back**, of `state_variable` fields with a `scale`: the inverse mapping from `location` to `scale`.

  | `location` → `scale` | Written |
  |---|---|
  | same | as is |
  | node → coarse (variable stored at the nodes, from a coarse property) | the D9 up mapping: `sum` for extensive, `mean` / `weighted_mean` for intensive. A broadcast variable written back is aggregated, not copied from one arbitrary node. |
  | coarse → coarse (stored at its own scale) | as is, at the vids **of that scale** (fixes the Q2 finding) |
  | edge, `child` / `parent` | at the owner endpoint |

- **When:** at the end of **every** component call (`Component.__call__`, after the Choregrapher steps), instead of only after a graph solve (D3; §8, N4).
  - The graph-solve write-back is removed, and the parameter refresh before each solve is kept.
  - DS4 (step 4) will later make the policy selectable: write after every call, or never (no MTG).
- **Cost:** one vectorised assignment per written variable on the fast path. Nothing is written for solver-only variables (`scale=None`).

## 4. Derived variables resolved at read (DS17, D10)

**Write counters.** `VariableStoreMixin` keeps `_writes[name]`, an int incremented by:
- `register`;
- `set` (which every framework write already uses: the Functor, `inject_result`, the graph outputs, the Coupler, `apply_input_tables`);
- `update_topology` (all variables).

**Freshness.**
- Each derived spec records `_computed_from = {source: _writes[source]}` at its last computation.
- `get(name)` on a derived variable (aliases resolved first) walks `_derivation_order([name])`, and recomputes each derived variable whose recorded counters differ from the current ones, in dependency order. The values are written in place, so views stay valid.
- `get` on a non-derived variable costs one extra dict lookup.

**Rules:**
- **`set()` on a derived variable raises** (§8, N3). Only the framework recomputes it. Today a receiver writing to its own input would have its write silently overwritten at the next refresh.
- `refresh(name)` keeps its meaning (force a recomputation). `pull_available_inputs` becomes a freshness check, a no-op when nothing changed. It stays the hook that DS10's sub-steps will use.
- **Views:** a view returned by `get` on a derived variable is up to date **at the time of the `get`**. Code holding a view across another component's writes must call `get` again. The Functor and the snapshot already call `get` at every step and every solve.
- **Unsupported:** writing through a view, `ds.get(x)[...] = v`, does not bump `x`'s counter, so variables derived from `x` would stay stale. This is documented (DS16), and `validate(strict=True)` (§6) detects it in tests by comparing a checksum of each source with the one recorded at the last derivation.
- A variable derived from another DataStructure (the soil, through the Coupler) is not concerned: the Coupler writes with `set`.

## 5. Output locations (DS5)

1. **Declared outputs** take their location from their declaration (§2), through `resolve_declaration`. This is the normal case: outputs coupled to other components are declared fields.
2. **Undeclared outputs** must give a location:
   - graph outputs: `@graph_output(name, location="node" | "edge")`;
   - extra step outputs: `@rate(location=...)` / `@state(location=...)`, or a location per output for steps with several outputs: `@rate(locations={"x_flux": "edge"})`.

   **Transition:** when it is missing, the location is inferred from the shape only if exactly one location has that shape, with a `DeprecationWarning`. When several locations match (n == m, or a coarse scale with n entities), it raises.
3. **Filtered evaluators** slice an argument by the node or edge mask according to its **resolved location**, not its length. A scalar or a coarse-scale argument is passed whole.

## 6. Validation and failure modes (DS11)

- **`ds.validate()`** checks that:
  - every registered array has its location's shape;
  - every alias resolves, with no cycle;
  - every derived variable's sources and weight exist at its recorded source location;
  - every variable has metadata;
  - `topology_version` and the entity maps agree.

  It is called by `assert_component_couplable`, at the end of `declare_data_and_couple_components`, and by tests. `validate(strict=True)` adds the checksum check of §4.
- **Missing variables raise instead of acting as zeros:**
  - `_read_array` raises `KeyError` with the solve name, the argument name, and the available variables at that location. The integrate-field amounts (`{fn}_amount`) are registered when the component is declared, instead of being read as zeros at the first solve.
  - A missing filter variable in `@node_balance` / `@edge_law` / `@boundary_condition` / step filters raises.
  - `@boundary_condition(location="edge")` raises `NotImplementedError` when the class is defined, until DS6.
- **`from_scale`.** Making it required would break about thirty constructions in the tests for no gain. Instead, `MPGDataStructure(g)` **infers** it from the populated graph (the scale of the vertices that the Compartments' `vertex_id` points to). It raises only when the graph is not populated and no `from_scale` is given. `update_topology()` can then never run without one.

## 7. Conventions page (DS16)

Add `docs/conventions.md`, linked from the README and from `downstream_migration.md`. It covers:
- the incidence sign: +1 at the parent end of an edge, −1 at the child;
- the local order: Compartment post-order from population, not sorted. Use `entity_ids` / `index_of`, never positions (D8);
- grids: axes are `(x, y, z)`, and `locate` / `cell_centers` work in that order;
- the edge mappings `child` / `parent` / `mean` (formerly `proximal` / `distal`), and which endpoint owns an edge variable;
- the plant scale is the `"scalar"` location;
- `scale` / `location` / `mapping`, with the D9 defaults table;
- `previous()`: the state at the start of the current solve;
- `on_grow`, and later the DS20 repartition;
- derived variables: up to date at `get`, read-only, and no writes through views.

## 8. Points to agree

- **N1, coarse-scale default.** `scale=<coarser int>` without `location` changes meaning, from "broadcast to the nodes" to "stored at that scale". It is the fix for D4.3, and downstream packages must migrate their declarations anyway. **Agreed.** Change it now, without a deprecation period. In this repo it affects no test: UC1-Organ uses Organ as the node scale. A clear error appears when an equation receives an Organ-length array where it expects nodes.
- **N2, rename** `proximal` / `distal` to `child` / `parent`, old names accepted for one release. **Agreed.**
- **N3, derived variables read-only:** `set()` on a derived variable raises. **Agreed.**
  - *Derived variables* are the inputs that the translator fills from other components' outputs, through a link with a factor, a sum of several sources, a formula or a scale change (`ds.derive` in `_couple_on_data_structures`). A component may also call `ds.derive` itself, under the same rule.
  - Identity links (same variable) and aliases (another name for the same array) are not derived variables: writing them writes the provider's variable, as today.
- **N4, write-back after every component call,** in step 1, rather than waiting for DS4. **Agreed.** It fixes `@rate`-only components, and DS4 only adds the policy choice later.
- **N5, scale names as location names.**
  - **Today:**
    - the scale the graph is built from (`from_scale`, e.g. SubOrgan) is stored at the location called `"node"`;
    - only the coarser scales are called by their name (`"Organ"`, `"Plant"`, …).

    So a SubOrgan variable is at `"node"`, and an Organ variable is at `"Organ"`.
  - **The problem, with the DS8 anatomy mode.** The nodes become the Compartments of the anatomies, and SubOrgan becomes a coarser scale. Then:
    - `"node"` would mean Compartment;
    - SubOrgan variables would move to a location called `"SubOrgan"`.

    Any code or translator link written with `location="node"` to mean "per segment" would silently change meaning: it would then receive one value per cell.
  - **Proposal:** accept the scale name everywhere a location is given (`location="SubOrgan"`, translator links, `ds.get` / `ds.mask` arguments):
    - without an anatomy, `"SubOrgan"` resolves to `"node"`;
    - with an anatomy, it resolves to the SubOrgan location.

    Code written with scale names keeps its meaning in both modes. `"node"` stays for solver-level variables, which genuinely follow the graph's nodes whatever they are.
  - **Scope:** the conventions page recommends scale names for biological variables. Declarations with `scale=scales.SubOrgan` already get the right location by default (§2), so N5 only matters where a location is written explicitly.
  - **Agreed (2026-10-02),** on the user's condition, which holds:
    - `"node"` and `"edge"` are the entities of the graph built by the MPG traversal (`populate_graph`, and DS8's assembly later);
    - model declarations anchored on biological scales are independent of them. They are resolved against the graph when the component is bound to its DataStructure, never the reverse.

## 9. Implementation order and tests

Each step lands with its tests. The existing suite passes at every step: 516 tests, with the wrapper `[contract]` tests and UC1–UC4 **unchanged**, except for in-repo declarations migrated to the new keys where they use `edge_mapping` (kept working through the legacy forms anyway).

| Step | Content | New tests |
|---|---|---|
| 1a | `VariableSpec` + `resolve_declaration`, used by registration and the snapshot. Legacy forms are mapped, the D9 defaults applied, and mapping errors raised. | `test/data_api_tests/test_declarations.py`: every legacy form gives the same location as today, except N1; the D9 defaults table; every declaration error |
| 1b | Coarse locations (N1), location synonyms (N5), `child` / `parent` (N2), and `read_mtg` / write-back with the inverse mapping, at every call (N4) | `test_scale_mapping.py`: Organ-scale pool read, stored and written back at Organ vids; broadcast parameter written back as a mean; extensive sum; edge `child` / `parent` round trip; a `@rate`-only component reaching the MTG |
| 1c | Write counters, lazy derived variables, read-only derived (N3), the `pull_available_inputs` check | `test_lazy_derived.py`: stale after a source `set`, fresh at `get`; chains of derivations; aliases of derived variables; no recomputation when nothing changed (call counts); `set` on derived raises; the existing composite tests unchanged |
| 1d | Output locations, filtered slicing by location | `test_output_locations.py`: n == m graph with node and edge outputs; undeclared output without location raises when ambiguous and warns when unambiguous; a scalar argument in a filtered evaluator |
| 1e | `validate()`, raising on missing names and filters, edge BCs rejected, `from_scale` inferred | `test_validation.py`: each failure mode raises with its message; `validate(strict=True)` detects a write through a view |
| 1f | `docs/conventions.md`, CHANGELOG, `downstream_migration.md` §2 updated for the new keys | a documentation test: every location and mapping name in the page exists in the code |

**Risks:**
- Tests may rely on silent zeros (1e) or on size inference (1d). They will show up as failures and be fixed one by one in the tests, never by keeping the silent path.
- Lazy derivation adds a check to every `get` of a derived variable. 1c includes a micro-benchmark on a 20 000-node DataStructure with ten derived variables, so that the cost is measured, not assumed.
