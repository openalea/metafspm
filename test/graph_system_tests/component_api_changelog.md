# Component API Changelog

Changes made to the component and data-structure API as use cases are updated.
Each entry lists what changed, where, and what must be updated in remaining UCs.

---

## UC1 — NitrogenAxialTransport  (applied)

### 1. `FunctionalComponent` now requires a DataStructure argument

**File:** `src/openalea/metafspm/coupling/component.py`

- `FunctionalComponent` gains a `data_structure: Optional[DataStructure]` field
  (default `None`, enforced non-None in `__post_init__`).
- `__post_init__` derives `self._graph_view` and `self.props` from the
  DataStructure — no more manual assignment in tests.
- Every `@dataclass` subclass of `FunctionalComponent` (UC1–UC4) must be
  instantiated as `MyModel(data_structure=ds)` instead of `MyModel()`.

**Before (old pattern):**
```python
model = NitrogenAxialTransport()
model.props = {...}
model._graph_view = graph
```

**After (new pattern):**
```python
ds = MPGDataStructure(g, scale=6)
ds.set_node_property("concentration", c_init)
ds.set_edge_property("K_axial", k_vals)
model = NitrogenAxialTransport(data_structure=ds)
```

### 2. `MPGDataStructure` — anchor filtering and new topology method

**File:** `src/openalea/metafspm/data_structure/data_api.py`

- `_build_index_map()` overridden in `MPGDataStructure` to exclude vertices
  where `isanchor=True` (scale anchor nodes inserted by `MPG.__init__`).
- `to_graph_view()` rewritten: builds `GraphView` directly from parent-child
  pairs at `self._scale` (via `self.edges()`), using 0-based integer edge IDs.
  **Old** used `edge_scale = self._scale + 1` which is wrong for SubOrgan transport.
- `to_props_dict()` added: converts `_node_data`/`_edge_data` numpy arrays to
  the `{name: {id: value}}` dict format the decorator machinery expects.
  Node props are keyed by MTG vertex ID; edge props by 0-based edge index.

### 3. Edge identity convention change

**Old:** edge properties in `props` were keyed by MTG vertex IDs of connection
vertices at a dedicated edge scale (Connection scale 10, from anatomy MPG).

**New:** edge properties in `props` are keyed by **0-based integer index**
matching the order of `MPGDataStructure.edges()` (parent-child pairs sorted by
child node ID).  This is consistent with `GraphView.edge_ids = np.arange(m)`.

**Impact on UC2–UC4 setups:**
- Any setup function that does `props["some_edge_field"][vid]` using a Connection
  vertex ID must switch to `ds.set_edge_property("some_edge_field", array)` before
  constructing the component.
- `GraphView` edge_ids will be `[0, 1, ..., m-1]` instead of MTG vids.

### 4. Test topology change (UC1)

**Old:** `_cell_chain_graph()` — 3 cell nodes + 2 symplastic edges (from anatomy seedling MTG).

**New:** `MPGDataStructure(g, scale=6)` from `generate_simple_mpg_seedling()` —
14 SubOrgan nodes + 13 axial edges (tree: 2 stem elements, 6 leaf elements,
6 root segments, branching at `internodeelement`).

`assert (n, e) == (14, 13)` replaces the old `assert (n, e) == (3, 2)`.

Physics tests use `np.random.default_rng(42)` for reproducible gradients;
assertions check `np.all(c > 0)` and `np.any(|q| > 0)` rather than scalar
edge indices.

---

## UC2–UC4 — pending updates

The following changes are required for UC2, UC3, UC4:

1. **Component base class**: Change `Component` → `FunctionalComponent` as the
   base class in all UC component declarations (or keep `Component` if that's
   still appropriate — check whether `Component` also needs the DataStructure).

2. **Constructor call**: Replace `MyModel()` with `MyModel(data_structure=ds)`.

3. **Setup function**: Replace manual `props`/`_graph_view` assignment with
   `ds.set_node_property(...)` / `ds.set_edge_property(...)` calls before
   constructing the model.

4. **Graph source**: Replace `_cell_chain_graph()` / `_anatomy_graph()` with
   an `MPGDataStructure` derived from `generate_simple_mpg_seedling()` at the
   appropriate scale, OR keep using `GraphView.from_mtg_subset` if the anatomy
   graph (UC3) is specifically needed.

5. **`_boundary_ports`** (UC3): `FunctionalComponent.__post_init__` does not yet
   handle boundary ports.  UC3 will need either:
   a. `ds.to_graph_view(boundary_ports=ports)` called explicitly, or
   b. `_boundary_ports` set after construction (as before).

6. **Edge ID keys** (UC2 `WaterMunchTransport`): UC2 has no edge unknowns but
   does use edge-located parameters (`K_xylem`, `K_phloem`).  These must be
   registered via `ds.set_edge_property(...)` before construction.
