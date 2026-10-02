# DataStructure API stabilisation plan

**Goal.** Settle the `DataStructure` API, and its use by the graph-system declaration tools (`@graph_system`, `@node_balance`, `@edge_law`, `@boundary_condition`, `@graph_output`, the Choregrapher steps), so that downstream components and decorators do not have to change their API afterwards.

**How to use this file.** Your questions and answers are kept verbatim. Updated 2026-10-01 with your answers to D1–D8 and Q4–Q5. Each answer cites the code (`release2026` at `b7f95ad`) and ends with what should change. The DS-items (§5) and the decisions (§6) are what I added; edit them freely, and record answers inline.

---

## 1. Your questions

### Q1a: Does the graph_system decorator implicitly only works on a component which whould have received a MPG data structure?

**Answer: in practice yes. MPG is the only data structure that can host a graph system today.** Nothing in the decorator mentions MPG, but it needs two things that only `MPGDataStructure` provides together:

1. **A graph view.** `GraphSystemBuilder.build` starts from `instance._graph_view` (`decorator.py:328`). `FunctionalComponent._graph_view` is `ds.to_graph_view()` when the data structure has one, and `None` otherwise (`component.py`, `_graph_view` property). `to_graph_view` exists only on `MPGDataStructure` (`data_api.py:1279`). `GraphView.from_mtg_subset` also builds views (`:103`), but no data structure uses it.
2. **A live variable store.** `_live_ds` requires `get` / `register` (`decorator.py:~248`). Only `MPGDataStructure` and `ArrayDataStructure` have the `VariableStoreMixin`. `LegacyMPGDataStructure` / `MTGDataStructure` have neither the store nor a graph view.

What happens elsewhere:
- **`ArrayDataStructure` (soil grid):** a `FunctionalComponent` works for Choregrapher steps, but a graph system fails as soon as it reads `gv.node_ids` from a `None` view.
- **`MultiGridDataStructure`:** it has no variable store, so it cannot even host a component.

**What should change: DS1**, a topology contract that graph systems target instead of MPG. Grids could expose their cells as nodes and their faces as edges, so soil transport can be declared with the same decorators.

### Q1b: does the specialized DataStructure for MPG actually use multiscale traversal methods like pre_order_mpg if this was needed by an iteration that would occur outside of the graph_system's usage of the graph_view?

**Answer: no.** `MPGDataStructure` uses the MPG's multiscale machinery only indirectly:
- The node set comes from `populate_graph(from_scale)`, which walks `post_order_mpg`. You call it before construction (`data_api.py:786`), and `update_topology` calls it through `repopulate_graph` (`:1243`).
- The node order is therefore the Compartment post-order (tips first). That order is an accident of population, not a guaranteed traversal order. The earlier `node_local_index` bug came from exactly that ambiguity.
- The scale operators and the scale-aware MTG mapping use `complex_at_scale` (`:1006`, `:1086`) to build node → coarse-scale membership. They never use a pre/post-order traversal.
- `pre_order_mpg`, `post_order_mpg`, `integrate_at_scale` and `average_at_scale` live on `MPG` only, and work on MTG properties (vid-keyed ArrayDicts), not on DataStructure arrays.

So an algorithm that must walk the topology in order would today either:
- iterate the MTG with `pre_order_mpg`, then map vids to local indices by hand (`ds._vid_to_idx`, a private attribute); or
- rely on the local order, which is not guaranteed.

Examples of such algorithms: cumulative sums from tip to base, sequential axial transport, growth propagation.

**What should change: DS2**, traversal orders exposed by the data structure as **local-index arrays** (`ds.order("pre" | "post", scale=None)`, plus `parents()`, `children()`), cached per `topology_version`. The iteration then happens on DS arrays and never touches vids or the MTG.

### Q2: About the graph_view, is it dynamically retreived at the relevant scale of the variables solved by the graph_system? And do I understand right it populates node and edge scales, solves, and then writes back on relevant output scale?

**Answer: partly.**

**1. The view is not built per variable scale.** A data structure has **one** node scale: the `from_scale` given to `populate_graph`. Every node is one Compartment per `from_scale` vertex, and every edge one Connection per parent-child pair. `to_graph_view()` always returns that single graph.
- The view is rebuilt only when the topology changes (`topology_version`). It carries the boundary ports set on the component.
- A graph system solves on that one graph whatever the declared `scale` of its fields. To solve at Organ scale, you need a second DS populated at Organ scale. UC1-organ does exactly that.

**2. The cycle, as implemented:**
1. **Declaration.** At `FunctionalComponent` construction, `_auto_declare_on_ds` registers every field that has a `scale`:
   - node or edge, by `_declared_locations` (`decorator.py:~197`): Compartment → node, Connection → edge, bio scale + `edge_mapping` → edge, bio scale alone → node;
   - its initial values come from the MTG property when one exists. A coarser bio scale is broadcast through `complex_at_scale`, using the scale-aware mapping of WD.P.
2. **Each solve:**
   - bio-scale **parameters** are refreshed from the MTG (`_refresh_from_bio_scale`, with their declared scale);
   - the required variables are **copied** from the DS (`_snapshot`); the unknowns' initial guess and previous state are copies too;
   - the solve runs;
   - unknowns and outputs are written **in place** into the DS (`inject_result`, graph outputs);
   - bio-scale **state variables** are written back to the MTG (`write_back_to_mtg`).
3. **Write-back scale: not what you expect.** `write_back_to_mtg` writes at the **node vids** (`ds.write_node_to_mtg(name, ds.get(name))`), whatever the declared scale.
   - A state variable declared at a coarser scale (e.g. `scale=Organ` on a SubOrgan DS) was broadcast down at declaration, and is written back at the SubOrgan vids. Its Organ value is neither aggregated back nor written at the Organ vids.
   - Outputs (`@graph_output`) and Choregrapher `@rate` / `@state` results stay in the DS; they reach the MTG only if they are bio-scale state variables of a component that ran a graph system (write-back runs inside `_invoke_graph_system` only).
   - Output locations are guessed from the array size (`size == n` → node), which is ambiguous when n == m.

**What should change:**
- **DS3:** an explicit variable location with a declared **up/down mapping** between the variable's scale and the solver nodes, used symmetrically when reading and when writing back.
- **DS4:** a single, explicit MTG synchronisation policy.
- **DS5:** declared output locations.

### Q3: If I want to apply a boundary condition of leaves, e.g. an air water potential, does this call for the creation of additionnal "environment" nodes in the graph_view? Since there is no explicit coupling with a grid like for soil.

**Answer: no extra node is needed.** Three mechanisms exist today, with limits.

1. **`@boundary_condition("node", "dirichlet" | "neumann", field=, filters=)`** (`decorator.py:165`, assembled at `:482-564`):
   - Dirichlet replaces the residual rows of the filtered nodes (`result[idx] = value`); Neumann adds a source.
   - The filter is a node variable, e.g. `{"is_leaf": [1]}` or a label code.
   - For leaves: Dirichlet imposes ψ_leaf = ψ_air. That is usually too strong: it ignores the stomatal/boundary-layer resistance.
2. **Robin boundary ports** (`BoundaryPort(node_id, value, weight)`, `data_api.py:66`, UC3). Each port is a **virtual environment node**: a column of `graph.boundary_incidence` (built by `to_graph_view(boundary_ports=)`).
   - The equations add `B_b · diag(w) · (B_bᵀ ψ − v)`, i.e. a conductance `w` to a fixed potential `v`. This is the physically right model for transpiration: `w` is the stomatal plus boundary-layer conductance, `v` the air water potential.
   - Limits:
     - ports are **frozen dataclasses** holding constant `value` / `weight`, so a changing air potential or stomatal conductance forces a new port set and a new view;
     - they are keyed by **vid** and attached to the component by hand (`model._boundary_ports`), not selected by a rule;
     - after growth, new leaves get no port;
     - the decorator does not assemble Robin terms itself: each model rewrites them in its equations (UC3), and `EquationContext.boundary_values/weights` have no test.
3. **Explicit environment nodes** (an extra "atmosphere" Compartment in the MPG). Possible, but it mixes the environment into the plant topology and the Choregrapher/MTG write-back. Not recommended.

**What should change: DS6**, first-class boundary sets. A boundary set is declared by a **node selection rule** (label, variable, predicate). Its **value and weight are DS variables**, read live at each solve (e.g. `air_water_potential` in the scalar store, `leaf_conductance` a node variable). It is refreshed with the topology. A `kind="robin"` option has the decorator assemble the term, and the same mechanism serves the soil side when a model wants a Robin coupling rather than a Coupler.

---

## 2. Current API map

| Concern | Where it lives today | Stable? |
|---|---|---|
| Variable storage, locations, aliases, derived variables, scale operators, export | `VariableStoreMixin` + `MPGDataStructure` / `ArrayDataStructure` | mostly (WD.P, WD.3) |
| Topology (nodes, edges, incidence) | `MTGDataStructure` / `MPGDataStructure` + `GraphView` | no: graph only on MPG, two conventions merged recently |
| Scales | MPG (`complex_at_scale`), `_membership`, the `scale` metadata | no: one node scale per DS, write-back ignores scale |
| Location of a field | `declare(scale=...)` overloaded: `"node"` / `"edge"` / `"cell"` / `"scalar"` / bio int (+ `edge_mapping`) | no: one keyword carries two concepts |
| Boundary | `BoundaryPort` (frozen, vid-keyed) + `@boundary_condition` (node-only filters) | no |
| Traversal | MPG methods on MTG properties | not exposed on DS |
| Growth | `update_topology` + `on_grow` | yes, but `from_scale` is optional and must be set |
| MTG sync | `_auto_declare` (read), `_refresh_from_bio_scale` (parameters), `write_back_to_mtg` (state variables, after graph solves) | no |

---

## 3. Design principle

Components and decorators should depend only on this contract:

1. **Named variables with an explicit `Location`:** entity kind, scale, and mapping from and to the solver entities.
2. **A topology view:** nodes, edges, incidence, boundary sets and traversal orders, in local indices.
3. **A lifecycle:** `topology_version`, growth, sync.

MPG, MTG, grids and multigrids are specialisations that implement it. Everything below either fills a gap in that contract or removes a place where downstream code reaches past it (vids, `_idx_to_vid`, MTG properties, `_boundary_ports`).

---

## 4. Behaviour changes already in place (for reference)

These are done: in-place writes, aliases, derived variables, scale operators (sum / mean / weighted_mean / broadcast / proximal / distal), the scalar store, growth carry-over with `on_grow`, `topology_version`, `(x, y, z)` grids with `locate`, the export API, the Coupler / Transport, live reading and `previous()`.

---

## 5. DS-items to stabilise (proposal)

- [x] **DS1 Topology contract, graph systems on any DataStructure.** (steps 3c–3d; MultiGrid later)
  - Define `TopologyView` (what `GraphView` already is: `node_ids`, `edge_ids`, `tail`, `head`, `incidence`, `boundary_incidence`), requested through one method, `ds.topology(boundary=...)`.
  - Implement it for `MPGDataStructure` (existing), for `ArrayDataStructure` (cells as nodes, faces as edges, with `face_area/dx` as a geometric factor) and later for `MultiGridDataStructure`.
  - `@graph_system` then solves soil transport too. Its equations use `B`, as the plant ones do.
  - Decide how grids expose face geometry to equations (D1).
- [x] **DS2 Traversal orders on the DS:** (step 2a) `ds.order("pre" | "post")`, `ds.parents()` (local index of the parent, −1 at the root), `ds.children()` (CSR), `ds.roots()`, `ds.tips()`, all in local indices and cached per `topology_version`.
  - The MPG implementation derives them from Connections, independently of the post-order accident of population.
  - Replace the private `_idx_to_vid` / `_vid_to_idx` accesses in tests and components with `entity_ids("node")` and `index_of(vids)`.
- [x] **DS3 Explicit `Location` for fields, symmetric scale mapping (D2 decided: split).** Split `declare(scale=...)` into:
  - `location`: `"node"` / `"edge"` / `"cell"` / `"scalar"`;
  - `scale`: a bio scale (live ScalesConfig reference);
  - `mapping`: how the variable maps to the solver entities — `broadcast` / `sum` / `mean` / `weighted_mean` (`weight=`), and for edges the `edge_mapping` values `proximal` / `distal` / `mean`.

  Rules:
  - At declaration and refresh, MTG values at `scale` are mapped **down** to the solver entities.
  - At write-back, the inverse mapping aggregates **up** (e.g. `sum` for extensive, `mean` for intensive) and writes at the vids **of that scale**.
  - Also rename or document the inverted `proximal`, which means "owned by the child".
  - Keep `scale="node"` etc. accepted for one release (D2).
  - **Answers D4.3.** A field declared at a coarser bio scale (e.g. an Organ-scale carbon pool) is registered **at that scale's location** (`"Organ"`, which already exists since WD.3), not broadcast to the nodes as today. The coupling then maps between locations explicitly (DS18). Today such a field is always stored at node location, so a cross-scale input "exists" only as a broadcast copy.
- [x] **DS4 MTG synchronisation policy (steps 1b, 4a) (D3 decided: MTG optional).** One explicit policy per component:
  - **read:** at construction, and for parameters before each call;
  - **write:** state variables after **every** `Component.__call__`, not only after graph solves; outputs on request.
  - Or no MTG at all, for DS-only models.

  It is implemented once in `FunctionalComponent.__call__`. It removes the silent case where `@rate` results never reach the MTG, and makes the MTG optional (D3).
- [x] **DS5 Declared output locations.** `@graph_output(name, location="node" | "edge")`, plus the same option on steps whose outputs are not declared fields. This replaces the `size == n` guess, which is ambiguous when n == m.
- [x] **DS6 Boundary sets (answers Q3).** (step 2e, with UC5) `boundary_set(name, select=..., value="air_water_potential", weight="leaf_conductance", kind="robin" | "dirichlet" | "neumann")`, declared on the component.
  - `select` is a label, a variable (> 0), or a callable over DS variables.
  - Value and weight are DS variables or constants, read live at each solve.
  - Ports are rebuilt when the topology changes.
  - For `kind="robin"`, the decorator assembles `B_b·diag(w)·(B_bᵀx − v)` into the field's balance, and supplies the matching Jacobian term.
  - This unifies `@boundary_condition` filters and `BoundaryPort`. `BoundaryPort` stays the internal representation; `_boundary_ports` set by hand is deprecated.
  - Also fix: `location="edge"` BCs are accepted but always masked over nodes (`decorator.py:~503`), and a missing filter variable silently deactivates a BC. Both should raise.
- [ ] **DS7 Several scales per plant (D4 decided: one topology per component, several components per DataStructure).**
  - The plant's components share **one** DataStructure, as they shared the central MTG, and as the WD.4 coupling already does. Each graph system solves on that DataStructure's topology, and the other equations of the component are operations at specific scales (DS3 locations, DS18 mappings).
  - A multiscale graph (anatomy within segments, plus segment connectivity) is **one** topology built at the Compartment/Connection scales (DS8), not several topologies.
  - The MPG ↔ MPG Coupler (two DataStructures of one plant) is needed only when a plant really uses two DataStructures. It is deferred until a use case asks for it.
- [x] **DS8 Multiscale graph assembly (step 2f; the repartition in anatomy mode waits for GRANAP as a StructuralComponent, A1): anatomies below, connectivity above (answers D4.2, Q6, Q9–Q11; priority raised).**
  - **The model (Q6).**
    - **Compartment / Connection** (below SubOrgan) hold each segment's anatomy graph. Within a cross-section, exchanges are Connection vertices whose `n_id_a` / `n_id_b` are Compartment vids (Q10).
    - **SubOrgan and coarser scales** hold the connectivity between segments.
  - **The MPG is the source of truth; the DataStructure is a dynamic interface to it (Q9, D11).**
    - Every edge of the solved graph exists in the MPG as a Connection vertex, including the junctions between segments.
    - **Populating** the MPG means running MPG methods:
      - `populate_graph`: one Compartment per segment;
      - `populate_graph_custom_connections` and its successor: junctions between anatomies.
    - **Building the graph view** means extracting nodes and edges with MPG methods: `array_filtering`, `n_id_a` / `n_id_b`, parent / complex relations. This is what `MPGDataStructure` does today in the one-Compartment-per-segment case.
  - **Junctions only join directly connected SubOrgans (Q11):**
    - a within-scale parent and child;
    - or, when the within-scale parent is missing, the tip of the complex parent (the `_tip_component` rule already used by `populate_graph`).

    Organ-scale information may decide *whether* or *how* to wire, but the pair of SubOrgans is always a direct MPG link.
  - **What has to change.**
    1. **Node identity.** `MPGDataStructure` keys nodes on `vertex_id`: the SubOrgan vid that `populate_graph` copies onto its Compartment. Anatomy Compartments are several per SubOrgan and carry no `vertex_id`.
       - *Anatomy mode:* the node key is the Compartment vid itself, and the SubOrgan owner is its MTG parent.
       - Edges join Compartment vids directly.
       - `entity_ids` returns those keys (D8). The DS3 scale locations and the WD.3 operators use the owner map (`parent`, then `complex_at_scale`).
    2. **Wiring rules as MPG methods.** Extend `populate_graph_custom_connections` with:
       - `match="all" | "nearest" | "equal"` on a property (e.g. vessel index);
       - a custom callable, `rule(g, suborgan_a, suborgan_b, compartments_a, compartments_b) -> pairs`, for the modeller's formalisms (e.g. angular sectors).

       Each generated Connection is marked, e.g. `edge_kind="junction"` against `"anatomy"`.
    3. **Growth without destroying anatomies (D12, Q13).** `repopulate_graph` deletes every Compartment and Connection and rebuilds them: correct for one Compartment per segment, but it would erase the anatomies.
       - In anatomy mode, the anatomies belong to the structural component that generates them (GRANAP, later, as a StructuralComponent; Q10), and they are never deleted by repopulation.
       - Only the junctions are rebuilt: preferably **incrementally**, for new SubOrgans only, so the existing Connection vids, and their variables, are kept.
       - Edge variables carry over by the endpoint pair `(n_id_a, n_id_b)` instead of `n_id_b` alone, which is not unique when a SubOrgan has several junctions.
       - **Conditional rewiring (Q13).** An existing SubOrgan's junctions are re-wired only when:
         - its **SubOrgan-scale neighbourhood** changed: a new or removed child, or another parent;
         - or its **anatomy** changed: its set of Compartments, or a label used by a wiring rule (differentiation).

         Every other junction is kept as is. Both conditions are detected by `update_topology` by comparing the MPG with the last extraction:
         - each SubOrgan's parent and children;
         - each SubOrgan's Compartment ids, and the rule properties' values on them.

         Structural components need no extra API for this.
    4. **Labels** (cell type, `edge_kind`) are categorical variables (DS12), usable in filters, rules and boundary sets.
  - **Owners in use:**
    - a segment's radial water uptake is the **sum** of its anatomy's boundary fluxes;
    - a SubOrgan metabolite concentration is **broadcast** to the symplastic Compartments of its cross-section (DS18, Q7).
  - **Validates on:** UC3 on a small synthetic anatomy per segment, generated in the test helpers (in this repo, since GRANAP is migrated after stabilisation):
    - xylem junctions matched by vessel index;
    - one growth step that adds a segment and keeps the old Connection vids and edge values;
    - then a sum to SubOrgan and a filtered broadcast from it.
- [x] **DS19 StructuralComponent contract (step 2b; the repartition after structural steps is DS20, step 2c) (answers the Q12 follow-up, Q14, Q15).** Today `StructuralComponent` is an empty class (`component.py:213`): no DataStructure, no scheduling.
  - **Reference: rhizodep's `RootGrowthModel`** (`rhizodep/root_growth.py`, read only):
    - its structural processes are steps scheduled like functional ones: `@potential @state potential_growth`, `@actual @state actual_growth_and_corresponding_respiration`, `@segmentation @state segmentation_and_primordia_formation`;
    - each is a **whole-MTG method without array arguments**: it loops over apices and segments with the MTG node API (`g.node(vid)`, `add_child`, attribute writes);
    - a component-level pass, `post_growth_updating`, runs at the end of `__call__`: distance from tip, root hairs, axis ids.

    GRANAP is expected to follow the same pattern.
  - **It edits the MPG directly.** It gets the shared DataStructure, `StructuralComponent(data_structure=ds)`, and edits through `ds.mtg` with the MPG's own methods: `add_child`, `add_component_with_topo`, the anatomy generation, pruning.
    - The DataStructure does **not** re-expose the edit methods (D11, D13).
  - **Scheduling (Q15, corrected by step 2's P2).** Its steps are decorated and placed in the Choregrapher rows like any FunctionalComponent step. The potential / allocation / actual / segmentation rows order the processes **within** the component. Components are called as a whole, in the composite's order, as today, and no interleaving across components is planned.
    - **Two step styles coexist (Q17).**
      - A step that names no variables is called **without arguments** and works on the MPG: for topology traversals and edits (segmentation, emergence), as in `root_growth.py`.
      - A step that names declared variables receives arrays, as functional steps do, for the vectorisable parts (e.g. thickening, root hairs). Its outputs are written to the DataStructure, then flushed to the MPG after the step, since structural outputs live in the MPG (Q14).
    - `post_growth_updating`-like passes become scheduled steps too (e.g. a `@postsegmentation` row), instead of code in `__call__`.
  - **Declared variables.** Its fields use the same `declare` metadata, so the translator, `assert_component_couplable` and the documentation treat it like any component:
    - **inputs** it reads, e.g. `C_hexose_root`, a functional state;
    - **structural outputs** it writes as MPG properties (Q14): length, radius, `x1 … z2`, `struct_mass`, labels, types.
  - **Sync around each structural step.** This is what makes option a enough:
    1. **before:** the framework writes the declared inputs from the DataStructure to the MPG, so every value the step reads is exact there;
    2. the step reads and edits the MPG;
    3. **after:** the framework re-reads the declared outputs from the MPG into the DataStructure, then applies the **repartition rules** (DS20) to every other registered variable.
       - If the MPG topology changed (vertices added or removed at the node scale, or anatomy changes), it calls `ds.update_topology()`. That extracts the topology, re-wires conditionally (DS8), re-reads the structural outputs, and carries the other variables over (Q16).
       - FunctionalComponents rebuild their graph views through `topology_version`, as today.
    - Detecting a topology change is cheap: an MPG modification counter, or the vertex count per scale. A pure elongation step therefore costs only the re-read.
  - **Validates on:** a test-helper growth component in this repo, a small rhizodep-like model with three steps:
    - potential elongation;
    - actual elongation, reading a functional carbon state;
    - segmentation, adding segments (and their synthetic anatomy for DS8).

    A FunctionalComponent runs after it in the same step. The test checks the inputs flushed before each step, the outputs re-read after, and carry-over at segmentation.
- [x] **DS20 Repartition of functional variables (step 2c) when structure changes (answers Q16; generalises rhizodep's `post_growth_updating`).**
  - **The reference** (`root_growth.py`, `post_growth_updating`, read only). For each new or growing vertex `v`, with parent `p`, and with `f = m_v / (m_v + m_p)` on the living structural mass:

    | `state_variable_type` | new `v` with mass > 0, or `v` elongating from zero mass | new `v` with zero mass (e.g. a lateral primordium) | existing `v` whose mass grew |
    |---|---|---|---|
    | massic_concentration | the amount `c_p · m_p⁰` is split: `c_v = amount·f/m_v`, `c_p = amount·(1−f)/m_p` | copied from `p` (or the grandparent) | diluted: `c_v ← c_v · m_v⁰ / m_v` (amount conserved) |
    | extensive | split: `x_v = f·x_p`, `x_p ← (1−f)·x_p` | 0 | unchanged |
    | non_inertial_intensive | copied | copied | unchanged |
    | non_inertial_extensive | `f·x_p`, `p` unchanged | 0 | unchanged |
    | descriptor | none | none | unchanged |

    Soil-boundary inputs are copied from the parent.
    - So **a lateral is not split when it is formed** (zero mass, `nil_properties=True`: it copies concentrations and holds no amounts). It is split from its parent **when it first elongates** from zero mass, like a new apex.
  - **In the framework.** The same table runs after every structural step, on **every registered variable** of the shared DataStructure. This uses each variable's `state_variable_type`, which the DataStructure records at registration (it now records only `default` / `on_grow`). This replaces `modules_to_update` and the per-module lists (`massic_concentration`, `extensive_variables`, …): the DataStructure knows them all.
  - **Nothing rhizodep-specific is built in (Q16).** The structural component declares, as **class attributes** (Q18), each entry accepting a callable for computed values:
    - the **partition weight**, e.g. `living_struct_mass` in rhizodep. Another model could use length, volume or a cell count;
    - the **previous weight** used for dilution and amounts, e.g. `initial_struct_mass + initial_living_root_hairs_struct_mass`;
    - the **active set**: which entities hold a functional system, as rhizodep's "focused" elements (`struct_mass > 0` with a label / type filter).

    Example: `partition_weight = "living_struct_mass"`, `previous_weight = ("initial_struct_mass", "initial_living_root_hairs_struct_mass")`, `active = {"struct_mass": ">0", "label": [...], "type": [...]}`.

    Without a declared weight, there is no split: new entities take `on_grow` (default or inherit), as today.
  - **The active set is a DataStructure-level mask for vectorised operations only (Q18).**
    - `ds.define_mask("active", rule)` is re-evaluated at each refresh and after each structural step. `ds.mask("active")` returns a boolean array at the rule's location.
    - **Vectorised steps** (functional `@rate` / `@state`, and array-style structural steps) compute on the active entities only. Values outside the mask are left unchanged. A step opts out with `where=None`, or chooses another mask with `where="name"`.
    - This replaces rhizodep's `focus_elements` and the Choregrapher-level `filter`.
    - **MPG-style structural steps always see every vertex**, since segmentation, emergence and death act precisely on inactive ones too.
    - The repartition rules use the same mask for their "focused" branch.
  - **Overrides:**
    - variables the structural component writes itself keep its values;
    - `on_grow` stays available per variable for the cases the table does not cover.
  - **Lineage:** "parent" is the nearest pre-existing ancestor of a new entity, as the WD.3 carry-over already computes. A chain of new segments created by one segmentation is processed from the base to the tip, so the pairwise splits compose like in rhizodep.
  - **Validates on:** the DS19 growth helper.
    - A segmentation conserves every extensive amount and every `c · m` amount, summed over the origin and its new segments.
    - A primordium copies concentrations and holds zero amounts.
    - Its first elongation splits it from its parent.
    - Elongation of an existing segment dilutes its concentrations.
    - The numbers equal those of `post_growth_updating` on the same small tree.
- [x] **DS21 Graph systems on the active subgraph (step 2d) (answers Q19, D16).** `@graph_system(where="active")` is opt-in. The default stays the whole graph, with the mask entering only through `filters`. Robust handling means:
  - **Views cached by `(topology_version, mask_version)`.** The mask has its own version, bumped when its values change; it is re-evaluated before the solve if any of its sources was written (DS17). The subgraph is rebuilt only when either version changed.
  - **Subgraph definition:**
    - the active nodes;
    - the edges whose **both** endpoints are active;
    - the boundary conditions, boundary sets (DS6) and ports restricted to active entities.

    Local ↔ full index maps are vectorised gather / scatter arrays, and writes go back in place into the full DataStructure arrays.
  - **Inactive entities:**
    - node unknowns keep their last value (frozen);
    - edge outputs on dropped edges are set to 0, i.e. no flux;
    - `@rate` outputs follow D15 (left unchanged).
  - **Activation and deactivation between steps:**
    - a newly active node enters with its current value, as set by the repartition (DS20) or `on_grow`;
    - its `previous()` value is that same value;
    - a deactivated node's last value stays in the DataStructure.
  - **Well-posedness check, at each rebuild.** Every connected component of the subgraph must have either a capacity term (a time derivative) on its nodes or at least one Dirichlet condition.
    - Isolated active nodes with a capacity term are valid: they reduce to local ODEs.
    - An ill-posed component raises, naming its entity ids and the system, instead of failing later as a singular matrix.
  - **Empty subgraph:** the solve is skipped, and outputs follow the inactive rules.
  - **Anatomy mode (DS8):** the mask is defined on Compartments. A SubOrgan-scale rule is broadcast to its Compartments through the owner map.
  - **Validates on:**
    - all-active equals the whole-graph solve, bit for bit;
    - a primordium becomes active between two steps, with the view rebuilt and fluxes and amounts conserved;
    - a dead segment drops out, with its edges at zero flux;
    - a disconnected algebraic component raises with its ids;
    - an empty mask skips the solve;
    - the mask is recomputed when its source variable changes, with no stale view.
- [x] **DS9 Solve-time data policy.** (step 4a: read-only views; the step-level previous state is DS10, step 4b)
  - `_snapshot` copies every required variable at each solve. Keep copies for unknowns and previous state; pass **read-only views** for parameters and inputs. This saves memory and time on large root systems.
  - Also define a policy for several graph systems in one step:
    - `previous()` is "state at the start of this solve";
    - decide whether a step-level previous state is needed for operator splitting (D5).
- [ ] **DS10 Graph systems with their own time loop and sub-stepping (D5 decided: both previous-state levels).**
  - Graph systems use `step_once` only; the adaptive `DAESolver.solve` loop is not reachable from `@graph_system`.
  - Offer `@graph_system(integrate="step" | "adaptive" | "substeps", n_substeps=..., t_span=...)`, now that B5/B6 made the loops correct, and pass `BoundaryConditions` updates through it.
  - Sub-stepping is **per component** (Q8): a component's own `sub_time_step` within the Choregrapher step, as today. Its inputs from other components are those at the start of its call; groups are not planned.
  - `previous(fn)` stays the state at the start of the current solve; `previous(fn, at="step")` gives the state at the start of the Choregrapher step, for operator splitting.
- [x] **DS11 Validation and failure modes:**
  - `DataStructure.validate()` checks that every registered array has its location's shape, and that aliases and derivations resolve;
  - `update_topology()` without `from_scale` becomes impossible (make `from_scale` required);
  - a missing filter variable, a missing `is_root`-style flag, or a derived variable with a missing source raises instead of acting as zeros (`_read_array` still returns zeros for missing names, `decorator.py`).
- [ ] **DS12 Non-float and categorical variables** (Q22 was decided but is not implemented).
  - An object store (`location` with `dtype=object`) for lists and records, not solver- or transport-eligible.
  - Integer categorical variables (labels, types) stored as ints, not coerced to float, with name ↔ code resolution through the per-instance `LabelsConfig`.
  - Filters then accept names: `filters={"label": ["RootSegment"]}`.
- [ ] **DS13 Choregrapher per-instance scheduling.**
  - Steps are registered by **class name**, and bound to the **last** instance of a class. Two plants of one class cannot run in one process, and same-named classes in different modules merge.
  - Bind schedules per instance, with registration per class object instead of name. That removes the one-plant-per-process constraint of the in-process scene and the name-collision hazard.
  - Also clarify the scheduling order: rows currently run in reverse of their listed order (`axial` before `rate`, see B8).
- [ ] **DS14 Performance paths:**
  - vectorise `_mtg_values` and `complex_at_scale` membership (Python loops per vid);
  - vectorise the `to_graph_view` vid → index dicts;
  - benchmark on a 20 000-segment root system;
  - decide whether a numba path is wanted on the vectorised steps (links to Q29: delete `specializer.py` or rewrite it).
  - **Found in 4a:** `MPG.populate_graph` on a single 20 000-segment chain raises `RecursionError` in openalea.mtg's recursive `pre_order` (through `post_order_mpg` → `components_iter`). Long axes need an iterative traversal there.
- [ ] **DS15 Persistence.** Checkpoint and restart of a DataStructure (arrays + locations + aliases + derivations + `topology_version`) independently of pickling the MPG, plus an MTG round trip for existing tooling. This builds on `export`.
- [x] **DS17 Derived variables resolved at read (answers D3).** Today aliases resolve at every read, but derived variables (factors, sums, scale changes) are recomputed only when the **receiver** refreshes them before its step, as the former `pull_available_inputs` did.
  - To keep "linked by name, read dynamically" for every link kind, a derived variable is recomputed **on `get()`** when one of its sources was written since its last computation. This needs per-variable write counters, maintained by `set()` and in-place step outputs.
  - Explicit `refresh()` stays available, and the receiver's pre-step refresh becomes a no-op when nothing changed.
  - **Rule to document:** writing through `ds.set` (or the framework) keeps links consistent; mutating `ds.get(x)[...]` by hand does not mark `x` as changed (D10).
- [x] **DS18 Cross-scale links in the translator (steps 3a–3b) (answers D4.3 and Q7).**
  - When a link joins two locations (e.g. SubOrgan → Organ, anatomy Compartments → SubOrgan, SubOrgan → Compartments), the coupling builds the derived variable with an explicit mapping:
    - **up (aggregation):** `sum` (extensive), `mean` / `weighted_mean` (intensive). This is the main direction: e.g. the anatomy water balance populates the SubOrgan radial water uptake.
    - **down:** `broadcast` (intensive), with an optional **target filter**, e.g. a SubOrgan concentration sent to the symplastic Compartments only (Q7). Compartments outside the filter keep their own value.
    - An extensive quantity going down has no default (Q7: not expected). A weighted `split` stays possible later as an explicit mapping, not implemented now.
  - When a link gives no mapping, the default follows the variables' `state_variable_type` (D9):
    - extensive / NonInertialExtensive → `sum` up;
    - intensive / massic_concentration → `mean` up and `broadcast` down;
    - extensive down, or a missing type → the coupling raises and asks for an explicit mapping.
  - The receiver's input therefore always exists at its declared location, with values for every entity of that scale, including after growth (the WD.3 carry-over).
- [x] **DS16 Documentation of conventions**, in one page:
  - incidence sign (+1 at the parent);
  - the local order is not sorted;
  - `(x, y, z)` grids;
  - `proximal` means owned by the child;
  - plant scale = `"scalar"`;
  - `previous()` semantics;
  - `on_grow`.

---

## 6. Decisions

| # | Decision | Status |
|---|---|---|
| D1 | Grid topology: faces as edges, with a geometric factor as an edge variable | **Agreed** (2026-10-01) |
| D2 | Split `scale=` into `location=` / `scale=` / `mapping=`, old forms accepted for one release | **Agreed** |
| D3 | The MTG is optional for components; write state variables after every call | **Agreed**, under the condition that links stay resolved dynamically by name. Today this holds for aliases (resolved at each read) and for derived variables refreshed before the receiver's step, as the former props coupling did. It becomes exact at read with DS17. |
| D4 | One topology per component, several components per DataStructure; multiscale graphs as one Compartment/Connection topology (DS8); MPG ↔ MPG Coupler deferred | **Agreed**, with the precisions in §8 |
| D5 | `previous(fn)` per solve and `previous(fn, at="step")`; per-component sub-stepping (DS10) | **Agreed**; per component only (Q8) |
| D6 | Boundary sets: rule declared on the class, values and weights from DataStructure variables | **Agreed** |
| D7 | Per-instance Choregrapher scheduling (DS13) | **Agreed** |
| D8 | Node identity public only through `entity_ids` / `index_of` | **Agreed** |
| D9 | Default cross-scale mapping inferred from `state_variable_type` when a link gives none (DS18); extensive going down, or a missing type, raises. Detailed in §8 "D9 in detail" | **Agreed**: option A (2026-10-01) |
| D11 | The MPG is the source of truth for topology. Every edge, junctions included, is a Connection vertex populated by MPG methods, and the DataStructure extracts its graph view from the MPG (DS8) | **Agreed** (Q9; replaces the earlier proposal of junctions held only in the DataStructure) |
| D12 | In anatomy mode, growth keeps the anatomies and adds the junctions of new SubOrgans. An existing SubOrgan is re-wired only when its neighbourhood or its anatomy changed (detected automatically). Edge variables carry over by endpoint pair (DS8) | **Agreed** (Q13) |
| D13 | Values: copies with explicit sync (option a, no MPG storage change); structural components edit the MPG through `ds.mtg`. Their declared inputs are flushed to the MPG before each structural step; their outputs are written as MPG properties and re-read after it, with `update_topology()` when the topology changed (DS19) | **Agreed** (Q12, Q14, Q15) |
| D14 | Repartition when structure changes (DS20): the rules of rhizodep's `post_growth_updating`, applied by the framework to every registered variable after each structural step, by `state_variable_type`. The partition weight, the previous weight and the active set are declared by the structural component as class attributes, each accepting a callable (nothing model-specific built in). Its own outputs and `on_grow` override | **Agreed** (Q16, Q18) |
| D15 | The active set is a named DataStructure mask, applied by default to vectorised steps (opt-out with `where=None`), and never to MPG-style structural steps (DS20) | **Agreed** (Q18) |
| D16 | Graph systems solve on the whole graph by default; `where="active"` solves on the active subgraph, with the robustness rules of DS21 (versioned views, frozen inactive nodes, zero flux on dropped edges, well-posedness check) | **Agreed** (Q19) |
| D10 | Derived variables recomputed lazily at read (DS17), with in-place mutation outside `set()` documented as unsupported | **Agreed** (2026-10-01) |

---

## 7. Suggested order

1. **Contract:** DS3 (Location, with scale locations for coarse fields) + DS17 (derived at read) + DS5 (output locations) + DS11 (validation) + DS16 (conventions). Design note: `docs/design/datastructure_contract.md` (N1–N5 agreed). Progress: **1a done** (`4d1353d`), **1b done** (`15f8976`), **1c done** (`cc3e588`), **1d done** (`bf008e3`), **1e done** (`bc37e79`), **1f done** (2026-10-02): step 1 complete.
2. **Multiscale topology and boundaries:** DS19 (StructuralComponent contract) + DS20 (repartition, active mask) + DS21 (active subgraph) + DS8 (assembled graph: anatomies + wiring rules, real anatomy UC3) + DS2 (traversal) + DS6 (boundary sets, validated on UC5 below). Design note: `docs/design/structure_and_boundaries.md` (agreed, P1–P7). Progress: **2a done** (`4fbe3dc`), **2b done** (`e659bd0`), **2c done** (`6430b08`), **2d done** (`9e0b00b`), **2e done** (`3ffb485`), **2f done** (2026-10-02): step 2 complete, except the repartition in anatomy mode (open question A1).
3. **Cross-scale coupling and grids:** DS18 (mappings, filtered broadcast, defaults) + DS1 (grid topology). Design note: `docs/design/cross_scale_and_grids.md` (agreed, R1–R5). Progress: **3a done** (`40657a2`), **3b done** (`1f6c926`), **3c done** (`ceb6e2f`), **3d done** (2026-10-02): step 3 complete.
4. **Time and data:** DS10 (sub-stepping, adaptive loop, `previous(at="step")`) + DS4 (sync) + DS9 (solve-time views) + DS12 (typed variables). Design note: `docs/design/time_and_data.md` (agreed: recommendations T1–T7). Progress: **4a done** (2026-10-02).
5. **Runtime:** DS13 (per-instance scheduling) + DS14 (performance) + DS15 (persistence). DS7's MPG ↔ MPG Coupler only if a use case requires it.

Every step follows the same practice as WD: a design note, tests first, and the UC1–UC4 and wrapper contract tests unchanged as regression gates.

**New use case, UC5 leaf transpiration (from Q4), as the DS6 acceptance test:**
- Robin boundary set on the leaf nodes, with per-leaf microclimate inputs (air water potential / vapour pressure deficit per leaf, from a light/microclimate model) as node variables.
- The conductance is the product of a stomatal conductance (a model variable, computed by a `@rate` of the same or another component) and an anatomical exchange surface (a parameter or a structural variable).
- The water potential is solved on the shoot + root graph.
- It is a use case designed to exercise the tool, not a validated plant model.

---

## 8. Open questions for you

Answer inline, as in `devplan.md`.

- **D1–D8** above.
  → answers: 
    - agree on D1 and D2
    - D3 I am worried about this decision because all couplings of metafspm used to be made possible by dynamically linking to the translator's mentionned names in MTG properties' dicts or array at read. Does this still hold on DS? If yes, I agree.
    - On D4 several questions: 1) One per component does not mean one par DataStructure right? Because the coupling on one plant relies on several components being anchored to the same DataStructure (which was a central MTG before all our developments to wrap it). 2) I introduced abstract Compartment and Connection scales initially to be able to hold a topology that would be build from the traversal of several scales (e.g. anatomies being natively stored at Connection and Compartment scales because this matches the node and edge logic of anatomy generator models, and then between each anatomy held in SubOrgan scales, link them conditionnally with a variable, for example vessel indices, so the graph system results from the assembly of all scales because then it calls for all SubOrgan connectivity in the plant), Is is possible to build a topology like that from the MPG and its wrapping DataStructure? If yes, I think one per component is enough yes, other equations of the component will mostly be scalar operations anchored on specific scales preparing or postprocessing graph solve. 3) a side question probably applying to another section: If a variable is simulated by a component at a specific scale, and that another coupling to this variable as an input operates at a different scale, upper needing aggregation or lower needing split if extensive, does the current translation logic populates that variable on relevant vids at these scales so that the input exist?
    - D5: agree, one component or component group might want its own substepping and therefore not just step once.
    - D6, D7, D8 agree
- **Q4.** For leaf boundary conditions (DS6): which variables should drive them? Air water potential only, as a scalar, or also per-leaf values (e.g. from the light model's microclimate)? Is the conductance a model variable (stomatal model) or a parameter?
  → answer: This answer really only should be to design an example use case, not specificities of the tool. That said, when implementing a model, per-leaf values of microclimate are also needed as part as model inputs, and leaf stomatal conductance and anatomical exchange surfaces to scale conductance are also part of the model.
- **Q5.** DS7: which cross-scale solves do you foresee first (e.g. SubOrgan water with an Organ-scale carbon balance)? The answer tells whether the MPG ↔ MPG coupler or several topologies per component is more urgent.
  → answer: I already answered above, but ask complementary questions if precision are needed.

### Replies to your D3 / D4 questions (2026-10-01)

- **D3: do links still resolve dynamically by name on DataStructures?** Yes, with one nuance.
  - Components that share a DataStructure share variables **by name**, like the MTG properties dict, so identity links need nothing.
  - **Aliases are name-level and resolved at every read** (`ds.get(alias)` is the source array).
  - **Factor, sum and scale links** are derived variables recomputed by the receiver right before its step. That is exactly the former `pull_available_inputs` timing, not every read.
  - DS17 makes them exact at read as well (D10).
- **D4.1: "one per component" means one topology per component, not one DataStructure per component.** Several components share one DataStructure; that is how the plant is coupled (WD.4, `test_composite_datastructure.py`).
- **D4.2: yes, a graph assembled across scales is possible from the MPG.** The Compartment/Connection scales and `populate_graph_custom_connections` (label pairing, `ordering`-based matching such as vessel indices) build it. The DataStructure cannot wrap it yet; DS8 adds that, and its priority is raised. With DS8, one topology per component is enough, as you said.
- **D4.3: no, not today.**
  - A field declared at a coarser scale is registered at node location (broadcast), so a cross-scale input exists only as a copy at the solver nodes, and nothing aggregates or splits.
  - The translator links accept `scale` / `aggregation` (WD.0, WD.4), but nothing infers the mapping, and there is no `split` for extensive quantities.
  - DS3 + DS18 make the input exist at its own scale, with the right mapping.

### Complementary questions

- **Q6 (DS8).** In your anatomy generators, are the **intra-segment** edges (cell to cell within one cross-section) also Connection vertices with `n_id_a` / `n_id_b` set to Compartment vids, as the inter-segment links are? When a segment is created by growth, is its anatomy generated immediately, before the inter-segment links are re-run?
  → answer: I don't get the question clearly, for me connection between segments, which are held at SubOrgan scale, are held at this scale. Compartment and Connection are scales bellow SubOrgan designed to hold anatomy graphs, and then the multiscale traversal is supposed to build nodes from anatomies' Compartments, and edges from both anatomies' Connections, but also from the information of connectivity at SubOrgan and coarser scales, with possible conditionnal rules of which Compartment to wire to which between SubOrgans using dedicated variables (e.g. vessel index or angular coordinates, custom formalisms will be declared for that by the modeller). Ask further precisions and questions if I missed the point.
- **Q7 (DS8 / DS18).** Should a SubOrgan-scale variable (e.g. radial water uptake of a segment) be distributed over specific Compartments only (e.g. epidermis, through a label filter on the split), or over all the Compartments of the segment?
  → answer: I don't think this kind of distribution will ever occur. If the SubOrgan scale gets an anatomy bellow, eventually it is rather the water balance on the anatomy that will be used to populate SubOrgan's scale radial water uptake. An example distribution could for example be: a SubOrgan scale metabolic model solves a metabolites concentrations, that can then be distributed to all symplastic Compartments of the cross section the update the osmotic potential.
- **Q8 (DS10).** Is sub-stepping configured per component (each component its own `sub_time_step`, as the Choregrapher allows today) or per explicit **group** of components that sub-step together, refreshing their mutual links at each sub-step?
  → answer: Keep it per component for now.
  

### Replies to Q6–Q8 (2026-10-01)

- **Q6:** you did not miss the point; my question assumed the wrong model. It assumed that the links between segments are stored as Connection vertices, which is what `populate_graph_custom_connections` does today. Your model:
  - anatomies are held below SubOrgan;
  - connectivity is held at SubOrgan and coarser scales;
  - the graph is assembled by the traversal, with modeller-declared wiring rules.

  DS8 is rewritten that way, and proposed decision D11 records it.
- **Q7:** the down mapping is a filtered `broadcast` of intensive quantities (e.g. to symplastic Compartments). Aggregation up is the main direction, so `split` is dropped from the defaults (DS18, D9 revised).
- **Q8:** sub-stepping is per component (DS10, D5).

### Complementary questions (2)

- **Q9 (D11).** Do you agree that the generated junction edges between SubOrgans should **not** be written to the MTG as Connection vertices? They would exist only in the DataStructure's graph view, rebuilt on growth, with their variables (e.g. an axial conductance per vessel junction) stored at the DataStructure's edge location. If something downstream needs them in the MTG (e.g. a visualisation or a Logger), they would be exported on demand.
  → answer: For me the DataStructure wrapping the MPG should always be a dynamic interface to it, so I would think that graph_view creation would need to be based on populating the MPG and extracting from it node and edges using MPG methods traversals, parent-child informations and array/ArrayDict exctraction. Does this idea calls for further questions?
- **Q10 (DS8).** In the anatomy generators you use, are the exchanges **within** a cross-section stored as Connection vertices with `n_id_a` / `n_id_b` set to Compartment vids? If not, how are they stored, e.g. as adjacency in a property, or implied by the labels?
  → answer: In my general idea yes, but the GRANAP model you can find in package/ has not been reimplemented to use MPG as its underlying DataStructure yet, this will be done after metafspm is stabilized by implementing it as a StructuralComponent.
- **Q11 (DS8).** Are junction edges only ever between SubOrgans that are parent and child at SubOrgan scale (axial continuity, including across organ boundaries as `populate_graph` does)? Or can a rule also join non-adjacent SubOrgans (e.g. the parent's tip to a lateral's base at another position, or root–soil contact)?
  → answer: No, even if multiscale information can be used to connect two suborgans based on a connection at Organ scale, normally the method you can see in test to differentiate a complex parent from a within-scale parent enables to always have a direct link encoded in the MPG topology.

### Replies to Q9–Q11 (2026-10-01)

- **Q9:** agreed. It is also how `MPGDataStructure` already works for one Compartment per segment:
  - `populate_graph` / `repopulate_graph` write the Compartments and Connections into the MPG;
  - `_build_index_map` and `to_graph_view` extract them through `array_filtering` and `n_id_a` / `n_id_b`.

  D11 now records that the junctions follow the same rule; my earlier proposal (junctions only in the DS) is withdrawn. Your idea does raise two follow-up points: what "dynamic interface" means for the **values** (Q12), and growth in anatomy mode (Q13).
- **Q10:** noted. GRANAP stays out of scope until it is reimplemented as a StructuralComponent. DS8 is validated on a synthetic anatomy built in this repo's test helpers.
- **Q11:** noted. Junctions only join direct MPG links (within-scale parent, or the complex parent's tip), so wiring rules only ever see parent/child SubOrgan pairs.

### Complementary questions (3)

- **Q12 (DS4 / D11), dynamic interface for values.** Today the topology is extracted from the MPG at every topology change. **Values**, however, are copied into the DataStructure's arrays (`_node_data`, `_edge_data`, read from the MPG properties at `declare` / `refresh`), and state variables are written back after each call.
  - True zero-copy views are not possible with the current storage: one ArrayDict per property spans every scale, so the Compartment values are a fancy-indexed copy.
  - Option **a**: keep copies, with the DS4 policy making the sync explicit (read at declare / topology change, write back after each call). Within a step, every component sees the shared DataStructure, and the MPG is up to date between steps. This is the recommendation.
  - Option **b**: the MPG stores each property per scale, so that the DataStructure arrays can be views on it. That is an MPG storage change.

  Is **a** enough for "dynamic interface", or do you need the MPG properties to be exact at every moment within a step (e.g. a structural component reading the MPG directly between two other components)?
  → answer: well the structural component will indeed need to operate on the MPG directly to access all its edit methods (except if I misunderstood and that operating on the DataStructure can communicate all these methods), which also calls on how a StructuralComponent uses the input DS that is provided to it? But a) if better because it doesn't ask for changes on MPG.
- **Q13 (D12), growth in anatomy mode.** When a segment grows, is its anatomy generated by the anatomy StructuralComponent **before** `ds.update_topology()` runs? `update_topology` would then only add the junctions of the new SubOrgans, and never delete Compartments or Connections it did not generate. Should an existing segment's anatomy also be able to change, e.g. by secondary growth adding cells? If yes, that segment's junctions must be re-wired as well.
  → answer: Existing should possibly rewire, but conditionnally, only if its SubOrgan scale neighboring changes, or if differentiation changes anatomy. Other from that existing connection should be kept to remain efficient.

### Replies to Q12–Q13 (2026-10-01)

- **Q12:** option a is kept (D13).
  - **Your follow-up:** yes, structural components operate on the MPG directly, with all its edit methods. The DataStructure should not re-expose them. It already gives access to its MPG (`ds.mtg`), and duplicating edits would create two sources of truth.
  - **How a StructuralComponent uses its DataStructure** (new item DS19):
    - it receives the shared DataStructure, and edits `ds.mtg`;
    - the framework flushes its declared inputs to the MPG before its step, and calls `update_topology()` after it.

    So the MPG is exact whenever a structural component reads it, which is exactly when option b would have mattered.
- **Q13:** D12 is agreed, with conditional rewiring. A SubOrgan's junctions are rebuilt only when its neighbourhood or its anatomy changed, which `update_topology` detects by comparing with the last extraction. Everything else is kept.

### Complementary questions (4)

- **Q14 (DS19).** Should a StructuralComponent write its structural outputs (length, radius, coordinates) as **MPG properties**, so that `update_topology()` re-reads them (recommended: structure stays in the MPG)? Or should it write them to the DataStructure, with the MPG updated at the next write-back?
  → answer: As MPG properties yes.
- **Q15 (DS19).** How should a structural step be scheduled: as decorated steps like a FunctionalComponent's (`@rate`-style, one per structural process: emergence, elongation, pruning), or as one `__call__` / `run` per component? That decides whether several structural processes can be interleaved with functional ones within a Choregrapher step.
  → answer: scheduled as FunctionalComponent's, you can see an example in package/Wheat-BRIDGES_framework/rhizodep/src/openalea/root_growth.py, and I guess that when GRANAP turns to a Component, it will be scheduled in multiple steps like that. We mostly talked about FunctionalComponent until now, so you can ask further questions about StructuralComponent is this helps stabilizing the DataStructure API beforehand.

### Replies to Q14–Q15 (2026-10-01)

- **Q14:** structural outputs are MPG properties, re-read after each structural step (DS19, D13).
- **Q15:** structural steps are scheduled like FunctionalComponent steps, interleaved in the Choregrapher rows. Following `root_growth.py`, they are whole-MPG methods without array arguments, and the `__call__` post-processing becomes a scheduled step. DS19 is rewritten from that example.

### Complementary questions (5): StructuralComponent

- **Q16 (D14), functional state at segmentation.**
  - When an apex is segmented, the new segments and the shortened apex share what the apex held. The TODO in `segmentation_and_primordium_formation` says it: the N amount is divided, the N concentration is unchanged.
  - With components sharing a DataStructure, the growth model does not know every functional variable of the other components (hexose, amino acids, water…). Today's `on_grow` only gives new entities a default or the parent's value. It cannot reduce the origin's amount.
  - Proposal D14: for every registered variable that the structural component does not write itself, the framework:
    - finds the **lineage**: each new vid's nearest pre-existing ancestor, the "origin";
    - **splits** extensive variables over the origin and its new descendants, in proportion to a weight such as `struct_mass` or `length`;
    - **copies** intensive variables.

    Variables the structural component writes keep its values (override).
  - Do you agree? Which weight is right by default: structural mass, or length? And should a lateral primordium formed on a new segment take part in the split, or start from defaults? 
  → answer: If it fits with the logic of the post_growth_updating method of the root_growth, yes the idea is the same and in this model only, the split is guided by structural mass because concentrations are massic, but another model could use something else for normalization so don't overspecialize, this was to give an exemple. For lateral again don't overspecialize, but in the root_growth component, I think it is fed from its parent but not split from it unlike growing apices on the same root order, right?
- **Q17 (DS19), structural step signature.** Is it fine that structural steps keep the whole-MPG style of `root_growth.py`: no array arguments, node-level MTG access, declared inputs and outputs used only for sync? Or do you want them to receive the declared variables as arrays, like functional steps, for the parts that can be vectorised (e.g. thickening)? Both can coexist: arguments are passed only when the step names them.
  → answer: Steps needing specific topology traversal will use the MPG style edits, but of course when operations can be vectorised I want to go for it.

### Replies to Q16–Q17 (2026-10-01)

- **Q16:** yes, it is the same idea, and DS20 now takes `post_growth_updating`'s rules as the reference, without building in the structural mass. **On laterals you are right, with one precision:**
  - at **formation**, a primordium is created with `nil_properties=True` (zero mass). It copies concentrations from its parent (or grandparent), holds zero amounts, and the parent is not reduced;
  - when it **first elongates** from zero mass, it goes through the same pairwise split as a new apex (`f = m_v / (m_v + m_p)`, the `else` branch for elements elongating from null structural mass). So a lateral is fed from its parent at emergence, not at formation.
- **Q17:** both step styles coexist. MPG-style steps for traversals and edits; array arguments for vectorisable steps, with their outputs flushed to the MPG (DS19).

### Complementary questions (6)

- **Q18 (DS20), declaring the repartition.** How should a structural component declare its partition weight, previous weight and active set?
  - **a**, class attributes: `partition_weight = "living_struct_mass"`, `previous_weight = ("initial_struct_mass", "initial_living_root_hairs_struct_mass")`, `active = {"struct_mass": ">0", "label": [...], "type": [...]}`;
  - **b**, a decorated method returning the weights, for computed ones.

  Recommendation: **a**, with a callable allowed for each entry. Also: should the "active set" be a DataStructure-level mask that FunctionalComponents use as well, replacing rhizodep's `focus_elements` and the Choregrapher filter, or stay local to the repartition?
  → answer: go for the a) proposition, and for the side question the focus_elements "active set" was for vectorised operations only, but the whole set of vertices should still be accessible for other structural steps.

### Replies to Q18 (2026-10-01)

- **a** is kept: class attributes, each accepting a callable.
- **Active set:** a DataStructure mask applied to vectorised operations only; MPG-style structural steps see all vertices (DS20, D15).

### Complementary questions (7)

- **Q19 (D15 / DS8), active set and graph systems.** Should a graph system also be restricted to the active set, e.g. solve water transport only on emerged, living segments?
  - **a**: the solved graph is the **subgraph** of active nodes and the edges between them. Inactive nodes are not unknowns and keep their values. This is a different topology whenever the mask changes, so graph views are rebuilt on a mask version as well as on `topology_version`.
  - **b**: the whole graph is solved, and the mask only enters the equations through `filters`, as today (e.g. zero conductance for inactive segments).

  Recommendation: **a** opt-in (`@graph_system(where="active")`), **b** by default. Disconnected inactive nodes (non-emerged primordia) would otherwise make the system singular unless every equation handles them.
  → answer: yes for a), as long as behind the argument where="active", there is a robust handling of the active set.

### Replies to Q19 (2026-10-01)

- **a** is kept as opt-in (`where="active"`), with **b** as the default. "Robust handling" is now specified as DS21:
  - views versioned by topology and mask;
  - a clear subgraph definition, and rules for inactive entities;
  - activation and deactivation semantics;
  - a well-posedness check per connected component that raises with ids;
  - empty-subgraph handling.

**No open question remains in this plan.** Q1–Q19 and D1–D16 are answered or agreed, except D9 and D10 (still *proposed*: the default cross-scale mapping and lazy derived variables). Mark them agreed or amend them, and step 1 of §7 (the contract: DS3, DS17, DS5, DS11, DS16) can start with its design note.

### D9 in detail (2026-10-01)

- **The situation.** A translator link connects an output of one component to an input of another. When both live at the **same** scale, the values are passed entity by entity. When they live at **different** scales, e.g. an output per SubOrgan feeding an input per Organ, there are more source values than target values (going up) or fewer (going down). Something must say how to combine or distribute them.
  - Today the link can say it explicitly (`aggregation="sum"`, `weight=...`); without it, the coupling raises.
- **D9 only decides what happens when the link says nothing.** The source variable's `state_variable_type` is used to pick the physically consistent mapping:

  | Source type | Going up (fine → coarse) | Going down (coarse → fine) |
  |---|---|---|
  | extensive / NonInertialExtensive (amounts, fluxes in mol.s-1) | **sum**: the total is conserved | **raise**: copying would multiply the amount by the number of children, and a split needs a weight that only the modeller knows |
  | intensive / NonInertialIntensive (potential, temperature) | **mean** | **broadcast**: each child gets its owner's value |
  | massic_concentration | **mass-weighted mean**, which needs a weight variable: raise if the link gives none | **broadcast** |
  | missing type | raise | raise |

  The source and target types must also agree (an extensive output cannot feed an intensive input); otherwise the coupling raises.
- **Examples:**
  - `radial_water_uptake` (mol.s-1, extensive) per SubOrgan → a plant-scale water balance input: summed;
  - `osmotic_potential` (intensive) per Compartment → its SubOrgan: averaged;
  - `C_sucrose` (massic concentration) per SubOrgan → the symplastic Compartments of the cross-section: broadcast, with the target filter (DS18, Q7);
  - `carbon_supply` (extensive) per Organ → SubOrgans: raises with "give an explicit mapping (split with weight=…, or broadcast)".
- **Choice to make:**
  - **A, defaults as above.** Shorter translators, physically safe choices only, and a raise whenever the choice is ambiguous.
  - **B, always explicit.** Every cross-scale link must give its mapping. This is today's behaviour, and the more verbose of the two.

  Recommendation: **A**. B remains available, since an explicit mapping always overrides the default.
  → answer: A

### Step 2f follow-ups (2026-10-02)

- **A1: the repartition in anatomy mode waits** until GRANAP becomes a StructuralComponent. GRANAP's anticipated use, in your words:
  - "a segment belongs to a diameter X differentiation class";
  - "we generate a dynamic library of anatomies, with one anatomy being able to populate several segments of the same class";
  - "only their between-SubOrgan links being added to connect adjacent anatomies".

  This will shape how anatomies are attached to segments (see Q-A4 below), and the repartition depends on it.
- **A3: agreed.** Junctions are marked with the integer `is_junction`, consistent with integer types and labels (see the configs), so that array operations stay possible.
- **A2:** re-explained in the devlog, awaiting an answer.
- **Q-A4 (for GRANAP, no change now).** When one library anatomy "populates" several segments of a class, does each segment get its **own Compartment vertices**, a copy of the template, which the anatomy mode supports today? Or should the segments **share** the template's description, and be instantiated as per-segment Compartments only in the DataStructure's graph?
  → answer:
