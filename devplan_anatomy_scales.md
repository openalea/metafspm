# Plan: anatomy mode with Layer and Cell scales (GRANAP P0)

Your request (GRANAP `devplan_metafspm.md`, Q5, 2026-10-07): before porting GRANAP, make metafspm's anatomy mode support SubOrgan → Layer → Cell → Compartment, with its own plan and tests here. The GRANAP mapping agreed there:
- a Layer vertex per tissue region (each bundle or pole its own Layer);
- a Cell vertex per biological cell;
- the cell's protoplast as a Compartment under its Cell;
- walls and wall junctions as Compartments shared under the SubOrgan;
- every MECHA edge as a Connection.

## 1. Today

- **Creation:** Compartments are components of the Compartment scale's anchor, linked to their SubOrgan by a topological parent (`add_component_with_topo(compartment_anchor, suborgan)`), as in the SPAC seedling. A Compartment added as a component of a SubOrgan gets scale 7, because the MTG sets scale = complex + 1.
- **Ownership:** a node's section is its MTG parent (`_owner_at`, `_membership`: `topology_arrays()["parent"]`, then the complex at the scale asked).
- **Coarse locations:** only the scales from Plant to `from_scale` (`_coarse_scale_names`: `value < from_scale + 1`). Layer and Cell are not locations.
- **Probe:** a SubOrgan with one Layer, two Cells and their Compartments, plus a wall Compartment. `MPGDataStructure(g, from_scale=SubOrgan, nodes="Compartment")` then fails: "'SubOrgan' is not a coarse location ([])".
- **Built on "the parent is the section":**
  - `compartments_by_owner` (anatomies per SubOrgan);
  - `wire_junctions` (Compartments of linked SubOrgans);
  - the anatomy signatures (partial rewiring on differentiation);
  - the repartition in anatomy mode;
  - `on_grow="inherit"` (same-label Compartment upstream).

## 2. Proposal

- **R1, structure.** Layer and Cell vertices become true MTG components (SubOrgan → Layer → Cell are consecutive scales, 6 → 7 → 8). A protoplast Compartment is a true component of its Cell (scale 9). Shared Compartments (walls, junctions) stay as today: components of the Compartment anchor, with a topological link to their SubOrgan. Connections are unchanged.
- **R2, ownership.** Each node has an owner vertex: its MTG complex when it is a true component (a Cell), else its topological parent (the SubOrgan, for anchored nodes). Its entity at any coarser scale is the complex at that scale of its owner, through Cell and Layer up to SubOrgan and the plant. Its section is that entity at `from_scale`.
- **R3, coarse locations.** In anatomy mode the locations are every scale coarser than Compartment that holds entities: Plant … SubOrgan, plus Layer and Cell when present. `ds.owner("Cell")` and `ds.owner("Layer")`, `ds.entity_ids("Cell")`, declarations at `scale=Cell`, derived variables and translator aggregations then work as at SubOrgan.
- **R4, nodes without an entity at a scale.** Walls and junctions have no Cell and no Layer. Their owner index there is −1:
  - aggregations towards the scale (sum, mean, weighted mean) skip them;
  - a broadcast from the scale gives them the variable's default;
  - masks and filters on a Cell variable read the default for them.
- **R5, everything built on the section.** It goes through R2: `compartments_by_owner(from_scale)` returns all Compartments of each section, whether through its Cells or directly. The wiring, the signatures, the repartition and `on_grow` then work unchanged.
- **R6, bulk creation.** Building a section in one call per scale:
  - `add_components_bulk` for the Layers of a SubOrgan, the Cells of a Layer and the protoplasts of the Cells;
  - a bulk variant for anchored Compartments with a topological parent (walls, junctions), and for Connections.

  A GRANAP section is 5–30 k vertices, so per-vertex calls would dominate.
- **Tests**, in `test/data_structures/anatomy/`, a synthetic two-section anatomy with Layers, Cells, protoplasts, walls and junctions:
  - the owner and entity ids at every scale;
  - declarations at `scale=Cell` and `scale=Layer`;
  - aggregations and broadcasts with walls (R4);
  - the solver graph (nodes, edges, GraphView);
  - junction wiring between sections on protoplasts by label and ordering;
  - partial rewiring after a differentiation;
  - the repartition on growth;
  - checkpoint and restore;
  - the bulk creation (R6), against per-vertex creation.

  The existing anatomy-mode tests, with Compartments directly under SubOrgans, must keep passing.

## 3. Questions

- **QA1 — structure (R1).** Should Layers, Cells and protoplasts be true MTG components (consecutive scales), with only the shared walls and junctions anchored? Or should every vertex below SubOrgan be anchored with topological links (Layer → SubOrgan, Cell → Layer, Compartment → Cell), as Compartments are today?
  - **Recommendation:** true components. It is the MTG's own decomposition, so `complex_at_scale` and openalea's traversals work as they are. The anchored form exists only because Compartments skipped scales.
  - → answer: yes to recommendation, as long as at the end we can build the whole hydraulic graph and structure as output of GRANAP, stored at Compartment and Connection scales as nodes and edges of the graph.
- **QA2 — walls and junctions at the Cell and Layer scales (R4).** Should they be skipped by aggregations, and get the default from broadcasts? Or should each be attached to a Cell (e.g. the lowest-id cell it bounds) so that every node has one?
  - **Recommendation:** skipped, with the default. An arbitrary owner would count a wall in one cell's sums and not in the other's.
  - → answer: We cannot explicitly encapsulate these edges under cell scale because they can be shared between two or more cells.
- **QA3 — scope of P0.** Is R6 (bulk creation) part of P0, or does it wait for the GRANAP adapter (P1), which will measure the need?
  - **Recommendation:** in P0, with the timings of one GRANAP-sized synthetic section (about 20 k vertices) per vertex and in bulk.
  - → answer: Yes to recommendation

## 4. Done (2026-10-12, your answers: the recommendations)

- **R1, structure:** Layers, Cells and protoplasts are true MTG components; walls and junctions are anchored to their SubOrgan. The whole hydraulic graph stays at the Compartment and Connection scales (QA1): the DataStructure's nodes are every Compartment, its edges every Connection.
- **R2, ownership:**
  - `MPG.compartment_owners()` gives each Compartment's owner, vectorised: its complex when it is a real vertex, else its topological parent;
  - `MPG.owners_at_scale(owners, scale)` gives the entity above an owner at a scale, −1 when there is none;
  - `compartments_by_owner(from_scale)` gives all Compartments of each SubOrgan.
- **R3, locations:** in anatomy mode the locations are the scales up to `from_scale`, plus Layer and Cell down to the finest scale that owns Compartments.
- **R4, walls and junctions (QA2):** their owner index at Layer and Cell is −1. Aggregations skip them, and broadcasts (parameter views, masks, MTG writes) give the declared default.
- **R5:** the junction rewiring (stale junctions) and `on_grow="inherit"` go through the owner chain (`_sections_of`). Signatures and wiring use `compartments_by_owner`.
- **R6, bulk (QA3):** `add_components_bulk` takes one complex per component.
  - Timings on a 55 × 55-cell section (45.7 k vertices: 12.3 k nodes, 30.4 k edges): creation 1.01 s per vertex, 0.17 s in bulk; `convert_properties_to_arraydict` 0.8 s (now the largest cost); DataStructure 0.06 s; owners 5 ms; graph view 10 ms.
- **Tests:**
  - `test/data_structures/mpg/test_layered_anatomy.py` (11): owners and locations; bulk = per vertex; aggregations and defaults; masks; Cell declarations and parameters in steps; a graph system solved on the layered graph; junctions; growth; differentiation; checkpoint and restore; bulk with one complex per component.
  - Helper: `test/helpers/layered_anatomy.py`. Benchmark: `test/benchmarks/bench_layered_anatomy.py`.
- **Not changed, by design:** a step receives a state variable at its own location (a Cell variable as one value per Cell), as SubOrgan variables in segment mode; only parameters are broadcast to the step's location.
