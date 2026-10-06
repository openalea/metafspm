# Design note: cross-scale links and grid topology (step 3)

Status: **agreed** (2026-10-02, R1–R5 as recommended). 3a done (`40657a2`), 3b done (`1f6c926`), 3c done (`ceb6e2f`), 3d done: **step 3 complete**. It covers step 3 of `devplan_datastructures.md` §7:
- DS18: cross-scale links in the translator (mappings, defaults, filtered broadcast);
- DS1: graph systems on any DataStructure, with grid topology.

Branch `data_structure_api`, written against `b8e9fba` (steps 1 and 2 complete). Code starts only after this note is agreed. The points to agree are in §5, with answer lines.

It builds on decisions D1 (grid faces as edges, with a geometric factor as an edge variable) and D9 option A (default mappings from `state_variable_type`), on the step 1 contract (locations and mappings) and on step 2e's boundary sets.

## 1. Where we start from (code facts)

**Links (DS18).**
- `CompositeModel._couple_on_data_structures` turns every factor, sum or formula link within one DataStructure into `ds.derive(variable, sources, location=<the receiver's registered location>, aggregation=link.aggregation, weight=link.weight)`. When the sources' location differs from the receiver's and the link gives no aggregation, `derive` raises "needs an aggregation".
- **What step 1 changed:** a receiver declared at a coarse scale (`scale=scales.Organ`) is now stored at `"Organ"`, so a SubOrgan output feeding it is a real scale change, which `derive` maps with the operators of `_map`:
  - node ↔ edge: `child` / `parent` / `mean`;
  - node → coarse: `sum` / `mean` / `weighted_mean`;
  - coarse → node: `broadcast`;
  - any ↔ scalar.

  **Coarse ↔ coarse** (e.g. Organ → Plant at Axis scale) is not supported.
- `Link.scale` and `Link.source_scale` are parsed (live `scales.*` or names) but only mark the link as `scale_change`. The coupling never reads them: locations come from the declarations.
- `input_variable` takes no `state_variable_type`, so a receiving input has no kind to check against its provider's.
- A variable cannot have the same name at two locations of one DataStructure. Step 1a raises when a second component declares it at another location. Same-name identity links across scales are therefore impossible: the receiver must use another name, linked by the translator.
- Links across DataStructures (plant ↔ soil) go through the Coupler and `Transport` (voxel mapping), and are outside this step.

**Grids (DS1).**
- `ArrayDataStructure(shape, dx, origin)` has the `"cell"` (shape `(nx, ny, nz)`) and `"scalar"` locations, and a finite-difference `laplacian()` (Neumann boundaries).
- It has no `to_graph_view`, so `FunctionalComponent._graph_view` is `None` on grids and `@graph_system` cannot run there. Soil transport is written by hand, or delegated to cmf in the reference model.
- **Reference soil model** (`test/provide_usage_examples/legacy/rhizosoil_core_model.py:503`): lateral neighbours wrap around in x and y when the scene is symmetric (`symetry`). The bottom layer has either no flux (pot) or a fixed groundwater moisture (field), and water and solute uptake are Neumann sources per cell.

## 2. Cross-scale links (DS18), sub-steps 3a–3b

**Default mapping (D9, option A).** When a link joins two locations and gives no `aggregation`, the mapping follows the **provider's** recorded `state_variable_type`:

| provider kind | up (node → coarse, coarse → coarser, → scalar) | down (coarse → node, coarser → coarse) |
|---|---|---|
| extensive, NonInertialExtensive | `sum` | error: give `aggregation` |
| intensive, NonInertialIntensive | `mean` | `broadcast` |
| massic_concentration | `weighted_mean` (the link's `weight` is required) | `broadcast` |
| missing, or sources of different kinds | error | error |

- Node → edge links have no default.
- The error names the link (`receiver.variable <- provider`), the two locations and the provider's kind.

**Kind agreement.** `input_variable(..., state_variable_type=...)` becomes optional. When the receiver declares a kind, it must belong to the provider's family: extensive with extensive, intensive or massic with intensive or massic. A mismatch raises at coupling, and is reported by `assert_component_couplable`. Without a declared kind, nothing is checked, as today.

**Coarse ↔ coarse mappings.** `_map` gains coarse → coarser (aggregation over the owner at the target scale, `complex_at_scale` of each source entity) and coarser → coarse (broadcast). Organ → Axis, Organ → Plant and SubOrgan → Organ in anatomy mode then all work. Scalars remain the plant scale.

**`Link.scale` / `source_scale` become checks (R1).** When a link states them, they must match the receiver's and the sources' declared locations; otherwise the coupling raises. They document a scale change in the translator file, but declarations stay the reference.

**Filtered broadcast in the translator (R2).** A link may give `target="<mask name>"`, passed to `derive(target=)` (step 2f, A2 option 1): the values go to the mask's entities, and the others get the derived variable's default. The mask must exist on the DataStructure when components are coupled. It is defined by a component (e.g. `active`) or by the composite before coupling.

**Validation:**
- every row of the default table, for the up and down directions;
- the kind-agreement errors;
- coarse ↔ coarse mappings on the seedling (Organ → Axis sums, Axis → Organ broadcasts);
- a composite whose translator links a SubOrgan output to an Organ input without an aggregation, and gets the sum;
- a translator `target` link;
- `assert_component_couplable` reporting a missing default.

## 3. Graph systems on grids (DS1, D1), sub-steps 3c–3d

**Topology.** `ArrayDataStructure.to_graph_view()` (and `topology()`, a common name on both DataStructures):
- **nodes:** the cells, in flat C order (`entity_ids("cell")`);
- **edges:** the internal faces between adjacent cells, axis by axis (x faces, then y, then z), each in C order of its lower cell;
- **orientation:** tail = the lower-index cell, head = the upper one. So `B[lower, e] = +1`, `(Bᵀc)_e = c_lower − c_upper`, and a positive edge flux goes towards increasing coordinates. This matches the graph convention (+1 at the tail);
- **periodic axes (R3):** `ArrayDataStructure(..., periodic=(True, True, False))` adds the wrap faces (last cell → first cell) along those axes, as the reference model does in x and y for symmetric scenes;
- **outer faces** are not edges, so an outer boundary has no flux by default (Neumann 0). Dirichlet, Robin and flux conditions are boundary sets on boundary cells (step 2e), e.g. `select=lambda ds: ds.layer_mask(z=-1)` for the bottom groundwater layer.

**Locations on grids.**
- A new `"edge"` location holds one value per face.
- `"cell"` plays the role of `"node"` in graph systems: snapshots flatten cell arrays to `(n,)`, and writes reshape them back (C order). Masks and boundary sets on cells work the same way.
- **Geometric factor (D1).** At construction, the grid registers two edge variables: `face_area` (the product of the other two `dx`) and `face_distance` (the `dx` of the face's axis). A Fickian or Darcian edge law reads `K · face_area / face_distance · (Bᵀc)`, and a balance divides by `cell_volume()`.
- **Helpers:** `face_axis()` (an int per face) and `layer_mask(x=, y=, z=)` (cells of given layers), for boundary sets and anisotropy.

**Validation:**
- `B · diag(face_area / face_distance) · Bᵀ / cell_volume == −laplacian()`, exactly, on 1-D, 2-D and 3-D grids (non-periodic);
- periodic faces counted and oriented;
- a soil diffusion written as a `@graph_system` with implicit Euler on a 3-D grid, matching the direct sparse solve of `(I − dt·L)·c = c_old`;
- a bottom Dirichlet layer (field) and a no-flux bottom (pot) through boundary sets;
- `where="active"` on cells (e.g. a frozen layer);
- the `GridSoil` double and the soil contract tests unchanged.

## 4. Sub-steps

| Step | Content |
|---|---|
| 3a | Default mappings and kind agreement in `_couple_on_data_structures` and `couplability_problems`; `input_variable(state_variable_type=)`; coarse ↔ coarse in `_map` |
| 3b | `Link.scale` / `source_scale` checks; translator `target=` |
| 3c | Grid topology: faces, periodic axes, `face_area` / `face_distance`, the `"edge"` location, `face_axis`, `layer_mask` |
| 3d | Graph systems on grids: cell ↔ node flattening in the builder, masks and boundary sets on cells, the soil diffusion test |

Each sub-step lands with its tests and a commit; UC1–UC5 and the contract tests stay unchanged.

## 5. Points to agree

- **R1, link scales as checks.** `Link.scale` / `source_scale` must agree with the declared locations, otherwise the coupling raises. The alternative is to let a link's `scale` override a declaration. **Recommendation:** checks only; declarations stay the reference.
  → answer: agree with Recommendation
- **R2, `target=` on translator links**, naming a DataStructure mask defined by a component or by the composite before coupling. **Recommendation:** yes. The other option is to keep filtered broadcasts out of the translator, as `ds.derive` calls in the composite.
  → answer: agree
- **R3, periodic grid axes** as a construction option of `ArrayDataStructure` (`periodic=(True, True, False)`), adding wrap faces. **Recommendation:** yes, matching the reference model's `symetry`. It also becomes the default `periodic` of `locate`.
  → answer: yes
- **R4, face orientation:** a positive flux goes towards increasing coordinates, so towards increasing depth along z if z increases downwards in the grid frame. **Recommendation:** yes, with the sign documented in the conventions page. Models choose the sign of gravity terms accordingly.
  → answer: yes agree with Recommendation
- **R5, order and scope:** 3a → 3b → 3c → 3d; `MultiGridDataStructure` topology stays later (DS1 says "later"). **Recommendation:** yes.
  → answer: yes
