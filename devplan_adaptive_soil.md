# Plan: an adaptive soil in the SPAC example

Your request (2026-10-07): add a soil model, wrapped as a CompositeModel alongside `Soil`, on an `AdaptiveGridDataStructure`. Show with plots how voxel size varies through the scene and how it relates to flux intensity, if that is the right metric. The run goes in `one_plant_adaptative.py`.

## 1. Is flux intensity the right metric?

**Not on its own.** A finite-volume two-point flux resolves a linear Ψ profile exactly at any cell size. The discretisation error comes from the curvature of Ψ, i.e. from places where the flux changes over space, not from places where it is large.
- **Large flux, linear profile:** the vertical column fed by the water table carries a strong, nearly uniform flux with an almost linear Ψ. Refining there would cost cells and gain nothing.
- **Where the flux changes:** in steady flow, the flux divergence in a cell equals its sources (`div j = −uptake`, plus evaporation at the surface). So the curvature of Ψ sits where the roots take up water: the sink density, uptake per volume, of order K·∇²Ψ.

**Candidate indicators:**

| indicator | what it is | for | against |
|---|---|---|---|
| \|j\| (flux intensity) | the face fluxes, per cell | easy to explain | refines linear zones (the water-table column) and misses the sinks |
| sink density | `plant_uptake / cell_volume` | exact for this model (the FV divergence), physically readable: refine where the roots take up | a priori: blind to other sources of curvature, e.g. K jumps between layers |
| gradient jump | per cell, the difference between the flux densities through its two opposite faces along each axis (a second difference) | a posteriori, general (sinks, K jumps, boundaries) | less readable; close to the sink density here |

**Recommendation:** refine on the sink density, coarsen where it is negligible. The plots would show voxel size against both |j| and sink density, to make the point visible: refined cells sit at high sink density, while |j| is high in coarse cells too.

## 2. Design

- **`AdaptiveSoil(CompositeModel)`** in `models.py`, next to `Soil`:
  - the same components (`SoilStructure`, `SoilWaterTransport`) and translator;
  - an `AdaptiveGridDataStructure(shape=base, dx=0.025, max_level=2)`, so cells go from 2.5 cm down to 0.625 cm;
  - its `run()` adapts the grid before its components run: refine where the sink density is high, coarsen where it is low, from the previous step's uptake. The coupling is already lagged, so adaptation is a fixed point too, and the scene stops when Ψ and the grid no longer change.
- **`SoilStructure` made grid-agnostic** (shared by both soils):
  - surface and water table from `layer_mask(z=0)` / `layer_mask(z=-1)`;
  - each cell's evaporating area from its own size;
  - `K_sat` by depth from `cell_centers()` without the reshape;
  - layer masks and areas recomputed after each refinement.
- **Framework fix (needed):** `CrossMapping` rebuilds its incidence when the source (plant) topology changes, but not the target grid's. After a refinement it would map roots onto stale cells. The target's topology version goes into its stamp, with a test.
- **`one_plant_adaptative.py`:** the one-plant scene with `AdaptiveSoil`, its summary, its figures, a smoke test.

## 3. Plots (proposed)

1. **Voxel-size slice:** a vertical slice through the plant, each cell drawn at its own size (rectangles, not pixels) and coloured by its size, with the roots over it.
2. **Soil ΔΨ on the adaptive cells:** the same slice coloured by Ψ minus its layer mean, which shows the refinement following the depletion.
3. **Size vs metric:** per cell, sink density and |j| against cell size (one panel each, cells at each level as a strip, log scale). The point of the figure: refinement follows the sinks, not the flux.
4. **Cost and accuracy:** cell counts (adaptive vs uniform at the finest size), and plant uptake and leaf Ψ against the uniform 2.5 cm and 0.625 cm grids.

## 4. Questions

- **Q1 — metric.** Refine on sink density (recommended), on |j| as you suggested, or on the gradient jump? I can also show |j| as the refinement criterion, as a counter-example.
  → answer: density yes
- **Q2 — depth.** Base 2.5 cm and `max_level=2` (down to 0.625 cm)? The uniform 0.625 cm reference takes about 100 s to run. **Recommendation:** yes. The reference is computed only by `one_plant_adaptative.py`, not in the smoke test.
  → answer: yes to Recommendation
- **Q3 — thresholds.** Refine where the sink density exceeds a fraction of its maximum (e.g. 5 %), coarsen below 0.5 %. Fractions are scale-free; absolute values would be in mm³ s⁻¹ m⁻³. **Recommendation:** fractions, as `AdaptiveSoil` parameters.
  → answer: fractions
- **Q4 — shared `SoilStructure`.** Make it grid-agnostic, as in §2, rather than adding an `AdaptiveSoilStructure`? **Recommendation:** shared: one declaration for both grids is the framework's point.
  → answer: shared.

## 5. Done (2026-10-07)

- **`AdaptiveSoil`** (`models.py`): the components of `Soil` on an `AdaptiveGridDataStructure` (2.5 cm, `max_level=2`). After each solve it refines where the sink density exceeds `refine_fraction` (5 %) of its maximum and coarsens below `coarsen_fraction` (0.5 %), then resets the layers and K_sat. The grid settles after two adaptations at 930 / 678 / 592 cells (2.5 / 1.25 / 0.625 cm), 2200 cells against 65 536 for the finest uniform grid. The scene converges in 8 steps with closed balances.
- **`SoilStructure.set_layers()`**: the surface, the water table and the evaporating areas, from the geometry of either grid.
- **Framework:**
  - `CrossMapping` now rebuilds when the target grid's topology changes. Before, roots were mapped onto stale cells after a refinement (tested).
  - `cell_sizes()` on both grids.
- **`one_plant_adaptative.py`:** the scene, plots 1–3 of §3, and the comparison with uniform 2.5 cm and 0.625 cm grids. A smoke test runs it without the reference.
- **Plot 4 changed:** leaf Ψ does not converge with the cell size (−2.646, −2.687 and −2.712 MPa at 2.5, 1.25 and 0.625 cm), for two reasons (F2 below, and the root-surface Ψ around a line sink falling as cells shrink). The comparison shows the depletion error near the roots instead.

## 6. Findings and questions

- **F1 — faces between cells of different sizes (accuracy).**
  - **What fails:** the adaptive grid's depletion error near the roots equals the uniform 2.5 cm one (0.0074 MPa against the 0.625 cm reference). A horizontal refined block brings it down to 0.0004.
  - **Cause:** the two-point flux through a face uses the two centres' distance along the face's axis only (`face_distance` = half sizes). Across a vertical coarse/fine interface, the centres also differ in depth (by h/4), and the vertical gradient (about 3.5 MPa m⁻¹) leaks into the lateral flux. Refining one more ring of neighbours makes it worse: more such interfaces.
  - **Fix:** a consistent flux at coarse/fine faces. The coarse value is interpolated at the fine cell's position, i.e. the coarse centre value plus the coarse cell's gradient times the offset, the gradient from its coarse neighbours. The grid would expose it as a face gradient operator `G` (faces × cells, sparse): `G @ Ψ` is ΔΨ/d for every face, exact for linear Ψ. A model's edge law then writes `j = K A (G @ Ψ)` instead of `incidence.T @ Ψ / d`. On regular grids, and between equal cells, `G` is the two-point difference, so results do not change.
  - **Recommendation:** add `face_gradient()` to both grids, and use it in the example's Darcy law.
  - → answer:
- **F2 — the water table's depth depends on the bottom cells' size.**
  - **Cause:** the Dirichlet condition holds the bottom cells' *centres* at Ψ_table, i.e. h/2 above the true bottom (1.25 cm at 2.5 cm, 0.31 cm at 0.625 cm). This shifts the whole vertical profile between grids, and with it leaf Ψ.
  - **Options:**
    - (a) keep the Dirichlet, but at the bottom face: the water table becomes an inflow through the bottom face, k_table (Ψ_table − Ψ), with k_table = K A / (h/2). It is still a `@boundary_condition` (Neumann), and its position is exact on every grid.
    - (b) keep the cell-centre Dirichlet and refine the bottom layer to the finest size.
  - **Recommendation:** (a). You asked for a Dirichlet here earlier ("why not dirichlet?"), so this is your call: (a) gives the same physics, a potential held at the bottom face, written as the flux through it.
  - → answer:
