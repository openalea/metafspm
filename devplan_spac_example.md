# Plan: an example of metafspm, water in the soil–plant–atmosphere continuum

Your request (2026-10-06): one simple example that shows every feature of metafspm on water flow from the soil through a plant to the atmosphere. It must contain:
- a transport component, j = k ΔΨ, steady (hydrostatic), with axial and radial fluxes;
- a conductance component computing k from the edge types and lengths, coupled to the transport through the DataStructure;
- both components on a whole seedling with root and leaf anatomies, and on a soil grid;
- plant ↔ soil coupling: soil Ψ is a boundary of the plant, and the plant's radial uptake is a Neumann sink of the soil;
- the water table and the atmosphere as boundaries;
- conductances scaled down where water turns to vapour;
- two scenes: one plant, and a population;
- plots of the DataStructures and of the converged potentials.

Answer the questions in §4 on the `→ answer:` lines; §1–§3 follow my recommendations until then.

## 1. What the example shows, feature by feature

| feature | where in the example |
|---|---|
| a `FunctionalComponent` with declarations (scale / location, kinds, units) | `WaterTransport`, `Conductance` |
| a graph system, residual form, Newton; an edge law; a `@graph_output` | `WaterTransport`: node balance Σ j = 0, edge law j = k ΔΨ, output of the radial uptake |
| boundary sets (Dirichlet, Robin) and `@boundary_condition` as an equation | water table (Dirichlet), atmosphere (Robin, vapour conductance), soil ↔ root exchange (an equation of coupled variables) |
| a `StructuralComponent` with `initiate_plant` | `SeedlingStructure` builds each plant (all scales, Plant → SubOrgan), with root and leaf anatomies and their geometry |
| anatomy mode and junction wiring | `MPGDataStructure(nodes="Compartment", wiring=...)`: radial edges inside segments, axial xylem junctions between them |
| coupling within one DataStructure (translator, an alias or a derived variable) | `Conductance.k` → `WaterTransport.conductance`, through the plant's translator |
| one component class on two DataStructures | the same `WaterTransport` and `Conductance` on the plant MPG and on the soil `ArrayDataStructure` (cells as nodes, faces as edges) |
| coupling across DataStructures | `CrossMapping` of the root epidermis onto the soil cells (`mask=`): soil Ψ broadcast down, radial uptake summed up |
| a scalar environment | the atmosphere's Ψ, a `ScalarMapping` (or a forcing, Q8) |
| scenes | `Scene` with one plant, then a planted population, the soil as an environment model, a `SceneRecorder` |
| plots | the DataStructures (graph, anatomy, grid) and the converged Ψ on them |

## 2. The model

- **Plant** (one MPG, anatomy mode). The nodes are Compartments:
  - **root segment:** epidermis → cortex → endodermis → xylem, a chain of radial edges;
  - **leaf element:** xylem → mesophyll → stomatal cavity, radial edges;
  - **stem element:** xylem only (Q5);
  - **axial edges:** junctions between the xylems of adjacent segments.

  Segment geometry (lengths, coordinates) comes from `SeedlingStructure.initiate_plant`, oriented in the soil and air at the plant's position. An edge's length L_e is the segment length for an axial edge, and a radial distance for a radial edge (a parameter per tissue).
- **Edge types:** a `edge_type` variable on the edges, from the Connection labels (transmembrane, apoplastic, symplastic, junction), plus "gas" for the edges leaving to the air (stomatal cavity → atmosphere, soil surface → atmosphere). Each type has a specific conductance `k_s` (parameters).
- **`Conductance`:** k = f(k_s[type], L_e) (Q1), written in a step at each call. It is a state variable, read by `WaterTransport` as an input through the translator. On a grid, the faces have type "soil": k = K_sat · face_area / face_distance.
- **`WaterTransport`:** a steady balance per node, Σ_edges j = boundary terms, with j = k (Ψ_tail − Ψ_head). The transport is the same class on both DataStructures; boundary terms are declared as boundary sets, active where their selecting variable exists:
  - **plant:** the stomatal cavities exchange with the atmosphere (Robin, weight = k_gas, value = Ψ_atm); the root epidermis exchanges with the soil, as an equation `k_root_soil * (Ψ_soil - Ψ_epidermis)` of the coupled Ψ_soil (Neumann, `@boundary_condition`);
  - **soil:** the bottom layer is held at the water table (Dirichlet); the top layer exchanges with the atmosphere (Robin, gas-scaled); each cell receives minus the plants' uptake (Neumann value: the mapped uptake).
- **Liquid → gas:** edges of type "gas" get k = k_s · L_e (or / L_e) times a factor f_vap (Q3), so that the vapour step limits the flow, as stomata and the soil surface do.
- **Coupling plant ↔ soil:** a lagged fixed point. At each scene step, the soil solves with the uptake of the plants' last solve, then the plants solve with the soil's new Ψ. A steady state is reached when the step-to-step changes vanish (Q4).

## 3. Code layout (proposed)

```
examples/soil_plant_atmosphere/
  components.py   Conductance, WaterTransport, SeedlingStructure (initiate_plant), anatomy and wiring rules
  models.py       PlantModel (population model: initiators, components, translator), SoilModel and Atmosphere
                  (environment models)
  translators.py  the plant translator (k → conductance) and the scene translator (soil ↔ plant links)
  plotting.py     plant graph and anatomy, soil grid slices, potentials on both
  one_plant.py    Scene with one plant: runs to steady state, plots
  population.py   Scene with a planted population (e.g. 3 × 3), same soil: runs, plots
  README.md       what each file shows, with the figures
test/examples/test_soil_plant_atmosphere.py
                  a smoke test of both scenes (few steps, small grid), checking mass balance (Σ uptake =
                  transpiration, water table inflow = uptake + soil evaporation at steady state)
```

The user guide gets a short page pointing to it.

## 4. Questions

- **Q1 — conductance and length.** You wrote k = k_s · L_e. A conductance usually decreases with the path length (Darcy, Poiseuille: k = k_s · A / L_e), so a longer segment carries less flow for the same ΔΨ. Should it be k = k_s / L_e (k_s in m³ s⁻¹ MPa⁻¹ · m, cross-section folded into k_s), and k = K_sat · A / d on soil faces? Or k = k_s · L_e as written (e.g. radial conductance per unit length of root, which does grow with L)? **Recommendation:** both, by edge type: radial edges k = k_s · L_e (the exchange surface grows with the segment length), axial edges and soil faces k = k_s / L_e (resp. K_sat · A / d).
  → answer: yes correct my conductance formula was messed up, conductances really should be defined from the lower anatomical scale radially, not radial length.
- **Q2 — component kinds.** You called the conductance component structural. In metafspm, a `StructuralComponent` builds or edits the structure (`initiate_plant`, growth). I propose:
  - `SeedlingStructure`, a `StructuralComponent`, builds the plant (all scales, anatomies, lengths, coordinates);
  - `Conductance`, a `FunctionalComponent`, reads the edge types and lengths and writes k.

  Or should `Conductance` itself be the `StructuralComponent` that initiates the plant? **Recommendation:** the split above. Each class shows one role, and `Conductance` can then run unchanged on the soil grid, which has no plant to initiate.
  → answer: Finally make the conductance come out of SeedlingStructure the structural component. For the soil, use the same logic the difference is that the conductance equation will be different between voxels.
- **Q3 — liquid to gas.** How should conductances be scaled where water evaporates (stomatal cavity → air, soil surface → air)?
  - (a) a constant factor f_vap ≪ 1 on the "gas" edges' k (a parameter, e.g. 1e-3);
  - (b) a physical vapour conductance: the flux driven by the vapour pressure difference, with Ψ ↔ relative humidity through Ψ = (RT/V_w) ln(RH), linearised, so k_gas = g_vap · e_sat(T) · V_w / (RT · P), with g_vap the stomatal / soil-surface conductance;
  - (c) (b) with the exact non-linear relation (Newton handles it).

  **Recommendation:** (b). It keeps a linear system and the physics of the phase change, and stays simple. (a) is not physical; (c) adds little for an example.
  → answer: (b)
- **Q4 — the plant–soil fixed point.** The Scene exchanges once per step, so the coupled steady state is reached over steps (lagged). Should the example:
  - (a) run steps until the changes fall under a tolerance (`stop_when`), and show the convergence;
  - (b) iterate the plant and soil solves to convergence within each step (a coupled Picard loop in the scene)?

  **Recommendation:** (a). It uses the Scene as it is, `stop_when` shows the services, and the steady problem converges quickly.
  → answer: (a), cause I might want to show this convergence at several specific time points.
- **Q5 — anatomies.** Root segments: epidermis, cortex, endodermis, xylem. Leaf elements: xylemsem, mesophyll, stomatal cavity. Stem elements: xylem only, so that the axial path is continuous. Is this right, or should stems have an anatomy too (e.g. xylem and a parenchyma)? **Recommendation:** as proposed. The stem only carries water up.
  → answer: stem has anatomy too.
- **Q6 — the plant.** The test segmentedling (2 phytomers with 3 leaf elements each, and a root axis with a lateral) built by `SeedlingStructure.initiate_plant` under each Plant vertex, with lengths and coordinates (root down into the soil, leaves up)? Or a slightly larger plant (e.g. 3 root axes with laterals), so that the soil uptake pattern shows on the plots? **Recommendation:** a slightly larger one, about 30 segments, with parameters for the numbers of axes and segments. The figures then show the uptake spread over several cells.
  → answer: slightly larger yes, but small enough to be a seedling
- **Q7 — the population scene.** For example 3 × 3 plants on a 0.3 m × 0.3 m stand over a 0.5 m deep soil, with the same soil model. Should the plants differ (e.g. per-plant `k_s` scenarios, or an emergence time for one row)? **Recommendation:** yes, per-plant root conductances from the planting table, so the plots show plants competing for water.
  → answer: yes, and use explicitly this paremeter variation as a varying value in the input per plant scenarios.
- **Q8 — the atmosphere.** A constant Ψ_atm (e.g. from RH = 50 % at 20 °C, about −93 MPa), or a forcing table over a day (RH and temperature varying, steady states followed hour by hour)? **Recommendation:** a day of hourly steady states from a forcing table. It shows forcings and makes the plots more telling (transpiration following RH); the steady states are cheap.
  → answer: Fixed, this is just a simple convergence simulation for now.
- **Q9 — location.** `examples/soil_plant_atmosphere/` at the repository root, with a smoke test in `test/examples/` and a user-guide page? **Recommendation:** yes.
  → answer: in examples yes.
- **Q10 — plots.** I propose:
  - the plant graph laid out from its coordinates, each segment drawn with its anatomy as small radial markers, coloured by Ψ;
  - the soil grid as a vertical slice through each plant (Ψ coloured), with the roots drawn over it;
  - for the population, a top view of the uptake per column and the transpiration per plant;
  - with Q8, a day plot of transpiration and soil evaporation.

  Anything to add or remove? (matplotlib only, PNG files.)
  → answer: a simplified plant graph limiting at SubOrgan scale, a full one including anatomy, and a simplified one including only one anatomy per type (root, stem, leaf). Yes for soil grid, Yes to topview limiting at SubOrgan for the plot. No to the last plot.

## 5. Decisions from your answers (2026-10-06)

- **Conductances (Q1, Q2)** come from the structural components, computed in one of their steps from the structure.
  - `HydraulicStructure` is a base class with the conductance step, specialised by `SeedlingStructure` (the plant, which also builds it in `initiate_plant`) and `SoilStructure` (the grid).
  - `WaterTransport` is one class, used on both DataStructures, and reads k through the translator.
  - **Plant, axial edges** (xylem junctions, and the stem's): k = k_s,axial / L_segment.
  - **Plant, radial edges:** from the anatomy, the exchange surface of the tissue boundary they cross: k = k_s,tissue · 2π r_boundary · L_segment, with the tissue radii as anatomy parameters (C1).
  - **Soil faces:** k = K_face · A_face / d_face, with K_sat varying between voxels (by layer), K_face the harmonic mean of the two voxels (C2).
- **Liquid → gas (Q3, b):** linearised vapour conductance. The stomatal cavities' exchange with the air, and the top soil layer's, have weight k_gas = g_vap · A · e_sat(T) · V_w / (R T P), with Ψ_atm from RH and T (fixed).
- **Coupling (Q4, a):** lagged, until `stop_when` (the largest Ψ change between steps under a tolerance). The recorder keeps every step, so that the convergence can be shown at chosen steps.
- **Anatomies (Q5):**
  - root: epidermis, cortex, endodermis, xylem;
  - stem: epidermis, cortex, xylem (no gas exchange);
  - leaf: xylem, mesophyll, stomatal cavity → air.
- **Plant (Q6):** a seedling a little larger than the test one (about 20–30 segments). The numbers of root axes, laterals and segments are parameters.
- **Population (Q7):** 3 × 3 plants. The root radial `k_s` varies per plant through the planting table's per-plant scenarios.
- **Atmosphere (Q8):** fixed RH and T, as an environment scalar.
- **Location (Q9):** `examples/soil_plant_atmosphere/`, with a smoke test.
- **Plots (Q10):**
  - the plant graph at SubOrgan scale;
  - the full graph with the anatomies;
  - a simplified graph with one anatomy per organ type (root, stem, leaf);
  - the soil grid slice with the roots;
  - a top view at SubOrgan scale.

  All coloured by Ψ; no day plot.

### To confirm

- **C1 — radial conductances.** I read "from the lower anatomical scale radially" as: each radial edge between two tissues gets k_s of the tissue boundary times that boundary's surface along the segment (2π r · L). So outer tissues exchange over larger surfaces, and L enters as a surface, not as a path length. Right?
  → answer: No, just apply conductances at each anatomical edge and then just extrapolate 3D from SubOrgan length.
- **C2 — soil voxels.** I read "the conductance equation will be different between voxels" as: the soil's face convergenceductance is its own equation (K · A / d), with K_sat varying between voxels (e.g. a denser lower layer) and averaged across each face. Or did you mean another equation per voxel type?
  → answer: Yes this equation.

## 6. Done (2026-10-06)

- **`examples/soil_plant_atmosphere/`:**
  - `components.py`, `models.py`, `plotting.py`, `one_plant.py`, `population.py`, a README with the figures (`figures/`);
  - `test/examples/test_soil_plant_atmosphere.py` runs both scenes and checks the closed water balances: plant uptake = transpiration = water taken from the soil, and water-table inflow = uptake + soil evaporation. It also checks the records and the five figures.
- **One plant** (33 segments, 120 Compartments): converges in 5 steps; leaf Ψ about −2 MPa, transpiration 0.076 mm³ s⁻¹.
- **Population:** `planting_table` lays out 8 plants on 0.2 m × 0.3 m (its rows, not 3 × 3), with per-plant `root_radial_k` (0.2, 0.5, 1.0) from `per_plant_scenarios`. It converges in 6 steps.
- **Framework changes the example needed:**
  - `location="node"` is a grid's cells, so one component class runs on plants and soil;
  - `scale=Connection` in anatomy mode reads and writes the Connections' properties;
  - an explicit `CrossMapping` replaces the inferred one (for `mask=`);
  - `CompositeModel` takes Translator objects;
  - grids have `dx`;
  - scalars accept one-value arrays;
  - node outputs on grids;
  - `Layer.Mesophyll` / `Layer.StomatalCavity` labels.

### Open points

- **E1 — the solver's tolerance is absolute** (`tol=1e-10` on the residual). In m³ s⁻¹ the residuals were about 1e-11, so Newton stopped at the initial guess without any warning; the example works in mm³. Add a relative criterion (the residual against its initial value, or per block against a scale), or a warning when the first residual is already under the tolerance? **Recommendation:** both — `rtol` on the residual relative to its initial norm (default on), and a warning when a solve converges in zero iterations from a non-trivial state.
  → answer:
- **E2 — plants competing for water.** With a dry air (Ψ_air ≈ −94 MPa) and a linear vapour exchange, the vapour step dominates: transpiration hardly depends on the roots, and the competition shows in leaf Ψ, not in fluxes. To make root differences change fluxes, should stomatal conductance close with leaf Ψ (a g_s(Ψ_leaf) law, non-linear, still solved by Newton)? **Recommendation:** yes, a simple sigmoid closure. It is one more equation of a coupled variable in `@boundary_condition` (then the atmosphere exchange is written as an equation instead of a boundary set).
  → answer:
- **E3 — `wiring=` and Connection properties.** The example builds its junctions in `initiate_plant`, because the Connections wiring creates have no properties (type, length) for the conductance step. Let wiring rules give properties to the junctions they create (constants, or a callable of the two segments)? **Recommendation:** yes, `properties=` in a rule. The example could then use `wiring=`.
  → answer:
