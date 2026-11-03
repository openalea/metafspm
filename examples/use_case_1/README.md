# Use case 1: a wheat plant in its soil and its light

A soil–plant–atmosphere assembly that tests metafspm before the release: ArchiSimple's roots, Adel's shoot, GRANAP's
anatomies, MECHA's water flow, CNW-Grass's gas exchange, Caribu's light, and the soil of this folder. The
environment is `environment.yaml` (`mamba env create -f environment.yaml`, the repositories cloned side by side).

## The soil (`component.py`)

`RichardsSoil` is the transient water flow of Richards' equation on a soil grid: a regular `ArrayDataStructure` or
an `AdaptiveGridDataStructure`.

- **Equation.** The mixed form (Celia et al. 1990), the pressure head h (m) as each cell's unknown:
  (θ(h) − θ(h_prev)) V + dt (Σ q + uptake − boundary inflows) = 0, written per cell volume.
- **Fluxes.** On each face, q = K_f A (Δh/d + g), with g = 1 on the z faces (z points down from the surface).
  K_f is upstream-weighted by default; the arithmetic mean is a setting (`face_conductivity`).
- **Soil.** van Genuchten–Mualem, with θr, θs, α, n, Ks and l per cell, set by layers (`set_soil_layers`). A loam
  by default (Carsel & Parrish 1988's classes in `SOILS`). Gardner's exponential soil is a setting
  (`retention_model="gardner"`).
- **Boundaries.**
  - The top: `rainfall` − `potential_evaporation` (inputs that a Scene's forcings of the same names write). Both
    are limited by what the soil can take or give: the rain it cannot take runs off.
  - The bottom: free drainage, a water table (`water_table_head`) or no flux (`bottom_boundary`).
  - The sides: the grid's (periodic over the stand).
- **Solver.** Newton with the analytic Jacobian, and adaptive time steps (step doubling).
- **Water balance.** A second unknown, `exchanged_water`, integrates what crosses the boundaries and what the roots
  take, so Σ θ V − Σ θ₀ V = Σ exchanged_water V over any run.

`RichardsSoilModel` is the soil of a Scene: the grid over the stand's pattern, adaptive by default.

- After each run, it refines the grid where the roots take up water or where h changes steeply, and coarsens it
  elsewhere.
- **Carry-over.** When the grid changes, the water is carried over as a volume (θ V), not as h. h is then recomputed
  from θ by the inverse retention curve (`RichardsSoil.carry_over`). metafspm has no hook for such a derived
  carry-over, so the model does it after its refinement.
- **Links with the plants.** `scene_translator()` gives them:
  - MECHA's `root_uptake` (cm3 d-1) summed into the cells' `plant_uptake` (m3 s-1);
  - the soil's `water_potential` (MPa) sent back to MECHA's `soil_psi` (hPa).

Its tests are in metafspm's `test/examples/test_use_case_1_soil.py`:

- the soil curves and the Jacobian;
- the hydrostatic equilibrium;
- a steady drainage to a unit gradient;
- the water balance with a root sink and regridding;
- Srivastava & Yeh (1991)'s exact infiltration towards a water table.
