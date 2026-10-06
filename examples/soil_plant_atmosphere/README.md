# Water in the soil–plant–atmosphere continuum

A small, complete example of metafspm: steady water flow from a water table, through the soil and seedlings, out to a
dry atmosphere. Every part shows a feature of metafspm:

| file | what it shows |
|---|---|
| `components.py` | **`PlantWaterTransport`** and **`SoilWaterTransport`**, two `FunctionalComponent`s, each declaring its own variables and graph system: a node balance, an edge law given explicitly (j = k ΔΨ), and every boundary written as an equation (`@boundary_condition`): the exchange with the air, k_vap (Ψ_air − Ψ), each selecting its nodes with explicit `filters=` key / values. The plant's are at the Compartment and Connection scales, read from and written to the MTG; it adds the root–soil exchange, an equation of the coupled soil Ψ (`@boundary_condition`). The soil's are at cell and edge locations; it adds the water table (Dirichlet, Ψ − Ψ_table) and the plants' uptake. `@graph_output`s give the uptake and the evaporation. |
| `seedling.py` | **The seedling generator**, standing for the structural models (growth, anatomy) a real simulation would couple: the architecture at every scale (three first-order roots from the collar, each with a lateral: a nearly vertical pivot, and two roots bending towards the vertical with a gravitropism coefficient), the anatomies and the axial junctions, with the codes the components read. |
| | **`SeedlingStructure`** (in `components.py`), a `StructuralComponent`: `initiate_plant` builds each seedling (with `seedling.py`) at every scale (Plant → Axis → GrowthUnit → Phytomer → Organ → SubOrgan) with an anatomy per segment (root: epidermis, cortex, endodermis, xylem; stem: epidermis, cortex, xylem; leaf: xylem, mesophyll, stomatal cavity) and the axial xylem junctions, then computes the conductances from the structure: k = k_s · L on radial edges, k = k_axial / L on axial ones. **`SoilStructure`** does the same for the soil grid (k = K · A / d on the faces, K varying between voxels). |
| | **The air**, a constant input: Ψ_air = (RT/V_w) ln RH at 50 % and 20 °C (about −93.9 MPa), and the vapour factor that turns a vapour conductance into a liquid one where water evaporates (the liquid–vapour step, linearised). |
| `models.py` | **Models and translators.** The plant population model (`SeedlingWater`: initiators, anatomy mode) and the soil environment model. Translators inside a DataStructure (k → conductance), and across DataStructures (soil Ψ → root epidermis, uptake → soil cells). A `CrossMapping` of the root surface only (`mask=`). A `stop_when` condition on the change of Ψ between steps. |
| `one_plant.py` | **A Scene with one seedling**, run until the lagged plant–soil fixed point converges, with a `SceneRecorder`. |
| `population.py` | **A Scene with a planted stand.** `planting_table` lays out the plants, and each plant's root radial conductance comes from its own scenario (`per_plant_scenarios`). |
| `upscaling.py` | **Upscaling** the solved water potentials from the Compartments to the plants, scale by scale (SubOrgan, Organ, Phytomer, GrowthUnit, Axis, Plant), each the mean of the scale below: one derived variable per scale (`ds.derive(..., location=scale, aggregation="mean")`), recomputed when the potentials change. |
| `plotting.py` | **Plots** of the DataStructures and of the converged Ψ: the plants at SubOrgan scale, the full graph with the anatomies, one anatomy per organ type, a soil slice with the roots, a top view of the uptake. |

Run, from this folder:

```
python one_plant.py            # writes to outputs/one_plant
python population.py           # writes to outputs/population
```

Units: water potentials in MPa, volumes in mm³ (fluxes in mm³ s⁻¹, conductances in mm³ s⁻¹ MPa⁻¹), lengths in m.
Volumes in mm³ keep the solves' residuals well above the solver's absolute tolerance.

## One plant

The scene converges in six steps (largest change of Ψ: 7.9e-01, 1.1e-02, 3.1e-04, 1.3e-05, 6.5e-07 MPa). Transpiration
(0.076 mm³ s⁻¹) equals the root uptake and the water taken from the soil; the water table supplies it and the soil
evaporation.

![plants at SubOrgan scale](figures/one_plant/plant_segments.png)
![the plant with its anatomies](figures/one_plant/plant_anatomy.png)
![one anatomy per organ type](figures/one_plant/anatomy_types.png)
![soil slice](figures/one_plant/soil_slice.png)
![top view](figures/one_plant/top_view.png)

### Upscaling, step by step

The solved potentials of the 164 Compartments are averaged into the 44 segments, then the 12 organs, the 6
phytomers, the 4 growth units, the 4 axes and the plant: every scale of the MPG (`upscaling.py`). Each step is one
`derive`; every graph uses the same colour scale.

![1. Compartments](figures/one_plant/upscale_1_compartment.png)
![2. SubOrgan](figures/one_plant/upscale_2_suborgan.png)
![3. Organ](figures/one_plant/upscale_3_organ.png)
![4. Phytomer](figures/one_plant/upscale_4_phytomer.png)
![5. GrowthUnit](figures/one_plant/upscale_5_growthunit.png)
![6. Axis](figures/one_plant/upscale_6_axis.png)
![7. Plant](figures/one_plant/upscale_7_plant.png)

## A population

The plants share the soil; their root radial conductances cycle through 0.2, 0.5 and 1.0. Under a dry air
(Ψ_air ≈ −94 MPa), the vapour step limits the flow, so transpiration varies little between plants. Their leaf water
potentials absorb the difference: the least conductive roots give the lowest leaf Ψ.

The side views show the first planting row (four plants, seen along the row, in the y–z plane).

![plants](figures/population/plant_segments.png)
![soil slice](figures/population/soil_slice.png)
![top view](figures/population/top_view.png)

The upscaling series of the first row's plants: [Compartments](figures/population/upscale_1_compartment.png), [SubOrgan](figures/population/upscale_2_suborgan.png), [Organ](figures/population/upscale_3_organ.png), [Phytomer](figures/population/upscale_4_phytomer.png), [GrowthUnit](figures/population/upscale_5_growthunit.png), [Axis](figures/population/upscale_6_axis.png), [Plant](figures/population/upscale_7_plant.png).

## Notes

- The axial junctions are built by `initiate_plant` with their own properties (type, length). `MPGDataStructure(...,
  wiring=rules)` can wire junctions too, but the Connections it creates carry no such properties.
- The plant–soil coupling is lagged: each scene step solves the soil with the plants' last uptake, then the plants
  with the soil's new Ψ, until `stop_when` holds.
