# metafspm in five minutes

This page is for those who want to **use** plant models built with metafspm: students, experimenters, modellers new
to these models. It explains the core ideas through one real assembly, use case 1 (`examples/use_case_1`): wheat
plants growing from their sowing, hour by hour, with the water of their soil and their light. Open its files beside
this page:
`run.py` (the run), `models.py` (the plant model and its links), `component.py` (the soil) and `README.md` (every
detail).

## What metafspm is for

A **functional-structural plant model** (FSPM) describes a plant as it grows in space: its **structure** (roots,
stems, leaves, each cut into small elements) and its **functions** (growth, the flows of water and sugars, the light
absorbed). Many FSPMs exist, each specialised in one process. metafspm lets them run together, with their soil and
their atmosphere, in one **scene**.

Use case 1 couples seven models this way. Adel grows the shoot and ArchiSimple the roots; GRANAP builds the
anatomy of every section; MECHA computes the water flow from the soil to the leaves; CNW-Grass computes the leaves'
gas exchange; Caribu computes the light. Richards' equation on a grid gives the soil's water. None of these models
knows about the others: metafspm passes each the values it needs, in its own units, at each time step.

What use case 1 does not do yet: its growth is prescribed. Adel grows the shoot on thermal time, and ArchiSimple
grows the roots at their potential, on the soil temperature. So growth responds neither to the soil's water nor to
the light. No carbon is allocated either: the leaves' photosynthesis feeds no other model. The water flow and the
light are computed every hour, in the plants as they grow.

## Components: one process each

Each model is a **component**, a Python class that declares its variables and computes them. Each declaration
carries a unit and a place. Here are a few lines of the soil (`component.py`, `RichardsSoil`):

```python
pressure_head: float = state_variable(unit="m", description="Pressure head of the soil water (negative: "
                                                            "suction).", default=-1., location="cell",
                                      kind="intensive")
water_flux: float = state_variable(unit="m3 s-1", description="Flux through a face, ...", default=0.,
                                   location="edge", kind="extensive")
plant_uptake: float = input_variable(unit="m3 s-1", description="Water taken up by the roots from a cell.", ...)
saturated_conductivity: float = parameter(unit="m s-1", description="Saturated hydraulic conductivity Ks.", ...)
```

- A **state variable** is what the component computes (the pressure head in each cell, in m).
- An **input** is what it receives from another component (the roots' uptake, m3 s-1).
- A **parameter** is a number of the model, with its default, its bounds and its references.
- `location=` (or `scale=` on a plant) says where the variable lives: in each soil cell, on each face between two
  cells, on each root segment. `kind=` says how it adds up: extensive quantities are summed (a flux), intensive
  ones averaged (a potential).

The equations are methods of the component. A simple process is a function of the variables, marked `@rate` or
`@state`. The soil's balance is a system of equations over the grid, `@graph_system`, which metafspm solves (here
by Newton's method). Running a model does not need these methods: the declarations say what goes in and out.

## DataStructures: where the variables live

The values are not stored in the components. They are columns of a **DataStructure**, one value per entity:

- a plant population's is an **MPG**, a multiscale graph of the plants (`MPGDataStructure`): the scales Plant,
  Axis, ..., Organ and SubOrgan (the elements), and in use case 1 the Compartments of each section's anatomy, the
  nodes MECHA's water flows between;
- the soil's is a grid of cells (`ArrayDataStructure`, or `AdaptiveGridDataStructure`, refined where the roots
  take up water, as here).

Steps receive whole arrays: the same component runs on one plant or on many.

## Plant models: components on one DataStructure

A **plant model** gathers the components of one plant and runs them in order. Use case 1's is `WheatPlant`, a
`CompositeModel` (`models.py`):

```python
class WheatPlant(CompositeModel):
    initiators = (WheatCrown, ScenarioShoot, RootArchitecture, GranapAnatomy)
    parts = {"crown": WheatCrown, "shoot": ScenarioShoot, "rule": NodalRootEmergence, "roots": RootArchitecture,
             "anatomy": GranapAnatomy, "barriers": Development, "optics": PlantOptics, "gas_exchange": GasExchange,
             "hydraulics": MechaHydraulics}
    translator = PLANT_TRANSLATOR
```

- The **initiators** build each plant at its sowing: the seed and the crown, then Adel's shoot, ArchiSimple's
  seminal roots and GRANAP's anatomies.
- The **parts** are the components; each step runs them in this order: the shoot grows, then the roots (the nodal
  roots asked by a rule), the new sections get their anatomy and the roots their barriers, then come the optics,
  the gas exchange and MECHA's water flow.
- A part can itself be a plant model: Adel's `ScenarioShoot` is nested here.

## Links: the Translator

Components exchange values through **links**, gathered in a `Translator`. A link names the receiver and its
variable, the provider and its source, and the factor between their units. Two of use case 1's:

```python
# within the plant: each organ's transpiration (mmol s-1, summed over its elements) into MECHA (cm3 d-1)
.link("MechaHydraulics", "leaf_transpiration", "GasExchange", {"transpiration": WATER_MOLAR_VOLUME},
      source_scale="SubOrgan", scale="Organ", aggregation="sum")
# between the soil and the plants (component.scene_translator, its defaults): the soil's water potential (MPa)
# to MECHA's roots (hPa)
.link("MechaHydraulics", "soil_psi", "RichardsSoil", {"water_potential": 1e4}, mapping=ROOT_SURFACE)
```

metafspm **checks every link** when the models are built: the two variables must be the same quantity, the factor
must be their units' conversion (1e4 from MPa to hPa), and both must be extensive, or both intensive. A wrong factor
raises an error and the run does not start. Where the quantity itself changes (an amount of water into a volume),
the factor is a named equivalence (`WATER_MOLAR_VOLUME`, `RADIUS_OF_DIAMETER`) whose number metafspm computes
from the units. A link can also change scale (the elements' values summed per organ) or map the plants onto the
soil cells (`mapping=`: the roots' surfaces overlapping each cell). Between the plants and the environment,
variables of the same name are linked by default (Caribu's absorbed light, `PARa`).

## The Scene

The **Scene** puts the plants in their environment and runs them. Here is the core of `run.py`:

```python
plants = stand(PLANTS, row_spacing=0.15, plant_spacing=0.05, depth=0.03, rotation=120.)

scene = Scene(
    planting(plants, WheatPlant, WHEAT_SCENARIO),
    environment=[RichardsSoilModel, CaribuLightModel],
    environment_scenarios=[SOIL_SCENARIO, LIGHT_SCENARIO],
    translator=SCENE_TRANSLATOR,
    forcings=weather(START),
    time_step="1 h",
    start="2026-05-10 00:00",
    compute=Compute(threads="auto"),
    recorder=SceneRecorder(FOLDER, tables=TABLES, frame_period="12 h", frame_views="default", ...),
)
scene.simulate(round(DAYS * 24))
```

- `stand` sows a row of plants, 5 cm apart in rows 15 cm apart, 3 cm deep, each turned by 120° from the one before.
  The scene is periodic: the stand repeats beyond its pattern, as in a field.
- `planting` gives each plant its model and its **scenario**, a flat dict of parameter values (`WHEAT_SCENARIO`:
  the leaf shapes, the anatomies, Tuzet's stomata, ...). A key that no component declares raises an error, so a typo
  cannot pass silently.
- The **environment models** are the soil (`RichardsSoilModel`, a loam 50 cm deep) and the light
  (`CaribuLightModel`, `LIGHT_SCENARIO`: the sun's and the sky's light, scattered in the canopy, `mode="nested"`),
  each with its own DataStructure and scenario. The scene's `translator` links them to the plants.
- The **forcings** are the weather, functions of the time (`weather(START)`: a clear May day's PAR, the air's
  daily course of temperature and humidity, 5 mm of rain on the second day). Each model reads their mean over its
  step (the rain's total).
- The **time step** is one hour, from the sowing date `start`. Durations carry their units (`"1 h"`, `"12 h"`,
  `"1 d"`).
- `Compute(threads="auto")` is how the run uses the machine: the processors' threads for the kernels, the solves
  and the sections. The results are the same, bit for bit, at any number of threads.

## One step of the scene

At each step, the Scene writes the forcings, then runs each environment model in the given order (the soil, then
the light), then each plant population (`WheatPlant`'s parts in their order). Before each model runs, the links into
it are exchanged.

So a model reads the values of the models before it at this step, and those of the models after it **at the step
before**. In use case 1, the soil runs first: it receives the plants' uptake of the hour before. Within the plant,
the gas exchange runs before MECHA: its stomata read the leaf water potential of the hour before. This one-step lag
is accepted at an hourly step (decision U-6); `plant_water.csv` and `soil_water.csv` record the water potentials'
change from one hour to the next, to check that it stays small.

The explicit coupling, with its one-step lag, is the default. Where the lag is too large, an option,
`Scene(coupling=FixedPoint(...))`, repeats chosen steps within the step until what they write stops changing. It requires steps that keep their topology: use case 1's growing plants and adaptive soil
grid do not, so it keeps the explicit coupling. `scene.schedule()` gives the step's order as a table: each model,
its steps, and the variables each reads and writes.

## The SceneRecorder: what a run keeps

The `SceneRecorder` writes the run's outputs into one folder, as it runs:

- **summaries** (in `summaries/Plant/` and `summaries/Soil/`): every component's state variables per plant and for the soil, after every step (the extensive ones
  summed, the intensive ones averaged by their declared weight);
- **tables** declared in a few words (in `summaries/hand_declared/`): use case 1's leaves table is the blades' green area, assimilation and
  transpiration summed, their light and stomatal conductance averaged by the green area:

  ```python
  LEAVES = Summary("SubOrgan", where={"label": "LeafElement", "green_area": ">0"},
                   sum={"green_area": None, "An": "green_area", "transpiration": None},
                   mean={"PARa": "green_area", "gs": "green_area", "leaf_water_potential": "green_area"},
                   min=["leaf_water_potential"])
  ```
- **the water balances** of the solves that declare their quantity conserved (MECHA's plant water, the soil's water),
  in `summaries/conservation.csv`: inflow, outflow, change and residual per step;
- **frames**: the whole scene in 3D, with its variables, as encoding3d files, every `frame_period`;
- **checkpoints**: the whole state, every `checkpoint_period`, from which `Scene.load_checkpoint` resumes the run
  bit for bit;
- `recorder.pack()`: the frames in one checked archive, `frames.fspm.zip`, at the end.

## Running use case 1

Create its environment from its folder (the repositories cloned side by side, as in "Installing them" below),
compile the numba kernels once, then run it:

```bash
cd examples/use_case_1
mamba env create -f environment.yaml
conda activate metafspm_use_case_1
python -m openalea.metafspm warm_up
python run.py 3 1 outputs/three_days      # 3 days, 1 plant; by default 14 days, 2 plants, outputs/from_sowing
```

Three days of one plant take about a minute and a half on a laptop, the default run (2 plants, 14 days) about
twenty minutes: a step grows with the plants' sections (use case 1's README gives measured times). The run prints
nothing: everything is in its folder, after every hour:

| file | what it holds |
|---|---|
| `summaries/Plant/WheatPlant/summaries.csv`, `summaries/Soil/RichardsSoilModel/summaries.csv` | every component's variables, per plant (column `plant`: `p0`, `p1`, ...) and for the soil, per hour (column `time`) |
| `summaries/hand_declared/leaves.csv`, `roots.csv` | the blades' gas exchange; the roots' water potential and uptake |
| `summaries/hand_declared/plant_water.csv`, `leaf_water.csv`, `soil_water.csv` | the water exchanged between the plants and the soil, and the potentials' hourly change (the lag's check) |
| `summaries/conservation.csv` | the plant's and the soil's water balances (their residuals are round-off) |
| `run.json` | the run's progress, then the time each model and component took |
| `frames/`, then `frames.fspm.zip` | a frame every 12 hours; at the end, the frames packed into one archive |
| `checkpoints/` | the last day's checkpoint |

## Looking at the results

The tables are CSV files, readable while the run goes on: `pandas.read_csv("outputs/three_days/summaries/hand_declared/leaves.csv")`, or a
spreadsheet.

The frames are **encoding3d** files: each holds the whole scene at one time, every element's variables, the soil and
the 3D shapes. The format is meant to be shared with other plant modelling platforms. For now CPlantBox, GroIMP and
VirtualPlantLab read the frames only through encoding3d's adapters, which are first proposals tested by hand.
In ParaView, encoding3d's reader plugin opens `frames/scene.fspm.json` during the run, or `frames.fspm.zip` after
it, and plays the time steps. `frame_views="default"` prepares the layers: the shoot coloured by its organs' rank,
the roots by their type, the soil by its pressure head. The first plant's sections are drawn once a day
(`display_plants=["p0"]`) as instances of their templates: show the layer "WheatPlant GranapAnatomy instances",
pick a section and apply "encoding3d Select at scale" (scale SubOrgan) to see MECHA's water potential `psi` in each
of its cells.

The plugin is the file `encoding3d/paraview/plugin_encoding3d.py`, loaded once from ParaView's Tools → Manage
Plugins... menu (Load New ..., tick Auto Load); `encoding3d/paraview/README.md` is the how-to. From Python,
`openalea.encoding3d.read_index` and `read` open the same files.

## Starting with one model

Each ported package offers a simpler entry point, `simulate`, which builds a plant model with its component alone
and a scene with one plant, runs it and returns tables of results. It is the way to start with one model; the
packages' developer tutorials show the full use, with several components and plants.

## The models available

The models ported to metafspm are in their own packages, on their development branches (the links below). Each has
**user tutorials** (`example/tutorials/user/`) that start from `simulate`, and **developer tutorials**
(`example/tutorials/developer/`).

| package (branch) | what it models | its components |
|---|---|---|
| [ArchiSimple](https://github.com/openalea/archisimple/tree/develop) (`develop`) | the architecture of a root system: its roots' elongation, branching, thickening and death | `RootArchitecture` (structural) |
| [Adel](https://github.com/openalea/adel/tree/to_metafspm) (`to_metafspm`) | the architecture of wheat shoots: their leaves, stems and tillers, in 3D | `ShootArchitecture` (structural), `ScenarioDevelopment` (the organs' lengths from a scenario); the plant model `ScenarioShoot` couples them |
| [Caribu](https://github.com/openalea/caribu/tree/to_metafspm) (`to_metafspm`) | the light absorbed by each element of a canopy | `CaribuLight` (functional), `CaribuLightModel` (the scene's light, an environment model), `SimplePlant` (a simple plant to light) |
| [GRANAP](https://github.com/openalea-incubator/GRANAP-dev/tree/to_metafspm) (`to_metafspm`) | the anatomy of plant organs' cross sections: their tissues and cells | `GranapAnatomy` (structural: the sections' cells), `RootAxis` and `Needle` (organs to carry the sections) |
| [MECHA](https://github.com/openalea-incubator/MECHA-dev/tree/to_metafspm) (`to_metafspm`) | the flow of water (and solutes) across a root's or a needle's tissues | `MechaHydraulics` (water), `MechaSolutes` (sucrose and starch), `MechaOsmoticPresets` (osmotic potentials); on GRANAP's anatomies |

Ports coming next, not available yet (the component, and the model it is ported from):

- `NitrogenMetabolism`, from openalea.rootcynaps
- `WaterTransport`, from openalea.hydroroot and openalea.hydroshoot
- `WaterTransportMunch`, from openalea.rootcynaps
- `CarbonMetabolism`, from openalea.rhizodep
- `CNMetabolism`, from openalea.rootbridges
- `RootGrowth`, from openalea.rhizodep
- `SimplifiedAnatomy`, from openalea.rhizodep
- openalea.cnwgrass's components (its `GasExchange` is already a metafspm component, on the `to_metafspm`
  branch, used by use case 1; the rest of the repository is not ported yet):
  - `Growth`
  - `Morphogenesis`
  - `Respiration`
  - `Senescence`
  - `Hydraulics`
  - `CNMetabolism`
- `ShootRootJunctionStructure`, from openalea.grassbridges
- `ShootRootJunctionCoordination`, from openalea.grassbridges
- `BrassicaArchitecture`, from openalea.colzette
- `LegumeArchitecture`, from openalea.colzette
- `ChamberLightModel`, from openalea.spice
- `SoilTemperature`, from openalea.soiltemp
- `SoilWaterTransport`, from the common Richards equations
- `SoluteAdvectionDiffusion`
- `MicrobialSoil`, from MIMICS-CN

## Installing them

metafspm and these ports are not published on the `openalea3` channel yet. Each is **cloned on its branch and
installed from source** (`pip install -e`), in an environment created from the package's own
`conda/environment.y(a)ml`, which takes the other dependencies (PlantGL, the MTG, numpy, ...) from `openalea3` and
`conda-forge` with mamba. Clone the repositories side by side, in one folder:

```bash
mkdir fspm && cd fspm
git clone -b release2026 https://github.com/openalea/metafspm.git metafspm
git clone -b other_platforms_bindings https://github.com/openalea/encoding3d.git encoding3d   # openalea.encoding3d (spec 0.4), a dependency of metafspm
```

| package | clone | environment, then install |
|---|---|---|
| ArchiSimple | `git clone -b develop https://github.com/openalea/archisimple.git archisimple` | `mamba env create -f archisimple/conda/environment.yml`, `mamba activate archisimple`, `pip install -e encoding3d`, `pip install -e metafspm` (the environment installs ArchiSimple) |
| Adel | `git clone -b to_metafspm https://github.com/openalea/adel.git adel` | `mamba env create -n adel_dev -f adel/conda/environment.yml`, `mamba activate adel_dev`, `pip install -e encoding3d`, `pip install -e metafspm` |
| Caribu | `git clone -b to_metafspm https://github.com/openalea/caribu.git caribu` (and Adel's, as above, for the wheat examples) | `mamba env create -n caribu -f caribu/conda/environment.yml`, `mamba activate caribu`, `pip install -e encoding3d`, `pip install -e metafspm` (and `pip install -e adel --no-deps`) |
| GRANAP | `git clone -b to_metafspm https://github.com/openalea-incubator/GRANAP-dev.git GRANAP` | `mamba env create -f GRANAP/conda/environment.yaml`, `mamba activate granap`, `pip install -e encoding3d`, `pip install -e metafspm`, `pip install -e GRANAP` |
| MECHA | GRANAP's, as above, and `git clone -b to_metafspm https://github.com/openalea-incubator/MECHA-dev.git MECHA-dev` | `mamba env create -f MECHA-dev/conda/environment.yaml`, `mamba activate mecha`, `pip install -e encoding3d`, `pip install -e metafspm`, `pip install -e GRANAP`, `pip install -e MECHA-dev` |

Keep the folder names above: the environment files find metafspm, encoding3d (and GRANAP) next to the package. Each package's
README gives the same steps. To update, `git pull` in each repository; the editable installs follow.

Then compile the numba kernels once, metafspm's and those of every port installed (Adel's, Caribu's): without it,
the first run spends about a minute and a half compiling them. Run it again after installing or updating a port; for
a read-only installation, set `NUMBA_CACHE_DIR` to a writable folder, the same when running:

```bash
python -m openalea.metafspm warm_up
```

Adel and Caribu are also on `openalea3` (`mamba install -c openalea3 -c conda-forge openalea.adel openalea.caribu`),
but these are their released versions, the original models **without** the metafspm components above: for the
components and the tutorials, install from the branches.


## Where to go next

- **Use case 1's README** (`examples/use_case_1/README.md`) details each of its models, links and outputs, with
  measured times and memory.
- **From a scenario to its outputs** ({doc}`tutorial`) goes through a scenario table, a recorded run, its outputs
  read back and parameter variants, on a seedling in a soil column.
- The **framework guide** ({doc}`user`) describes each part in full: DataStructures, components, graph systems,
  structure, coupling, scenes, upscaling and calibration, performance, and how to port a model.
- Each package's **user tutorials** (`example/tutorials/user/`) start from `simulate`, for beginners; its
  **developer tutorials** (`example/tutorials/developer/`) open with the formalism, then read its components
  through it.
- metafspm's workshop tutorial `examples/soil_plant_atm_fspm2026_workshop/tutorial_fspm2026_general_usage.ipynb`
  builds components step by step and couples them with a soil.
