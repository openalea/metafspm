# From a scenario to its outputs

The one path through a metafspm simulation, five steps:

1. write the scenarios in a table;
2. plant;
3. build the scene with its forcings and its recorder, and run it;
4. read the outputs back;
5. vary a parameter within its bounds.

The script `examples/soil_plant_atmosphere/scenario_to_outputs.py` does all of it on a seedling in a column of soil
under a day-and-night air, and runs in a few seconds:

```bash
cd examples/soil_plant_atmosphere
python scenario_to_outputs.py outputs/scenarios
```

Each model's tutorials follow the same path through its own `simulate()` (`simulate(..., outputs="outputs")`).

## 1. The scenarios, in a table

`scenarios/scenarios.csv` has one row per entry and one column per scenario. Here the scenarios are `humid` and
`dry`:

```text
name,model,type,unit,description,humid,dry
root_radial_k,SeedlingWater,parameter,mm3 s-1 MPa-1 m-1,Root radial conductance per length,0.5,0.5
stomatal_conductance,SeedlingWater,parameter,mol m-2 s-1,Stomatal conductance to water vapour,0.2,0.1
topsoil_K,Soil,parameter,mm3 m-1 s-1 MPa-1,Hydraulic conductivity of the topsoil,1.5,0.6
air_water_potential,,forcing,MPa,The air's water potential over the day (a table of time in s),air_humid.csv,air_dry.csv
```

- **The columns.** `model`, `type`, `unit` and `description` describe the entries and are not scenarios. An empty
  cell leaves the model's default.
- **The `model` column** sends each entry to its model: the plants (`SeedlingWater`) or the soil (`Soil`).
- **A `forcing` row** is a time-varying input. Its cell names a CSV file with the columns `time` (s) and the
  variable, here the air's water potential hour by hour over two days. It needs no model: forcings are the scene's.

```python
from openalea.metafspm.scene.scenarios import read_scenarios

scenarios = read_scenarios("scenarios/scenarios.csv")
scenarios["dry"]
# {"SeedlingWater": {"root_radial_k": 0.5, "stomatal_conductance": 0.1}, "Soil": {"topsoil_K": 0.6},
#  "forcings": {"air_water_potential": <Series indexed by time>}}
```

A scenario is a flat dict, `{name: value}`, one per model. Its entries are the parameters and settings of the
model's components, or the model's own arguments (`topsoil_K` is an argument of `Soil`). A name nothing knows
raises an error, so a typo doesn't go unnoticed.

## 2. Planting

`planting` builds the planting table, one row per plant. *plants* are `Planting(x, y, z, rotation)`, given (here
one, at the centre of a 0.2 m × 0.2 m column) or sown (`stand(xrange, yrange, sowing_density, row_spacing)`). They
are named `p0`, `p1`, …

```python
from openalea.metafspm.scene.population import Planting, planting

table = planting([Planting(x=0.1, y=0.1)], SeedlingWater, scenario["SeedlingWater"], xrange=0.2, yrange=0.2)
```

`plant_scenarios=[...]` gives each plant its own values on top of the shared scenario. Numeric parameters may
differ per plant; the model's other entries are one for all.

## 3. The scene: forcings, recorder, run

```python
from openalea.metafspm.scene.recorder import SceneRecorder
from openalea.metafspm.scene.scene import Scene

forcings = scenario.pop("forcings")
recorder = SceneRecorder("outputs/dry", frame_period=6, checkpoint_period=12, keep_checkpoints=1)
scene = Scene(table, environment=[Soil], environment_scenarios=[scenario["Soil"]], translator=SCENE_TRANSLATOR,
              time_step=3600, mappings=root_surface_mappings, forcings=forcings, recorder=recorder)
scene.simulate(48)
```

- **Forcings** are the one way in for time-varying inputs. A forcing is a number, a function of the time (s), a
  pandas Series indexed by time, or a `(times, values)` pair, read by linear interpolation. Here
  `air_water_potential` is a parameter of both the plant's and the soil's water transport: at the start of each
  step, the Scene writes its value into both. `as_forcing(value, time_unit=86400.)` converts an input given by day.
  `events=[(time, action)]` stay for one-off actions, such as a cut or an irrigation.
- **The recorder** decides what is written, every output switched on or off on its own:
  - summaries every `summary_period` steps (on by default);
  - the `segment_plants`' segments every `segment_period` steps;
  - frames every `frame_period` steps;
  - checkpoints every `checkpoint_period` steps;
  - tables of your own, `tables={name: function(scene)}`, each function returning a row.

  Without a folder, it keeps its tables in memory.
- **`simulate(n)`** runs *n* steps, or fewer if `stop_when=` holds. It then records the last state and the run's
  status: finished, stopped, or failed with its error. A loop of `scene.run()` ends with `scene.finish()`.
- **In one call.** `Scene.from_scenario(scenarios["dry"], [Planting(x=0.1, y=0.1)], SeedlingWater, environment=[Soil],
  xrange=0.2, yrange=0.2, translator=SCENE_TRANSLATOR, time_step=3600, mappings=root_surface_mappings,
  recorder=recorder)` does steps 2 and 3 from the table's column: each model's entries go to it, the forcings to
  the scene.

The folder then holds:

```text
outputs/dry/
  run.json                           status, start and end, wall time per model, versions
  scenario.json                      the planting, each model's scenario, the forcings, the recorder's settings
  forcings/air_water_potential.csv   the forcing, as given
  Plant/SeedlingWater/summaries.csv  one row per plant and step: the plant's state variables, summed or averaged
  Soil/Soil/summaries.csv            one row per step: the soil grid's state variables
  frames/frame_000000.glb ...        the whole scene every 6 hours (fspm-TF), and their index scene.fspm.json
  checkpoints/step_000048/           the last checkpoint, to restart from
```

Plant summaries sum the extensive variables over each plant's elements (the transpiration, `evaporation`) and
average the others (a water potential). An environment model's summaries go to `Soil/` or `Atmosphere/`, given by
its `compartment` attribute. Every table has its time column, `time`, in s.

## 4. Reading the outputs back

```python
import json
import pandas as pd
from openalea import fspm_tf

status = json.load(open("outputs/dry/run.json"))           # status["status"] == "finished", status["timing"]
plant = pd.read_csv("outputs/dry/Plant/SeedlingWater/summaries.csv")
soil = pd.read_csv("outputs/dry/Soil/Soil/summaries.csv")
frames = fspm_tf.read_index("outputs/dry/frames/scene.fspm.json")["frames"]
last = fspm_tf.read(frames[-1]["path"])                     # every variable of the plant and the soil, at 48 h
```

The script prints, for each scenario:

```text
dry: finished after 48 steps in 1.5 s (the seedling 0.3 s, the soil 1.0 s)
  transpiration (mm3 s-1) on day 2, at 0, 6, 12, 18 h: 0.0234, 0.0175, 0.0443, 0.0545
  9 frames, the last at 48 h: SeedlingWater, Soil
```

The dry air pulls more water through the seedling than the humid one, despite its stomata half closed.

- **ParaView.** fspm-TF's reader plugin opens `frames/scene.fspm.json` and plays the frames as time steps.
- **Restart.** `Scene.restore("outputs/dry/checkpoints/step_000048", **the same arguments)` continues the run.
  The recorder then cuts its tables back to the checkpoint's time, so a run that went past it leaves no duplicate
  rows.
- **Another platform.** `openalea.metafspm.io.fspm_tf.import_frame(path)` rebuilds the DataStructures from a frame,
  bit for bit.

## 5. Variants within the declared bounds

`parameter_table(*components)` lists a model's parameters: value, unit, description, and the bounds the component
declares (`min_value`, `max_value`). `variants` builds scenarios around a base one, within those bounds:

```python
from openalea.metafspm.scene.scenarios import parameter_table, variants

parameter_table(SeedlingStructure, main=("root_radial_k", "stomatal_conductance"))
variants(scenarios["humid"]["SeedlingWater"], {"stomatal_conductance": None}, design="grid", levels=3,
         declared=[SeedlingStructure])
# {"stomatal_conductance=0.02": {...}, "stomatal_conductance=0.31": {...}, "stomatal_conductance=0.6": {...}}
```

- **`ranges`** take a list of values, a `(low, high)` range, or `None` for the declared bounds.
- **`design`** is one of:
  - `"grid"`: every combination;
  - `"one_at_a_time"`: the base, then each value of each entry alone;
  - `"sampling"`: a Latin hypercube of `samples` scenarios (`seed=` gives the same draws again).

A value outside its declared bounds raises an error. Each variant is run like any scenario. Computing sensitivity
indices from the outputs (e.g. with SALib) is left to you.
