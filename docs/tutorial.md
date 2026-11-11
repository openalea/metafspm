# From a scenario to its outputs

The one path through a metafspm simulation, five steps:

1. write the scenarios in a table;
2. plant;
3. build the scene with its forcings and its recorder, and run it;
4. read the outputs back;
5. vary a parameter within its bounds.

The script `examples/soil_plant_atm_fspm2026_workshop/scenario_to_outputs.py` does all of it on a seedling in a
column of soil under a day-and-night air, and runs in a few seconds:

```bash
cd examples/soil_plant_atm_fspm2026_workshop
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
from openalea.metafspm import read_scenarios

scenarios = read_scenarios("scenarios/scenarios.csv")
scenarios["dry"]
# {"SeedlingWater": {"root_radial_k": 0.5, "stomatal_conductance": 0.1}, "Soil": {"topsoil_K": 0.6},
#  "forcings": {"air_water_potential": <Series indexed by time>}}
```

A scenario is a flat dict, `{name: value}`, one per model. Its entries are the parameters and settings of the
model's components, or the model's own arguments (`topsoil_K` is an argument of `Soil`). A name nothing knows
raises an error, so a typo doesn't go unnoticed.

## 2. Planting

`planting` builds the planting table, one row per plant. *plants* are `Planting(x, y, depth, rotation)`, given (here
one, at the centre of a 0.2 m × 0.2 m column) or sown (`stand(plants, rows=, row_spacing=, plant_spacing=)`). They
are named `p0`, `p1`, …

```python
from openalea.metafspm import Planting, planting

table = planting([Planting(x=0.1, y=0.1)], SeedlingWater, scenario["SeedlingWater"], x_range=0.2, y_range=0.2)
```

`plant_scenarios=[...]` gives each plant its own values on top of the shared scenario. Numeric parameters may
differ per plant; the model's other entries are one for all.

## 3. The scene: forcings, recorder, run

```python
from openalea.metafspm import Scene, SceneRecorder

forcings = scenario.pop("forcings")
recorder = SceneRecorder("outputs/dry", frame_period="6 h", checkpoint_period="12 h", keep_checkpoints=1)
scene = Scene(table, environment=[Soil], environment_scenarios=[scenario["Soil"]], translator=SCENE_TRANSLATOR,
              time_step="1 h", forcings=forcings, recorder=recorder)
scene.simulate(48)
```

- **Forcings** are the one way in for time-varying inputs. A forcing is a number, a function of the time (s), a
  pandas Series indexed by time, or a `(times, values)` pair, read by linear interpolation. Here
  `air_water_potential` is a parameter of both the plant's and the soil's water transport: at the start of each
  step, the Scene writes its value into both. `as_forcing(value, time_unit="d")` converts an input given by day.
  `events=[("36 h", action)]` stay for one-off actions, such as a cut or an irrigation.
- **Times carry their units**: `"1 h"`, `"12 h"`, `"1 d 6 h"` (or a number of seconds); a date, `"2026-05-10
  00:00"`, for the Scene's `start=` (the sun's calendar).
- **The recorder** decides what is written, every output switched on or off on its own:
  - summaries every `summary_period` (every step by default, `"step"`);
  - the `segment_plants`' segments every `segment_period` (`"1 d"` by default);
  - frames every `frame_period` (`"6 h"` here);
  - checkpoints every `checkpoint_period`;
  - tables of your own, `tables={name: function(scene)}`, each function returning a row.

  Without a folder, it keeps its tables in memory.
- **`simulate(n)`** runs *n* steps, or fewer if `stop_when=` holds. It then records the last state and the run's
  status: finished, stopped, or failed with its error. A loop of `scene.run()` ends with `scene.finish()`.
- **In one call.** `Scene.from_scenario(scenarios["dry"], [Planting(x=0.1, y=0.1)], SeedlingWater, environment=[Soil],
  x_range=0.2, y_range=0.2, translator=SCENE_TRANSLATOR, time_step=3600, recorder=recorder)` does steps 2 and 3 from the table's column: each model's entries go to it, the forcings to
  the scene.

The folder then holds:

```text
outputs/dry/
  run.json                           status, start and end, wall time per model, provenance
  scenario.json                      the planting, each model's scenario, the forcings, the recorder's settings
  forcings/air_water_potential.csv   the forcing, as given
  Plant/SeedlingWater/summaries.csv  one row per plant and step: the plant's state variables, summed or averaged
  Soil/Soil/summaries.csv            one row per step: the soil grid's state variables
  frames/frame_000000.glb ...        the whole scene every 6 hours (encoding3d), and their index scene.fspm.json
  checkpoints/step_000048/           the last checkpoint, to restart from
```

Plant summaries sum the extensive variables over each plant's elements (the transpiration, `evaporation`) and
average the others (a water potential). An environment model's summaries go to `Soil/` or `Atmosphere/`, given by
its `compartment` attribute. Every table has its time column, `time`, in s.

**Several plants, several cores.** A scene of many plants can step its populations in worker processes, the scene's
own process keeping the soil, the light and the outputs: give each group of plants its population and ask the
processes. The outputs are the same, to the last digit, at any number of processes.

```python
from openalea.metafspm import Compute

table = planting([Planting(x=0.05 + 0.1 * k, y=0.1) for k in range(4)], SeedlingWater, scenarios["dry"]["SeedlingWater"],
                 population=["a", "b", "a", "b"])                 # two populations of two plants
scene = Scene(table, environment=[Soil], translator=SCENE_TRANSLATOR, time_step=3600, recorder="outputs/dry_4",
              compute=Compute(processes=2))                         # one worker per population; "auto": as many as fit
scene.simulate(48)
```

`run.json` then says where each population ran (`"processes"`: its groups, cores and memory). It pays when the plants
are large (days of growth, an anatomy each); on young seedlings like these the exchanges between the processes cost
more than they save ([user guide](user.md#populations-in-processes-computeprocesses)).

## 4. Reading the outputs back

```python
import json
import pandas as pd
from openalea import encoding3d

status = json.load(open("outputs/dry/run.json"))           # status["status"] == "finished", status["timing"]:
                                                           # per model and per component ("SeedlingWater.PlantWaterTransport")
plant = pd.read_csv("outputs/dry/Plant/SeedlingWater/summaries.csv")
soil = pd.read_csv("outputs/dry/Soil/Soil/summaries.csv")
frames = encoding3d.read_index("outputs/dry/frames/scene.fspm.json")["frames"]
last = encoding3d.read(frames[-1]["path"])                     # every variable of the plant and the soil, at 48 h
```

The script prints, for each scenario:

```text
dry: finished after 48 steps in 1.5 s (the seedling 0.3 s, the soil 1.0 s)
  transpiration (mm3 s-1) on day 2, at 0, 6, 12, 18 h: 0.0234, 0.0175, 0.0443, 0.0545
  9 frames, the last at 48 h: SeedlingWater, Soil
```

The dry air pulls more water through the seedling than the humid one, despite its stomata half closed.

- **ParaView.** encoding3d's reader plugin opens `frames/scene.fspm.json` and plays the frames as time steps.
- **Shapes drawn for display** (e.g. GRANAP's root and leaf cross-sections). They are a recording choice:
  `SceneRecorder(..., display_period="1 d")` adds them to a frame every day, `display_plants=("p0",)` for some
  plants only. When the plant model *reduces* its sections (MECHA's water flow keeps a few nodes per section and
  computes the others from them), the frames store each section as one copy of its shape plus a few numbers, not
  every cell's value: a frame stays small (under 1 MB where a fully drawn one takes 30 MB or more), and the shapes
  and the formulas shared by many sections go once into `frames/parts/`. In ParaView, the layer
  "<plant model> <component> instances" shows these sections; pick one and run "encoding3d Select at scale" to
  colour its cells by their water potential, or tick the reader's "Evaluate Derived Fields" to colour every section
  (slower on large stands). `SceneRecorder(display="full")` writes every cell's value in every frame instead.
- **One file to share.** `SceneRecorder(..., pack_at_finish=True)` packs `frames/` into `frames.fspm.zip` when the
  run finishes (about a third of its size), checks the archive, then removes the folder (`pack_keep_folder=True`
  keeps it); ParaView and encoding3d open the archive as they open the folder.
- **Restart.** `Scene.restore("outputs/dry/checkpoints/step_000048", **the same arguments)` continues the run.
  The recorder then cuts its tables back to the checkpoint's time, so a run that went past it leaves no duplicate
  rows. To go on with a changed parameter, build the scene with the changed scenario and resume with
  `scenario="current"`: `Scene.restore(path, scenario="current", **arguments)` applies the scene's scenarios after
  the load (by default, `"checkpoint"`, the run keeps the saved values).
- **Another platform.** `openalea.metafspm.io.encoding3d.import_frame(path)` rebuilds the DataStructures from a frame,
  bit for bit.

## 5. Variants within the declared bounds

`parameter_table(*components)` lists a model's parameters: value, unit, description, and the bounds the component
declares (`min_value`, `max_value`). `variants` builds scenarios around a base one, within those bounds:

```python
from openalea.metafspm import parameter_table, variants

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
