# MetaFSPM
[![Anaconda-Server Badge](https://anaconda.org/openalea3/openalea.metafspm/badges/license.svg)](https://anaconda.org/openalea3/openalea.metafspm)

[![Build Status](https://github.com/openalea/metafspm/actions/workflows/conda-build.yml/badge.svg?branch=main)](https://github.com/openalea/metafspm/actions/workflows/conda-build.yml?query=branch%3Amain)
[![Docs](https://readthedocs.org/projects/metafspm/badge/?version=latest)](https://metafspm.readthedocs.io/)

[![Anaconda-Server Badge](https://anaconda.org/openalea3/openalea.metafspm/badges/version.svg)](https://anaconda.org/openalea3/openalea.metafspm)
[![Anaconda-Server Badge](https://anaconda.org/openalea3/openalea.metafspm/badges/platforms.svg)](https://anaconda.org/openalea3/openalea.metafspm)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![Downloads](https://anaconda.org/openalea3/openalea.metafspm/badges/downloads.svg)](https://anaconda.org/openalea3/openalea.metafspm)

[![DOI](https://zenodo.org/badge/994653515.svg)](https://zenodo.org/badge/latestdoi/994653515)

## Purpose 

This package is intended to provide minimal tools to assist Functionnal Structural Plant Modellers (FSPM) so that they can make their production more readable regarding scientific content. It is also intended as a minimal constraint in object oriented programming to ease the model coupling in OpenAlea around a Multiscale Tree Graph (MTG). However the idea is to remain very generic in order to accept a wide variety of model types. You can't impose formalism on others, that's how great civilizations die. Our aim here is rather to favor research projects among different scientific communities.

## Just want to run a model?

metafspm is the framework; the models built on it are in their own packages, on their development branches, each
with step-by-step user tutorials (`example/tutorials/user/`) for newcomers and developer tutorials
(`example/tutorials/developer/`):

| package (branch) | what it models | its components |
|---|---|---|
| [ArchiSimple](https://github.com/openalea/archisimple/tree/develop) (`develop`) | root system architecture | `RootArchitecture` |
| [Adel](https://github.com/openalea/adel/tree/to_metafspm) (`to_metafspm`) | wheat shoots in 3D | `ShootArchitecture`, `ScenarioDevelopment` (plant model `ScenarioShoot`) |
| [Caribu](https://github.com/openalea/caribu/tree/to_metafspm) (`to_metafspm`) | light in canopies | `CaribuLight`, `CaribuLightModel` (environment model), `SimplePlant` |
| [GRANAP](https://github.com/openalea-incubator/GRANAP-dev/tree/to_metafspm) (`to_metafspm`) | cross-section anatomies | `GranapAnatomy`, `RootAxis`, `Needle` |
| [MECHA](https://github.com/openalea-incubator/MECHA-dev/tree/to_metafspm) (`to_metafspm`) | water and solutes across root and needle tissues | `MechaHydraulics`, `MechaSolutes`, `MechaOsmoticPresets` |

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

None of them is on the `openalea3` channel in these versions yet (Adel's and Caribu's packages there are the original
models, without these components): clone each one on its branch next to metafspm and install both with
`pip install -e`, in the environment of the package's `conda/environment.y(a)ml` (mamba, from `openalea3` and
`conda-forge`). The steps for each package are in [metafspm in five minutes](docs/five_minutes.md#installing-them),
which also explains metafspm's core ideas in plain words through use case 1: components, plant models, links, the
scene and its step, the recorder.

The rest of this page is for those who build or couple models.

## Installation

Prerequisites to installation :
- miniforge (https://conda-forge.org/download/) 
- git (https://git-scm.com/downloads)

### User installation
In your folder current model, just use
```
mamba install -c conda-forge -c openalea3 openalea.metafspm
```
or simply add openalea.metafspm to your environment requirements.


### Developer installation
From terminal, clone this package and then , then run the command :
```
git clone https://github.com/openalea/metafspm.git
```

- Create an environment dedicated to your model with the necessary requirements and install MetaFSPM in it
```
mamba create -f ./conda/environment.yaml
mamba activate metafspm
pip install -e .
```

- Compile the numba kernels once (metafspm's and every installed port's; a first run otherwise spends about 1.5 min
  compiling them). Run it again after installing a port:
```
python -m openalea.metafspm warm_up
```
  For a read-only installation, set `NUMBA_CACHE_DIR` to a writable folder (numba's cache), the same when running.

## Example use

A component is a dataclass deriving from `FunctionalComponent`, built on a DataStructure (here a plant population
in one MPG). Its fields declare its variables, and its steps are methods named after the variable they write:

```python
from dataclasses import dataclass
from openalea.metafspm import (FunctionalComponent, input_variable, MPGDataStructure, parameter, rate, state,
                               state_variable)
from openalea.metafspm.data_structure.configs import ScalesConfig as scales

DOC = dict(unit="mol g-1", description="", min_value=0., max_value=1.)

@dataclass
class RootCarbon(FunctionalComponent):
    hexose: float = state_variable(**DOC, default=1e-3, scale=scales.SubOrgan,
                                   kind="massic_concentration", on_grow="inherit")
    exudation: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")
    struct_mass: float = input_variable(**DOC, by="RootGrowth", default=0., scale=scales.SubOrgan)
    exudation_rate: float = parameter(**DOC, by="RootCarbon", default=1e-6)

    @rate
    def _exudation(self, hexose, struct_mass, exudation_rate):
        return exudation_rate * hexose * struct_mass

    @state
    def _hexose(self, hexose, exudation, struct_mass):
        return hexose - self.step_duration * exudation / struct_mass

data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)   # mtg: an MPG holding one or several plants
carbon = RootCarbon(data_structure=data_structure, time_step=3600.)
carbon()                                                 # one time step, steps in the scheduled order
```

Coupled equations (transport, diffusion) are declared as `@graph_system`s and solved on the graph; components are
coupled within a DataStructure by a `CompositeModel` and its translator, and across DataStructures (plants and soil
grids, light, environment) by mappings; a `Scene` builds populations from a planting table and runs them with their
environment. See the documentation in `docs/`: the user guide (`docs/user.md`), the declaration conventions
(`docs/conventions.md`) and the guide for porting an existing model (`docs/migration.md`).

A simulation follows one path: a scenario table (`read_scenarios`), `planting`, a `Scene` with its `forcings=` and
a `SceneRecorder`, `simulate`, then the outputs read back (`run.json`, the `summaries/Plant/` and `summaries/Soil/` tables, the
frames), and `variants` of a scenario within the declared bounds. [From a scenario to its outputs](docs/tutorial.md)
goes through it on a seedling in a soil column
(`examples/soil_plant_atm_fspm2026_workshop/scenario_to_outputs.py`).

How a run uses the machine is its `Compute` (`Scene(..., compute=Compute(threads="auto", device="cpu",
memory="trim", linear_solver=None))`, the default): the threads (`"auto"`: the processors, each parallel part taking
those that pay, a reduction's archetypes included, the results the same at any number), an optional GPU (experimental:
it should work on suitable hardware but has not been tested on a GPU yet), the memory (glibc's allocator held to two arenas, the memory freed given
back after each step) and the graph systems' sparse solver (`linear_solver="superlu_reuse"`, say, for every system;
a mapping such as `{"default": "superlu", "symmetric": "cholmod"}` gives each system its own, by its qualified name
`"Component._system"`, else a property it declares, else `"default"`). `scene.timing` gives the time per model and per component.
See the user guide's *Performance notes* and [Running on a GPU](docs/gpu.md), which also covers repeated runs
(`run_variants`).

For a model built on MetaFSPM, see [Root_CyNAPS](https://github.com/GeraultTr/Root_CyNAPS).

## Exchange with other platforms

`openalea.metafspm.io.encoding3d` writes scenes and DataStructures as [encoding3d](https://github.com/openalea/encoding3d)
files: one glTF file per time step, which ParaView displays with encoding3d's reader plugin. CPlantBox, GroIMP and
VirtualPlantLab read them only through encoding3d's adapters, first proposals tested by hand. A file holds:

- each MPG scale as a table, with every variable at its declared unit, kind and standard name;
- the soil grids, the mappings between plants and soil;
- the geometry.

`import_frame` rebuilds the DataStructures from a file, bit for bit, and a simulation continues from them as from
the original.

```python
from openalea.metafspm.io.encoding3d import export_scene, import_frame

export_scene("scene.glb", scene)                         # one frame now
data_structures = import_frame("scene.glb")              # {id: DataStructure}
scene = Scene(..., recorder=SceneRecorder("outputs", frame_period="1 d"))   # a frame every day, in outputs/frames
```

A reduction's groups (e.g. the cells of reduced cross sections) are written as instances of their templates
(`SceneRecorder(display="instances")`, the default; `"full"` draws every recovered interior, `None` leaves them out):
the templates and the archetypes' recovery operators are shared parts written once, and the reader rebuilds the values
of the instances it selects. `SceneRecorder(pack_at_finish=True)` packs the frames into one `frames.fspm.zip` at
the end (`recorder.pack()`), checks it (every file a member, read back unchanged, the archive valid), then removes
the folder (`pack_keep_folder=True` keeps it; `encoding3d unpack ARCHIVE.fspm.zip FOLDER` restores it; encoding3d
and its ParaView reader also read the archive as it is, and a run resumed from a checkpoint unpacks it).
Each frame's file is written by a writer thread while the next steps run (`SceneRecorder(frame_writing=
"background")`, the default; `"synchronous"` in the step's thread; the same files): a checkpoint, `pack()` and the
run's end wait for it, `recorder.wait_for_frames()` too.

encoding3d's Python package, `openalea.encoding3d`, is a dependency of metafspm (`pyproject.toml` and
`conda/environment.yml`, from `openalea3`). Until it is published there, clone
[encoding3d](https://github.com/openalea/encoding3d) next to metafspm and install it from the clone
(`pip install -e ../encoding3d`), leaving out the environment's `openalea.encoding3d` line.

## Code Structure

Here is a dynamic view of OpenAlea.MetaFSPM's code structure : https://openalea.github.io/metafspm/

## Contributing

We are open to contributions on the develop branch of this package.

## Authors and acknowledgment

Gerault T., Rees F., Barillot R., Pradal C.

## License
This project is licensed under the CeCILL-C License - see file [LICENSE](LICENSE) for details
