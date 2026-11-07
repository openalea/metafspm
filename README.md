# MetaFSPM
[![Docs](https://readthedocs.org/projects/mtg/badge/?version=latest)](https://metafspm.readthedocs.io/)
(Not yet)
[![Build Status](https://github.com/openalea/mtg/actions/workflows/conda-package-build.yml/badge.svg?branch=master)](https://github.com/openalea/mtg/actions/workflows/conda-package-build.yml?query=branch%3Amaster)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License--CeCILL-C-blue)](https://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html)
(Not yet)
[![Anaconda-Server Badge](https://anaconda.org/openalea3/mtg/badges/version.svg)](https://anaconda.org/openalea3/metafspm)

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
| [CNW-Grass](https://github.com/openalea/cnwgrass/tree/to_metafspm) (`to_metafspm`) | leaves' gas exchange (photosynthesis, stomata, transpiration) | `GasExchange` (plant model `ExchangingWheat`, on Adel's shoots lit by Caribu) |

None of them is on the `openalea3` channel in these versions yet (Adel's and Caribu's packages there are the original
models, without these components): clone each one on its branch next to metafspm and install both with
`pip install -e`, in the environment of the package's `conda/environment.y(a)ml` (mamba, from `openalea3` and
`conda-forge`). The steps for each package are in [metafspm in five minutes](docs/five_minutes.md#installing-them),
which also explains, in plain words, what a component, a plant model and a scene are.

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
from openalea.metafspm import FunctionalComponent
from openalea.metafspm.coupling.component import input_variable, parameter, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.solve.decorator import rate, state

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
        return hexose - self.dt * exudation / struct_mass

ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)   # g: an MPG holding one or several plants
carbon = RootCarbon(data_structure=ds, time_step=3600.)
carbon()                                                 # one time step, steps in the scheduled order
```

Coupled equations (transport, diffusion) are declared as `@graph_system`s and solved on the graph; components are
coupled within a DataStructure by a `CompositeModel` and its translator, and across DataStructures (plants and soil
grids, light, environment) by mappings; a `Scene` builds populations from a planting table and runs them with their
environment. See the documentation in `docs/`: the user guide (`docs/user.md`), the declaration conventions
(`docs/conventions.md`) and the guide for porting an existing model (`docs/migration.md`).

A simulation follows one path: a scenario table (`read_scenarios`), `planting`, a `Scene` with its `forcings=` and
a `SceneRecorder`, `simulate`, then the outputs read back (`run.json`, the `Plant/` and `Soil/` summaries, the
frames), and `variants` of a scenario within the declared bounds. [From a scenario to its outputs](docs/tutorial.md)
goes through it on a seedling in a soil column (`examples/soil_plant_atmosphere/scenario_to_outputs.py`).

How a run uses the machine is its `Compute` (`Scene(..., compute=Compute(threads="auto", workers="auto",
device="cpu", memory="trim", linear_solver=None))`, the default): the threads (`"auto"`: the processors, each
parallel part taking those that pay, the results the same at any number), the process pool of a reduction's kinds, an
optional GPU, the memory given back after each step and the Newton steps' sparse solver. `scene.timing` gives the
time per model and per component. See the user guide's *Performance notes* and [Running on a GPU](docs/gpu.md),
which also covers repeated runs (`run_variants`).

For a model built on MetaFSPM, see [Root_CyNAPS](https://github.com/GeraultTr/Root_CyNAPS).

## Exchange with other platforms

`openalea.metafspm.io.encoding3d` writes scenes and DataStructures as [encoding3d](https://github.com/openalea/encoding3d)
files: one glTF file per time step, which CPlantBox, GroIMP or VirtualPlantLab can read, and ParaView displays with
encoding3d's reader plugin. A file holds:

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
the templates and the kinds' recovery operators are shared parts written once, and the reader rebuilds the values
of the instances it selects. `SceneRecorder(pack_at_finish=True)` packs the frames into one `frames.fspm.zip` at
the end (`recorder.pack()`; `encoding3d unpack ARCHIVE.fspm.zip FOLDER` restores the folder; encoding3d and its
ParaView reader also read the archive as it is).

encoding3d's Python package, `openalea.encoding3d`, is a dependency of metafspm. Until it is published, clone
[encoding3d](https://github.com/openalea/encoding3d) next to metafspm; `conda/environment.yml` installs it from there
(`pip install -e ../encoding3d`).

## Code Structure

Here is a dynamic view of OpenAlea.MetaFSPM's code structure : https://openalea.github.io/metafspm/

## Contributing

We are open to contributions on the develop branch of this package.

## Authors and acknowledgment

Gerault T., Rees F., Barillot R., Pradal C.

## License
This project is licensed under the CeCILL-C License - see file [LICENSE](LICENSE) for details
