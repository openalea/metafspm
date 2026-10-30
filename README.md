# MetaFSPM
[![Docs](https://readthedocs.org/projects/metafspm/badge/?version=stable)](https://metafspm.readthedocs.io/)
[![Build Status](https://github.com/openalea/metafspm/actions/workflows/conda-build.yml/badge.svg?branch=main)](https://github.com/openalea/metafspm/actions/workflows/conda-build.yml?query=branch%3Amain)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License--CeCILL-C-blue)](https://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html)
(Not yet)
[![Anaconda-Server Badge](https://anaconda.org/openalea3/metafspm/badges/version.svg)](https://anaconda.org/openalea3/metafspm)

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

DOC = dict(unit="mol.g-1", unit_comment="", description="", min_value=0., max_value=1., value_comment="",
           references="", DOI=[])

@dataclass
class RootCarbon(FunctionalComponent):
    hexose: float = state_variable(**DOC, initialize=1e-3, scale=scales.SubOrgan,
                                   state_variable_type="massic_concentration", on_grow="inherit")
    exudation: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="extensive")
    struct_mass: float = input_variable(**DOC, by="RootGrowth", initialize=0., scale=scales.SubOrgan)
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

For a model built on MetaFSPM, see [Root_CyNAPS](https://github.com/GeraultTr/Root_CyNAPS).

## Exchange with other platforms

`openalea.metafspm.io.fspm_tf` writes scenes and DataStructures as [fspm-TF](https://github.com/openalea/fspm-TF)
files: one glTF file per time step, which CPlantBox, GroIMP or VirtualPlantLab can read, and ParaView displays with
fspm-TF's reader plugin. A file holds:

- each MPG scale as a table, with every variable at its declared unit, kind and standard name;
- the soil grids, the mappings between plants and soil;
- the geometry.

`import_frame` rebuilds the DataStructures from a file, bit for bit, and a simulation continues from them as from
the original.

```python
from openalea.metafspm.io.fspm_tf import export_scene, import_frame

export_scene("scene.glb", scene)                         # one frame now
data_structures = import_frame("scene.glb")              # {id: DataStructure}
scene = Scene(..., output_dirpath="outputs", frame_period=24)   # a frame every 24 steps, in outputs/frames
```

fspm-TF's Python package, `openalea.fspm_tf`, is a dependency of metafspm. Until it is published, clone
[fspm-TF](https://github.com/openalea/fspm-TF) next to metafspm; `conda/environment.yml` installs it from there
(`pip install -e ../fspm-TF`).

## Code Structure

Here is a dynamic view of OpenAlea.MetaFSPM's code structure : https://openalea.github.io/metafspm/

## Contributing

We are open to contributions on the develop branch of this package.

## Authors and acknowledgment

Gerault T., Rees F., Barillot R., Pradal C.

## License
This project is licensed under the CeCILL-C License - see file [LICENSE](LICENSE) for details
