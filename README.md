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

A full example is available in [Usage example](https://github.com/openalea/metafspm/examples/soil_plant_atmosphere/tutorial_fspm2026_general_usage.ipynb) files for a real functional example. For advanced modellers, you can look at the [Solver implementation example](https://github.com/openalea/metafspm/examples/soil_plant_atmosphere/tutorial_fspm2026_new_solver.ipynb).

Bellow is a short presentation of basic usage of Components.

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

## Code Structure

Here is a dynamic view of OpenAlea.MetaFSPM's code structure : https://openalea.github.io/metafspm/

## Contributing

We are open to contributions on the develop branch of this package.

## Authors and acknowledgment

Gerault T., Rees F., Barillot R., Pradal C.

## License
This project is licensed under the CeCILL-C License - see file [LICENSE](LICENSE) for details
