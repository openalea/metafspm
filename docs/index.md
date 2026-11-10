# MetaFSPM

MetaFSPM gives Functional-Structural Plant Models (FSPM) a common way to declare, schedule, solve and couple their
processes, around the Multiscale Tree Graph (MTG) of OpenAlea.

- **Components** declare their variables (unit, scale, kind) and their processes as steps or as graph systems
  (coupled equations on the plant's graph, solved by Newton or time integrators).
- **DataStructures** hold the variables: a plant population on one MPG (an MTG with a solver graph), regular and
  adaptive soil grids, and unions of populations. Steps receive their arrays, so the same model runs on one plant or
  a thousand.
- **Coupling** goes through translators: within a DataStructure by aliases and derived variables, across
  DataStructures (plants and soil, light, environment scalars) by mappings exchanged at fixed points.
- **Scenes** build plant populations from a planting table and run them with their environment models.
- **Exchange**: scenes and DataStructures are written as encoding3d files (one glTF file per time step), which other
  platforms read and ParaView displays, and read back bit for bit; a metafspm model runs with another platform's
  model in an encoding3d coupling session, that model a component (`RemoteComponent`).

## Using models

For those who want to run a plant model built with metafspm: its core ideas in plain words, through use case 1
(components, plant models, links, the scene and its step, the recorder), and the models available with their
tutorials.

```{toctree}
:maxdepth: 1

metafspm in five minutes <five_minutes>
From a scenario to its outputs <tutorial>
Running on a GPU <gpu>
```

## Building models

For those who write, port or couple models: the framework's parts, its declaration conventions, the migration of an
existing model, the API.

```{toctree}
:maxdepth: 2

Framework guide <user>
Conventions <conventions>
Migrating a model <migration>
API reference <ref>
```

## Contact and contributing

Questions, bugs and feature ideas: open an issue on the project's repository.

## Authors

Tristan Gérault and Christophe Pradal.

## License

MetaFSPM is released under the CeCILL-C license.
