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

```{toctree}
:maxdepth: 2

User guide <user>
Conventions <conventions>
Migrating a model <design/downstream_migration>
API reference <ref>
```

## Contact and contributing

Questions, bugs and feature ideas: open an issue on the project's repository.

## Authors

Tristan Gérault and Christophe Pradal.

## License

MetaFSPM is released under the CeCILL-C license.
