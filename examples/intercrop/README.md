# Intercropping: two populations of one model

Wheat and pea root systems in one row, each species a population of its own, in one soil
(`intercrop.py`). Both are ArchiSimple's `RootArchitecture` (its `WHEAT` and `PEA` parameter sets) in one plant model,
`Roots`; the soil gives them its temperature by depth and hour and gathers their roots.

| what | where |
|---|---|
| **Two populations of one model**: the planting's `population` column, one MPG, model instance and output folder each (`Plant/wheat/`, `Plant/pea/`), the scenario given per population name | `scene()`: `planting(..., [Roots] * 4, {"wheat": WHEAT, "pea": PEA}, population=["wheat", "pea", ...])` |
| **Links by class and by scene name**: `"RootArchitecture"` reaches both populations' components (the soil's temperature, the roots' length into the soil), `"pea.RootArchitecture"` the pea's only (`pea_root_length`) | `TRANSLATOR` |
| **Finding them back** by name or class | `scene.population("pea")`, `scene.model(Soil)` |
| **Times with their units** | `Scene(time_step="1 h", start="2026-05-10 00:00")`, `SceneRecorder(summary_period="6 h")` |
| **Tables per population** | `Summary(..., population="wheat")` |

Run, from this folder (about 5 s):

```
python intercrop.py [days]     # 6 days by default; writes to outputs/intercrop
```

`RootArchitecture` works on a view of the plant's roots (its `entities`); the soil exchanges with that view directly
(its soil temperature, its roots' length). The test (`test/test_intercrop.py`) runs a day in memory.
