# Changelog

## Unreleased

### Added

- Populations in worker processes (route B, B2): `Compute(processes=1 | n | "auto")` (default 1). With n, the scene's
  process forks n workers at the first step, each running a group of whole populations on a core set of its own
  (`os.sched_setaffinity`; on a hybrid CPU the scene's process on the performance cores, the workers on the
  efficient ones, the low-power cores left out), its threads held to it; the scene's process keeps the environment
  models, the exchanges, the recorder and the checkpoints, and a copy of each population's DataStructure, published
  by its worker after each step (its arrays raw through /dev/shm, its state, its MTG only when the topology moved;
  loaded in place) (`scene.runner`: `SequentialRunner`,
  `ProcessRunner`, `placement`, `cpu_topology`, `memory_budget_mb`). `"auto"` picks the most workers the
  populations, the cores and the memory budget allow. `scene.apply(function, population=)` runs a function on the
  live DataStructure where it lives. A worker's death or exception ends the run naming its group, plants, step and
  memory; run.json gains `"processes"` (groups, core sets, threads, memory predicted and measured, balance) and the
  publications' time. The same bits at any process count: use case 1, 2 plants as 2 populations x 3 days, every
  table and frame byte-identical at 1 and 2 processes; checkpoints resume at any count. Limits: whole populations
  per worker (split with `planting(population=)`), `coupling="explicit"`, the cpu, fork (Linux, macOS).
  docs/user.md "Populations in processes", docs/conventions.md "Owners, publication, copies".
- Variable lookups cached (route A, A4): the DataStructure's stores dict is kept while its coarse scales stay and
  each name's store is remembered while it holds the name (`_find`, `_var_stores`, the located views' alike):
  `data_structure.get` of every variable of use case 1's 9 structures 9.2 -> 7.2 ms; use case 1, 2 plants x 3 days
  at 1 thread, run 50.4 -> 49.5 s (-1.7 %, two runs each). Results unchanged, byte for byte.
- The plant-order rule (route B, B1): every scene-level sum over plants is the sum of per-plant partial sums in the
  planting table's order (`CrossMapping.up_partials`, `cross.sum_in_plant_order`, `cross.plant_ranks`,
  `data_structure.plant_ranks`), the plants' union is in plant order (`UnionDataStructure.part_positions(k)` replaces
  `part_slice(k)`; `part_of()`, `source_ids()`, `labels()` and `triangles()` in that order), and the recorder's
  per-scene summaries sum per plant then in that order (per-plant rows in plant order). The same plants in one
  population or split by `planting(..., population=)` now give the same bits. **The single-process results change
  once** by round-off (use case 1, 2 plants x 3 days: the soil's plant_uptake by 2.4e-16 relative, one row; the roots'
  psi by up to 2.8e-15 relative, propagated by the water solve; root_uptake by 1.6e-16 of its scale), not again.
  A per-scene summary over no entity now writes 0.0 (was the integer 0). docs/conventions.md "The plant order".
- Thread and fork safety of the module-level state (route B, B0): the archetype extraction's flag, the systems a
  reduction's library skips and `Compute.current()`'s stack are per context (ContextVars; the archetypes' threads
  run in a copy); the shared positions' cache holds its arrays weakly under a lock, `solve.reduction._FIELDS` is
  locked; the BLAS/OpenMP hold of `Compute.limits()` lasts while any thread is inside one; `os.register_at_fork`
  hooks renew the locks and clear the timings, numba's thread count and the recorder's writer in a child;
  `compute.fork_safe_threading()` refuses numba's workqueue layer. Two populations stepped in two threads give the
  sequential bits (they could fail: "piece has no Dirichlet anchor", or wrong values); use case 1 unchanged, byte
  for byte. docs/conventions.md "Threads and forks".
- The Organ label `Awn` (`labels.Organ.Awn`): a grass ear's awns, an organ of their own (Adel's, formerly
  labelled `Ear`).

### Breaking

- The environment variables `METAFSPM_CHECK_TOPOLOGY` and `METAFSPM_CHECK_ARCHETYPES` are removed (QA-10, no
  fallback): the self-checks are `Compute(checks={"topology", "archetypes"})` (any collection of names among
  `compute.CHECKS`, none by default; an unknown name raises ValueError), recorded in run.json's provenance
  (`"checks"`) and in scenario.json's `"compute"`. docs/migration.md section 8.
- run.json's `"versions"` is replaced by `"provenance"` (QA-9): python's version, every package the scene's models
  and components come from with its version and, for a clone, git commit and modified flag (numpy, scipy, pandas,
  numba, openalea.mtg, encoding3d too), the start date, coupling, links, seeds and the parameters (their hash and
  the non-default values); `"forcings"` summarises the forcings applied (mean, least, greatest of the per-step
  means). Checkpoints (`scene.json`) carry the provenance and each component's declared variables, frames' `writer`
  the versions and commits (and metafspm's `"version"`); `Scene.load_checkpoint` warns with each difference
  (another commit, a parameter, a seed, a variable the checkpoint lacks) and loads. No format change.

- The 2026 renames (QA-6): one word per concept and full words in the public API (`archetype`, `condition`,
  `role`, `aggregation`, `where=`, `absolute_tolerance`/`relative_tolerance`, `data_structure`, `mtg`,
  `step_duration`, `Scene.simulate(steps)`, `Scene.step_count`, `x_range`, `spacing`), the internal helpers
  `_`-prefixed, the public names imported from `openalea.metafspm`. No aliases. Checkpoints and frames carry a
  format number (2); older ones are refused. The table: docs/migration.md, section 7.
