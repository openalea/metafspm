# Changelog

## Unreleased

### Added

- Use case 1 keeps what a user reads and runs (`run.py`, `models.py`, `component.py`, its README with a short note on
  its speed): its benchmark (`benchmark.py`, `memory_probe.py`) and the measurements are in
  `test/benchmarks/use_case_1/` (its README), its tests in `test/examples/test_use_case_1_plant.py` and
  `test_use_case_1_assembly.py` (in metafspm's suite, skipped without the ports).
- A fixed-point coupling's convergence leaves out a NaN at the same entries in both iterates (a variable a model
  leaves NaN where it does not apply, as GRANAP's `Kx`, `PAR_reflectance`, `Ts`): a FixedPoint over a GRANAP step
  can now converge; a NaN that appears or moves, or an infinity, is still no convergence and still raises.
- `Compute(processes=n)` runs a fixed-point coupling (`Scene(coupling=FixedPoint(...))`, route B, B6): the
  iteration in the coordinator, the populations' iterated steps restored and run again in their workers (their
  snapshots kept there), the environment's in the coordinator; the same iterate counts, warnings, FloatingPointError
  messages and bits as one process. Links between two populations are no longer refused: a population waits for the
  publication of the populations before it that its exchanges read (rounds), the same bits as one process.

- Shared archetype store (route B, B4; `solve.archetype_store`): a reduction's archetype kind evaluated by one
  population is read by every other one needing it, the same bits (`Compute(checks={"archetypes"})` checks the kinds
  read too). In one process the populations hold one object per kind (`SequentialRunner`); with `Compute(processes=n)`
  a kind is written once to the run's `/dev/shm` folder, claimed by the first worker needing it (the others wait for
  it instead of evaluating it at the same step), and the publications refer to it by name
  (`SharedStore(external=)`), the coordinator reading it once for all its copies. A shared kind is let go, never
  released in place. Use case 1, 6 plants, first 12 h: 92.0 / 80.5 / 71.0 s at 1 / 2 / 4 processes -> 41.1 / 39.6 /
  35.8 s, the archetypes' evaluation 60 / 95 / 145 -> 9 / 15 / 17 s, the scope's peak 6.4 / 7.1 / 7.4 -> 3.7 / 4.5 /
  5.2 GB; tables and frames byte-identical at 1, 2 and 4. `runner.PER_PLANT_MB` and `COPY_PER_PLANT_MB` retuned
  500 -> 250 from these runs.
- Run folders of `Compute(processes=n)` left in `/dev/shm` by a coordinator killed without its cleanup (out of
  memory) are removed at the next run's start once their process is gone (`runner.remove_stale_folders`, this user's
  folders only); a worker's failure removes its run's folder.

- MTG delta publication for `Compute(processes=n)`: when the topology moves, a worker sends its MPG's change since
  the last publication (`MPG.copy_delta`: the MPG's attributes but its structure caches, the property columns that
  may have changed; unchanged numeric columns left out) and the coordinator patches its copy in place
  (`MPG.apply_structure_delta`, `MPGDataStructure._load_copy_delta`; arrays written in place when their shape
  stayed); the whole MPG only when the variables or the MPG changed and before checkpoints. The first publication
  after the fork is a delta, and the shared parts both processes hold since the fork go by id
  (`SharedStore(forked=)`, `shared_objects`), so the coordinator's copies keep their pages shared with the workers;
  parts no copy refers to any more are let go (`SharedStore.load_referring`, `retain`). New check
  `Compute(checks={"publication"})` compares each patched copy with the MPG sent whole. A reduction's `merged_into`
  cache is no longer pickled (made again; MBs a plant). Use case 1, 6 plants resumed at hour 48 for 24 h at 2
  processes: publication 12.6 -> 4.3 s, scope peak 7.5 -> 5.1 GB, total 59.8 -> 56.6 s; tables and frames
  byte-identical to 1 process.
- Populations in worker processes (route B, B2): `Compute(processes=1 | n | "auto")` (default 1). With n, the scene's
  process forks n workers at the first step, each running a group of whole populations on a core set of its own
  (`os.sched_setaffinity`; on a hybrid CPU the workers on the efficient cores, the low-power cores left out; the
  scene's process unpinned, its environment phase on every core while the workers wait: measured faster, B3), its threads held to it; the scene's process keeps the environment
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
