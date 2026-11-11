# Changelog

## Unreleased

### Added

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
