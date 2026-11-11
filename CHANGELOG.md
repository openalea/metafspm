# Changelog

## Unreleased

### Breaking

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
