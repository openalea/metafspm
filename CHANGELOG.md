# Changelog

## Unreleased

### Breaking

- The 2026 renames (QA-6): one word per concept and full words in the public API (`archetype`, `condition`,
  `role`, `aggregation`, `where=`, `absolute_tolerance`/`relative_tolerance`, `data_structure`, `mtg`,
  `step_duration`, `Scene.simulate(steps)`, `Scene.step_count`, `x_range`, `spacing`), the internal helpers
  `_`-prefixed, the public names imported from `openalea.metafspm`. No aliases. Checkpoints and frames carry a
  format number (2); older ones are refused. The table: docs/migration.md, section 7.
