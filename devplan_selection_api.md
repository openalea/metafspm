# Plan: one selection argument for every decorator

Your request (2026-10-06): the decorators select their elements in too many ways. All should take `filters=`, with one or several explicit key / value pairs, to operate on chosen elements of their scale.

## 1. Today

| decorator | argument | forms accepted | outside the selection |
|---|---|---|---|
| `@node_balance`, `@edge_law` | `filters=` | `{variable: values}` | the block contributes 0 |
| `@boundary_condition` | `select=` (or `filters=`) | dict, variable name (> 0), mask name, callable | no term |
| `boundary_set` | `select=` | dict, variable name, mask name, callable | no term |
| `@graph_output` | `select=` | dict, variable name, mask name, callable | 0 |
| steps (`@rate`, `@state`, …) | `where=` | a mask name; `"active"` by default, `None` for all | values kept |
| `@graph_system` | `where=` | a mask name | the system is solved on that subgraph |

## 2. Proposal

**One argument, `filters=`, on every decorator:**
- the steps, `@node_balance`, `@edge_law`, `@boundary_condition`, `boundary_set`, `@graph_output` and `@graph_system`;
- `select=` and `where=` are renamed;
- the meaning outside the selection stays each decorator's: 0 for a contribution or an output, no term for a boundary, values kept for a step, the subgraph for a system.

**Its primary form, a dict of key / values written explicitly.** All pairs must hold (AND):

```python
filters={"label": "RootSegment"}                       # equal to a value (label names resolved)
filters={"tissue": [EPIDERMIS, CORTEX]}                # one of several values
filters={"is_evaporating": ">0", "length": "<=0.03"}   # conditions, as in masks
```

**Is a dict the most flexible?** It is the most readable, and it covers most uses. Three things it cannot express:
1. **Geometry:** e.g. the top layer of a soil grid, which has no variable to test, or a radius around a point. A callable `ds -> boolean array` does it.
2. **"Or", and reuse:** a named mask, defined once with `ds.define_mask`, shared with the mappings (`CrossMapping(mask=)`) and the lifecycle (`active`, emergence), and versioned so that the structures built on it update when it changes.
3. **Elements of a coarser scale:** e.g. the Compartments of root segments, the root label being a SubOrgan property. Proposed (Q3): a key may name a variable at a coarser scale, its values read at each element's entity at that scale.

So I propose `filters=` takes the dict (the form used and documented everywhere), plus a mask name and a callable for the cases above, under the same argument.

**Steps and the `active` mask.** Today a step computes on the `active` entities by default (emergence, dead tissues), and `where=None` computes on all of them. With `filters=`, a step would compute on the active entities among the filtered ones; the active mask is the lifecycle, not a selection.

## 3. Questions

- **Q1 — forms.** Should `filters=` take the dict of key / values plus a mask name and a callable, or the dict only? **Recommendation:** all three. The dict is the documented form; the other two cover geometry, "or" and masks shared with mappings and emergence.
  → answer:
- **Q2 — the `active` mask in steps.** Should filtered steps stay restricted to the active entities, with an explicit opt-out (`include_inactive=True`, rare: e.g. a step setting the state of tissues before they emerge)? Or should a step see only its `filters`, and add `{"active": ...}` itself? **Recommendation:** stay restricted by default, with the opt-out. Otherwise every step of every model must remember emergence.
  → answer:
- **Q3 — keys at coarser scales.** Allow a key naming a variable at a coarser scale than the elements, read at each element's entity? For example, `filters={"label": "RootSegment"}` on anatomy Compartments, `label` being the segments'. **Recommendation:** yes. It is what "elements of the targeted scale" needs in anatomy mode.
  → answer:
- **Q4 — `@graph_system(where=)`.** Rename it `filters=` too? It solves the whole system on the subgraph of the selected nodes (dropped edges, frozen nodes, a well-posedness check), a stronger effect than the others. **Recommendation:** yes, for one name everywhere. Its docstring states the subgraph meaning.
  → answer:

## 4. Then

1. One selection resolver shared by every decorator: dict, mask name or callable → a boolean array at the element location, cached on the DataStructure's mask machinery so it follows variables and topology.
2. The renames: `select=` and `where=` become `filters=`. Downstream models are not ported yet, so no aliases are kept.
3. Tests per decorator, docs (user guide, conventions), the CHANGELOG, and the example rewritten on `filters=`.

## 5. Done (2026-10-06)

- **Your answers:** yes to every recommendation (Q1–Q4).
- **One resolver**, `Filters` in `solve/decorator.py`: a dict, a mask name or a callable becomes a DataStructure mask at the elements' location (cached, following its variables and the topology). Mask rules now take any comparison threshold and variables of coarser scales.
- **Every decorator takes `filters=`:** steps, `@node_balance`, `@node_rate`, `@edge_law`, `@boundary_condition`, `boundary_set`, `@graph_output`, `@graph_system`.
  - `select=` and `where=` are gone; `include_inactive=True` replaces `where=None` on steps.
  - A string is a mask name, so a variable is written `{"is_collar": ">0"}`.
- **Tests:** `test/components/test_filters.py` (9). The tests and the example are migrated.
- **Example:** the two transports declare their graph systems separately (no `_WaterFlow`). The structures' steps use dict filters; the `root_surface` mask stays, for the soil mapping. The results are unchanged.
- **Found:** inside a filtered balance, values read through `self` (e.g. `self.previous()`) are not sliced to the selection. This was already the case, and is now documented. Slicing `previous()` too would need the framework to know which block calls it.
