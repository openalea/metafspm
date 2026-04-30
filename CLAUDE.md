# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in development mode (inside activated conda env)
pip install -e .

# Run all tests
python -m pytest

# Run a single test file
python -m pytest test/test_graph_system_decorators.py -v

# Run a single test function
python -m pytest test/test_graph_system_decorators.py::test_node_balance_field_tagging -v
```

pytest is configured in `pyproject.toml` (testpaths = `test/`).

**Known failing test:** `test/test_model_assembly.py::test_dummy_model_assembly` fails because it looks for `inputs/dummy_temperatures.csv` relative to the working directory — unrelated to the graph-system work.

## Architecture

### Core layers

```
component_factory.py   — Choregrapher singleton, Functor, and scheduling decorators
component.py           — Model base class (dataclass mixin); declare() field factory
composite_wrapper.py   — CompositeModel for multi-module assemblies
graph_system.py        — GraphView, FieldState, GraphSystem, solvers (Newton / direct)
graph_system_decorators.py — @graph_system method decorator + @node_balance etc.
utils.py               — ArrayDict (numba-compatible dict-like array store)
specializer.py         — JIT specialization helper for Numba acceleration
```

### Choregrapher singleton + scheduling

`Choregrapher` is a **singleton** (`Singleton` base class). All model classes share one instance. Each `@rate`, `@state`, `@axial`, etc. decorator calls `Choregrapher().add_process(Functor(func), name="<step>")` at **class-definition time**, registering the function under its owning class name (taken from `func.__qualname__`).

`Choregrapher().add_time_and_data(instance, sub_time_step, data)` is called in `__init__` to bind instances and data. After this call, all functors are wrapped with `partial(functor, instance, data, data_type)`.

Execution order follows `consensus_scheduling` (list of lists): rows are parallel groups, columns within a row run in order. Steps: `priorbalance → selfbalance → stepinit → rate/state/totalrate/totalstate → axial → potential/deficit/allocation/actual/segmentation/postsegmentation`.

A functor with **no non-self arguments** sets `iterating=True`, which causes the Choregrapher to call `fun(instance)` directly instead of per-vertex dispatch. This is the mechanism used by `@graph_system` to hook whole-graph solves.

### Model base class

`Model` (in `component.py`) provides:
- `__call__` → `pull_available_inputs()` + `choregrapher(module_family=self.__class__.__name__)`
- `link_self_to_mtg()` — initialises `self.props` (MTG properties dict) entries for all declared fields
- `pull_available_inputs()` — merges provider outputs into input fields via `self.pullable_inputs`
- `post_coupling_init()` — override hook for post-coupling setup (e.g. building `_graph_view`)

Field declarations use `declare(...)` which wraps `dataclasses.field(metadata=...)`. The metadata carries: `variable_type` (`"input"`, `"state_variable"`, `"plant_scale_state"`, `"parameter"`), `state_variable_type` (`"intensive"`, `"extensive"`, `"massic_concentration"`, etc.), `location` (`"node"` / `"edge"`, used by the graph-system decorator).

### Graph system decorator API (`graph_system_decorators.py`)

`@graph_system` is a **method decorator placed inside the class body**. It returns a `_GraphSystemDescriptor` whose `__set_name__` hook fires when the class body finishes. That hook:
1. Builds a no-arg `_trampoline(instance)` and registers it as a Choregrapher Functor (step controlled by `schedule_as`, default `"axial"`).
2. Stores the solver spec in `owner._graph_system_specs[method_name]`.
3. Installs `_invoke_graph_system` on the class if not already present.

Each timestep `_invoke_graph_system` (in `graph_system_decorators.py`):
1. Scans the MRO for methods tagged with `__graph_tag__` (`@node_balance`, `@edge_law`, `@graph_jacobian`, `@graph_output`).
2. Inspects their argument names to discover which props to snapshot from `self.props`.
3. Snapshots props once before the Newton loop.
4. Wraps methods as `evaluator(ctx)` callables.
5. Assembles and solves a `GraphSystem`, then writes solution + outputs back to `self.props`.

Method arguments are classified at snapshot time: names in `node_unknowns`/`edge_unknowns` receive the current Newton iterate; everything else is snapshotted once from props. Scalar model attributes are accessed directly via `self`.

### Graph system core (`graph_system.py`)

`GraphView` — compact topology: `node_ids`, `edge_ids`, CSC incidence matrix, optional boundary incidence. `GraphSystem` assembles `EquationBlock` residuals, drives Newton (analytic or FD Jacobian) or `linear_direct` solves via scipy sparse. `EquationContext` is the context object passed to all residual / Jacobian / output methods, giving access to `node_unknowns`, `edge_unknowns`, and `parameters`.

### MTG integration

All models operate on a Multiscale Tree Graph (MTG) from `openalea.mtg`. `self.g` is the MTG object; `self.props = self.g.properties()` is a flat dict of `{property_name: {vertex_id: value}}`. `self.vertices` and `self.focus_elements` filter which vertices are active.

### Module coupling

`CompositeModel` (in `composite_wrapper.py`) orchestrates multi-module assemblies. `declare_and_couple_components()` reads a `coupling_translator.yaml` to wire `state_variable` outputs of one module to `input` fields of another. `pullable_inputs` on each model instance carries these linkages.
