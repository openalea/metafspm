# Design note: time loops, MTG sync, solve-time data and typed variables (step 4)

Status: **agreed** (2026-10-02: "Go on with step 4", taken as agreement with the recommendations T1–T7; the answer lines were left empty). 4a done (`6bfcaf7`), 4b done. It covers step 4 of `devplan_datastructures.md` §7:
- DS10: graph systems with their own time loop and sub-stepping;
- DS4: MTG synchronisation policy;
- DS9: solve-time data policy;
- DS12: non-float and categorical variables.

Branch `data_structure_api`, written against `f648b06` (steps 1 to 3 complete). Code starts only after this note is agreed. The points to agree are in §6, with answer lines.

It builds on D3 (MTG optional, write after every call), D5 (`previous()` per solve and per step; per-component sub-stepping, Q8) and Q22 (non-float variables are not solver or transport variables).

## 1. Where we start from (code facts)

- **One solver step per call.** A `@graph_system` call is one `solver.step_once(spec, previous_node_fields, dt=self.time_step)`.
  - `DAESolver.solve(spec, t_span)`, the adaptive loop, is not reachable from the decorator.
  - `NewtonSolver` and `ImplicitEulerSolver` return a **zero error estimate** from `_integrate_step`, so that loop would accept every step and grow `h`. It is not adaptive for them.
- **Equations read time from the instance.** User equations write their time terms with `self.time_step` and `self.previous("c")` (UC1, UC3, the grid diffusion of 3d).
  - `previous()` returns the state saved when the solve was built, so it is fixed during the solve.
  - The solver passes its own `previous_node_fields` and `dt` in the `EquationContext` (`ctx.previous_node_fields`, `ctx.dt`), but the decorator's evaluators do not hand them to the equations.
  - So **sub-stepping inside one solve would silently keep the previous state and the step length of the whole call.**
- **The Choregrapher already sub-steps per component.** `__call__` repeats the component's whole schedule `simulation_time_step / sub_time_step` times, and each repetition of a graph system is a new `step_once` with `dt = self.time_step`. That is per-component sub-stepping at the level of the whole schedule (Q8).
- **MTG sync (DS4, mostly done in step 1b).**
  - State variables are written after every call (N4).
  - MTG-backed parameters are re-read before each graph solve, not at the start of a call.
  - There is no way to opt out of MTG writes for a component.
- **Solve-time data (DS9).** `_snapshot` copies every required variable at every solve (`_read_array` returns `np.array(...)`), parameters and inputs included.
- **Typed variables (DS12).**
  - `register` stores every variable as `float64`. Labels and types are integer codes in the MPG (`LabelsConfig`), coerced to float in the DataStructure. `boundary_set` and `define_mask` rules compare those floats with integer codes.
  - Lists such as `xylem_vessel_radii` cannot be registered: Q22 kept them on the MTG or as instance attributes.
  - Label codes are assigned at runtime (UC5 used a callable select for that reason), and the same name can exist in several label groups (e.g. a `Symplastic` Compartment and a `Symplastic` Connection).

## 2. Time loops and sub-stepping (DS10), sub-step 4b

```python
@graph_system(node_unknowns=["water_potential"], solver="implicit_euler", integrate="substeps", n_substeps=4)
@graph_system(node_unknowns=["solute"], solver="implicit_euler", integrate="adaptive", rtol=1e-4, atol=1e-8)
```

- **`integrate="step"` (default):** today's single step of length `time_step`.
- **`integrate="substeps"`:** `n_substeps` steps of `time_step / n_substeps` within one call. Boundary sets and their values are re-read between sub-steps. Inputs from other components stay those at the start of the call (Q8).
- **`integrate="adaptive"`:** error control by **step doubling**, done by the framework (T3):
  - one step of `h` is compared with two steps of `h/2`;
  - the step is accepted when the difference is within `rtol·|x| + atol`, and `h` is adapted;
  - `min_step` / `max_step` bound `h`.

  This works for every solver, including those whose estimate is zero. The solver's own estimate is used instead when it is non-zero (IVP).
- **What the equations see during a sub-step (T1, T2):**
  - `self.dt`: the length of the current (sub-)step, managed by the framework. It equals `time_step` with `integrate="step"`.
  - `self.previous(fn)`: the state of `fn` at the start of the current (sub-)step;
  - `self.previous(fn, at="solve")`: the state at the start of this call's solve;
  - `self.previous(fn, at="step")`: the state at the start of the component's call (the Choregrapher step), for operator splitting between several graph systems of one component.

  The evaluators hand `ctx.dt` and `ctx.previous_node_fields` to the instance during each evaluation. With `integrate="step"`, these give exactly today's values.
- **Required:** a graph system declared with `substeps` or `adaptive` must use `self.dt` and `self.previous()` for its time terms. Equations using `self.time_step` would integrate with the wrong step. The framework cannot detect this. It is documented, and the 4b tests show the difference.
- **As implemented (4b):** `BoundaryConditions` is not attached to the specs built by the decorator, so there is no `t` to pass. Time-varying boundaries on the decorator path are boundary sets, whose values are re-read at every sub-step.
- **Validation:**
  - linear diffusion (UC1 and the grid of 3d) with `substeps` matches n implicit Euler steps of `dt/n` done by hand;
  - `adaptive` on a stiff 2-node exchange matches the analytic solution within the tolerance, with fewer steps when the exchange is slow;
  - `previous(at="step")` across two graph systems of one component (operator splitting);
  - `integrate="step"` results unchanged everywhere (UC1–UC5).

## 3. MTG synchronisation policy (DS4), sub-step 4a

- **`mtg_sync = "after_call"`** (the default, today's behaviour since 1b) or **`"never"`**, as a class attribute of `DataStructureComponent`.
  - `"never"`: no write-back. For components whose results are read only through the DataStructure (DS-only models, or performance), the Logger reads the DataStructure.
  - With no MTG (grids), nothing is synchronised, as today.
- **Parameters are re-read from the MTG at the start of every call**, in `pull_available_inputs` (T4). The re-read before each graph solve is kept, since a solve can also be called directly.
- **Validation:** a `"never"` component leaves the MTG untouched; a parameter changed on the MTG is seen by a `@rate` at the next call.

## 4. Solve-time data (DS9), sub-step 4a

- **Read-only views (T5).** Parameters and inputs in the snapshot become read-only views of the DataStructure arrays, with no copy. Unknowns, previous states and integrated amounts stay copies.
  - On an active subgraph, the fancy-indexed arrays are copies anyway.
  - An equation that writes into an input or a parameter (`K[...] = ...`) now raises `ValueError: assignment destination is read-only`, instead of silently corrupting the DataStructure.
- **Validation:** a benchmark on a 20 000-node chain with ten parameters (snapshot time before and after, reported in the devlog), and an equation writing into a parameter raising.

## 5. Typed variables (DS12), sub-step 4c

- **Integer variables.** `register(..., dtype=int)`, and `declare(..., dtype="int")` / `parameter(..., dtype="int")` for labels, types and indices.
  - Values stay integers in the DataStructure, read from the MTG as integers.
  - Writing a non-integral value raises.
  - They can be read by steps, filters, masks and boundary sets. They are not solver unknowns: a graph-system unknown must be a float.
- **Label names (T6).** Filters, mask rules and `boundary_set` selects on a variable named `label` (or declared `labels="<group>"`) accept names: `{"label": ["RootSegment", "LeafElement"]}`.
  - Names are resolved through the MTG's `LabelsConfig`: first in the group of the variable's scale (SubOrgan labels for a SubOrgan variable, Compartment and Cell labels in anatomy mode), then in the other groups.
  - An ambiguous or unknown name raises with the candidates.
  - This replaces callable selects such as UC5's.
- **Object variables.** `register(..., dtype=object)` and `declare(..., dtype="object")` for lists and records (e.g. `xylem_vessel_radii`).
  - They are stored per entity, carried over by growth, readable by steps (`vectorized=False`), and written to and from the MTG by identity.
  - They are rejected as graph-system arguments and by derivations, mappings, the Coupler and the Transport, with a clear error (Q22).
- **Validation:**
  - labels kept as integers from the MTG;
  - name filters on SubOrgan labels, and in anatomy mode on Compartment labels;
  - an ambiguous name raising;
  - an object variable carried over growth and rejected by a graph system and a link.

## 6. Points to agree

- **T1, `self.dt` in equations.** A graph system using `substeps` or `adaptive` writes its time terms with `self.dt` (the framework's current step length) instead of `self.time_step`. **Recommendation:** yes. `self.time_step` stays the component's step, and `self.dt == self.time_step` with `integrate="step"`, so nothing changes for existing models.
  → answer:
- **T2, `previous()` levels.**
  - `previous(fn)`: the start of the current (sub-)step;
  - `previous(fn, at="solve")`: the start of the call's solve;
  - `previous(fn, at="step")`: the start of the component's call.

  **Recommendation:** yes. With `integrate="step"`, the first two are the same, and they are today's `previous()`.
  → answer:
- **T3, adaptive by step doubling,** in the framework, for every solver (Newton and implicit Euler return no error estimate). The alternative is to restrict `adaptive` to solvers with their own estimate (IVP). **Recommendation:** step doubling.
  → answer:
- **T4, MTG sync:** `mtg_sync = "after_call" | "never"`, and parameters re-read at the start of every call. **Recommendation:** yes.
  → answer:
- **T5, read-only views** for parameters and inputs in graph-system snapshots; writing into them raises. **Recommendation:** yes.
  → answer:
- **T6, typed variables:**
  - integer variables kept as integers;
  - label names resolved through the MTG's `LabelsConfig` (the variable's own scale group first);
  - object variables stored per entity and rejected by solvers, links and transport.

  **Recommendation:** yes.
  → answer:
- **T7, order:** 4a (DS4 + DS9, small) → 4b (DS10) → 4c (DS12). **Recommendation:** yes.
  → answer:
