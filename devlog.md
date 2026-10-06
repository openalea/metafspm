# metafspm devlog

## 2026-09-29: Test suite state (branch `release2026`, HEAD `557e4a8`)

This is a snapshot of the suite as it stands. No code was changed to produce it. How it was gathered:
- 7 parallel read-only audits: one per test area, one on source coverage, one on hygiene and CI.
- An adversarial critic re-checked 20 of their claims and corrected 7.
- A final pass of manual spot-checks.

Probe scripts were run from a scratch directory only.

---

### 1. Headline numbers

| Metric | Value |
|---|---|
| Collected tests | **312** (275 `def test_` in 27 active files; parametrization adds the rest) |
| Result | **312 passed, 0 failed, 0 skipped, 0 xfail** |
| Warnings | 7, all `scipy.linalg.LinAlgWarning` (ill-conditioned matrix) from `scipy_anderson` cases in `test_scipy_root_solver.py` |
| Hidden warnings | none: `-W error -W ignore::LinAlgWarning` still gives 312 passed |
| Wall time | 2–4 s (collection alone takes about 1.5 s because several modules build MPGs at import time) |
| Line coverage | **67.4 %** (2454/3643 executable lines), measured with `sys.monitoring` because `coverage`/`pytest-cov` are not installed |
| Disabled test files | 4 (`est_*.py`); **all 4 fail at import** |
| Determinism | Stable: 2 full runs, reverse-order run, and all 27 files run alone. The only randomness is seeded `default_rng` |
| Side effects | No collected test writes to disk; the tree stays clean |
| Doctests | none in `src/` |

Environment: Python 3.13, pytest 9.1.1, conda env `metafspm`, package installed editable. Not installed: pytest-cov, coverage, nbmake, pytest-randomly, pytest-xdist.

### 2. Per-area inventory

| Area | Tests | Nature | Linked issues |
|---|---|---|---|
| `data_api_tests/` | 124 | Real unit tests of `data_api.py`, `arraydict.py` and the MPG DS. Several tautologies (see §5) | #15 #16, `test_update_topology` #17 |
| `solver_tests/` | 135 | Numerical one-step tests against closed forms (good tolerances). About 40 are interface/shape/issubclass checks | #15 #16 |
| `graph_system_tests/` | 35 | UC1 nitrogen transport at SubOrgan (18) and Organ (17) scale. Real, tight physics checks; the two files are near-duplicates (about 800 redundant lines) | #17 of #23 |
| `mpg_tests/` | 13 | MPG init, traversal, aggregation, filtering, graph building. Mostly real asserts; `test_partial_traversal` only prints | #11 #14 |
| `component_use_tests/` | 4 | 1 real assert (`test_reinit_Choregrapher`). The other 3 are import/instantiation smoke tests | #13 |
| `components_coupling_tests/` | 1 | `test_import_composite`: instantiation only, no assert | – |

Per-file counts:
- `data_api_tests`: 1 / 17 / 44 / 15 / 17 / 15 / 15
- `solver_tests`: 11 / 11 / 11 / 10 / 18 / 18 / 21 / 35
- `graph_system_tests`: 18 / 17
- `mpg_tests`: 1 / 4 / 5 / 2 / 1
- component tests: 1 each

8 files contain exactly one test.

#### Disabled `est_*` files

| File | Current failure | Distance to green |
|---|---|---|
| `graph_system_tests/est_mms_solver.py` (12 tests) | `ImportError: SolverSpec` from `system_specs` (it moved to `solver.py`) | **One-line import fix** gives 12/12 passing (plus 3 anderson warnings). Covers MMS convergence order, `jac_sparsity_matrix`, 6-solver agreement |
| `graph_system_tests/est_uc2_water_munch.py` (4) | `No module named 'conftest'`: `graph_system_tests/conftest.py` was deleted in `557e4a8` | Deep. After restoring conftest: `declare(location=)` was removed, and then `generate_anatomy_in_mtg.array_at_scale` crashes (`'dict' object has no attribute 'indices_of'`). It also needs the `FunctionalComponent(data_structure=ds)` migration listed in `component_api_changelog.md` |
| `graph_system_tests/est_uc3_uc4_anatomy_bc.py` (5) | same `conftest` import error | Same chain as UC2, plus the `_boundary_ports` wiring |
| `components_coupling_tests/est_model_assembly.py` | `No module named openalea.metafspm.composite_wrapper` (and `component_factory`, `dummy_components`) | Stale paths plus cwd-relative I/O. A rebuilt copy runs and gives `DOC={1: 3600.0}` |

`component_api_changelog.md` is partly stale: it shows `MPGDataStructure(g, scale=6)`, and it does not mention `edge_mapping`, the conftest deletion, the `declare(location=)` removal, or the MMS file.

### 3. Coverage map (line-level, "executed", not "asserted")

| Module | % | Main untested parts |
|---|---|---|
| `solve/specializer.py` | **9.3** | everything (numba `specialize_method_recursive`) |
| `scene/scene_wrapper.py` | **8.9** | everything (`play_Orchestra`, workers, `plan_affinity`, `free_cpu`) |
| `coupling/composite_wrapper.py` | **10.1** | everything past an empty `__init__` (coupling, translator, input tables, docs) |
| `solve/legacy_functor.py` | 41.0 | `total`, supplementary output, ArrayDict and ndarray paths |
| `solve/solver.py` | 60.6 | **`ODESolver.solve` / `DAESolver.solve` (the whole adaptive time loop)**, step-size helpers, `ODESystemSpec` path, edge-unknown/algebraic recovery, `.order` |
| `solve/system_specs.py` | 75.9 | `ODESystemSpec`, `BoundaryConditions.update`, analytic `jacobian_evaluator`, `GraphSystem` wrapper methods |
| `coupling/choregrapher.py` | 76.6 | `add_schedule`, inherited-process merge, `focus_elements`, numba branch |
| `data_structure/arraydict.py` | 78.6 | `keys_array`, `assign_all`, `reindex_sorted_inplace` |
| `data_structure/data_api.py` | 80.6 | `from_mtg_subset` happy path, `validate`, non-ArrayDict slow path (`distal`/`mean` edge mapping), boundary ports, 3-D Laplacian |
| `data_structure/mpg.py` | 82.9 | `from_mtg`, `filter_in`/`filter_out` in `populate_graph*`, `_tip_component` cross-complex wiring |
| `coupling/component.py` | 84.1 | variable-category properties, `__post_init__` TypeErrors, bio-scale refresh |
| `solve/decorator.py` | 88.4 | `@graph_jacobian`, `type_filter` masking, `graph_system` argument error paths |
| `data_structure/configs.py` | 98.0 | `ScalesConfig.name_translator/names` |

**Executed but never asserted.** These run during the suite, but no test checks their result:
- MTG write-back: `write_node_to_mtg` / `write_edge_to_mtg`, called via `component.write_back_to_mtg`
- `to_props_dict`
- `_refresh_from_bio_scale`
- `derive_outputs`

### 4. Source bugs that the tests currently mask (confirmed by probe or by reading the code)

1. **Solvers ignore the `x` argument.** Newton, ImplicitEuler, LinearDirect and ScipyRoot restart from `spec.pack_unknowns()` (`solver.py:519,584,706,742`). Single-step tests rebuild the spec each step, so they never notice. Inside the untested `DAESolver.solve` loop, the initial guess would never advance.
2. **`ScipyIVPSolver` with edge unknowns:**
   - it returns only the node DOFs (`solver.py:869,876`);
   - the inner edge recovery uses `spec.pack_unknowns()` instead of `y_node` (`:851`).
3. **`make_solver(method, config)` silently discards a non-`SolverConfig` config** (for example a dict) and uses the defaults (`solver.py:943-945`).
4. **`GraphView.node_local_index` is wrong on views built by `to_graph_view`.** It uses `searchsorted`, which assumes sorted ids, but MPG ids come out descending. 0 of 14 seedling lookups are correct, and missing ids return an out-of-range index instead of raising.
5. **`MPGDataStructure.from_legacy` is wrong:**
   - it yields a length-1 property array when `n_nodes()==3`, and the value is 0.0;
   - the test compares the copy against its own source, so it cannot fail.
6. **`MultiGridDataStructure` R/P are 1-D operators applied to flattened N-D state.** On a 3-D ramp, restriction gives an all-zero coarse field. The 1-D approximate-identity test uses `atol=0.5`, and restriction does not preserve constants at the boundary.
7. **`LabelsConfig.Connection.Apoplastic == Compartment.Apoplastic == "ApoplasticNode"`** (`configs.py:184,190`). This is a label collision, probably a typo.
8. **`MPG.graph()` crashes on graphs wired by `populate_graph_custom_connections`**, because `vertex_id` is never set (`'dict' object has no attribute 'order'`).
9. **`array_filtering(filter_out=...)` keeps only the vertices that carry the property.** `populate_graph` does the opposite, and this difference is undocumented.
10. **`CompositeModel.documentation` / `.inputs` crash** on any `FunctionalComponent` model (`None.__format__`, `KeyError 'variable_type'`). There is also a stale loop variable at `composite_wrapper.py:184`.
11. **`scene_wrapper.stand_initialization`** compares `model_picker <= frequency` instead of the running total (`:183`), so it picks the wrong model when there are more than 1 model. Found by reading the code, not executed.
12. **`boundary_condition(location="edge")`** is accepted but always masks over nodes (`decorator.py:503`). A Dirichlet BC whose filter property is absent is **silently inactive**.
13. **`data_api.py:741,775`** have `except Exception: pass` in the MTG write-back, and `choregrapher.py:75-84` has a bare `except: pass` around numba specialization. Both swallow errors silently.
14. **Example script `data_api_tests/examples/example_mpg_data_structure.py` crashes at line 68**, so its committed PNG is stale. Running the other example scripts rewrites tracked PNGs.

### 5. Weak, tautological, or misleading tests

- **No assertion:** `test_bare_component`, `test_dummy_components_declaration`, `test_import_composite` and `test_partial_traversal`. `test_field_consensus` only checks the default it just set.
- **Cannot fail on logic:**
  - `test_newton_ignores_prev_fields` and `test_newton_ignores_dt`: the residual reads neither argument.
  - `test_graph_view` incidence tests: they check a matrix the test helper built itself.
  - `test_data_structure_abstract.py:163-216`: checks the in-file stub class.
  - `test_invalidate_topology_rebuilds_index_map`: 3 == 3.
  - `test_from_legacy_*`.
- **Too loose:**
  - Root-solver tests use `abs(x) ≈ 2`, so they accept the wrong root.
  - `test_registry_has_expected_methods` uses `issubset`, and its expected set omits `newton_optional_jacobian`.
  - `test_graph_building` uses uniform conductance, so index permutations go undetected.
  - `test_mtg_props_auto_mapped` uses uniform values, so it cannot tell proximal, mean and distal mapping apart. Its docstring says "mean"; the code uses "proximal".
  - `test_filtering` checks label presence only.
- **Pass for the wrong reason:**
  - `test_uc1_stepinit_and_graph_system_via_choregrapher`: all 8 graph systems run, and the "axial" ones run **before** `@rate`, the opposite of the docstring. The test passes only because the initial state is uniform.
  - In the SubOrgan UC1 file, the "root" Dirichlet/Neumann BC is applied to local index 0, which is a **leaf tip** (VID 39) because local order is descending. The Organ file correctly finds the real root.
- **Neumann with an implicit balance:** it is claimed in the test docstring but not tested.
- **Dirichlet only checks the pinned node,** not the rest of the field or the residual.
- **Stale docstrings:** `decorator.py:143` (`_mean` vs `_amount`), UC1 `:560,:587,:804`, organ `:18,:512`, and `test_scipy_ivp_solver.py:74`.

### 6. Infrastructure and hygiene

- **CI and local runs differ.**
  - The conda recipe runs `cd test && pytest -v` with only `test/**` copied, so `pyproject.toml` (with `pythonpath`/`testpaths`) is absent.
  - The suite survives through per-file `sys.path.insert` hacks and prepend-mode `from conftest import ...` in 6 solver files. These break under `--import-mode=importlib`.
  - Simulating CI locally still gives 312 passed.
- **The GitHub workflow runs no pytest step of its own.**
  - It calls the openalea reusable workflow with a matrix of Ubuntu, macOS, macOS-Intel and Windows × Python 3.11–3.14.
  - `release2026` does not trigger CI until a PR is opened.
- **`test/utils.py::deep_reload_package` bugs:**
  - With a string argument, used by the 3 `component_use_tests` files, it iterates the characters, so it does nothing.
  - With a list argument, used by the UC1 files, it purges all of `openalea.*` at collection time, so different modules hold different class objects. This is a latent isinstance/registry hazard.
  - The docstring says it returns the module; it returns None.
- **Global state leaks:**
  - `test_choregrapher_reinit` resets `Choregrapher._instance` and never restores it. A probe shows later-defined components would silently no-op.
  - `LabelsConfig` mutates class attributes on the first `MPG()` (a second MPG gets an empty translator).
  - Module-level shared `simple_seedling.g` is mutated by the aggregation tests.
- **Dead or half-wired items:**
  - `nbmake` is in the test extras but there are no notebooks.
  - `pytest-cov` is in the dev extras with no config.
  - `requires-python>=3.8` and the 3.8–3.13 classifiers do not match the 3.11–3.14 CI matrix.
  - `.pytest_cache/` is not in `.gitignore`.
  - Helpers never run by the suite: `generate_anatomy_in_mtg.py` (616 lines, broken) and `mpg_tests/plotting.py` (316 lines).
  - `test/inputs/*` is used only by the disabled assembly test.
- **Structure:**
  - There are no pytest fixtures outside `solver_tests/conftest.py`.
  - `_make_linear_mpg` is duplicated three times with small differences.
  - Seedling counts (14/13/15/16) are hard-coded in 6 tests.
  - Private attributes are used heavily (`_vid_to_idx`, `_invoke_graph_system`, `_previous_fields`, …).
  - Disabling tests by renaming them `est_` is invisible in reports.
  - The directory names `component_use_tests` / `components_coupling_tests` are inconsistent.

### 7. Largest gaps, ranked by risk

1. The adaptive time loop (`ODESolver.solve`, `DAESolver.solve`) and step control: 0 %. Combined with bug #1, multi-step integration is effectively unverified.
2. DAEs with edge unknowns (the UC2/UC3 kind): algebraic recovery and IVP inner Newton are untested, and bug #2 is present.
3. Multi-model coupling (`CompositeModel`, 10 %) and scene/multiprocessing orchestration (9 %).
4. MTG ↔ data-structure bridge: write-back, `distal`/`mean` mapping and bio-scale refresh are executed but never asserted, and errors are swallowed silently.
5. `@graph_jacobian`, analytic Jacobians, `type_filter` / `filters=` masking, edge BCs and edge `graph_output`.
6. `ODESystemSpec` and time-varying `BoundaryConditions`.
7. Numba specialization path.

### 8. Quick wins noted, none applied

- Re-enable `est_mms_solver.py` with a one-line `SolverSpec` import fix: +12 tests and real coverage of the sparse Jacobian and convergence order.
- Fix `deep_reload_package` to accept a string. Add a fixture that saves and restores `Choregrapher._instance`.
- Add a `filterwarnings` policy for the anderson LinAlgWarning, or drop anderson from the tiny-system tests.
- Switch the conda test command to run from the repo root with `pyproject.toml` copied, or add a `test/conftest.py` for paths. Replace `from conftest import` with a helper module.
- Mark the disabled files with `pytest.mark.skip(reason=...)` or `collect_ignore` instead of the `est_` prefix.

---

## 2026-09-29 (later): Scope set to wrapper coverage; `devplan.md` created

- New planning file: `devplan.md` (user-editable). Epic **W** covers `scene_wrapper` and `composite_wrapper`; the backlog B1–B10 is carried over from the audit above.
- The user added reference usage in `test/provide_usage_examples/`: `composite_wrapper_example.py` (`GrassBRIDGES`) and `scene_wrapper_example.py` (`play_Orchestra` with RhizoSoil and LightModel). These files are untracked and not collected by pytest.
- **The examples do not match `release2026`.** They target the API and module layout of `origin/publish_WB`, which has 3 wrapper commits that `release2026` lacks:
  - `717172f`: `translator_path` is the full YAML path.
  - `dba0143`: `sowing_depth` is a kwarg. On `release2026`, the example call to `play_Orchestra` raises `TypeError`.
  - `5bd0645`: cosmetic.
- The examples also use flat import paths that no longer exist here: `openalea.metafspm.utils` (including `mtg_to_arraydict`), `.composite_wrapper`, `.scene_wrapper` and `.component_factory`.
- None of the downstream packages are installed in the `metafspm` env: rootbridges, rootcynaps, cnwgrass, fspm, adel, caribu, rhizosoil, grassbridges. The new wrapper tests will therefore use in-repo stand-in models. numba, psutil, yaml and pandas are available.
- Newly spotted wrapper issues (to be pinned by xfail-first tests, see devplan W2–W5):
  - `play_Orchestra` always waits at least 10 s, because of a hard-coded `sleep(10)`.
  - `del b` raises `NameError` when no plant was created.
  - `plan_affinity` / `free_cpu` depend on a cwd-relative `outputs/` directory and crash when it is missing.
  - `psutil.cpu_affinity` does not exist on macOS, which is in the CI matrix.
  - `light_worker` hard-codes the cwd-relative `inputs/meteo_Ljutovac2002.csv`, and `light_scenario` is unused.
  - `soil_worker` requires a `logger_class`.
  - The soil component name `"SoilModel"` and the SharedMemory shape `(35, 20000)` are hard-coded in both the wrapper and the plant model.
- No code has been changed. Open questions Q1–Q10 are in `devplan.md`.

---

## 2026-09-29 (later): Decisions Q1/Q2/Q3/Q10; `publish_WB` wrapper port applied

- **Decisions:**
  - Q1: port the `publish_WB` wrapper API.
  - Q2: no compatibility shims; downstream packages pass explicit paths.
  - Q3: coupling must work dynamically through the variables of `DataStructure` derivatives, which opens design epic **WD** in `devplan.md`.
  - Q10: reference files were added to `test/provide_usage_examples/`: RhizoSoil composite and core SoilModel, LightModel, and the WheatBRIDGES translator. The Logger interface is still missing.
- **W0.1 done (not committed):** ported `717172f`, `dba0143` and `5bd0645` into `scene/scene_wrapper.py` and `coupling/composite_wrapper.py`.
  - `open_or_create_translator` now takes the full YAML path.
  - `play_Orchestra` gained `sowing_depth=[0.025]`, replacing the unused `max_depth`.
  - `debug_runs=False`.
  - Both files now match `publish_WB` except for import paths. Suite: **312 passed, 7 warnings**; no test covered these paths.
- **Findings from the reference files:**
  - **The handshake has no spare capacity.** The real translator needs 25 plant-side rows and 10 soil-output rows. That is exactly the hard-coded SharedMemory height of 35, so one more coupled variable gives an `IndexError`.
  - **Unit conversions are silently dropped.** In `couple_current_with_components_list`, a same-name link is skipped whatever its factor (`composite_wrapper.py:162-164`). The real translator therefore loses 6 conversions, `SoilModel ← RootCNUnified` × `12 * 6` (cells_release, hexose_exudation, hexose_uptake_from_soil, mucilage_secretion, phloem_hexose_exudation, phloem_hexose_uptake_from_soil). The soil reads the raw plant values. Q11 asks whether the translator or the code is wrong.
  - **SoilModel is on an older API.** It still uses the pre-refactor `openalea.metafspm.component.Model` + `component_factory`, which is older than `Component`.
  - **Downstream `LightModel.run` can crash.** If the first step triggers no Caribu run, it raises (B11).
- Plan updated: W-tests are tagged `[contract]` (they must survive the WD refactor) or `[legacy]`. New open questions are Q11–Q14.

---

## 2026-09-29 (later): Q4–Q14 answered; plan re-sequenced

- **Answers recorded** in the `devplan.md` decisions log. Your wording is kept verbatim under *Answered*. The ones that change the plan:
  - **Q13, live reading:** components read DataStructure arrays directly; the props snapshot is removed.
  - **Q14, full switch:** plants use `MPGDataStructure` and the soil uses the 3-D grid, with no MTG-props path left. As a result, the `[legacy]` test tier is dropped (W2.7 and W2.8). The wrapper tests assert only on observable values, so they can be retargeted after WD.
  - **Q6:** `recursive_reload` is replaced by an in-place `Choregrapher.reset()`.
  - **Q7:** the soil and light workers get pinned cores too.
  - **Q9:** no pinning on macOS.
  - **Q8:** the meteo file comes from `light_scenario`.
  - **Q5:** the stale input-table cache gets fixed.
  - **Q12:** queue timeouts are out of scope for now.
- **Q11:** you renamed the soil-side receivers in the translator and SoilModel with a `_massic` suffix. The metafspm code still ignores a same-name factor ≠ 1, so W2.5 stays.
- **Q10b, Logger** (`logger_api_reference.py`): the Logger accepts only an exact `openalea.mtg.MTG` (root/shoot) or a `dict` (soil) in `model_instance.data_structures`, and raises on any other type. With Q14 it will reject both MPG and grid DataStructures, so it needs a migration path (new task WD.9, question Q15).
- **Q4:** you asked about a scale-aware translator and YAML vs Python. The recommendation is in WD.0 (YAML stays the source of truth, links are loaded into typed Python objects, and custom code is referenced by registered name). It awaits your confirmation in Q4b.
- Added an ordered *Next steps* section, WD.P (DataStructure prerequisites taken from B7), and the new questions Q4b, Q15 and Q16.

---

## 2026-09-29 (later): Step 1 done (W0.1, W0.3–W0.5, W4.6–W4.8)

- **Commits on `release2026`:**
  - `251636d`: W0.1, the `publish_WB` wrapper port.
  - `4bbc834`: the scene wrapper changes plus the CHANGELOG migration table.
- **Changes to `scene/scene_wrapper.py`:**
  - `play_Orchestra(debug_runs=False, poll_interval=10)` are new keyword arguments.
  - The soil and light workers now get dedicated cores: `plan_affinity(n_plants + n_env)`, still keeping one core free. They are assigned in the order plants, soil, light.
  - New `available_cpu_ids()` / `pin_to_cpus()`. Pinning is a no-op when `psutil.Process` has no `cpu_affinity` (macOS).
  - The light worker now receives `light_scenario` instead of `plant_scenarios[0]`. `meteo` is taken out of it through `load_meteo` (a csv path indexed by `t`, or a DataFrame), and a missing `meteo` raises `KeyError`. The caller's dict is not mutated.
- **`CHANGELOG.md`** (previously empty) now has the old → new import table and the wrapper behaviour changes.
- **`test/provide_usage_examples/`** now uses the new import paths.
  - `mtg_to_arraydict` has been replaced by `MPG.convert_properties_to_arraydict(g, g=g, ...)`, marked `TODO(WD)`.
  - `SoilModel` uses `Component as Model`, also marked `TODO(WD)`.
  - The scene example passes `light_scenario=dict(scenario, meteo=...)`.
  - The folder is still untracked.
- **Tests:** new file `test/wrappers_tests/test_scene_wrapper_step1.py`, with 8 tests:
  - pinning with and without `cpu_affinity`;
  - `plan_affinity` / `free_cpu` without affinity;
  - the light worker meteo from a DataFrame, from a csv, and when missing;
  - `play_Orchestra` wiring with `mp` faked (the core allocation covers plants, soil and light, `light_scenario` is forwarded, and `sowing_depth` is applied).
- **Results:** the full suite gives **320 passed**, 7 warnings (the same anderson warnings as before). A CI-style run (`cd test`, no pyproject) of `wrappers_tests` gives 8 passed. No `/dev/shm` segments are left behind.

---

## 2026-09-29 (later): Step 2 done (W1 infrastructure), commit `409a0e9`

- **Answers recorded in the plan:**
  - Q4b: you lean toward a Python translator, for live `scales.SubOrgan` references and free-form formulas. WD.0 is revised to be Python-first, with the YAML loader kept.
  - Q15: Logger scope reduced to the xarray and csv writers on DataStructures (WD.9).
  - Q16: the soil starts on a 3-D `ArrayDataStructure`. The axis order is still open (Q16b).
- **`Choregrapher.reset()`** clears the run state in place: `scheduled_groups`, `sub_time_step`, `data_structure` and `simulation_time_step`. It keeps the processes registered by the decorators, which are keyed by class *name* when the class is defined, and it keeps the singleton, since `Component.choregrapher` is bound when the class is created.
- **`test/wrappers_tests/`:**
  - `doubles.py` contains:
    - `RootCarbon` and `RootNitrogen` (legacy `Component` on MTG props);
    - a voxel `SoilModel`, which mirrors rhizosoil's `get_from_plant` / `send_to_plant` / `__call__`;
    - `FakePlant` (GrassBRIDGES shape) and `FakeSoil` (RhizoSoil shape);
    - `FakeLight` (the LightModel queue protocol, with PARa = PARi × leaf area);
    - `FakeLogger` (the Logger call surface, recorded to `calls.txt`);
    - a translator covering every link kind.
  - `conftest.py` resets the Choregrapher (autouse) and provides a translator file, a zeroed SharedMemory factory, a meteo table and a threaded `in_process_scene`.
  - 7 new sanity tests, including 2 in-process scene steps with and without light.
- **Hazards found while building this:**
  1. **The collection-time `deep_reload_package(["openalea"])` in the UC1 files duplicated class objects** when test directories were passed explicitly: pytest loads the initial conftests before collection. `wrappers_tests` then failed 4 tests on `isinstance` and singleton mismatches. The reloads were removed from all active tests (B3).
  2. With the reload gone, **`test_uc1_stepinit_and_graph_system_via_choregrapher` depended on global state.** A leftover `simulation_time_step = 3600` made the Choregrapher run 3600 sub-steps (80 s) and gave wrong values. The UC1 files and the Choregrapher tests now reset per test.
  3. **`from conftest import` broke** once a second `conftest.py` existed: `solver_tests` followed by `wrappers_tests` gave 6 errors. The builders moved to `test/solver_tests/solver_specs.py` (B4, partial).
  4. **The Choregrapher schedules per class *name*.** Same-named component classes in different test modules merge their processes, so the declaration-test dummies were renamed `Declared*`. Only one live instance per component class per process is possible (the latest instance wins), which is why the in-process scene is limited to one plant.
- **New suspected scene bugs (to verify in W5):**
  - **W5.0: likely final hang with a light model.** A plant with `n` iterations needs `n + 1` light replies, while `light_worker` runs `n` times. The soil avoids this with its init reply.
  - **W5.2a:** the plant SharedMemory is initialised from `np.empty`.
- **Results:** full suite **328 passed**, 7 warnings (2.6–3.2 s). The CI-style run from `test/` gives 328 passed. Directory orders checked, all passing:
  - `solver`, then `wrappers`;
  - `wrappers`, then `solver`;
  - `solver`, then `wrappers`, then `graph`;
  - `wrappers`, then `graph`, then `mpg`, then `data_api`;
  - `component_use`, then `graph`.

---

## 2026-09-29 (later): Step 3 done (W2–W4 contract tests, test-first fixes), commit `1f6a416`

- **Decisions:**
  - Q16b: the canonical soil axis order is `(x, y, z)`; the RhizoSoil `(y, z, x)` indexing gets converted in the Coupler.
  - Q17: the light model sends an initial reply like the soil does (applied in step 4).
- **New tests** (all in `test/wrappers_tests/`):
  - `test_composite_wrapper.py`, 24 tests:
    - translator file I/O;
    - the scripted interactive builder;
    - WheatBRIDGES soil inputs/outputs (10 outputs, 18 inputs, 25 plant-side rows; 25 + 10 = 35 = `HANDSHAKE_SHAPE[0]`);
    - one test per link kind, asserting the value the receiver sees;
    - input tables;
    - documentation.
  - `test_composite_contract.py`, 8 tests: the in-process plant/soil/light protocol (handshake, buffer rows, voxel mapping, the ×72 and ×5 conversions, containers updated in place, light round trip) and **the two-cycle regression anchor for WD**, with hand-derived values.
  - `test_scene_wrapper.py`, 20 tests: stand layout, CPU registry, and the plant and soil workers run in-process with `os._exit` patched.
- **Bugs confirmed by failing tests on the old code, then fixed:**
  - W2.5: a same-name link with a factor ≠ 1 was dropped. It is now applied across data structures, and within one it raises `ValueError`. The contract anchor fails on the old wrapper; this was checked by restoring it.
  - W2.6: a multi-source link with a subcategory kept only its last source.
  - W2.9: the stale input-table cache raised `IndexError` when targets were added.
  - W2.10: the documentation crashed with `None.__format__`, which also broke the interactive translator builder.
  - W4.2: the multi-model stand pick was wrong for more than 2 models, and a draw of 0 matched no model.
  - W4.4: a missing `outputs/` folder raised `FileNotFoundError`.
  - W4.5: the soil worker required a logger, and the plant worker needed one whenever `logging=True`.
- **Other changes:**
  - `recursive_reload` removed.
  - `CompositeModel.soil_handshake_inputs()` extracted.
  - New constants `scene_wrapper.HANDSHAKE_SHAPE` and `CPU_REGISTRY_FOLDER`.
  - The reference translator was copied, tracked, to `test/inputs/wheatbridges_coupling_translator.yaml`.
- **Results:**
  - Full suite **380 passed**, 7 warnings (about 3 s). The CI-style run gives 380 passed, and the mixed directory orders pass.
  - Line coverage (`sys.monitoring`): **`composite_wrapper` 96.5 %** (was 10.1 %), **`scene_wrapper` 92.6 %** (was 8.9 %), package 78.3 % (was 67.4 %).
  - The remaining `scene_wrapper` misses are the orchestrator poll and stop-file loop and the SharedMemory re-creation path, both in W5 scope.
- **Noted, not changed:** `play_Orchestra` always passes `plant_model_frequency=[1.]`, so only the first plant model is ever used, even though `plant_models` is a list.

---

## 2026-09-29 (later): Step 4 done (W5, `play_Orchestra` with real processes), commit `062c61c`

- **New file `test/wrappers_tests/test_scene_orchestration.py`** (slow tests), 3 tests × {fork, spawn, forkserver} = 9:
  - 2 `FakePlant` + `FakeSoil` + `FakeLight` + `FakeLogger` over 3 iterations. Checks run counts, logger calls, PARa at t=2, soil DOC and C_hexose_soil received, one pinned core per worker, and no leftover `/dev/shm` segments or CPU registry.
  - The stop file deleted mid-run gives `clean_exit` False.
  - A failing plant stops the scene.
  - A thread-free `SIGALRM` watchdog writes a report and kills only its own scene workers. The first watchdog (`faulthandler.dump_traceback_later`) started a thread, and forking with that thread alive was a likely cause of hangs in the test process.
- **Three scene protocol bugs found and fixed**, each shown failing first:
  1. **Truncated runs with a light model.** Every model ran `n_iterations - 1` steps while `clean_exit` stayed True; the plant logger showed 2 of 3 calls. The suspected hang was really this truncation. Q17 fix: the light model answers the plants' initialization in its constructor. **This breaks the downstream `LightModel` constructor** (B11).
  2. **Race in the end of scene.** An environment worker that finished set `stop_event`, and the other one could then skip its last step while the plants waited for it. Environment workers now stop the scene only on failure.
  3. **Lost last replies.** `soil_worker` ended with `os._exit(0)`, which kills the queue feeder threads. It was reproduced in 2 of 10 runs under full CPU load, and a per-worker `SIGUSR1` stack dump showed both plants blocked on the soil reply after the soil had completed all its runs. Fix: `flush_queues()` before `os._exit`. After the fix, **0 of 30 stress runs failed**.
- **Other changes:**
  - The plant buffers are zero-initialised, the numpy handle is released before `shm.close()`, and the `del b` in `finally` is removed.
  - CHANGELOG entries added for all behaviour changes.
- **Portability:**
  - fork is skipped on macOS.
  - The scene tests are skipped on Windows: `play_Orchestra` closes its creation handle before the plants open the block, and Windows destroys a shared memory block when its last handle closes (W5.6).
- **Process-safety note:** during diagnosis I used broad `pgrep`-style kills on the pattern "multiprocessing". Processes from the user's `wheat-bridges` env were running at the same time. A later listing showed them alive, but one earlier kill targeted an unknown pid that had already exited. From then on only explicit pids of my own test runs were killed, and the watchdog kills only its own children.
- **Results:**
  - Full suite **395 passed**, 7 warnings (about 5.5 s). CI-style run: 395 passed. Excluding slow tests (`-m "not slow"`): 386 passed in 1.4 s. The slow tests take about 4.2 s.
  - Coverage: `scene_wrapper` 94.5 % in-process (worker bodies run in child processes, which are not traced), `composite_wrapper` 96.5 %, package 78.5 %.
- **New questions:**
  - Q18: `clean_exit` is True even when a worker failed.
  - Q19: only `plant_models[0]` is ever used.

---

## 2026-09-29 (later): Q18 / Q19 implemented, commit `a3a9877`

- **Q18:** plant and soil workers exit with code 1 when their model raised, and `play_Orchestra` returns `clean_exit=False` when any worker has a non-zero exit code. The light worker already exited with code 1, because it re-raises.
- **Q19:** new argument `play_Orchestra(plant_model_frequency=None)`. It defaults to uniform frequencies (`[1.]` for one model). It is validated to have one frequency per model summing to 1, and `plant_scenarios` must match `plant_models`. A single `sowing_depth` is shared by all models.
- Each fix started from a failing test: 5 tests failed first. Results: full suite **398 passed**, and the slow scene tests passed 3 runs in a row.

---

## 2026-09-29 (later): Step 5, design note drafted (WD.0 + WD.1)

- **New file `docs/design/coupling_through_datastructures.md`**, for review. It was built from three read-only code surveys:
  1. how components and the solver consume props;
  2. DataStructure storage, scales and mapping;
  3. downstream coupling requirements from `test/provide_usage_examples/`.
- **Key survey findings recorded in the note:**
  - **Three copies of every variable drift apart:** the MTG props, the DS arrays and the `FunctionalComponent.props` snapshot. After construction, all solver and Functor writes go to the snapshot only, so the DS arrays go stale, and no test checks them.
  - **Already array-native:** the spec layer and `make_evaluator`. The hard part is the per-vid `Functor` branch.
  - **Blocking bugs B-a…B-i:**
    - `node_local_index` is wrong on MPG views (confirmed again);
    - the two incidence matrices have opposite signs;
    - the bio-scale mapping helpers have no scale argument, and their fast path matches on size only;
    - `write_*_to_mtg` swallow every exception;
    - `update_topology` clears every array;
    - there is no common accessor for graphs and grids;
    - `set_*` rebinds arrays;
    - grid axes have no names, and there is no locate/volume helper;
    - `LabelsConfig` mutates its class attributes;
    - `populate_graph` adds Compartments to the SubOrgan `children()`.
  - **The real WheatBRIDGES translator has 98 links:** 66 identity, 10 alias, 4 numeric factor, 18 string expression, 0 same-name factor, 0 multi-source. Recount checked.
  - **Plant-scale values live at "vertex 1".**
  - **`state_variable_type` is blank for 7 of the 16 soil inputs**, so the aggregation must be declared on the link or the coupler.
- **Proposal:**
  - `Translator` / `Link`: Python-first, with a YAML loader and a restricted parser replacing `eval`.
  - `DataStructure.get` / `set` / `register` with in-place writes, plus a version counter.
  - Name-level alias table; derived variables refreshed by the receiver.
  - Sparse scale operators (sum, mean, weighted mean, broadcast, proximal, distal) and a scalar store for the plant scale.
  - `Coupler` + `VoxelLocator` + `Transport` in `(x, y, z)`; growth preserves arrays, with a per-variable on-grow policy.
  - Live-reading Functor branch with a vectorisation contract; explicit per-solve copies in the solver.
  - DS export API for the Logger writers.
  - A gated migration sequence, whose final gate is that all `[contract]` tests pass unchanged.
- **Open decisions** Q20–Q26 are added to `devplan.md`, each with a recommendation. No WD code will be written before your review.

---

## 2026-09-29 (later): Design note accepted; WD.P done, commit `42e00eb`

- **Review:** you accepted the design note and the §11 recommendations, with precisions on Q24 and Q26:
  - Q24: the growth model may overwrite from parent states, for concentrations and extensive quantities.
  - Q26: the tests keep identity links written explicitly, even in Python translators.
  - Decisions Q20–Q26 are recorded in the devplan. New memory: write a design note before complex steps.
- **WD.P implemented, tests first.** 21 new tests; before the fixes, all but one of the original 20 failed.
  - **Variable store** (`VariableStoreMixin` on `MPGDataStructure` and `ArrayDataStructure`):
    - `get` / `set` / `register` / `location` / `has` / `alias` / `aliases` / `version`;
    - writes are in place with shape checks, and the legacy setters now write in place too;
    - aliases are name-level, with cycle and shadowing checks.
  - **Graph fixes:** `node_local_index` is correct on the MPG's unsorted ids, and `incidence_matrix` is aligned on the GraphView convention (+1 at the parent).
  - **MTG mapping:**
    - the mapping helpers take `scale=` (coarser scales via `complex_at_scale`), and `FunctionalComponent` passes the declared scale;
    - the fast path is key-checked;
    - partial coverage raises;
    - write-back creates missing properties and raises on shape errors.
  - **Growth:** `update_topology` carries registered variables over (nodes by vid, edges by child vid). New entities take the default or inherit from the parent (`on_grow`, declared on fields via `declare(..., on_grow=)`).
  - **Grid:** `ArrayDataStructure` gets axes `("x", "y", "z")`, `cell_centers`, `cell_volume` and `locate(periodic=, clip=)`.
  - **`LabelsConfig`:** integers live on per-instance copies, and `Connection.Apoplastic = "ApoplasticEdge"`.
- **Tests updated because they pinned old behaviour (3):**
  - a 4-value array on 3 nodes;
  - `update_topology` clearing the variables (2 tests);
  - plus the abstract incidence stub.
- **Results:**
  - Full suite **419 passed**, 7 warnings; the same in the CI-style run from `test/`, and in the mixed directory orders.
  - The UC1 physics suites pass with write-back errors no longer swallowed, so nothing was failing silently.
- **Noticed:** `test/components_coupling_tests/` (`test_wrapper.py`, `est_model_assembly.py`) was deleted in the working tree, not by me and not committed. Its only live test was the assertion-free `test_import_composite`, which `wrappers_tests` supersedes.
- **Next:** WD.3, meaning derived variables, scale operators and the scalar store.

---

## 2026-09-29 (later): WD.3 done (links on DataStructures)

- **Derived variables:** `derive` / `refresh` on both DataStructures. They take weighted sums or formulas, refresh in place so views stay valid, refresh dependencies first, reject cycles, check locations (and reject a location change without an aggregation), and resolve aliases.
- **New locations on `MPGDataStructure`:** the coarser biological scales (named after `ScalesConfig`, with entity ids from `complex_at_scale` of the node vids, so anchors are excluded and B-i does not bite), and `scalar` for plant-scale values (0-d arrays).
- **Operators:**
  - node → coarse: `sum`, `mean`, `weighted_mean` (via `bincount`);
  - coarse → node: `broadcast`;
  - node → edge: `proximal`, `distal`, `mean`;
  - node ↔ scalar.
- **`ArrayDataStructure`:** `cell` ↔ `scalar`.
- **Growth:** `update_topology` carries every location over.
- **Tests:** 12 new tests, 11 of them failing first. Full suite **431 passed**; CI-style run 431 passed.
- **Next:** WD.2, live reading in the solver path and the Functor DataStructure branch. It is the most invasive step, and the UC1 suites are its gate.

---

## 2026-09-29 (later): WD.2 done (live reading)

- **`FunctionalComponent`:**
  - binds its DataStructure to the Choregrapher, under compartment `"graph"`, so the legacy `"root"` slot is untouched;
  - `props` is now a read-only `DataStructurePropsView` built from the DS at each access;
  - write-back and bio-scale refresh use the DS arrays directly.
- **Solver path (`decorator.py`):**
  - `_snapshot`, the initial guesses, `amount_olds` and the first-tick implicit previous fields are **copies read from the DS at each solve**;
  - `inject_result` and `@graph_output` write in place, and register on first write (outputs are located by size);
  - the second `build()` is removed (`builder.last_spec`);
  - `_type_mask` uses `np.isin`.
- **Functor:** a new DS branch, where one vectorised call writes its outputs in place. Every step decorator accepts `vectorized=False` for one call per element; supplementary outputs are supported, and scalar outputs go to the `scalar` store.
- **`self.previous(fn)`:** unknowns at the start of the current solve, managed by the framework (Q21).
- **Tests:** 7 new tests, all failing first. They show among other things that a parameter changed on the DS after construction is used by the next solve, which the snapshot prevented before.
- **Results:** full suite **438 passed**; CI-style run 438 passed; mixed directory orders pass. The UC1 suites and the wrapper `[contract]` tests are unchanged.
- **Follow-ups recorded in the devplan:**
  - the stale `_graph_view` after `update_topology` (needs a topology version);
  - moving `pull_available_inputs` / `apply_input_tables` to WD.4;
  - migrating the UC1 equations to `self.previous()`.

---

## 2026-09-29 (later): WD.0 implemented (translator objects)

- **New `coupling/translator.py`:**
  - `parse_factor`: restricted AST arithmetic over numbers and `+ - * / **`; it rejects names, calls, subscripts and conditionals.
  - `Link`: a frozen object carrying `receiver`, `variable`, `provider`, `sources` (`{name: factor}`, or names when a `formula` is given), and optionally `scale` / `source_scale` (ScalesConfig integers or names), `aggregation` and `weight`. Its `kind` is identity / alias / derived, and `detail` refines it (identity / alias / factor / expression / multi_source / same_name_factor / formula / scale_change).
  - `Translator`: a chained `link()` builder, plus `links_of`, `components`, `inputs_outputs` (the same semantics as `get_component_inputs_outputs`), `to_nested`, `from_yaml` / `from_dict` (short and long YAML forms), `from_module` and `load`.
- **`CompositeModel`:** string factors go through `parse_factor` instead of `eval`, and `open_or_create_translator` accepts `.py` translator modules.
- **Gate met:** the WheatBRIDGES YAML gives **98 links, split 66 identity / 10 alias / 4 factor / 18 expression**. It round-trips to the nested format with the same numeric factors, and gives the same soil inputs/outputs as the legacy method in both modes.
- **Results:** 24 new tests (22 + 2), all failing first. Full suite **462 passed**; CI-style run 462 passed.
- **Next:** WD.4, `CompositeModel` on Translator and DS links.

---

## 2026-09-29 (later): WD.4 done (CompositeModel on DataStructure links)

- **DataStructure-backed coupling:** when every component is DataStructure-backed, `couple_components` loads the translator into `Link` objects:
  - identity links need nothing;
  - an alias replaces the receiver's own auto-declared default (new `DataStructure.unregister`) with `ds.alias`;
  - factor, multi-source, formula and scale links become `ds.derive(...)`, recorded in the receiver's `_derived_inputs` and refreshed by `FunctionalComponent.pull_available_inputs` at the start of each call;
  - a same-name factor within one DS is rejected.
- **Soil side:** soil outputs are registered on the plant DS and set to 0, as the props path did. Links with components outside the composite (the soil) are left to the scene.
- **Input tables:** `apply_input_tables` writes DS variables, applying the table value to the whole variable.
- **New doubles:** `test/wrappers_tests/doubles_ds.py`, with `PlantCarbon` and `PlantNitrogen` (the same equations as the legacy doubles) as `FunctionalComponent`s on the seedling MPG, plus a renamed translator.
- **Tests:** 7 new tests. The DS coupling reproduces **exactly the props-based contract numbers**: nitrogen_status 4, hexose 0.904, carbon_supply 0.2, amino_acids 2.378.
- **Results:** full suite **469 passed**; CI-style run 469 passed.
- **Next:** WD.5, the plant↔soil Coupler and Transport.

---

## 2026-09-29 (later): WD.5a done (Coupler); checkpoint before the wire-protocol change

- **New `coupling/coupler.py`:**
  - `VoxelLocator`: segment barycentres from the plant variables x1..z2 (names configurable), `flip_z`, periodic x and y, depth clipped, flat `(x, y, z)` cell indices through `ArrayDataStructure.locate`.
  - `Coupler`: `update_map`, `zero_soil_inputs`, `push` (`np.add.at` scatter-add of Σ factor·plant variable, several plants into one soil) and `pull` (in-place gather). It also has `from_translator`, and `plant_variables` / `soil_variables` for the transport plan.
  - A stale map (topology changed since `update_map`) raises through the new `DataStructure.topology_version`.
- **Tests:** 9 new tests, including **equality with the reference soil model's `apply_to_voxel_fast` sums after the `(y, z, x)` → `(x, y, z)` permutation**, and growth.
- **Results:** full suite **478 passed**; CI-style run 478 passed.
- **Checkpoint.** The next step (WD.5b) changes the plant↔soil wire protocol that the downstream GrassBRIDGES and RhizoSoil code implements. Questions Q27 (buffer sizing and migration) and Q28 (when to remove the legacy props path) are added to the devplan, each with a recommendation.
- **Commits in this implementation run:**
  - `df50f2b`: design note precisions;
  - `42e00eb`: WD.P;
  - `b671415`: WD.3;
  - `d7546f3`: WD.2;
  - `d69d9d8`: WD.0 implementation;
  - `ee7bbc4`: WD.4;
  - `26433fc`: WD.5a.

---

## 2026-09-29 (later): Q27/Q28 decided; WD.5b and WD.6 done

- **Decisions:**
  - Q27, option (a): `play_Orchestra(handshake_shape=...)`, defaulting to the legacy `(35, 20000)`.
  - Q28: keep the legacy props path, deprecated, until the end of WD.7.
  - `test/components_coupling_tests/` removed as outdated, at your request (`892534c`).
- **WD.5b (`804d082`):**
  - `Transport`: buffer rows from the translator (`_n_nodes`, `_node_id`, coordinates, the plant variables read by the soil, the soil states), with a capacity check;
  - `BufferPlantView`, so the soil runs the same Coupler on the plant's buffer;
  - `play_Orchestra(handshake_shape=)`;
  - `FunctionalComponent` on grids.
  - The WheatBRIDGES transport needs 36 rows: 2 headers, 6 coordinates, 18 plant variables and 10 soil states.
- **Fixed while building the DS scene:**
  - `FunctionalComponent` registered a sub time step of 1, so with a 3600 s simulation step it would run 3600 times per call. It now runs once per simulation step.
  - The soil name is now `CompositeModel.soil_name`.
- **WD.6, the DataStructure scene doubles:**
  - `DSFakePlant` puts PlantCarbon and PlantNitrogen on a 3-segment MPG with the legacy geometry, and exchanges through a Transport.
  - `DSFakeSoil` runs `GridSoil` (a FunctionalComponent) on an `(x, y, z)` grid, with one Coupler per plant view.
  - **The in-process DS scene reproduces the props-based two-cycle anchor exactly:** DOC 33.8 / 16.9 (now at `(0, 0, 0)` / `(0, 0, 1)` in `(x, y, z)`), and all the plant values.
  - `play_Orchestra` runs end to end with the DS doubles under fork, spawn and forkserver: **0 failures in 10 runs under full CPU load**.
- **Test hygiene:** a capacity test left a daemon thread blocked on a queue, and later fork-based tests warned about forking a multi-threaded process (10 extra warnings). It was rewritten so that no thread is started; back to the 7 known warnings.
- **Results:** full suite **496 passed**; CI-style run 496 passed.

---

## 2026-09-29 (later): WD.8, WD.9 and the WD.7 guide done; state of the refactor

- **`3c85536`:**
  - DataStructure `export` / `to_dataframe` / `summarize` for the Logger writers;
  - `openalea.metafspm.testing.assert_component_couplable`;
  - `FunctionalComponent._graph_view` rebuilt when the topology changes, so a solve after growth converges on the grown plant.
- **New guide:** `docs/design/downstream_migration.md` (WD.7), covering imports, components, the plant composite with Transport, the soil on an `(x, y, z)` grid with Coupler, the light model's init reply, the Logger writers, and a checklist. Each step points to a tested DS double.
- **State:**
  - Everything planned inside metafspm for Epic W and Epic WD is implemented and tested. The full suite passes **507 tests**, including 12 slow real-process tests under fork, spawn and forkserver.
  - What remains is the migration of the downstream packages (rootbridges, rootcynaps, cnwgrass, rhizosoil, grassbridges, and the `openalea.fspm` Logger), then the removal of the legacy props path in one commit (Q28).
- **Open follow-ups in the devplan:**
  - migrate the UC1 test equations from `_previous_fields` to `self.previous()`;
  - backlog B1, B2, B4 (importlib mode), B5, B6, B8–B11.

---

## 2026-09-29 (later): The three last steps — UC migration, legacy removal, backlog

- **Scope rule (from you, saved to memory):** only this repo is edited. The downstream `Wheat-BRIDGES_framework` tree was inspected read-only (branches, status, sizes) and is unchanged; "migration" means this repo's use-case tests.
- **Step 1, UC migration (`8f6f055`):**
  - UC2 (Münch water transport) and UC4 (Dirichlet / Neumann) now run on a 3-segment `MPGDataStructure` chain.
  - UC3 (Robin ports, typed conductances, `@graph_output`, analytic Jacobian) runs on the seedling root graph. The cross-sectional anatomy helper no longer builds with the MPG API and remains to be ported.
  - 9 tests are re-enabled; the suite has **no skipped test**.
- **Step 2, legacy props path removed (`b3c11c8`):** see the CHANGELOG section "Breaking: legacy props path removed".
  - Found along the way: `translator_matrix_builder` crashed on any `FunctionalComponent`; fixed.
  - The legacy doubles and contract tests are removed; the DS doubles reproduce the same numbers.
  - The reference RhizoSoil voxel math is kept as a standalone reference in `test_coupler.py`.
- **Step 3, follow-ups and backlog:**
  - `89cc443`: the UC1 equations use `self.previous()`. The previous state is captured at build time, so direct builder + solver use works too.
  - `b423d00`: MMS tests re-enabled (B2); explicit module skips instead of the `est_` prefix (B10).
  - `b4b9ffd`: coverage configuration; `nbmake` dropped (B1).
  - `09eb138`, solvers (B5, B6):
    - the first tests of the multi-step `solve()` loops, on 2-node diffusion with an analytic solution;
    - edge unknowns are now recovered with the nodes fixed. Explicit Euler and IVP used a whole-system quasi-static Newton, which gave wrong fluxes, or a singular matrix;
    - IVP returns the full state, and its fake midpoint error estimate (which collapsed the step size) is gone;
    - solvers start from `x`;
    - `make_solver` accepts dict configs.
  - `39978cd`: the UC1 SubOrgan boundary conditions now sit on the real root (they were on a leaf tip), and the scheduling test asserts the actual step order (B8).
  - `57c5e04`: root `test/conftest.py`; importlib import mode supported (B4).
  - `b7f95ad`: `from_legacy` maps values by vid (B7); `test/utils.py` removed (B3).
- **Results:**
  - **516 passed, 0 skipped**, 10 warnings (anderson `LinAlgWarning`s), in about 4.6 s. The same holds in the CI-style run from `test/` and with `--import-mode=importlib`.
  - The 9 slow real-process tests pass repeatedly, with no leftover shared memory.
  - **Line coverage 84.1 %** (67.4 % at the first audit). `composite_wrapper` 94.7 %, `scene_wrapper` 94.5 %, `coupler` 95.2 %, `translator` 95.8 %, `decorator` 90.0 %.
- **Open:** Q29, what to do with the now-unused `solve/specializer.py` (0 % coverage).

---

## 2026-09-29 (later): DataStructure API audit (`devplan_datastructures.md`)

- **Your questions Q1a, Q1b, Q2, Q3 are answered from the code in `devplan_datastructures.md`.**
  - **Q1a:** graph systems effectively work only on `MPGDataStructure`, the only DS with both a graph view and a variable store. On `ArrayDataStructure` the graph view is `None`.
  - **Q1b:** `MPGDataStructure` uses no traversal order. Its local order is the Compartment post-order produced by population, and the MPG traversals act on MTG properties only.
  - **Q2:** there is one node scale per DS (the `from_scale`), so graph views are not built per variable scale.
    - The cycle is: declare/refresh (MTG → DS), then snapshot copies at each solve, then solve, then in-place write, then write-back of bio-scale state variables.
    - Found: **write-back ignores the declared scale.** Coarser-scale state variables are written at the node vids, without aggregation.
    - `@rate` results reach the MTG only when the component also ran a graph system.
    - Output locations are inferred by size.
  - **Q3:** no extra environment nodes are needed. Dirichlet/Neumann BCs via filters exist; Robin boundary ports act as virtual environment nodes. Their values are frozen per port, they are keyed by vid, they do not follow growth, and the models have to assemble them by hand.
- **Plan:** 16 DS-items (DS1–DS16), decisions D1–D8 with recommendations, questions Q4–Q5, and a suggested order: the Location contract first, then traversal and boundary sets.

---

## 2026-10-01: DataStructure plan refined from your answers

- **Decisions recorded** in `devplan_datastructures.md` §6:
  - D1, D2 and D5–D8 are agreed.
  - D3 is agreed, on condition that links stay resolved dynamically by name.
  - D4 is agreed, with your precisions.
  - Two new decisions are proposed, D9 and D10.
- **Your D3/D4 questions are answered with code evidence** (§8, "Replies"):
  - **D3:** name-linking holds on the DataStructure.
    - Identity links: components share variables by name.
    - Aliases: resolved at every `get`.
    - Factor, sum and scale links: recomputed before the receiver's step, the same timing as the former props coupling.
    - The new **DS17** makes derived variables exact at read too: lazy, tracked by write counters (D10).
  - **D4.1:** "one topology per component" ≠ one DataStructure per component. Components still share the plant's DataStructure.
  - **D4.2:** a cross-scale graph (per-segment anatomy plus links between neighbouring segments, matched by e.g. vessel index) can already be built in the MPG with `populate_graph_custom_connections`. `MPGDataStructure` cannot wrap it, because it is keyed on SubOrgan `vertex_id`.
    - **DS8 is rewritten** as a Compartment/Connection mode of `MPGDataStructure`: Compartment nodes, each mapped to its owner at every scale, and connection rules re-run on growth. Its priority is raised.
  - **D4.3:** no, cross-scale inputs are not populated at their own scale today.
    - Coarse-scale fields are stored at node location, as broadcast copies.
    - Nothing infers the mapping, and there is no extensive `split`.
    - Added **DS18**: cross-scale links with sum/mean up and broadcast/`split` down. The default mapping comes from `state_variable_type` (D9) and raises when the type is missing.
- **Other plan items updated:**
  - **DS3** now places coarse fields at their scale's location.
  - **DS7** is reduced: the MPG ↔ MPG Coupler is deferred.
  - **DS10** gains per-component or per-group sub-stepping and `previous(fn, at="step")`.
- **UC5 leaf transpiration is specified from Q4** as a tool use case:
  - a Robin boundary on the leaves, driven by per-leaf microclimate inputs;
  - conductance = stomatal conductance (a model variable) × exchange surface.
- **New questions:**
  - Q6: the intra-segment anatomy edges, and the anatomy/link order at growth;
  - Q7: whether a split targets labelled Compartments or all of them;
  - Q8: whether sub-stepping is per component or per group.
- **Order (§7):**
  1. the contract (DS3, DS17, DS5, DS11, DS16);
  2. the multiscale topology and boundaries (DS8, DS2, DS6 with UC5);
  3. cross-scale coupling and grids (DS18, DS1);
  4. time and data;
  5. runtime.
- **Rule saved for commits:** no Claude signature or mention in commit messages. Earlier commits up to `b7f95ad` still carry `Co-Authored-By` trailers; they are rewritten only if you ask.
- No code changes, nothing committed.

---

## 2026-10-01 (later): DS8 / DS18 / DS10 refined from Q6–Q8

- **Q6, your multiscale model:**
  - anatomies are held at Compartment/Connection, below SubOrgan;
  - connectivity between segments is held at SubOrgan and coarser scales;
  - the solved graph is assembled by the traversal, with modeller-declared wiring rules (vessel index, angular coordinates).
- **DS8 is rewritten as an assembled graph view on `MPGDataStructure`:**
  - Compartment nodes, the anatomy Connections, and junction edges between adjacent SubOrgans, generated by declarative or callable rules;
  - junctions are not written to the MTG: proposed **D11**, asked as Q9;
  - owners at every scale, so that aggregation and broadcast work;
  - rules re-applied on growth;
  - `populate_graph_custom_connections` stays for existing code, as one declarative rule.
- **Q7, DS18:** the down mapping is a filtered `broadcast` of intensive quantities (e.g. to symplastic Compartments). `split` is dropped from the defaults: extensive going down raises (D9 revised).
- **Q8, DS10 / D5:** sub-stepping is per component only.
- **New questions:**
  - Q9: whether junctions should stay out of the MTG;
  - Q10: how the anatomy generators store the intra-section edges;
  - Q11: whether junctions only join parent/child SubOrgans.
- No code changes, nothing committed.

---

## 2026-10-01 (later): DS8 aligned on "the MPG is the source of truth" (Q9–Q11)

- **Q9:** the DataStructure is a dynamic interface to the MPG, and its graph view is extracted from the populated MPG. **D11 is reversed** and now agreed:
  - every edge, junctions between anatomies included, is an MPG Connection vertex, populated by MPG methods;
  - this matches what `MPGDataStructure` already does for one Compartment per segment.
- **Q10:** GRANAP is out of scope until it becomes a StructuralComponent. DS8 is validated on a synthetic anatomy in the test helpers.
- **Q11:** junctions only join direct MPG links (within-scale parent, or the complex parent's tip).
- **DS8 is rewritten:**
  - node keys are Compartment vids in anatomy mode, with the SubOrgan as owner;
  - wiring rules are MPG methods (`all` / `nearest` / `equal`, or a callable), and their Connections are marked `edge_kind`;
  - growth keeps the anatomies and adds junctions incrementally (proposed **D12**). `repopulate_graph` would otherwise delete them;
  - edge values carry over by endpoint pair.
- **New questions:**
  - Q12: copies with explicit sync versus zero-copy views for values. Zero-copy is impossible with the current mixed-scale ArrayDicts;
  - Q13: the anatomy-before-junctions order at growth, and whether existing anatomies can change.
- No code changes, nothing committed.

---

## 2026-10-01 (later): Q12–Q13 → D12/D13 agreed, new DS19 (StructuralComponent contract)

- **Q12:** option a is kept. Values are copied with explicit sync, and the MPG storage is unchanged.
  - Your follow-up showed that the contract is missing for structural components: `StructuralComponent` is an empty class today.
  - Added **DS19**: it receives the shared DataStructure and edits the MPG through `ds.mtg` with the MPG's own methods (they are not re-exposed on the DS).
  - The framework flushes the component's inputs to the MPG before its step, and calls `update_topology()` after it. The MPG is therefore exact whenever it is edited.
- **Q13:** D12 is agreed with **conditional rewiring**. A SubOrgan's junctions are rebuilt only when its neighbourhood or its anatomy (Compartments, rule labels) changed. `update_topology` detects both automatically.
- **Order:** DS19 is placed before DS8 in step 2.
- **New questions:**
  - Q14: whether structural outputs are written as MPG properties or as DS variables;
  - Q15: whether structural processes are scheduled as decorated steps or as one call per component.
- No code changes, nothing committed.

---

## 2026-10-01 (later): Q14–Q15 → DS19 rewritten from rhizodep's RootGrowthModel

- **Q14:** structural outputs are MPG properties. **Q15:** structural steps are scheduled like functional ones.
- **Read (not edited):** `Wheat-BRIDGES_framework/rhizodep/src/openalea/rhizodep/root_growth.py`.
  - Its steps are `@potential` / `@actual` / `@segmentation` `@state` whole-MTG methods with no array arguments, using node-level MTG edits.
  - A `post_growth_updating` pass runs in `__call__`.
- **DS19 now specifies:**
  - decorated structural steps, interleaved with functional ones in the Choregrapher rows;
  - called without arrays, and editing `ds.mtg`;
  - declared inputs flushed to the MPG before each step;
  - declared outputs re-read after it, with `update_topology()` only when the topology changed (cheap detection);
  - the post-processing pass becomes a scheduled step;
  - validation on an in-repo rhizodep-like growth helper.
- **D13 is agreed.** D14 is proposed: at segmentation, the framework splits extensive functional variables between the origin and its new descendants by a weight, and copies intensive ones. The structural component can override for the variables it writes.
  - This is where `split` is needed after all: in carry-over at topology changes, not in coupling (Q7).
- **New questions:**
  - Q16: D14, the default weight, and primordia;
  - Q17: whether structural steps also get array arguments.
- No code changes, nothing committed.

---

## 2026-10-01 (later): Q16–Q17 → new DS20 (repartition when structure changes)

- **Read (not edited):** rhizodep `post_growth_updating`, `segmentation_and_primordium_formation`, `primordium_formation`, `ADDING_A_CHILD`.
- **Q16:** **DS20** generalises `post_growth_updating`'s rules by `state_variable_type`:
  - massic concentrations: the amount is split, or diluted on growth;
  - extensive: split;
  - non-inertial intensive: copied;
  - non-inertial extensive: scaled copy;
  - descriptors: none.

  The framework applies them after each structural step, to every registered variable of the shared DataStructure, which replaces `modules_to_update`. The weight, previous weight and active set are declared by the structural component; no structural mass is built in.
  - **On laterals:** a primordium is not split at formation (zero mass: it copies concentrations, with zero amounts). It is split from its parent at its first elongation.
- **Q17:** MPG-style steps for traversal and edits, and array-argument steps for vectorisable parts (outputs flushed to the MPG). Both are in DS19.
- **D14:** agreed in principle.
- **New question:** Q18, the declaration form, and whether the active set becomes a DataStructure-level mask shared with FunctionalComponents.
- No code changes, nothing committed.

---

## 2026-10-01 (later): Q18 → D14 agreed, D15 (active-set mask)

- **Q18:** the repartition is declared by class attributes (`partition_weight`, `previous_weight`, `active`), each accepting a callable.
- **The active set is a named DataStructure mask** (`define_mask` / `mask`), re-evaluated at refresh and after each structural step:
  - vectorised steps compute on active entities by default, with values outside left unchanged; `where=None` opts out;
  - MPG-style structural steps always see every vertex;
  - it replaces rhizodep's `focus_elements` and the Choregrapher `filter`.
- **D14** is agreed; **D15** is added.
- **New question:** Q19, whether graph systems solve on the active subgraph (opt-in) or on the whole graph with filters (the default).
- No code changes, nothing committed.

---

## 2026-10-01 (later): Q19 → D16 and DS21 (graph systems on the active subgraph)

- **Q19:** `@graph_system(where="active")` is opt-in, and the whole graph stays the default. **DS21** specifies the robust handling:
  - views cached by `(topology_version, mask_version)`, with the mask recomputed when its sources change;
  - the subgraph holds active nodes and the edges with both ends active, with boundary conditions restricted to it;
  - inactive nodes are frozen, and dropped edges have zero flux;
  - activation semantics, including `previous()`;
  - a well-posedness check per connected component (capacity or Dirichlet), raising with ids;
  - empty-mask skip;
  - anatomy-mode broadcast;
  - six validation tests.
- **The plan has no open question left.** Only D9 and D10 are still *proposed*. The next step is the §7 step 1 design note (DS3, DS17, DS5, DS11, DS16) once they are confirmed.
- No code changes, nothing committed.
- **Follow-up:** D10 is agreed. D9 is detailed in the plan (§8 "D9 in detail"): defaults table by source type and direction, worked examples, a type-agreement check, and a choice between A (defaults) and B (always explicit).

---

## 2026-10-01 (later): D9 = A; step 1 design note drafted

- **D9:** option A, defaults from `state_variable_type`. **D10:** agreed.
- **New design note `docs/design/datastructure_contract.md`** for §7 step 1 (DS3, DS17, DS5, DS11, DS16), written against `1bf8356` after a code survey:
  - **Declarations:** one interpreter, `resolve_declaration` → `VariableSpec`, with `scale` / `location` / `mapping` keys, the D9 defaults, and the legacy forms mapped. It replaces the duplicated logic of `_auto_declare_on_ds` and `_declared_locations`.
  - **Scales:** coarse-scale fields are stored at their own scale.
  - **Write-back:** the inverse mapping at the vids of the declared scale, after every component call (today it happens only after a graph solve, at the node vids).
  - **Derived variables:** lazy, through per-variable write counters checked at `get`, and read-only.
  - **Outputs:** locations from declarations, or explicit; size guessing only when unambiguous, with a warning. Filtered slicing by location.
  - **Validation:** `validate()`; missing variables and filters raise instead of giving zeros; edge BCs rejected until DS6; `from_scale` inferred rather than required.
  - **Conventions:** a page at `docs/conventions.md`.
- **Points to agree (N1–N5):**
  - N1: the coarse-scale default changes without deprecation;
  - N2: `proximal`/`distal` → `child`/`parent`;
  - N3: derived variables read-only;
  - N4: write-back after every call, now;
  - N5: scale names as location synonyms.
- **Implementation plan:** sub-steps 1a–1f, each with its test file.
- No code changes, nothing committed.
- **2026-10-02, review of the step 1 note:**
  - N1, N2 and N4 are agreed.
  - N3 is agreed. The note now defines derived variables: inputs filled by translator links with a factor, sum, formula or scale change. Identities and aliases are not derived variables.
  - N5 is re-explained in the note: the node scale is called `"node"`, which changes meaning when anatomies make Compartments the nodes, so scale names are proposed as stable location names. It is still open.

---

## 2026-10-02: N5 agreed; step 1a implemented (declaration resolver)

- **N5** is agreed, with your condition: `node` / `edge` are the entities of the graph built by the MPG traversal, and biological-scale declarations are resolved against them. It is recorded in the design note.
- **New `coupling/declaration.py`:**
  - `VariableSpec` (location, MTG scale, mapping, weight, kind, variable_type, default, on_grow);
  - `resolve_declaration` / `declared_specs`, one interpreter for all forms:
    - the legacy `scale="node"`… forms, `edge_mapping`, and Compartment/Connection;
    - N1: coarse scales stored at their own location;
    - N5: scale names as locations;
    - N2: `child` / `parent`, with deprecation warnings for the old names;
    - D9: default mappings by `state_variable_type`;
  - `DeclarationError` with the class and field named.
- `declare` and its wrappers gain `location=`, `mapping=` and `weight=`.
- **`FunctionalComponent._auto_declare_on_ds` is rewritten on the specs:**
  - it registers at the resolved location, from the MTG property mapped from its scale (coarse own-scale values, broadcast down, aggregated up), or from the default;
  - it records the spec in the DS metadata, which `register` now preserves across growth;
  - it rejects a pre-registered variable at another location;
  - scalar fields are now registered on graph DataStructures too.
- The solver's `_declared_locations` uses the same specs. `_snapshot` broadcasts scalars as before, and raises with a hint for coarse-located variables.
- **Tests:**
  - in-repo UC1 declarations are migrated to `location="edge", mapping="child"`;
  - new `test/data_api_tests/test_declarations.py`: 29 tests on the legacy forms, N1, N2, N5, the D9 defaults, 14 declaration errors, registration from the MTG, metadata surviving growth, the grid case and the snapshot hint.
- **Suite:** 545 passed (516 before, plus 29), same 10 warnings.
- **Not yet:** write-back of coarse-located state variables (1b), until then they are not written to the MTG. CHANGELOG is updated.
- Nothing committed.

---

## 2026-10-02 (later): 1a committed (`4d1353d`); step 1b implemented (MTG write-back through the mapping)

- **`MPGDataStructure.read_mtg(spec)` / `write_mtg(spec)`:** reading and writing a declared variable at the vertices of its scale, through its mapping and its inverse.
  - **Own scale:** as is.
  - **Broadcast down:** written back as the (weighted) mean of the nodes.
  - **Mean up:** written back by broadcast.
  - **Edges:** at the child or parent endpoint. A `parent` write raises where several edges share a parent.

  The edge readers, writers and `_map` accept `child` / `parent`.
- **`FunctionalComponent`:**
  - registration and the pre-solve parameter refresh use `read_mtg`, so the refresh now also covers coarse-located parameters;
  - `write_back_to_mtg` writes every state variable with a scale through `write_mtg`;
  - the `_bio_scale_*_fields` bookkeeping is removed.
- **N4:** `Component.__call__` writes back after every call, and the write-back at the end of a graph solve is removed. No existing test depended on it.
- **Resolver:** state variables that could not be written back are rejected (summed to a coarser scale; edge state at a scale coarser than the nodes). The 1a tests using such declarations now use parameters.
- **Tests:** new `test/data_api_tests/test_scale_mapping.py`, 9 tests:
  - Organ pool written at Organ vids only;
  - `@rate`-only component reaching the MTG;
  - broadcast written back as the node mean;
  - mean written back by broadcast;
  - child edge write;
  - parent ambiguity raising;
  - parameters refreshed and never written back;
  - `child` / `parent` operators.

  Plus 2 resolver error cases.
- **Suite:** 555 passed, the 9 slow scene tests included, same 10 warnings. CHANGELOG, plan progress and design note status are updated.
- Not committed yet.

---

## 2026-10-02 (later): 1b committed (`15f8976`); step 1c implemented (derived variables resolved at read)

- **`VariableStoreMixin`:**
  - per-variable write counters (`write_count`, `mark_written`), bumped by `register`, `set` (through `_write`) and re-registration at growth;
  - each derived spec stores the source stamps of its last computation;
  - `get()` on a derived variable recomputes the stale ones in dependency order, in place (`_update_derived`);
  - `refresh()` forces a recomputation; `is_stale()` is new;
  - `set()` and the legacy graph setters raise on derived variables (N3);
  - internal recomputation reads sources raw, so it never recurses.
- **`pull_available_inputs`** now only reads its derived inputs (a no-op when fresh).
- **Coupler `push`:** writes through `set` instead of `np.add.at` on a view, so derived variables see soil updates.
- **One contract expectation changed, with care.** `test_ds_scene_contract.py::test_two_cycles_regression_anchor` asserted `nitrogen_status == 4.103` after the run. That was the value PlantCarbon last refreshed, before PlantNitrogen's later update of `amino_acids`. Resolved at read (D10), it is now the current `amino_acids + 0.5·nitrate`.
  - The assertion now checks that relation, with a comment.
  - Every computed value of the anchor is unchanged (verified with the line skipped), including `hexose`, which consumes `nitrogen_status = 4.103`.
- **Tests:** new `test/data_api_tests/test_lazy_derived.py`, 10 tests:
  - stale then fresh, with the same view;
  - no recomputation without writes (call counts);
  - refresh forcing;
  - chains;
  - aliases both ways;
  - read-only, including the graph setters;
  - writes through views needing `mark_written`;
  - counters;
  - export reading the current value.
- **Benchmark (20 000 entries):** `get` of a plain variable 0.3 µs, of a fresh derived variable 2.2 µs, of a 10-deep fresh chain 8.8 µs; stale two-source recomputation 23 µs. Recorded in the design note.
- **Docs:** CHANGELOG, design note §4, and `downstream_migration.md` (derived variables now "recomputed when read, read-only").
- **Suite:** 565 passed, same 10 warnings. Not committed yet.

---

## 2026-10-02 (later): 1c committed (`cc3e588`); step 1d implemented (output locations)

- **Graph outputs:**
  - `@graph_output(name, location=)`;
  - the builder records the given locations;
  - an undeclared output uses that location, or else `infer_output_location`: a warning when exactly one location matches, an error when the shape is ambiguous (n == m) or unmatched;
  - a mismatch with an already registered variable raises.
- **Step decorators** accept `location=` and `locations={...}`. The Functor registers undeclared outputs at that location, resolving scale names (e.g. `"Organ"`), or as scalars for total steps and 0-d values, or else by unambiguous inference among node / edge / cell. Coarse locations are never inferred.
- **Filtered evaluators and BCs** slice an argument according to its entity: node or edge unknown, or node or edge snapshot, instead of `shape[0] == size`.
  - On trees (m = n − 1) this does not change any result: the old guess only went wrong for n == m.
  - It cannot be exercised by a solve before graphs with cycles exist (DS8). The ambiguity itself is tested on `infer_output_location`.
- **In-repo hooks** now give their location: the UC1 `axial_divergence` at node, the UC3 `edge_water_flux` at edge.
- **Tests:** new `test/graph_system_tests/test_output_locations.py`, 5 tests:
  - inference (unique, n == m ambiguous, unmatched);
  - the `graph_output` location check;
  - step outputs at edge, Organ (by scale name), inferred node and total scalar;
  - declared outputs without warnings;
  - grid inference and an unmatched shape raising.
- **Suite:** 570 passed, same 10 warnings. CHANGELOG, plan and note updated. Not committed yet.

---

## 2026-10-02 (later): 1d committed (`bf008e3`); step 1e implemented (validation, no silent zeros)

- **`VariableStoreMixin.validate_variables(strict=False)`** backs `MPGDataStructure.validate` and the new `ArrayDataStructure.validate`. It reports, all at once:
  - shape mismatches with the location;
  - dangling or cyclic aliases;
  - derived variables with missing or moved sources, or at the wrong location.
- **`strict=True`** recomputes each up-to-date derived variable (`_derived_values`, split out of `_compute_derived`) and compares it with the stored values. It detects writes through views without any checksum.
- **Where validation runs:**
  - at the end of `declare_data_and_couple_components`, once per distinct DataStructure;
  - in `couplability_problems` / `assert_component_couplable` when given `data_structure=` (plus declarations that do not resolve).
- **No silent zeros:**
  - `_read_array` raises `KeyError`, naming the component and the registered variables;
  - `_type_mask` raises on a missing filter variable;
  - `{field}_amount` is registered at 0 before the first integrated solve.
- **Boundary conditions:** `@boundary_condition("edge")` raises `NotImplementedError`, and other non-node locations raise `ValueError`.
- **`MPGDataStructure`** infers `from_scale` from the populated graph.
  - It cannot be built on an unpopulated graph anyway: `vertex_id` does not exist yet, which predates 1e.
  - So the `update_topology` error is a safeguard, and its test now clears `from_scale` by hand.
- **Tests that relied on the silent paths, made explicit:**
  - two UC1 tests (`test_uc1_stepinit_and_graph_system_via_choregrapher`, `test_rate_output_lands_in_the_data_structure`) used the `is_root` filter without registering it, so the Dirichlet condition was silently inactive. They now register `is_root = 0` everywhere, with the same physics and the same expected values;
  - the `from_scale` test is adjusted as above, plus a new inference test.
- **Tests:** new `test/data_api_tests/test_validation.py`, 7 tests:
  - clean validation, normal and strict, on MPG and grid;
  - every inconsistency reported in one error;
  - strict detection of a write through a view, and `mark_written`;
  - missing graph variable;
  - missing filter;
  - edge BC rejection;
  - couplability with a DataStructure.
- **Suite:** 578 passed, same 10 warnings. CHANGELOG, note and plan updated. Not committed yet.

---

## 2026-10-02 (later): 1e committed (`bc37e79`); step 1f done, so step 1 is complete

- **New `docs/conventions.md`**, a one-page reference:
  - graph: entities from the MPG traversal; incidence `B[parent,e]=+1`, `B[child,e]=−1`; unsorted post-order with `entity_ids`; edges identified by their child;
  - grids: `(x, y, z)` axes;
  - the locations table, and scale names as locations;
  - `scale` / `location` / `mapping`, with the mappings table and the D9 default table;
  - write-back rules and the declarations rejected because they could not be written back;
  - output locations;
  - identity / alias / derived couplings (recomputed at read, read-only, `mark_written`, `validate(strict=True)`);
  - `previous()` and `on_grow`;
  - failure modes.
- **`test/data_api_tests/test_conventions_doc.py`** (7 tests) checks the page's tables against `coupling.declaration`: locations, mappings, former names, and every row of the default table against `default_mapping`.
- **`downstream_migration.md` §2** is rewritten for the new keys:
  - coarse scales stored at their own scale;
  - `edge_mapping="proximal"` becomes `location="edge", mapping="child"`;
  - read-only derived inputs, output locations, and missing variables raising.

  Checklist step 3 now passes `data_structure=`. The README model-design section points to the guide and the conventions (it still describes the former API).
- **Plan:** DS3, DS5, DS11, DS16 and DS17 are ticked. The design note status says "step 1 complete".
- **Gap noted:** D8 mentions `index_of`, which does not exist yet. The conventions page documents `entity_ids` only.
- **Suite:** 585 passed, same 10 warnings. Not committed yet.
- **Next in the plan (§7 step 2):** DS19 (StructuralComponent contract), DS20 (repartition and active mask), DS21 (active subgraph), DS8 (multiscale assembly), DS2 (traversal), DS6 (boundary sets with UC5). Each needs its design note first.

---

## 2026-10-02 (later): 1f committed (`a987cba`); step 2 design note drafted

- **New `docs/design/structure_and_boundaries.md`** for §7 step 2 (DS2, DS19, DS20, DS21, DS6 with UC5, DS8), written against `a987cba`.
- **Code facts that change the plan:**
  - Scheduling is per component. A component's call runs all its rows, and components interleave only in the composite's call order.
  - The reference composite calls `root_growth()` first, so its potential, actual and segmentation steps read the previous step's carbon state. The "potential → allocation → actual across components" interleaving assumed in DS19 does not exist today. Proposed: defer it to DS13 (P2).
  - The MPG has no modification counter. Proposed: detect topology changes by a cheap signature (P3).
  - The residual convention `storage + B·outflux − sources = 0` fixes the Robin sign.
  - `index_of` (D8) is still missing; it is added in 2a.
- **Proposed sub-steps:** 2a traversal → 2b StructuralComponent → 2c repartition and active mask → 2d active subgraph → 2e boundary sets and UC5 → 2f anatomy mode. Each has its validation.
- **Points to agree:**
  - P1: sync per MPG-style step;
  - P2: defer cross-component interleaving to DS13;
  - P3: signature-based change detection;
  - P4: a `transient=` flag for the well-posedness check;
  - P5: boundary-set membership on write counters;
  - P6: edge ids by Connection vid in anatomy mode, by child vid in segment mode;
  - P7: the sub-step order.
- No code changes; the note and plan pointer are not committed yet.
- **Review of the step 2 note:**
  - P1, P3, P5, P6 and P7 are agreed.
  - **P2 is answered "no".** There is no cross-component interleaving, neither now nor deferred. The potential / allocation / actual rows order processes within one component, and components are called whole, in the composite's order. The note §1 and the plan's DS19 are corrected.
  - **P4 is re-explained** in the note: disconnected pieces of an active subgraph; why a steady balance needs an anchor (Dirichlet or positive-weight Robin) and a transient one does not; why the framework needs a `transient=` flag; its default by method; the risks of a wrong value; the scope (active subgraphs only). It is awaiting your answer.
  - **P4 is agreed** ("most of the time models won't have graph discontinuity", but the check is accepted). The note's status is now agreed; next is 2a.

---

## 2026-10-02 (later): step 2 note committed (`7796833`); step 2a implemented (traversal, index_of)

- **`MPGDataStructure`:**
  - `parents()`, `children()` (CSR), `roots()`, `tips()`, `order("pre" | "post")` (iterative DFS) and `owner(location)`;
  - all derived from the Connections, cached by `(topology_version, n_nodes, n_edges)`;
  - several parents per node, or a cycle, raise.
- **`VariableStoreMixin.index_of(ids, location)`:** vectorised (sorted entity ids plus `searchsorted`), cached per location and topology, raising `KeyError` on unknown ids. Traversal stubs raise `NotImplementedError` on grids.
- **On the seedling:** 14 nodes, 13 edges, one root, 4 tips. The multiscale branch (`root_segment4` under `root_segment2`) has its within-scale parent.
- **Tests:** every `_idx_to_vid` / `_vid_to_idx` use outside `test_legacy_mpg.py` is replaced by `entity_ids("node")` / `index_of`. The root and tip helpers of UC1, UC3/UC4 and the coupler tests use `roots()` / `tips()`.
- **New `test/data_api_tests/test_traversal.py`**, 10 tests:
  - parents against the edges;
  - the multiscale branch;
  - CSR, tips and parents agreeing;
  - pre/post orders respecting the tree;
  - the order kind checked;
  - `index_of` on node, edge and Organ, with unknown ids raising;
  - `owner`;
  - traversal after growth;
  - grids.
- **Docs:** conventions page (traversal and `index_of`), CHANGELOG, plan (DS2 ticked) and note status updated.
- **Suite:** 595 passed, same 10 warnings. Not committed yet.

---

## 2026-10-02 (later): 2a committed (`4fbe3dc`); step 2b implemented (StructuralComponent)

- **Component classes:** new `DataStructureComponent` base, holding the DataStructure binding, declaration resolution and registration, `pull_available_inputs`, MTG write-back and the parameter refresh. `FunctionalComponent` keeps `_graph_view` and `previous()`.
- **`StructuralComponent`** (formerly an empty stub) adds `mtg` and `_run_mpg_step`.
  - The Functor calls MPG-style (argument-less) steps of a StructuralComponent through it:
    - flush its MTG-backed declared variables (`write_mtg`);
    - take the MPG signature `(nb_vertices(), _id)`, run the step, compare (P3);
    - call `ds.update_topology()` if changed;
    - re-read the declared state variables (`read_mtg`).
  - Array-style steps are unchanged, vectorised on the DataStructure.
- **Bug found and fixed: metadata precedence.** A variable read as an input by one component and owned (state variable) by another took its default, `on_grow` and kind from whichever registered first. In the test, the growth model registered `C_hexose_root` as an input before the carbon model declared it with `on_grow="inherit"`, so new segments got the default instead of their parent's concentration. The owner's metadata now wins, and inputs only fill unknown keys.
- **Test helper `test/structure_tests/growth.py`:** a rhizodep-like model on a root chain:
  - potential, actual and segmentation steps, MPG-style;
  - radius thickening, array-style;
  - distance from tip as post-segmentation;
  - plus a `CarbonProbe` FunctionalComponent.

  `structure_tests` is added to the root conftest's import paths.
- **New `test/structure_tests/test_structural_component.py`**, 9 tests:
  - the rows;
  - `mtg`;
  - inputs flushed before an MPG-style step;
  - outputs re-read;
  - array-style outputs reaching the MTG at the end of the call;
  - `update_topology` only on segmentation;
  - FunctionalComponents following the new topology, with inherited concentration;
  - post-segmentation on the new structure;
  - metadata precedence.
- **Docs:** CHANGELOG, conventions (structural components), migration guide (growth models), plan (DS19 ticked, repartition in 2c), note status.
- **Suite:** 604 passed, same 10 warnings. Not committed yet.

---

## 2026-10-02 (later): 2b committed (`e659bd0`); step 2c implemented (repartition, active mask)

- **Named masks on `VariableStoreMixin`:**
  - `define_mask(name, rule, location)`, with a `{variable: ">0" | value | values}` rule or a callable;
  - `mask(name)`, recomputed on write counters and topology version;
  - `mask_version(name)`, bumped when the values change;
  - `has_mask`, `masks`.
- **Masked steps:**
  - step decorators accept `where=` ("active" by default, applied only if the DataStructure defines it; `None` opts out);
  - the Functor restricts arguments at the mask's location and scatters outputs at that location onto the selected entities;
  - an undeclared output under a mask is inferred at the mask location, with a warning;
  - a named mask that is not defined raises.
- **`StructuralComponent`:**
  - `partition_weight` and `active` as class attributes (instances may override them);
  - `active` defines the `"active"` mask at construction;
  - `_run_mpg_step` records the weight and activity by vid before the step, then after the re-read calls `_repartition`.
- **`_repartition`** runs in pre-order. New active entities, and existing ones becoming active, are split pairwise with their parent. New inactive ones copy intensive values and hold zero amounts. Then the remaining active entities are diluted by `w_before / w_after`. The structural component's own variables and derived variables are skipped.
- **Deviation from the note, recorded there:** no `previous_weight` attribute. The framework records the weight before each MPG-style step; dilutions compose to rhizodep's `initial_struct_mass` result.
- **New `test/structure_tests/test_repartition.py`**, 7 tests:
  - elongation dilution;
  - segmentation split, with c·w conserved and every kind checked by hand;
  - no weight means only `on_grow`;
  - an inactive primordium copying concentrations, with zero amounts and the carrier not reduced;
  - emergence splitting it from its carrier;
  - masked `@rate` and `where=None`;
  - mask recomputation and version.
- **Docs:** CHANGELOG, conventions (repartition, masks), note §4 and status, plan (DS20 ticked).
- **Suite:** 611 passed, same 10 warnings. Not committed yet.

---

## 2026-10-02 (later): 2c committed (`6430b08`); step 2d implemented (graph systems on the active subgraph)

- **`@graph_system(where=, transient=)`.** `transient` defaults to `True` for `ExplicitEulerSolver`, `ImplicitEulerSolver` and `ScipyIVPSolver`.
- **`_invoke_graph_system`** builds a `_Restriction` for `where=`:
  - active node indices, kept edges (both ends active), and the sub-GraphView (sliced incidence, local tail/head);
  - cached by `(topology_version, mask_version)`;
  - during the solve, `self._graph_view` returns the sub-view (`_solve_view`), so user equations are unchanged;
  - the body moved into `_solve_graph_system`.
- **Builder:**
  - `_read_array(take=)` slices snapshots, initial guesses, `_previous_state` and integrated amounts;
  - `inject_result` and graph outputs scatter back: inactive nodes frozen; dropped edges at 0, except `{field}_amount`, which is kept;
  - the solution saved for the next solve's previous values is now full-size (`ds.get` after inject), so it survives mask changes; it is sliced when used.
- **Well-posedness check (P4):** for a steady system on a restriction, `scipy.sparse.csgraph.connected_components` on the sub-view. Every piece needs a Dirichlet node (the union of the Dirichlet BC filters); otherwise a `ValueError` names the system and up to 10 vids.
- **Empty mask:** the solve is skipped, and edge unknowns are set to 0.
- **Hand-set `_boundary_ports` with `where=`** raise `NotImplementedError` (boundary sets in 2e).
- **New `test/graph_system_tests/test_active_subgraph.py`**, 8 tests, with a transient diffusion and a steady Darcy potential on the seedling:
  - all active is bit-for-bit equal to the whole-graph solve over two steps;
  - a dead interior node is frozen, its edges carry zero flux, and mass is conserved over the active set;
  - activation between steps, with mass conserved;
  - an empty mask;
  - a steady anchored solve;
  - an unanchored piece raising;
  - `transient` defaults;
  - hand ports rejected.
- **Docs:** CHANGELOG, conventions, plan (DS21 ticked), note status.
- **Suite:** 619 passed, same 10 warnings. Not committed yet.

---

## 2026-10-02 (later): 2d committed (`9e0b00b`); step 2e implemented (boundary sets, UC5)

- **`boundary_set`** is a declared data object in the graph-system class, collected by the builder before method binding.
  - Its membership is a DataStructure mask (`__boundary_set:<Class>.<name>`): a dict or variable-name select is recomputed on write counters and topology, a callable select at each solve. It is sliced to the active subgraph.
  - `value` and `weight` are read from the snapshot (they are added to the required names) or are constants.
  - The default field is the only node unknown; with several, `field=` is required.
- **Assembly** in `make_combined_node_ev`, per field:
  - Robin `+ w(x − v)` and Neumann `− v·scale` after the bulk terms;
  - the existing `@boundary_condition` rows;
  - then Dirichlet sets `x − v`, last.
- **User `@graph_jacobian`:** the framework adds `w` on the Robin diagonal and replaces Dirichlet rows by identity, at the field's offset (`index · n`), for dense or sparse Jacobians.
- **Well-posedness anchors** now include Dirichlet sets and positive-weight Robin sets. The error says "no Dirichlet or positive-weight Robin anchor"; the 2d test pattern is updated.
- **Deprecation:** `FunctionalComponent._graph_view` warns when building a view with hand-set `_boundary_ports`. UC3/UC4 deliberately keep the former path, with a module-level `filterwarnings` for that message.
- **UC5, `test/graph_system_tests/test_uc5_leaf_transpiration.py`** (6 tests): a steady water potential on the seedling, with leaves Robin to a per-leaf air water potential (callable select on the runtime label code) and roots Robin to the soil (variable-name select). The tests:
  - match the direct linear solve `(B diag K Bᵀ + diag w) ψ = w v`, with and without a user Jacobian (the framework adds the Robin terms);
  - see a changed leaf microclimate at the next solve without a view rebuild;
  - include a grown leaf in the set;
  - check that Robin sets anchor both steady pieces of a cut active subgraph;
  - check the declarations.
- **Observations:**
  1. The leaf conductance `@rate` sits in a separate `Stomata` component. A step declared on a base dataclass is not inherited by subclasses: the Choregrapher registers steps by the function's class name, which is the DS13 class-name issue (step 5).
  2. A parameter read from the MTG gets its values for new entities only at the parameter refresh before the next solve, so a mask on it lags until then. This is documented in the conventions.
  3. The suite time grew from about 5 s to about 11 s. It is machine load on the multiprocessing scene tests, not 2e: those tests take 6.7 s before and 6.5 s after the 2e source changes, measured with the changes stashed.
- **Suite:** 625 passed, same 10 warnings. CHANGELOG, conventions, migration guide, plan (DS6 ticked) and note updated. Not committed yet.

---

## 2026-10-02 (later): 2e committed (`3ffb485`); step 2f implemented (anatomy mode), so step 2 is complete

- **Rule saved:** when a step's conclusion raises no decision or question, commit and go on unasked. 2f raises three questions (below), so I stopped after it.
- **MPG:**
  - `wire_junctions(from_scale, rules, children=None)`: label rules with `match` all / nearest / equal on an ordering property, or a callable rule; it marks junctions `is_junction = 1`;
  - `populate_graph_custom_connections` delegates to it, with the existing MPG tests unchanged;
  - new helpers `linked_parent` (the `populate_graph` link logic), `compartments_by_owner`, `_valid_vids_at`, `junction_vids`, `remove_connections`.
- **`MPGDataStructure(..., nodes="Compartment", wiring=)`:**
  - nodes are the Compartments (MTG parent = owning SubOrgan), and edges are every Connection, keyed by Connection vid (`_connection_vids`, ascending like `edges()`);
  - `_owner_at` routes owners through the parent;
  - coarse locations go up to `from_scale` included, so `"SubOrgan"` is one;
  - `update_topology` calls `_rewire_junctions` instead of `repopulate_graph`. Per-SubOrgan signatures (linked parent, Compartments with label and ordering values) select the vertices to rewire: new, re-linked, or own/parent anatomy changed. Stale junctions are removed, the others kept; the rebuilt vertices are recorded in `ds.rewired`;
  - `on_grow="inherit"` takes the same-label upstream Compartment.
- **Declarations:** `scale=Compartment` is MTG-backed in anatomy mode. This was found by the UC3-like solve: labels were silently never read before the fix.
- **Derived variables:** `target=mask`, with the mask version in the derivation stamps.
- **StructuralComponent:** `_repartition` raises `NotImplementedError` in anatomy mode.
- **Test helper `test/structure_tests/anatomy.py`:** four Compartments per segment (epidermis, cortex, two xylem vessels with `vessel_index`), three anatomy Connections, and rules (xylem equal on `vessel_index`, cortex all).
- **New `test/structure_tests/test_anatomy_mode.py`**, 8 tests:
  - node and edge identities;
  - junction pairs equal to `populate_graph_custom_connections`;
  - owners, and a sum to SubOrgan;
  - a target-filtered broadcast to symplastic Compartments;
  - growth keeping every old Connection vid and value, wiring only the new segment, with same-label inheritance;
  - differentiation rewiring only the touched links;
  - traversal, missing `from_scale` and repartition raising;
  - a steady hydraulic solve on the assembled (cyclic) graph with a soil Robin set and a collar Dirichlet set, matching the direct linear solve.
- **Suite:** 633 passed, same 10 warnings. CHANGELOG, conventions, note (deviations recorded) and plan (DS8 ticked) updated. Not committed yet.
- **Questions:**
  - A1, repartition in anatomy mode: the lineage would follow the owner SubOrgans (a new segment's Compartments split from the same-label Compartments of its parent segment). Implement it now, or when GRANAP becomes a StructuralComponent?
  - A2, `derive(target=)`: entities outside the target get the default. OK, or should a targeted mapping write into an existing variable instead (keeping the others' values)?
  - A3, `is_junction = 1` instead of `edge_kind = "junction"`: OK?
- **Answers on 2f:**
  - **A1:** wait for GRANAP as a StructuralComponent. GRANAP will assign anatomies from a dynamic library per diameter differentiation class: one anatomy can populate several segments of a class, with only the inter-SubOrgan links added. Recorded in the plan, with question Q-A4 (per-segment copies or a shared template).
  - **A3:** agreed (integer markers, consistent with integer types and labels).
  - **A2:** re-explained (options: a default outside the target, or a targeted write into an existing variable).
  - **A2: option 1** (a separate derived variable with a default outside the target; other cell types in their own variables, combined with `np.where` by the model). No code change.

---

## 2026-10-02 (later): 2f committed (`b8e9fba`), so step 2 is complete; step 3 design note drafted

- **New `docs/design/cross_scale_and_grids.md`** for §7 step 3 (DS18 cross-scale links, DS1 grid topology), written against `b8e9fba`.
- **Code facts:**
  - the composite maps a cross-scale link with `derive(location=<receiver's>, aggregation=link.aggregation)`, which raises when no aggregation is given;
  - `Link.scale` / `source_scale` are parsed but never used;
  - inputs carry no `state_variable_type`;
  - coarse ↔ coarse mappings are missing;
  - grids have no `to_graph_view`, so `@graph_system` cannot run on soil;
  - the reference soil model wraps lateral neighbours in x and y for symmetric scenes (`symetry`), and uses Neumann or groundwater boundaries at the bottom.
- **Proposed:**
  - 3a: defaults by provider kind (D9 table), kind agreement (optional `state_variable_type` on inputs), coarse ↔ coarse mappings;
  - 3b: link scale checks, and translator `target=`;
  - 3c: grid topology (cells as nodes, internal faces as edges, optional periodic axes, `face_area` / `face_distance` edge variables, `face_axis`, `layer_mask`);
  - 3d: graph systems on grids. Validated by `B·diag(A/d)·Bᵀ/V == −laplacian()` exactly, and by an implicit-Euler soil diffusion against the direct sparse solve.
- **Points to agree:**
  - R1: link scales as checks only;
  - R2: `target=` on translator links;
  - R3: periodic axes as a grid option;
  - R4: face orientation (positive flux towards increasing coordinates);
  - R5: the order, with multigrid later.
- No code changes; note and plan pointer not committed yet.
- **R1–R5 agreed as recommended**; the note is committed and 3a starts.

---

## 2026-10-02 (later): step 3 note committed (`650eb14`); step 3a implemented (default link mappings, kinds)

- **Composite, `_couple_on_data_structures`:**
  - `_check_link_kinds`, for every link: the receiver's declared kind against the provider's;
  - `_default_link_mapping`, for links between two locations without an aggregation: `link_direction` (ranks: scalar < coarse scales < nodes; edges have no default) and `default_mapping` on the sources' recorded kind; mixed kinds raise; errors name the link and both locations.
- **Found while testing:** a single-source factor-1 link was always an alias, so across scales the receiver silently aliased a node-sized array. Such links now become mapped derived variables.
- **`declaration.py`:** `KIND_FAMILIES`, `kinds_agree`, `link_direction`.
- **`input_variable(state_variable_type=)`** is optional.
- **`MPGDataStructure._map`:** coarse → coarser aggregation (via `complex_at_scale` of the finer entities) and coarser → coarse broadcast.
- **`couplability_problems(..., data_structure=)`** also reports kind conflicts and missing default mappings for links whose providers are already registered on the DataStructure.
- **New `test/wrappers_tests/test_cross_scale_links.py`**, 8 tests on the seedling:
  - extensive sum up;
  - intensive mean up;
  - massic weight required (and given);
  - broadcast down, with extensive down raising;
  - untyped provider raising, or given an aggregation;
  - kind conflict;
  - Organ ↔ Axis;
  - couplability reports.
- **Suite:** 641 passed, same 10 warnings. No open question, so 3a is committed and 3b starts (your rule).

---

## 2026-10-02 (later): 3a committed (`40657a2`); step 3b implemented (link scale checks, targets)

- **`Link.target`:** new field, part of the `scale_change` detail, so the link is always derived. It is read from nested specs (`"target"`), and Python translators pass it to `link()`.
- **`CompositeModel._check_link_scales`:** a link's `scale` / `source_scale` must match `location_of_scale` of the declared receiver and sources (R1).
- **Targeted links** call `derive(target=, default=<receiver's default>)`. An unknown mask raises at coupling.
- **Tests:** 3 more in `test_cross_scale_links.py`:
  - matching scales pass, and a wrong `scale` or `source_scale` raises;
  - a targeted broadcast reaches only the root segments, the others keeping the receiver's default;
  - Python translator `target`, and an unknown mask raising.
- **Plan:** DS18 ticked (3a–3b).
- **Suite:** 644 passed, same 10 warnings. No open question: 3b committed, 3c starts.

---

## 2026-10-02 (later): 3b committed (`1f6c926`); step 3c implemented (grid topology)

- **`ArrayDataStructure`:**
  - `_build_faces` (axis by axis, internal then wrap faces for `periodic` axes with more than 2 cells); `face_axis`, `n_nodes`, `n_edges`, `edges`, `incidence_matrix`, `to_graph_view` / `topology` (hand ports rejected), `layer_mask`, `periodic`;
  - an `"edge"` store, with `face_area` and `face_distance` registered at construction;
  - `entity_ids("edge")`;
  - `locate(periodic=None)` defaults to the grid's axes. The Coupler passes its own, so it is unchanged.
- **`MPGDataStructure.topology()`** is an alias of `to_graph_view()`.
- **Declarations:** grids accept `location="edge"`.
- **Bug found and fixed:** `_is_graph` used `hasattr(ds, "to_graph_view")`, true for grids now, so cell declarations were rejected. It now tests for an MTG.
- **Test-harness robustness:** with that bug, `test_ds_scene_contract.py` and `test_doubles.py` **hung** instead of failing. The soil, built in a thread, failed, and the plant waited on its queue forever (the first full run timed out after 600 s and was stopped). `RecordingQueue.get` in `test/wrappers_tests/conftest.py` now waits at most 20 s and raises `TimeoutError("did a model fail in another thread?")`. Checked by reintroducing the bug: the tests fail in 20 s.
- **New `test/data_api_tests/test_grid_topology.py`**, 8 tests:
  - the face graph reproduces `−laplacian()` exactly on 1-D, 2-D and 3-D grids with anisotropic dx;
  - orientation;
  - periodic wrap counts and orientation, with no wrap for 2 cells;
  - face geometry and layer masks;
  - `locate` defaults;
  - edge declarations on grids.
- **Suite:** 652 passed, same 10 warnings. No open question: 3c committed, 3d starts.

---

## 2026-10-02 (later): 3c committed (`ceb6e2f`); step 3d implemented (graph systems on grids), so step 3 is complete

- **Builder:**
  - `_read_array` flattens multi-dimensional arrays (C order);
  - `_snapshot` treats `"cell"` as nodes;
  - `_Restriction.scatter` writes through a flat view;
  - where-masks and boundary-set masks may be cell masks (boundary-set masks are defined at `"cell"` on grids);
  - previous and saved fields are flattened;
  - new unknowns and graph outputs at "node" are registered at `"cell"` on grids (`_entity_location`).
- **`VariableStoreMixin._write`** reshapes a flat array of the right size onto a multi-dimensional variable.
- **New `test/graph_system_tests/test_grid_graph_systems.py`**, 5 tests, with Fickian diffusion (`D · face_area / face_distance · Bᵀc` per face, balance divided by `cell_volume`, implicit Euler through `previous()`) on an anisotropic 4×3×5 grid:
  - it matches the direct sparse solve of `(I − dt·D·L)c = c_old + dt·s`, within the Newton tolerance;
  - a groundwater Dirichlet layer as a boundary set matches the direct solve with replaced rows;
  - a pot conserves mass (outer faces are not edges);
  - a frozen top layer through `where="active"` on cells;
  - periodic wrap faces carry flux and conserve mass.
- **Plan:** DS1 ticked (MultiGrid later).
- **Suite:** 657 passed, same 10 warnings.
- **Next:** §7 step 4 (DS10 sub-stepping and the adaptive loop, DS4 sync policy, DS9 solve-time views, DS12 typed variables) needs its design note first.
- **Step 4 design note drafted:** `docs/design/time_and_data.md`, covering DS10, DS4, DS9 and DS12.
  - **Facts:**
    - graph systems do one `step_once` per call;
    - Newton and implicit Euler return zero error estimates, so the existing adaptive loop is not adaptive for them;
    - equations read `self.time_step` and a `previous()` fixed per solve, while the solver's sub-step `ctx.dt` and `ctx.previous_node_fields` never reach them, so sub-stepping would be silently wrong;
    - parameters are re-read only before solves;
    - snapshots copy everything;
    - labels are floats, and lists cannot be registered.
  - **Points to agree:**
    - T1: `self.dt`;
    - T2: `previous()` levels (sub-step, solve, step);
    - T3: adaptive by step doubling;
    - T4: `mtg_sync` and parameter re-read per call;
    - T5: read-only snapshot views;
    - T6: integer, label-name and object variables;
    - T7: the order 4a → 4b → 4c.
  - Stopped for agreement (design note before a complex step).
- **Step 4:** "Go on with step 4", with empty answer lines, is taken as agreement with the recommendations T1–T7 (recorded in the note). Note committed; 4a starts.

---

## 2026-10-02 (later): step 4 note committed (`2c0ccff`); step 4a implemented (DS4, DS9)

- **`DataStructureComponent.mtg_sync`:** `"after_call"` (default) or `"never"`, checked when the component writes back.
- **`pull_available_inputs`** now re-reads the MTG-backed parameters at the start of every call (and still before each solve).
- **Snapshots:** `_read_array(read_only=True)` returns read-only views for parameters and inputs (float64, no restriction); unknowns, previous states and amounts stay copies.
- **Benchmark (20 001-node branched tree, 10 parameters):** a snapshot takes 95.9 µs with copies (1.6 MB copied) against 44.6 µs with views (nothing copied).
- **Found (recorded under DS14):** `populate_graph` on a single 20 000-segment chain raises `RecursionError` in openalea.mtg's recursive `pre_order` (via `post_order_mpg` → `components_iter`). The benchmark used 200 axes of 100 segments instead. This matters for long roots, and belongs to step 5 (performance).
- **New `test/data_api_tests/test_sync_policy.py`**, 4 tests:
  - an MTG parameter change is seen by a `@rate` at the next call;
  - `"never"` leaves the MTG untouched;
  - an invalid policy raises;
  - an equation writing into a parameter raises `read-only`, with the DataStructure intact.
- **Plan:** DS4 and DS9 ticked.
- **Suite:** 661 passed, same 10 warnings. No open question: 4a committed, 4b starts.

---

## 2026-10-02 (later): 4a committed (`6bfcaf7`); step 4b implemented (DS10 time integration)

- **`@graph_system`** gains `integrate="step" | "substeps" | "adaptive"`, `n_substeps`, `rtol`, `atol`, `min_step` and `max_step`, all validated.
- **`_solve_graph_system`** dispatches:
  - `_solve_once` (the former body; dt from `_current_dt`);
  - `_solve_substep` (sets the solver's previous fields to the state at the sub-step start, and `_current_dt`);
  - `_integrate_adaptive`: step doubling, comparing one step of h with two of h/2 on the node unknowns (`rtol·|x| + atol`), with h adapted by `0.9/√err` within [0.2, 2], and `_last_integration = {steps, rejected}`.

  The unknowns at the start of the solve are kept for `previous(at="solve")`.
- **FunctionalComponent:**
  - `dt` property;
  - `previous(name, at="substep" | "solve" | "step")`;
  - `pull_available_inputs` records the unknowns of all its graph systems, with their location, for `at="step"` (sliced on active subgraphs).
- **`BoundaryConditions`** is not attached to decorator specs, so time-varying boundaries are boundary sets, re-read at every sub-step. Recorded in the note.
- **Bug found and fixed (pre-existing):** `ArrayDataStructure.laplacian()` put `−1/h²` on the diagonal for axes with a single cell, a spurious sink: a `(6, 1, 1)` column lost mass. Found because the implicit-Euler reference disagreed with the face-graph solve. The 3c Laplacian test now also covers `(6, 1, 1)` and `(1, 4, 3)`.
- **New `test/graph_system_tests/test_time_integration.py`**, 7 tests, with diffusion along a `(6, 1, 1)` column written with `self.dt` and `previous()`:
  - one step is unchanged, and `dt == time_step`;
  - 4 sub-steps equal 4 hand implicit-Euler solves of dt/4;
  - adaptive is 5 times closer than a single step to the exact `expm(dt·D·L)·c0`, conserving mass;
  - slow dynamics take fewer adaptive steps;
  - a step below `min_step` raises;
  - operator splitting with `previous(at="step")`;
  - the options are checked.
- **DS13 seen again:** two instances of one class cannot both be called, because the Choregrapher binds a class's steps to its last instance. The adaptive comparison builds and runs them one after the other.
- **Plan:** DS10 ticked.
- **Suite:** 670 passed, same 10 warnings. No open question: 4b committed, 4c starts.

---

## 2026-10-02 (later): 4b committed (`c254840`); step 4c implemented (DS12 typed variables), so step 4 is complete

- **Variable store:**
  - `register(dtype=float | int | object)`, with the dtype in the metadata (kept across growth);
  - `_converted` writes values: integers checked; objects one per entity from a sequence of the right length, otherwise the same value for every entity (ragged lists handled);
  - `_carry_over` builds object arrays for object variables;
  - `derive` rejects object sources;
  - `export` keeps dtypes.
- **Declarations:**
  - `dtype=` on `declare` and the three wrappers, carried by `VariableSpec.dtype`, and registration passes it;
  - object defaults are not converted.
- **MTG:** `read_mtg` reads object node variables at their own scale by identity, and `write_mtg` writes them back by identity; `_write_at` keeps integers.
- **Label names:** `MPGDataStructure.label_code(name, variable)` / `resolve_codes`, through `LabelsConfig`:
  - a label value (`labels.filters`, unique) first;
  - then an attribute of the group whose `scale` is the variable's declared scale;
  - then any group;
  - ambiguous or unknown names raise.

  Used by `_evaluate_mask` (masks and boundary-set dict selects) and `_type_mask(ds=)` (graph-system filters and anchors).
- **Rejections:**
  - `_read_array` rejects object variables in graph systems;
  - `Coupler.push` rejects object sources.
- **New `test/data_api_tests/test_typed_variables.py`**, 5 tests:
  - integer labels kept as integers, with non-integral writes raising;
  - label names in masks and filters;
  - anatomy mode, where `"Symplastic"` resolves to the Compartment group from the variable's scale, and is ambiguous without one, with unknown names raising and label values unique;
  - object vessel radii round-tripping through the MTG, used by a per-element `@rate`, carried over growth;
  - object variables rejected by a graph system and by `derive`.
- **Docs:** CHANGELOG, conventions (types and label names), migration guide (non-float variables), plan (DS12 ticked), note (step 4 complete).
- **Suite:** 675 passed, same 10 warnings.
- **Next:** §7 step 5 (DS13 per-instance scheduling, DS14 performance, including the `RecursionError` on long chains found in 4a, and DS15 persistence) needs its design note first.
- **Your `devplan_scene_paralellization.md`:** questions A and B are answered in the file, below your text (no code changed, as you asked):
  - **A:** traversal-dependent computations vectorise from the cached topology arrays: level-by-level accumulation and propagation, segmented scans along axes (rhizodep's `distance_from_tip` is a reverse cumulative sum per axis, checked read-only), pointer jumping, or sparse triangular solves. The cost is O(tree height) numpy calls, independent of the number of plants. Order-dependent sequential rules (allocation, segmentation while iterating) need a decide-then-apply reformulation, or numba.
  - **B:** the MPG traversals (`pre/post_order_mpg`, `components_iter`, `complex_at_scale`, `populate_graph`'s link search) are Python loops or recursion over openalea.mtg dicts. Array mirrors (parent, complex, scale) would make them vectorised, and remove the `RecursionError` found in 4a.
  - Also covered: implications for a population held in one MPG (heterogeneity as per-plant variables, one Coupler, DS13 becoming a smaller need).
  - Proposed DS14a (tree kernels), DS14b (MPG array mirrors) and DS13 reconsidered, with questions PA1 (sequential allocation rules) and PA2 (start step 5 with a population prototype).
- **Step 5 design note:** not drafted yet; it waits for PA1/PA2, which change its order and scope.

---

## 2026-10-02 (later): PA1–PA2 answered; step 5 design note drafted

- **PA1:** vectorising the C supply is fine if the result is numerically the same. **PA2:** start step 5 with the population prototype, to opt out of the multiprocessing constraint.
- **Read (not edited):** rhizodep's `calculating_supply_for_elongation` and `actual_growth_and_corresponding_respiration`.
  - Each apex walks its ancestors until a volume budget is reached, with a fractional last segment, reading without depletion.
  - Consumption is accumulated (`+=`) onto the supplying segments.
  - So there is no visiting-order dependence, except floating-point rounding where windows overlap.
- **New `docs/design/population_and_performance.md`:**
  - 5a: MPG array mirrors (`topology_arrays()`: parent, complex, scale, edge type), with `populate_graph`, `wire_junctions` and the memberships vectorised and iterative orders, giving the same Compartments in the same order;
  - 5b: tree kernels (`levels`, `depth`, `accumulate`, `axes`, `axis_scan`, and `path_window` in numba `prange`), bitwise identical to rhizodep's order for scans and windows;
  - 5c: N plants in one MPG, one instance per component, one Coupler, an in-process soil, and benchmarks for N = 1 to 1000 with a decision point;
  - 5d: DS13 reduced to per-class-object registration with per-instance binding;
  - 5e: DS15 checkpoints (npz + JSON).
- **Your parallelism question,** answered in §7 of the note: numba `prange` (multi-threaded compiled loops), numexpr, multi-threaded BLAS, `pypardiso` / `scikit-umfpack` for sparse direct solves, per-component block solves, GPU later. The scene's `OMP_NUM_THREADS = 1` per worker would become several threads per population process.
- **Questions S1–S8:**
  - S1: exactness of accumulated consumption;
  - S2: target sizes, and anatomies;
  - S3: numba as a dependency;
  - S4: rebuilding the mirrors;
  - S5: persistence format;
  - S6: the prototype scene in one process;
  - S7: rhizodep rules reimplemented in test helpers;
  - S8: the order.
- Not committed yet.
- **Step 5 note, after your answers S1–S8:**
  - **S1:** bitwise by emitting contributions in rhizodep's visiting order, with an `(k − 1)·ε` bound otherwise.
  - **S2:** per-connected-component solves (`split="components"`), threaded, benchmarked against one system.
  - **S5:** re-explained (what a checkpoint is; pickle versus npz + JSON).
  - **S7:** a read-only survey of cnwgrass, adel and GRANAP gave a kernel catalogue of 12 patterns (chain scans with max, lagged and forward chain writes, cross-chain gathers, transform composition, clamped recurrences, budgeted group depletion, …).
  - **Readability (your new question):** rules (no indices in models, named kernels, per-element opt-in), with before/after examples for `distance_from_tip` and the supply window.
  - **New questions:** S9 (chains), S10 (whether geometry and turtle frames are in scope), S11 (the 5b scope).
- **S9–S11 answered:**
  - S9: chains declared by name on StructuralComponents;
  - S10: geometry in scope, since `x1 … z2` position elements for soil and light;
  - S11: only the prototype's kernels, plus what decides the API's generality, namely vector-valued values and a minimal `path_compose`.

  The note is agreed and committed; 5a starts.

---

## 2026-10-02 (later): step 5 note committed (`207272d`); step 5a implemented (traversals without recursion, topology arrays)

- **`MPG.components_iter`** is overridden with an iterative version in openalea's exact order (component roots; then '+' children before '<' successors; children inside the complex when they have no own complex, or theirs is it). This removes the `RecursionError` from every traversal that uses it.
- **`MPG.topology_arrays()`** (cached by vertex count and last id): `parent`, `complex` (resolved by pointer doubling), `scale`, `edge_type`, `is_anchor`. **`complex_at_scale_array`** goes with it. `MPGDataStructure._membership` uses them, in segment and anatomy modes.
- **New `test/mpg_tests/test_topology_arrays.py`**, 7 tests:
  - `components_iter` equals openalea's on every complex of the seedling and of a branched root system;
  - a 20 000-segment chain: openalea raises `RecursionError`, and ours populates and wraps;
  - arrays equal `parent`, `complex` and `scale` of the MTG;
  - `complex_at_scale_array` equals the scalar one;
  - arrays follow edits;
  - `populate_graph` gives the same Compartment order and edges as with openalea's traversal.
- **Benchmark (20 001 segments, 200 axes):** `populate_graph` 2.13 → 1.93 s; Organ owners 131 → 53 ms.
  - Profile: 1.30 s of 2.26 s is creating 40 001 vertices, through 160 000 single-item `ArrayDict` inserts. The traversal is 0.36 s. Vectorising the link search would gain little.
  - At population scale (2·10⁶ segments), full repopulation at each growth step would take minutes.
- **Scene robustness gap found (F1, not caused by 5a):** with an intermediate 5a bug, plant workers failed in their constructor, and `test_data_structure_scene[fork]` hung for 60 s until the watchdog killed pytest. The cause: environment model construction is outside the worker's `try` and blocks on the plants' queues, and the main loop does not watch worker exit codes. A fix is proposed. The stray `hung_scene_traceback.txt` written by the watchdog was removed.
- **Timing noise:** the suite takes 9 s instead of 3–5 s, the same with and without 5a's source changes (checked twice, with stash). It is machine load (load average about 2.5).
- **Suite:** 682 passed.
- **Questions:**
  - F1: the scene fix;
  - F2: incremental segment-mode population and bulk vertex creation as the first part of 5c.

  5a is committed; stopped for F1/F2.

---

## 2026-10-04: F1–F2 agreed; plan for a scene of populations in one process

- **F1** (stopping the scene on a worker failure) and **F2** (incremental segment-mode population, bulk vertex creation) are agreed.
- **Your questions 1–3** (environment models as components with scene-level exchanges, the scene as a CompositeModel, heterogeneity between plants) are answered in the new plan file **`devplan_population_scene.md`**:
  - environment models become FunctionalComponents on their grids;
  - translator links between DataStructures become cross-DataStructure mappings (sparse incidence from a locator, recomputed on topology or geometry changes), generalising the existing `Coupler` / `VoxelLocator`;
  - the light model is a component reading every population's geometry;
  - `Scene(CompositeModel)` runs environment then populations, with one MPG per sub-population and intercropping through the environment;
  - planting comes from `stand_initialization`;
  - heterogeneous parameters are Plant-scale declarations, broadcast to segments.
- **Steps P1–P8:** robustness, incremental population, tree kernels, population builder, cross-DataStructure links, Scene, benchmarks and decision, then DS13 / DS15.
- **Questions Q1–Q10:**
  - Q1: exchange timing;
  - Q2: barycentre or overlap;
  - Q3: light geometry;
  - Q4: initial structures;
  - Q5: staggered emergence;
  - Q6: parameter laws;
  - Q7: logging;
  - Q8: two cultivars;
  - Q9: environment time steps;
  - Q10: `play_Orchestra`.
- The step 5 note and the plan point to the new file. No code changes.
- **2026-10-05:** Q1–Q10 answered and recorded in `devplan_population_scene.md` §5:
  - fixed-point exchanges;
  - barycentre by default;
  - light on an MPG or a grid;
  - per-plant initialisation;
  - staggered emergence later;
  - one scenario per plant;
  - selected-plant logging;
  - cultivars as one population, so DS13 is reduced to its hazard fix;
  - one time step;
  - `play_Orchestra` removed after P7.

  Follow-up QH1 (how per-plant parameters reach the equations; (b) declared-scale recommended) blocks only P4. P1 starts.

---

## 2026-10-05: P1 implemented (a failing worker stops the scene)

- **`scene_wrapper`:**
  - plant, soil and light model construction (and the loggers) moved inside the workers' `try`, so a construction failure sets `stop_event` and exits 1;
  - the main loop sets `stop_event` when a worker has a non-zero exit code;
  - `finally` joins within `shutdown_timeout` (new argument, 30 s), then terminates (and kills if needed) the workers still alive, and returns `clean_exit = False`.
- **`doubles.MinimalPlant`** accepts `fail_at="construction"`. A new slow test (fork, spawn, forkserver) runs a plant failing in its constructor next to a `DSFakeSoil`, which waits for the plants in its constructor. It returns `False` with no shared memory left, in about 2 s per start method.
- **Limit of the check:** running the new test against the old `scene_wrapper` fails at once, because the old function does not accept `shutdown_timeout`. So it does not reproduce the hang. The hang itself was observed during 5a: 60 s, then the watchdog.
- **Suite:** 685 passed.
- **Correction:** P1 was committed (`e528e46`) before I read that run's result, and 5 scene unit tests failed. Their fake process classes lacked `join(timeout)` and `is_alive()`. Fixed in a follow-up commit; 685 pass.

---

## 2026-10-05 (later): P1 committed (`e528e46`, test fakes fixed in `d0a32a7`); P2 implemented (graph at population scale)

- **`MPG.add_components_bulk(complex, count, topo_parents=, **properties)`:** the MTG bookkeeping per vertex, then one `update` per property (`ArrayDict` appends in one go since new vids are larger). Same vids and properties as one-by-one creation (tested).
- **`populate_graph`:** passes 1 and 2 use it (compartments first, then connections, so the vids are unchanged).
  - **Pass 3 bug found and fixed:** orphans with no filtered ancestor were all chained together, which would link the plants of a population. They are now chained per plant (`complex_at_scale(vid, Plant)`).
  - This chaining is how organ-scale roots of one plant get linked: the UC1-Organ graph needs it, and the first version, which removed it, broke that test.
- **`MPG.extend_graph(from_scale)`:**
  - adds Compartments and Connections for new vertices;
  - removes the Connections whose endpoints are gone, and the Compartments of removed vertices still present;
  - relinks children whose linked parent changed;
  - falls back to `repopulate_graph` only for a new parentless vertex in a plant that has others.

  Segment-mode `update_topology` uses it, and records `last_extension`.
- **Pruning:** use `remove_tree`. `remove_vertex` refuses, because the segment's Compartment is its MTG child.
- **New `test/mpg_tests/test_population_graph.py`**, 4 tests:
  - bulk creation equals one-by-one creation;
  - 3 plants in one MPG give 3 connected components;
  - growth keeps every Compartment and Connection vid, and equals a full repopulation of a twin (modulo the new vids);
  - pruning removes the vertex and its edges.
- **Benchmark (20 001 segments):**
  - `populate_graph` 1.99 → 1.30 s;
  - `update_topology` after 10 new segments **398 s → 0.42 s**. The old full repopulation deleted properties vertex by vertex: `ArrayDict.__delitem__` is O(n), so the rebuild was quadratic.
- **Suite:** 689 passed. No open question: P2 committed, P3 (tree kernels) starts.

---

## 2026-10-05 (later): P2 committed (`e5c864f`); P3 implemented (tree kernels)

- **New `src/openalea/metafspm/data_structure/tree_kernels.py`:**
  - chains from predecessors (the '<' edge type) or from group and rank;
  - `_segmented_scan` and `_path_window` as numba `parallel=True` kernels, sequential within a chain or a walk, so the order is the reference loops';
  - `chain_shift` and `chain_write` vectorised;
  - `depth` (pointer doubling; the first version was wrong and caught by the subtree check), `levels`, `accumulate` by levels (`np.add.at` / `np.maximum.at`), `path_compose` (batched matmul by levels).
  - **Exactness detail:** the fractional contribution is computed as `value * remaining / extent`, rhizodep's order, not `value * (remaining / extent)`.
- **`MPGDataStructure`:** `define_chain`, `chain` (cached by topology version and by the group and rank write counters; `"axis"` defined by default; edge-type chains not in anatomy mode), `chain_scan`, `chain_shift`, `chain_write`, `depth`, `levels`, `accumulate`, `path_window`, `path_compose`.
- **New `test/structure_tests/test_tree_kernels.py`**, 7 tests, against plain loops on a branched root system:
  - distance from tip equals rhizodep's recursion **bit for bit**;
  - the supply window equals rhizodep's walk (zero-length elements skipped, fractional last element) **bit for bit**;
  - subtree sum and max, and 3-D root-path sums;
  - rank chains: cnwgrass-style prefix max and exclusive sums;
  - chain shift and forward write;
  - `path_compose` against the product from the root;
  - a 3-plant population computed at once.
- **Deferred to the population growth helper (P4–P7):** emitting the supply contributions in visiting order for the consumption scatter (S1's bitwise accumulation). The window sums are done here.
- **Suite:** 696 passed.
- **P4** (population builder) **waits for QH1**, in `devplan_population_scene.md` §6: how per-plant parameters reach the equations.

---

## 2026-10-06: QH1 answered (every parameter stored at Plant scale)

- **Your choice:** every parameter is stored at Plant scale, heterogeneous only when the scenarios differ. Your question was how parameters and variables are told apart in method arguments while staying vectorisable and numba-compatible.
- **Answer** (devplan_population_scene.md §7): by declaration, not signature.
  - Each argument is resolved through its `VariableSpec`.
  - Parameters arrive broadcast to the equation's entity: node arrays for balances and rates, edge arrays from the child segment's plant.
  - Heterogeneous parameters come through a cached owner-map gather; homogeneous ones as a zero-stride read-only view.
  - Float arrays always, so one numba specialisation serves both cases.
- **New questions:**
  - QH2: whether `self.k` is forbidden inside equations;
  - QH3: whether `parameter()` without a scale defaults to Plant on plant DataStructures.
- P4 waits for them. No code changes.
- **QH2 / QH3 agreed:** `self.k` is forbidden inside equations, and numeric parameters default to Plant scale. The P4 design is drafted in the plan's §8:
  - Plant-scale storage;
  - per-argument broadcast (owner gather, or zero-stride);
  - a class data descriptor enforcing QH2 and setting every plant's value on write;
  - the planting table;
  - `build_population`;
  - a new `StructuralComponent.initiate_plant(g, plant_vid, parameters)` contract;
  - staggered emergence through a Plant-level active mask;
  - migration of about 20 in-repo `self.<parameter>` reads.

  Questions QP4a (`initiate_plant`) and QP4b (who computes positions). Stopped for agreement (design note before a complex step).

---

## 2026-10-06 (later): QH2/QH3 and QP4a/QP4b agreed; P4 implemented (populations, per-plant parameters)

- **QP4a:** several structural components may initiate a plant in sequence. **QP4b:** structural components compute positions; the P5 translator maps from the initialised variables.
- **Resolver:** a numeric parameter without scale or location resolves to `"Plant"` (scale Plant) on plant DataStructures, `"scalar"` otherwise; other unplaced fields return `None` before any default conversion.
  - **Latent 1a bug fixed:** string parameters without a scale raised "needs a numeric default".
- **`parameter_view(name, to)`** (mixin, with an MPG override for edges by their child node): a zero-stride read-only view when uniform, else an owner gather, cached by write count and topology.
- **Functor:** `_arguments` broadcasts parameter arguments to the step's location, before masking; `_in_equation` is flagged around the evaluation. Split into `_arguments`, `_evaluate` and `_write_outputs`.
- **Graph systems:**
  - `_snapshot` puts per-plant parameters into both the node and the edge snapshots;
  - evaluators look up their own entity's snapshot first, and filtered slicing is entity-aware;
  - `_in_equation` wraps the equation calls.
- **`_PlantParameter` data descriptor** (installed once per class, with values set by the dataclass `__init__` moved to pending): QH2's refusal inside equations; outside, the uniform value or the per-plant array; writes set every plant. Registration uses the constructor's value; a pre-registered variable is overwritten only by an explicit non-default value. `parameter_values()` serves MPG-style steps.
- **`StructuralComponent.initiate_plant`** contract; new `scene/population.py` (`planting_table` on `stand_initialization` with a seed and `per_plant_scenarios`, `build_population`, `apply_plant_scenarios`).
- **Test changes:**
  - `VectorisedProbe._clipped` takes `threshold` as an argument (the one migration needed in the suite);
  - the growth helper gained `initiate_plant` (chain under the Plant vertex, with positions from the plant's `x, y, z`) and reads `elongation_rate` per vertex.
- **New `test/structure_tests/test_population.py`**, 9 tests:
  - 100 identical plants give each plant exactly the single-plant result, in one call;
  - alternating scenarios match separate single-plant runs;
  - initial structures follow per-plant scenarios;
  - the planting table: size, seed reproducibility, per-plant scenarios;
  - parameters at Plant, broadcast and settable;
  - `self.k` inside a step refused;
  - zero-stride read-only views;
  - a numba step with uniform and varied parameters;
  - grid parameters at `"scalar"`.
- **DS13 hazard seen in the suite:** a test class named `GridDecay` collided with another in `test_transport.py`, because the Choregrapher merges same-named classes. The test class is renamed; P8 fixes the cause.
- **Moved to P6:** staggered emergence (Q5), which needs the scene clock.
- **Suite:** 705 passed.

- **P4 committed** (`4f097f5`): CHANGELOG, conventions (Parameters section), migration guide (parameters as arguments), plan ✓.
- **P5 design note** drafted in `devplan_population_scene.md` §9, with questions QP5a–c. Waiting for the answers before any code, since they decide the default mappings and the light-over-populations design.

## 2026-10-06 (later): QP5a–c agreed; P5 implemented (links between DataStructures)

- **Answers:** all three recommendations agreed. Any kind of variable must be exchangeable, even though in practice only extensive plant → soil and intensive soil → plant values are passed.
- **`coupling/cross.py`:**
  - **`CrossMapping`:** a (rows, columns, weights) incidence, rebuilt when the source's `topology_version` or a coordinate's write count changes.
    - *Barycentre*: the same cells as `VoxelLocator`, bit for bit.
    - *Overlap*: a numba traversal cuts each segment at the cell faces (piece middles located with `grid.locate`, so periodic and clipped axes behave as in barycentre); weights are length fractions, and zero-length pieces are dropped.
    - Up: `sum`, `mean`, `weighted_mean`, returned as a numerator and denominator so populations pool. Down: `broadcast`, and `split` with cell totals over every receiving population.
  - **`Exchanges`:**
    - resolves the cross links (kinds checked with `kinds_agree`);
    - defaults from `cross_default_mapping` (QP5a: intensive up requires `weight=` or an explicit `mean`; QP5c: extensive down requires `weight=`);
    - formula links are evaluated on the provider;
    - `exchange(into=ds)` pools every provider of a variable into one `set()`; a cell mean with no plant gets the receiver's default.
  - **`UnionDataStructure`:** node and scalar stores; `element_scale` is the parts' node scale.
    - The resolver accepts that scale, or `location="node"`, on it. Before this, a CARIBU-like component declared with `scale=SubOrgan` silently had no declared variables there.
    - `update_topology()` carries values over by (part, id).
  - **`UnionMapping`:** one-to-one exchanges with the parts.
- **`Coupler`** builds its map through `VoxelLocator.mapping()` (a `CrossMapping`); its behaviour (explicit `update_map()`, `zero_soil_inputs`) is unchanged until P7. The Coupler and Transport tests pass unchanged.
- **Timing:** incidence builds take 0.11 s (barycentre) and 0.66 s (overlap) on 2·10⁶ segments; an up exchange takes about 10 ms. With overlap, the incidence is rebuilt each time the coordinates are written (e.g. at every growth step).
- **New `test/structure_tests/test_cross_datastructures.py`**, 13 tests:
  - barycentre equals `VoxelLocator`;
  - hand-computed overlap fractions (vertical and oblique segments);
  - the map follows growth and moves;
  - two populations into one soil in one write;
  - gathering by overlap;
  - intensive pooled weighted mean, and the default in empty cells;
  - split between populations (conservation, and a dense reference);
  - link checks;
  - formula links;
  - CARIBU-like light over two populations through a union, shaded by both;
  - the union following growth;
  - RATP-like light on a grid.
- **Suite:** 718 passed.
- **P6 design note** drafted in `devplan_population_scene.md` §10 (Scene contract, step order, logging, emergence), with questions QP6a–d. Stopped for the answers.

## 2026-10-06 (later): QP6a–d agreed; P6 implemented (the population scene)

- **Answers:** all recommendations agreed. QP6b: environment components that generate a structure expose it to the scene (as root growth does), and other components of the same compartment may operate on it. QP6d: (a).
- **`scene/scene.py`:**
  - **`Scene`:**
    - groups the planting table by model;
    - for each model: `build_population`, `populate_graph`, `MPGDataStructure`, the model, then `apply_plant_scenarios`;
    - builds the environment models with the population DataStructures;
    - refuses duplicate component class names;
    - infers the mappings from the translator links;
    - `run()`: emergence, then for each environment model `exchange(into=its DataStructures)` and its `run()`, then for each population `exchange(into=population)` and its `run()`;
    - `simulate()` / `stop()`, and the `logger_class` hook.
  - **`_shared_scenario`:** the model is built with its first plant's scenario. Numeric parameters may differ, or be missing for some plants; other entries must be equal (`_same` compares dicts, DataFrames and arrays).
  - **Emergence:** a Plant `emergence_time`, a node `emerged` (on_grow inherit) updated at the start of each step, the masks `emerged` and `active`; a model's `active` rule is kept as `_model_active` and combined.
  - **`SceneRecorder`:** CSV appends per population (pyarrow and netCDF are not installed in the env).
- **`CrossMapping(mask=)`:** rows outside the mask are dropped (the stamp includes `mask_version`), and `Exchanges` keeps their values on the way down.
- **`DataStructureComponent.active_ids()`;** the growth helper's `_apices` uses it.
- **`planting_table`:** `emergence_times=`, and the stand size in `table.attrs`.
- **New `test/structure_tests/test_scene.py`**, 9 tests:
  - one population per model, and the inferred mappings;
  - the fixed-point step order;
  - per-plant numeric parameters, and shared other entries;
  - duplicate component classes refused;
  - frozen until emergence (no growth, no exudation, not exchanged), then growing;
  - the recorder files;
  - a light model on the union of the populations;
  - the planting table's stand size and emergence;
  - emergence combined with a model's `active` mask.
- **Found while testing:** a step named `_seedling_exudation` silently wrote a new `seedling_exudation` variable (only a DeprecationWarning). Step names must match their output.
- **Suite:** 727 passed.
- **P6 committed** (`0867486`). **P7 design note** drafted (§11: benchmark protocol and the list of what is removed), with questions QP7a–b. Stopped for the answers: the removal is breaking for downstream users.

## 2026-10-06 (later): QP7a–b agreed; P7 done (benchmarks, removal of the per-process scene)

- **QP7a:** the in-repo doubles are enough. **QP7b:** remove regardless; the user considers the per-process parallelisation unfit for populations, and runtime optimisation comes later if needed.
- **`test/benchmarks/bench_population.py`:**
  - doubles with the real models' cost structure: vectorised carbon, an implicit diffusion graph system, bulk segmentation (`add_components_bulk` with predicted consecutive vids for chains), a soil grid, the recorder;
  - plants of 2 000 segments (a main axis of 1000 and 5 laterals of 200).
- **Results:** 1 → 1000 plants is linear, 6.7 s per step for 2·10⁶ segments; anatomy mode reaches 0.6 s per step for 8·10⁵ Compartments.
- **Profile at 100 plants:**
  - `write_back_to_mtg` is 97 % of rates-and-states (F3);
  - `extend_graph` traverses the whole MTG through openalea's `components_at_scale` at each growth (F4).

  Both are recorded in §14 and asked as QF3 / QF4.
- **Reference** (`bench_reference.py`, run, then deleted with `play_Orchestra`): `play_Orchestra` with `DSFakePlant` / `DSFakeSoil` against a Scene of the same toy plants, as (T(13) − T(3)) / 10 to cancel the start-up. Per step: 0.47 / 1.7 / 6.6 ms for 1 / 4 / 12 plants against a flat 0.3 ms.
- **Pitfalls hit while writing the benchmark:**
  - bulk creation needs `PropsConfig` (the `scale` property), or `populate_graph` silently skips the vertices;
  - an MPG-style step must give new vertices every MTG-backed variable, or `read_mtg` raises.
- **Removal:**
  - `scene_wrapper.py` (with `stand_initialization` moved to `population.py`) and `coupler.py`;
  - the soil members of `CompositeModel`;
  - six test files (79 tests); the queue, logger and light doubles and fixtures; the `slow` marker; `scene_wrapper_example.py`, rewritten as `scene_example.py`.
- **Test changes:**
  - the composite contract test now sets the soil values that coupling used to zero;
  - `test_translator`'s comparison with the removed method became fixed expectations;
  - the CrossMapping barycentre test computes the reference map itself.
- **New `test/wrappers_tests/test_scene_contract.py`:** the DS doubles in a Scene, with 3 plants feeding one soil without zeroing.
- **Docs:** migration guide §3–§5 rewritten for the population contracts and the Scene.
- **Suite:** 649 passed.

## 2026-10-06 (later): QF3 and QF4 agreed; F4 implemented (growth bookkeeping follows the growth)

- **`MPG.extend_graph`:** Compartments, Connections and endpoints are read as property arrays (`_vertices_at_scale`, `_property_at`) instead of `components_at_scale` traversals. The valid vertices are a sorted array (`_SortedIds`, with set-like membership for `linked_parent`), and added, removed and stale entities come from numpy set operations. 3.4 s → 0.21 s at 2·10⁵ segments.
- **`MPGDataStructure.update_topology`:** the carry-over of registered variables matches ids in bulk (`_KeyedValues`). Only new entities go through the per-key inherit walk, and the `default` policy is one assignment (object variables excepted).
- **Latent issue fixed:** in anatomy mode, the edge ids read at the start of `update_topology` are live, and growth has already added Connections. The old `zip(keys, arr)` only paired the values correctly because new vids are larger. The DataStructure now keeps the ids its arrays were built for (`_stored_ids`).
- **`MPG.topology_arrays`:** bulk `fromiter` reads of the MTG dicts instead of per-element numpy writes.
- **Result:** one growth step with an update at 100 plants (2·10⁵ segments) goes from 5.1 s to about 0.6 s. The remaining cost is O(population) numpy work (`unique`, `searchsorted`) and the `edges()` list.
- **Suite:** 649 passed.
- **F3 implemented: lazy MTG synchronisation.**
  - `mtg_sync = "lazy"` is the default. Components track their MTG-backed state variables on the DataStructure (`track_mtg`).
  - `flush_mtg(names=None)` writes those whose (write count, topology version) changed since their last synchronisation, through the existing `write_mtg`. It runs from:
    - `MPGDataStructure.mtg` (now a flushing property; internal code uses `_mtg`);
    - before MPG-style steps (with the running component's inputs and parameters tracked too);
    - before parameters are re-read from the MTG, for those parameters only.
  - Values read back after an MPG-style step are marked synchronised. `"after_call"` keeps the eager write; the policy is checked at declaration and at each call.
  - Fallback MTG writes (properties not keyed like the nodes) are vectorised (`_scatter`).
  - Tests: the scale-mapping tests read through `ds.mtg`; the parent-mapping write error now surfaces at the flush; 3 new tests cover lazy, only-changed and after_call.
- **DS13 seen again:** a subclass without its own steps (`EagerCounter(Counter)`) silently ran nothing. Inherited steps are dropped unless the subclass defines steps (the `inheriting` globals hack). It goes into P8.
- **Suite:** 652 passed.

## 2026-10-06 (later): P8.1, per-instance scheduling (DS13 hazard fix)

- **Choregrapher:**
  - `add_process` keys functors by `Functor.family` (`module:class qualname`, the function's qualname minus its name; graph-system trampolines take `owner.__qualname__`);
  - `schedule_of(cls)` merges the families along the MRO, a subclass's same-named step replacing its base's, categories included;
  - `add_time_and_data` stores the instance's bound groups in `instance.__dict__["_choregraphy"]`;
  - `__call__(instance=)` runs them. The `module_family=` call stays for the usage examples (it resolves a bare class name to the last bound family). The `inheriting` hack is removed.
- **Tests:** the three schedule-introspection tests use `schedule_of` / the instance schedule. New `test_per_instance_scheduling.py` (4 tests): two instances in any order, subclass steps (inherited and redefined), same-named classes from two modules (built with `exec`), a function-local class. `EagerCounter` is a plain subclass again.
- **Suite:** 656 passed.

## 2026-10-06 (later): P8.2, checkpoints (DS15), and F3 refined

- **DS15, `VariableStoreMixin.checkpoint` / `restore`:**
  - stores are written per location (`location::name` keys in the npz; object arrays in the pickle), together with the entity ids;
  - `_CHECKPOINT_STATE` (metadata, aliases, derivations, masks, write counters, versions, stored ids, MTG tracking) plus class extras (`_wiring`, `_anatomy_signature`, `last_extension`), and the MPG flushed first;
  - restore constructs the class from the JSON construction (MPG: `from_scale`, `nodes`, wiring only when the MTG has no junctions; grid: shape, dx, origin, periodic), checks the entity ids, sets the stores and the state, and drops the caches.
- **Pickling:** `LabelsConfig.__getstate__` / `__setstate__` rebuild the label-group types, the only thing that kept an MPG from pickling. Lambdas in derivations or masks get a clear `TypeError`; the Scene's emergence mask became `AllMasks`.
- **New `test_checkpoint.py`**, 6 tests:
  - a population with growth, a formula derivation and a mask: checkpoint at step 2, then 2 more steps, equals 4 uninterrupted steps bit for bit;
  - the layout, with object variables pickled;
  - restoring with an MTG given back, and a different MTG refused;
  - lambdas refused;
  - a grid continues bit for bit;
  - an anatomy keeps rewiring incrementally after restore.
- **F3 refined after the benchmark:** growth had not improved at 1000 plants, because the flush before an MPG-style step wrote every changed variable, so the write-back cost had only moved into growth. Now:
  - only the stepping component's declared variables are flushed;
  - numeric properties created by the write-back are ArrayDicts (they were plain dicts, filled one key at a time on each flush);
  - integer properties stay dicts;
  - `edges()` is cached by (topology version, vertex count, last vid).

  One scene step at 100 plants: 0.69 s → about 0.4 s.
- **Suite:** 662 passed.
- **More superlinear costs removed** (profile at 400 plants):
  - `np.unique` hashing on sorted ids became sort + neighbour comparison (`_sorted_unique`);
  - the inherit carry-over walks ancestors level by level for all new entities at once (segment mode);
  - `topology_arrays` reads the dict values in bulk (object array, None → -1), and edge-type codes by string comparison.

  Growth (4 calls) at 400 plants: 5.7 → 2.9 s.
- **Final benchmark:** 1000 plants of 2 000 segments take 3.3 s per step (6.7 s before F3 and F4); rates and states take 0.03 s (0.77 s before). Growth is still superlinear at 1000 plants (F5 in the design doc §15: O(MTG) dict reads per growth event); left open.
- **Suite:** 662 passed. **P8 done:** P1–P8 of `devplan_population_scene.md` are complete.

## 2026-10-06 (later): S1 and S2 asked; S1 implemented (bitwise consumption scatter)

- **Request:** implement S1 and `split="components"`; the anatomy repartition waits for GRANAP, and simplified anatomies are fine meanwhile.
- **S1:**
  - `_count_window` / `_window_contributions` (numba) follow `path_window`'s walk, recording each supplier and its contribution (the fraction computed in rhizodep's order);
  - `scatter_contributions` uses `np.add.at`, which is sequential in emission order, with terms `amount · contribution / total` (rhizodep's operand order);
  - `ds.order("post", convention="openalea")` rebuilds `post_order2`'s order from the MTG children lists (`reversed(plus + successor)`), cached per topology. Nodes linked across complexes fall back to the graph order.
- **Tests:** the reference is rhizodep's `actual_growth` sharing written as a loop over `post_order2`, equal bit for bit with partial and maximal overlap; the openalea order equals `post_order2`. Index order instead of visiting order differs on 6 of 110 segments (1.9e-16 relative), so the test is sensitive to the order.
- **Suite:** 665 passed.
- **S2 implemented, `@graph_system(split="components")`:**
  - `_pieces_of` (connected components, local incidences from tail/head, cached by topology version and base subgraph);
  - `_solve_restricted` for both where= and pieces;
  - `_Restriction.scatter` writes in place (O(subgraph)), and dropped edges are computed lazily (none for pieces);
  - `_capture` / `_restore` of the subgraph for adaptive integration;
  - `_saved_fields` updates the previous fields at the subgraph's nodes; in piece loops one shared dict is copied once.
- **Also fixed:** `n_edges()` no longer copies the cached edge list (it was called per piece through `_location_shape`).
- **Tests** (`test_split_components.py`, 4): pieces equal plants alone (1e-15, adaptive, slow and fast plants), split against whole within tolerances, pieces of an active subgraph, option check. The first version of the toy used rtol=1e-6, which meant about 1300 adaptive steps per call (slow, not hung).
- **Timing:** split is 1.3–1.6× slower than whole on one core at 10–100 plants (per-piece builder and FD Jacobian overhead), so the default stays "whole", deviating from §9 and reported. Thread pool not done (state on the instance; GIL).
- **Suite:** 669 passed.

## 2026-10-06 (later): audit, housekeeping, then F5

- **Audit given in the conversation:** every plan is done, apart from the anatomy repartition (A1 / Q-A4), Q29, F5, the S7 kernels deferred to the model ports, and minor scope limits.
- **Housekeeping:**
  - `devplan_datastructures.md`: DS13, DS14 and DS15 ticked with their commits;
  - `devplan.md`: W2.12 and W5.6 dropped (the per-process scene is gone); Q29 answered from your message (remove it unless numba needs it; it does not) and `specializer.py` deleted;
  - `test_conventions_doc` closes its file.
- **F5:**
  - incremental `topology_arrays` (`_extended_topology_arrays`: new vids only, children of inserted vertices re-linked, capacity doubling, full rebuild on removals);
  - lazy `_vid_to_idx` (property) and `_vid_index` (bulk sorted lookup) in `incidence_matrix`, `to_graph_view` and node → edge `_map`;
  - one id match per location in the carry-over.

  Test: incremental arrays equal a full rebuild after elongation, a bulk lateral, an inserted parent and a removal. 1000 plants: 3.3 → 2.0 s per step; growth 1.6–2.2 → 0.9–1.1 s.
- **Porting audit** (four read-only agents on rhizodep / Root-CyNAPS / Root_BRIDGES, cnwgrass / WheatFspm / adel, RhizoSoil / soiltemp / Wheat-BRIDGES / fspm-utility, GRANAP):
  - results in the new `devplan_porting.md`: gaps G1–G15, steps PT1–PT9, questions QPa–QPf plus GRANAP's;
  - checked against the code, three reported gaps are already covered: multi-output steps (`-> tuple[...]`), weighted sums (a formula link plus `sum`), and per-vertex lists (`dtype=object`).
- **Suite:** 670 passed.

## 2026-10-06 (later): porting questions answered; PT1 (tree kernels, round 2)

- **Answers recorded** in `devplan_porting.md` §5:
  - QPa: per-vertex streams;
  - QPb: MTG scales, so the shoot data model is no gap;
  - QPc: an extra unknown in the solve, which extends graph systems without changing the API;
  - QPd: the light model triangulates by itself from the MPG, so metafspm stores no triangles;
  - QPe: cmf kept as an opaque solver;
  - QPf: GRANAP first in practice, with its guidelines after the general gaps;
  - GRANAP: per-segment copies; class by diameter, distance from tip and from the collar; replaced on class change; xylem and phloem axial junctions; all nodes Compartments; at most about 100 classes.
- **PT1 design** (§6), then implementation:
  - `tree_kernels.FoldLevel` / `fold`: CSR children of the level gathered with repeat and offsets; reductions with ufunc `.at`; edge filter from the MTG edge types (None in anatomy mode);
  - `chain_gather`: group chains use one sorted search on (chain, rank) keys, edge chains take a source vid and a position;
  - `chain_recurrence`: position by position.
- **Tests** (`test_tree_kernels_round2.py`, 7), each exact against a reference loop written in the test: rhizodep's pipe model (post_order2, `<` successor, `+` emerged non-nodule laterals, 0.1 % threshold), death with the minimum time since death, Root-CyNAPS' filtered max, elongwheat's tiller cohort gather, a clamped whorl-like recurrence, a downward heading fold.
- **Cost:** a fold takes 0.2 s on 2·10⁵ segments (about 1000 levels; per-level numpy overhead, independent of the number of plants).
- **Suite:** 677 passed.
- **PT2:** `random_streams` (SplitMix64 mixing of seed, crc32 of the stream name, step, draw and id; Box-Muller normals), `ds.random`, and `DataStructureComponent.random` with per-stream step counters on the DataStructure (`_random_steps`, in `_CHECKPOINT_STATE`) and a `random_seed` attribute. Conventions: a "Random draws" section, and chained creation in MPG-style steps is plain Python drawing with `ids=new_vids`. Tests (5): order and other entities don't matter, moments, component streams advance and replay, continuation across a checkpoint, unknown distributions refused.
- **PT3 (edits and inheritance):**
  - **Found:** `remove_vertex(reparent_child=True)` on a segment raised, because its Compartment is a topological child of another scale. Once that was handled, openalea's `MTG.replace_parent` recursion (`replace_parent(old_complex, complex(new_parent))`) made complexes their own parents (0:0 … 5:5) when both share a complex, and `_full_topology_arrays` then looped forever.
  - **Fixes:** `MPG.remove_vertex` removes the owned Compartments and re-links children at the tree level within a complex; `extend_graph` adds the valid heads of stale Connections to the relink candidates; the pointer doubling is bounded and raises on cycles.
  - **Steps:** a functor output `None` is skipped; `Choregrapher.schedule_of` honours `steps_removed` along the MRO and raises on unknown names.
  - **Tests** (`test_structure_edits.py`, 5): insertion mid-chain (edges, kept and inherited values), removal with relinking, a None-returning override, `steps_removed`, unknown names.
  - Templated components are not done; a question is asked (QPg).
- **Suite:** 687 passed.
- **PT4 design and questions QPg–QPj** written in `devplan_porting.md` §7. Stopped for the answers: they add API to graph systems and choose the templating mechanism.

## 2026-10-06 (later): QPg–QPj answered; PT4 (pools, per-node boundary kinds, forcings)

- **Answers:** QPg duplication by hand (no templating); QPh pools at a scale; QPi a per-node kind variable; QPj the end of the (sub-)step for implicit solvers, the evaluation time for IVP. Your note said the plan was not saved at first; it was re-read and QPj picked up.
- **Pools:**
  - `GraphDAESpec` gains `pool_fields` / `pool_coupling`, a third packed block, `unpack_pools`, `EquationContext.pool_unknowns`, and a block sparsity (`_sparsity_with_pools`: a pool couples to its exchange nodes, the edges at them, and itself);
  - the builder takes the pools of the solved nodes' owners (`ds.owner(location)` restricted by the take), builds the coupling from the exchange set, records previous values, and adds `@pool_balance` blocks;
  - pools are written back at their scale and captured whole in adaptive steps;
  - Newton only, and an analytic Jacobian with pools is refused.
- **Boundary kinds:** `boundary_set(kinds=var)` splits members by code into the three kinds at each build. A first version took `kind=` itself as a variable name, which would have hidden typos (a UC5 test caught it).
- **Forcings:** `_solve_offset` is set per (sub-)step in `_solve_graph_system` and `_integrate_adaptive`; the `time_hook` reports `_ivp_time`; the clock advances in `Component.__call__` and is set by `Scene.run`.
- **Tests** (`test_pools_and_kinds.py`, 5):
  - pools against a hand-assembled linear solve, with conservation and each plant fed by its own collar;
  - split against whole;
  - Dirichlet and Neumann collars switched per node, then removed;
  - forcings at the 8 sub-step ends (two calls) and the implicit Euler of `d(sugar)/dt = t`;
  - the Newton-only check.
- **Suite:** 692 passed.
- **PT6 / PT5 / PT7 designs and questions QPk–QPo** written in `devplan_porting.md` §8. Stopped for the answers.

## 2026-10-06 (later): QPk–QPo answered (all as recommended; QPo: the cmf / MIMICS wrappers are to be replaced by native components later); PT6

- **`ScalarMapping`:** made per link by `Exchanges._mapping_between` when the receiver is stored at `"scalar"` (up) or all sources are (down), checked before the explicit mappings. Up: sum, mean, or weighted mean as a numerator and denominator, pooled over populations; down: broadcast, or split with totals over every receiving population. The default for an intensive value going up without a weight is the plain mean (QPk).
- **`LayerMapping`:** overlap matrix of the column and grid layer intervals; `transfer(values, from_column, aggregation)`, with "mean" / "sum" by kind; Exchanges direction "layer". `Scene(mappings=)` appends explicit mappings, or a callable of the scene.
- **Tests** (`test_environment_mappings.py`, 4): sums, pooled means, weighted means and Plant-scale totals to scalars; broadcast and a mass-weighted split of a scalar over two populations; column ← grid means and conserved totals with 0.25 m against 0.2 m layers; grid ← column broadcast and conservation.
- **Suite:** 696 passed.
- **PT5:** `Scene(forcings=, events=, stop_when=)`; `_due(model)` with `run_every` (on the iteration) and `run_when`; `spin_up(scene)` called after the exchanges are built; events popped in time order at the step start; `forcing()` falls back on the scene table set on every component as `_scene_forcings`. Tests (`test_scene_services.py`, 4): shared forcings read at each step end, a soil run every 2 steps with a single spin-up, a population run only after a time, a fertilisation event and a stop condition.
- **Suite:** 700 passed.
- **PT7:**
  - **Checkpoints:** `VariableStoreMixin.load_checkpoint` (in place, through `_load_construction`: the MPG replaces its MTG and rebuilds its index maps, a grid checks its construction, a union takes its saved `_ids`), with `restore` refactored onto `_read_checkpoint` + `_apply_checkpoint` and the `_TOPOLOGY_CACHES` dropped on both sides of the state update.
  - **Scene:** `checkpoint` writes the DataStructures in a fixed order (`_all_data_structures`), the pickled hook states (`_stateful`) and `scene.json`. `restore` builds the scene, then `load_checkpoint` loads them in place, drops the components' graph-view, restriction and piece caches and previous fields, refreshes the mappings, applies the hooks, and sets the time, iteration and remaining events; the recorder resumes under the existing header.
  - **Vector variables:** declarations carry `shape` (`VariableSpec.shape`, not MTG-backed); `register(shape=)` and the meta `shape`; validation, carry-over and union growth use the entity shape; masked step outputs with trailing dimensions; `_per_column` in `Exchanges` from the provider's declared shape (plain grid variables are 3-D, so `ndim` cannot tell); `to_dataframe` and the recorder expand the columns.
- **Tests** (`test_scene_checkpoint.py`, 5): a growing population with emergence and an external solver's state continued bit for bit; recorder resume; a mismatched scene refused; vector pools through steps, growth, exchanges, outputs and checkpoints; per-component sums.
- **Suite:** 705 passed.
- **PT8** (`test_light_component.py`): a toy CARIBU on a `UnionDataStructure` of a Wheat and a Pea population (classes made with `dataclass(type(...))` to share field definitions under distinct names). It triangulates elements as ribbons, shades with Beer's law per horizontal cell, writes `absorbed` back through the union, and runs with `run_every = 4`. Exact against the same functions applied to the concatenated geometry; outputs held between runs, updated at the next. The first versions found no shading because vertical ribbons project no area (a test issue, not an API one). No API change was needed.
- **Status written** in `devplan_porting.md` §9: everything before porting is done except PT9 (GRANAP, waiting for your guidelines).
- **Suite:** 706 passed.

## 2026-10-06 (later): your question before GRANAP

- **Warnings:** the 10 were SciPy `LinAlgWarning`s from Anderson's singular history matrix at convergence; they are silenced inside `ScipyRootSolver` for Anderson only, since the residual is checked after the call. The suite is warning-free, also with `-W default`.
- **Found:**
  - `MultiGridDataStructure` has no variable store, graph topology or checkpoint, and builds 1-D-only restriction and prolongation on flat indices;
  - Compartments are topological children of segments, so openalea's `children()`, `Sons()` and `post_order2` return them, a hazard for ported MPG-style code (B-i was deferred since WD.P);
  - coverage tools are not installed in the environment.
- **Written** as `devplan_porting.md` §10, with questions QPp–QPr.
- **Suite:** 706 passed, 0 warnings.
- **Answers to QPp–QPr:**
  - **QPq (a):** `MPG.children` / `children_iter` / `nb_children` keep the same-scale children, so openalea traversals see segments only (test `test_openalea_traversals_see_segments_only`).
  - **QPr:** `pytest-cov` installed in the env; coverage 86 % (numba bodies untraced in `tree_kernels`; legacy solver paths).
  - **QPp:** answered with a question about adaptive discretisation. I proposed cell-based octree refinement on the grid graph contract (PT10), with QPs–QPu.
  - The "parallel pieces" deferred item is explained in the conversation.
- **Suite:** 707 passed.

## 2026-10-06 (later): QPs–QPu answered (PT10 now, so that the whole DataStructure API settles); PT10

- **`AdaptiveGridDataStructure`** (new module):
  - leaves as (level, index) arrays sorted by finest corner (C order when unrefined);
  - a finest-lattice map `_leaf_of`; faces as unique (low, high) leaf pairs along each axis, with wrap faces on periodic axes (skipped when they would duplicate an internal face, as `ArrayDataStructure` does), areas from finest-face counts, distances from half sizes;
  - `refine` (split, then rebalance until 2:1), and `coarsen` (complete families only, undone if it breaks the balance);
  - carry-over through a sparse new × old overlap matrix (extensive: fractions of old cells; others: volume means), with edge variables reset;
  - stable cell ids (finest corner and level); `locate` via the finest map; `_dx` is the finest spacing, so `CrossMapping` overlap cuts at every leaf face;
  - checkpoints with `_level` / `_index`.
- **Tests** (`test_adaptive_grid.py`, 6):
  - balance and face closure;
  - conservation through refine and coarsen, and exact return;
  - a uniformly refined grid equals the regular fine grid in a diffusion solve (rtol 1e-10);
  - locally refined diffusion conserves through refinement between steps;
  - plants mapped onto refined cells (barycentre and overlap);
  - checkpoint round trip.

  Also checked by hand: unrefined adaptive and regular grids agree (cells, faces, areas, centres), periodic or not.
- **`MultiGridDataStructure` removed:** the class, 20 tests, two example scripts with their images, and the plotting helper.
- **Found:** `example_mpg_data_structure.py` was already broken before this step (QPv asked).
- **Suite:** 693 passed.

## 2026-10-05: documentation and test audit

- **Docs:**
  - `docs/index.md`, `user.md` (a full user guide) and `ref.md` (autodoc of every public module) rewritten for the current API;
  - `conventions.md`, the migration guide, the CHANGELOG and every docstring and comment in `src/` and `test/` cleaned of plan step numbers;
  - the README example rewritten (FunctionalComponent on an MPGDataStructure);
  - docstrings fixed to build as RST: Sphinx now builds with no warning from metafspm, except the six `docs/design` notes outside the toctree (QPβ);
  - module docstrings of `decorator`, `solver` and `data_api`, and the `MPGDataStructure` class docstring, describe the current API.
- **A cleaning slip, caught:** the reference-stripping regex took `next(answers)` for a plan reference in a test (two lines). Restored, and the other removals were checked to be in strings or comments only.
- **Test hygiene:**
  - one autouse Choregrapher reset in `test/conftest.py`, replacing 30 copies;
  - removed: `test_component_base`, `test_field_consensus`, `test_partial_traversal` (no assertions), `generate_anatomy_in_mtg.py` (broken), `component_api_changelog.md` (stale), the duplicate `example_translator.yaml`, and the tautological explicit-form MMS tests (`explicit=` is covered through UC1);
  - redundant `children` filters removed, now that `MPG.children` is same-scale;
  - scene doubles in `structure_tests/scene_doubles.py`;
  - subclasses instead of copied declarations (Wheat/Pea, light organs, seedling leaves);
  - `solver=` and `self.dt` in the graph-system tests;
  - a tighter lazy-sync assertion; unused imports removed.
- **New tests (72):** `test_variable_store_api.py` (16), `test_graph_system_options.py` (24), `test_scene_options.py` (15), `test_mapping_options.py` (10), plus small additions.
- **Bugs they found, fixed:**
  - `forcing()` took `(times, values)` tuples for Series;
  - `to_nested()` refused long-form links;
  - a same-name link stating only its scales derived itself;
  - anatomy-mode populations got the segment graph on top of their anatomies, with no way to wire junctions (now a `wiring` class attribute);
  - scene checkpoints dropped events timed between the last step and the checkpoint;
  - `CompositeModel` turned Python translators into dicts, losing link options.
- **Checked:** the audit's claim that `_ivp_time` is never set is wrong. The graph system's `time_hook` sets it at each IVP evaluation.
- **Questions QPw–QPζ** added to `devplan_porting.md` §13: legacy APIs, the solver layer, the time-term convention, `@boundary_condition`, the old example files, `docs/design`, the example scripts, the test layout, the UC rewrites, `legacy_functor.py`.
- **Suite:** 754 passed.

## 2026-10-05 (later): QPw–QPζ answered; QPy detailed; the decided items

- **QPy:** you asked for more detail. `devplan_porting.md` §14 now explains the current meaning of graph-system equations for each solver, and proposes two forms: the residual form for the Newton family, and a new `@node_rate` form for every solver. `implicit_euler` would become an alias of `newton, transient=True`. Waiting for your answer.
- **QPζ:** `solve/legacy_functor.py` → `solve/functor.py`.
- **QPβ:** `docs/design/*` → `dev/design/`; the migration guide → `docs/migration.md`. Sphinx builds with no metafspm warning, and nothing is left outside the toctree.
- **QPα:** `composite_wrapper_example.py` and `rhizosoil_component_example.py` are rewritten as current-API sketches (a population model, an environment model). `rhizosoil_core_model.py` and `logger_api_reference.py` move to `provide_usage_examples/legacy/` with a README. `light_component_example.py` is removed.
- **QPz:** `@boundary_condition` emits a DeprecationWarning. UC4 and the active-subgraph test use `boundary_set`; UC1 keeps the decorator until QPε, with the warning filtered there.
- **QPw, removed:**
  - `LegacyMPGDataStructure` and `from_legacy`, with their tests and example scripts;
  - `GraphView.from_mtg_subset` and `_array_at_scale`; its test, which asserted the method's bug, is replaced by a `to_graph_view()` test;
  - `set_node_property`, `set_edge_property`, `add_field`, `_get_field` / `_set_field` and `_set_or_register`; the abstract `inject_state` / `extract_state` go through `get` / `set`;
  - `MPG.graph`, `integrate_at_scale` and `average_at_scale`, with their tests;
  - the Choregrapher's `build_schedule`, `add_schedule` and by-name runs;
  - `CompositeModel.declare_data`, and the unused DataStructure keywords of `declare_data_and_couple_components`;
  - `Translator.inputs_outputs`.

  The `props` view and `_last_graph_system` wait for QPε.
- **QPx:** `system_specs` and the solver internals are documented as internal and removed from the API reference. The solver unit tests stay, in `test/graph_systems/solver_layer/`.
- **QPγ:** the array example script becomes `test_examples.py` (the soil column, with conservation and smoothing asserted, and the plant graph plots, written to `tmp_path`).
- **QPδ:**
  - tests are organised by feature, with the helpers in `test/helpers/`; `conftest.py` puts only that folder on the path, and the 35 per-file `sys.path` insertions are removed;
  - helpers imported from test modules move to `helpers/plants.py`: `branched_root_system`, `population`, `seedling_ds`, `grow_root`;
  - `test_datastructure_prerequisites.py` is split by subject (graph view, variable store, grid topology, scale mapping, update_topology);
  - the two composite files are merged into `coupling/test_composite_model.py`.
- **Slips, caught by the suite:**
  - a method-removal regex also cut the header of `DataStructurePropsView`, restored;
  - a failed `git rm` left a merged test file in place, so its tests ran twice once, then removed;
  - my plant sketch called a method that does not exist, corrected.
- **Suite:** 728 passed, no warnings.

## 2026-10-05 (evening): `@boundary_condition` kept; QPy and QPε

- **`@boundary_condition` (your question):** its methods already took their arguments by name from the DataStructure, coupled variables included, read at each solve. The tests only used `self` attributes and constants, so nothing showed it.
  - Following your choice (option 2), the decorator stays, for conditions given by equations.
  - Neumann values are now inflows (`boundary_set`'s sign).
  - `select=` takes a dict, a variable, a mask name or a callable; `boundary_set` also accepts mask names.
  - Three tests cover it: a coupled inflow changed between calls, an exchange equation of the unknown checked against a hand solve, and a Dirichlet condition on a mask following a coupled value.
- **Found:** placed numeric parameters have two copies (QPη, asked). `model.c_dirichlet = 2.0` sets an attribute while the DataStructure keeps its own value.
- **QPy:**
  - `@node_rate` added; `implicit_euler` deprecated as an alias of `newton`, transient; explicit and IVP solvers refuse residual forms and Dirichlet conditions.
  - Explicit Euler now recovers the fluxes at the current state before stepping. It used the previous step's, zero at the start, which a forward-Euler test caught.
  - Tests in `test_rate_form.py` (9).
- **QPε:**
  - UC1 and UC1-organ (1880 lines) became one parametrised file of 40 tests through the public API, with shared components in `helpers/nitrogen.py`. The `implicit_euler` tests asserting the spurious edge term `q(1 + 1/dt)` are replaced by the rate form solved three ways.
  - UC2 now checks its Jacobian by one Newton step (a deliberately wrong Jacobian fails it, checked).
  - UC3 uses boundary sets instead of hand-set ports; UC4 checks its balances by hand.
- **QPw finished:** `FunctionalComponent.props`, `DataStructurePropsView`, `_last_graph_system`, `_last_graph_solution` and `_make_compat_graph_system` are removed.
- **Docs:** the user guide explains the two forms; the migration guide, conventions and API reference follow the new paths and API.
- **Suite:** 743 passed, no warnings.

## 2026-10-06: QPη and QPθ; the SPAC example designed

- **QPη:** every numeric parameter now goes through the DataStructure, placed ones included. `model.k = v` writes every entity, and reading `self.k` in a step raises. One helper read `self.exudation_rate` in a step and now takes it as an argument.
- **QPθ:** hand-set boundary ports are removed (component, decorator, and their test).
- **The suite took 16 s instead of 9 s.** It was as slow at an older commit; it is the machine on battery, as you said.
- **The example of water flow in the soil–plant–atmosphere continuum** is designed in `devplan_spac_example.md`: a feature map, the model, the code layout, and questions Q1–Q10 (conductance and length, component kinds, the liquid–gas scaling, the plant–soil fixed point, anatomies, plant size, population, atmosphere forcing, location, plots). Waiting for your answers.
- **Suite:** 742 passed.

## 2026-10-06 (later): the SPAC example

- **C1 and C2 answered:**
  - radial conductance per anatomical edge is k_s · L_segment; axial conductance is k_axial / L;
  - soil face conductance is K · A / d, with K varying between voxels (harmonic mean across a face).
- **Framework changes, each with a test:**
  - `"node"` on grids is the cells;
  - Connection-scale variables in anatomy mode are the Connections' own properties;
  - explicit CrossMappings replace the inferred ones;
  - Translator objects in `CompositeModel`;
  - grid `dx`;
  - scalars from one-value arrays;
  - `@graph_output("node")` on grids;
  - leaf anatomy labels.
- **The example:**
  - `examples/soil_plant_atmosphere/` with one transport class on plants and soil, structural components computing the conductances (`SeedlingStructure` builds 33-segment seedlings with root, stem and leaf anatomies), an atmosphere of environment scalars (Ψ_air, the vapour factor), and two scenes;
  - plots: SubOrgan graph, full anatomy graph, one anatomy per organ type, soil slice, top view;
  - a smoke test checks convergence and closed water balances.
- **Found while building it:** Newton's absolute tolerance stopped the plant solve at the initial guess in m³ s⁻¹ units, silently. The example uses mm³, and E1 is asked.
- **Other open points in the plan:** E2 (stomatal closure, so that the competition between plants shows in their fluxes), E3 (properties on wired junctions).
- **Suite:** 751 passed.
- **Atmosphere removed (your request):** the air is now a constant input, `air_water_potential` and `vapour_factor` as parameters defaulting to their values at 50 % RH and 20 °C. Results are unchanged; the smoke test and the figures were rerun.
- **Split declarations (your request):** plant and soil transports and structures each declare their own variables (plant at Compartment / Connection scales, MTG-backed; soil at cell / edge), on a shared flow-equations mixin. The edge law is explicit. The results are identical; the MTG write-back of the plant variables was checked.
- **Selective computation (your question):** `@graph_output` gets `select=` (the boundary-set forms, zero elsewhere; tested). The example's evaporation and root uptake now use it instead of multiplying by a flag. Its structures define masks (root surface, stomatal cavities, soil surface), and the conductance steps run on them only (`where=`). The results are unchanged.
- **`filters=` everywhere (devplan_selection_api.md, your answers: yes to all):**
  - one `Filters` resolver on DataStructure masks (dict with any comparisons, label names and coarser-scale keys; mask name; callable);
  - every decorator migrated, `select=` / `where=` removed, `include_inactive=` for steps;
  - 9 new tests;
  - the example's transports are declared separately, without the mixin.

  One bug of my own was caught: the functor read the mask dict before the first mask existed. Results unchanged; suite 763 passed.
- **Example boundaries as decorators (your request):** the air exchange (plant and soil) and the water table are `@boundary_condition` equations instead of boundary sets. Results identical.
- **Seedling generator externalised (your request):** the architecture, anatomies and junctions (and their codes) moved from `components.py` to `seedling.py`; `SeedlingStructure.initiate_plant` calls it. Results identical.
- **Root architecture (your request):** three first-order roots emerging 15° below the horizontal at the collar and bending towards the vertical by a gravitropism coefficient (0.35 per segment), with a seeded tortuosity per plant; laterals with a weaker gravitropism. Converges as before (5 and 6 steps), with balances closed.
- **Pivot root (your request):** the first of the three first-order roots is a pivot (80° below the horizontal, gravitropism 0.8, 8 segments, down to about 20 cm); the two others spread along ±x and bend down. Converges in 5 and 6 steps.
- **Longer laterals (your request):** 4 segments of 2 cm (8 cm, from 4.5 cm). Converges in 5 and 6 steps.
- **Population side views (your request):** the side plots (SubOrgan segments, anatomy graph, soil slice) show only one planting row (row=0), seen along the row in the y–z plane.
- **Upscaling series (your request):** `upscaling.py` derives the water potential at SubOrgan, Organ, Axis and Plant, each the mean of the scale below (one `derive` per scale); `plotting.upscaling_series` draws one graph per scale on one colour scale (the first row for the population). The smoke test checks the means and the five figures.
- **Top view (your request):** shoot segments drawn last and twice as thick as the roots (2.4 vs 1.2).
- **Upscaling over every MPG scale (your remark):** Phytomer and GrowthUnit added between Organ and Axis (7 graphs). Layer and Cell hold no entity in this generator (the Compartments are directly below the segments).
- **Soil-limited uptake (your request, option 2):** soil conductivities divided by about 3.3 (topsoil 1.5, subsoil 0.6). The horizontal anomaly of soil Ψ grows about 3× (population 0.019 → 0.052 MPa), but the vertical gradient grows as much (surface −1.4 → −4.4 MPa), so the slices still look layered. Leaf Ψ min −2.7 (one plant) and −5.8 MPa (population); converges in 6 and 9 steps.
- **Soil anomaly view (your request):** `soil_slice(anomaly=True)` shows Ψ minus its layer mean on a diverging scale centred on 0; written as `soil_slice_anomaly.png` by both scenes. Root depletion columns (to about 0.03 MPa) are now visible.
- **Upscaling figure (your requests):** each scale drawn as its own graph (entities as nodes at the centroid of their segments, edges = links between entities in the solver graph, coloured by the mean of their nodes), and all seven scales side by side in one figure, `upscaling.png`, with a shared colour bar. While doing it, a per-scale plotting version hung (fixed by the rewrite).
- **Upscaling figure in one plot (your request):** the seven scale graphs drawn in a single plot, each shifted by the same horizontal distance, labelled below, with a colour bar as tall as the plot.
- **Time step at construction (your question):** components did not take the scene's time step: `dt` came from a class attribute or fell back to 1. `time_step=` is now a constructor argument (default: class attribute, else the simulation step; exempted from the couplability check), the example and the sketches pass it, and a test covers the three cases.
- **Sub-stepping by the time step (your go-ahead):** `sub_time_step` defaults to the component's `time_step`; the choregrapher runs the schedule `simulation step / time_step` times, advancing the component's clock per sub-step (forcings read at each sub-step's end), and construction raises when the ratio is not a whole number ≥ 1. Two tests (4 runs of 900 s in 3600 s, the last ending at 3600 s; 1000 s and 7200 s refused); two grid tests got a simulation-step fixture. Found: a step without arguments is an iterating step, its return discarded. Suite 766 passed.
- **Scenarios applied by the Scene (your question, then go-ahead):** models swallowed `**scenario`. Only numeric plant parameters reached components (through `apply_plant_scenarios`); non-numeric ones and environment parameters were silently dropped, and so were typos. `apply_model_scenario` now sets every entry on the components as a constructor keyword would (checked: the DataStructure value and the dataclass field match a hand-built component), leaving per-plant values to `apply_plant_scenarios`. Model-named arguments and keys read by initiators (recorded during `build_population`) are accepted, anything else raises, and a parameter not stored per plant must be equal across plants. Placed in the Scene rather than `declare_data_and_couple_components`, since plant models need not be `CompositeModel`s. Found: a parameter declared at an MPG scale on a grid is not registered, and its constructor value is ignored. 6 tests, suite 772 passed, example unchanged.
- **`**scenario` removed from model signatures (your request):** models name only their own arguments (the Scene passes them just those, and applies the rest to the components). Example, test doubles, usage examples, benchmark and the contracts in docs and `scene.py` updated; the legacy migration snippet kept. Suite 772 passed.
- **Finer soil voxels hung (your report):** at voxel 0.0125 the soil Newton step spent 22 s in SuperLU. The cause is the system of 8192 cell Ψ plus 24320 edge fluxes: cell rows hold only fluxes (7680 zero diagonals), so COLAMD pivoting gave 25.7 M fill-in. The linear step now eliminates edge unknowns whose Jacobian block is diagonal (an exact Schur complement, checked to 1e-13), and other systems use MMD_AT_PLUS_A. Times: 0.0125 m runs in 2.3 s in total (7 steps); 0.00625 m (65k cells) in 100 s, 92 s of which is the direct solve of the cell system (6.6 s each). Results at the default voxel are unchanged. 2 tests, suite 774 passed. Open: an iterative solver for larger grids.
- **Second lateral on the pivot (your request):** `pivot_lateral_ranks=(2, 5)`, on alternate sides; 48 segments and 180 Compartments per plant. Converges in 7 steps (one plant) and 10 (population); balances closed.
- **Soil ΔΨ behind the side views (your request):** `plant_segments` and `plant_anatomy` take `soil=`. The soil's Ψ minus its layer mean, in the plants' plane (the planting row, or the collar's y), is drawn behind the architecture on a diverging RdBu scale centred on 0 (colour bar under the plot), with the plant on viridis (right). The slice selection is shared with `soil_slice`. Figures and README regenerated; README counts (and the stale radial-k values) corrected.
- **Separate figures (your request):** the side views exist alone (`plant_segments.png`, `plant_anatomy.png`) and over the soil ΔΨ (`*_soil.png`); README and smoke test updated. Your answer: the 2.5 cm default stays (restored by you); figures regenerated at 2.5 cm. No iterative solver yet.
- **Adaptive soil (your request):** plan in `devplan_adaptive_soil.md`. Flux intensity is argued not to be the right refinement metric (FV resolves linear profiles exactly; error follows the curvature, i.e. the sink density); questions Q1–Q4. Found: `CrossMapping` does not rebuild after the target grid's topology changes.
