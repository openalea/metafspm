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
