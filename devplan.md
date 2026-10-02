# metafspm dev plan

This is the working plan, and **you can edit it directly between sessions.** Claude re-reads it at the start of each session and merges with your edits instead of overwriting them. Outcomes are logged in `devlog.md`.

**Conventions**
- Task status: `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` dropped · `[?]` blocked on a decision (see *Open questions*).
- Task IDs are stable (`W2.3` …). Refer to them in commits and devlog entries.
- **xfail-first**: write a test that shows the current bug and mark it `xfail(strict=True, reason=...)`. Then fix the source and remove the marker in the same commit.
- **Test tags**: `[contract]` tests pin behaviour that must survive the WD refactor (translator semantics, scene protocol, numeric results). Per Q14 there is no `[legacy]` tier any more; tests of MTG-props internals are not written.
- Source references are against `release2026` at `557e4a8` plus the uncommitted W0.1 port, unless noted.

## Next steps (ordered)

Each phase ends green and gets its own commit.

1. ~~Commit W0.1, then W0.3–W0.5, W4.6–W4.8~~ **done 2026-09-29** (`251636d`, `4bbc834`).
2. ~~W1 infrastructure~~ **done 2026-09-29** (`409a0e9`).
3. ~~W2–W4 `[contract]` tests + xfail-first fixes~~ **done 2026-09-29** (`1f6a416`). Coverage: composite_wrapper 96.5 %, scene_wrapper 92.6 %.
4. ~~W5 end-to-end `play_Orchestra`~~ **done 2026-09-29** (`062c61c`). Three scene protocol bugs fixed; 9 slow tests; 0 failures in 30 runs under full CPU load.
5. ~~WD.0 translator schema + WD.1 design note~~ **accepted 2026-09-29** (`docs/design/coupling_through_datastructures.md`, Q20–Q26 decided).
6. ~~WD.P prerequisites~~ **done 2026-09-29** (`42e00eb`). B-a…B-h are fixed. B-i is deferred to WD.3, where the scale operators must exclude anchors and Compartments.
7. **WD.2–WD.6 implementation** (WD.3 done `b671415`, WD.2 done `d7546f3`, WD.0 implemented). WD.4 done (`ee7bbc4`), WD.5a done (`26433fc`). Epic WD complete: the legacy path is removed (`b3c11c8`), the UC tests are migrated (`8f6f055`), and the backlog is closed except B11 (downstream, out of scope). live DataStructure reading, links on the DataStructure, a Coupler across DataStructures, and a scene transport sized from the handshake. Retarget the W doubles to MPG and 3-D grid and re-run the `[contract]` suite unchanged.
8. **WD.7–WD.9 downstream migration**: guide, Logger adapter, `assert_component_couplable` (W6.2).

**See also `devplan_datastructures.md`:** the DataStructure API stabilisation plan (DS1–DS21, decisions D1–D16), with its step 1 design note `docs/design/datastructure_contract.md`.

## Decisions log

| Date | Q | Decision |
|---|---|---|
| 2026-09-29 | Q1 | `publish_WB` is the reference for the wrapper API. Port its 3 wrapper commits (done: W0.1). |
| 2026-09-29 | Q2 | **No backward-compatibility shims.** Downstream packages must pass explicit paths and imports (`openalea.metafspm.coupling.composite_wrapper`, `.scene.scene_wrapper`, `.data_structure.arraydict`, `.coupling.choregrapher`). |
| 2026-09-29 | Q3 | **Coupling must be dynamic, through the variables of `DataStructure` derivatives**, no longer through `g.properties()` dicts. This adds design epic **WD**. |
| 2026-09-29 | Q4 | Keep `eval` of string factors for now. The translator format is reopened as **WD.0** (scale-aware links, and YAML vs Python); see Q4b. |
| 2026-09-29 | Q5 | The `models_data_required` cache going stale is **not intended**, so it gets fixed (W2.9). |
| 2026-09-29 | Q6 | `recursive_reload` existed to fully reset the Choregrapher singleton between tests. It is **replaced by an in-place `Choregrapher.reset()`** (W1.6), and `recursive_reload` is removed (W2.11). |
| 2026-09-29 | Q7 | Keeping one core free is intentional. **The soil and light workers must also be pinned to dedicated cores** (W4.7). |
| 2026-09-29 | Q8 | The meteo file comes from `light_scenario` (W4.6). |
| 2026-09-29 | Q9 | On platforms without `cpu_affinity` (macOS), there is **no pinning**: it becomes a no-op (W4.8). |
| 2026-09-29 | Q10 | Reference files were added to `test/provide_usage_examples/`: RhizoSoil composite, SoilModel core, LightModel, the WheatBRIDGES translator, and `logger_api_reference.py` (Logger). |
| 2026-09-29 | Q11 | The dropped ×72 conversions went unnoticed. You fixed it downstream by renaming the soil-side receivers with a `_massic` suffix (they are different quantities). **The code still silently ignores a same-name factor ≠ 1**, so this gets fixed in metafspm (W2.5). |
| 2026-09-29 | Q12 | Queue timeouts and poison pills are **out of scope for now**; the blocking `get()` calls stay as they are. |
| 2026-09-29 | Q13 | **Components read DataStructure arrays live.** The props snapshot is removed (WD.2). |
| 2026-09-29 | Q15 | The Logger has to change anyway, but it isn't this package's focus. **Scope: make the xarray and csv writing work with `DataStructure` and its subclasses** (WD.9). No full reimplementation. |
| 2026-09-29 | Q16 | **Start the soil on `ArrayDataStructure`** (3-D); move to `MultiGridDataStructure` later. The axis order is still open (Q16b). |
| 2026-09-29 | Q16b | **The canonical axis order is `(x, y, z)`** for the soil `ArrayDataStructure`. The Coupler converts from the RhizoSoil `(y, z, x)` voxel indexing during migration (WD.P, WD.1). |
| 2026-09-29 | Q17 | **The light model sends an initial reply in `__init__`, like the soil does** (W5.0). |
| 2026-09-29 | Q18 | A worker failure makes `clean_exit` False: plant and soil workers exit with code 1, and the orchestrator checks the exit codes (`a3a9877`). |
| 2026-09-29 | Q19 | `plant_model_frequency` is an argument of `play_Orchestra`, uniform by default, so it is implicit for one model (`a3a9877`). |
| 2026-09-29 | WD.0/WD.1 | **Design note accepted:** `docs/design/coupling_through_datastructures.md`, strategy and §11 recommendations. |
| 2026-09-29 | Q20 | Step functions receive arrays; a per-element loop only by opt-in (`vectorized=False`). |
| 2026-09-29 | Q21 | A framework-managed `self.previous(fn)` replaces the user-managed `_previous_fields`, with a one-release deprecation. |
| 2026-09-29 | Q22 | Non-float variables go in an object store on the DS, couplable by identity or alias only. |
| 2026-09-29 | Q23 | Plant-scale values move from "vertex 1" to a scalar store; the Logger CSV is adapted in WD.9. |
| 2026-09-29 | Q24 | On-grow policy per variable, default `"default"`. The growth model can still overwrite with specific initialisation from parent states, for concentrations and extensive quantities. |
| 2026-09-29 | Q25 | The growth model registers the segment geometry `x1..z2` as plant DS node variables. The locator's variable names are configurable. |
| 2026-09-29 | Q26 | The soil handshake is derived from the Couplers, and identity links are optional in Python translators. **The tests still write identity links out explicitly**, even in Python, for readability. |
| 2026-09-29 | Q27 | Option (a): `play_Orchestra(handshake_shape=...)`, defaulting to today's `(35, 20000)`. Migrated models pass `Transport.from_translator(...).shape`, and the default goes away at WD.7. |
| 2026-09-29 | Q28 | Keep the legacy props path, deprecated; remove it in one commit at the end of WD.7. The legacy scene tests stay as the legacy contract, and the DS scene doubles get their own contract tests with the same numbers. |
| 2026-09-29 | Q14 | **Full switch, no dual support.** Plants use `MPGDataStructure` and the soil uses the 3-D grid `DataStructure` already on this branch (WD). |

---

## Epic W: coverage for `scene_wrapper` and `composite_wrapper`

**Goal.** The coupling layer used by the real FSPM packages (rootbridges, rootcynaps, cnwgrass, rhizosoil, grassbridges) becomes a tested contract. It covers three things:
- declared components can be coupled through the translator;
- plant composites exchange data with the soil and light models;
- `play_Orchestra` runs a scene end to end.

**Baseline.** Line coverage is 10.1 % for `composite_wrapper.py` and 8.9 % for `scene_wrapper.py`. The one existing test has no assert.

**Target.** At least 90 % line coverage on both files. Every public function gets at least one behavioural assertion. An end-to-end scene test runs in CI.

**Constraint.** No downstream package is installed in the `metafspm` env, and CI runs `cd test && pytest`. Everything uses in-repo doubles that reproduce the interfaces in `test/provide_usage_examples/`.

**Rule from Q14.** Assert only on observable behaviour: translator results, values that reach the receiver, buffer contents, and messages. Don't assert on how props dicts are wired internally. That way the same tests stay valid after WD, once the doubles are switched to DataStructures.

### Interface contract extracted from the examples

**Translator YAML** (`example_translator.yaml`):
- Layout: `translator[receiver][provider][receiver_var] = {provider_var: factor}`.
- `factor` is a number, or a string arithmetic expression such as `"12 * 6"` or `0.000001 / 3600` that gets `eval`'d.
- `{}` means no link.
- Several `provider_var` keys mean the receiver variable is their weighted sum.
- The soil component is keyed as `"SoilModel"`. That name is hard-coded in `couple_components`.

**Plant composite** (`GrassBRIDGES`). Calls `declare_data_and_couple_components(root=, shoot=, translator_path=, components=)`, then uses:
- `self.components`;
- `self.plant_side_soil_inputs`: 7 fixed names (`vertex_index, x1, x2, y1, y2, z1, z2`) plus every provider-variable name under `translator["SoilModel"]`;
- `self.soil_outputs`;
- `soil_handshake = {v: k for k, v in enumerate(plant_side_soil_inputs + soil_outputs)}`.

It also calls `apply_input_tables` and `pull_available_inputs` on its components.

**Soil composite** (`RhizoSoil`):
1. `__init__` blocks until one message from each plant has arrived on `queue_plants_to_soil`.
2. For each message:
   - `couple_current_with_components_list(receiver=soil, components=msg["carried_components"], translator=..., subcategory=msg["model_name"])`, which fills `soil.pullable_inputs[model_name]`;
   - `get_component_inputs_outputs(..., target_name="SoilModel", names_for_others=False)`;
   - `soil.get_from_plant(msg)`, then `soil.send_to_plant(msg, soil_outputs)`;
   - `queues_soil_to_plants[plant_id].put("finished")`.
3. `run()` calls `apply_input_tables` on voxels, then `soil(queue_plants_to_soil=, queues_soil_to_plants=, soil_outputs=)`. That call gathers every plant, applies the fluxes to voxels, steps the soil, sends the states back, and puts `"finished"`.

**Plant ↔ soil wire:**
- one `SharedMemory` per plant, named after `plant_id`, holding a float64 buffer of shape `(35, 20000)`;
- rows follow `handshake[var]`;
- columns are positional, in ArrayDict sorted-vid order;
- the valid-column mask is `buf[hs["vertex_index"]] >= 1`.

The two directions:
- **Plant → soil (extensive)**: the soil maps each vertex to its voxel with the barycenter of `x1..z2` (`flip_z`, periodic in x and y). It then does `np.add.at(voxels[name], idx, Σ factor·buf[hs[src]])`.
- **Soil → plant (intensive)**: `buf[hs[name], mask] = voxels[name][idx]`. The plant then does `root_props[name].scatter(vertices, buf[row][mask])`. **The objects are kept the same so that aliases stay valid.**

The messages:
- first message: `{"plant_id", "model_name", "carried_components", "handshake"}`;
- later messages: the same, without `carried_components`.

**Plant ↔ light wire.** Each plant puts `{"plant_id", "data": {"coordinates", "rotation", "scene": <cscene triangles per vid>, "class_name": {vid: str}}}`. The light model waits for `len(queues_light_to_plants)` messages, then replies per plant with `{var: {vid: value}}`. Here `var` is `PARa`, optionally `Erel` and `*_prim`, and the vids are remapped through `indexer`.

**Constructors called by `play_Orchestra`:**
- Plant: `plant_model(queues_soil_to_plants=, queue_plants_to_soil=, queues_light_to_plants=, queue_plants_to_light=, name=, time_step=, coordinates=, rotation=, translator_path=, **scenario)`.
- Soil: `soil_model(queues_soil_to_plants=, queue_plants_to_soil=, time_step=, scene_xrange=, scene_yrange=, translator_path=, **scenario)`.
- Light: `light_model(scene_xrange=, scene_yrange=, meteo=<DataFrame>, **scenario)`, then `.run(queues_light_to_plants=, queue_plants_to_light=)`.
- Logger (`openalea.fspm.utility.writer.logging.Logger`, see `logger_api_reference.py`): `logger_class(model_instance=, components=, outputs_dirpath=, time_step_in_hours=, logging_period_in_hours=, echo=, **log_settings)`, which exposes `__call__`, `run_and_monitor_model_step` and `stop`. It reads `model_instance.data_structures` and **accepts only an exact `openalea.mtg.MTG` (`root`/`shoot`) or a `dict` (`soil`)**; it rejects anything else, MPG subclasses included (see WD.9). It also reads `props['root']['struct_mass']`, `props['soil']['length']` / `['soil_temperature']`, `fields(component)` metadata, and `model_instance.shoot` if present.

### W0: Align `release2026` with the reference API

- [x] W0.1 Ported `publish_WB` commits `717172f` (translator_path is the full YAML path), `dba0143` (`sowing_depth` kwarg) and `5bd0645` (print). After the port, both files match `publish_WB` apart from import paths. Suite: 312 passed. Committed as `251636d`.
- [-] W0.2 Backward-compatibility import shims: dropped per Q2.
- [x] W0.3 Drop `mtg_to_arraydict` from the API. The DataStructures replace it (Q14). Record the old → new import mapping for downstream packages in `CHANGELOG.md`. Done in `4bbc834` (CHANGELOG table).
- [x] W0.4 Make `debug_runs` and `poll_interval` (the current `sleep(10)`) keyword arguments of `play_Orchestra`. Done in `4bbc834`.
- [x] W0.5 Update `test/provide_usage_examples/` to the new explicit import paths. They stay reference only and are not collected. Done. The files are still untracked, and `light_scenario` with `meteo` has been added to the scene example.

### W1: Test infrastructure

- [x] W1.1 Create `test/wrappers_tests/` with `conftest.py` (fixtures only) and `doubles.py`. `doubles.py` is an importable module with its classes at module level, so they can be pickled under `spawn`. Done in `409a0e9`.
- [x] W1.2 Dummy components with known one-step values. The set is `Carbon`, `Nitrogen` and a voxel `SoilModel` double. Together they must cover these translator cases: Done: `RootCarbon`, `RootNitrogen` (legacy `Component` on MTG props) and a voxel `SoilModel`, with every link kind in `doubles.TRANSLATOR`.
  - identity;
  - alias with a different name;
  - numeric factor;
  - string expression (`"12 * 6"`);
  - same name with a factor ≠ 1;
  - a `_massic` rename with a factor (the post-Q11 real case);
  - multi-source sum;
  - an empty `{}`.
- [x] W1.3 Tiny root fixture (3–5 vertices) with `struct_mass`, `living_struct_mass`, `vertex_index` and `x1..z2` placed in known voxels. Build it as a plain MTG now; WD.6 retargets it to `MPGDataStructure`. Done: `make_root_mtg` gives 3 segments at depths 0.02/0.04/0.06 m, which fall in voxel layers 0/0/1.
- [x] W1.4 Translator fixtures under `tmp_path`: Done: `translator_path` fixture. The trimmed WheatBRIDGES copy is left for W2.3.
  - one minimal file per case in W1.2;
  - a trimmed copy of the current `example_translator.yaml`, which includes the `_massic` entries.
- [x] W1.5 Scene doubles that follow the protocol exactly: Done. There is also a threaded `in_process_scene` fixture: one plant, with soil and light in threads.
  - `FakePlant(CompositeModel)`, the GrassBRIDGES shape without Adel/Caribu;
  - `FakeSoil(CompositeModel)`, the RhizoSoil shape: voxel grid, barycenter mapping, `np.add.at`, gather;
  - `FakeLight`;
  - `FakeLogger`, which follows the Logger call surface and records its calls to `outputs_dirpath/calls.txt`.
- [x] W1.6 Add `Choregrapher.reset()`. It clears the scheduling state **in place** and keeps the same instance. Replacing `_instance` doesn't work, because `Component.choregrapher` is bound when the class is created (devlog §6: the new singleton silently no-ops). Add an autouse fixture that calls it in the wrapper tests. Migrate `test_choregrapher_reinit.py` to it. Done: `Choregrapher.reset()`, autouse in `wrappers_tests`, the UC1 files and `test_choregrapher_reinit.py`.
- [x] W1.7 A `slow` marker in pyproject. Done, in pyproject and again in `wrappers_tests/conftest.py`, because CI runs without pyproject.

### W2: `composite_wrapper` unit tests

- [x] W2.1 `[contract]` `open_or_create_translator`: Done in `1f6a416`.
  - it loads the full path;
  - a missing file triggers the interactive builder (monkeypatch `input`), and the result round-trips through the YAML file;
  - a directory path is rejected.
- [x] W2.2 `[contract]` `translator_matrix_builder` with scripted `input()` answers. Cover "0 for None", an empty answer (same name), `name*factor`, a `;` multi-source answer, and an out-of-range index. Done in `1f6a416`.
- [x] W2.3 `[contract]` `get_component_inputs_outputs` with `names_for_others` set to True and to False. Assert exact sets on the trimmed WheatBRIDGES translator. Done in `1f6a416`. Uses `test/inputs/wheatbridges_coupling_translator.yaml`, a tracked copy of the reference translator.
- [x] W2.4 `[contract]` Translator semantics, one case per W1.2 link kind. Assert **the value the receiver sees**: identity and alias give the source value; a factor or expression gives factor · source; multi-source gives the weighted sum. Also cover the per-`model_name` subcategory, and the case where it already exists. Done in `1f6a416`.
- [x] W2.5 xfail-first: a **same-name link with a factor ≠ 1 is ignored** (`composite_wrapper.py:162-164`, `if source_name == name: continue`). The real translator no longer hits this since the `_massic` rename (Q11), but the code bug remains. Fix: skip only when `factor == 1`, and otherwise register a derived link. Done in `1f6a416`. Fix: the factor is applied across data structures (soil subcategory). Within one data structure it raises `ValueError`, since the variable would be converted into itself on every pull.
- [x] W2.6 xfail-first: `composite_wrapper.py:184` uses stale loop variables when a link is multi-source and has a subcategory. Fix: `= source_variables`. Done in `1f6a416`.
- [-] W2.7 Tests of `couple_components` props-alias internals: dropped per Q14. The end-to-end values are covered by W2.4 and W3.
- [-] W2.8 Tests of `declare_data` internals: dropped per Q14, covered through W3.
- [x] W2.9 `[contract]` `apply_input_tables`: Done in `1f6a416`. The selection cache is now keyed on the target components and the table variables.
  - `None` is a no-op;
  - a `voxels` target is filled;
  - a `props` target gets vertex 1;
  - any other target raises `TypeError`;
  - both variable-selection rules are exercised.

  Also xfail-first (Q5): the stale `models_data_required` cache. The fix is to compute the selection at coupling time, or key it on the `to` components.
- [x] W2.10 xfail-first: `get_documentation`, `documentation` and `inputs` crash on any current component (`None.__format__`, `KeyError 'variable_type'`, a column chosen by position). Fix them. Done in `1f6a416`.
- [x] W2.11 Remove `recursive_reload` (Q6). Point `test/utils.py::deep_reload_package` users at W1.6 (see B3). Done in `1f6a416`.
- [~] W2.12 xfail-first **handshake capacity**: the real translator fills exactly the hard-coded 35 rows (25 plant-side + 10 soil outputs), and the column count is capped at 20000 vertices. The interim fix is a height derived from `len(handshake)` and passed in the first message. The full fix is WD.5. **Partly done (`1f6a416`):** `scene_wrapper.HANDSHAKE_SHAPE` is the single constant (the doubles import it), `CompositeModel.soil_handshake_inputs()` was extracted, and a test pins that the WheatBRIDGES translator fills all 35 rows. Dynamic sizing is left to WD.5.

### W3: Composite contract test, in-process

- [x] W3.1 `[contract]` `FakePlant` + `FakeSoil` with `queue.Queue` and one real `SharedMemory`. Check: Done in `1f6a416`.
  - the first message carries `carried_components`, and later ones don't;
  - the handshake rows;
  - the plant values sit in their rows;
  - the soil sends `"finished"`.
- [x] W3.2 `[contract]` Plant → soil: the voxel sums equal Σ factor · plant values in the expected voxels, including the `_massic` ×72 case. Soil → plant: every valid vertex receives its voxel value, and the plant-side containers keep their identity. Done in `1f6a416`.
- [x] W3.3 `[contract]` Plant ↔ light round trip: the vids are remapped through the indexer, and `shoot_props` is merged. Done in `1f6a416`. The Caribu indexer remapping is not reproduced: the double replies with plant vids.
- [x] W3.4 `[contract]` Two full `run()` cycles of plant + soil + light, interleaved in-process, checked against hand-computed values. **These numbers are the regression anchor for WD.** Done in `1f6a416`. `test_two_cycles_regression_anchor` fails on the pre-fix wrapper (checked).

### W4: `scene_wrapper` unit tests (no subprocesses)

- [x] W4.1 `stand_initialization`, seeded or with `exact=True`. Check: Done in `1f6a416`.
  - the row and per-row counts, and `actual_xrange`;
  - the ID format;
  - coordinates with `-sowing_depth[i]`;
  - the rotation range;
  - the density floor.
- [x] W4.2 xfail-first: the multi-model pick uses `model_picker <= frequency` instead of the running total (`scene_wrapper.py:184`). Also cover `picker == 0`. Done in `1f6a416`.
- [x] W4.3 `plan_affinity` / `free_cpu`, with a chdir to `tmp_path` and `cpu_affinity` mocked. Cover: Done in `1f6a416`.
  - a fresh file;
  - `debug_runs`;
  - a corrupted file;
  - the lock wait;
  - the one-free-core margin: `OverflowError` when free cores equal the need, pinned as intended (Q7);
  - the last free removes the file;
  - the file was already deleted.
- [x] W4.4 xfail-first: a missing `outputs/` in the cwd gives `FileNotFoundError`. Anchor the lock files on `output_folder`, or create the directory. Done in `1f6a416`. The registry stays cwd-relative (`CPU_REGISTRY_FOLDER`), because it is shared machine-wide by the scenes launched from the same directory. The folder is now created when missing.
- [x] W4.5 Workers run in-process, with `os._exit` patched to raise a sentinel. Check: Done in `1f6a416`.
  - the iteration count;
  - `stop_event` both honoured and set;
  - `logger.stop()`;
  - the `record_performance` routing;
  - `logging=False`.

  Also xfail-first: `soil_worker` crashes without a `logger_class`, and `plant_worker` does when `logging=True`.
- [x] W4.6 (Q8) xfail-first, then fix: `light_worker` must use `light_scenario`, including where the meteo file comes from. Today it passes `plant_scenarios[0]` and reads the hard-coded, cwd-relative `inputs/meteo_Ljutovac2002.csv`. Done in `4bbc834`. A missing `meteo` now raises `KeyError`. Tests: `test/wrappers_tests/test_scene_wrapper_step1.py`.
- [x] W4.7 (Q7) Allocate cores for the soil and light workers too: `need = n_plants + 1 (soil) + 1 (light, if any)`, keeping the one-free-core margin. `soil_worker` and `light_worker` then call `cpu_affinity` like `plant_worker` does. Done in `4bbc834`. Order: plants, then soil, then light.
- [x] W4.8 (Q9) Pinning is a no-op when `psutil.Process` has no `cpu_affinity` (macOS). The file-based allocation still runs, so the logic is testable everywhere. Test it with the attribute monkeypatched away. Done in `4bbc834` (`available_cpu_ids`, `pin_to_cpus`).

### W5: `play_Orchestra` end to end (real multiprocessing, `slow`)

- [x] W5.1 Use the `poll_interval` from W0.4. Done in `062c61c`.
- [x] W5.0b **Found in W5: a race in the stop protocol.** Soil and light set `stop_event` when they finished. The first to finish could make the other skip its last step while the plants still waited for it, so `play_Orchestra` hung in `join`. Fix: environment workers set `stop_event` only on failure, and plants end the scene. Unit tests are in `test_scene_wrapper.py` / `_step1.py`. Done in `062c61c`.
- [x] W5.0c **Found in W5: lost queue messages.** `soil_worker` ended with `os._exit(0)`, which kills the queue feeder threads before its last `"finished"` replies are written, so the plants waited forever. This hung **2/10 runs under full CPU load**. Fix: `flush_queues()` (close plus `join_thread`) before `os._exit`. After the fix: **0/30 runs failed under full CPU load** (270 scene executions). Plant workers keep `os._exit` without a flush, because their final status is never read and a real shoot scene may not fit in the pipe buffer. Done in `062c61c`.
- [x] W5.2a Done in `062c61c`. Fix: `play_Orchestra` fills the plant SharedMemory from `np.empty` (uninitialised memory). The soil's `vertex_index >= 1` mask can therefore pick up garbage columns past the plant's vertex count. Fill with `np.zeros` instead. (Fresh OS pages are usually zero, which is why this goes unnoticed.)
- [x] W5.2 xfail-first: `del b` raises `NameError` when no plant was created or the run fails early. Also assert that no `/dev/shm/<plant_id>` segment is left over on any exit path. Done in `062c61c`. The `NameError` could not be triggered (there is always at least 1 plant). The numpy handle is now released right after initialisation, and the `del b` in `finally` is gone. Segments are unlinked on every tested exit path: normal, stop file and failing plant.
- [x] W5.0 **Suspected hang with a light model.** Decision (Q17): the light model sends an initial reply in its `__init__`, like the soil does. Apply it to `FakeLight` and report it for the downstream `LightModel` (B11). Each plant step needs the light's reply to its previous status, so the plant needs `n_iterations + 1` light replies (its init plus `n` runs). `light_worker` only runs `n_iterations` times, so the plant's last `get()` should block forever. The soil gets this right, because its `__init__` sends the extra reply. The threaded double scene is consistent with this: 2 plant steps need 3 light runs. Confirm with real processes, then decide on a fix (Q17). **Done in `062c61c`.** Measured: the scene did not hang, it was **truncated**. With a light model, every model ran `n_iterations - 1` steps, because the light worker finished first and set `stop_event`, and `clean_exit` was still True. Fix: `light_worker` now passes the queues to the light model's constructor, which answers the initialization messages (`FakeLight`). **Downstream action: `LightModel.__init__` must take the queues and answer the plants' initialization messages** (B11).
- [x] W5.3 2 × `FakePlant` + `FakeSoil` + `FakeLight` + `FakeLogger`, `n_iterations=3`, in `tmp_path`. Check: Done in `062c61c`. Checks run counts, logger calls, PARa, soil DOC, buffer and registry cleanup, and one core per worker.
  - `clean_exit`;
  - the folder layout;
  - each fake ran 3 steps;
  - `log_only_one`;
  - shared memory is released;
  - the CPU file is freed;
  - soil and light were pinned (W4.7).
- [x] W5.4 Deleting `Delete_to_Stop` mid-run gives `clean_exit is False`, and every process joins. Done in `062c61c` (the plant deletes the stop file after 2 runs).
- [x] W5.5 A plant whose `run()` raises sets `stop_event`, and the scene exits. Hangs caused by blocking queues are out of scope (Q12): the fake soil and light must not block when this scenario runs. Done in `062c61c`. The scene ends and cleans up, `clean_exit` is False since `a3a9877` (Q18).
- [~] W5.6 The doubles must be importable under `spawn` when tests run from `test/`. Run the suite once with `mp.set_start_method("spawn")`. **Partly done (`062c61c`):** fork, spawn and forkserver run on Linux; fork is skipped on macOS. **Skipped on Windows:** a Windows shared memory block is destroyed when its last handle closes, and `play_Orchestra` closes its creation handle before the plants open theirs. Keep the handle open until the join for Windows support.

### W6: Keep the real packages coupled (optional)

- [ ] W6.1 An `integration` marker plus `importorskip`, running a trimmed real composite for 1–2 steps. It is skipped in metafspm CI and run in downstream CI.
- [ ] W6.2 A reusable `openalea.metafspm.testing.assert_component_couplable(cls, translator)` for downstream CI.

---

## Epic WD: dynamic coupling through `DataStructure` variables

**Decided.**
- Links live on DataStructures (Q3).
- Components read DataStructure arrays live; there is no props snapshot (Q13).
- The switch is total: `MPGDataStructure` for plants and the 3-D grid DataStructure for the soil, with no MTG-props path left (Q14).
- The W `[contract]` tests, and the W3.4 numbers in particular, must pass unchanged once the doubles are retargeted.

- [x] WD.0 **Implemented 2026-09-29** (`coupling/translator.py`: `Link`, `Translator`, `parse_factor`; the WheatBRIDGES gate is met, 98 links 66/10/4/18; `eval` removed from `CompositeModel`; `.py` translators accepted). **Translator schema** (Q4 / Q4b). **Updated after Q4b: Python-first.** You want live references such as `scales.SubOrgan` and free formulas. YAML can only hold strings, which would have to be resolved by name when loading: that works, but it is not a live reference and it is not checked when you refactor. The revised proposal: **Drafted in the design note §4:** `Link` / `Translator` objects, a Python-first builder with live `scales.*` references, a `formula=` callable, a YAML loader for the existing files, and a restricted arithmetic parser in place of `eval`. Link kinds are derived from the link, not declared.
  - The **primary format is a Python module** (for example `coupling_translator.py`) that builds `Link` objects. It uses real references (`scales.SubOrgan`, `LabelsConfig` members, aggregation functions) and allows arbitrary formulas, since links can take a callable (`sources={"hexose_exudation": 12 * 6}` or `formula=lambda ds: ...`).
  - **YAML stays loadable** (the existing `example_translator.yaml`) through a loader that turns it into the same `Link` objects, with string factors parsed by a restricted arithmetic parser. It covers the plain factor/sum links, and scale names are looked up in `ScalesConfig` by name.
  - Keep the component-pair structure (`receiver → provider → variable`), so that `get_component_inputs_outputs` and the soil handshake derivation carry over.
  - The cost: a Python translator runs code when loaded (acceptable, since it is project code) and is harder to edit without a Python editor.
  - Settle this in the WD.1 design note.

  Superseded first recommendation:
  - Keep a **declarative file (YAML)** as the source of truth. It is diffable, can be edited without code, can't execute arbitrary code, and matches the current files.
  - Load it into **typed Python objects** (for example `Link(receiver, receiver_var, sources={var: factor}, scale=..., aggregation=..., edge_mapping=...)`), with factor strings parsed by a restricted arithmetic parser. That replaces `eval`.
  - Let each link optionally carry `scale` (a scale name from `ScalesConfig`) and `aggregation` (`sum` | `mean` | `mass_weighted_mean` | `proximal` | `distal`, the same vocabulary as `integrate_at_scale` / `average_at_scale` / `edge_mapping`). The resolver then maps between the MPG anchoring scales.
  - Anything that needs real code is referenced **by name** from a Python registry (`@register_aggregation("my_fn")`), not embedded in the file.
  - Accept a Python `dict` or `Link` list as an alternative input for programmatic use. It loads into the same objects.
  - The short form `name: {src: factor}` stays valid, with the defaults "same scale" and `sum`.
- [x] WD.1 **Design note** in `docs/`, for your review before any code: **Drafted:** `docs/design/coupling_through_datastructures.md`, covering the current state, the blocking bugs B-a…B-i, requirements R1–R12, DS `get` / `set` / `register` in place, aliases, derived variables, scale operators and a scalar store, the Coupler / VoxelLocator / Transport, live reading in Functor and decorator, the Logger export, and a gated migration sequence.
  - **Links**: identity is a no-op; an alias is a name-level alias table on the DS; a derived link is a vectorised `ds[r] = Σ fᵢ · A(ds[sᵢ])`, with `A` the scale mapping or aggregation from WD.0, evaluated before the receiver's step.
  - **Across DataStructures**, a `Coupler(ds_a, ds_b, mapping)`:
    - MPG ↔ 3-D grid: vertex → voxel index arrays built from the barycenters, recomputed after `update_topology`;
    - plant → soil is a scatter-add of extensive variables;
    - soil → plant is a gather of intensive variables;
    - across processes, the SharedMemory carries the Coupler's variable subset.
  - Where the environment component names (today `"SoilModel"`) and the plant-side coordinate variables come from.
  - The lifecycle: registration, `update_topology`, growth.
- [x] WD.P Prerequisites on the target DataStructures (these come from backlog B7): **Done in `42e00eb`:** variable store (`get` / `set` / `register` / `alias` / `version`, in place), `node_local_index`, one incidence convention (GraphView's), scale-aware MTG mapping with errors on partial coverage, write-back errors raised, growth carrying variables over with `on_grow` (declared on fields), grid `(x, y, z)` axes and `cell_centers` / `cell_volume` / `locate`, per-instance `LabelsConfig`. There are 21 tests in `test_datastructure_prerequisites.py`. **B-i deferred:** `populate_graph` still adds Compartments to the SubOrgan `children()`, and anchors match `scale ==` filters. The WD.3 scale operators must build membership from `components_iter` and skip anchors. N-D MultiGrid restriction/prolongation stays out of scope while the soil is on `ArrayDataStructure` (Q16).
  - MPG node order versus `GraphView.node_local_index` (it is wrong on real MPG views);
  - N-D grid restriction/prolongation, and a numeric test of the 3-D Laplacian, if the soil grid uses them;
  - fix the soil axis convention: the RhizoSoil voxels are indexed `(y, z, x)`, so settle this against the `ArrayDataStructure` shape and coordinates.
- [x] WD.2 Live reading. Components access `self.data_structure` variables (views) and the `self.props = ds.to_props_dict()` snapshot is removed. Rework the `decorator.py` consumers (`:429, :732, :769`) and `write_back_to_mtg`, so they read and write the DS directly. The UC1 suites must stay green. **Done 2026-09-29:**
  - no props snapshot; the solver snapshots the DS at each solve and writes in place;
  - the Functor runs on DS arrays (vectorised, with a `vectorized=False` opt-in);
  - `props` is a read-only view;
  - `self.previous(fn)`;
  - 7 tests in `test_live_datastructure.py`; UC1 suites unchanged and green.

  **Follow-ups:**
  - (a) ~~After `ds.update_topology()`, `FunctionalComponent._graph_view` is stale~~ **done (`3c85536`)**. Add a topology version, separate from the variable `version`, and rebuild views lazily.
  - (b) `pull_available_inputs` and `apply_input_tables` still use props on FunctionalComponents; this belongs to WD.4.
  - (c) The UC1 equations still read the user-managed `_previous_fields` (deprecated); migrate them to `self.previous()`.
- [x] WD.3 Alias table and derived variables on `GraphDataStructure` / `FieldDataStructure`. Test: **Done 2026-09-29:** `derive` / `refresh` (weighted sums or formulas, dependency order, cycle and location checks, alias-aware), coarse bio-scale locations plus a `scalar` store, and the `sum` / `mean` / `weighted_mean` / `broadcast` / `proximal` / `distal` operators, all carried over growth. 12 tests in `test_datastructure_links.py`.
  - survival across `update_topology`;
  - alias chains and cycle detection;
  - a missing source gives a clear error.
- [x] WD.4 Rework `CompositeModel` on WD.0–WD.3: **Done 2026-09-29:** links inside a shared DS become aliases or derived variables refreshed before the receiver's step; input tables go to the DS; soil outputs are registered; the legacy props path is kept until WD.6. 7 tests reproduce the legacy contract numbers. **Remaining:** coupling across DataStructures inside one plant composite (root MPG ↔ shoot structure) raises `NotImplementedError`, and belongs to WD.5 (Coupler).
  - `couple_components` registers links on the component DataStructures;
  - `apply_input_tables` writes DS variables;
  - `documentation` reads field metadata.
  - Remove `declare_data` MTG assumptions such as `props["struct_mass"]` and `.properties()`.
- [x] WD.5 Scene transport: the handshake height comes from the translator and the width from the DS size, with capacity growth. Columns are DS node positions, and an explicit id row replaces the `vertex_index >= 1` convention. **WD.5a done (`26433fc`):**
  - in-process `Coupler` + `VoxelLocator` in `(x, y, z)`: map from barycentres, `push` / `pull`, `from_translator`, stale-map detection through `topology_version`;
  - it matches the reference soil model's sums.

  **WD.5b done 2026-09-29** (`804d082`, and the scene commit):
  - `Transport` sized from the translator, with an explicit node-id row and count instead of the `vertex_index >= 1` convention;
  - a buffer-backed plant view on the soil side, so the soil process runs the Coupler on it;
  - a configurable `play_Orchestra` buffer shape;
  - DataStructure-backed scene doubles, with contract tests reproducing the regression-anchor numbers.
- [x] WD.6 Retarget the W1 doubles and fixtures to `MPGDataStructure` / the 3-D grid, then re-run all of W2–W5 unchanged. **Done, following Q28:** the DataStructure doubles `doubles_ds.py` (PlantCarbon / PlantNitrogen, GridSoil, DSFakePlant, DSFakeSoil) have their own contract tests that **reproduce the props-based regression anchor exactly**, in process and with real processes. The legacy doubles and tests stay as the legacy contract until the legacy path is removed at the end of WD.7.
- [x] WD.7 Migration guide for downstream packages (RootGrowth, RootAnatomy, RootWater, RootCN, CNW_Grass, and SoilModel, which is still on the legacy `openalea.metafspm.component.Model` + `component_factory`). **Guide written 2026-09-29:** `docs/design/downstream_migration.md`. It gives per-role steps pointing to the tested DS doubles, plus a checklist. The actual migration happens in the downstream repositories, followed by the removal of the legacy path (Q28). **Scope set (your 2026-09-29 rule):** the downstream packages are not edited from this repo. The in-repo UC tests are migrated (B9), and the legacy path is removed (`b3c11c8`).
- [x] WD.8 `assert_component_couplable` (W6.2), updated for DS-backed components and the WD.0 schema. **Done (`3c85536`):** `openalea.metafspm.testing.couplability_problems` / `assert_component_couplable`.
- [x] WD.9 **Logger compatibility (scope reduced by Q15).** Add read-only export helpers in metafspm (`DataStructure.available_vars()` plus a per-variable array with coordinates and ids) and adapt only the Logger's xarray (`mtg_to_dataset` / `recording_raw_MTG_properties_in_xarray`) and csv (`recording_summed_MTG_properties_to_csv`) writers onto them. Test them on `MPGDataStructure` and `ArrayDataStructure`. The remaining Logger features stay as they are. Original analysis: `openalea.fspm` `Logger` only accepts an exact `openalea.mtg.MTG` or a `dict` in `model_instance.data_structures`, and reads `props["root"]` / `props["soil"]`. MPG and grid DataStructures will be rejected. Option (a): metafspm provides a small read-only export API on `DataStructure` (`available_vars()`, `to_dataset()` / `variable(name)` with coordinates and ids) and the Logger is migrated onto it. Option (b): an adapter lives in metafspm. See Q15. **metafspm side done (`3c85536`):** `export`, `to_dataframe` (entity and time index, cell centres, `.to_xarray()` ready) and `summarize`. The Logger's two writers are adapted downstream, following the guide §6.

---

## Backlog (from the 2026-09-29 audit; details in devlog §4–§8)

- [x] B1 Add `coverage`/`pytest-cov` to the dev env with a `[tool.coverage]` config. Remove the unused `nbmake`. **Done (`b4b9ffd`):** `[tool.coverage]` config added; `nbmake` dropped.
- [x] B2 Re-enable `est_mms_solver.py` with a one-line `SolverSpec` import fix (+12 tests). **Done (`b423d00`):** `test_mms_solver.py`, 12 tests.
- [x] B3 Fix `test/utils.py::deep_reload_package`: it is a no-op for a string argument and purges too much for a list. Once W1.6 exists, replace its uses with `Choregrapher.reset()`. **Partly done (`409a0e9`):** no active test calls it any more (the collection-time purge duplicated class objects when directories were passed explicitly). What remains: delete `test/utils.py` once `est_model_assembly.py` is migrated or dropped. **Done (`b7f95ad`):** `test/utils.py` removed.
- [x] B4 Make test imports work the same in CI and locally: remove the `from conftest import` pattern, and run from the root or ship pyproject to the conda test env. **Partly done (`409a0e9`):** the solver builders moved to `solver_specs.py`, so every tested directory order passes. What remains: `--import-mode=importlib` still fails, because the helper modules (`solver_specs`, `doubles`, `simple_seedling`) are resolved through `sys.path`, and the disabled `est_uc2/3` files still use `from conftest import`. **Done (`57c5e04`):** root `test/conftest.py`; the suite passes with `--import-mode=importlib`.
- [x] B5 Test the multi-step time loop (`ODESolver.solve`, `DAESolver.solve`). Fix the solvers that ignore the `x` passed to them. **Done (`09eb138`):** first tests of the `solve()` loops; solvers start from `x`.
- [x] B6 Fix `ScipyIVPSolver` with edge unknowns. Fix `make_solver` silently dropping a dict config. **Done (`09eb138`):** edge recovery with the nodes fixed, IVP returns the full state and relies on `solve_ivp`'s error control, `make_solver` accepts dict configs.
- [x] B7 The parts WD needs moved to WD.P (done in `42e00eb`). What stays here: the `from_legacy` length bug, which may be moot once the MTG path is removed (Q14). **Done (`b7f95ad`):** `from_legacy` maps values by vid.
- [x] B8 Fix the UC1 SubOrgan boundary condition that lands on a leaf tip, and the Choregrapher test that passes for the wrong reason. **Done (`39978cd`):** boundary conditions on the real root; the scheduling test asserts the actual order.
- [x] B9 Migrate UC2/UC3/UC4 (`est_*`) to `FunctionalComponent` on DS live reading, after WD.2. **Done (`8f6f055`):** UC2 / UC3 / UC4 on `FunctionalComponent` + `MPGDataStructure`. The anatomy graph helper remains unported.
- [x] B10 Replace the `est_` file prefix with explicit skip markers. **Done (`b423d00`):** no `est_` files left.
- [-] B11 Downstream, not metafspm. (a) New contract (Q17, W5.0): `LightModel.__init__(queues_light_to_plants, queue_plants_to_light, scene_xrange, scene_yrange, meteo, **scenario)` must answer the plants' initialization messages, as `FakeLight` does. (b) `LightModel.run` crashes if the first step has no Caribu run (`previous_Erel` is None, `previous_indexer` is undefined). **Out of scope (your 2026-09-29 rule):** downstream packages are not edited from this repo; it is documented in the migration guide §5.

---

## Open questions (answer inline; Claude reads them next session)

- **Q29.** `solve/specializer.py` (numba specialisation of step functions) is now unused: its only caller was the ArrayDict-only branch removed with the legacy path, and its coverage is 0 %. Options:
  - (a) delete it;
  - (b) keep it, and plan its reuse on the vectorised DataStructure step path (a numba-compiled `fun(*arrays)`).

  Recommendation: (a) now. Vectorised numpy steps have most of the speed-up, and a DS-path specialiser would be written against the new Functor anyway.
  → answer:

### Answered 2026-09-29 (kept verbatim; decisions are in the log above)

- **Q4** (translator eval / format): "We could keep this, but I also wonder considering the MPG now has introduced multiscale anchoring of properties, should we also include the scale itself in the translator? Is it still right to store it as yaml then and should I instead store it in python format so that conversions, aggregations and scales etc are python methods and objects directly?" → answer proposed in WD.0, to be confirmed in Q4b.
- **Q5** (stale cache): "No"
- **Q6** (recursive_reload): "I think this comes from the fact that choregrapher, called at each component's __call__ to solve the component according to its internal scheduling, is a singleton and to make independant tests I needed this to be fully reinitialized in each test. Is it still relevant?" → No. W1.6 `Choregrapher.reset()` in place replaces it, because swapping the instance breaks already-defined classes.
- **Q7** (core margin): "keeping one core free yes this is a safety margin to keep the machine responsive, but indeed soil and light workers should be allocated specific resources as well"
- **Q8** (meteo): "yes"
- **Q9** (macOS): "no pinning on this platform then"
- **Q10b** (Logger): "provided" (`logger_api_reference.py`)
- **Q11** (×72): "This went unoticed, so I changed names on soil side since they do not track the same quantitity (_massic suffix)"
- **Q12** (timeouts): "Don't touch that yet"
- **Q13** (live vs snapshot): "Live reading"
- **Q14** (dual support): "Fully switch to the current branch DataStructure classes that have been implemented (MPG for plant and 3D grid for soil)"
- **Q4b**: "I just wonder if yaml might not Constrain too much the formulas and also I like the idea that scales keep being a dynamic link like scales.SubOrgan in other from configs.py, which is not possible with yaml right?"
- **Q15**: "This is so untertwined with the evolutions of DataStructure that it surely needs to evolve, but it is not the focus of this specific package yet, so you can evolve it for xarray and csv write functions to check we actually can output DataStructure and inherited classes, but don't bother reimplementing everything yet."
- **Q16**: "as of right now begin with ArrayDataStructure, we will transition to the other later"
- **Q16b**: "Since x, y, z is more intuitive to everybody, let's keep this"
- **Q17**: "do like soil does"
- **Q18**: "yes make it false"
- **Q19**: "Yes make it an argument, and give a default so that when using only one plant model it doesn't have to be passed, it is implicit."
- **Q20–Q26**: "agree with the strategy and the recommendations to questions", with precisions on Q24 and Q26 written into the design note §11.
- **Q27**: "option (a)"
- **Q28**: "follow recommandation yes"
