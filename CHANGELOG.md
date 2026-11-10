# Changelog

## Unreleased (release2026)

### Outside the solvers: the allocator, the recorder and the lookups (bit-identical)

- **`Compute(memory="trim")`, the default, holds glibc's allocator to two arenas** (`mallopt(M_ARENA_MAX, 2)`,
  `compute.ARENAS`, `Compute.limit_arenas`), once per process when a Compute's `limits()` is first entered (the
  Scene's build, before its worker threads), besides `malloc_trim(0)` after each step. It matches the environment
  variable `MALLOC_ARENA_MAX=2` at the process's start (use case 1, 8 plants resumed from day 12: resident memory
  2.73-2.76 GB against 3.17-3.31, peak 3.68-3.76 GB against 4.13-4.23, the step unchanged), which the use case's
  `environment.yaml` no longer sets. An arena limit set in the environment (`MALLOC_ARENA_MAX`, `GLIBC_TUNABLES`) is
  kept; `memory="keep"` does neither; both are no-ops where the C library is not glibc.
- **The frame build in bulk** (`io.encoding3d._mpg_data_structure`): each scale's instanced vertices located once
  (`Instances.plan`) and every property read through that plan (`_lookup(prop, vids, plan)`,
  `_InstancedProperty.lookup`, `MPG._explicit_property_mask`), the scale's vids read-only so that the
  DataStructure's columns share their positions (`columns._positions_shared`); the vertices by scale, their own
  complex and parent, the labels, the edge types and the MTG edge codes by array operations instead of per vertex.
  The frames are byte-identical (use case 1, 2 plants, 3 days: every frame file and table; 8 plants resumed from
  day 12: both frames and their parts). 8 plants at day 12: a frame 13.0 → 2.5 s, with p0's sections 16.6 → 6.1 s
  (3.7 s of it GRANAP's `display_triangles`), the build's peak +0.47 → +0.40 GB.
- **The checkpoints' fast path** (`scene.shared`): the pickler's `persistent_id`, called for every object pickled
  (about 13 million at 8 plants, day 12: the MPG's dicts), returns at once for the builtin scalars and containers
  (one `type()` lookup in a frozenset) and looks a class' `shared_in_checkpoints` up once (`_SHARED_BY_TYPE`). The
  pickles are streamed to their files instead of made in memory first: a DataStructure's `state.pkl`
  (`SharedStore.dump`, or protocol 5 straight to the file without a store) and the shared parts, their digest
  computed on the way (`SharedStore.key_of`: a temporary file of the folder, then named by its key; `SharedStore.
  dumps` removed). The files and the parts' keys are the former writer's bytes (former checkpoints load and keep
  their parts), but for a store-less `ds.checkpoint(path, shared=None)`, now protocol 5 (pickle.dumps' default 4
  before; both load). 8 plants at day 12: the first checkpoint of a resumed scene (every kind's part pickled for its key)
  4.0 → 2.5 s, the next one 0.65 → 0.30 s, the peak over the resident memory +0.41 → +0.06 GB and +0.33 →
  +0.05 GB. Protocol 5's out-of-band buffers were measured and not adopted: protocol 5 already writes the arrays
  from their memory in band when the pickle goes straight to a file (the same time and peak), and the format stays.
- **The instanced lookups** (`Instances.plan`, `structure`): the located vertices grouped by template with one stable
  sort of their codes (`Instances._group_indices`) instead of one comparison per template; `MPG._property_at` takes
  an instanced vertex's own entry by its mask (`_explicit_property_mask`) instead of `np.isin`. Bit-identical. The
  plans and `ArrayDict` searches kept per topology change, as planned, were measured and dropped: within a step the
  reads are at fresh arrays (`values_at`, `values_of`), 535 plans a step at 8 plants, day 12, the repeated ones
  small (28 % of the vertices located), no cache hit for the read-only ones (`_read_plan` keeps its own); the step
  is unchanged within the runs' spread (3.80-3.84 s against 3.82).
- The user guide (*Cores of a hybrid CPU*) and use case 1's README document the main thread's core on a hybrid CPU:
  `taskset -c 0-3` keeps a run on the performance cores when the machine is shared (8 plants at day 12: +29 % on the
  efficient cores, +122 % on the low-power ones; nothing gained on an idle machine). No Compute setting.

### Breaking: the former keyword names removed

- `graph_system(max_iter=, tol=, fd_eps=, n_substeps=)` raise `TypeError` (use `max_iterations=`, `tolerance=`,
  `finite_difference_step=`, `substeps=`), and `integrate="step"` raises `ValueError` (use `integrate="once"`); they
  warned for one release. Likewise `state_variable`, `input_variable` and `parameter` no longer take `initialize=`,
  `state_variable_type=` or `mapping=` (use `default=`, `kind=`, `aggregation=`): `TypeError`.
- An output that is neither a declared field nor given a location (`@rate(location=, locations=)`,
  `@graph_output(location=)`) raises a `ValueError` naming the component and the output; its location was inferred
  from its shape with a `DeprecationWarning`. No port relied on it.

### Breaking: the linear solver per graph system (fix-D)

- **`Compute(linear_solver=)` is a backend's name (every system, as before) or a mapping, one per system**:
  `Compute(linear_solver={"default": "superlu", "symmetric": "cholmod", "MechaHydraulics._plant_problem":
  "superlu_reuse"})`. A graph system's backend is its qualified name's (`"Component._system"`, its component's
  class or a base class declaring it, as `Reduction(systems=)` names it), else that of a property it declares
  (`solve.linear.SYSTEM_PROPERTIES`: `"symmetric"`, `@graph_system(symmetric=True)`), else `"default"` (the device's
  when the mapping does not give it). Components declare properties, never a library; the nonlinear method stays the
  system's own (`@graph_system(solver=)`). The mapping is checked when the Compute is made (each backend registered,
  `"cupy"` only on `"cuda"`, each key `"default"`, a known property or a `"Component._system"` name) and when the
  Scene is built (each backend's package, `Compute.check`; each qualified name one of the scene's graph systems, one
  mapped to `"cholmod"` declared symmetric, `Compute.check_systems`): a mistake raises naming its key.
  `Compute.linear(system)` resolves (one LinearSolver per backend, its caches kept); `solve.linear.solve(matrix,
  rhs, system=)` takes a `solve.linear.LinearSystem` (key, qualified names, properties) instead of `key=`,
  `symmetric=`; `spec.parameters["linear_system"]` is one; `solve.decorator.linear_system(component, name)` and
  `linear_systems(components)` build them.
- **The static condensation and the Kron reduction go through the registry**: a backend gives `solve_once(matrix,
  rhs, symmetric)` (a system solved once: `condensed_solve`'s, `equivalent_conductance`'s) and `factorize(matrix,
  symmetric)` (a factorisation kept for many right-hand sides: a condensation's kinds, `KronReduction`'s interior),
  resolved for the system they belong to (`condensed_solve(..., system=)`, `kron_reduction(matrix, kept, system)`,
  `equivalent_conductance(..., system=)`; a reduction's kinds its reduced system's), else `"default"`. `"superlu"`
  keeps SuperLU's default ordering (COLAMD) there and the device's factorisation (backend.HostBackend, or its factors
  on the GPU): use case 1 (2 plants, 3 days) is bit-identical, every table and frame. `"cholmod"` factorises a
  symmetric system's there too; `"pardiso"` and `"mumps"` solve once with their own, factorize with SuperLU.
- **There is no `dense_solver`**: the dense solves (a system declared `prefer_sparse=False`, the edge unknowns'
  recovery) are LAPACK's, with no other backend to choose. `solve_ivp`'s and scipy `root`'s factorisations stay
  inside scipy, outside the registry (documented).
- **CHOLMOD with scikit-sparse 0.5** (the current release: `cho_factor`, then `factorize` on the same pattern, `solve`);
  the former adapter called 0.4's `analyze`/`cholesky_inplace`, absent from 0.5, and a pattern coming back after
  another one's (the solver frees the others' factors) raised `KeyError: 'factor'`: both found running it for real
  (it had been tested on a stand-in only).
- **`solver="implicit_euler"` removed** (a deprecated alias): a backward-Euler step is `solver="newton"` with
  `transient=True`. `register_solver` documents nonlinear methods only (the linear algebra is the Compute's).
- Use case 1's `benchmark.py --linear-solver` takes a JSON mapping; `memory_probe` counts every backend's caches.
- Docs: user.md (the run configuration, the solvers), README, gpu.md, ref.md; `tutorial_fspm2026_new_solver`
  (the method and the linear algebra within it: a mapping, its resolution order) and
  `tutorial_fspm2026_general_usage` (the plain form), both re-run. Caribu's own `linear_solver` setting is renamed
  `radiosity_method` (its radiosity system's iterative method).

### Fixed: a variable two components of one model read from the same provider exchanged once

- `coupling.cross.Exchanges` joined each (receiver component, provider) pair a link matches, and added the values of
  the links writing the same variable of the same DataStructure, as several providers' are: two components of one
  plant model (or environment model) on one DataStructure, each reading the same variable from the same provider,
  received it **twice** (a broadcast soil temperature 20 °C for 10, an extensive lit area doubled). Such a link is
  now kept once per (receiver DataStructure, variable, provider DataStructure, sources, formula, aggregation,
  weight). Found when Caribu's `PlantOptics` declared `lit_area` beside `AbsorbedLight` (fix-C); test
  `test_two_components_reading_one_soil_variable_receive_it_once`. Use case 1 has no such pair (2 plants, 3 days
  bit-identical).
- The soil-plant-atmosphere examples give the Scene's time step as the duration `"1 h"`; both tutorials re-run.

### Breaking: a packed run's frames folder removed once its archive is checked

- **`SceneRecorder.pack(keep_folder=None)`** and **`pack_at_finish`** check `frames.fspm.zip` once written
  (encoding3d's `pack`: every file of the frames folder a member, of the same size and CRC-32, each member read back;
  the archive valid, 0 errors), then **remove the frames folder**. `SceneRecorder(pack_keep_folder=True)` (or
  `pack(keep_folder=True)`) keeps it; when the check fails, the folder stays, no archive replaces the former one and
  `encoding3d.ArchiveCheckError` is raised. scenario.json's recorder records `pack_keep_folder`.
- A run resumed from a checkpoint (`Scene.load_checkpoint`, `Scene.restore`), or going on after a pack, **unpacks
  `frames.fspm.zip` into `frames/`** when the folder is gone, cuts it back to the restored step and adds its frames;
  the next pack holds them all. The frame files the cut removes from the index are removed too (they were left
  behind), and so are a former run's frames when a new run starts in the same folder, so that the folder holds its
  sequence only and can be removed whole.
- Use case 1's `run.py`: `keep_checkpoints=1` (the last day's only) and `scene.recorder.pack()` removing the folder;
  a 2-plant 3-day run takes 343.6 MB at its end (`frames.fspm.zip` 16.3 MB for its 45.0 MB folder, `checkpoints/`
  327.1 MB: the last 8.5 MB, `shared` 318.6 MB); the archive validates with 0 errors (the soil's 3 "cell ids
  reused" errors are gone with encoding3d's validator fix: an adaptive grid's cell refined again is the same cell),
  ParaView's reader opens it, and the run resumes from its one checkpoint (frames 0 to 84 packed again).
  `benchmark.py --outputs` measures the instances' folder before packing it.

### Breaking: frames' missing property values as row ranges

- An MPGDataStructure's frame writes its properties' missing values compactly: the extras' **`absent_rows`** (the
  rows of a scale without the property) and **`none_rows`** (those holding it as None) are flat `[start, stop, ...]`
  lists of half-open ranges over the scale's rows (`io/encoding3d._row_ranges`, read back by `_ranges_mask`); they
  replace `absent` and `none`, one vid per row. Use case 1's last frame (2 plants, day 23): 838,773 vid runs in
  15.9 M vids become 209,388 row ranges (these lists were most of the frame's 162 MB JSON chunk,
  which drops to an estimated 3 to 4 MB); 2 plants at day 2, the JSON chunk 0.68 → 0.15 MB, at day 3 written unfiltered
  1.01 → 0.36 MB, the MPG rebuilt identical (264 properties, 445,775 values). Every reader parses a large frame faster
  and in less memory (encoding3d's ParaView reader: about 1 GB less at day 23). `from_frame` refuses a
  frame of the former lists (`ValueError`: rewrite the frame). The extras are metafspm's own: encoding3d's
  specification, its conformance files and its codecs do not read them and are unchanged.

### Breaking: the process pool removed (devplan_generic_kernels QG-3)

- **`Compute.workers` is gone, with no alias** (`Compute(workers=...)` raises a TypeError): a reduction's kinds are
  worked out in the main process, each in a thread of `Compute.threads` (GK-5). Removed with it: `WorkerPool`,
  `WorkerError`, `worker_state`, `in_worker`, `FORK_HOOKS`, `START_METHOD`, `POOL_WORK`, `SECONDS_PER_CELL`,
  `Compute.pool_size`, `pool`, `pool_info` and `close` (compute.py); the pool's section of solve/reduction.py
  (`_evaluate_in_pool`, `_worker_evaluate`, the processes' copies and syncs, the fork records and the re-fork when
  new templates appear); `solve.linear.block_threads`' one thread in a pool's process; `Scene.finish`'s closing of
  the pool; run.json's `"compute": {"pool"}`; `benchmark.py --workers` and its per-step pool record;
  `memory_probe`'s pool report; `bench_reduction.py --workers`.
- A component's setting named **`threads`** (no longer `workers`) that its scenario does not give takes the
  Compute's (Caribu's projection threads, renamed in Caribu).
- **`run_variants(run, scenarios, processes="auto", compute=Compute())`** and `variant_processes(processes, ...)`:
  the variants' processes are named `processes` (was `workers`); a scene's `threads="auto"` is shared among them
  (`processors // processes`), an integer kept when `processes × threads` fits the processors.
- Measured (use case 1, 8 plants, the user's choice after the GK-5 gate): to day 8, 298.0 s at a scope peak of
  3.21 GB with the kinds' threads, against 286.5 s and 4.06 GB for a 4-process pool (4 % slower, 0.85 GB less); at
  day 13, 4.15 against 4.13–4.14 s per step, 3.65 against 4.73–5.09 GB. Every result the same, bit for bit.

### The kinds in threads (devplan_generic_kernels GK-5)

- **Each kind of a batch is worked out in its own thread** after the library's run (`_kinds_of_batch`:
  `Compute.section_threads(kinds, "compute")`, the largest interiors first): its blocks, its Kron reduction (one
  sequential SuperLU factorisation, outside the GIL), inputs, outputs, `@kind_derived` values, recovery and release.
  The kinds are added in the requests' order and are the same bit for bit at any thread count; numba's kernels
  within a kind run on one thread. run.json times them under `kinds.tails` (with their threads).
- `Compute.count` is thread-safe (the kinds' threads count their solves).
- The QG-3 gate (the threads at least as fast as the pool) was not met: at 8 plants to day 8 the threads take
  298.0 s against the 4-worker pool's 286.5 s, at a peak 0.85 GB lower; level at day 13. The user chose the
  threads and the memory: the pool is removed (above).

### The kinds' batches (devplan_generic_kernels GK-4)

- **One library population per reduction and time step, one plant per plant parameters:** the kinds of plants of
  different parameters (signatures) are evaluated in the same batches (`kind_batch` cells at most), each kind's
  groups under its signature's plant, each plant given its parameters; a batch removes the plants of the
  signatures it has not and adds one for a new signature. It was one library, and one batch, per signature.
- **`Reduction.assignment`'s plant parameters and signatures are kept** while the library components' settings and
  parameters are the same (their write counts, or their values): not read and digested at every call.
- **A library's components keep their own caches across its batches** (`keep(name, saved=False)`, rebuilt by the
  component when its key no longer matches); their other state is reset as before. MECHA's per-template unit
  volumes (`@kind_derived` condensed volumes) are kept that way.
- **`KronReduction.release()` also frees S** (`reduced`), with the factorisation and X; `pairs()` and `shunts()` need
  S and are not available after `release()`.
- **`METAFSPM_CHECK_KINDS=1`** (new): every kind evaluated is evaluated again alone, in a library built for it, and
  compared field by field, bit for bit (`solve.reduction.check_kinds`).
- Each kind's work after the library's run (its blocks, Kron reduction, inputs, outputs, `@kind_derived`, recovery,
  release) is one function of the batch's arrays (`_kind_from_batch`).
- Not done (measured, devplan_generic_kernels GK-4 note): slot pools reused across batches and the library running
  only the reduced systems' closure.
- Bit-identical.

### Assembly into kept patterns (devplan_generic_kernels GK-3)

- **`GraphView.laplacian(weights)`** (new): B diag(w) Bᵀ through a pattern kept per view (each entry's edges with
  their signs, `kernels.laplacian_pattern`: linear in the edges), filled by `kernels.scatter_edges` (parallel over
  chunks of entries made from the work only): each entry summed over its edges in ascending order and the exact
  zeros dropped, as scipy's product does, so the same matrix bit for bit. MECHA's Jacobians of the plant and standard
  problems and its osmotic input Jacobian call it.
- **Not done: the Jacobian's terms summed into a kept union pattern.** Tried (a `TermSum` in `jac_evaluator`) and
  removed: use case 1's topology changes nearly every hour, so the union is built at almost every step, and building
  it (a sort of every term's entries) costs seconds at 8 plants where scipy's chain of additions costs 13 ms; kept,
  its scatter is no faster than that chain. `jac_evaluator` adds its terms with scipy, as before.
- Bit-identical.

### Gathers by owner and tree orders as kernels (devplan_generic_kernels GK-2)

- **`DataStructure.local_gather(owner, owner_key, tables, fill=)`** (new, `kernels.gather_local`, nogil): per entity,
  its owner's table at its local row (its rank among its owner's entities). MECHA's per-node geometry of each
  section (cell ranks, relative x, root nodes) is one gather over its distinct geometries.
- **`MPGDataStructure.preorder(branches_first=True, defer=None)`** (new, `kernels.depth_first_orders`): the nodes in a
  depth-first pre-order over the MPG's children (in its children order), branches before successors, *defer*'s
  vertices visited as the last branches of others. Adel's `turtle.traversal_order` is it.
- **`chain_scan(..., cut=mask)`**: a chain cut after the entities of *mask* (its next one starts a chain). ArchiSimple's
  distance from tip is a reverse scan cut at its apices and nodules instead of a level-by-level fold (its terms in the
  fold's order).
- Bit-identical.

### Data access without a sweep of the vid index (devplan_generic_kernels GK-1)

- **`MPG.column(name)` on an MPG with instances** is built from the explicit column and each template's values
  gathered at its instances (`Instances.property_entries`: one gather per template), the explicit entries over them:
  no `np.arange` over the vid index, no `np.isin`, no instance located. The values, their type and their presence
  are those of `values_at` over every vid (a test on an instanced toy MPG).
- **`FunctionalComponent.parameter_table(names, location)`** (new): {name: one value per entity} of several per-plant
  parameters. `parameter_view` (and so `parameter_values` and the steps' parameter arguments) gathers a parameter's
  per-plant values through each entity's owner index, found once per topology and location and shared by every
  parameter (`_parameter_owners`), not one broadcast map per parameter.
- **`MPG.compartments_csr(from_scale)`** (new, `OwnerCompartments`: `owners`, `indptr`, `compartments`, `has`,
  `has_all`, `compartments_of`, `items`): the Compartments of each owner as one CSR, the owners in order of their first
  Compartment, built in one sort from `compartment_owners`; `compartments_by_owner` is its dict.
- **Traversal orders and levels as kernels:** `kernels.depth_first_orders` (one explicit stack, the pre and post
  orders of the recursive visit), `tree_depths` and `tree_levels` (a stable counting sort by depth) replace the Python
  stack walks of `MPGDataStructure._traversal` and the coarse locations' traversals (`tree_kernels.traversal`, new;
  `_tree_traversal` gone) and the pointer doubling of `tree_kernels.depth`/`levels`: the same orders, depths and
  levels, element for element (tests against the loops).
- Bit-identical.

### Each step timed in run.json (devplan_generic_kernels GK-0)

- **`run.json`'s `"compute"`/`"timing"` holds each step's wall time and calls**, `functor.<Class>.<step>` beside the
  kernels' and solvers' (`Functor.__call__` under `Compute.timed`): inclusive of the steps it runs inside (a
  reduction's library components while it evaluates kinds). `benchmark.py` records them per step (`"kernels"`).
- Results unchanged.

### The carry-over of every numeric dtype in bulk (devplan_scene_parallel, after P5: bulk carry-over)

- **`update_topology` no longer copies every variable's array before carrying it over:** the carry replaces the
  stores' arrays, it never writes into the old ones.
- **One block per dtype and location:** the float and int scalar variables (an int's default an integer) are each a
  row of their dtype's block, taken from the old array through the location's match in one gather (`np.take`), no
  stack of the old arrays; the new entities get the default or, in anatomy mode, their inherited source's value.
  The int variables, carried one by one through floats before, keep their dtype throughout. Object and shaped
  variables are carried one by one, as before.
- The functors' `_write_outputs` were measured, not changed: their writes are one `ds.set` per variable and call
  (0.03 s a step at 8 plants, day 12); the 0.5 s a profile shows under them is the `update_topology` that
  `_create_nodes` triggers.
- Bit-identical.

### The states an MPG-style step leaves, read back in bulk (devplan_scene_parallel, after P5: bulk reads)

- **One lookup per location, not per variable:** a view's `at()` (`DataStructureColumn`) keeps the positions of
  read-only vids (a read plan's, `_read_plan`) in the location's sorted ids (shared by its variables) for the next
  read; the presence of the entities synchronised before is found the same way; a small overlay's keys are looked
  up in ascending vids instead of the vids in the overlay; the overlay's entries at the DataStructure's entities are
  dropped by looking its keys up (`drop_at_entities`).
- **`contains(vids)`** (new, `ArrayDict` and `DataStructureColumn`): which vids have an entry, with no merge of a
  view's keys (`_kept_at_instances` used `np.isin(ids, prop.keys_array())`); the instanced entities known before the
  step are found once per location.
- **An object variable is read in one bulk read** (`MPGDataStructure._mtg_objects`: the column's `at()`, with the
  templates' values on an MPG with instances), not one dict lookup per vertex.
- Bit-identical.

### Wiring rules in bulk (devplan_scene_parallel, after P5: bulk wiring)

- **A wiring rule is evaluated once over every pair it joins, not once per pair** (`MPG.wire_junctions`, new
  `data_structure.wiring`): the (linked parent, child) pairs and each side's Compartments come as one
  `wiring.SectionPairs` (`parents`, `children`, the sections' Compartments as one CSR by label in order of first
  appearance, `parent_compartments(i)`, `child_compartments(i)`, `compartments_of(vertex)`). A callable rule is
  `{"pairs": f, "edge_label": E}` with `f(g, section_pairs) -> (pair, tails, heads)`, its properties
  `p(g, section_pairs, pair, tails, heads) -> {name: one value per junction}`. **The per-pair form
  `{"rule": f(g, parent, child, parent_compartments, child_compartments)}` is gone** (a dict with `"rule"` raises,
  naming `"pairs"`), as is `MPG._junction_pairs`; the declared rules (`node_label`, `ordering`, `match`) are evaluated
  through the same entry. The junctions keep their order (pair by pair, then rule by rule, as each rule gives them),
  so their vids and every system's node and edge order are unchanged.
- **`_rewire_junctions` compares arrays:** `MPG.linked_parent_arrays(scale)` (new: every vertex and its linked
  parent, -1 for none; `linked_parents` is its dict) gives the changed sections in a few array operations; the anatomy
  signature (checkpointed) holds those arrays. Checkpoints of an earlier version do not load (their wiring rules and
  signature are of the per-pair form).
- Bit-identical; use case 1 at 8 plants, day 13: see the use case's README.

### One knob: `Compute(threads="auto")` (devplan_scene_parallel P5)

- **`threads="auto"`, the default:** the threads' budget is the processors the process may run on (its CPU affinity,
  `compute.machine()`, which also records a hybrid CPU's core types), one in a pool's process (the pool is the
  parallel work there), `processors // processes` for each scene of several `run_variants` processes; an integer is
  the budget itself. `Compute()`'s default (threads 1 since P3) solved the plants' blocks in sequence: they are in
  threads again.
- **Each parallel section takes the threads that pay:** at most the budget and one per independent work unit (a
  kernel's chunks, a solve's blocks), within its ceiling (`Compute.section_threads`, `Compute.parallel` around a
  kernel's call). The reduction's scatter, bound by the memory bandwidth, takes at most
  `compute.BANDWIDTH_THREADS_RULE` (4). SuperLU's blocks find their count on the system's own solves
  (`solve.linear.BlockThreads`: 1, 2, 4, ... and the most allowed tried twice each, the fewest within 10 % of the
  fastest kept, tried again when the system has grown fourfold); it replaces `solve.linear.SUPERLU_BLOCK_THREADS`.
  A solver says how its blocks run (`LinearSolver.block_section`: `"bandwidth"`, `"compute"`, `"sequence"`; it
  replaces `block_threads_max`). A synthetic calibration was measured and rejected: SuperLU on synthetic blocks
  scales to 2 threads or not at all on a laptop where the plant system's blocks reach 4. The results are the same,
  bit for bit, at any number of threads.
- **run.json** records the threads chosen (`"compute": {"threads": {"requested", "budget", "bandwidth_threads",
  "processors", "core_types", "blas", "numba"}}`) and each kernel's and solver's time, calls and threads
  (`"compute": {"timing": {...}}`: `solve.<solver>`, `solve.kron_reduction`, `kernels.scatter_weighted_blocks`,
  `kernels.group_pattern`, `kernels.chain_scan`, `kernels.path_window`); `benchmark.py --threads auto|n` (default
  `auto`) records them per step.

### The solve and the threads' budget (devplan_scene_parallel P3)

- **One thread budget, `Compute(threads=)`:** numba's kernels and the independent blocks of a linear solve (in
  threads; they followed `workers`, the process pool's setting). BLAS and OpenMP are held at one thread within the
  scene (`compute.BLAS_THREADS`), whatever `threads`: a threaded BLAS changed the results' last bits with its thread
  count (up to 1.5e-11 relative on use case 1's potentials) and only oversubscribed the step's small dense work. The
  results are now the same, bit for bit, at any number of threads. SuperLU's blocks use at most
  `solve.linear.SUPERLU_BLOCK_THREADS` (4) threads: its factorisations are bound by the memory bandwidth and got
  slower beyond on a laptop. `Compute()`'s default (threads 1) solves the blocks in sequence.
- **A block's matrix gathered, not sliced:** a matrix's independent blocks are closed (their columns hold their rows
  only), so each block's matrix is a gather of the matrix's values at positions found once per pattern
  (`solve.linear._Block`), bit-identical to `matrix[rows][:, rows]` (8 plants, day 13: 0.14 s a solve → 0.09 with a
  new pattern, 0.02 with the same).
- **Optional solvers PARDISO and MUMPS** (`Compute(linear_solver="pardiso" | "mumps")`, extras
  `openalea.metafspm[pardiso]`, `[mumps]`): registered only when their package (pypardiso, python-mumps) is
  importable; naming a missing one raises with its install line. Each keeps its analysis per block while the pattern
  does not change. PARDISO's blocks run in threads (each call on one MKL thread), MUMPS's in sequence (it is not
  thread-safe). SuperLU stays the default: use case 1's plants grow every hour, so every step has a new pattern, and
  SuperLU is then the fastest at 2, 8 and 16 plants. Their solutions differ from SuperLU's by up to 3e-12 relative
  a solve (another pivoting, the same residual), 8e-10 on the potentials over 3 days at 2 plants; each gives the
  same bits at any thread count.
- **Kernel:** `scatter_weighted_blocks` keeps one k × k buffer a chunk instead of allocating one per group in the
  parallel loop (bit-identical).

### Kernels for the scene's serial parts (devplan_scene_parallel P2)

- **Traversals of the whole scene:** `MPG.components_at_scale` lists each scale's components in one loop, in
  openalea.mtg's order (its nested generator per vertex cost 0.25 s a step at 8 plants, day 12).
  `MPG.linked_parents(scale)` gives every vertex's linked parent at once, cached until the next structural edit;
  the anatomy signatures read it. GRANAP's anatomies read their classes, records and 3D quantities in bulk, and
  sum the pass-through junctions' lengths in a kernel.
- **`update_topology` in bulk:** a location's float variables are carried over as one gather (their old arrays
  stacked, the known entities' columns gathered at once, the new ones filled per variable: default or inherited
  source), each variable a contiguous row of the new block; the anatomy signatures take each Compartment's owner from
  one gather; `wire_junctions` reads the involved Compartments' labels in one read and every target's linked parent
  from `linked_parents`.
- **A reduction's terms by kernels** (`kernels.group_pattern`, `kernels.scatter_weighted_blocks`): the sparse pattern
  of the reduced groups' dense blocks built directly as scipy's canonical CSR, each entry's slot in its data int32
  (half the permutation it replaces); the values, each group's kinds' D weighted corner by corner, scattered by a
  parallel kernel over chunks of groups made from their nonzeros (`kernels.work_chunks`): bit-identical to the numpy
  path, and at any thread count.
- **Benchmark:** `benchmark.py --threads` (`Compute(threads=)`); `--thread-scaling` also times the reduction's terms
  kernel.

### The MPG's properties are columns, one storage (devplan_scene_parallel P1)

- **A tracked variable is not copied to the MPG:** when a DataStructure tracks a variable held at its own scale
  (its location's entities are the vertices of its `scale`, no mapping), the MPG's property is a view of the
  DataStructure's column (`data_structure.columns.DataStructureColumn`). It reads the current values at any time, no
  flush needed. The values written through the MPG (an MPG-style step's, the vertices the DataStructure does not hold
  yet) stay in the view's overlay until the DataStructure reads them back. New entities have no MPG entry until the
  next synchronisation, as with the copies. A variable mapped from another scale is still written back through the
  inverse mapping. Checkpoints keep the columns once (`arrays.npz`): the pickled MPG holds its views without their
  DataStructure, bound again on load.
- **Every other property is a typed column:** `ArrayDict` (float64, or an object column for any value). The MPG's
  property store (`ColumnStore`) keeps a dict assigned to it (`g.properties()[name] = {...}`, `setdefault(name, {})`)
  as an object column. A write of one key builds no key dict.
- **Bulk access:** `g.values_at(name, vids)`, `g.set_values(name, vids, values)` (existing and new vertices),
  `g.column(name)` (new: the values and their presence over the vid index), `g.view(name)` (vertex by vertex).
  **`g.property(name)` is openalea.mtg's read API only: a read-only mapping built on demand.** A write
  `g.property(name)[v] = x` raises: write `g.set_values(name, [v], [x])`. openalea.mtg's per-vertex readers (`label`,
  `edge_type`, `class_name`, `index`, `order`, `EdgeType`, `get_vertex_property`) read the columns.
- The platform reads the columns in bulk (`at()`, `items_arrays()`); the ports' sites moved (GRANAP, Adel, Caribu,
  ArchiSimple; their tests, examples and tutorials), CNW-Grass's coupled CN-Wheat keeps openalea.mtg's API on its own
  MTG.

### A nodal root's bearer is the element at its phytomer's node

- **`ds.bearer(location)`** gives each entity's element at its end: the last along its axis from its first (its
  successors `'<'` within the entity), whose end is `ds.position(location, at="end")`. A Phytomer's is its internode's
  top element, where its leaf's sheath and its nodal roots are; it was the phytomer's first element (its internode's
  base). A SubOrgan still bears itself.
- **ArchiSimple's nodal roots** grow `'+'` on that element and start at its end (`x2, y2, z2`), as any root borne by
  another model's element: its special case for a site's node is gone. `ds.check_connectivity()` finds no gap in use
  case 1 once the internodes elongate (a nodal root's base was an internode's length above its bearer's end).
- **MECHA's network:** a nodal root's xylem and phloem now join the shoot at its phytomer's internode top (the
  StemElement), not at the nearest anatomy below the phytomer's first element (a HiddenElement, without anatomy: the
  StemElement of the internode below, or the crown's top for the first phytomer). The water of a phytomer's nodal
  roots no longer crosses that phytomer's internode. This takes effect once the internode elongates (an element
  without length has no anatomy, its junctions pass through it): use case 1, 2 plants for 3 days, is bit-identical,
  its internodes unelongated (with nodal roots from the first leaf too, `nodal_root_rank_offset` 0).

### Memory per node and the scene's serial parts measured (devplan_scene_parallel P0)

- **A kind's group description builds no key dict of the plants' properties:** `Reduction`'s `_describe` reads each
  array-backed MPG property (`ArrayDict`) at its group by binary search. Its one-key reads for every new kind had
  built each property's key dict (`vid2idx`, about 100 B per entry), kept while the keys lasted: 8 plants at day 8,
  the MPG's properties 277 → 74 MB (111 → 30 B per entry), the process 2.62 → 2.41 GB, results bit-identical.
- **Use case 1's benchmark measures memory by owner and thread scaling:** `--memory` (`memory_probe.report`: the
  DataStructure's columns, the MPG's properties and topology, the Instances, the registries, the kinds' parts, the
  group tables, the libraries, the solver's caches, the pool's processes), `--tracemalloc FRAMES` (the allocations
  alive at the end by line and by stack), `--thread-scaling` (the last step's large solves and the bandwidth at 1–12
  threads); `memory_probe.profile_split` splits a cProfile's call sites into Python overhead and numeric work.

### Frames, small items and use case 1's run (O-12 part c)

- **Frames keep what their views show:** a `frame_options` variables filter names the data only; the recorder adds
  the columns the frames' views select, colour and size tubes by (`SceneRecorder.frame_variables`,
  `io.encoding3d.with_view_columns`, `view_columns`), instead of raising when the filter drops one. A filter's value
  (a whole one or a table's) may be `"recorded"`: the variables the declarations record, a declaration's new
  `record=` (`state_variable(..., record=False)`; by default state variables and inputs, not parameters, settings or
  a grid's geometric descriptors: `io.encoding3d.unrecorded`, `Excluding`).
- **Frames folded into the periodic pattern, for display:** `scene_to_frame(fold=True)` (and `frame_options={"fold":
  True}`), `io.encoding3d.fold_frame`: each triangle, segment and instance moved by the whole periods bringing its
  centre into `(0, xrange) × (0, yrange)`; a folded frame is not rebuilt (`from_frame` refuses it).
- **Grids hold their geometric descriptors** (`data_api.GRID_DESCRIPTORS`, `register_grid_descriptors`):
  `cell_volume`, `cell_size_<axis>`, `cell_section_<axis>`, `boundary_low_<axis>`, `boundary_high_<axis>`,
  `face_normal_<axis>`, set again by an adaptive grid after each refinement; its components' carry-over hooks follow
  (`@after_regrid`, `ds.add_regrid_hook`). The testing soil and use case 1's soil read them (their `set_layers` and
  `set_geometry`, `is_evaporating`, `is_water_table`, `evaporating_area`, `water_table_area`, `water_table_distance`,
  `volume`, `is_surface`, `is_bottom`, `boundary_area`, `half_height`, `gravity` are gone; use case 1's soil summary
  loses its `volume` and `boundary_area` columns).
- **Named physical equivalences in links** (`coupling.units.Equivalence`, `WATER_MOLAR_VOLUME`, `RADIUS_OF_DIAMETER`,
  `EQUIVALENCES`, `register_equivalence`): a factor between two quantities, its number computed from the declared
  units and checked (`declaration.check_link` returns the resolved link, `resolve_equivalences`; at initiation too);
  given by name in a YAML translator. Use case 1's two `unit_check=False` links are gone.
- **Chains at coarse locations:** `define_chain(name, location="Phytomer", group="Axis", rank="phytomer_rank")`, the
  group an owner at a coarser location or a variable; `chain_gather(..., source=None)` reads along each entity's own
  chain; a LocatedView has every chain kernel.
- **A component's values at its last call:** `kept_from_last_call = (names)` and `self.last_call(name)` (aligned by
  id, checkpointed), replacing hand-kept `previous_*` states (use case 1's nodal root rule).
- **Fixed-point coupling, as an option** (`coupling.iteration.FixedPoint`, `Scene(coupling=...)`; explicit stays the
  default): chosen schedule steps of several models iterated within a scene step from their start, after the
  exchanges into their models, until what they write stops changing; `Scene.coupling_iterations` and run.json.
- **Means over owners of no entity are NaN** explicitly (`np.divide(where=)`), the MPG's duplicated aggregations one.
- `SceneRecorder.pack()` after the run's end records the archive in run.json.
- **Use case 1's `run.py` rewritten** as a short script (settings, stand, `Scene(..., compute=Compute(...))`, recorder,
  `scene.simulate`, `scene.recorder.pack()`), its `report()` lines and `run.log` gone: the recorder's outputs hold what
  they printed. `models.py` holds the environment's scenarios, `weather(start)`, `TABLES` and `FRAME_VARIABLES` (now
  without the views' columns); `benchmark.py` builds its scene itself (`build_scene`).
- **Docs:** MECHA's `leaf_ports="bundles"` is an option only, not a recommendation for large scenes (it gives up each
  vessel's own axial path and junction to its successor, a bundle's vessels held at one potential; it biases the
  leaves' water potential): `docs/user.md`, use case 1's README, MECHA's `doc/equations.rst`.

### Composition and schedule, positions, plant parameters, one stand (O-12 part b)

- **A model declares its parts and its schedule** (`coupling.composite_wrapper`, `coupling.schedule`):
  `parts = {name: class | Part(class, **keywords)}` (components, or nested models: `ScenarioShoot` within a larger
  plant, its components, initiators and translator's links the model's; built on the model's DataStructure and
  coupled once), `schedule = (part names, method names, Step(name, stages=, cadence=, name=), Together(...))`
  (default: the parts with steps, in order); `__init__` overrides hold only what a model adds (a mask). A plant
  model's `__init__`/`couple()`/`run()` listing is gone (`couple()` stays an environment model's for its own
  components). `model.steps()` gives the schedule bound, each step's `reads` and `writes`; `conflicts(a, b)`;
  `Scene.schedule()` the scene's table.
- **The cadence is a schedule step's:** `Step("anatomy", cadence="1 d")`, `Cadence("4 h", at="start")`, the model's
  scenario `{"cadence": {step: duration}}` (and `set_cadences`). `Component.cadence` and `EnvironmentModel.interval`
  with `due()`, `full_run()`, `between()`, `after_run()`, `last_run` and the scenario key `"interval"` are removed. A
  `Step(stages=)` runs a component's stages only: a split schedule (Caribu's lighting every 4 h covering the
  interval, its PARa every step). A geometry input relights a start cadence early. `Scene.calls` counts the steps a
  component ran in.
- **Positions:** `x1 ... z2` are topological (each element's base at its parent's end); `display_end("x1")` ...
  declare where an element is drawn (`display_x1 ... display_z2`, NaN at its topological place); the default
  cylinders draw the display ends (`coupling.geometry.segment_cylinders`, `drawn_ends`), also the DataStructure's
  without a geometry provider. `ds.position(location, at="base" | "end")` (a Phytomer's node: its end) and
  `ds.check_connectivity(tolerance=)`.
- **Plant parameters and builders:** `StructuralComponent.plant_parameters(scenario)` (the scenario over the
  declared defaults, checked against their bounds and dtype); `MPG.add_organ(within, label, kind=)` and
  `MPG.add_element(label, complex=, parent=, edge=, direction=, display=, **values)` (topological ends).
- **One stand:** `stand(plants, rows=1, row_spacing=0.15, plant_spacing= | sowing_density=, depth=0.025,
  jitter=0., rotation="random" | degrees, models=, frequencies=, seed=)`, the rows along x (replaces the former
  `stand(xrange, yrange, ...)`, Adel's `sown_stand` and the use case's). `Planting(x, y, depth, rotation)` replaces
  z (the planting table's `depth` column; the plant's z is the soil surface minus it). `planting(planting_times=)`,
  the column `planting_time` and the mask `"planted"` replace `emergence_times`, `emergence_time` and `"emerged"`.
- **A component on an entity view exchanges with a grid** (`Exchanges`, through a CrossMapping: its entities'
  values among the mapped ones): the intercrop example's `SoilContact` is gone.
- Use case 1: `WheatPlant` nests `ScenarioShoot` (one listing of its components), `WheatCrown` places its elements
  with the builders (its `triangles()` gone), `NodalRootEmergence`'s site positions are gone (ArchiSimple reads the
  phytomer's node), the soil's adaptation is a method step, `run.stand` is metafspm's `stand`.

### Times with their units, intercropping, per-component cadence (O-12 part a)

- **Times and durations carry their units** (`openalea.metafspm.coupling.durations`: `duration("12 h")`,
  `"1 d 12 h"`, `"30 min"`; a number is seconds, the one internal unit; `instant()`, `start_date()`,
  `day_and_hour()`): `Scene(time_step="1 h")`; `Scene(start="2026-05-10 00:00")` (ISO 8601, UTC) replaces
  `start=(day_of_year, hour)` (`Scene.start` is a datetime; `Component.calendar()` still gives (day of year, hour),
  `calendar(default=)` and `set_forcings(start=)` take a date); `events=[("36 h", action)]`;
  `planting(emergence_times=["2 d", ...])` (or dates); `pulse("38 h", "39 h", total)`; `as_forcing(time_unit="h")`;
  `EnvironmentModel.interval = "4 h"`. Every `SceneRecorder` period is a duration (`frame_period="12 h"`,
  `checkpoint_period="1 d"`, `segment_period="1 d"` by default) or `"step"` (every step: the summaries' and tables'
  default), no longer a number of steps: an output is due when the scene time is a multiple of its period, each
  period checked at the first step to be a whole number of steps. `forcings.seconds`, `forcings.TIME_UNITS` and
  `forcings.time_unit_label` are removed.
- **Several populations of one model (intercropping):** the planting table's `population` column
  (`planting(..., population="deep")`, one name or one per plant; by default the model's class name; a scenario may
  be given per population name) makes one population per name, several possibly of one model. Components are named
  `"<population>.<Component>"` in a scene (`"<EnvironmentModel>.<Component>"`): a link by class reaches every
  population's component of that class, a qualified one that component only (two populations' components only by
  qualified names); the identity links the Scene adds name them so. Two models sharing a component class no longer
  raise. `scene.model(key)` and `scene.population(key)` (by name or class) replace the `isinstance` loops;
  `Population.name` is the population's name; `population_data_structure(name=)`. Checkpoint states are keyed by
  the scene names (a checkpoint written before restores no component state).
- **A component's cadence:** `Component(cadence="1 d")` (a constructor keyword like `time_step`, or set after) or
  `Cadence("3 h", at="start")` (`openalea.metafspm.coupling.cadence`): its calls cover the interval (its clock at the
  interval's start, its step the interval, its forcings over it); `at="end"` (default) at the interval's last step
  with the inputs it reads through its DataStructure averaged over the interval's steps (each entity's over the
  steps it existed, by id; given under private names), `at="start"` at the first step with its inputs then; its
  outputs held between its calls; the means checkpointed; `run_together` requires one cadence. `Component.calls()`
  and `Scene.calls` (in run.json) count the calls that ran per component. No default changes.

### Scaling fixes (the final benchmark)

- **Incremental topology updates** (devplan_release_optimisation.md O-14): each structural change of an MPG is
  logged as the ids it removed and added (`MPG.structure_epoch`, `MPG.structure_delta(since)`, computed from the
  topology arrays: no hook in the editing methods, so every port takes part as it is). The arrays derived from the
  structure are patched from it instead of rebuilt after each `update_topology`: `compartment_owners()`,
  `owned_compartments()` (no longer building `compartments_by_owner`'s dict, now built only when asked),
  `connection_vids()` (new), `structural_links()` (new: the Connections' endpoints at a scale), the nodes' owners
  at each coarse scale and the vertices of each scale in the plants (anatomy mode). A delta exists when no kept
  vertex changed its scale, complex, parent or anchor flag and no id was given again; otherwise the arrays are
  rebuilt as before. `METAFSPM_CHECK_TOPOLOGY=1` checks every patch against a build. `Instances.vids_having` and
  `vids_at_scale` are one vectorised gather per template (a Python loop per instance before) and, after instances
  are appended, extended at their next read by the new rows alone (not at each addition: 541 additions in 3 steps
  at day 14 cost 1.8 s when each extended every table). New `ds.edge_endpoints()`: the edges as a read-only (edges, 2) array
  (`edges()` builds Python tuples); `_parent_pairs` uses it. Use case 1, 8 varied plants resumed at h240 to h264:
  every step's results identical, the step 4.51 → 3.95 s.

- **Plants differing in a parameter the library does not read share their kinds:** a kind is keyed by the
  parameters and settings of the components the reduction's library runs (`_library_components`: the reduced
  systems' and those giving them an input at the cells), no longer by every component's. Before, plants with their
  own root draws (ArchiSimple's `random_choice`) had kinds of their own only: use case 1, 8 plants with
  `plant_variability`, every plant evaluated its 6 first kinds again (48 at the first hour instead of 6); to day 2: the kinds
  729 → 154, the run 160 → 61 s, the scope's peak 6.4 → 3.2 GB, every step's results identical. A
  checkpoint written before keeps its kinds under the former keys: they are evaluated again after a resume.
- **A forked pool is forked again when new templates appear:** when templates (or registry entries) were added
  since its processes started, the kinds' pool is closed and forked anew (`_lacks_templates`) instead of sending
  them; its processes then share the main process' templates instead of each holding its own unpickled copy (a
  day-6 anatomy: 16 templates, 480 MB pickled, once per process before). Use case 1, 8 varied plants to day 8: the
  scope's peak 7.27 → 5.08 GB, the time unchanged (358 s), every step's results identical. A spawned pool is sent
  them as before.
- **The linear solvers' patterns are bounded in bytes** (`PATTERN_BYTES_KEPT`, 64 MB over the systems, the least
  recently solved dropped first, the one being solved kept): a growing plant's pattern changes at every step, and
  its 8 former ones per system were kept (159 MB at 8 plants, day 7, growing with the plants' nonzeros).
- **`ArrayDict` reads a few keys by binary search:** the first `SEARCHES_BEFORE_INDEX` (64) single-key reads after
  a change of the keys no longer build the {key: index} dict (a Python object per key); a reduction's group
  descriptions (one key of every property) built one per property at each batch of kinds (96 MB at 8 plants).
- **A reduction's library keeps its ids bounded:** each batch of kinds replaces the library's former groups, and
  their ids are now given again (`MPG.reuse_ids()`: the id counter back to the last id in use, any property entry
  above it deleted; `MPGDataStructure.forget_ids_above(last)`: the removed entities forgotten, so that a new entity
  of a reused id takes its default, not the removed one's values). Before, the library's ids grew with every batch
  (3.8M at day 3 with 2 plants) and its topology arrays, rebuilt after each batch, with them. The kinds are the same
  bit for bit (use case 1, 2 plants to day 8: every step's results identical); the kinds' time drops by a fifth at
  day 8 and no longer grows with the run's length; the main process' peak 3.0 → 2.4 GB.
- **`Instances.add` is amortised O(1)** when the ids grow (the usual case): its columns (`owner`, `first`, `size`,
  `code`, now read-only views) live in buffers that double, and the owners' rows are kept instead of rebuilt at each
  addition (80k instances: 0.14 s, was minutes). Pickles hold the columns as before.
- **The pool pickles what it sends once:** new templates and registry entries (tens to hundreds of MB when new
  anatomies appear) are pickled once for all the processes that lack the same (`_sync_blob`), not once per process;
  use case 1, 2 plants to day 7: the kinds' time 87.7 → 58.8 s (a pool call with new templates 8.5 → 1.6 s).
- **`"auto"`'s estimate recalibrated:** `SECONDS_PER_CELL` 1e-5 → 3e-5 s per interior cell, measured on use case 1
  (3.2e-5 s over 2 plants to day 8, the library's update included), so that a batch of about a second of work does
  go to the pool.

### An optional GPU backend, and repeated runs in a process pool (O-8)

- **`Compute(device="cuda")`** selects the CuPy backend (`openalea.metafspm.backend`: `HostBackend` numpy/scipy,
  `CupyBackend` cupy/cupyx; `make_backend`, `cupy_modules`, `Compute.backend()`, `Compute.device_info()`):
  - a reduction's kinds (`KronReduction`) solve their many right-hand sides on the GPU, SuperLU's factors copied
    once (the factorisation stays SuperLU's, on the host), A_IK sent sparse, X copied back once; a right-hand side
    below `DEVICE_MIN_VALUES` stays on the host;
  - the new `"cupy"` LinearSolver (the default of `linear_solver` on cuda: `Compute.linear_solver=None` now means
    "superlu" on the cpu, "cupy" on cuda; "cupy" on the cpu raises): a system of at least `DEVICE_MIN_NONZEROS`
    entries solved whole on the device, its independent blocks together in one call (the batch), by cuDSS when
    nvmath-python is installed, else cupyx's spsolve; its pattern copied once per pattern;
  - the transfers explicit (`to_device`, `to_host`, counted: `run.json`'s `"compute": {"device": ...}`); the
    components' `device` setting defaults to it (Caribu's near field, unchanged);
  - CuPy is optional: nothing imports it unless the device is cuda; a missing CuPy, or no CUDA device, raises when
    the Scene is built (`Compute.check()`, replacing `check_linear_solver()`), with the install line; the pool's
    processes are spawned on cuda (CUDA does not survive a fork).
- Tested without a GPU on a fake CuPy (`test/helpers/fake_cupy.py`: device arrays numpy refuses, every crossing
  logged): the dispatch, the transfers, the results equal to the cpu path bit for bit (also use case 1, 1 plant,
  6 hours); GPU runs by hand: `benchmark.py --device cuda`, docs/gpu.md ("Running on a GPU").
- **`scenarios.run_variants(run, scenarios, workers="auto", compute=Compute(workers=1))`**: one scene per scenario
  in a pool of fresh processes (forked on Linux, spawned elsewhere and on cuda), the results in the scenarios'
  order; `variant_processes` bounds the total (a scene's "auto" pool becomes 1 when several variants run at once;
  `workers × Compute.workers` and `× threads` within the processors, else ValueError).

### Parallel kinds and the sparse solvers (O-6)

- **The kinds in a process pool:** `Compute(workers="auto")` (the default) starts a pool when a batch of missing
  kinds is worth it (at least 2 kinds and about a second of estimated work, 3e-5 s per interior cell: measured on use case 1), of
  `min(processors // 2, kinds)` processes, one processor left free, kept for the run and grown to the largest
  batch's need; an integer forces it (`1`: in the process). Each process holds a copy of the reduction (its
  declaration, plant model, templates and the plants' saved registries: sent only when new) and its own library
  population per plant parameters; the kinds are dealt by their interior cells and added in the requests' order:
  the same kinds, bit for bit, whatever the number. Forked on Linux (`compute.START_METHOD`), spawned on macOS and
  Windows (the plant model class importable by name); `compute.WorkerPool`, `WorkerError`, `worker_state`,
  `in_worker`; `Compute.pool()`, `pool_info()`, `close()` (called by `Scene.finish` and when a Scene is
  collected); `run.json` records the pool used.
- A library places a group from its description (`solve.reduction._describe`: its complexes, label and
  properties), not from the plants' MPG.
- **`solve.linear`**, the registry of the Newton steps' sparse solvers, `Compute(linear_solver=)`: `"superlu"`
  (default), `"superlu_reuse"` (the ordering kept while the pattern does not change, the factorisation while the
  matrix does not: SuperLU's solution bit for bit), `"cholmod"` (scikit-sparse, optional; the systems declared
  `@graph_system(symmetric=True)`, their held rows solved first; the others SuperLU's). A missing optional package
  raises when the Scene is built, with its install line. `LinearSolver`, `LINEAR_SOLVERS`, `linear_solver()`,
  `solve()`, `held_rows()`, `SymmetryError`.
- **`@graph_system(symmetric=True)`**: the matrix checked once per pattern, its held rows (a Dirichlet node's
  diagonal alone) aside. MECHA's standard problem declares it.
- A matrix of independent diagonal blocks of at least 200k entries (`solve.linear.BLOCKS_MIN_NONZEROS`: the plants
  of a population) is solved block by block, the blocks in threads per `Compute.workers`: SuperLU releases the GIL
  (measured: 4 solves in 4 threads as fast as in 4 processes, 1.5–2.1× on a 2 P + 10 E core laptop). The same
  blocks whatever the threads: the same results.
- `benchmark.py --workers`, `--linear-solver`; the peak memory of the run's control group (the pool included).
- Measured (use case 1, this laptop: 14 threads, 2 P + 10 E cores): 2 plants to day 8 with 1, "auto" (6) and 4
  processes, 8 and 16 plants to day 4 with 1 and "auto": the same results bit for bit. 2 plants day 8: run 235 →
  212 s ("auto") → 208 s (4), last day's step 1.97 → 1.83 → 1.68 s, the first step 6.3 → 4.1 s, the scope's peak
  3.4 → 4.3 → 5.0 GB; 8 plants day 4: last day 1.08 → 0.97 s; 16 plants day 4: 1.78 → 1.52 s. "superlu_reuse": no
  gain there and more memory (above).

### A reduction's groups in frames as instances (O-13, encoding3d 0.4)

- **`display="instances"`** (the new default of `SceneRecorder`, `to_frame` and `scene_to_frame`; `"full"`: the
  former drawing, every interior recovered as display rows; `None`: the reduced groups not drawn): any reduction, at
  any location, its groups written as an encoding3d instance set per display component, each group an instance of
  its template (its owner, transform, kept values of the unknown and kinds' weights); the templates (the shape and
  the full template's nodes, their constant properties) and each kind's recovery operator are shared parts written
  once in `frames/parts/<key>.glb`; the unknown on every template node is a derived field rebuilt by the reader for
  the instances selected (`encoding3d.evaluate`, ParaView's Select at scale), equal to `"full"`'s values.
- **The display hook:** a display component's `display_instances(owners) -> coupling.geometry.TemplateShapes`
  (shapes once per template in template coordinates, each triangle owned by its full template row; a 4×4 per group;
  `TemplateShapes.placed()`); without it, a DataStructure's groups are drawn as with `"full"`.
- **The reduction's side:** `Reduction.recovery_terms(ds, groups=, system=)` (`RecoveryTerms`: the groups' kept
  values then 1, their kinds and weights: a class blend two weighted terms), `Kind.recovery_operator()` (the
  recovery as M · [x_kept; 1] over the full template's Compartments: the identity on the kept nodes, a merged member
  its node's value), `Kind.recovery` (dense, derived while the factorisation is live up to
  `solve.reduction.DISPLAY_DENSE_MAX` values; above, the sparse system, solved by the reader), `Kind.rows()`.
- **`SceneRecorder`:** `display=`, `pack_at_finish=` and `pack_compression=` ("deflate", "lzma", "stored"):
  `frames.fspm.zip` when the run finishes (`recorder.pack()`), noted in `run.json`; the frames' index lists the
  parts its frames use, the unreferenced ones removed at the first frame written and when a run resumes; the
  scenario records the options.
- `filter_frame` keeps the instances of the kept owners (none when their owners' scale is dropped); a frame's
  variables filter also selects the templates' node columns. `default_views` declares a hidden instances layer per
  instance set, coloured by its derived field (`SECTIONS_COLORMAP`).
- GRANAP's `GranapAnatomy.display_instances`: its reduced sections' protoplasts once per full template, in the
  section's plane, each segment's transform.

### Dense shoot blocks (O-7): merged kept nodes

- **`Reduction(merge=)`**: kept nodes lumped by a declared grouping (`{"where": rule, "by": property names |
  callable labels, "properties": {name: "sum" | "mean" | "min" | "max" | "root_sum_square" | callable}}`, or a tuple
  of disjoint such dicts): one node per group (its representative, the first member), the members' explicit edges
  joined to it; the kind's full matrix and sources contracted (PᵀAP, Pᵀc) before the Kron reduction (an
  approximation, exact when the members' potentials are equal). `Reduction.merged_into(full)`; `Kind.contract`
  (a `@kind_derived` method's per-Compartment values summed into the merged nodes); `recover_interiors` gives the
  dropped members their node's value; `SectionTemplate.restricted(redirect=)`.
- Use case 1's `benchmark.py`: `--scenario KEY=VALUE`, the dense blocks' nonzeros per step, the kinds held and their
  pickled size, the leaves' water potential in the results.

### API generalisations (O-11, second half: checkpoint scenario, no process-wide state, balances, timing)

- **Resuming with the current scenario.** `Scene.load_checkpoint(path, scenario="checkpoint" | "current")` and
  `Scene.restore(path, scenario=..., **arguments)`: `"checkpoint"` (the default) runs on with the saved parameters
  and settings; `"current"` applies the scene's own scenarios (its planting table's, its environment scenarios)
  again after the load (only the entries they give).
- **Breaking: the time step is the Scene's.** `ds.simulation_time_step` (every DataStructure of a Scene, set by
  `Scene(time_step=)` and `population_data_structure(time_step=)`), `Component.simulation_step()`. The Choregrapher
  is no singleton and holds no run state (the steps registered at class definition only): **`Choregrapher().reset()`,
  `set_simulation_time_step`, `simulation_step()` and `scheduled_groups` are removed**; a component run without a
  Scene is given its DataStructure's `simulation_time_step` (none: one of its own steps per call). Scenes of other
  time steps run in one process. A component built without a time step runs one of the scene's steps.
- **Objects shared by content.** `ds.objects(name, saved=)` (`MPG.objects`, `MPG.share_objects` for a reduction's
  library), `data_structure.objects.ObjectRegistry`: one registry per DataStructure, its objects keyed by content,
  checkpointed with it, each once (a shared part, in its saved form; `saved=False` a cache). GRANAP's module-level
  registries are gone (its templates are `g.objects("anatomy_templates")`).
- **One store of caches.** `Component.caches()` (cached()'s values, `keep(saved=False)`'s, the graph systems' views,
  pieces and previous solutions), `clear_caches()`: `Scene.load_checkpoint` drops them all (no list of keys).
- **Declared conservation.** `graph_system(conserved={quantity: Conservation(amount=, inflow=, outflow=,
  unit=)})` (`solve.conservation`): the inflow and outflow integrated over the accepted sub-steps (a rejected
  adaptive trial counts for nothing), the change of what is held, the residual; `component.conservation(reset=)`;
  the SceneRecorder's `conservation.csv` (`SceneRecorder(conservation=True)`).
- **The values before the step.** `ds.previous(name, missing=nan)` (aligned by id), kept by the Scene
  (`ds.remember_previous`, `scene.keep_previous(ds, *names)`) for the variables a `Summary(max_change=[...])` asks
  for (`<name>_max_change`); `Summary(population=)` may name an environment model's DataStructure (a grid's
  `"cell"`).
- **Fixed: a leak of every reduction library.** Its structure class closed over the library and was held by the
  planting table's object column (invisible to the garbage collector): each library population, its plants'
  reduction and their templates stayed alive until the process ended (the port suites' memory growth).
- **Timing per component.** `scene.timing` holds `"<Model>.<Component>"` entries next to the models'
  (`Component.wall_time(reset=)`: its calls and its steps run together), in run.json.

### API generalisations (O-11: soil surface, receiver-private names, entity sets, link mappings)

- **One soil surface.** `Scene(soil_surface=0.)` (m): every DataStructure's `ds.soil_surface` (a component's
  `soil_surface`), the cross mappings' grid top (`CrossMapping(soil_surface=)`, the grid's depth `soil_surface - z`),
  the MPG's `g.soil_surface` while the initiators build (`population_data_structure(soil_surface=)`).
- **Geometry in a domain, vectorised.** `ds.geometry` (`coupling.geometry.Geometry`): `clipped(domain, where=)`,
  `fraction_in(domain, location=, by="area" | "length", weight=, empty=)`, `centre_in(...)`; domains `"air"`,
  `"soil"` (the soil surface's half spaces) or `HalfSpace(level, axis, above)`. `Triangles.clip(domain)` (one or two
  triangles per crossing one, in the triangles' order) and `Triangles.fraction_in(domain)` (exact area fractions)
  replace **`Triangles.clip_below` (removed)** and its per-triangle loop. Triangles carry per-triangle `attributes`
  (`"green"`: a triangle's green share; through `select`, `clip`, `concatenate`, an entity view's geometry).
  Geometric link sources `FractionIn`, `CentreIn`: variables the DataStructure derives (`ds.derive_geometry`) and
  recomputes when the geometry changes (`Geometry.stamp`, `StructuralComponent.geometry_stamp()`).
- **Receiver-private link names.** `Link(..., names={"green_area": "exchanging_area"})`: the link's values kept
  under the private name, which the receiver reads as its variable during its calls and `read()`
  (`ds.private_names(...)`, `Component.private_names()`), the shared variable untouched: a same-name variable can
  receive a formula or factor link.
- **Named entity sets.** `Component.entity_sets = {name: {variable: condition}}`, named `"Class.set"`
  (`declaration.entity_set`): `ds.define_mask(name, entities="GranapAnatomy.soil_nodes")`, and a Reduction's rules.
  **Breaking**: `Reduction(labels=)` is `Reduction(where=)`, a filter of the groups (`"label"`, `"<Scale>.label"`,
  properties) or a set name; `reduction.reduces(g, vertex)` (was `reduces(label)`); `ports=`, `keep=`,
  `explicit_edges=` take set names, `{variable: condition}` rules on a template's properties
  (`declaration.template_selection`), callables or tuples of them; `Reduction.is_reduced(template)`.
- **Mappings from the links.** `Link(..., mapping={"mask": ..., "method": ...})`: the Scene infers one CrossMapping
  per (population, grid, options) (`Scene._infer_mappings`, the mask combined with `"emerged"` under emergence
  times), `Exchanges` picks each link's; the hand-made `mappings=` CrossMappings of the examples are gone.
  `CrossMapping.outside()`: the entities with a part outside the grid's non-periodic axes, clipped into its border
  cells, warned once.
- **Forcings by panel.** `Component.forcing(name, over=(start, end))` and `forcing_panels(over=)` (the step in panels
  of at most an hour, `forcings.panels`): a model whose response to a forcing is not its mean's reads it panel by
  panel (Caribu's sun over a daily step).

### Vectorisation: no loop over groups or nodes; numba's kernels in kernels.py, warm_up

- A reduction's groups are arrays (`Reduction.group_table`, cached per topology; `Reduction.assignment`: each group's
  corners, weights and kinds, a loop over the distinct kinds only). `terms()` fills a sparse pattern cached per
  topology and node set (the toy's 1k/10k/80k groups: 0.017/0.38/16.6 s before, 0.0004/0.004/0.032 s after,
  bit-identical); `outputs`, the new `kept_outputs` (a kind's output of a value per kept node, its groups × kept)
  and `recover_interiors` are batched by kind; `Relative.corner_arrays`.
- The DataStructure's topology arrays are read-only and shared until the topology changes: `entity_ids` of every
  location (no copy per call), `owner`. **Breaking**: a caller writing into one copies it first.
- `update_topology` carries object variables over in bulk; no junction rewiring without wiring rules (a
  reduction's library).
- numba: metafspm's kernels in `openalea.metafspm.kernels` (the tree kernels' compiled loops, the turtle's frame
  operations, **moved from `coupling.geometry`**: `follow_parents`, `left_of`, `roll_right`, `down`, `turn_left`,
  `roll_to_vertical`; the segment pieces); `python -m openalea.metafspm warm_up` (`compute.warm_up`) compiles every
  kernel of metafspm and of the ports registered under the entry points `openalea.metafspm.warm_up`, at install and
  before the conda recipe's tests. conventions.md "Performance rules".

### Breaking: forcings read over the step; summaries of every component; checkpoint parts saved once

- Forcings are read over the step, not at an instant (the results no longer depend on the time step through them):
  `self.forcing(name, over="substep" | "step")` gives the forcing over the (sub-)step being computed `[t, t + dt]`
  (or the call's whole step) as `forcing_variable(aggregate="mean" | "total" | "instant")` declares it (the mean by
  default; the total, a rain's depth; the value at the step's end). `forcing_variable(at=)` and `forcing(name, at=)`
  are gone. `coupling.forcings`: `forcing_mean` (exact for numbers, tables and pulses, the 4-point Gauss-Legendre
  rule on panels of at most an hour for callables), `forcing_total`, `forcing_over`, `spatial_mean`, `pulse(start,
  end, total)` (a rate given by its total); `as_forcing`'s inputs have an exact mean. The Scene writes the
  undeclared forcings' means over the step into their inputs and parameters (a spatial forcing's per entity).
  `Component.forcing_interval(over)`.
- The recorder's summaries hold every component on the plants' DataStructure and on its views (located and entity
  views: Adel's, ArchiSimple's, the gas exchange's, the light's union), at every location, each variable over the
  entities of the components declaring it; intensive variables averaged by their declared weight
  (`state_variable(weight=)`, now also kept on grid and union declarations), a grid's by its cells' volumes, plain
  otherwise with one warning listing them; descriptors, edge variables and NaN values left out
  (`SceneRecorder.summary_plan`, `recorder.grouped`). Declarative tables: `Summary(location, where=, sum=, mean=,
  min=, max=, per=)`.
- `SceneRecorder(display_plants=)` and `scene_to_frame(display_plants=)`: the display shapes drawn for some plants
  only (the others' reduced interiors not recovered); display meshes share their vertices (`Mesh.indices`;
  `filter_frame` filters indexed meshes).
- `scene.shared`: `Shared(name, value)` parts of a `checkpoint_state()` and the sections' templates saved once in
  the checkpoints' shared folder (`Scene.checkpoint(path, shared=)`, `DataStructure.checkpoint(path, shared=)`;
  `shared.json`; the recorder's `checkpoints/shared`, pruned with its checkpoints). Use case 1 at day 3: 261 MB per
  checkpoint before, 4 MB per checkpoint and 169 MB shared once after.

### Reduction kinds by interpolated length classes, one library, bounded batches, checkpointed

- `Relative(width, interpolate=True)`: a group between two classes (1 + width)^k gets their two kinds' blend,
  weighted linearly by its value's place between them (`Relative.corners`; a blended `Kind` has `parts`, its D, e
  and outputs weighted, its inputs' terms and its interior, `Kind.recover`, its parts'); `interpolate=False` is the
  nearest class. The class kinds no group has are kept while their template, exact kind values and parameters are
  in use (they saturate); the exact ones are released as before. MECHA's `plant_reduction(length_classes=0.1)` and
  use case 1 by default; `"exact"` stays the exactness tests' reference.
- The missing kinds are evaluated in batches of at most `Compute.kind_batch` interior cells (`Compute.current()`:
  the Scene's Compute within its build and steps), in one library population per plant parameters and time step
  kept on the Reduction (`_libraries`, not checkpointed): each batch's groups replace the former's, its components
  reset to their built state, its MTG-backed variables read again.
- The kinds are checkpointed (`Reduction.__getstate__` keeps them; `Kind.shared_in_checkpoints`: a shared part
  of the checkpoints, saved once): a resumed run evaluates no kind again.

### Reduction of graph systems by groups

- `solve.reduction.Reduction`, declared on a plant model (`reduction = Reduction(location=, labels=, ports=, keep=, explicit_edges=, kind=, inputs=, systems=)`): each group of the location holds only its kept nodes (its ports and the nodes the model keeps), its interior eliminated exactly, once per kind (its full template, its kind variables' values or `Relative` classes, the plant's parameters, the settings), from a library population of the plant model's own components; the group's term is the full instance's Kron reduction minus what the model computes on the reduced structure. `Reduction.template_for` (a structural model's reduced templates), `full_template(key)` (the full template of a reduced one's key), `recover` (a group's interior), per-group inputs through the graph system's `@input_jacobian`.
- `@explicit_term(field, jacobian=)`: a graph system's term kept explicit (a nonlinear one, its nodes kept); `@input_jacobian(name)`: the balance's derivative with respect to a node variable.
- `solve.network.kron_reduction` / `KronReduction` (`reduced`, `pairs`, `shunts`, `inject`, `injection`, `recover`, `release`: its interior's factorisation freed, made again at each later call); it keeps the sparse blocks, not A_II⁻¹ A_IK (dense, interior × kept); `SectionTemplate.restricted` and `extended`.
- Memory of a reduction: a kind keeps no factorisation of its full interior (`KronReduction.release` once evaluated); `Reduction.group_kinds(release=True)`, at each solve's terms, releases the kinds no group has any more (a growing section's former length classes: `Kind.release`, its arrays freed even where a cache still holds it). A growing plant's kinds no longer pile up (36 to 120 MB each for a wheat stem's or blade's full anatomy: 30 GB by use case 1's third day).
- Reduced sections drawn whole in the frames' display shapes: `Reduction.recover_interiors(ds, groups=None, system=None)` (every reduced group's interior values from its kept nodes', batched by kind: one factorisation of each kind's interior for all its groups, freed after), `Reduction.full_rows(key)` (the full template's rows a reduced template holds), `Reduction.unknown(system)` (the reduced system's node unknown), `solve.reduction.display_ids` / `is_display_id` (the ids an output writes a reduced group's interior vertices under, stable from frame to frame, above every vertex id). `coupling.geometry.Triangles(owner_ids=)`: display triangles owned by id (owner -1), e.g. a reduced section's interior cells. An encoding3d frame with display shapes writes each reduced group's interior (its eliminated Compartments, their Cells and Layers) as display rows of their entity sets (column `metafspm_display_only`; extras `display_only`), linked to their Cells, Layers and section, with the reduced system's unknown recovered (e.g. MECHA's psi), their template's properties and NaN otherwise; a display mesh's triangles on them by id (up the complex links); `from_frame` leaves them out (the DataStructure stays reduced), `filter_frame(entities=)` keeps a section's with it.
- Frames filtered by variables per entity set: `filter_frame(variables=)` (and `scene_to_frame`'s, `SceneRecorder(frame_options={"variables": ...})`) takes, per data structure, the variables kept in every table or a `{table: names}` dict (an entity set, a relation or the scalars, by name; `"*"` every table it does not name, the others unfiltered), e.g. `{"Wheat": {"Compartment": ["psi", "tissue"], "SubOrgan": ["length", "radius"], "*": []}}`; it applies to the display rows too (a reduced section's interiors). A filtered table always keeps its structure (encoding3d's reserved columns: ids, labels, complex and parent links, edge types, a relation's ends, a grid cell's level and index), its segment ends and the display rows' marker `metafspm_display_only`; metafspm's own links (`metafspm_complex`, `metafspm_parent`, `metafspm_edge_type`, which only `from_frame` reads) only when named. The frame's extras note the filter (`{table: names}`). `io.encoding3d.filtered_view_columns(views, variables)`: the columns views select, colour or size tubes by that a variables filter drops; `SceneRecorder` raises a ValueError when its filter drops one (given views at construction; the default views, those of the frame unfiltered by variables, at the first frame). Use case 1: its frames keep a few variables (`run.FRAME_VARIABLES`), 89 MB instead of 389 MB for the third day's frame drawing the sections.
- `Kind.key` and `Kind.system`: a kind's key in its Reduction (its reduced template key, kind values and plant parameters' signature) and its system, the same when the kind is evaluated again; a component caching values derived from kinds keys them so and drops the keys its groups no longer have (MECHA's condensed volumes, which grew by one entry per kind, held by the kind object).

### Instanced sections and condensed solves (memory and time at the size of a plant)

- `data_structure.instances`: a section's vertices (e.g. an anatomy's Layers, Cells, Compartments and Connections) held once in a `SectionTemplate` (`SectionTemplate.capture` from one copy), each segment carrying an instance: `MPG.add_instance(owner, template)` reserves the ids a copy would get, with no MTG vertex; `MPG.remove_instance`. The MPG's array queries (`topology_arrays`, `_property_at`, `array_filtering`, the vertices of a scale), the DataStructures (nodes, edges, owners, labels, reads, junction wiring, updates, checkpoints) and the encoding3d frames see an instance as its copy: the same results, bit for bit. A property written at an instanced vertex overrides its template's value; the write-back skips instanced vertices (the DataStructure is their reference). `MPG.view(name)` reads a property vertex by vertex, instanced vertices included; `MPG.values_of(name, vids)` in bulk.
- `solve.network.condensed_solve(matrix, rhs, groups, kept=, cache=)`: a sparse system by static condensation on node groups, one factorisation per kind of group (equal blocks, or equal up to a factor); `graph_system(condense=)` (a coarse location, or a function of the component) uses it for a Newton system's linear steps, the nodes of a Robin or Dirichlet condition kept.
- A structural (MPG-style) step reads back only the state variables whose MPG property changed (`ArrayDict.version`, incremented by every change) or all of them when the structure changed; at instanced vertices known before the step and without an entry of their own, the DataStructure's values are kept (the MPG holds their template's). Within the step, `component.mtg` is the MPG as it is (the component's variables written just before), not every tracked variable written again.
- `solve.network.equivalent_conductance(groups=, cache=)`: its solve by condensation.
- Memory of the MTG write-back: `ArrayDict`'s key dict built at its first lookup of one key (bulk operations by binary search, `insert_many`); a structural step's topology signature without flushing to the MTG; the graph view's topology (incidence) and the edges cached per topology, as arrays.

### Upscaled variables and calibration

- `state_variable(upscaling=upscaling(of=, law=, protocol=, references=))` declares what a coarse-scale variable is the equivalent of and under which protocol; `declarations()` gains the `upscaling` column, and the encoding3d frames carry it in the variable's description.
- `solve.network.equivalent_conductance(view, edge_conductance, sources, sinks, source_conductance=)`: a network's equivalent conductance between two node sets and each source's share (e.g. MARSHAL's Krs and SUF on MECHA's anatomical network).
- `scene.calibration`: `read_observations`, `compare` (each observation's model value from the recorder, in its unit, and its residual), `residuals`, `objective` (the residuals as a function of parameters within their declared bounds, for scipy, MCMC or ABC tools; several virtual experiments).
- `scene.scenarios.from_samples` (a SALib sample as scenarios, bounds checked) and `design_table`.

### Breaking: one public way per thing (the API cleanup)

- **Fixed:**
  - `@graph_system(rtol=, atol=, max_step=)` reach the IVP solvers (`scipy_ivp_bdf`, `scipy_ivp_radau` always ran at their own defaults); one set of defaults, 1e-6 / 1e-9.
  - A missing translator file raises `FileNotFoundError` instead of starting an interactive builder.
  - A model whose `__init__` takes `**kwargs` no longer receives its components' scenario entries: they are set once, by the Scene (a setting's `choices` checked then too).
  - On grids, `index_of`, `random` and `define_mask` default to the cells; `available_vars()` lists the edge and scalar variables (`export` missed them).
- **Coupling:**
  - `CompositeModel.couple(*components, translator=None)` replaces `declare_data_and_couple_components` and `couple_components`; no translator means no link (an input named as another component's variable reads it). `couplings()` lists how each input is coupled: shared, alias, derived, scene, not provided.
  - `Translator.load(source)` is the one loader (a Translator, a dict, a module, a .py or YAML path); `load_translator`, `from_yaml` and `from_module` are gone, and so are `open_or_create_translator`, `translator_matrix_builder` and the text documentation (`get_documentation`, `.documentation`, `.inputs`: `declarations()` is the route). `Link.kind` is `Link.coupling`.
  - One default aggregation rule within and across DataStructures (`declaration.default_mapping`, with the column-grid layers); `ScalarMapping` and `cross_default_mapping` are private.
- **Declarations:**
  - `default=` everywhere (`initialize=` on state variables and inputs), `kind=` for `state_variable_type=`, `aggregation=` for `mapping=`: the former names warn for one release.
  - `declarations()` has a `role` column (state_variable, input, parameter, setting, forcing) and a `kind` column (extensive, intensive, ...), and shows `references`, `DOI`, `unit_comment`, `value_comment`. `edit_by` is removed; `by=` is optional.
  - Removed: `scale="node"` / `"edge"` / ... as a location (give `location=`), `edge_mapping=`, the edge aggregations `proximal` / `distal` (`child` / `parent`), `declare` (private), `plant_scale_state` and the Component list properties (`state_variables`, `extensive_variables`, ...), the per-component `forcings` table.
- **Time step:** no silent 1 s default. A component alone runs one step of its own `time_step` per call; without one, reading `self.dt` raises. The Scene's `time_step` sets the simulation step; `Choregrapher().add_simulation_time_step` is private, and `Component.__call__` takes no argument.
- **Scene:**
  - `Scene.from_scenario(scenario, plants, model, environment=[...], **options)` builds a scene from a scenario table's column (by model, with its forcings).
  - `Scene.finish()` replaces `Scene.stop()`; a recording scene dropped unfinished warns.
  - `Scene` is no `CompositeModel`; `scene.data_structures()` is the one walk of the scene (recorder, frames, checkpoints); `__call__` is removed.
  - The stand's size is `xrange` / `yrange` everywhere; environment models are `Model(plant_data_structures, xrange, yrange, time_step, ...)`, with `plant_data_structure()` and a required `compartment`.
  - `run_every` is removed (`EnvironmentModel.interval` is the cadence); `run_when` is internal.
  - A forcing may vary in space: a callable `(time, ids) -> one value per entity`.
  - `population_data_structure(table, model, time_step=)` is the public population recipe; `build_population`, `apply_plant_scenarios`, `apply_model_scenario`, `model_keywords` and `flat_scenario` are private. `initiate_plant(g, plant, scenario)`.
  - `SceneRecorder(folder=, segment_plants=)`; its `write_frame` and `checkpoint` are private.
- **DataStructures:** the framework hooks are private (the MTG readers and writers, `node_property`, `edge_property`, `export`, `incidence_matrix`: use `to_graph_view().incidence`); `topology()`, `to_props_dict`, `summarize` (the recorder summarises), `extract_algebraic` / `inject_algebraic`, `MPG.from_mtg` and `populate_graph_custom_connections` (`wire_junctions`) are removed. One word per concept: `parameter_view(name, location)`, `labels(location=)`, `order(traversal=)`, `derive(where=)`, `AdaptiveGridDataStructure.refinement_level()`, `to_dataframe(time=)` indexed by `time`.
- **Solving:** `graph_system(max_iterations=, tolerance=, finite_difference_step=, substeps=, integrate="once")` (the former names warn for one release); solvers by key only (the class form, `method=` and `SolverSpec` removed; `newton_optional_jacobian`, `linear_direct` and `ImplicitEulerSolver` removed, `implicit_euler` warns as an alias of `newton`); `node_balance(explicit=)`, `@priorbalance` and `@selfbalance` removed; the builder's internals private.
- **Testing:** `openalea.metafspm.testing` keeps the couplability checks; the legacy comparisons move to `testing.comparison`, and `metafspm.corrections` to `testing.corrections`.
- **Time units:** `as_forcing(time_unit=)` and `to_frame(time_unit=)` take seconds or a label (`"s"`, `"min"`, `"h"`, `"d"`).

### Breaking: one way to plant, force, record and describe a scenario

- **Planting:**
  - `planting(plants, model, scenario, plant_scenarios, emergence_times)` is the one planting function. Plants are named `p0`, `p1`, ... in the table's order.
  - `stand(xrange, yrange, sowing_density, row_spacing, ...)` lays out a sown stand: the former random stand, its size adjusted to the rows, with models drawn by frequency.
  - **Removed:** `planting_table` and `stand_initialization`. `planting`'s arguments `parameters=` and `plant_parameters=` are renamed `scenario=` and `plant_scenarios=`.
- **Scenarios** are flat, `{name: value}`. The `{"parameters": {...}}` nesting raises. `Component.apply_scenario` is removed.
- **Forcings, the one route for time-varying inputs:**
  - `Scene(forcings=)` takes `{name: forcing}` or a DataFrame indexed by time. A forcing is a number, a callable of the time, a Series or `(times, values)`.
  - A name no component reads as a `forcing_variable` is written by the Scene, at each step's start, into the uncoupled inputs and parameters of that name. A forcing nothing takes raises.
  - `as_forcing(value, time_unit=)` converts inputs given by day.
  - **Removed:** `CompositeModel.apply_input_tables`.
- **SceneRecorder:**
  - `Scene(recorder=SceneRecorder(folder, ...))` (or `recorder=folder`) replaces `output_dirpath`, `log_plants`, `heavy_log_period`, `frame_period`, `frame_options`, `logger_class` and `log_settings`.
  - It writes `run.json` (status, error, timing, versions) and `scenario.json` (with `forcings/`).
  - Summaries go in `Plant/<Model>/summaries.csv` and in `Soil/` or `Atmosphere/<Model>/summaries.csv` for the environment grids, by the model's new `compartment` attribute. The time column is now `time`.
  - It also writes `segments.csv` (`segment_period`), your own `tables=`, frames, and `checkpoints/` (`checkpoint_period`, `keep_checkpoints`).
  - Every output can be switched off on its own. The initial and last states are recorded. Without a folder, the tables stay in memory (`recorder.table(name)`).
  - `Scene.simulate` sets the run's status (finished, stopped, failed); `Scene.stop()` (now `finish()`) ends a `run()` loop. `simulate(0)` records a static scene's state, e.g. a section built once.
  - `Scene.timing` gives the wall time per model.
  - `frame_views="default"` gives each part its own colour map (ParaView presets, `io.encoding3d`'s `SHOOT_COLORMAP` "Cool to Warm" for the shoots, `ROOTS_COLORMAP` "Inferno (matplotlib)" for the roots, `GRID_COLORMAP` "Viridis (matplotlib)" for the grids): the shoot, the roots and the soil told apart at a glance.
- **Fixed:** an environment model's scenario entry that is a per-plant parameter on the plants' DataStructure (a light model's optics) was silently skipped. It is now set for every plant.
- **Fixed:** after a restore, the recorder's CSVs are cut back to the checkpoint's time. A run that went past its checkpoint duplicated their rows.
- **Warned:** a StructuralComponent working at a scale coarser than an anatomy-mode DataStructure's from_scale (its finest declaration at Organ, say) registers its geometry on that scale's view (`ds.at("Organ")`), which the frames and the light models never draw (they take `ds.at("SubOrgan").triangles()`): its construction now warns (`UserWarning`, naming the component and the view) instead of its elements silently missing from the frames and the light. Not routed to the drawn view: its triangles are owned by that scale's entities, which no element's mesh row or absorbed light can be given back to. The Compartments (an anatomy component's) are not drawn by design and do not warn.
- **Fixed:** a run resumed with a changed translator silently used the checkpoint's derived-variable formulas (`load_checkpoint` replaced the DataStructure's links by the saved ones). A DataStructure its models coupled (`CompositeModel.couple`) now keeps its own derivations on `load_checkpoint`: a changed one is recomputed from the loaded sources at its next read, a new one registered, an unchanged one kept as saved, a dropped one's variable left as saved; a changed alias raises `ValueError`. `DataStructure.restore` (no model) still takes the checkpoint's.
- **Scenario layer (`openalea.metafspm.scene.scenarios`):**
  - `read_scenarios` reads a scenario table: one column per scenario, with type, unit and description columns; forcing rows read from CSV files; an optional `model` column.
  - `parameter_table(*components, main=)` lists the parameters with their declared bounds.
  - `variants(base, ranges, design="grid" | "one_at_a_time" | "sampling")` builds scenarios within the declared `min_value` / `max_value`.
  - `declarations()` gains the `min_value` and `max_value` columns.

### Breaking: legacy APIs removed

- **DataStructures:** `LegacyMPGDataStructure` and `MPGDataStructure.from_legacy`, `GraphView.from_mtg_subset`, and the setters `set_node_property`, `set_edge_property`, `add_field`, `_get_field` / `_set_field`. Use `register()` to create a variable, and `set()` / `get()` for its values. The abstract `extract_state` / `inject_state` contracts go through `get()` / `set()`.
- **MPG:** `graph()`, `integrate_at_scale` and `average_at_scale`. Scale changes are DataStructure mappings: coarse-scale declarations, derived variables, translator links with `aggregation=`.
- **Choregrapher:** `build_schedule`, `add_schedule`, and running a schedule by class name (`Choregrapher()(module_family)`). A component is run by calling it, i.e. `Choregrapher()(instance=component)`.
- **Coupling:**
  - `CompositeModel.declare_data` is removed;
  - `declare_data_and_couple_components(translator_path=, components=)` no longer takes `shoot=` / `root=` / `atmosphere=` / `soil=`, since the components carry their DataStructures;
  - `Translator.inputs_outputs` is removed.
- **Examples:** the legacy-to-MPG migration demos are removed.
- **Internal solver layer:** `system_specs` (GraphSystem, GraphDAESpec, ODESystemSpec, BoundaryConditions), the `solve()` time loop, `SolverResult`, `LinearDirectSolver` and `make_solver` are internal. Graph systems choose their solver with `@graph_system(solver=...)`, as a class or a key.

### One time-term convention: `@node_rate` and `@node_balance`

- **`@node_rate(field)`** (new) writes a balance in the rate form: the method returns du/dt, with no time term. The framework adds the time term for the solver chosen:
  - backward Euler with the Newton family, `(u - previous(u))/dt - rate`, also with `integrate="substeps"` / `"adaptive"`;
  - forward Euler with `explicit_euler`;
  - `solve_ivp` with `scipy_ivp_bdf` / `scipy_ivp_radau`.

  One model can thus be compared across solvers by switching `solver=`.
- **`@node_balance`** (the residual form, time term written by the equation) is for the Newton family. `explicit_euler` and the IVP solvers refuse it, and refuse Dirichlet conditions. The node unknowns of one graph system use one form.
- **Breaking:**
  - `solver="implicit_euler"` is a deprecated alias of `"newton"` with `transient=True`. It used to add `(u - u_prev)/h` to every unknown, edge fluxes included, so equations writing their own time term got it twice.
  - `explicit_euler` evaluates the edge unknowns at the current state before stepping (it used the previous step's fluxes), and leaves the step's fluxes in the DataStructure.

### Components take their time step

- **`time_step=`** is a constructor argument of every component, the `dt` of its equations. Its default is the class's `time_step` attribute when set, else the simulation time step. Before, a component without that attribute had `dt` = 1 whatever the scene's step.
- **Sub-stepping by the time step:** a component whose `time_step` is shorter than the simulation step runs its schedule once per `time_step` within each simulation step (e.g. 4 times for 900 s in a 3600 s step). Its clock advances at each sub-step, so forcings are read at each sub-step's end. `sub_time_step` defaults to `time_step`.
- **Breaking:** the time step must divide the simulation step a whole number of times; otherwise construction raises `ValueError` (a longer step, or 1000 s in 3600 s, used to be silently rounded).

### Anatomy mode with Layer and Cell scales

- **Anatomies may hold Layers and Cells:** SubOrgan → Layer → Cell → Compartment as true MTG components (a cell's protoplast under its Cell), alongside Compartments anchored to the SubOrgan (walls and junctions shared by several cells). A Compartment's owner is its Cell, or its SubOrgan when anchored, and its entity at each coarser scale is the one above its owner (`MPG.compartment_owners`, `owners_at_scale`; `compartments_by_owner(from_scale)` gives every Compartment of each SubOrgan).
- **`"Layer"` and `"Cell"` are locations** when they own Compartments: `ds.owner("Cell")`, `entity_ids("Cell")`, `scale=scales.Cell` declarations, aggregations, masks and parameters.
- **Nodes without an entity at a scale** (walls at the Cell and Layer scales) have owner −1. Aggregations skip them, and broadcasts give them the variable's default.
- **Junction wiring, differentiation rewiring, growth inheritance and checkpoints** go through the owner chain.
- **Pruning a segment removes its whole anatomy:** `MPG.remove_vertex` removes the vertex's decomposition (Layers, Cells, their protoplasts, deepest first) and the Compartments and Connections anchored to it, with their properties. Before, a segment holding Layers could not be removed (the MTG refused a vertex with components).
- **Anatomy-mode inheritance in bulk, with a model's own rule:** `on_grow="inherit"` finds the sources of the new Compartments once per topology update, for every variable, instead of a search per Compartment and per variable. `MPGDataStructure.inheritance` takes a callable `(ds, new_vids, is_known) -> source vids` (−1: none); the default stays the first same-label Compartment upstream. GRANAP uses it to inherit from the same cell of the same template.
- **`convert_properties_to_arraydict` builds each property in one pass** (`ArrayDict.from_items`: one sort, one copy, one index map), instead of inserting the keys one by one (each insert shifted the arrays and remapped the suffix). GRANAP's 886 k-vertex seedling: 14.6 s → 3.6 s.
- **Wiring rules may give their junctions' properties:** a rule `{"rule": ..., "edge_label": ..., "properties": f}`, with `f(g, parent, child, pairs) -> {name: [values]}`, writes them on the junctions it creates (e.g. their geometry). The junctions then carry, from their creation, the values a declared edge variable needs on every edge.
- **Topology arrays kept up to date at bulk creation (DS14b):**
  - `add_components_bulk` records its vertices' parents and complexes, and the incremental `topology_arrays` reads these records instead of the MTG's dicts, vertex by vertex;
  - `replace_parent` counts as a structural edit, so the cached arrays are rebuilt (they used to stay stale after a reparenting);
  - `_valid_vids_at` is cached per structure (without filters);
  - the anatomy signature reads its columns in bulk.

  The same arrays: a new test checks the incremental ones against a full rebuild. GRANAP's seedling (1.46 M vertices): build 21 s → 13.6 s, a 9-segment growth step 2.2 s → 1.6 s, identical to its growth reference.
- **`ArrayDict.extend` and `MPG.set_values(name, vids, values)`:** values of new vertices, with ascending vids, are appended in one copy, without going through a dict and `update`. Other keys fall back to `update`. `add_components_bulk` uses it. Instantiating 9 GRANAP anatomies in a 1.46 M-vertex seedling: 2.2 s → 1.3 s (profiled); the seedling's build 27 s → 21 s; metafspm's suite 29 s → 17 s.
- **Removing vertices in one pass:** `MPG.remove_connections` removes the vertices with what `remove_vertex` removes with each (their decomposition, their anchored Compartments and Connections). Their properties are deleted once, and each component or children list they were in is filtered once. It used to go vertex by vertex: one bulk property deletion per Cell's protoplast, and a list search in the Compartment anchor's list for each wall. Where `remove_vertex` would refuse a vertex, it falls back to the vertex-by-vertex path. Replacing a segment's GRANAP anatomy in a 1.46 M-vertex seedling: 669 s → under 3 s.
- **Anatomy updates follow the change, not the plant's size:**
  - `compartments_by_owner` is cached until the next structural edit and grouped with numpy; `owned_compartments` gives the sorted array;
  - the anatomy signature is a table of arrays, and the changed vertices are found by comparing two tables;
  - stale junctions are found with sorted arrays;
  - `wire_junctions` groups only the vertices it wires and their parents, and creates its junctions in bulk.

  The same junctions and values as before. A 9-segment growth step of that seedling: junction rewiring 1.6 s → 0.56 s (profiled).
- **`ArrayDict.update` in bulk:** keys all above the current last one (new vertices, e.g. `add_components_bulk`) skip the per-key membership test, the index of appended keys is built in one call, and the sorted merge is one stable sort instead of a Python loop. A section's stored network (42 k Connections): its MPG writes 0.25 s → 0.07 s.
- **Bulk vertex removal:** `MPG.remove_connections` deletes the vertices' property entries with one compaction per array-backed property (`ArrayDict.delete_many`); per-key deletion rebuilt the index of the shifted suffix each time. Removing a 27 k-vertex network from a section: 301 s → 3.6 s.
- **`MPG.move_components(vids, complex)`** moves vertices to another complex at the same scale, their components following (e.g. Cells changing Layer on differentiation); the topology arrays are refreshed.
- **`add_components_bulk` takes one complex per component** (e.g. every protoplast of a section in one call). On a 45.7 k-vertex section, bulk creation takes 0.17 s against 1.0 s per vertex (`test/benchmarks/bench_layered_anatomy.py`).

### Adaptive grids in scenes

- **`CrossMapping` follows the target grid's refinement:** the incidence is rebuilt when the grid's topology changes, not only the plants'. Before, after an `AdaptiveGridDataStructure.refine()` or `coarsen()`, the roots were mapped onto stale cells.
- **Consistent fluxes between cells of different sizes:** `GraphView.edge_difference` (edges × nodes, tail − head, `incidence.T` by default). On adaptive grids, it moves the coarse cell's value onto the fine cell's axis through the face (`AdaptiveGridDataStructure.cell_gradient()`, a least-squares gradient), exact for linear fields. Edge laws written `k * (view.edge_difference @ u)` are consistent on any grid; with `incidence.T @ u`, the two centres' offset across a coarse/fine face lets the gradient along the face leak into its flux. Sub-views slice it, and the FD Jacobian's sparsity follows its stencil.
- **`cell_sizes()`** on `ArrayDataStructure` and `AdaptiveGridDataStructure`: each cell's size along each axis, so that structures can be written once for both grids.
- **Example:** `examples/soil_plant_atmosphere/one_plant_adaptative.py`, a soil refined where the roots take up water (`AdaptiveSoil`).

### Anatomy-mode DataStructures: variables read in bulk

- **Reading a coarse-scale MTG property onto the Compartments** (a component's construction registers its variables this way) mapped each Compartment to its owner one at a time, each through the whole owner map: quadratic in the number of Compartments. The owners are now found in one pass, and numeric ArrayDict properties are looked up by `searchsorted`. On GRANAP's seedling (212k Compartments), constructing the anatomy component took 477 s and now takes 0.6 s.
- **The Connection vertices** (`entity_ids("edge")` in anatomy mode) are cached by topology, as `edges()` is: the anatomy component's call went from 10 s to 0.8 s on the same seedling.

### Sparse analytic Jacobians

- **A `@graph_jacobian` may return a scipy sparse matrix**, kept sparse up to the Newton solve (`spsolve`). It used to be converted to a dense array, out of reach for anatomies of 10⁵ nodes.
- **The boundary sets' terms are added to it sparse:** robin weights on the diagonal, Dirichlet rows replaced by the identity's. The previous sparse path failed on a robin set (fancy indexing of a LIL matrix).

### Faster Newton steps on graph systems with edge fluxes

- **Edge unknowns are eliminated from the linear step** when their block of the Jacobian is diagonal, i.e. each edge law depends on its own flux only, as explicit laws do. The node system is then solved alone (an exact Schur complement). Before, the node rows, which hold only fluxes, had a zero diagonal, so the direct solver pivoted and filled in. On the SPAC soil at 1.25 cm voxels (8192 cells), one solve took 25 s and now takes 0.1 s, with the same results; the example now runs at 0.625 cm. Other systems use a symmetric fill-reducing ordering.

### Scenarios applied by the Scene

- **The Scene sets each scenario's entries on the model's components** after building the model, as constructor keywords would: numeric ones go to the DataStructure (per plant for plant models), the others to the components' fields. Before, only numeric plant parameters reached the components; non-numeric ones and every environment parameter were dropped unless the model forwarded them, and none did. (Scenarios were then nested in a `"parameters"` dict; they are flat since.)
- **Breaking:** an entry that is no component parameter, no argument named by the model's `__init__` and not read by its initiators raises `ValueError`. So does a parameter that differs between plants without being stored per plant. Models receive only the keys their `__init__` names.

### Breaking: one selection argument, `filters=`

- **Every decorator selects its elements with `filters=`:** steps, `@node_balance`, `@node_rate`, `@edge_law`, `@boundary_condition`, `boundary_set`, `@graph_output` and `@graph_system`. It replaces `select=` (boundary sets, boundary conditions, outputs) and `where=` (steps, graph systems).
- **Forms:**
  - a `{variable: condition}` dict, every pair holding: a value, a list of values, a comparison string (`">0"`, `"<=0.03"`, `"!=2"`), label names; a key may name a variable at a coarser scale, read at each element's entity;
  - a mask name;
  - a callable `ds -> boolean array`.

  A string is a mask name: write `{"is_collar": ">0"}` where `select="is_collar"` meant that variable.
- **Steps** still skip the entities outside the `"active"` mask; `include_inactive=True` replaces `where=None`.
- **Masks:** `define_mask` rules take any comparison threshold and variables of coarser scales.

### Example: water in the soil–plant–atmosphere continuum

- **`examples/soil_plant_atmosphere/`** (new): a seedling and a population on a soil grid under a dry air (a constant water potential), steady water flow, with plots. A smoke test checks convergence and the water balances. The user guide and the example's README describe it.
- **Framework changes it needed:**
  - `CompositeModel` takes a `Translator` (or a nested dict) directly;
  - grids expose their cell sizes (`dx`);
  - a scalar variable accepts a one-value array (a step computed from one-value parameters);
  - `@graph_output(location="node")` writes on a grid's cells;
  - `@graph_output(select=...)` computes an output on selected nodes only (the boundary-set forms), zero elsewhere.

### Declarations shared by plants and grids, and Connection properties

- **`location="node"` on a grid is its cells** (the nodes of its graph), so one component class declared on nodes and edges runs on a plant MPG and on a soil grid.
- **In anatomy mode, `scale=Connection`** declares the Connections' own MTG properties, read at registration and written back like node properties. The edges are the Connections themselves; before, such variables had no MTG property.

- **Scene `mappings=`:** a `CrossMapping` given there replaces the one the scene would infer between the same DataStructures, e.g. to map only a mask of the plant (`mask=`).
- **Labels:** `Layer.Mesophyll` and `Layer.StomatalCavity` for leaf anatomies.

### Parameters: one value, the DataStructure's

- **Breaking:** every numeric parameter now goes through the DataStructure, placed ones (`scale=` / `location=`) included, not only those stored per plant:
  - `model.k = value` writes it on every entity;
  - reading `model.k` outside steps gives the uniform value, or the array;
  - reading `self.k` inside a step or equation raises, so take `k` as an argument.

  Placed parameters used to be plain attributes, diverging from their DataStructure copy.
- **Breaking:** boundary ports set by hand (`component._boundary_ports`) are removed. Use `boundary_set`, or `@boundary_condition` for conditions given by equations.

### Breaking: compatibility views removed

- **`FunctionalComponent.props`** (the read-only `DataStructurePropsView`) is removed. Use `data_structure.get(name)`, or take the variable as a step argument.
- **`_last_graph_system` / `_last_graph_solution`** (the test introspection shim) are removed. Results are read from the DataStructure.

### Boundary conditions as equations: `@boundary_condition`

- **The method form stays** for conditions given by an equation. A `@boundary_condition` method takes its arguments by name, like the balances: unknowns and DataStructure variables, read at each solve and sliced to the selected nodes. So coupled values reach it dynamically.
- **`select=`** chooses the nodes as `boundary_set` does: a `{variable: values}` dict, a variable name (> 0), the name of a DataStructure mask, or a callable. `filters=` stays as the dict form. A `boundary_set` can also select by mask name now.
- **Breaking, Neumann sign:** a Neumann method returns an **inflow**, subtracted from the residual, as a `boundary_set`'s value is. It used to be added to the residual. Flip the sign of existing Neumann methods.
- **Errors:** kinds other than dirichlet / neumann, and `select=` with `filters=`, are refused.
- **Module rename:** `solve/legacy_functor.py` is now `solve/functor.py` (the step wrapper).

### Fixes found by the test audit

- **`forcing()`** accepts a `(times, values)` pair; it was taken for a Series, since tuples have an `index` method.
- **`Translator.to_nested()`** writes links with a scale, aggregation, weight or target in the long form
  `{"sources": ..., "scale": name, ...}` that `from_dict()` reads, so they round trip through YAML. Formulas and
  callable aggregations still cannot be written.
- **A same-name link that only states its scales** (`scale=` / `source_scale=`, no aggregation or target) is an
  identity once the scales are checked. The composite derived the variable from itself (a derivation cycle).
- **Population models in anatomy mode** (`nodes = "Compartment"`): the scene no longer populates the segment graph
  on top of the anatomies the initiators built, and takes the junction rules from a `wiring` class attribute (a
  list, or a callable `g -> rules`).
- **Scene checkpoints** keep the events not fired yet, whatever their time. An event timed between the last step's
  start and the checkpoint was dropped on restore.
- **`CompositeModel`** keeps a Python translator as a `Translator`, so its links' options (scale, aggregation,
  formula) apply.

### Breaking: module layout

The flat modules used before the 2026 restructure have been removed. No compatibility aliases are provided, so downstream packages must import from the new paths:

| Old import | New import |
|---|---|
| `openalea.metafspm.utils.ArrayDict` | `openalea.metafspm.data_structure.arraydict.ArrayDict` |
| `openalea.metafspm.utils.mtg_to_arraydict` | removed. Store variables in a `DataStructure` (`MPGDataStructure` for plants, the 3-D grid for soil). `MPG.convert_properties_to_arraydict` covers the transition |
| `openalea.metafspm.composite_wrapper.CompositeModel` | `openalea.metafspm.coupling.composite_wrapper.CompositeModel` |
| `openalea.metafspm.scene_wrapper.play_Orchestra` | `openalea.metafspm.scene.scene_wrapper.play_Orchestra` |
| `openalea.metafspm.component_factory.Choregrapher` | `openalea.metafspm.coupling.choregrapher.Choregrapher` |
| `openalea.metafspm.component_factory` step decorators (`rate`, `state`, `actual`, `potential`, …) | `openalea.metafspm.solve.decorator` |
| `openalea.metafspm.component_factory.Functor` | `openalea.metafspm.solve.functor.Functor` |
| `openalea.metafspm.component.Model`, `declare` | `openalea.metafspm.coupling.component.Component` / `FunctionalComponent`, `declare` |
| `openalea.metafspm.specializer` | `openalea.metafspm.solve.specializer` |

### Adaptive grids; MultiGridDataStructure removed

- **New `data_structure/adaptive_grid.py`, `AdaptiveGridDataStructure`:** cell-based octree refinement over a regular base grid (periodic axes, `max_level`, 2:1 balance between neighbours).
  - **Graph:** cells are nodes and the faces between leaves are edges, with their real `face_area` and `face_distance`, so graph systems, boundary sets (`layer_mask(z=-1)`), masks, steps, `CrossMapping` (barycentre and overlap) and checkpoints work unchanged. `cell_volume()` gives one value per cell.
  - **Refinement:** `refine(criterion)` / `coarsen(criterion)`, between steps. Values are carried by volume overlaps: extensive values split or summed, the others volume-averaged or copied; edge variables are reset.
  - **Tested:** an unrefined grid equals `ArrayDataStructure`; a uniformly refined grid solves diffusion like the equivalent fine regular grid; locally refined solves conserve mass.
- **Breaking:** `MultiGridDataStructure` and `GridLevel` are removed. They had no variable store, 1-D-only operators and no users. Their tests, example scripts and plotting helper go too.

### A light model triangulating the populations itself

- **Test only, no new API:** a CARIBU-like toy component on the union of two populations of different models. It builds its own triangles from each element's geometry, shades by Beer's law, writes the absorbed light back to every population, and runs every 4 steps with its outputs kept.

### State outside variables

- **`Scene.checkpoint(path)` / `Scene.restore(path, *scene arguments)`:**
  - every DataStructure, loaded in place by the restored scene (`ds.load_checkpoint`, which takes the checkpointed MTG and drops the topology caches);
  - the models' and components' `checkpoint_state()` / `restore_state(state)` hooks, for external solvers (cmf, Campbell);
  - the scene's time, iteration and pending events.

  A restored scene continues bit for bit, and its recorder appends to the existing files.
- **Vector-valued variables:** `state_variable(..., shape=(k,))` (or `register(..., shape=)`) store `(entities, k)` arrays, e.g. MIMICS' pools per cell. They are:
  - carried over by growth, and validated;
  - used by steps;
  - exchanged component by component;
  - expanded into one column per component by `to_dataframe` and the recorder;
  - kept by checkpoints.

  They are not MTG properties nor graph-system unknowns.
- `UnionDataStructure` can be checkpointed and loaded in place.

### Scene services

- **`Scene(forcings=table)`:** one forcing table (a DataFrame indexed by time) shared by every model. A component's `forcing(name)` falls back on it.
- **Scheduling:** models may set `run_every = n` or `run_when(scene)`. On the other steps they and the exchanges into them are skipped, and their outputs keep their values.
- **`spin_up(scene)`:** an optional model method, run once after the scene is built.
- **`Scene(events=[(time, action)], stop_when=condition)`:** actions run at the start of the first step at or after their time; `simulate` stops when the condition holds after a step (`scene.stopped`).

### Mappings for the environment

- **Environment scalars:** a link between a population variable (at nodes or a coarse scale) and an environment variable stored at `"scalar"` is exchanged over every plant of every population. Extensive values are summed, intensive values averaged (weighted with `weight=`), and a scalar is broadcast, or split by weight for extensive values (`ScalarMapping`, created by `Exchanges`).
- **Column ↔ grid:** `LayerMapping(column, grid, axis="z")` links a 1-D column to the layers of a 3-D grid by the overlaps of the layer intervals, so thicknesses may differ:
  - grid → column: overlap-weighted layer means, or totals shared by fraction;
  - column → grid: broadcast over x and y, or totals shared over each layer's cells.

  Pass it to `Scene(mappings=[...])` (or a callable building it from the scene).
- **A plant in a stand, its roots out of the pattern:** `CrossMapping(periodic=(True, True, False))`, the default, folds a segment's pieces outside the grid's x and y into the cells of their periodic images (the stand repeated, the grid its pattern: a neighbour's roots come in as this plant's go out), each piece's length fraction kept, nothing dropped; z is clipped into the grid (a root below the grid in its bottom cells). Documented in the user guide and tested on the overlap and barycentre methods (`test_overlap_folds_segments_leaving_the_pattern_into_its_periodic_images`); the plants stay drawn at their own positions.

### Graph-system extensions

- **Pool unknowns at a coarse scale:** `@graph_system(pool_unknowns={"shoot_sugar": {"location": "Plant", "exchange": "collar"}})` adds one unknown per entity (e.g. the shoot phloem pool of each plant), solved with the graph by Newton.
  - Its residual is a `@pool_balance(field=...)`.
  - Equations take the pool as an argument and exchange through `self.pool_exchange(name)`, the sparse map between the exchange set's nodes and their entity's pool.
  - Pools follow `where=` and `split=` solves, are written back at their scale, and take part in adaptive steps.
- **`boundary_set(select=..., kind=None)`** is a selection only (for exchanges).
- **`boundary_set(..., kinds="variable")`:** each node's kind is read at every solve from a node variable (`boundary_set.CODES`: 1 Dirichlet, 2 Neumann, 3 Robin; otherwise no condition), e.g. a collar switching between a pressure and a flux.
- **Forcings:** `self.forcing(name)` interpolates a table of `self.forcings` (a Series indexed by time, `(times, values)` or a callable) at `self.forcing_time()`:
  - at the end of the current (sub-)step for implicit solves and steps;
  - at the evaluation time inside an IVP solve (a time hook of `ScipyIVPSolver`).

  Components keep a clock advanced at each call, set to the scene time by the Scene.

### Structure edits and inheritance

- **`MPG.remove_vertex(vid, reparent_child=True)`** works on segments:
  - their Compartments are removed first;
  - children are re-linked within their complex. openalea's `MTG.replace_parent` made a shared complex its own parent, which then looped `topology_arrays` forever.
- **The graph follows:** `extend_graph` relinks the children of a removed vertex, and a vertex inserted in a chain (`insert_parent`) is linked and its variables carried over (tested).
- **`topology_arrays` raises** on a cycle in parent links instead of looping.
- **Disabling inherited steps:** a step returning `None` writes nothing (Root_BRIDGES' way), and a subclass can list `steps_removed = ("name", ...)`.

### Reproducible random draws

- **New `data_structure/random_streams.py`:** counter-based draws (uniform, normal, exponential, integers), a pure function of (seed, stream, step, entity id).
- **`ds.random(...)`** and the component helper **`self.random(stream, ...)`**: its step counters live on the DataStructure and are kept by checkpoints; `random_seed` chooses the seed.
- Draws do not depend on the visiting order or on the other entities (rhizodep's global re-seeding is replaced by per-vertex streams with the same distributions).

### Tree kernels, round 2

- **`ds.fold(update, values, direction="up" | "down")`:** a level-by-level fold with a custom vectorised function. Each `FoldLevel` gives its nodes, their edge types, `children(values, op, edge=, where=, fill=)` (sum, max, min, all, any, count) and `parent(values)`. It is for nonlinear pipe models, death propagation, filtered maxima over laterals and turtle-like frames; tested exactly against rhizodep's and Root-CyNAPS' loops.
- **`ds.chain_gather(values, rank, chain=, source=)`:** read the value at a rank of another chain (a tiller's metamer reading the main stem's at `cohort + n − 1`).
- **`ds.chain_recurrence(update, values, chain=)`:** non-associative recurrences along chains, vectorised across chains.
- `ds.levels()` is cached per topology.

### Growth bookkeeping proportional to the growth

- `MPG.topology_arrays()` is extended for the vertices created since the last read (rebuilt only after removals). The vid → index dict is built lazily, graph builders look vids up in bulk, and the carry-over matches ids once per location.
- 1000 plants of 2 000 segments: 3.3 → 2.0 s per step.

### Removed: `solve/specializer.py`

- It was unused since the legacy path was removed. Numba is used by steps directly: a step calls an `@njit` function on the arrays it receives.

### Graph systems solved per connected piece

- `@graph_system(split="components")` solves each connected piece of the graph (of the active subgraph with `where=`), e.g. each plant of a population, on its own, with its own Newton convergence and adaptive steps. A piece gives the same values as its plant solved alone (to 1e-15 in the tests).
- The pieces are computed once per topology, each with a local incidence. The restricted-solve bookkeeping is now proportional to the subgraph: in-place scatters, previous fields updated at the subgraph's nodes, adaptive capture and restore of the subgraph only. This also helps `where=` solves.
- `split="whole"` stays the default: on one core the pieces' per-solve overhead makes `components` about 1.3–1.6× slower at 100 plants (see the design doc §16).

### Consumption shared in visiting order

- `tree_kernels.path_contributions` / `ds.path_contributions(budget, extent, values, where=, include=)` give each supply window of `path_window` element by element as (owner, supplier, contribution), emitted in visiting order. `tree_kernels.scatter_contributions` / `ds.scatter_contributions` then add `amount[owner] · contribution / total[owner]` onto the suppliers one after the other.
- Bit for bit equal to rhizodep's sharing of each apex's consumption between its supplying segments (tested with partial and maximal window overlap). Accumulating in array order differs in the last bits.
- `ds.order("post", convention="openalea")`: the post order of openalea's `post_order2` (the successor subtree first, then the branches in reverse insertion order), the visiting order of models written with it.

### Checkpoints

- **`ds.checkpoint(path, include_mtg=True)` / `MPGDataStructure.restore(path, mtg=None)` / `ArrayDataStructure.restore(path)`:** a folder holding:
  - `arrays.npz`: numeric variables, and the entity ids, checked at restore;
  - `manifest.json`: class, construction, variables with location and dtype, counters;
  - `state.pkl`: object variables, metadata, aliases, derivations, masks, write counters, the anatomy wiring state, and the MPG.

  A restored DataStructure continues bit for bit; components are built again on it and keep its values.
- Derivation formulas and mask rules are kept by reference: module-level functions or picklable objects. Lambdas are refused with that hint. The Scene's combined emergence mask is now a picklable `AllMasks`.
- **`LabelsConfig` pickles** (its per-instance label groups are rebuilt on load), so MPGs can be pickled.

### Per-instance scheduling

- Steps are registered per class, keyed by module and qualified name (`choregrapher.family_of`), and bound per instance at construction; `Component.__call__` runs its own instance's schedule. Several instances of one class run on their own DataStructures in any order, and same-named classes of different modules no longer collide.
- **Fix:** subclasses run their bases' steps, a redefined step replacing its base's. Before, a subclass without steps of its own silently ran nothing. The `inheriting` globals mechanism is removed.
- **Introspection:** `Choregrapher().schedule_of(cls)` gives a class's unbound schedule. `scheduled_groups` is keyed by `module:qualname`.

### Performance at population scale

- **Breaking: lazy MTG synchronisation.** `mtg_sync = "lazy"` is the new default, and the DataStructure is the reference. A state variable changed since its last synchronisation is written to the MTG:
  - before an MPG-style step;
  - when the MTG is read through `ds.mtg` / `component.mtg`;
  - by `ds.flush_mtg()`.

  Code reading the MTG object itself calls `ds.flush_mtg()` first; `mtg_sync = "after_call"` restores the former behaviour. Rates and states no longer pay the MTG write after each call, which was 97 % of their time at 2·10⁵ segments.
- **An MPG-style step flushes the variables its component declares** (states, inputs, parameters): it sees what it declares. Numeric MTG properties written by the DataStructure are `ArrayDict`s, so writes are array assignments (a plain dict made by MPG-style code is converted once); integer properties stay dicts. `MPGDataStructure.edges()` is cached until the topology or the MTG changes.
- **Faster growth bookkeeping:** `extend_graph`, the variable carry-over of `update_topology` and `topology_arrays` work on property arrays instead of MTG traversals and per-vertex loops. Writes to MTG properties that are not keyed like the nodes are batched.
- **Fix:** in anatomy mode, `update_topology` pairs the stored edge values with the ids they were built for. It used to rely on new Connections having the largest vids.

### Breaking: the one-plant-per-process scene is removed

- **Removed:**
  - `openalea.metafspm.scene.scene_wrapper`: `play_Orchestra`, the plant, soil and light workers, the CPU-affinity helpers and the queues;
  - `openalea.metafspm.coupling.coupler`: `Coupler`, `Transport`, `BufferPlantView`, `VoxelLocator`.

  Use `openalea.metafspm.scene.scene.Scene`, which also takes the package-level import `from openalea.metafspm import Scene, planting_table`. The plant ↔ environment exchanges go through `coupling.cross` (`CrossMapping`, `Exchanges`).
- **`stand_initialization`** moved to `scene/population.py`.
- **`CompositeModel`:** `soil_name`, `soil_inputs`, `soil_outputs` and `get_component_inputs_outputs` are removed; the translator query is `Translator.inputs_outputs`. A plant composite no longer registers its soil inputs and resets them to 0 at coupling: they keep their declared `initialize` until the scene's first exchange.
- **Tests:** the multiprocessing tests and their doubles are removed, with the `slow` marker. The plant / soil contract of the DataStructure doubles now runs in a `Scene` (`test/wrappers_tests/test_scene_contract.py`).
- **Benchmarks:** `test/benchmarks/bench_population.py`. On toy plants, the Scene's per-step overhead is flat, while the per-process scene's grew with the plants (0.27 ms against 6.6 ms for 12 plants). 1000 plants of 2 000 segments take 6.7 s per step in one process, of which about 4.5 s were later removed.

### The population scene

- **New `scene/scene.py`, `Scene(CompositeModel)`:** the populations and the environment models of a stand, in one process.
  - **Populations:** one per model in the planting table. Each is built once on an MPG holding all its plants, as `Model(data_structure, time_step, **scenario)`, with the class attributes `initiators`, `from_scale` and `nodes`. Numeric parameters come per plant from the table; other scenario entries must be shared.
  - **Environment models:** `Model(populations, scene_xrange, scene_yrange, time_step, **scenario)` builds its own grid or `UnionDataStructure`.
  - **Coupling:** mappings are inferred from the scene translator (a `CrossMapping` per population and grid pair, a `UnionMapping` per union) and run through one `Exchanges`.
  - **Step:** each environment model after the exchanges into it, then each population after the exchanges into it.
  - Component classes must differ between models.
- **Staggered emergence:** an `emergence_time` column (s) keeps a plant frozen until it emerges. Its nodes leave the `active` mask, combined with a model's own `active` rule, and the mappings (`CrossMapping(mask=)`); values of entities left out are kept.
  - MPG-style steps restrict themselves with the new `DataStructureComponent.active_ids()`.
- **`SceneRecorder`:** for each population, `summaries.csv` gets one row per plant per step (sums of the extensive and means of the intensive state variables). `segments.csv` gets the `log_plants`' segments every `heavy_log_period` steps. A `logger_class` hook remains for adapting an external Logger.
- **`planting_table`:** a new `emergence_times=` argument, and the stand size is kept in `table.attrs`.

### Links between DataStructures

- **New `coupling/cross.py`:**
  - `CrossMapping(plants, grid, method="barycentre" | "overlap")`: plant segments to grid cells. Barycentre is the reference soil map; overlap weights each cell by the segment's length fraction in it. The map rebuilds itself after growth or a coordinate write.
  - `Exchanges(translator, components, mappings)`: translator links between components on different DataStructures, run by `exchange(into=ds)` at the scene's fixed points.
- **Defaults** follow the variables' kinds:
  - extensive plant → cell: `sum`;
  - intensive plant → cell: `weighted_mean`, `weight=` required;
  - intensive cell → plant: `broadcast` (the overlap-weighted cell values);
  - extensive cell → plant: `split` by `weight=`.
- **Several populations** feeding one grid variable are pooled in one write: sums add up, means pool their weights, and splits share each cell among every receiving population. `zero_soil_inputs()` is no longer needed.
- **`UnionDataStructure` and `UnionMapping`:** the nodes of several populations, one after the other, so that one component (a CARIBU-like light model) sees every population. A component declared at the populations' node scale runs on a union unchanged. Values follow the parts' growth, kept by (part, entity id).
- **`Coupler`** now uses `CrossMapping` (same map); it and `Transport` stay until the soil port.

### Plant populations and per-plant parameters

- **Parameters per plant.** A numeric `parameter(...)` declared without a scale is stored per plant (`"Plant"` location) on plant DataStructures, and at `"scalar"` on grids. String, boolean and object parameters stay plain attributes.
- **Parameters are arguments.** Steps and graph-system equations take parameters as arguments (`def _rate(self, hexose, k)`), broadcast to the equation's entities: nodes, or edges through their child's plant. Uniform values come as zero-stride read-only views, varied ones as cached gathers, so numba steps get float arrays in both cases.
- **`self.<parameter>`:**
  - inside a step or an equation, reading it raises `AttributeError`, naming the argument to add;
  - outside, it reads the population value (or the per-plant values), and writing it sets every plant.

  MPG-style structural steps read per-vertex values with `self.parameter_values(name)`.
- **New `scene/population.py`:**
  - `planting_table(...)`: `stand_initialization`'s layout, with one scenario per plant possible;
  - `build_population(table, initiators)`: one Plant vertex per plant, each plant's initial structure built by `StructuralComponent.initiate_plant(g, plant, parameters)`, in order;
  - `apply_plant_scenarios(ds, components, table, plants)`: per-plant parameters from the scenarios.
- **Fix:** the declaration resolver no longer converts the defaults of fields that are not DataStructure variables, so string parameters without a scale raised before.

### Tree kernels

- **New `data_structure/tree_kernels.py`**, through `MPGDataStructure` methods, vectorised over every plant of the DataStructure:
  - `define_chain(name, edge_type="<" | group=, rank=)` / `chain(name)`: chains along '<' successors (`"axis"` by default) or by rank within a group;
  - `chain_scan(values, chain=, op="sum" | "max", reverse=, exclusive=)`;
  - `chain_shift(values, k)` and `chain_write(event, values, k)` (lagged neighbours and forward writes);
  - `accumulate(values, direction="up" | "down", op=)` (subtree or root-path sums or maxima, by levels);
  - `path_window(budget, extent, values, where=, include=)` (sums over ancestors up to a budget);
  - `path_compose(transforms)` (4×4 frames from the root);
  - `depth()`, `levels()`.
- **Values** may be vector-valued (`(n, k)`).
- **Implementation:** chain scans and path windows are numba loops, parallel over chains and over nodes. They add their terms in the order of the loops they replace, so rhizodep's distance from tip and supply for elongation are reproduced bit for bit (tested).

### Building the graph at population scale

- **Incremental growth updates.** `MPGDataStructure.update_topology()` (segment mode) calls the new `MPG.extend_graph(from_scale)`: only new vertices get Compartments and Connections, removed ones (pruned with `remove_tree`) lose theirs, and children whose linked parent changed are relinked. Every other Compartment and Connection keeps its vid, so edge values carry over by identity.
  - It falls back to a full rebuild only when a new vertex has no linked parent and its plant has others.
  - On 20 000 segments, an update after 10 new segments takes 0.42 s instead of 398 s: the full repopulation deleted the properties vertex by vertex, which is quadratic.
- **`MPG.add_components_bulk`** creates many components with one batched write per property. `populate_graph` uses it, with the same vids and properties (1.99 → 1.30 s on 20 000 segments).
- **Plants stay disconnected.** `populate_graph` chains orphan vertices only within one plant, so the plants of a population are no longer linked to each other.

### Scene: a failing worker stops the scene

- **Model construction** of the plant, soil and light workers now runs inside their `try`: a failure there stops the scene like a failure in a step. Before, a plant failing in its constructor left the soil blocked in its own constructor, waiting for that plant's first message, and the scene hung.
- **Exit codes.** `play_Orchestra` also stops the scene when a worker exits with a non-zero code.
- **`shutdown_timeout`** (default 30 s) bounds the wait for the workers after the scene stops. Workers still alive are terminated (then killed), the shared memory is released, and the scene returns `False`.

### MPG traversals without recursion, and topology arrays

- **`MPG.components_iter`** keeps openalea.mtg's order (component roots, then '+' children before '<' successors) without its recursive `pre_order`. `populate_graph` and the multiscale traversals work on long axes: a 20 000-segment chain raised `RecursionError` before.
- **`MPG.topology_arrays()`**: vid-indexed `parent`, `complex`, `scale`, `edge_type` and `is_anchor` arrays, cached until the MPG changes. `complex` is resolved for all vertices at once; openalea's `complex()` walks the parent chain of each vertex.
- **`MPG.complex_at_scale_array(vids, scale)`**, used by `MPGDataStructure`'s coarse-scale owner maps: 131 → 53 ms on 20 000 segments.

### Typed variables

- **`dtype=`** on `register` and declarations (`declare`, `state_variable`, `input_variable`, `parameter`):
  - `"int"`: labels, types and indices kept as integers, read from the MTG as integers; non-integral writes raise;
  - `"object"`: lists and records, one per entity, carried over by growth and written to and from the MTG by identity. Graph systems, derivations and soil transport reject them.
- **Label names.** Masks, boundary-set dict selects and graph-system filters accept label names, e.g. `{"label": ["RootSegment"]}`. They are resolved by `MPGDataStructure.label_code` / `resolve_codes` through the MTG's `LabelsConfig`: a label value (unique) first, then an attribute of the variable's own scale group, then any group. Ambiguous or unknown names raise.
- **`export()`** keeps each variable's dtype.

### Sub-stepping and adaptive integration of graph systems

- **`@graph_system(integrate=...)`**:
  - `"step"` (default, unchanged): one solve of `time_step`;
  - `"substeps"`: `n_substeps` solves of `time_step / n_substeps`, re-reading boundary sets between them;
  - `"adaptive"`: step doubling with `rtol` / `atol`, bounded by `min_step` / `max_step`. A step below `min_step` raises.
- **Time terms in equations.** Equations use `self.dt` (the current sub-step's length, equal to `time_step` with `"step"`) and `self.previous(fn)` (the state at the start of the current sub-step).
- **`previous(fn, at=...)`**: `at="solve"` gives the state at the start of the call's solve, and `at="step"` the state at the start of the component's call, for operator splitting between graph systems.
- **Fix: `ArrayDataStructure.laplacian()` on axes with a single cell.** It added a spurious `−1/h²` sink on such axes, e.g. a `(nx, 1, 1)` column lost mass. Such an axis now contributes nothing, like the face graph.

### MTG sync policy and read-only snapshots

- **`mtg_sync`.** The class attribute `DataStructureComponent.mtg_sync = "after_call"` (default) writes the state variables to the MTG after every call; `"never"` leaves the MTG untouched.
- **Parameter refresh.** MTG-backed parameters are re-read at the start of every call, as well as before each graph solve.
- **Read-only snapshots.** Graph-system snapshots of parameters and inputs are read-only views instead of copies: an equation writing into one raises `ValueError`. On 20 000 nodes, a snapshot of ten parameters takes 45 µs instead of 96 µs and copies nothing.

### Graph systems on grids

- **`@graph_system` solves on `ArrayDataStructure`s.**
  - Cell variables are flattened for the solve (C order) and written back on the cells, edge variables live on the faces, and graph outputs at "node" are cell variables.
  - Masks and boundary sets on cells (e.g. `boundary_set(select=lambda ds: ds.layer_mask(z=-1), kind="dirichlet", ...)`) work as on plant graphs.
  - `where=` takes a cell mask.
- **Flat writes.** `DataStructure.set()` accepts a flat array for a grid variable of the same size.

### Grid topology

- **`ArrayDataStructure` has a graph topology**: cells are nodes (flat C order), faces between adjacent cells are edges, axis by axis, oriented towards increasing coordinates (`B[lower, e] = +1`).
  - `topology()` / `to_graph_view()`, `incidence_matrix()`, `edges()`, `n_nodes()`, `n_edges()`, `face_axis()`.
  - The DataStructure itself also gets `topology()` as an alias of `to_graph_view()`.
- **Periodic axes.** `ArrayDataStructure(..., periodic=(True, True, False))` adds wrap faces (not with 2 cells, where they would duplicate the internal face). `locate()` uses the grid's periodic axes by default.
- **Edge location on grids.** An `"edge"` location holds face variables. `face_area` and `face_distance` are registered at construction, so `B · diag(face_area / face_distance) · Bᵀ / cell_volume()` equals `−laplacian()`.
- **`layer_mask(x=, y=, z=)`** gives cell masks of given layers, for boundary sets.
- **Declarations on grids** accept `location="edge"`.

### Translator link scales and targets

- **`scale` / `source_scale` are checks.** When a link states them, the receiver's and the sources' declared locations must be those scales', otherwise the coupling raises. Declarations stay the reference.
- **`target=` on links.** A link (Python `Translator().link(..., target=)`, or a nested spec with `"target"`) names a DataStructure mask. The mapped values go to its entities only, the others getting the receiver's default. The mask must be defined before coupling.

### Cross-scale links: default mappings and kinds

- **Default mapping.** A translator link between two locations that gives no `aggregation` is mapped from its provider's `state_variable_type`:
  - extensive: `sum` up, and an error down;
  - intensive: `mean` up, `broadcast` down;
  - massic concentration: `weighted_mean` up (the link's `weight` is required), `broadcast` down;
  - missing or mixed kinds: an error naming the link.
- **Aliases across scales.** A single-source factor-1 link whose ends are at different locations becomes a mapped derived variable instead of an alias.
- **Kinds on inputs.** `input_variable(..., state_variable_type=)` is optional. When given, it must belong to the provider's family (extensive or intensive), otherwise the coupling raises. `couplability_problems(..., data_structure=)` reports kind conflicts and missing mappings.
- **Coarse ↔ coarse mappings** (e.g. Organ → Axis, Axis → Organ) in `MPGDataStructure._map`.

### Anatomy mode: multiscale graphs assembled in the MPG

- **`MPGDataStructure(g, from_scale=SubOrgan, nodes="Compartment", wiring=[...])`**:
  - nodes are the Compartments of the anatomies held below the SubOrgans, keyed by their own vid;
  - edges are every Connection (anatomy edges and junctions), keyed by their own vid;
  - `"SubOrgan"` is a coarse location (owner = the Compartment's MTG parent), so step 1's mappings and `owner()` work across scales;
  - `scale=Compartment` declarations read the Compartments' MTG properties.
- **`MPG.wire_junctions(from_scale, rules, children=None)`** creates junctions between the Compartments of linked vertices:
  - `match="all" | "nearest" | "equal"` on an ordering property, or a callable rule;
  - junctions get `is_junction = 1`.

  `populate_graph_custom_connections` delegates to it, with unchanged results. New helpers: `linked_parent`, `compartments_by_owner`, `junction_vids`, `remove_connections`.
- **Growth and differentiation** (`update_topology()` in anatomy mode): only the junctions of vertices that are new, re-linked, or whose anatomy (or their parent's) changed are rebuilt. Every other Connection keeps its vid and values. `on_grow="inherit"` takes the same-label Compartment upstream.
- **Derived variables with a target.** `derive(..., target=mask)` gives values on the mask's entities only, the others getting the default (e.g. a SubOrgan concentration broadcast to symplastic Compartments).
- **Not in anatomy mode yet:** traversal (the assembled graph has cycles) and the repartition after structural steps (`NotImplementedError`).

### Boundary sets

- **`boundary_set(select=, kind="robin" | "dirichlet" | "neumann", value=, weight=, field=)`**, declared in a graph-system class, is assembled by the framework:
  - Robin `+ w·(x − v)` in the field's residual, Dirichlet rows `x − v`, Neumann `− v` (an inflow);
  - `select` is a `{variable: values}` dict, a variable name (> 0) or a callable, and membership follows its variables and the topology;
  - `value` and `weight` are DataStructure variables, read at each solve, or constants.
- **Jacobian.** With a user `@graph_jacobian`, the framework adds the sets' Robin and Dirichlet terms.
- **Active subgraphs and anchors.** Sets follow active subgraphs. Dirichlet sets and positive-weight Robin sets anchor steady pieces in the well-posedness check.
- **Boundary ports set by hand** (`component._boundary_ports`) are deprecated (`DeprecationWarning`).
- **New use case UC5, leaf transpiration** (`test/graph_system_tests/test_uc5_leaf_transpiration.py`), checked against the direct linear solve.

### Graph systems on the active subgraph

- **`@graph_system(where="active")`** solves on the nodes of a mask and the edges between them. The equations see the subgraph's view through `self._graph_view`; inactive nodes keep their values, and dropped edges carry no flux (their integrated amounts are kept).
  - The subgraph is rebuilt only when the topology or the mask's values changed.
  - An empty mask skips the solve.
- **`transient=`** tells whether the balance has a time derivative. It defaults to `True` for explicit and implicit Euler and IVP, and `False` otherwise. On an active subgraph, a steady system raises when a connected piece has no Dirichlet anchor, naming its nodes.
- **Boundary ports set by hand** cannot be combined with `where=` (`NotImplementedError`); boundary sets come in step 2e.
- **Previous values across solves** are now kept on every node, so that they survive a change of active subgraph.

### Repartition and the active mask

- **Repartition.** `StructuralComponent.partition_weight` (a node variable or a callable) enables the repartition, after each MPG-style step, of the other components' node variables by `state_variable_type`:
  - massic concentrations: amount split by weight, or dilution when the weight changed;
  - extensive: split;
  - intensive: copied;
  - NonInertialExtensive: weighted copy;
  - descriptors: `on_grow`.

  The rules are those of rhizodep's `post_growth_updating`. New inactive entities copy concentrations and hold no amount, and they are split when they become active. The weight before each step is recorded by the framework, so no `previous_weight` bookkeeping is needed.
- **Named masks.** `DataStructure.define_mask(name, rule)`, `mask(name)`, `mask_version(name)`, `has_mask(name)`.
- **Masked steps.** `StructuralComponent.active` defines the `"active"` mask. Vectorised steps compute on its entities when it is defined; `@rate(where=None)` opts out, and `where="name"` chooses a mask.

### StructuralComponent

- **New shared base.** `StructuralComponent` and `FunctionalComponent` now share `DataStructureComponent`: the DataStructure binding, declaration resolution, registration and MTG sync.
- **`StructuralComponent`** edits the plant's structure through `self.mtg`, the DataStructure's MPG, with the MPG's own methods.
  - **MPG-style steps** (decorated steps without arguments, e.g. `@potential`, `@segmentation`) run with synchronisation around them:
    - before: the component's declared variables are written to the MPG;
    - after: `ds.update_topology()` runs if the MPG's vertex count or last vertex id changed, then the declared state variables are re-read from the MPG.
  - **Array-style steps** (with arguments) are vectorised on the DataStructure, like functional steps.
  - The growth rows (`potential`, `actual`, `segmentation`, `postsegmentation`) order the steps within the component.
- **Metadata precedence.** When several components declare a variable, the one that does not declare it as an input sets its default, `on_grow` and `state_variable_type`. A component reading it as an input only fills what is still unknown. Before, the first component to register it won.

### Traversal and entity identity on the DataStructure

- `MPGDataStructure` gains, in local indices derived from the Connections and cached per topology version:
  - `parents()`, `children()` (CSR), `roots()`, `tips()`;
  - `order("pre" | "post")`;
  - `owner(location)`.

  A graph with several parents per node, or with a cycle, raises.
- `index_of(ids, location="node")` (every DataStructure) inverts `entity_ids`, and unknown ids raise `KeyError`. Grids have `index_of` but no traversal (`NotImplementedError`).
- Tests no longer use the private `_idx_to_vid` / `_vid_to_idx`, except the tests of the legacy DataStructure.

### Variable declarations: `scale` / `location` / `mapping`

- `declare` and its wrappers (`state_variable`, `input_variable`, `parameter`) take three new keys:
  - `location`: where the DataStructure stores the variable;
  - `mapping` (with `weight`): how values go between the MTG `scale` and the `location` when they differ;
  - `weight`: the weight variable of `weighted_mean`.

  One interpreter, `coupling.declaration.resolve_declaration`, now serves registration and the solver snapshot.
- **Behaviour change.** `scale=<a scale coarser than the graph's nodes>` without `location` is now **stored at that scale** (location `"Organ"`, …). Before, it was broadcast to the nodes. For the former behaviour, write `location="node", mapping="broadcast"`.
- Scale names are accepted as locations, and resolved against the graph built by the MPG traversal: `"SubOrgan"` is `"node"` when the graph is built from SubOrgan.
- Without `mapping`, the mapping follows `state_variable_type`:
  - `sum` up for extensive variables;
  - `mean` up and `broadcast` down for intensive variables;
  - `weighted_mean` up (a `weight` is required) and `broadcast` down for massic concentrations.

  Ambiguous cases raise `DeclarationError`.
- Edge mappings are renamed: `proximal` → `child`, `distal` → `parent`. `edge_mapping=` and the old names stay accepted for one release, with a `DeprecationWarning`.
- Declarations that cannot be resolved raise `DeclarationError` when the component is constructed, naming the field. This covers:
  - an unknown location;
  - a mapping on a variable stored at its own scale;
  - an edge variable without a mapping;
  - an edge `mean` mapping on a state variable;
  - a scale finer than the nodes;
  - a non-numeric default.
- `location="scalar"` fields are now registered on graph DataStructures too.
- A graph equation receiving a variable stored at a coarse scale raises, and points to `location="node", mapping="broadcast"`.

### MTG write-back through the scale mapping

- **State variables are written to the MTG after every component call** (`Component.__call__`, N4). Before, they were written only at the end of a graph solve, so `@rate`-only components never reached the MTG.
- **Values are written at the vertices of the declared scale, through the inverse mapping** (`MPGDataStructure.write_mtg`):
  - stored at its scale → as is (an Organ-scale pool is written at Organ vertices, no longer at the node vids);
  - broadcast from a coarse scale → the (weighted) mean of its nodes;
  - averaged to a coarse scale → broadcast back to the nodes;
  - on edges → at the child or parent endpoint.

  `MPGDataStructure.read_mtg` is the matching reader, used at registration and for the parameter refresh before each graph solve. That refresh now covers parameters at coarse locations too.
- **New declaration errors**, for state variables that could not be written back:
  - summed to a coarser scale;
  - an edge state at a scale coarser than the nodes.

  Writing a `parent`-mapped edge state where several edges share a parent raises.
- The DataStructure scale operators accept `child` / `parent` next to `proximal` / `distal`.

### Derived variables resolved at read

- **Per-variable write counters.** `register`, `set` and topology changes bump them; `write_count(name)` reads them.
- **Recomputed when read.** A derived variable is recomputed at `get()` when one of its sources was written since its last computation, in dependency order and in place, so earlier views stay valid. Before, it was recomputed only by the receiver's `pull_available_inputs`, at the start of its call. `refresh()` still forces a recomputation, and `is_stale(name)` tells whether `get` would recompute.
- **Read-only.** `set()`, or `set_node_property` / `set_edge_property`, on a derived variable raises `ValueError`.
- **Writes through views.** `ds.get(x)[...] = v` is not seen by the variables derived from `x`. Call `ds.mark_written(x)` after such a write. The Coupler's `push` now writes through `set`.
- **Observable change.** Reading a derived input after other components ran now gives its current value, not the value its receiver last used. The scene contract anchor (`test_ds_scene_contract.py`) is updated accordingly, and every computed value is unchanged.

### Output locations

- **Declared outputs** take their declared location.
- **Undeclared outputs give theirs:**
  - `@graph_output(name, location="node" | "edge")`;
  - `@rate(location=...)` and the other step decorators, with `locations={name: location}` for supplementary outputs. A scale name is resolved against the graph.
- **Total steps and 0-d values** stay scalar.
- **Without a location,** it is inferred from the shape only when exactly one of node / edge / cell matches, with a `DeprecationWarning`. When the shape is ambiguous (n == m) or unmatched, it raises.

  This replaces two rules:
  - the `size == n` guess for graph outputs;
  - "the location of the first input" for step outputs, which was wrong when that input was a scalar.
- **Filtered equations and boundary conditions** slice an argument by its location (node or edge), no longer by its length.
- A `@graph_output` whose location differs from that of the registered variable raises.

### Validation and failure modes

- **`DataStructure.validate(strict=False)`** (`MPGDataStructure`, `ArrayDataStructure`, through `validate_variables`) raises one `ValueError` listing every inconsistency:
  - an array whose shape is not its location's;
  - a dangling or cyclic alias;
  - a derived variable whose source or weight is missing or moved, or which is not stored at its declared location.

  `strict=True` also recomputes the up-to-date derived variables and detects writes made through views.
- **Where it runs:**
  - at the end of `CompositeModel.declare_data_and_couple_components`;
  - in `testing.couplability_problems` / `assert_component_couplable`, when given `data_structure=`. These also report declarations that do not resolve on it.
- **Missing variables raise instead of acting as zeros:**
  - a graph-system argument that is not registered raises `KeyError`, naming the component;
  - a missing filter variable raises (it used to select every entity);
  - integrated `{field}_amount` unknowns are registered explicitly at zero before the first solve.
- `@boundary_condition(location="edge")` raises `NotImplementedError`, since conditions were always applied on nodes; any other location raises `ValueError`.
- `MPGDataStructure(g)` infers `from_scale` from the populated graph.

### Conventions page

- **New `docs/conventions.md`:** graph incidence and order, grid axes, locations, `scale` / `location` / `mapping` with the default-mapping table, write-back, outputs, derived variables, `previous()`, `on_grow`, and failure modes.
- **Documentation test:** `test/data_api_tests/test_conventions_doc.py` checks its tables against the declaration resolver.
- **Updated for the new keys:** `downstream_migration.md` §2 and the checklist. The README points to both.

### Scene and coupling wrappers

- `CompositeModel.open_or_create_translator(translator_path)` now takes the **full path of the translator YAML file**. It no longer takes a directory to which `/coupling_translator.yaml` was appended.
- `play_Orchestra`:
  - New `sowing_depth` argument, which replaces the unused `max_depth`.
  - New `debug_runs` and `poll_interval` arguments.
  - The light model now receives `light_scenario` instead of `plant_scenarios[0]`. The meteo table is read from `light_scenario["meteo"]`, given as a csv path indexed by `t` or as a DataFrame. It replaces the hard-coded `inputs/meteo_Ljutovac2002.csv`.
  - The soil and light workers are now pinned to dedicated cores, like the plant workers, and one core is still left free.
  - On platforms without `cpu_affinity` support (macOS), pinning is skipped.
  - New `plant_model_frequency` argument: the probability of each plant model at each sowing position. It defaults to uniform, so it is implicit for a single model. Before, only `plant_models[0]` was ever used. A single `sowing_depth` value is shared by all plant models.
  - `clean_exit` is now False when a worker failed. Plant and soil workers exit with code 1 when their model raised.
- **Breaking, light model contract:** `light_model` is now built as `light_model(queues_light_to_plants=, queue_plants_to_light=, scene_xrange=, scene_yrange=, meteo=, **scenario)`. Its constructor must answer the plants' initialization messages, as the soil model does. Before this change, the light worker ran one step short of the plants: the first worker to finish stopped the scene, and every model ran `n_iterations - 1` steps.
- Soil and light workers no longer stop the scene when they finish normally. Only plants end a scene, and an environment worker sets `stop_event` only on failure. Before, the first environment worker to finish could make the other one skip its last step, so plants waited forever.
- The soil worker flushes its reply queues before `os._exit`. Its last replies could be lost, which made the scene hang under load.
- The plant shared buffers are zero-initialised (they used `np.empty`).
- New constants `scene_wrapper.HANDSHAKE_SHAPE` (the plant/soil buffer, still `(35, 20000)`) and `CPU_REGISTRY_FOLDER` (the `outputs/` folder is created when missing).
- `stand_initialization` picks plant models from cumulative frequencies (the pick was wrong with more than 2 models).
- The soil and plant workers run without a `logger_class`.
- `CompositeModel`:
  - A same-name link with a factor ≠ 1 is now applied across data structures (it was silently dropped) and rejected within one data structure.
  - Multi-source links of the soil keep all their sources.
  - `apply_input_tables` recomputes its variable selection when its targets change.
  - `documentation` / `inputs` work on any component.
  - `recursive_reload` is removed.

### DataStructures: prerequisites of the coupling refactor

- **New variable API on `MPGDataStructure` and `ArrayDataStructure`:**
  - `get`, `set`, `register`, `location`, `has`, `alias`, `aliases`, `version`.
  - `set` and the legacy `set_node_property` / `set_edge_property` / `_set_field` now **write in place** and check the shape. They used to rebind the array and accepted any length.
  - Aliases are name-level: `get(alias) is get(source)`. Cycles and shadowing are rejected.
- **Breaking, incidence sign convention:** `incidence_matrix()` now uses the solver's `GraphView` convention, `B[parent, e] = +1`, `B[child, e] = -1`. It used to be the opposite of `to_graph_view().incidence`.
- **`GraphView.node_local_index`** no longer assumes sorted ids (every lookup was wrong on MPG views) and raises `KeyError` for unknown ids.
- **`update_topology` carries registered variables over to the new topology.** Nodes are matched by vid, edges by child vid. New entities take the declared default, or the parent's value with `on_grow="inherit"`. The variables used to be cleared.
- `declare`, `input_variable`, `state_variable` and `parameter` take `on_grow="default" | "inherit"`, and `FunctionalComponent` registers its fields with it.
- **Scale-aware MTG mapping:**
  - `_mtg_to_node_array` / `_mtg_to_edge_array` take `scale=`, which maps a coarser-scale property through each node's complex.
  - The fast path checks the keys, not only the size.
  - Partial coverage raises `ValueError`; it used to fall back silently to the default.
- `write_node_to_mtg` / `write_edge_to_mtg` create a missing property and raise on shape errors. Every exception used to be swallowed.
- **`ArrayDataStructure`:** named axes `("x", "y", "z")`, plus `cell_centers()`, `cell_volume()` and `locate(points, periodic=, clip=)`, which returns flat C-order cell indices.
- **`LabelsConfig`:**
  - Integers now live on per-instance copies, so every MPG has a complete translator. The class attributes are no longer mutated.
  - `Connection.Apoplastic` is `"ApoplasticEdge"`; it used to collide with `Compartment.Apoplastic`.
- **Links on DataStructures:**
  - `derive(name, {source: factor} | sources + formula, location=, aggregation=, weight=)` and `refresh(name=None)` recompute derived variables in place. Dependencies are refreshed first, and derivation cycles are rejected.
  - `MPGDataStructure` gets new locations:
    - the coarser biological scales, named after `ScalesConfig` (`"Organ"`, `"Plant"`, …);
    - `"scalar"`, which holds plant-scale values.
  - Scale operators: node → coarse (`sum` / `mean` / `weighted_mean`), coarse → node (`broadcast`), node → edge (`proximal` / `distal` / `mean`), and to or from `"scalar"`. `entity_ids(location)` gives the entity order.
  - `ArrayDataStructure` gets a `"scalar"` location, with `cell` ↔ `scalar` aggregation and broadcast.
  - `update_topology` carries every location over growth.
- **Live reading:** `FunctionalComponent` no longer copies its DataStructure into a props snapshot.
  - **Graph-system solves:**
    - read the variables from the DataStructure at each solve (per-solve copies);
    - write unknowns, integrated amounts and `@graph_output` results back in place, registering them on first write;
    - take the implicit-Euler first-tick previous fields from it;
    - write the MTG directly from the DataStructure arrays.
  - `self.props` is a **read-only** `DataStructurePropsView` for compatibility, kept for one release.
  - Choregrapher steps run on the DataStructure arrays and **receive whole arrays**. `@rate(vectorized=False)` (and the same option on every step decorator) opts in to one call per element for scalar logic.
  - New `self.previous(fn)`: an unknown's value at the start of the current solve, managed by the framework. The user-managed `_previous_fields` is deprecated.
- **Translators as objects:**
  - New `openalea.metafspm.coupling.translator`, providing `Link`, `Translator` and `parse_factor`:
    - Python-first translators with live `ScalesConfig` references, `formula=` callables, and optional `scale` / `aggregation` / `weight` on each link;
    - `Translator.from_yaml`, `from_dict`, `from_module(path)`, `load`, `to_nested`, `inputs_outputs`.
  - The YAML short form is unchanged. A long form is available: `variable: {sources: {...}, scale: Organ, aggregation: sum}`.
  - String factors are parsed by a restricted arithmetic parser (numbers and `+ - * / **`). **`eval` is no longer used** in `CompositeModel`.
  - `open_or_create_translator` also accepts a `.py` translator module.
- **`CompositeModel` on DataStructures:** when every component is DataStructure-backed, `couple_components` registers the translator links on the shared DataStructure:
  - identity: nothing;
  - alias: `ds.alias`, which replaces the receiver's own default;
  - factor, sum, formula or scale change: `ds.derive`, refreshed by the receiver's `pull_available_inputs` before its step. A same-name factor is rejected.

  Soil outputs are registered on the plant DataStructure and initialised to 0. `apply_input_tables` writes DataStructure variables. The props path is kept for legacy components until the doubles are retargeted. `DataStructure.unregister(name)` is new.
- **Plant ↔ soil `Coupler`:** new `openalea.metafspm.coupling.coupler`.
  - `VoxelLocator` finds each segment's cell from its barycentre (x1..z2 plant variables, names configurable), with `flip_z` and periodic x and y.
  - `Coupler.update_map` / `zero_soil_inputs` / `push` / `pull` do an extensive scatter-add to the soil, with factors, and an intensive in-place gather to the plant. It gives the same sums as the reference soil model's `apply_to_voxel_fast`, with the `(y, z, x)` → `(x, y, z)` permutation.
  - `Coupler.from_translator` builds a Coupler from a Translator.
  - `DataStructure.topology_version` makes a stale map raise.
- **Plant ↔ soil transport:**
  - `coupler.Transport` sets the buffer layout from the translator: rows `_n_nodes`, `_node_id`, the segment coordinates, the plant variables read by the soil, and the soil states. This replaces the `vertex_index >= 1` convention.
  - It provides `write_plant` / `read_soil` for the plant side, and `plant_view` for the soil side: a `BufferPlantView` on which the same `Coupler` runs. Capacity overflow raises `OverflowError`.
  - `play_Orchestra(handshake_shape=...)` defaults to the legacy `(35, 20000)`.
  - `FunctionalComponent` also runs on `ArrayDataStructure`: fields declared with `scale="cell"` or `"scalar"` are registered automatically.
- **Soil component name and sub time step:**
  - `CompositeModel.soil_name` sets the soil component name, which is no longer hard-coded to `"SoilModel"`.
  - `FunctionalComponent` runs once per simulation step, or per its own `sub_time_step`. It used to register a sub time step of 1, so a 3600 s simulation step ran it 3600 times per call.
- **Export for loggers:** `ds.export(names)`, `ds.to_dataframe(names, location, time)` and `ds.summarize(sums, means, scalars, where)` are available on `MPGDataStructure` and `ArrayDataStructure`.
  - `to_dataframe` is indexed by vid, edge child vid or voxel, plus `t`; cells also get x/y/z centre columns. `.to_xarray()` turns the result into a dataset.
  - `summarize` builds the plant-scale csv row.
- **`openalea.metafspm.testing`:** `couplability_problems` / `assert_component_couplable(component_cls, translator, name=)`, for downstream test suites.
- `FunctionalComponent._graph_view` is rebuilt when the DataStructure topology changes, so components keep solving after growth.

### Solvers (backlog B5, B6)

- Edge (algebraic) unknowns are recovered by solving the edge rows with the node unknowns held fixed. `ExplicitEulerSolver` and `ScipyIVPSolver` used to run a Newton solve on the whole quasi-static system, which gave the wrong fluxes, or a singular matrix for transient problems. `DAESolver.solve` stores the recovered values in the accepted state.
- `ScipyIVPSolver` returns the full packed state (nodes and edges) for specs with edge unknowns. It used to return the node unknowns only.
- `ScipyIVPSolver` no longer fakes an error estimate from the midpoint state. `solve_ivp` controls the error within each step, and the outer loop accepts the step. The old estimate made the step size collapse.
- `NewtonSolver`, `ImplicitEulerSolver` and `ScipyRootSolver` start from the state they are given instead of the spec's initial guess.
- `make_solver(method, config)` accepts a dict of `SolverConfig` fields (it used to replace them silently with the defaults). Unknown keys and other types raise `TypeError`.

### Breaking: legacy props path removed

Components are coupled only through DataStructures; the props-based path of the former downstream models is gone. See `docs/migration.md`.

- **Components:**
  - `FunctionalComponent` requires a DataStructure with a variable store (`MPGDataStructure`, `ArrayDataStructure`).
  - `Component.link_self_to_mtg` and the props-based `pull_available_inputs` are removed.
  - `self.props` remains as a read-only view.
- **Functor and Choregrapher:**
  - Steps run only on DataStructures: the `dict`, `ArrayDict` and `ndarray` branches of the Functor are removed, and so are the Choregrapher's `"length"` type probe and `focus_elements` computation.
  - The ArrayDict-only numba specialisation branch (`use_njit`) is removed.
  - `add_time_and_data(instance, sub_time_step, data, compartment="graph")`.
- **Graph systems:** they read and write only DataStructures; the props fallbacks and the `_prop_location` heuristic are removed.
- **`CompositeModel`:**
  - It couples only DataStructure-backed components.
  - `couple_current_with_components_list` (soil-side props coupling) is removed; the scene uses `coupler.Transport` and `Coupler`. `soil_handshake_inputs`, `soil_handshake_prefix` and `plant_side_soil_inputs` are removed too.
  - `apply_input_tables` writes DataStructures only.
  - `translator_matrix_builder` skips fields without metadata; it used to crash on `FunctionalComponent`.
- **`play_Orchestra`:** `handshake_shape` is required, and `scene_wrapper.HANDSHAKE_SHAPE` is removed.
- `MPGDataStructure.from_legacy` copies the values by vertex. It used to copy them in the legacy sorted order into the MPG's post-order, which misaligned them, or a length-1 array from a legacy structure at another scale; that case now raises.

