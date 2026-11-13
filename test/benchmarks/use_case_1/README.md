# Use case 1's benchmarks

Development tools measuring [use case 1](../../../examples/use_case_1/README.md) (its `run.py` scene, built in memory
from `models.py` and `component.py`), and what they measured on a 14-thread laptop (Core Ultra 7 255U, 2 performance,
8 efficient and 2 low-power cores, 30 GB):

- `benchmark.py` times the run hour by hour (`python test/benchmarks/use_case_1/benchmark.py [--plants 2] [--days 3]
  [--label text] ...`, `--help` for its options), each entry in `outputs/benchmarks` here;
- `memory_probe.py`, the memory of a scene (`benchmark.py --memory`, `--tracemalloc`) and a profile's split
  (`profile_split`).

## The run's configuration and its speed

The run's configuration, `Compute(threads="auto")` (metafspm's default, shown in `run.py`): the
processors for numba's kernels, the plant system's blocks and the reduction's archetypes (each archetype's Kron reduction in
a thread), each part taking those that pay (the plant system's blocks in the count found fastest on its own solves:
4 on this laptop; BLAS stays at one thread at any `threads`: the results are the same bit for bit at any number;
8 plants, day 13: 4.27 s per hour at 1 thread, 3.92 with "auto", measured 2026-11-08 after the generic kernels'
GK-1 to GK-5 and the process pool's removal, against 4.93 and 4.75 the same day before them), the memory freed
given back after each hour (`malloc_trim`) and glibc's allocator held to two arenas (`Compute(memory="trim")`, the
default: resumed from day 12, 8 plants keep 2.73-2.76 GB against 3.17-3.31 GB with glibc's default arenas, the
peak 3.68-3.76 GB against 4.13-4.23, at the same time per hour, as `MALLOC_ARENA_MAX=2` in the environment gives;
measured 2026-11-10). Its sparse solver may be
chosen per graph system: `Compute(threads="auto", linear_solver={"default": "superlu", "symmetric": "cholmod"})`
sends MECHA's standard problem (declared symmetric) to CHOLMOD (scikit-sparse 0.5, optional), the plant problem, its
Kron reductions and the soil keeping SuperLU (`benchmark.py --linear-solver '{"symmetric": "cholmod"}'`). Measured
2026-11-09 at 2 plants over 3 days: its 38 solves 0.50 s against SuperLU's 0.88 s, the run 60.2 s against 56.0-59.4
(within the runs' spread), the results within 1.6e-14 relative (the standard uptake), the frames identical: not
worth it at this size. The reduction's archetypes
are worked out in the run's process, each in a thread (metafspm's process pool was removed on 2026-11-08, QG-3 of
its devplan_generic_kernels): measured that day on this 14-thread laptop, 8 plants to day 8 take 292.6 s at
"auto" (the scope's peak 3.15 GB) against 283.1 s and 3.83 GB with the former pool ("auto": 7 processes), every
step's results identical bit for bit; resumed from day 8, day 12 and 13 take 3.59 and 4.27 s per hour at 1 thread,
3.27 and 3.92 at "auto" (the scope's peak 3.9 GB, the checkpoint's load included). `benchmark.py --threads auto|n
--linear-solver superlu|superlu_reuse|cholmod|pardiso|mumps` compares them (`--threads`: the budget of numba's
kernels, the plant system's blocks and the archetypes, "auto" by default; BLAS stays at one thread; the benchmark's entry
records the threads chosen and each kernel's and solver's time, per step).

**Cores of a hybrid CPU.** A step is one chain of components on the main thread (the other threads only take
Caribu's projections, the archetypes, the plant system's blocks and numba's kernels), so its time follows the core the
main thread runs on. This laptop's Core Ultra 7 255U has 2 performance cores (processors 0-3), 8 efficient cores
(4-11) and 2 low-power efficient cores (12-13): resumed from day 12, 8 plants take 4.02 s per hour at 1 thread on
the performance cores, **5.19 s (+29 %) on the efficient cores and 8.92 s (+122 %) on the low-power ones** (measured
2026-11-10, `taskset`; the results identical whatever the cores). On an idle machine the scheduler already keeps the
busy main thread on a performance core (4.02 s unpinned too): pinning gains nothing there. Beside other work (a test
suite, a browser, a second run), the run can be moved to the slower cores; `taskset -c 0-3 python run.py` keeps it
on the performance cores (`lscpu --extended` or `/sys/devices/cpu_core/cpus` lists them; Linux only, macOS has no
affinity setting). `threads="auto"` then counts the processors given (4 here).

**Populations in processes** (`Compute(processes=n)`, metafspm's `scene.runner`; measured 2026-11-11, each plant
its own population, `planting(..., population=["wheat0", ...])`, 8 GB scope, OPENBLAS/OMP/MKL at 1). Every table and
frame byte-identical at 1, 2 and 4 processes. Mean step over day 2, 4 plants: **1.72 s at 1 process, 1.48 s at 2
(x1.16), 1.16 s at 4 (x1.48)** (totals 126.5 / 109.9 / 90.5 s); 2 plants, day 3: 0.91 s at 1, 0.94 s at 2. The
publication (each worker's arrays raw through /dev/shm, its MTG pickled only when it grew: here every hour) costs
about 0.05-0.06 s per plant and step, half the first runner's checkpoint-based one (ccb2d2f: 1.58 s at 2, 1.25 s at
4). Workers' resident memory 1.9-2.3 GB each (shared pages counted; the scope held them all), balance 1.01-1.04.

The largest scene within the 8 GB bound (B3, the scene's process unpinned): **6 plants**. From the sowing, 12 hours:
95.4 s at 1 process, 85.3 s at 2, 75.8 s at 4 (the archetypes' first evaluations, the scope's peak 6.5 / 7.6 /
8.0 GB); 8 plants at 4 processes ran out of memory in the first steps. Resumed at day 2 for a day (hours 48-72): **60 s
at 1 process (59.8, 60.7), 59-61 s at 2 (59.4-61.1)**, 4 processes out of memory at hour ~60; tables and frames
byte-identical. At 2 processes the publication takes 10-12 s of the day, and a plant's step is ~1.5x slower in a
worker (efficient cores) than in the scene's process (performance cores): 37 s per worker of 3 plants against 24 s for
3 plants in one process. A forked worker costs ~470 MB per plant beyond one process's run (the shared pages the scene's
process rewrites stay alive): the scope's peak 4.3-4.7 GB at 1 process, 7.3-7.7 GB at 2. On this laptop processes
pay at the sowing's heavy first steps, not later; machines of equal cores and more memory are where they pay.

With the MTG's delta publication (2026-11-12: a worker sends its MTG's change since the last step, the scene's copy
patched in place; the first publication after the fork a delta, the shared parts both processes hold since by their
id, so the copies keep the pages they share with the workers): the same day resumed at hour 48, 6 plants, **59.8 s at
1 process, 56.6 s at 2** (was 59.8), the publication **4.3 s** of the day (was 12.6), the scope's peak **5.1 GB**
at 2 (was 7.5; 4.2 GB at 1), and 4 processes now fit: 53.0 s, 6.9 GB (out of memory before); 4 plants from the sowing, 2 days: mean step over day 2 1.69 / 1.50 / 1.15 s at 1 / 2 / 4
processes, the publication 7.5 s at 2 and 9.2 s at 4 (was 11.4 and 13.5). Tables and frames byte-identical at every
count; `Compute(checks={"publication"})` passed on the 6-plant day.

With the shared archetype store (2026-11-12, B4: a reduction's archetype kind evaluated once for all the populations,
in memory in one process, through `/dev/shm` between processes, a kind claimed by the first worker needing it):
tables and frames byte-identical at 1, 2 and 4 processes and to before.

| scene | processes | total before -> after (s) | archetypes' evaluation (s, summed over processes) | scope's peak (GB) |
|---|---|---|---|---|
| 6 plants, first 12 h | 1 / 2 / 4 | 92.0 / 80.5 / 71.0 -> **41.1 / 39.6 / 35.8** | 60 / 95 / 145 -> 9 / 15 / 17 | 6.4 / 7.1 / 7.4 -> 3.7 / 4.5 / 5.2 |
| 4 plants, 2 days | 1 / 2 / 4 | 122.6 / 108.5 / 89.3 -> **68.6 / 71.8 / 60.8** (mean step 1.43 / 1.49 / 1.26) | 71 / 111 / 171 -> 18 / 24 / 31 | 4.7 / 5.8 / 6.3 -> 3.0 / 4.2 / 5.0 |
| 6 plants, resumed at hour 48 for 24 h | 1 / 2 | 59.7 / 54.8 -> **41.2 / 42.8** | 22 / 33 -> 4 / 6 | 3.7 / 4.5 -> 3.2 / 4.2 |

Growing the 6 plants to hour 48 at 1 process: 90.8 s, 3.7 GB (185.1 s, 6.5 GB before). One process gains most (its
six populations evaluated each kind six times); processes still pay at the sowing (x1.15 at 4), not later on this
laptop.

**Scaling** (measured 2026-11-07 on the same laptop, metafspm f3a2148, GRANAP c626201, after the incremental
topology updates; `benchmark.py --variability 1`: each plant its own root draws, the plants sharing their archetypes;
under an 8 GB bound, the memory the scope's peak, the process pool of the time included (its processes in the
"workers" column; the pool is removed since); each row the day's mean step, and
before them, at metafspm d2e708e, every step's results identical):

| plants, day | pool processes (then) | s per step (before) | archetypes evaluated that day (s per step) | nodes | memory peak |
|---|---|---|---|---|---|
| 2, day 8 (identical plants) | auto (7) | 1.44 (1.49) | 139 | 13,924 | 3.6 GB |
| 8, day 8 (the day-13 run) | 1 | 2.65 (2.85) | 115 (0.60) | 56,598 | |
| 16, day 8 | 2 | 4.06 (4.59) | 122 (0.46) | 113,438 | 5.2 GB |
| 8, day 12 (the day-13 run) | 1 | 4.86 (5.68) | 91 (0.50) | 193,672 | |
| 8, day 13 | 1 | 6.05 (7.21) | 83 (0.44) | 252,478 | 6.2 GB |
| 8, day 14 (metafspm db12072; stopped at h334 by the run's 6.5 GB guard) | 1 | 7.8 at h313–329, 8.5 at h333 (9.8, 10.2: final pass, GRANAP 9464738) | 60 | 326,034 | 7.0 GB |

The archetypes saturate (a few hundred a day in the first week, 50 to 90 a day from day 9: the organs' new classes, a
few tenths of a second per step); the step grows with the plants' nodes (the roots' sections and their kept nodes:
about 25 µs per node per step from day 10 on, MECHA's plant solve first, then GRANAP's anatomies and the topology
updates), and so does the memory (about 10 kB per node beyond 1.5 GB). Extrapolated, 40 plants at day 14 would hold
about 1.7 million nodes, take about 40 to 45 s per step and need 13 to 18 GB: beyond this laptop (the run is a
workstation's, or fewer plants, or a coarser root anatomy). The process pool of the time added 1 to 2.5 GB on a
large scene (the pages of the main process its forked processes touched were copied); without it, the scope's peak
is the run's process alone. Recorded (`benchmark.py --frames`: a frame a day, p0's sections
drawn as instances), 8 plants to day 8 take 357 s (334 s unrecorded), the frames 1.1 MB at the sowing to 25 MB at day
8 (about 0.45 kB per node), the parts 81 MB (77 files), the folder 176 MB.
On a machine with an NVIDIA GPU and CuPy, `benchmark.py --device cuda` runs the archetypes' Kron reductions and the
linear solves on the device (metafspm's doc/gpu.md, "Running on a GPU": what is offloaded, how to compare it with
the CPU run, the transfers in the entry's `device`). The GPU path is experimental: it should work on suitable
hardware but has not been tested on a GPU yet (no GPU on the development laptop), and its results would agree with
the CPU's within round-off, not bit for bit.

`benchmark.py` times it (its `build_scene`: `run.py`'s scene, in memory, with the benchmark's options; `python
benchmark.py [--plants 2] [--days 3]`, under a memory bound): each hour's time by
model, the archetypes of the reduction evaluated, the nodes, the memory (after each hour, after `malloc_trim` and the
peak), written to `outputs/benchmarks/<commit>.json`, one entry per mode; `--checkpoint-at 168` checkpoints the
scene after those hours and `--resume <checkpoint>` runs on from one (its archetypes loaded, none evaluated again).
`--scenario KEY=VALUE ...` updates the plants' scenario, e.g. MECHA's lumped shoot nodes, `--scenario
leaf_ports=bundles vapour_path=air_spaces` (`plant_reduction`'s options: the shoot's conduits per vascular bundle,
a leaf's evaporating walls per air space); each step records the reduced groups' dense blocks (their nonzeros), the
end the archetypes held and their pickled size. `--memory` adds what the scene holds at the end by owner
(`memory_probe.report`: the DataStructure's columns, the MPG's properties and topology, the Instances, the GRANAP
templates, the archetypes' parts, the group tables, the libraries, the solver's caches, and the
process before and after `malloc_trim`), `--tracemalloc 8` the allocations alive at the end by line and by stack
(about 8 times slower), `--thread-scaling` the last step's large linear solves, its parallel kernels (a
reduction's terms among them) and the memory bandwidth timed at 1 to 12 threads, `--threads` the run's
(`Compute(threads=)`); `memory_probe.profile_split` splits a `--profile-hours` profile into Python overhead and numeric
work by call site (devplan_scene_parallel.md, P0).

The use case keeps MECHA's defaults (each conduit and evaporating wall its own node). Measured with the options (2
plants, day 8, against the defaults; 2026-11-06): `leaf_ports="bundles"` keeps 12,060 nodes (13,924), 1.09M dense
nonzeros (1.96M), the plant solve 19 s over the run (30 s; the step 1.66 s on the last day, 1.77 s), the archetypes 494 MB
(613 MB), the transpiration, uptake, An and gs within 0.04 %, the leaves' water potential 0.45 % above (0.8 % of
their drop below the roots'); with `vapour_path="air_spaces"` too, 10,250 nodes, 0.30M nonzeros, the solve 12 s (the
step 1.55 s), the archetypes 337 MB, the fluxes within 0.2 %, the leaves' water potential 3 % above. Both are options
only, not recommendations, at any scene size. `leaf_ports="bundles"` gives up each vessel's own axial path and its
own junction to its successor section (a bundle's vessels are held at one potential and joined bundle to bundle),
and the leaves' water potential, which Tuzet's stomata read, comes out biased (beyond the 0.5 % bound as the plants
grow). At 16 plants (their own root draws), day 8:
`leaf_ports="bundles"` keeps 98,526 nodes (113,438), 8.8M dense nonzeros (15.7M), MECHA 1.44 s per step (2.18;
the step 3.28 s, 4.66 measured before GRANAP's faster junctions, 5f5b34b), the scope's peak 4.7 GB (5.3), the transpiration and uptake unchanged, gs within 0.04 %, the
leaves' water potential 0.36 % above on average, 0.54 % at most (0.8 % of their drop below the roots').

**Running a model less often** (a schedule step's cadence, metafspm's `Step(cadence=)` or the model's scenario
`{"cadence": {step: duration}}`; measured 2026-11-07, 2 plants, 3 days from the sowing, `benchmark.py --cadence
WheatPlant.anatomy="3 h" ...`, against the hourly run; the steps' time without the build). The use case keeps every
model hourly:

| | calls | steps (hourly 56 s) | the model's own time | what changes |
|---|---|---|---|---|
| GRANAP's anatomies every 3 h (`WheatPlant.anatomy="3 h"`) | 24 | 52 s | 10.0 → 8.8 s; MECHA 34 → 31 s | fluxes within 1e-7; the roots' mean ψ up to 74 hPa off (its range −131 to −50 hPa): the sections wait for their anatomy |
| GRANAP's anatomies every day | 3 | 36 s | 10.0 → 7.6 s; MECHA 34 → 17 s (fewer sections), the archetypes 27 → 11 s | fluxes within 1e-6; the roots' ψ up to 98 hPa off (−4 instead of −102 hPa at the first day's end) |
| Caribu's whole light (lighting and PARa) every 3 h, from the interval's start | 24 | 50 s | 2.8 → 1.0 s | PARa held at its 3-hour mean: up to 36 % off at an hour; over the run PARa +1.5 %, An +6 %, transpiration +0.5 % |
| Caribu's light every day, from its start | 3 | 52 s | 2.8 → 0.2 s | PARa held at the day's mean, night included: transpiration ×1.9, An −14 % |
| the former light model's `interval` of 3 h or 1 d | 72 | 52 s | unchanged | identical: the light is relit whenever elements appear (its geometry_input), every hour here |

A call of a cadenced component costs what it builds (the sections new since the last call), so GRANAP gains little
until the day; the whole light's cadence holds PARa (its output) between calls. Caribu's light model now schedules
its lighting (`CaribuLightModel.light`, its rate stage, `CaribuLightModel.light="3 h"`) apart from its PARa (its
state stage, every step), the lighting relit early when elements appear (its geometry_input), its call then
covering to the interval's end. Measured the same way (`CaribuLightModel.light="3 h"`): the light's time is
unchanged (2.8 s: elements appear every hour while the plants grow, so it is relit every hour), and its relative
light, the mean over the rest of the interval rather than over the hour, moves PARa up to 38 % at an hour (An up to
41 %, transpiration 26 %). A sparser lighting would need the relight rule relaxed.

**The nested light** (`LIGHT_SCENARIO`'s `mode="nested"` since 2026-11-10; the measures above were made with the
direct light). Measured that day (`benchmark.py`, from the sowing, "auto"): the step is the direct light's within
the runs' spread, Caribu 0.20 s per step at 8 plants day 8, 0.27 s at days 12 and 13 (the direct light's 0.17 and
0.21), 0.23, 0.28 and 0.33 s at 2 plants days 16, 19 and 22 (0.18, 0.24, 0.26); at 1 thread, 8 plants day 13,
0.59 s of a 4.32 s step. The canopy is small (376 triangles at 8 plants day 13, 268 at 2 plants day 22) and its
pattern narrower than the 0.5 m sphere: the near field holds every triangle's images, 50 to 64 pairs per triangle,
and costs 0.04 to 0.06 s; the sky's and the sun's projections (direct light's z-buffer) remain about three
quarters of Caribu's time, the radiosity solve (5 MGCR steps) under a millisecond. The plants absorb 1.19 to 1.27 times the
direct light's PAR (the same triangles): the soil's reflection reaching the leaves from below and, less, their
scattering.

## The recorder's cost

Measured with `benchmark.py --outputs` (2026-11-06; the display frames draw p0's sections, the use case's frame
variables; "+parts": the parts the frame added to `frames/parts`, the second display frame one hour later):

| | data frame | display frame, instances (+parts) | display frame, full | 2nd frame, instances (+parts) | 2nd, full | 2 frames' folder: instances / packed / full |
|---|---|---|---|---|---|---|
| 1 plant, day 3 | 0.7 MB | 0.8 MB, 1.2 s (+33 MB) | 33.0 MB, 3.1 s | 0.8 MB, 0.3 s (+2.6 MB) | 33.6 MB, 3.1 s | 37 / 14 / 67 MB |
| 2 plants, day 3 | 1.4 MB | 1.4 MB, 1.3 s (+33 MB) | 33.7 MB, 3.1 s | 1.4 MB, 0.4 s (+2.6 MB) | 34.3 MB, 3.1 s | 39 / 14 / 68 MB |
| 1 plant, day 8 | 2.9 MB | 3.0 MB, 2.3 s (+52 MB) | 143.5 MB, 11.1 s | 3.1 MB, 0.9 s (+5.0 MB) | 145.3 MB, 11.5 s | 63 / 24 / 289 MB |
| 2 plants, day 8 | 5.8 MB | 6.0 MB, 2.9 s (+52 MB) | 146.5 MB, 11.5 s | 6.0 MB, 1.4 s (+5.0 MB) | 148.3 MB, 11.3 s | 69 / 25 / 295 MB |

A display frame costs about 0.1 MB more than the data frame; the parts are written once (the templates, then a
archetype's operator, 0.6 to 5.7 MB as its sparse system, when a new archetype is drawn: they saturate with the archetypes), so a
run's frames grow by the archetypes drawn, not by the frames. A 2-day run (1 plant, 5 frames, 3 drawing the sections):
frames 0.3 to 0.6 MB, parts 34 MB, the folder 37 MB, packed 14 MB. A checkpoint is 8 MB at day 3 (1 plant), 36 MB at
day 8 (2 plants), the anatomy templates and the archetypes (0.4 to 0.9 GB) written once in `checkpoints/shared`.

The recorder's cost at larger sizes (measured 2026-11-10, after the bulk frame build, the copy-free GLB writer,
the streamed checkpoints, the summaries' groupings and the background frame writer; resumed scenes, one step, the
process's peak over its resident memory): 8 plants at day 12 (220 k nodes), a frame 2.5 s (13.0 s before, +0.40 GB
against +0.47), with p0's sections 2.8 s (16.6 s), its write 0.2 s (+0.05 GB against +0.19), a checkpoint 0.3 s
after the first (0.65 s, +0.05 GB against +0.33), the summaries and tables 0.05 s a step (0.11 s); 2 plants at day 21
(427 k nodes), a frame 10.3 s (+0.89 GB), its write 0.6 s (+0.17 GB), a checkpoint 0.9 s (+0.10 GB). The frames
are written by a writer thread while the next hours run (`SceneRecorder(frame_writing="background")`, the default):
a run's day of 2 frames and a checkpoint at 2 plants, day 21, costs about 20 s on top of its 24 hours' steps
(about 170 s; from the sowing to hour 520 the recording took 144 s of the run's 1441 s).
Every hour's frame (`frame_period="step"`) is not cheap: 8 plants at day 12, 9.5-9.7 s per hour against 4.6 s
without frames (background or synchronous alike), 1.1 GB of frames a simulated day (195 MB packed, `pack()` 73 s);
2 plants at day 21, 18.7 s per hour in the background against 23.8 s synchronous and 8.3 s without frames, 2.3 GB a
day (386 MB packed, `pack()` 2.3-2.9 min). The build stays in the hour's thread: built in a second thread beside the
step, both slow down (8 plants, day 12: the step 5.7 → 13.0 s, the build 3.5 → 9.2 s; the GIL), more than building
in turn.

The disk a run takes (`python run.py 3 2`, 2026-11-09): 343.6 MB at its end, of which `frames.fspm.zip` 16.3 MB
(7 frames and 27 parts; the folder it replaces 45.0 MB, its parts 41.2 MB), `checkpoints/` 327.1 MB (the last
one, `step_000072`, 8.5 MB; `shared` 318.6 MB) and the tables 0.2 MB. Before, the run kept the folder beside its
archive and the last two checkpoints: about 45 MB and a checkpoint (8 MB at day 3, 36 MB at day 8) more. The
shared parts, not the checkpoints themselves, take most of the disk, written once whatever `keep_checkpoints`.
