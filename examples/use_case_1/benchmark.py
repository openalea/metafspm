"""
Use case 1's benchmark (devplan_release_optimisation.md §5): the use case's scene (build_scene: run.py's, in memory:
no frames, no checkpoints) run hour by hour from the sowing, each step's time by model and by component
(Scene.timing: "Model" and "Model.Component"), the kinds of the reduction evaluated (their number and time), the plants' nodes, the step's
results (results(): transpiration, uptake, the leaves' gas exchange, the roots' water potential, the plant's and the
soil's balance residuals, to compare runs), and the memory (RSS after each step; after malloc_trim and the peak at the
end).

    python benchmark.py [--plants 2] [--days 3] [--label text] [--outputs] [--checkpoint-at 168 ...]
                        [--resume outputs/benchmarks/checkpoints/<label>/step_000168]
                        [--scenario leaf_ports=bundles vapour_path=air_spaces ...]
                        [--threads auto|n] [--workers auto|n] [--linear-solver superlu|superlu_reuse|cholmod|cupy]
                        [--device cpu|cuda] [--variability SEED] [--frames] [--profile-hours FIRST LAST]
                        [--cadence WheatPlant.anatomy="1 d" CaribuLightModel.light="3 h" ...]
                        [--memory] [--tracemalloc FRAMES] [--thread-scaling]

By default 2 plants to day 3 (the use case's fast regime); --days 8 for the day-8 numbers, --plants 8 --days 4 for the
scaling; --outputs measures the outputs at the end (outputs()): a frame without the sections drawn, two display
frames one step apart with the sections of DISPLAY_PLANTS drawn as instances (display="instances", the
recorder's default: their frames, the parts they share, the folder packed with deflate) and whole ("full"), the use
case's frame variables, and two checkpoints sharing their unchanging parts: their size and time. --checkpoint-at writes the scene's checkpoint after those hours (outputs/benchmarks/checkpoints/<label>/, the
shared parts once); --resume loads one into the freshly built scene and runs on from its hour to --days (the load's
time and memory, then each step's: a resumed run evaluates no kind again); --scenario updates the plants' scenario
(models.WHEAT_SCENARIO; a value read as JSON when it is, e.g. length_classes=0.2, else a string); --threads, --workers
and --linear-solver the Compute's (the threads' budget of the kernels and solves, "auto" by default, the pool of the
kinds, the sparse solver; the threads chosen recorded, "threads", and each kernel's, solver's and step's
(functor.<Class>.<step>) time, per step and in all, "kernels"),
--device its device (cuda: the kinds' Kron
reductions and the linear solves on the GPU, CuPy needed, docs/gpu.md; the device's transfers recorded at the end,
"device"), the pool recorded per step and the peak
memory of the process's control group (the pool's processes included: run it in its own systemd-run --scope), now
and at its peak. --variability gives each plant its own seeded root draws (plant_variability: ArchiSimple's
random_choice; identical plants share every kind, which flatters a scaling); --frames records the run (a frame every
24 h, the sections of DISPLAY_PLANTS drawn as instances, no checkpoint) under outputs/benchmarks/frames/<label>,
its size at the end; --profile-hours cProfiles the main process over those steps (outputs/benchmarks/profiles/);
--cadence gives a model's schedule steps ("<model>.<step>": WheatPlant.anatomy, GRANAP's; CaribuLightModel.light, the
lighting, its PARa every step) a cadence (coupling.cadence.Cadence: the duration, "@start" or "@end", the step's declared
one by default: CaribuLightModel.light's at its start, the others' at the interval's end; set_cadences, the scenario's
"cadence"), to measure what running them less often saves and changes.
--memory measures at the end what the scene holds by owner (memory_probe.report: the DataStructure's columns, the
MPG's properties and topology, the Instances, the registries, the kinds, the components, the solver's caches, the
pool's processes; before and after malloc_trim), "memory"; --tracemalloc traces the allocations from before the
scene's build (FRAMES frames each: slower, more memory) and records those alive at the end by file and line,
"allocations"; --thread-scaling captures the last step's large linear solves and parallel kernels' inputs and times
them at 1, 2, 4, 8 and 12 threads (memory_probe.thread_scaling), "thread_scaling". The process's memory after its
imports and after the scene's build is recorded either way ("baseline").
Each step also records its kinds' batches (kinds, interior cells, seconds, processes: "auto"'s estimate checked
against them). Each step records the
plants' nodes and the reduced groups' dense blocks (their nonzeros: kept nodes squared, summed), the end the kinds
held and their pickled size (as checkpointed). The results go to
outputs/benchmarks/<commit>.json (untracked), one entry per mode ("2p_3d") updated in place, and a summary is
printed. Run it under a memory bound, one at a time (systemd-run --user --scope -p MemoryMax=8G ...).
"""
import argparse
import ctypes
import json
import os
import platform
import resource
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from openalea.metafspm.solve import reduction as reduction_module    # noqa: E402

from openalea.caribu.component import CaribuLightModel             # noqa: E402
from openalea.metafspm.scene.population import planting, stand       # noqa: E402
from openalea.metafspm.scene.recorder import SceneRecorder           # noqa: E402
from openalea.metafspm.scene.scene import Scene                      # noqa: E402

from component import RichardsSoilModel                              # noqa: E402
from models import (FRAME_VARIABLES, LIGHT_SCENARIO, SCENE_TRANSLATOR, SOIL_SCENARIO, TABLES,  # noqa: E402
                    WHEAT_SCENARIO, WheatPlant, weather)

START = "2026-05-10 00:00"                                           # run.py's
DISPLAY_PLANTS = ("p0",)                                             # the plants whose sections frames draw


def build_scene(plants=2, folder=None, frame_period="12 h", display_period="1 d", checkpoint_period="1 d", scenario=None,
          compute=None, plant_scenarios=None):
    """run.py's scene with the benchmark's options: *plants* plants (WHEAT_SCENARIO updated by *scenario*, each plant's
    *plant_scenarios* added), recorded in *folder* (None: in memory: no frames, no checkpoints) with run.py's tables,
    frames and views; *compute*: the run configuration (None: Compute())."""
    sown = stand(plants, row_spacing=0.15, plant_spacing=0.05, depth=0.03, rotation=120.)
    table = planting(sown, WheatPlant, {**WHEAT_SCENARIO, **(scenario or {})}, plant_scenarios=plant_scenarios)
    recorder = SceneRecorder(folder, tables=TABLES, frame_period=frame_period if folder else None,
                             frame_views="default" if folder else None,
                             frame_options={"variables": FRAME_VARIABLES},
                             display_period=display_period if folder else None,
                             display_plants=DISPLAY_PLANTS if folder and display_period else None,
                             checkpoint_period=checkpoint_period if folder else None)
    return Scene(table, environment=[RichardsSoilModel, CaribuLightModel],
                 environment_scenarios=[SOIL_SCENARIO, LIGHT_SCENARIO], translator=SCENE_TRANSLATOR,
                 forcings=weather(START), time_step="1 h", start=START, compute=compute, recorder=recorder)

BENCHMARKS = os.path.join(HERE, "outputs", "benchmarks")
HOUR = 3600.


def commit() -> str:
    """The metafspm commit (short hash), "-dirty" when the tracked files differ from it."""
    root = os.path.dirname(os.path.dirname(HERE))
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True,
                           text=True)
    return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def resident_memory() -> float:
    """The process's resident memory now (GB), from /proc (NaN elsewhere)."""
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return float("nan")


def scope_peak_memory() -> float:
    """The peak memory of the process's control group (GB: its memory.peak, cgroup v2), the pool's processes
    included (a systemd-run --scope's), NaN where there is none."""
    try:
        with open("/proc/self/cgroup") as cgroup:
            relative = cgroup.read().strip().split("\n")[0].split("::", 1)[1]
        with open(os.path.join("/sys/fs/cgroup", relative.lstrip("/"), "memory.peak")) as peak:
            return int(peak.read()) / 2 ** 30
    except (OSError, IndexError, ValueError):
        return float("nan")


def scope_memory() -> float:
    """The memory of the process's control group now (GB: its memory.current, cgroup v2), the pool's processes
    included, NaN where there is none."""
    try:
        with open("/proc/self/cgroup") as cgroup:
            relative = cgroup.read().strip().split("\n")[0].split("::", 1)[1]
        with open(os.path.join("/sys/fs/cgroup", relative.lstrip("/"), "memory.current")) as current:
            return int(current.read()) / 2 ** 30
    except (OSError, IndexError, ValueError):
        return float("nan")


def peak_memory() -> float:
    """The process's peak resident memory (GB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2 ** 30 if sys.platform == "darwin" else peak / 2 ** 20


def trim_memory() -> None:
    """glibc's malloc_trim(0): the freed memory given back (nothing elsewhere)."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class KindCounter:
    """Counts the kinds the reductions evaluate (Reduction._evaluate_requests' requests: in the process or in the
    Compute's pool) and the time they take (the main process's wait)."""

    def __init__(self):
        self.kinds, self.seconds, self.templates = 0, 0., 0
        self.batches = []                 # each evaluation's (kinds, interior cells, seconds, processes)
        self.seen = set()                 # the (reduced template, signature) of the kinds evaluated so far
        self._evaluate = evaluate = reduction_module.Reduction._evaluate_requests
        counter = self

        def counted(reduction, missing):
            from openalea.metafspm.compute import Compute, in_worker
            started = time.perf_counter()
            try:
                return evaluate(reduction, missing)
            finally:
                seconds = time.perf_counter() - started
                counter.seconds += seconds
                kinds = sum(len(requests) for _, requests in missing.values())
                cells = sum(reduction._interior_cells(request[1]) for _, requests in missing.values()
                            for request in requests.values())
                processes = 1 if (reduction.library or in_worker()) else Compute.current().pool_size(kinds, cells)
                counter.batches.append((kinds, cells, seconds, processes))
                for _, requests in missing.values():
                    counter.kinds += len(requests)
                    for key, _, signature in requests:    # the kinds of a template not seen before (a new anatomy)
                        if (key, signature) not in counter.seen:
                            counter.seen.add((key, signature))
                            counter.templates += 1

        reduction_module.Reduction._evaluate_requests = counted

    def take(self) -> tuple:
        out = (self.kinds, self.seconds, self.templates, self.batches)
        self.kinds, self.seconds, self.templates, self.batches = 0, 0., 0, []
        return out


def results(scene) -> dict:
    """The step's results, to compare runs (a change's effect on them): the organs' transpiration and the roots'
    uptake (models.WATER_TABLES), the leaves' green area, PARa, An and gs (models.LEAVES), the roots' water potential
    (models.ROOTS), the plant's and the soil's water balance residuals (their declared conservation)."""
    rows = {name: scene.recorder.table(name).iloc[-1] for name in ("leaf_water", "plant_water", "leaves", "roots")}
    picked = {"transpiration": rows["leaf_water"]["leaf_transpiration"],
              "root_uptake": rows["plant_water"]["root_uptake"],
              "green_area": rows["leaves"]["green_area"], "PARa": rows["leaves"]["PARa"], "An": rows["leaves"]["An"],
              "gs": rows["leaves"]["gs"], "leaf_psi": rows["leaves"]["leaf_water_potential"],
              "leaf_psi_min": rows["leaves"]["leaf_water_potential_min"], "roots_psi": rows["roots"]["psi"]}
    balances = scene.recorder.table("conservation")
    balances = balances[balances["time"] == scene.time]
    for component, name in (("MechaHydraulics", "plant_residual"), ("RichardsSoil", "soil_residual")):
        row = balances[balances["component"] == component]
        picked[name] = row["residual"].iloc[0] if len(row) else np.nan
    return {name: float(value) for name, value in picked.items()}


def dense_blocks(scene) -> int:
    """The nonzeros of the reduced groups' dense blocks (each group's kept nodes squared), over the populations."""
    total = 0
    for population in scene.populations:
        ds = population.data_structure
        reduction = getattr(ds.structure, "reduction", None)
        table = reduction.group_table(ds) if reduction is not None else None
        if table is not None:
            total += sum(int(vids.shape[0]) * int(vids.shape[1]) ** 2 for vids in table.kept_vids)
    return total


def kinds_size(scene) -> tuple:
    """(the kinds the reductions hold, their pickled size in MB: as checkpointed), one kind pickled at a time."""
    import pickle
    count, size = 0, 0
    for population in scene.populations:
        reduction = getattr(population.data_structure.structure, "reduction", None)
        for by_system in (reduction._kinds.values() if reduction is not None else ()):
            count += 1
            size += len(pickle.dumps(by_system, protocol=pickle.HIGHEST_PROTOCOL))
    return count, size / 1e6


def plant_variability(plants: int, seed: int) -> list:
    """Each plant's own scenario entries, seeded: ArchiSimple's random_choice (its root system's random draws: the
    daughter roots' diameters, CVDD), so that the plants differ in shape (identical plants share every kind)."""
    draws = np.random.default_rng(seed).integers(1, 2 ** 31 - 1, size=plants)
    return [{"random_choice": int(draw)} for draw in draws]


def benchmark(plants: int, days: float, measure_outputs: bool = False, checkpoint_at=(), resume: str = None,
              label: str = "run", scenario: dict = None, compute=None, variability: int = None,
              frames: bool = False, profile_hours=(), cadences: dict = None, memory: bool = False,
              trace_frames: int = 0, scaling: bool = False) -> dict:
    import memory_probe
    counter = KindCounter()
    baseline = {"imported": memory_probe.process_memory()}
    if trace_frames:
        import tracemalloc
        tracemalloc.start(trace_frames)
    started = time.perf_counter()
    plant_scenarios = plant_variability(plants, variability) if variability is not None else None
    folder = os.path.join(BENCHMARKS, "frames", label) if frames else None
    options = dict(frame_period="1 d", display_period="1 d", checkpoint_period=None) if frames else {}
    scene = build_scene(plants, folder=folder, scenario=scenario, compute=compute, plant_scenarios=plant_scenarios,
                        **options)
    from openalea.metafspm.coupling.cadence import Cadence
    for name, given in (cadences or {}).items():           # "<model>.<step>": a model's schedule step
        model, _, step = name.rpartition(".")
        if not model:
            raise KeyError(f"--cadence {name}: give the model and its step, e.g. WheatPlant.anatomy")
        interval, _, at = given.partition("@")
        declared = next((s.cadence for s in scene.model(model).steps() if s.name == step), None)
        at = at or (declared.at if isinstance(declared, Cadence) else "end")
        scene.model(model).set_cadences({step: Cadence(interval, at=at)})
    built = time.perf_counter() - started
    baseline["built"] = memory_probe.process_memory()
    counter.take()
    loaded, first_hour = None, 0
    if resume is not None:
        load_started = time.perf_counter()
        scene.load_checkpoint(resume)
        first_hour = int(round(scene.time / HOUR))
        loaded = {"path": resume, "hour": first_hour, "seconds": time.perf_counter() - load_started,
                  "rss_gb": resident_memory(), "kinds": sum(len(population.data_structure.structure.reduction._kinds)
                                                           for population in scene.populations
                                                           if population.data_structure.structure.reduction)}
        print(f"loaded {resume} (hour {first_hour}, {loaded['kinds']} kinds) in {loaded['seconds']:.1f} s, "
              f"RSS {loaded['rss_gb']:.2f} GB", flush=True)
    checkpoints = os.path.join(BENCHMARKS, "checkpoints", label)
    steps = []
    previous = dict(scene.timing)
    previous_kernels = scene.compute.timing_info()
    profiler, profiled = None, os.path.join(BENCHMARKS, "profiles", f"{label}.prof")
    last_hour, captured = int(round(days * 24)), None
    for hour in range(first_hour, last_hour):
        if scaling and hour + 1 == last_hour:
            captured = memory_probe.Captured().install()
        if profile_hours and hour + 1 == profile_hours[0]:
            import cProfile
            profiler = cProfile.Profile()
            profiler.enable()
        step_started = time.perf_counter()
        scene.run()
        seconds = time.perf_counter() - step_started
        if captured is not None:
            captured.remove()
        if profiler is not None and hour + 1 == profile_hours[-1]:
            profiler.disable()
            os.makedirs(os.path.dirname(profiled), exist_ok=True)
            profiler.dump_stats(profiled)
            profiler = None
            print(f"profile of h {profile_hours[0]}-{profile_hours[-1]} written to {profiled}", flush=True)
        kinds, kind_seconds, templates, batches = counter.take()
        timing = {name: value - previous.get(name, 0.) for name, value in scene.timing.items()}
        previous = dict(scene.timing)
        kernels = scene.compute.timing_info()
        kernel_seconds = {name: entry["seconds"] - previous_kernels.get(name, {}).get("seconds", 0.)
                          for name, entry in kernels.items()}
        previous_kernels = kernels
        nodes = sum(population.data_structure.n_nodes() for population in scene.populations)
        dense = dense_blocks(scene)
        steps.append({"hour": hour + 1, "seconds": seconds, "dense_nnz": dense,
                      "pool": scene.compute.pool_info(),
                      "models": {name: value for name, value in timing.items() if "." not in name},
                      "components": {name: value for name, value in timing.items() if "." in name}, "kinds": kinds,
                      "kernels": kernel_seconds,
                      "new_templates": templates,
                      "kind_seconds": kind_seconds, "kind_batches": batches, "nodes": nodes,
                      "rss_gb": resident_memory(), "scope_gb": scope_memory(),
                      "results": results(scene)})
        print(f"h {hour + 1:4d}  {seconds:7.2f} s  kinds {kinds:3d} ({kind_seconds:6.2f} s)  nodes {nodes:6d}  "
              f"dense {dense:8d}  RSS {steps[-1]['rss_gb']:.2f} GB  scope {steps[-1]['scope_gb']:.2f} GB "
              f"(peak {scope_peak_memory():.2f})", flush=True)
        if hour + 1 in checkpoint_at:
            checkpoint_started = time.perf_counter()
            path = os.path.join(checkpoints, f"step_{hour + 1:06d}")
            scene.checkpoint(path, shared=os.path.join(checkpoints, "shared"))
            steps[-1]["checkpoint"] = {"path": path, "seconds": time.perf_counter() - checkpoint_started,
                                       "MB": folder_size(path) / 1e6,
                                       "shared_MB": folder_size(os.path.join(checkpoints, "shared")) / 1e6}
            print(f"checkpoint {path}: {steps[-1]['checkpoint']['MB']:.1f} MB (shared "
                  f"{steps[-1]['checkpoint']['shared_MB']:.1f} MB) in {steps[-1]['checkpoint']['seconds']:.1f} s",
                  flush=True)
    pool = scene.compute.pool_info()
    allocations = None
    if trace_frames:
        import tracemalloc
        allocations = memory_probe.allocations(tracemalloc.take_snapshot())
        tracemalloc.stop()
    held = memory_probe.report(scene) if memory else None
    if held is not None:
        print(memory_probe.summary(held), flush=True)
    scaled = memory_probe.thread_scaling(captured) if captured is not None else None
    scene.finish()
    timing, total = dict(scene.timing), time.perf_counter() - started    # the steps' (outputs() runs one more)
    rss = resident_memory()
    trim_memory()
    after_trim, peak = resident_memory(), peak_memory()
    kinds_held, kinds_mb = kinds_size(scene)
    measured = outputs(scene) if measure_outputs else None
    compute = getattr(scene, "compute", None)
    result = {"commit": commit(), "date": time.strftime("%Y-%m-%d %H:%M:%S"), "plants": plants, "days": days,
              "machine": {"processors": os.cpu_count(), "platform": platform.platform(),
                          "python": platform.python_version()},
              "compute": compute.as_dict() if compute is not None else None,
              "threads": compute.thread_info() if compute is not None else None,
              "kernels": compute.timing_info() if compute is not None else None,
              "environment": {name: os.environ.get(name) for name in
                              ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "MALLOC_ARENA_MAX")},
              "build_seconds": built, "total_seconds": total, "resumed": loaded,
              "rss_end_gb": rss, "rss_after_trim_gb": after_trim, "peak_rss_gb": peak,
              "scope_peak_gb": scope_peak_memory(), "pool": pool, "scenario": scenario or {},
              "device": compute.device_info() if compute is not None else None,
              "kinds_held": kinds_held, "kinds_MB": kinds_mb, "variability": variability,
              "cadences": dict(cadences or {}), "calls": dict(scene.calls),
              "plant_scenarios": plant_scenarios,
              "frames": {"folder": folder, "MB": folder_size(folder) / 1e6} if folder else None,
              "timing": timing, "steps": steps, "outputs": measured, "baseline": baseline, "memory": held,
              "allocations": allocations, "thread_scaling": scaled}
    return result


def folder_size(path: str) -> int:
    """The bytes of the files under *path*."""
    return sum(os.path.getsize(os.path.join(folder, name)) for folder, _, names in os.walk(path) for name in names)


def outputs(scene) -> dict:
    """The outputs of *scene* now, written under outputs/benchmarks/outputs (removed after): a frame of the data; a
    display frame with the sections of DISPLAY_PLANTS drawn as instances (display_instances_<k>: the frame, the
    parts it adds to the frames' parts/ folder, their count) and whole (display_full_<k>), at this step (k 1) and
    the next (k 2: the parts it shares with the first not written again), the instances' two frames and parts packed
    (display_instances_packed: deflate, encoding3d.pack); two checkpoints one step apart into one shared folder (the
    second refers to the first's parts); their size (MB) and time (s)."""
    import shutil
    from openalea.metafspm.io.encoding3d import _encoding3d, default_views, scene_to_frame, with_view_columns
    folder = os.path.join(BENCHMARKS, "outputs")
    variables = with_view_columns(FRAME_VARIABLES, default_views(scene), list(scene.data_structures()))  # run.py's
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    encoding3d, found = _encoding3d(), {}

    def write(name, subfolder, file, **options):
        os.makedirs(os.path.join(folder, subfolder), exist_ok=True)
        parts = os.path.join(folder, subfolder, "parts")
        before = folder_size(parts) if os.path.isdir(parts) else 0
        started = time.perf_counter()
        frame = scene_to_frame(scene, variables=variables, **{"display_meshes": False, **options})
        path = os.path.join(folder, subfolder, file)
        keys = encoding3d.write(path, frame)
        found[name] = {"seconds": time.perf_counter() - started, "MB": os.path.getsize(path) / 1e6}
        if options.get("display") == "instances":
            found[name].update(parts=len(keys), new_parts_MB=(folder_size(parts) - before) / 1e6,
                               parts_MB=folder_size(parts) / 1e6)
        del frame
        return keys
    drawn = {"display_meshes": True, "display_plants": list(DISPLAY_PLANTS)}
    write("frame", "data", "frame.glb")
    keys = set()
    for k in (1, 2):
        keys |= set(write(f"display_instances_{k}", "instances", f"frame_{k:06d}.glb", display="instances", **drawn))
        write(f"display_full_{k}", "full", f"frame_{k:06d}.glb", display="full", **drawn)
        if k == 1:
            shared = os.path.join(folder, "checkpoints", "shared")
            for c in (1, 2):
                path = os.path.join(folder, "checkpoints", f"step_{c}")
                started = time.perf_counter()
                scene.checkpoint(path, shared=shared)
                found[f"checkpoint_{c}"] = {"seconds": time.perf_counter() - started, "MB": folder_size(path) / 1e6,
                                            "shared_MB": folder_size(shared) / 1e6}
                if c == 1:
                    scene.run()
    instances = os.path.join(folder, "instances")
    encoding3d.write_index(os.path.join(instances, "scene.fspm.json"),
                           [{"time": float(k), "step": k, "file": f"frame_{k:06d}.glb"} for k in (1, 2)],
                           time_unit="s", parts=sorted(keys))
    started = time.perf_counter()
    packed = encoding3d.pack(instances, os.path.join(folder, "instances.fspm.zip"))
    found["display_instances_packed"] = {"seconds": time.perf_counter() - started,
                                         "MB": os.path.getsize(packed) / 1e6,
                                         "folder_MB": folder_size(instances) / 1e6}
    found["display_full_folder"] = {"seconds": found["display_full_1"]["seconds"]
                                    + found["display_full_2"]["seconds"],
                                    "MB": folder_size(os.path.join(folder, "full")) / 1e6}
    shutil.rmtree(folder)
    return found


def summary(result: dict) -> str:
    steps = result["steps"]
    seconds = np.array([step["seconds"] for step in steps])
    kinds = np.array([step["kinds"] for step in steps])
    lines = [f"{result['plants']} plants, {result['days']} days ({len(steps)} steps) at {result['commit']}: "
             f"built in {result['build_seconds']:.1f} s, run {seconds.sum():.1f} s, first step {seconds[0]:.1f} s, "
             f"mean step {seconds[1:].mean() if seconds.size > 1 else seconds[0]:.2f} s, "
             f"last day's mean {seconds[-24:].mean():.2f} s"]
    lines.append(f"kinds evaluated {kinds.sum()} ({kinds[-24:].sum()} on the last day), "
                 f"{sum(step['kind_seconds'] for step in steps):.1f} s")
    lines.append(f"memory: RSS at the end {result['rss_end_gb']:.2f} GB, after trim {result['rss_after_trim_gb']:.2f} GB, "
                 f"peak {result['peak_rss_gb']:.2f} GB, the scope's peak (pool included) "
                 f"{result.get('scope_peak_gb', float('nan')):.2f} GB; pool {result.get('pool')}")
    lines.append(f"nodes at the end {steps[-1]['nodes']}, dense blocks {steps[-1]['dense_nnz']} nonzeros; kinds held "
                 f"{result['kinds_held']} ({result['kinds_MB']:.1f} MB pickled)"
                 + (f"; scenario {result['scenario']}" if result.get("scenario") else ""))
    if result.get("frames"):
        lines.append(f"frames: {result['frames']['folder']}, {result['frames']['MB']:.1f} MB")
    if result.get("device"):
        lines.append(f"device: {result['device']}")
    by_model = {name: value for name, value in result["timing"].items() if "." not in name}
    by_component = {name: value for name, value in result["timing"].items() if "." in name}
    lines.append("per model (s): " + ", ".join(f"{name} {value:.1f}" for name, value in
                                               sorted(by_model.items(), key=lambda item: -item[1])))
    lines.append("per component (s): " + ", ".join(f"{name} {value:.1f}" for name, value in
                                                   sorted(by_component.items(), key=lambda item: -item[1])))
    if result.get("outputs"):
        lines.append("outputs: " + ", ".join(
            f"{name} {entry['MB']:.1f} MB in {entry['seconds']:.2f} s"
            + (f" (shared {entry['shared_MB']:.1f} MB)" if "shared_MB" in entry else "")
            + (f" (+{entry['new_parts_MB']:.1f} MB of parts, {entry['parts']} used)" if "parts" in entry else "")
            + (f" (folder {entry['folder_MB']:.1f} MB)" if "folder_MB" in entry else "")
            for name, entry in result["outputs"].items()))
    return "\n".join(lines)


def scenario_entries(entries) -> dict:
    """--scenario's KEY=VALUE entries as a dict, each value read as JSON when it is (a number, true), else a
    string."""
    out = {}
    for entry in entries:
        key, _, value = entry.partition("=")
        try:
            out[key] = json.loads(value)
        except ValueError:
            out[key] = value
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--plants", type=int, default=2)
    parser.add_argument("--days", type=float, default=3.)
    parser.add_argument("--label", default=None, help="the entry's name (default: <plants>p_<days>d)")
    parser.add_argument("--outputs", action="store_true", help="measure a frame and two checkpoints at the end")
    parser.add_argument("--checkpoint-at", type=int, nargs="*", default=(), help="hours after which to checkpoint")
    parser.add_argument("--resume", default=None, help="a checkpoint folder to resume from")
    parser.add_argument("--scenario", nargs="*", default=(), metavar="KEY=VALUE",
                        help="the plants' scenario entries (e.g. leaf_ports=bundles vapour_path=air_spaces)")
    parser.add_argument("--threads", default="auto",
                        help="Compute(threads=): auto (default) or a number, the budget of numba's kernels and the "
                             "solves' blocks (BLAS stays at 1)")
    parser.add_argument("--workers", default="auto", help="Compute(workers=): auto (default) or a number")
    parser.add_argument("--linear-solver", default=None,
                        help="Compute(linear_solver=): the device's default (superlu on cpu, cupy on cuda), "
                             "superlu_reuse, cholmod, pardiso, mumps")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="Compute(device=)")
    parser.add_argument("--variability", type=int, default=None, metavar="SEED",
                        help="the plants differ (plant_variability: each its own seeded root draws)")
    parser.add_argument("--frames", action="store_true",
                        help="record to outputs/benchmarks/frames/<label>: a frame every 24 h, the sections of "
                             "DISPLAY_PLANTS drawn as instances (no checkpoint)")
    parser.add_argument("--profile-hours", type=int, nargs=2, default=(), metavar=("FIRST", "LAST"),
                        help="cProfile the main process over these steps (outputs/benchmarks/profiles/<label>.prof)")
    parser.add_argument("--cadence", nargs="*", default=(), metavar="MODEL.STEP=DURATION[@start|@end]",
                        help='schedule steps\' cadences, e.g. WheatPlant.anatomy="1 d" CaribuLightModel.light="3 h"')
    parser.add_argument("--memory", action="store_true",
                        help="measure what the scene holds by owner at the end (memory_probe.report)")
    parser.add_argument("--tracemalloc", type=int, default=0, metavar="FRAMES",
                        help="trace the allocations (FRAMES frames each) and record those alive at the end")
    parser.add_argument("--thread-scaling", action="store_true",
                        help="time the last step's large solves and parallel kernels at 1-12 threads")
    arguments = parser.parse_args(argv)
    from openalea.metafspm.compute import Compute
    compute = Compute(threads=arguments.threads if arguments.threads == "auto" else int(arguments.threads),
                      workers=arguments.workers if arguments.workers == "auto" else int(arguments.workers),
                      linear_solver=arguments.linear_solver, device=arguments.device)
    label = arguments.label or f"{arguments.plants}p_{arguments.days:g}d"
    result = benchmark(arguments.plants, arguments.days, arguments.outputs, tuple(arguments.checkpoint_at),
                       arguments.resume, label, scenario_entries(arguments.scenario), compute, arguments.variability,
                       arguments.frames, tuple(arguments.profile_hours),
                       dict(given.split("=", 1) for given in arguments.cadence), arguments.memory,
                       arguments.tracemalloc, arguments.thread_scaling)
    os.makedirs(BENCHMARKS, exist_ok=True)
    path = os.path.join(BENCHMARKS, f"{result['commit']}.json")
    entries = {}
    if os.path.exists(path):
        with open(path) as f:
            entries = json.load(f)
    entries[label] = result
    with open(path, "w") as f:
        json.dump(entries, f, indent=1)
    print(summary(result))
    print(f"written to {path} ({label})")


if __name__ == "__main__":
    main()
