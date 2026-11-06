"""
Use case 1's benchmark (devplan_release_optimisation.md §5): the use case's scene (run.scene, in memory: no frames,
no checkpoints) run hour by hour from the sowing, each step's time by model (Scene.timing), the kinds of the
reduction evaluated (their number and time), the plants' nodes, and the memory (RSS after each step; after
malloc_trim and the peak at the end).

    python benchmark.py [--plants 2] [--days 3] [--label text] [--outputs] [--checkpoint-at 168 ...]
                        [--resume outputs/benchmarks/checkpoints/<label>/step_000168]

By default 2 plants to day 3 (the use case's fast regime); --days 8 for the day-8 numbers, --plants 8 --days 4 for the
scaling; --outputs measures the outputs at the end (outputs()): a frame without and with the sections drawn (those of
run.DISPLAY_PLANTS, the use case's frame variables), two checkpoints sharing their unchanging parts, their size and
time. --checkpoint-at writes the scene's checkpoint after those hours (outputs/benchmarks/checkpoints/<label>/, the
shared parts once); --resume loads one into the freshly built scene and runs on from its hour to --days (the load's
time and memory, then each step's: a resumed run evaluates no kind again). The results go to
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

import run                                                           # noqa: E402

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
    """Counts the kinds the reductions evaluate (solve.reduction._evaluate's requests) and the time they take."""

    def __init__(self):
        self.kinds, self.seconds, self.templates = 0, 0., 0
        self.seen = set()                 # the (reduced template, signature) of the kinds evaluated so far
        self._evaluate = reduction_module._evaluate

        def counted(reduction, parameters, requests):
            started = time.perf_counter()
            try:
                return self._evaluate(reduction, parameters, requests)
            finally:
                self.kinds += len(requests)
                self.seconds += time.perf_counter() - started
                for key, _, signature in requests:        # the kinds of a template not seen before (a new anatomy)
                    if (key, signature) not in self.seen:
                        self.seen.add((key, signature))
                        self.templates += 1

        reduction_module._evaluate = counted

    def take(self) -> tuple:
        out = (self.kinds, self.seconds, self.templates)
        self.kinds, self.seconds, self.templates = 0, 0., 0
        return out


def benchmark(plants: int, days: float, measure_outputs: bool = False, checkpoint_at=(), resume: str = None,
              label: str = "run") -> dict:
    counter = KindCounter()
    started = time.perf_counter()
    scene = run.scene(plants, folder=None)
    built = time.perf_counter() - started
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
    for hour in range(first_hour, int(round(days * 24))):
        step_started = time.perf_counter()
        scene.run()
        seconds = time.perf_counter() - step_started
        kinds, kind_seconds, templates = counter.take()
        timing = {name: value - previous.get(name, 0.) for name, value in scene.timing.items()}
        previous = dict(scene.timing)
        nodes = sum(population.data_structure.n_nodes() for population in scene.populations)
        steps.append({"hour": hour + 1, "seconds": seconds, "models": timing, "kinds": kinds,
                      "new_templates": templates,
                      "kind_seconds": kind_seconds, "nodes": nodes, "rss_gb": resident_memory()})
        print(f"h {hour + 1:4d}  {seconds:7.2f} s  kinds {kinds:3d} ({kind_seconds:6.2f} s)  nodes {nodes:6d}  "
              f"RSS {steps[-1]['rss_gb']:.2f} GB", flush=True)
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
    scene.finish()
    timing, total = dict(scene.timing), time.perf_counter() - started    # the steps' (outputs() runs one more)
    rss = resident_memory()
    trim_memory()
    after_trim, peak = resident_memory(), peak_memory()
    measured = outputs(scene) if measure_outputs else None
    compute = getattr(scene, "compute", None)
    result = {"commit": commit(), "date": time.strftime("%Y-%m-%d %H:%M:%S"), "plants": plants, "days": days,
              "machine": {"processors": os.cpu_count(), "platform": platform.platform(),
                          "python": platform.python_version()},
              "compute": compute.as_dict() if compute is not None else None,
              "environment": {name: os.environ.get(name) for name in
                              ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "MALLOC_ARENA_MAX")},
              "build_seconds": built, "total_seconds": total, "resumed": loaded,
              "rss_end_gb": rss, "rss_after_trim_gb": after_trim, "peak_rss_gb": peak,
              "timing": timing, "steps": steps, "outputs": measured}
    return result


def folder_size(path: str) -> int:
    """The bytes of the files under *path*."""
    return sum(os.path.getsize(os.path.join(folder, name)) for folder, _, names in os.walk(path) for name in names)


def outputs(scene) -> dict:
    """The outputs of *scene* now, written under outputs/benchmarks/outputs (removed after): a frame of the data, a
    frame with the sections of run.DISPLAY_PLANTS drawn (the recorder's display frames), two checkpoints one step
    apart into one shared folder (the second refers to the first's parts); their size (MB) and time (s)."""
    import shutil
    from openalea.metafspm.io.encoding3d import _encoding3d, scene_to_frame
    folder = os.path.join(BENCHMARKS, "outputs")
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    encoding3d, found = _encoding3d(), {}
    for name, options in (("frame", {}), ("display_frame", {"display_meshes": True,
                                                              "display_plants": list(run.DISPLAY_PLANTS)})):
        started = time.perf_counter()
        frame = scene_to_frame(scene, variables=run.FRAME_VARIABLES, **{"display_meshes": False, **options})
        path = os.path.join(folder, name + ".glb")
        encoding3d.write(path, frame)
        found[name] = {"seconds": time.perf_counter() - started, "MB": os.path.getsize(path) / 1e6}
        del frame
    shared = os.path.join(folder, "checkpoints", "shared")
    for k in (1, 2):
        path = os.path.join(folder, "checkpoints", f"step_{k}")
        started = time.perf_counter()
        scene.checkpoint(path, shared=shared)
        found[f"checkpoint_{k}"] = {"seconds": time.perf_counter() - started, "MB": folder_size(path) / 1e6,
                                    "shared_MB": folder_size(shared) / 1e6}
        if k == 1:
            scene.run()
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
                 f"peak {result['peak_rss_gb']:.2f} GB")
    lines.append("per model (s): " + ", ".join(f"{name} {value:.1f}" for name, value in
                                               sorted(result["timing"].items(), key=lambda item: -item[1])))
    if result.get("outputs"):
        lines.append("outputs: " + ", ".join(
            f"{name} {entry['MB']:.1f} MB in {entry['seconds']:.2f} s"
            + (f" (shared {entry['shared_MB']:.1f} MB)" if "shared_MB" in entry else "")
            for name, entry in result["outputs"].items()))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--plants", type=int, default=2)
    parser.add_argument("--days", type=float, default=3.)
    parser.add_argument("--label", default=None, help="the entry's name (default: <plants>p_<days>d)")
    parser.add_argument("--outputs", action="store_true", help="measure a frame and two checkpoints at the end")
    parser.add_argument("--checkpoint-at", type=int, nargs="*", default=(), help="hours after which to checkpoint")
    parser.add_argument("--resume", default=None, help="a checkpoint folder to resume from")
    arguments = parser.parse_args(argv)
    label = arguments.label or f"{arguments.plants}p_{arguments.days:g}d"
    result = benchmark(arguments.plants, arguments.days, arguments.outputs, tuple(arguments.checkpoint_at),
                       arguments.resume, label)
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
