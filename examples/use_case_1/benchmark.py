"""
Use case 1's benchmark (devplan_release_optimisation.md §5): the use case's scene (run.scene, in memory: no frames,
no checkpoints) run hour by hour from the sowing, each step's time by model (Scene.timing), the kinds of the
reduction evaluated (their number and time), the plants' nodes, and the memory (RSS after each step; after
malloc_trim and the peak at the end).

    python benchmark.py [--plants 2] [--days 3] [--label text]

By default 2 plants to day 3 (the use case's fast regime); --days 8 for the day-8 numbers, --plants 8 --days 4 for the
scaling. The results go to outputs/benchmarks/<commit>.json (untracked), one entry per mode ("2p_3d") updated in place,
and a summary is printed. Run it under a memory bound, one at a time (systemd-run --user --scope -p MemoryMax=8G ...).
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
        self.kinds, self.seconds = 0, 0.
        self._evaluate = reduction_module._evaluate

        def counted(reduction, parameters, requests):
            started = time.perf_counter()
            try:
                return self._evaluate(reduction, parameters, requests)
            finally:
                self.kinds += len(requests)
                self.seconds += time.perf_counter() - started

        reduction_module._evaluate = counted

    def take(self) -> tuple:
        out = (self.kinds, self.seconds)
        self.kinds, self.seconds = 0, 0.
        return out


def benchmark(plants: int, days: float) -> dict:
    counter = KindCounter()
    started = time.perf_counter()
    scene = run.scene(plants, folder=None)
    built = time.perf_counter() - started
    counter.take()
    steps = []
    previous = {}
    for hour in range(int(round(days * 24))):
        step_started = time.perf_counter()
        scene.run()
        seconds = time.perf_counter() - step_started
        kinds, kind_seconds = counter.take()
        timing = {name: value - previous.get(name, 0.) for name, value in scene.timing.items()}
        previous = dict(scene.timing)
        nodes = sum(population.data_structure.n_nodes() for population in scene.populations)
        steps.append({"hour": hour + 1, "seconds": seconds, "models": timing, "kinds": kinds,
                      "kind_seconds": kind_seconds, "nodes": nodes, "rss_gb": resident_memory()})
        print(f"h {hour + 1:4d}  {seconds:7.2f} s  kinds {kinds:3d} ({kind_seconds:6.2f} s)  nodes {nodes:6d}  "
              f"RSS {steps[-1]['rss_gb']:.2f} GB", flush=True)
    scene.finish()
    rss = resident_memory()
    trim_memory()
    compute = getattr(scene, "compute", None)
    result = {"commit": commit(), "date": time.strftime("%Y-%m-%d %H:%M:%S"), "plants": plants, "days": days,
              "machine": {"processors": os.cpu_count(), "platform": platform.platform(),
                          "python": platform.python_version()},
              "compute": compute.as_dict() if compute is not None else None,
              "environment": {name: os.environ.get(name) for name in
                              ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "MALLOC_ARENA_MAX")},
              "build_seconds": built, "total_seconds": time.perf_counter() - started,
              "rss_end_gb": rss, "rss_after_trim_gb": resident_memory(), "peak_rss_gb": peak_memory(),
              "timing": dict(scene.timing), "steps": steps}
    return result


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
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--plants", type=int, default=2)
    parser.add_argument("--days", type=float, default=3.)
    parser.add_argument("--label", default=None, help="the entry's name (default: <plants>p_<days>d)")
    arguments = parser.parse_args(argv)
    result = benchmark(arguments.plants, arguments.days)
    os.makedirs(BENCHMARKS, exist_ok=True)
    path = os.path.join(BENCHMARKS, f"{result['commit']}.json")
    entries = {}
    if os.path.exists(path):
        with open(path) as f:
            entries = json.load(f)
    label = arguments.label or f"{arguments.plants}p_{arguments.days:g}d"
    entries[label] = result
    with open(path, "w") as f:
        json.dump(entries, f, indent=1)
    print(summary(result))
    print(f"written to {path} ({label})")


if __name__ == "__main__":
    main()
