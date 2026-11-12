"""
The runner layer (scene.runner): a scene's populations in its own process (processes=1) or in worker processes
(Compute(processes=n)) give the same bits: the grid's state, every plant's values, the light's union, the recorder's
tables, the frames' values and the checkpoints; the placement of processes on the cores; the failures.
"""
import json
import os
import signal
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.compute import Compute
from openalea.metafspm.coupling.cadence import Cadence
from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, state_variable
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.iteration import FixedPoint
from openalea.metafspm.coupling.schedule import Step
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.scene import runner
from openalea.metafspm.scene.environment import EnvironmentModel
from openalea.metafspm.scene.recorder import SceneRecorder, Summary
from openalea.metafspm.scene.scene import Scene
from openalea.metafspm.solve.decorator import rate

from growth import DOC, CarbonProbe, RootGrowthProbe
from scene_doubles import DT, RootPopulation, SceneExudation, SceneSoil, doubles_planting, soil_translator

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="the workers fork")

PLANTS = 4
RATES = [0.1 + k / 7. for k in range(PLANTS)]
LAYOUT = ["a", "b", "c", "d"]                     # one population per plant: up to four workers


@dataclass
class OrderedShade(FunctionalComponent):
    """A light's toy on the plants' union: each element shaded by those before it in the union's order."""
    struct_mass: float = input_variable(**DOC, by="RootGrowthProbe", default=0., scale=scales.SubOrgan)
    shade: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="intensive")

    @rate
    def _shade(self, struct_mass):
        before = np.cumsum(np.asarray(struct_mass) / 3.) - np.asarray(struct_mass) / 3.
        return np.exp(-before)


class ShadeModel(EnvironmentModel):
    compartment = "Atmosphere"
    schedule = (Step("shade", cadence=Cadence(DT, at="start")),)

    def __init__(self, plant_data_structures, x_range, y_range, time_step):
        super().__init__(plant_data_structures, x_range, y_range, time_step)
        self.shade = OrderedShade(data_structure=self.plant_data_structure())
        self.components = [self.shade]


def _scene(processes=1, recorder=None, populations=LAYOUT, coupling="explicit", threads=1, checks=()):
    table = doubles_planting([RootPopulation] * PLANTS, scenarios=[{"exudation_rate": rate} for rate in RATES],
                             population=list(populations))
    translator = soil_translator("SceneExudation")
    translator.link("OrderedShade", "struct_mass", "RootGrowthProbe", {"struct_mass": 1.})
    return Scene(table, environment=[SceneSoil, ShadeModel], translator=translator, time_step=DT,
                 compute=Compute(threads=threads, processes=processes, checks=checks), recorder=recorder,
                 coupling=coupling)


def _outcome(scene) -> dict:
    grid = scene.environment[0].grid
    union = scene.environment[1].plant_data_structure()
    outcome = {"grid " + name: np.asarray(grid.get(name)).copy() for name in ("exudation", "soil_nitrate")}
    for population in scene.populations:
        data_structure = population.data_structure
        for variable in ("exudation", "soil_nitrate", "struct_mass"):
            outcome[f"{population.name} {variable}"] = np.asarray(data_structure.get(variable)).copy()
    outcome["union shade"] = np.asarray(union.get("shade")).copy()
    table = Summary("SubOrgan", sum=["exudation", "struct_mass"], mean=["soil_nitrate"])(scene)
    for column in table.columns:
        outcome["summary " + column] = table[column].to_numpy()
    return outcome


def _run(processes, steps=6, **options) -> dict:
    scene = _scene(processes, **options)
    try:
        for _ in range(steps):
            scene.run()
        return _outcome(scene)
    finally:
        scene.finish()


@pytest.mark.parametrize("processes", [2, PLANTS])
def test_the_workers_give_the_bits_of_one_process(processes):
    reference = _run(1)
    assert reference["grid exudation"].max() > 0.
    outcome = _run(processes)
    assert outcome.keys() == reference.keys()
    for key, values in reference.items():
        np.testing.assert_array_equal(outcome[key], values, err_msg=f"processes={processes}: {key}")


def test_the_copies_patched_from_the_workers_deltas_are_checked_against_their_mpgs():
    """The plants grow every step: each copy is patched from its worker's delta publication (scene.runner), checked
    against the MPG sent whole (Compute(checks={"publication"})), and the outcome is one process's."""
    reference = _run(1)
    outcome = _run(2, checks={"publication"})
    for key, values in reference.items():
        np.testing.assert_array_equal(outcome[key], values, err_msg=key)


def test_the_layouts_and_the_process_counts_give_the_same_bits():
    """One population of four plants in one process, and four populations in four workers: the same bits."""
    one = _run(1, populations=["roots"] * PLANTS)
    four = _run(PLANTS)
    names = {"a": 0, "b": 1, "c": 2, "d": 3}
    for key in ("grid exudation", "grid soil_nitrate", "union shade"):
        np.testing.assert_array_equal(four[key], one[key], err_msg=key)
    for column in [key for key in one if key.startswith("summary ")]:
        np.testing.assert_array_equal(four[column], one[column], err_msg=column)
    assert sum(four[f"{name} exudation"].size for name in names) == one["roots exudation"].size


def _tables(folder) -> dict:
    found = {}
    for root, _, files in os.walk(folder):
        for name in files:
            if name.endswith(".csv"):
                with open(os.path.join(root, name), "rb") as file:
                    found[os.path.relpath(os.path.join(root, name), folder)] = file.read()
    return found


def test_the_recorder_tables_and_checkpoints_are_the_same_at_two_processes(tmp_path):
    options = dict(summaries=True, segment_plants=["p0", "p3"], segment_period="1 h", checkpoint_period="2 h",
                   keep_checkpoints=None)
    for processes in (1, 2):
        scene = _scene(processes, recorder=SceneRecorder(str(tmp_path / str(processes)), **options))
        scene.simulate(4)
    one, two = _tables(tmp_path / "1"), _tables(tmp_path / "2")
    assert one and one.keys() == two.keys()
    for name in one:
        assert one[name] == two[name], name
    with open(tmp_path / "2" / "run.json") as file:
        run = json.load(file)
    assert run["status"] == "finished" and run["processes"]["count"] == 2
    assert sorted(sum(run["processes"]["groups"], [])) == LAYOUT
    assert all(entry["peak"] > 0 for entry in run["processes"]["measured_MB_per_process"].values())


def test_a_checkpoint_of_two_processes_resumes_at_one_and_at_two(tmp_path):
    scene = _scene(2)
    for _ in range(3):
        scene.run()
    scene.checkpoint(str(tmp_path / "checkpoint"))
    for _ in range(3):
        scene.run()
    reference = _outcome(scene)
    scene.finish()
    for processes in (1, 2):
        resumed = _scene(processes)
        resumed.load_checkpoint(str(tmp_path / "checkpoint"))
        for _ in range(3):
            resumed.run()
        outcome = _outcome(resumed)
        resumed.finish()
        for key, values in reference.items():
            np.testing.assert_array_equal(outcome[key], values, err_msg=f"resumed at {processes}: {key}")


def test_a_user_write_reaches_the_worker_and_apply_runs_there():
    scene = _scene(2)
    scene.run()
    data_structure = scene.population("a").data_structure
    data_structure.set("exudation_rate", np.full(np.shape(data_structure.get("exudation_rate")), 5.))
    scene.run()
    rates = scene.apply(_exudation_rates, population="a")
    assert len(rates) == 1 and np.all(rates[0][0] == 5.) and rates[0][1] != os.getpid()
    scene.finish()


def _exudation_rates(data_structure):
    return np.asarray(data_structure.get("exudation_rate")).copy(), os.getpid()


def _fail(data_structure):
    raise KeyError("no such thing")


def test_an_exception_in_a_worker_is_raised_again_with_its_group():
    scene = _scene(2)
    scene.run()
    with pytest.raises(KeyError, match="group .*population.* a.*no such thing"):
        scene.apply(_fail, population="a")
    scene.finish()


def test_a_killed_worker_ends_the_run_naming_its_group(tmp_path):
    scene = _scene(2, recorder=SceneRecorder(str(tmp_path)))
    scene.run()
    os.kill(scene.runner._handles[1]["pid"], signal.SIGKILL)
    with pytest.raises(RuntimeError, match=r"group 1 .*plants .*ended at step 1: exit status -9 \(killed"):
        scene.simulate(2)
    with open(tmp_path / "run.json") as file:
        assert json.load(file)["status"] == "failed"
    assert all(handle["pid"] is None for handle in scene.runner._handles)
    assert not os.path.exists(scene.runner._folder)              # its shared memory removed with the run


def test_the_folders_of_runs_killed_without_their_cleanup_are_removed_at_the_next_start(tmp_path):
    import subprocess
    ended = subprocess.Popen(["true"])
    ended.wait()                                                # a coordinator that is no longer running
    stale = tmp_path / f"{runner.FOLDER_PREFIX}{ended.pid}-abc"
    running = tmp_path / f"{runner.FOLDER_PREFIX}{os.getpid()}-def"
    other = tmp_path / "metafspm-notapid-ghi"
    for folder in (stale, running, other):
        (folder / "shared").mkdir(parents=True)
    assert runner.remove_stale_folders(str(tmp_path)) == [str(stale)]
    assert not stale.exists() and running.exists() and other.exists()


def test_the_archetype_store_is_the_runs_and_its_folder_goes_with_it():
    from openalea.metafspm.solve import archetype_store
    scene = _scene(2)
    scene.run()
    folder = scene.runner._folder
    assert archetype_store.current() is None                   # current while the populations run only
    assert os.path.isdir(os.path.join(folder, "archetypes"))
    scene.finish()
    assert not os.path.exists(folder)


# ── a fixed-point coupling and links between populations across processes ──

ITERATED = ("SceneSoil.nitrate", "a.exudation", "b.exudation", "c.exudation", "d.exudation")
INTERLEAVED = ("a.exudation", "b.exudation", "SceneSoil.nitrate", "c.exudation", "a.exudation", "d.exudation")


def _iterated(processes, steps, **options) -> tuple:
    scene = _scene(processes, coupling=FixedPoint(steps, **options))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(4):
                scene.run()
        not_converged = [str(warning.message) for warning in caught if "not converged" in str(warning.message)]
        return _outcome(scene), dict(scene.coupling_iterations), dict(scene.calls), not_converged
    finally:
        scene.finish()


@pytest.mark.parametrize("steps", [ITERATED, INTERLEAVED], ids=["soil first", "interleaved"])
def test_a_fixed_point_across_processes_gives_the_bits_and_iterations_of_one_process(steps):
    """The iteration runs in the coordinator, the populations' iterated steps in their workers, restored there from
    their starts: the same iterates (bit for bit), the same iteration counts and calls at 1, 2 and 4 processes."""
    reference, counts, calls, _ = _iterated(1, steps, relative_tolerance=1e-12, max_iterations=50)
    assert counts["steps"] == 4 and counts["most"] > 2 and counts["not_converged"] == 0
    for processes in (2, PLANTS):
        outcome, other_counts, other_calls, _ = _iterated(processes, steps, relative_tolerance=1e-12,
                                                          max_iterations=50)
        assert other_counts == counts and other_calls == calls
        for key, values in reference.items():
            np.testing.assert_array_equal(outcome[key], values, err_msg=f"processes={processes}: {key}")


def test_a_fixed_point_not_converged_across_processes_warns_as_one_process():
    reference, counts, _, warned = _iterated(1, ITERATED, relative_tolerance=1e-15, max_iterations=2)
    assert counts["not_converged"] == 4 and len(warned) == 4
    outcome, other_counts, _, other_warned = _iterated(2, ITERATED, relative_tolerance=1e-15, max_iterations=2)
    assert other_counts == counts and other_warned == warned
    for key, values in reference.items():
        np.testing.assert_array_equal(outcome[key], values, err_msg=key)


def test_a_fixed_point_ending_on_a_non_finite_iterate_raises_as_one_process():
    messages = []
    for processes in (1, 2):
        scene = _scene(processes, coupling=FixedPoint(ITERATED, max_iterations=3))
        soil = scene.model("SceneSoil").grid
        nitrate = np.array(soil.get("soil_nitrate"), dtype=float)
        nitrate.flat[0] = np.nan
        soil.set("soil_nitrate", nitrate)
        with pytest.raises(FloatingPointError) as raised:
            scene.run()
        messages.append(str(raised.value))
        scene.finish()
    assert messages[0] == messages[1] and "soil_nitrate" in messages[0]


@dataclass
class PlantSignal(FunctionalComponent):
    """A plant's toy signal: its total mass (a scalar), received by another population's plants (a link between two
    populations), and what they make of it."""
    struct_mass: float = input_variable(**DOC, by="RootGrowthProbe", default=0., scale=scales.SubOrgan)
    received: float = input_variable(**DOC, by="PlantSignal", default=0., scale=scales.SubOrgan)
    total: float = state_variable(**DOC, default=0., location="scalar", kind="intensive")
    echo: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")

    @rate
    def _total(self, struct_mass):
        return np.sum(struct_mass)

    @rate
    def _echo(self, struct_mass, received):
        return struct_mass * (1. + received)


class SignalPopulation(CompositeModel):
    initiators = (RootGrowthProbe,)
    parts = {"growth": RootGrowthProbe, "carbon": CarbonProbe, "exudation": SceneExudation, "signal": PlantSignal}


def _signals(processes) -> dict:
    """b reads a (before it: this step's total), a reads d (after it: the step before's), c reads b (two rounds)."""
    table = doubles_planting([SignalPopulation] * PLANTS, population=LAYOUT)
    translator = soil_translator("SceneExudation")
    for receiver, provider in (("b", "a"), ("a", "d"), ("c", "b")):
        translator.link(f"{receiver}.PlantSignal", "received", f"{provider}.PlantSignal", {"total": 1.})
    scene = Scene(table, environment=[SceneSoil], translator=translator, time_step=DT,
                  compute=Compute(threads=1, processes=processes))
    try:
        for _ in range(4):
            scene.run()
        return {f"{population.name} {name}": np.asarray(population.data_structure.get(name)).copy()
                for population in scene.populations for name in ("received", "echo", "total", "exudation")}
    finally:
        scene.finish()


def test_links_between_populations_across_processes_give_the_bits_of_one_process():
    reference = _signals(1)
    assert np.all(reference["b received"] > 0.) and np.all(reference["c received"] > 0.)
    for processes in (2, PLANTS):
        outcome = _signals(processes)
        for key, values in reference.items():
            np.testing.assert_array_equal(outcome[key], values, err_msg=f"processes={processes}: {key}")


def test_the_rounds_follow_what_the_exchanges_read():
    """A population after the earlier ones it reads, its own earlier task, and not before an earlier one reading it."""
    process_runner = runner.ProcessRunner(runner.placement(2, [1] * PLANTS))
    a, b, c, d = (type("Population", (), {"name": name})() for name in LAYOUT)
    process_runner._reads = {"a": {"d"}, "b": {"a"}, "c": {"b"}, "d": set()}
    rounds = process_runner._rounds([(a, None), (b, None), (c, None), (d, None)])
    assert [[population.name for population, _ in tasks] for tasks in rounds] == [["a", "d"], ["b"], ["c"]]
    process_runner._reads = {name: set() for name in LAYOUT}
    rounds = process_runner._rounds([(a, 1), (b, 2), (a, 4), (d, 5)])
    assert [[(population.name, step) for population, step in tasks] for tasks in rounds] == \
        [[("a", 1), ("b", 2), ("d", 5)], [("a", 4)]]


def test_what_processes_cannot_run_is_refused():
    with pytest.raises(ValueError, match="4 population"):
        _scene(5)
    with pytest.raises(ValueError, match="processes is 'auto' or a positive integer"):
        Compute(processes=0)


# ── placement ──

def _topology(performance=0, efficient=0, low_power=0, plain=0, smt=False):
    cores, kinds, processor = [], {}, 0
    for kind, count in (("performance", performance), ("efficient", efficient), ("low_power", low_power),
                        (None, plain)):
        for _ in range(count):
            siblings = [processor, processor + 1] if smt and kind in ("performance", None) else [processor]
            cores.append(siblings)
            kinds.update({sibling: kind for sibling in siblings})
            processor += len(siblings)
    return {"cores": cores, "kinds": kinds}


def test_the_placement_on_a_hybrid_cpu():
    """The laptop's: 2 performance cores (two threads each), 8 efficient, 2 low-power left out."""
    plan = runner.placement(4, [1] * 8, _topology(performance=2, efficient=8, low_power=2, smt=True), budget_mb=1e6)
    assert plan["count"] == 4 and plan["disjoint"]
    assert plan["coordinator_cores"] == [0, 2]                      # the performance cores, one thread each
    assert plan["worker_cores"] == [[4, 5], [6, 7], [8, 9], [10, 11]]
    assert plan["groups"] == [[0, 4], [1, 5], [2, 6], [3, 7]]
    every = plan["coordinator_cores"] + sum(plan["worker_cores"], [])
    assert len(every) == len(set(every)) and not {12, 13} & set(every)
    assert plan["threads_per_process"] == {"coordinator": 2, "workers": [2, 2, 2, 2]}


def test_the_placement_on_equal_cores_and_the_larger_groups():
    plan = runner.placement(2, [3, 1, 1], _topology(plain=8), budget_mb=1e6)
    assert plan["groups"] == [[0], [1, 2]]
    assert plan["coordinator_cores"] == [0, 1, 2, 3]
    assert plan["worker_cores"] == [[4, 5], [6, 7]]
    few = runner.placement(3, [1, 1, 1], _topology(plain=2), budget_mb=1e6)
    assert not few["disjoint"] and few["count"] == 3


def test_auto_follows_the_cores_and_the_memory_budget():
    topology = _topology(performance=2, efficient=8)
    assert runner.placement("auto", [1] * 8, topology, budget_mb=1e6, coordinator_mb=0)["count"] == 8
    fits = 2 * runner.predicted_mb(3) + runner.predicted_mb(2) + 1000 + 8 * runner.COPY_PER_PLANT_MB + 1   # 3, 3, 2
    assert runner.placement("auto", [1] * 8, topology, budget_mb=fits, coordinator_mb=1000)["count"] == 3
    with pytest.warns(UserWarning, match="memory budget"):
        assert runner.placement("auto", [1] * 8, topology, budget_mb=10, coordinator_mb=0)["count"] == 0
    assert runner.placement("auto", [5], topology, budget_mb=1e6)["count"] == 0     # one population


def test_the_topology_of_this_machine_and_a_worker_on_its_cores():
    topology = runner.cpu_topology()
    assert topology["cores"] and set(sum(topology["cores"], [])) == set(topology["kinds"])
    scene = _scene(2, threads="auto")
    affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    scene.run()
    plan = scene.runner.plan
    if hasattr(os, "sched_getaffinity"):                            # the coordinator is not pinned
        assert os.sched_getaffinity(0) == affinity
    for handle, cores in zip(scene.runner._handles, plan["worker_cores"]):
        if hasattr(os, "sched_getaffinity"):
            assert os.sched_getaffinity(handle["pid"]) == set(cores)
    assert scene.apply(_threads, population="a")[0] == len(plan["worker_cores"][scene.runner._handles[0]["group"]])
    scene.finish()


def _threads(data_structure):
    from openalea.metafspm.compute import Compute
    return Compute.current().thread_count()
