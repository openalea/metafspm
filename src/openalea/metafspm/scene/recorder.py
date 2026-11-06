"""
A scene's outputs, written as it runs::

  Scene(..., recorder=SceneRecorder("outputs", segment_plants=["p0"], frame_period=24, checkpoint_period=240))
  Scene(..., recorder="outputs")              the defaults: the summaries of every DataStructure, every step

  outputs/
    run.json                          the run: status (running, finished, stopped, failed, interrupted; a failure's
                                      error), started, ended, wall time, iterations, simulated time, time step, wall
                                      time per model, versions, the threads in use (BLAS libraries, numba) and the
                                      process pool (None: in the process); rewritten after every step (a running
                                      run's progress)
    scenario.json                     the scene: time step, size, populations (model, plants and their scenarios),
                                      environment models and their scenarios, forcings, the run configuration
                                      (Compute), the recorder's settings
    forcings/<name>.csv               the forcings given as tables
    Plant/<Model>/summaries.csv       one row per plant and recorded step: time, plant, its float state variables
                                      (extensive ones summed over its nodes, the others averaged, Plant ones as is)
    Plant/<Model>/segments.csv        the node state variables of the segment_plants, every segment_period steps
    Soil/<Model>/summaries.csv        one row per recorded step: time, the grid's float state variables (extensive
    Atmosphere/<Model>/summaries.csv  ones summed, the others averaged), by the environment model's compartment
    <name>.csv                        the tables given as tables={name: function(scene) -> a row (dict), a
                                      DataFrame, or None for no row}
    frames/frame_<step>.glb, frames/scene.fspm.json
                                      the whole scene (encoding3d), every frame_period steps: every scale, every
                                      row; the components' display shapes (e.g. GRANAP's sections) every
                                      display_period steps
    frames/views.json                 with frame_views: the layers ParaView shows (encoding3d's views), written at the
                                      first frame
    checkpoints/step_<iteration>/     Scene.checkpoint every checkpoint_period steps, the last keep_checkpoints kept

Every output is switched on or off by its own argument (summaries=False, frame_period=None, ...). The initial state
is recorded when the run begins, then every period, and the last state when it finishes (Scene.stop). Without an
output folder, the summaries and tables are kept in memory: recorder.table(name) gives any of them back, e.g.
table("Plant/RootModel/summaries") or a table's name. After a restore (Scene.load_checkpoint), every table is cut
back to the restored time and appended to from there.
"""
import json
import os
import platform
import shutil
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from openalea.metafspm.coupling.declaration import EXTENSIVE_KINDS, INTENSIVE_KINDS, MASSIC_KINDS

COMPARTMENTS = ("Plant", "Soil", "Atmosphere")


def compartment(model) -> str:
    """The folder of an environment model's outputs: its compartment attribute, "Soil" or "Atmosphere"."""
    declared = getattr(model, "compartment", None)
    if declared not in COMPARTMENTS[1:]:
        raise ValueError(f"{type(model).__name__}.compartment: 'Soil' or 'Atmosphere' (the folder of its outputs), "
                         f"got {declared!r}")
    return declared


def _described(value):
    """*value* as JSON: numbers, text, lists and dicts as they are; arrays as lists, tables as columns, classes and
    functions by name, anything else by its repr."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value) if isinstance(value, (float, np.floating)) else int(value)
        return number if not isinstance(number, float) or np.isfinite(number) else str(number)
    if isinstance(value, np.ndarray):
        return _described(value.tolist())
    if isinstance(value, pd.DataFrame):
        return {"table": {str(column): _described(value[column].tolist()) for column in value.columns}}
    if isinstance(value, pd.Series):
        return {"series": {"index": _described(value.index.tolist()), "values": _described(value.tolist())}}
    if isinstance(value, dict):
        return {str(key if not isinstance(key, type) else key.__name__): _described(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_described(item) for item in value]
    if isinstance(value, type) or callable(value):
        return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', repr(value))}"
    return repr(value)


def _write_json(path: str, content: dict) -> None:
    temporary = path + ".part"
    with open(temporary, "w") as file:
        json.dump(content, file, indent=1)
    os.replace(temporary, path)                      # a reader never sees half a file


def _versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    found = {"python": platform.python_version()}
    for package in ("openalea.metafspm", "numpy", "pandas", "openalea.encoding3d"):
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            pass
    return found


class SceneRecorder:
    """
    The outputs of a Scene (see the module): *folder* (None: in memory, no files); *summaries* and their
    *summary_period* (steps); *segment_plants* (planting names) and *segment_period*; *frame_period* and *frame_options*
    (io.encoding3d.scene_to_frame's: geometry, standard_names, data_structures, scales, entities, variables);
    *display_period*: the frames that also carry the components' display shapes (display_triangles(), e.g. GRANAP's
    sections: drawings only, their data rows are in every frame), every display_period steps (a multiple of
    frame_period) and in the last frame; None (the default): none, the frames hold the data and the geometry
    light models get. A light log leaves out data explicitly, by frame_options (e.g. scales={"Wheat": ["Plant",
    "Axis", "Organ", "SubOrgan"]}: no anatomy rows in population Wheat's; variables={"Wheat": {"Compartment": ["psi",
    "tissue"], "SubOrgan": ["length", "radius"], "*": []}}: a few variables, the structure kept, see
    io.encoding3d.filter_frame); a variables filter must keep what the frames' views select, colour and size tubes
    by (a ValueError: given views at construction, the default ones, those of the frame unfiltered by variables, at
    the first frame).
    *checkpoint_period* and *keep_checkpoints* (None: all); *tables* {name: function(scene)} and *table_period*.
    *frame_views*: the frames' views.json (encoding3d's views: layers for ParaView), written at the first frame: None
    (none), "default" (io.encoding3d.default_views: the shoots, the roots as tubes, the soil grids), a views dict, or the
    path of a views file; its names are checked against that frame, with a warning for each one it lacks. A restored
    run keeps the views.json it finds.
    """

    def __init__(self, folder: str = None, summaries: bool = True, summary_period: int = 1, segment_plants=(),
                 segment_period: int = 24, frame_period: int = None, frame_options: dict = None,
                 checkpoint_period: int = None, keep_checkpoints: int = 2, tables: dict = None,
                 table_period: int = 1, frame_views=None, display_period: int = None):
        self.folder = os.fspath(folder) if folder is not None else None
        self.summaries, self.summary_period = bool(summaries), int(summary_period)
        self.segment_plants, self.segment_period = {str(name) for name in segment_plants}, int(segment_period)
        self.frame_period = int(frame_period) if frame_period else None
        self.frame_options = dict(frame_options or {})
        if "display_meshes" in self.frame_options:
            raise ValueError("SceneRecorder: the frames' display shapes are given by display_period (steps; None: "
                             "none), not frame_options['display_meshes']")
        self.display_period = int(display_period) if display_period else None
        if self.display_period and (not self.frame_period or self.display_period % self.frame_period):
            raise ValueError(f"SceneRecorder(display_period={display_period}): the frames carrying display shapes, a "
                             f"multiple of frame_period ({self.frame_period})")
        self.checkpoint_period = int(checkpoint_period) if checkpoint_period else None
        self.keep_checkpoints = keep_checkpoints
        self.tables, self.table_period = dict(tables or {}), int(table_period)
        if self.folder is None:
            for name, value in (("segment_plants", self.segment_plants), ("frame_period", self.frame_period),
                                ("checkpoint_period", self.checkpoint_period)):
                if value:
                    raise ValueError(f"SceneRecorder({name}=...) writes files: give it a folder")
        self.resume = False            # after a restore: files are appended to, under their existing header
        self._columns = {}
        self._memory = {}              # {table: [DataFrame]}, without an output folder
        self._frames = None            # the frames index's entries, read at the first frame written
        self._called = {}              # {table: the last iteration its function was called at}
        self._run = None
        self._started = None
        if self.folder is not None:
            os.makedirs(self.folder, exist_ok=True)
        if self.frame_period:
            from openalea.metafspm.io.encoding3d import _encoding3d
            _encoding3d()                 # fails now, not after the first steps, without openalea.encoding3d
        self.frame_views = self._views_given(frame_views)
        self._views_written = False

    # ── the run ───────────────────────────────────────────────────────────────

    def begin(self, scene) -> None:
        """At the scene's first step (a restored one's too): run.json and scenario.json, and the initial state."""
        self._started = datetime.now()
        if self.folder is not None:
            previous = self._read_run() if self.resume else {}
            self._run = {"status": "running", "started": previous.get("started", self._started.isoformat()),
                         "time_step": float(scene.time_step), "versions": _versions(), "host": platform.node(),
                         "command": " ".join(sys.argv),
                         "compute": {"threads": scene.compute.thread_info(), "pool": None}}
            if self.resume:
                self._run["wall_time_before"] = previous.get("wall_time", 0.)
                self._run["resumed"] = previous.get("resumed", []) + [
                    {"at": self._started.isoformat(), "iteration": int(scene.iteration), "time": float(scene.time)}]
            self._write_run(scene)
            self._write_scenario(scene)
        if scene.iteration == 0 and not self.resume:
            self._record(scene, initial=True)

    def record(self, scene) -> None:
        """After a step: what is due at this iteration, then run.json's progress (its iterations, time, wall time and
        timing: a running run's state, read while it runs)."""
        self._record(scene)
        if self.checkpoint_period and scene.iteration % self.checkpoint_period == 0:
            self._checkpoint(scene)
        elif self.folder is not None and self._run is not None:
            self._write_run(scene)

    def finish(self, scene, status: str = "finished", error: BaseException = None) -> None:
        """The run's end: the last state (when not recorded at this step) and its status."""
        self._record(scene, final=True)
        if self.folder is not None and self._run is not None:
            self._run["status"] = status
            if error is not None:
                self._run["error"] = f"{type(error).__name__}: {error}"
            self._write_run(scene, ended=True)

    def _record(self, scene, initial: bool = False, final: bool = False) -> None:
        iteration = scene.iteration
        if self.frame_period and (iteration % self.frame_period == 0 or final) and self._due("frames", iteration):
            self._write_frame(scene, display=bool(self.display_period)
                              and (iteration % self.display_period == 0 or final))
        if self.summaries and (iteration % self.summary_period == 0 or final) and self._due("summaries", iteration):
            self.write_summaries(scene)
        if (self.segment_plants and (iteration % self.segment_period == 0 or final)
                and self._due("segments", iteration)):
            self.write_segments(scene)
        for name, function in self.tables.items():
            if (iteration % self.table_period == 0 or final) and self._due(f"table {name}", iteration):
                row = function(scene)
                if row is not None:
                    frame = pd.DataFrame([row]) if isinstance(row, dict) else pd.DataFrame(row)
                    self._append(name, frame)

    def _due(self, output: str, iteration: int) -> bool:
        """Whether *output* was not written at *iteration* yet (and note it is now)."""
        if self._called.get(output) == iteration:
            return False
        self._called[output] = iteration
        return True

    # ── summaries, segments, tables ──────────────────────────────────────────

    @staticmethod
    def _components_on(scene, ds) -> list:
        """The scene's components working on *ds* (an environment component on a population's MPG included)."""
        return [c for c in scene.components if getattr(c, "data_structure", None) is ds]

    @classmethod
    def _state_variables(cls, scene, ds) -> dict:
        """{name: kind} of the float state variables the components on *ds* declare."""
        found = {}
        for component in cls._components_on(scene, ds):
            for name, spec in getattr(component, "_variable_specs", {}).items():
                if spec.variable_type == "state_variable" and ds.has(name) and ds.get(name).dtype.kind == "f":
                    found.setdefault(name, spec.kind)
        return found

    @staticmethod
    def _columns_of(ds, name: str) -> dict:
        """{column: values per entity} of variable *name*: one column, or one per component of a vector variable."""
        stored = np.asarray(ds.get(name), dtype=float)
        n_entities = int(np.prod(ds._location_shape(ds.location(name))))
        flat = stored.reshape(n_entities, -1)
        if flat.shape[1] == 1:
            return {name: flat[:, 0]}
        return {f"{name}_{j}": flat[:, j] for j in range(flat.shape[1])}

    def plant_summaries(self, scene, population) -> pd.DataFrame:
        """One row per plant: time, plant (its planting name), its float state variables."""
        ds = population.data_structure
        plants = np.asarray(ds.entity_ids("Plant"))
        owner = np.asarray(ds.owner("Plant"))
        counts = np.bincount(owner, minlength=plants.size)
        names = population.plant_names()
        table = {"time": float(scene.time), "plant": [names.get(int(v), str(v)) for v in plants]}
        for name, kind in self._state_variables(scene, ds).items():
            location = ds.location(name)
            for column, values in self._columns_of(ds, name).items():
                if location == "Plant":
                    table[column] = values
                elif location == "node" and kind in EXTENSIVE_KINDS:
                    table[column] = np.bincount(owner, weights=values, minlength=plants.size)
                elif location == "node" and (kind in INTENSIVE_KINDS or kind in MASSIC_KINDS):
                    sums = np.bincount(owner, weights=values, minlength=plants.size)
                    table[column] = np.divide(sums, counts, out=np.full(plants.size, np.nan), where=counts > 0)
        return pd.DataFrame(table)

    def environment_summary(self, scene, ds) -> pd.DataFrame:
        """One row: time, the float state variables of *ds* (an environment model's), extensive ones summed over
        their entities, the others averaged."""
        row = {"time": float(scene.time)}
        for name, kind in self._state_variables(scene, ds).items():
            for column, values in self._columns_of(ds, name).items():
                if kind in EXTENSIVE_KINDS:
                    row[column] = float(np.nansum(values))
                else:
                    row[column] = float(np.nanmean(values)) if values.size else np.nan
        return pd.DataFrame([row])

    def environment_data_structures(self, scene) -> list:
        """[(table, DataStructure)] of the environment models' own DataStructures (Scene.data_structures(): not the
        populations', nor their unions), each in its compartment's folder: <compartment>/<id>/summaries."""
        population_names = {population.name for population in scene.populations}
        return [(f"{compartment(model)}/{identifier}/summaries", ds)
                for identifier, (ds, model, _) in scene.data_structures().items() if identifier not in population_names]

    def write_summaries(self, scene) -> None:
        for population in scene.populations:
            self._append(f"Plant/{population.name}/summaries", self.plant_summaries(scene, population))
        for name, ds in self.environment_data_structures(scene):
            self._append(name, self.environment_summary(scene, ds))

    def selected_segments(self, scene, population) -> pd.DataFrame:
        """The node state variables of the segment_plants' nodes: time, plant, one row per node (indexed by vid)."""
        ds = population.data_structure
        names = population.plant_names()
        selected = [vid for vid, name in names.items() if name in self.segment_plants]
        if not selected:
            return None
        variables = [n for n in self._state_variables(scene, ds) if ds.location(n) == "node"]
        frame = ds.to_dataframe(names=variables, location="node")
        plant_of = np.asarray(ds.entity_ids("Plant"))[np.asarray(ds.owner("Plant"))]
        frame.insert(0, "plant", [names[int(v)] for v in plant_of])
        frame.insert(0, "time", float(scene.time))
        return frame[np.isin(plant_of, selected)].reset_index()

    def write_segments(self, scene) -> None:
        for population in scene.populations:
            frame = self.selected_segments(scene, population)
            if frame is not None:
                self._append(f"Plant/{population.name}/segments", frame)

    def _append(self, name: str, frame: pd.DataFrame) -> None:
        if self.folder is None:
            self._memory.setdefault(name, []).append(frame)
            return
        path = os.path.join(self.folder, name + ".csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if name not in self._columns and self.resume and os.path.exists(path):
            self._columns[name] = list(pd.read_csv(path, nrows=0).columns)
        if name not in self._columns:
            self._columns[name] = list(frame.columns)
            frame.to_csv(path, index=False)
        else:
            frame.reindex(columns=self._columns[name]).to_csv(path, mode="a", header=False, index=False)

    def table(self, name: str) -> pd.DataFrame:
        """A recorded table (a tables= name, or "Plant/<Model>/summaries" ...), from its file or from memory."""
        if self.folder is None:
            if name not in self._memory:
                raise KeyError(f"no table '{name}' recorded (recorded: {sorted(self._memory)})")
            return pd.concat(self._memory[name], ignore_index=True)
        path = os.path.join(self.folder, name + ".csv")
        if not os.path.exists(path):
            raise KeyError(f"no table '{name}' recorded ({path} missing)")
        return pd.read_csv(path)

    def table_names(self) -> list:
        """The names of the tables recorded so far."""
        if self.folder is None:
            return sorted(self._memory)
        found = []
        for folder, _, files in os.walk(self.folder):
            relative = os.path.relpath(folder, self.folder)
            if relative.split(os.sep)[0] in ("frames", "checkpoints", "forcings"):
                continue
            found += [os.path.normpath(os.path.join(relative, file[:-4])).replace(os.sep, "/")
                      for file in files if file.endswith(".csv")]
        return sorted(found)

    # ── frames and checkpoints ────────────────────────────────────────────────

    def _views_given(self, frame_views):
        """frame_views as kept: None, "default", or a views dict (a file read now), checked against the schema."""
        if frame_views is None:
            return None
        if not self.frame_period:
            raise ValueError("SceneRecorder(frame_views=...) are the frames' views: give it a frame_period")
        if isinstance(frame_views, str) and frame_views == "default":
            return frame_views
        from openalea.metafspm.io.encoding3d import _encoding3d
        _encoding3d()
        from openalea.encoding3d import views
        if isinstance(frame_views, (str, os.PathLike)):
            frame_views = views.read_views(os.fspath(frame_views))
        if not isinstance(frame_views, dict):
            raise TypeError(f"SceneRecorder(frame_views=...): None, 'default', a views dict or a file's path, got "
                            f"{type(frame_views).__name__}")
        report = views.check_views(frame_views)
        if not report.ok:
            raise ValueError(f"SceneRecorder(frame_views=...): {report}")
        self._check_views_kept(frame_views)
        return frame_views

    def _check_views_kept(self, content: dict) -> None:
        """A ValueError when frame_options' variables filter drops a column the views select, colour or size by."""
        from openalea.metafspm.io.encoding3d import filtered_view_columns
        if self.frame_options.get("variables") is None:
            return
        dropped = filtered_view_columns(content, self.frame_options["variables"])
        if dropped:
            raise ValueError("SceneRecorder: frame_options' variables filter drops what the frames' views show (keep "
                             "these variables): " + "; ".join(dropped))

    def _write_views(self, scene, frame, folder: str, display: bool = False) -> None:
        """views.json next to the first frame, its names checked against *frame* (a warning for each it lacks)."""
        from openalea.metafspm.io.encoding3d import default_views, scene_to_frame, write_views
        from openalea.encoding3d import views
        path = os.path.join(folder, views.VIEWS_FILE)
        if not (self.resume and os.path.exists(path)):
            content = self.frame_views
            if self.frame_views == "default":       # those of the frame unfiltered by variables, which must keep theirs
                unfiltered = frame
                if self.frame_options.get("variables") is not None:
                    options = {key: value for key, value in self.frame_options.items() if key != "variables"}
                    unfiltered = scene_to_frame(scene, display_meshes=display, **options)
                content = default_views(scene, unfiltered)
                self._check_views_kept(content)
            for issue in views.check_views(content, frame).issues:
                warnings.warn(f"SceneRecorder(frame_views=...): {issue.where}: {issue.message} (in the first frame)")
            write_views(path, content)
        self._views_written = True

    def _write_frame(self, scene, display: bool = False) -> str:
        """Write the scene's frame now (frames/frame_<step>.glb), with the components' display shapes when
        *display*, and the frames' index; the frame's path."""
        from openalea.metafspm.io.encoding3d import _encoding3d, scene_to_frame
        encoding3d = _encoding3d()
        folder = os.path.join(self.folder, "frames")
        os.makedirs(folder, exist_ok=True)
        index_path = os.path.join(folder, encoding3d.codec.INDEX_FILE)
        if self._frames is None:
            self._frames = []
            if self.resume and os.path.exists(index_path):       # a restored run: the frames up to its step
                self._frames = [entry for entry in encoding3d.read_index(index_path)["frames"]
                                if entry["step"] <= scene.iteration]
                for entry in self._frames:
                    entry.pop("path", None)
        name = f"frame_{scene.iteration:06d}.glb"
        frame = scene_to_frame(scene, display_meshes=display, **self.frame_options)
        encoding3d.write(os.path.join(folder, name), frame)
        if self.frame_views is not None and not self._views_written:
            self._write_views(scene, frame, folder, display)
        self._frames = [entry for entry in self._frames if entry["step"] != scene.iteration]
        self._frames.append({"time": float(scene.time), "step": int(scene.iteration), "file": name})
        encoding3d.write_index(index_path, self._frames, time_unit="s")
        return os.path.join(folder, name)

    def _checkpoint(self, scene) -> str:
        """Scene.checkpoint into checkpoints/step_<iteration>, the older ones beyond keep_checkpoints removed."""
        folder = os.path.join(self.folder, "checkpoints")
        path = os.path.join(folder, f"step_{scene.iteration:06d}")
        scene.checkpoint(path)
        if self.keep_checkpoints is not None:
            kept = sorted(name for name in os.listdir(folder) if name.startswith("step_"))
            for name in kept[:max(0, len(kept) - int(self.keep_checkpoints))]:
                shutil.rmtree(os.path.join(folder, name))
        if self._run is not None:
            self._run["last_checkpoint"] = os.path.relpath(path, self.folder)
            self._write_run(scene)
        return path

    def resume_at(self, scene) -> None:
        """After a restore at *scene*'s time: every table cut back to its rows up to that time (the interrupted
        run's later rows removed), the frames' index to its steps up to that iteration; appended to from here."""
        self.resume = True
        self._columns, self._frames = {}, None
        self._called = {output: scene.iteration for output in ("frames", "summaries", "segments",
                                                              *(f"table {name}" for name in self.tables))}
        if self.folder is None:
            self._memory = {}
            return
        for name in self.table_names():
            path = os.path.join(self.folder, name + ".csv")
            table = pd.read_csv(path)
            if "time" in table.columns:
                kept = table[table["time"] <= scene.time + 1e-9 * max(1., abs(scene.time))]
                if len(kept) < len(table):
                    kept.to_csv(path, index=False)
        index_path = os.path.join(self.folder, "frames", "scene.fspm.json")
        if os.path.exists(index_path):
            from openalea.metafspm.io.encoding3d import _encoding3d
            encoding3d = _encoding3d()
            frames = [entry for entry in encoding3d.read_index(index_path)["frames"] if entry["step"] <= scene.iteration]
            for entry in frames:
                entry.pop("path", None)
            encoding3d.write_index(index_path, frames, time_unit="s")

    # ── run.json, scenario.json ───────────────────────────────────────────────

    def _read_run(self) -> dict:
        path = os.path.join(self.folder, "run.json")
        if not os.path.exists(path):
            return {}
        with open(path) as file:
            return json.load(file)

    def _write_run(self, scene, ended: bool = False) -> None:
        now = datetime.now()
        self._run.update(iterations=int(scene.iteration), time=float(scene.time),
                         wall_time=(now - self._started).total_seconds() + self._run.get("wall_time_before", 0.),
                         timing={key: round(value, 6) for key, value in scene.timing.items()})
        if ended:
            self._run["ended"] = now.isoformat()
        else:
            self._run.pop("ended", None)
        _write_json(os.path.join(self.folder, "run.json"), self._run)

    def _write_scenario(self, scene) -> None:
        """scenario.json: what the scene was built with; the forcings given as tables in forcings/<name>.csv."""
        populations = []
        for population in scene.populations:
            plants = []
            for row in population.table.itertuples(index=False):
                plant = {key: _described(getattr(row, key)) for key in ("plant", "x", "y", "z", "rotation")}
                if population.emergence:
                    plant["emergence_time"] = float(row.emergence_time)
                plant["scenario"] = _described(row.scenario)
                plants.append(plant)
            populations.append({"model": _described(population.model), "plants": plants})
        forcings = {}
        for name, forcing in scene.forcings.items():
            if callable(forcing):
                forcings[name] = {"function": _described(forcing)}
            elif np.ndim(forcing) == 0 and not hasattr(forcing, "to_numpy"):
                forcings[name] = float(forcing)
            else:
                if hasattr(forcing, "to_numpy"):
                    times, values = np.asarray(forcing.index, dtype=float), np.asarray(forcing, dtype=float)
                else:
                    times, values = (np.asarray(x, dtype=float) for x in forcing)
                os.makedirs(os.path.join(self.folder, "forcings"), exist_ok=True)
                pd.DataFrame({"time": times, name: values}).to_csv(
                    os.path.join(self.folder, "forcings", f"{name}.csv"), index=False)
                forcings[name] = {"table": f"forcings/{name}.csv"}
        content = {
            "time_step": float(scene.time_step), "xrange": _described(scene.xrange),
            "yrange": _described(scene.yrange), "populations": populations,
            "environment": [{"model": _described(type(model)), "compartment": compartment(model),
                             "scenario": _described(scenario)}
                            for model, scenario in zip(scene.environment, scene.environment_scenarios)],
            "forcings": forcings, "compute": scene.compute.as_dict(),
            "recorder": {"summary_period": self.summary_period if self.summaries else None,
                         "segment_plants": sorted(self.segment_plants), "segment_period": self.segment_period,
                         "frame_period": self.frame_period, "frame_options": _described(self.frame_options),
                         "display_period": self.display_period,
                         "frame_views": self.frame_views if self.frame_views in (None, "default") else "given",
                         "checkpoint_period": self.checkpoint_period, "tables": sorted(self.tables),
                         "table_period": self.table_period}}
        _write_json(os.path.join(self.folder, "scenario.json"), content)
