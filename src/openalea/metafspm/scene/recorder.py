"""
A scene's outputs, written as it runs::

  Scene(..., recorder=SceneRecorder("outputs", segment_plants=["p0"], frame_period="1 d", checkpoint_period="10 d"))
  Scene(..., recorder="outputs")              the defaults: the summaries of every DataStructure, every step

Its periods are durations ("12 h", "1 d": coupling.durations; a number is seconds), each a whole number of the
scene's steps, or "step" (every step); an output is written when the scene's time is a multiple of its period.

  outputs/
    run.json                          the run: status (running, finished, stopped, failed, interrupted; a failure's
                                      error), started, ended, wall time, iterations, simulated time, time step, wall
                                      time per model and per component ("<Model>.<Component>": Scene.timing), the
                                      calls that ran per component (Scene.calls: a cadence's fewer than the steps),
                                      versions, the threads chosen and in use ("compute": "threads": the
                                      Compute's threads, its budget, the bandwidth-bound sections' ceiling and its
                                      source, the processors and core types, BLAS libraries, numba:
                                      Compute.thread_info), the time, calls and threads of each kernel, solver and
                                      step ("timing": Compute.timing_info; functor.<Class>.<step>);
                                      rewritten after every step (a running run's progress)
    scenario.json                     the scene: time step, size, populations (model, plants and their scenarios),
                                      environment models and their scenarios, forcings, the run configuration
                                      (Compute), the recorder's settings
    forcings/<name>.csv               the forcings given as tables
    Plant/<Population>/summaries.csv  one row per plant and recorded step: time, plant, the float state variables of
                                      every component on its DataStructure or a view of it (each over the entities of
                                      the components declaring it; extensive ones summed, intensive ones averaged by
                                      their declared weight, Plant ones as is; descriptors and edges left out)
    Plant/<Population>/segments.csv   the node state variables of the segment_plants, every segment_period
    Soil/<Model>/summaries.csv        one row per recorded step: time, the grid's float state variables (extensive
    Atmosphere/<Model>/summaries.csv  ones summed, the others averaged by their weight or the cells' volumes), by the
                                      environment model's compartment
    <name>.csv                        the tables given as tables={name: function(scene) -> a row (dict), a
                                      DataFrame, or None for no row}, or declared: Summary(location, where, sum, mean,
                                      min, max, max_change, per)
    conservation.csv                  the balances of the quantities the graph systems conserve
                                      (graph_system(conserved=...)), at the summaries' period: time, model,
                                      component, system, quantity, inflow, outflow, change, residual (change - inflow
                                      + outflow, since the last row), unit
    frames/frame_<step>.glb, frames/scene.fspm.json
                                      the whole scene (encoding3d), every frame_period: every scale, every
                                      row; the components' display shapes (e.g. GRANAP's sections) every
                                      display_period, a reduction's groups as instances of their templates
                                      (display="instances", the default)
    frames/parts/<key>.glb            the parts the frames share (encoding3d 0.4): each template drawn and each
                                      kind's recovery operator, written once (its key: its content's hash), listed
                                      in scene.fspm.json; those no frame refers to removed (a resumed run)
    frames.fspm.zip                   with pack_at_finish (or recorder.pack()): the frames folder in one archive
                                      (encoding3d.pack), written when the run finishes and checked (every file a
                                      member, its size and CRC-32 read back, the archive valid), then the folder
                                      removed (pack_keep_folder=True keeps it); ParaView and encoding3d.read open it
                                      as the folder, and a run resumed from a checkpoint unpacks it to go on
    frames/views.json                 with frame_views: the layers ParaView shows (encoding3d's views), written at the
                                      first frame
    checkpoints/step_<iteration>/     Scene.checkpoint every checkpoint_period, the last keep_checkpoints kept (2:
                                      the last two; 1: the last only, each removed once the next is written)
    checkpoints/shared/               the checkpoints' parts saved once (scene.shared: anatomy templates, sections),
                                      those no kept checkpoint refers to removed

Every output is switched on or off by its own argument (summaries=False, frame_period=None, ...). The initial state
is recorded when the run begins, then every period, and the last state when it finishes (Scene.stop). Without an
output folder, the summaries and tables are kept in memory: recorder.table(name) gives any of them back, e.g.
table("Plant/RootModel/summaries") or a table's name. After a restore (Scene.load_checkpoint), every table is cut
back to the restored time and appended to from there.
"""
import json
import os
import platform
import re
import shutil
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from openalea.metafspm.coupling.declaration import EXTENSIVE_KINDS, INTENSIVE_KINDS, MASSIC_KINDS

COMPARTMENTS = ("Plant", "Soil", "Atmosphere")
_PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # openalea.metafspm's folder
_NOT_SUMMARISED = ("edge", "face", "Connection")      # the relations' variables (flows, conductances between nodes)
CELL_VOLUME = "<cell volume>"                         # the weight of a grid's intensive variables: its cells' volumes


def _warn(message: str) -> None:
    """A warning shown at the caller's line outside metafspm (the user's scene.run()), not the recorder's."""
    if sys.version_info >= (3, 12):
        warnings.warn(message, stacklevel=2, skip_file_prefixes=(_PACKAGE,))
    else:
        warnings.warn(message, stacklevel=2)


def base_of(ds):
    """The DataStructure *ds* is a view of (a component's located or entity view), else *ds* itself."""
    while ds is not None and "_base" in getattr(ds, "__dict__", {}):
        ds = ds.__dict__["_base"]
    return ds


def grouped(values, owner, groups: int, how: str, weights=None) -> np.ndarray:
    """
    *values* aggregated by group (*owner*: each entity's group index, -1 for none): "sum", "mean" (weighted by
    *weights* when given), "min" or "max"; the entities with a NaN value (or a NaN weight) left out; NaN for a group
    without entities (a mean, an extreme) or with no weight. (Grouping(owner, groups).aggregate: the same, the owners
    sorted out once for several variables.)
    """
    return Grouping(owner, groups).aggregate(values, how, weights)


class Grouping:
    """The entities' groups (*owner*: each entity's group index, -1 for none; *groups* of them), sorted out once for
    every variable aggregated over them (a summary's columns): aggregate(values, how, weights) is grouped()'s, the
    same entities in the same order."""

    def __init__(self, owner, groups: int):
        owner = np.asarray(owner, dtype=np.int64)
        self.groups = groups
        self.valid = owner >= 0
        self.every = bool(self.valid.all())
        self.owner = owner if self.every else owner[self.valid]

    def aggregate(self, values, how: str, weights=None) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if not self.every:
            values = values[self.valid]
        kept = np.isfinite(values)
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if not self.every:
                weights = weights[self.valid]
            kept &= np.isfinite(weights)
        owner = self.owner
        if not kept.all():
            owner, values = owner[kept], values[kept]
            weights = None if weights is None else weights[kept]
        groups = self.groups
        if how == "sum":
            return np.bincount(owner, weights=values, minlength=groups)
        if how == "mean":
            weights = np.ones(values.size) if weights is None else weights
            total = np.bincount(owner, weights=weights, minlength=groups)
            sums = np.bincount(owner, weights=values * weights, minlength=groups)
            return np.divide(sums, total, out=np.full(groups, np.nan), where=total > 0)
        if how in ("min", "max"):
            out = np.full(groups, np.inf if how == "min" else -np.inf)
            (np.minimum if how == "min" else np.maximum).at(out, owner, values)
            out[~np.isfinite(out)] = np.nan
            return out
        raise ValueError(f"grouped(how=...): 'sum', 'mean', 'min' or 'max', got '{how}'")


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


def _period(value, name: str):
    """A recorder period: None (never), "step" (every step) or a duration in seconds (coupling.durations: "12 h")."""
    if value is None or value == "step":
        return value
    from openalea.metafspm.coupling.durations import positive_duration
    return positive_duration(value, f"SceneRecorder({name}=)")


def _whole(length: float, unit: float) -> bool:
    """Whether *length* is a whole number (at least 1) of *unit*."""
    ratio = float(length) / float(unit)
    return round(ratio) >= 1 and abs(ratio - round(ratio)) <= 1e-9 * max(1., ratio)


def _on(period, time: float) -> bool:
    """Whether scene time *time* (s) is on *period* ("step": every step; seconds: a multiple of it)."""
    if period == "step":
        return True
    ratio = float(time) / float(period)
    return abs(ratio - round(ratio)) <= 1e-9 * max(1., abs(ratio))


def _remove_file(path: str) -> None:
    """The file *path* removed, when there is one."""
    if os.path.isfile(path):
        os.remove(path)


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


def _scene_location(ds, location: str) -> str:
    """*location* named as *ds* stores it: "node" for the scale of its nodes (SubOrgan on a segment-mode MPG,
    Compartment on an anatomy-mode one), else as given."""
    if location in ("node", "scalar") or location in getattr(ds, "_coarse_scale_names", lambda: [])():
        return location
    try:
        if ds._scale_name(ds._node_scale()) == location:
            return "node"
    except (AttributeError, StopIteration, IndexError, TypeError, ValueError):
        pass
    if location == "cell" and hasattr(ds, "cell_volume"):        # a grid's cells
        return location
    raise ValueError(f"Summary: location '{location}' is not one of the DataStructure's ('node', the scale of its "
                     f"nodes, or {ds._coarse_scale_names() if hasattr(ds, '_coarse_scale_names') else []})")


_COMPARISON = re.compile(r"^\s*(<=|>=|==|!=|<|>)\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")
_OPERATORS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal, "==": np.equal,
              "!=": np.not_equal}


class Summary:
    """
    A table of the recorder declared by what it holds (SceneRecorder(tables={"leaves": Summary(...)})): after each
    recorded step, one row (per="scene": every population's entities together; per="plant": one row per plant, its
    planting name in column plant) of aggregates over the entities at *location* ("node", a scale: "SubOrgan",
    "Organ", ...; the scale of a DataStructure's nodes is its "node") where every condition of *where* holds:

      where   {name: condition}: name a variable (at *location* or at a coarser one: each entity reads its owner's),
              "label" (the entities' MTG labels) or "<Scale>.label" (their owner's at that scale); condition a value,
              a list of values (label names for labels), or a comparison (">0", "<=0.03", "!=4")
      sum     [names] the sums, or {name: factor}: the sum of name × factor (an integral: An per area × the area)
      mean    [names] the means weighted by their declared weight (state_variable(weight=...), else plain), or
              {name: weight}: weighted by that variable (None: plain)
      min, max  [names]: the extremes (columns <name>_min, <name>_max)
      max_change [names]: the largest change since the step before, |value - ds.previous(value)| (column
              <name>_max_change), over the entities there at both (NaN at the first row): the Scene keeps the values
              before each step (ds.previous)

    The entities without a value (NaN) are left out; a mean or an extreme over no entity is NaN, a sum 0.
    *population*: one population's (its model's name), by default every population holding *location*; or an
    environment model's DataStructure (its id in Scene.data_structures(), e.g. "RichardsSoilModel": a grid's
    "cell"), per="scene" only.
    """

    def __init__(self, location: str, where: dict = None, sum=(), mean=(), min=(), max=(), max_change=(),
                 per: str = "scene", population: str = None):
        if per not in ("scene", "plant"):
            raise ValueError(f"Summary(per=...): 'scene' or 'plant', got '{per}'")
        self.location, self.where, self.per, self.population = location, dict(where or {}), per, population
        self.sum = dict(sum) if isinstance(sum, dict) else {name: None for name in sum}
        self.mean = dict(mean) if isinstance(mean, dict) else {name: "declared" for name in mean}
        self.min, self.max, self.max_change = list(min), list(max), list(max_change)
        columns = list(self.sum) + list(self.mean) + [f"{n}_min" for n in self.min] + [f"{n}_max" for n in self.max]
        columns += [f"{n}_max_change" for n in self.max_change]
        twice = sorted({column for column in columns if columns.count(column) > 1})
        if twice:
            raise ValueError(f"Summary: columns {twice} asked twice")
        if not columns:
            raise ValueError("Summary: give what it holds (sum=, mean=, min=, max=, max_change=)")

    def __repr__(self) -> str:
        return (f"Summary({self.location!r}, where={self.where}, sum={self.sum}, mean={self.mean}, min={self.min}, "
                f"max={self.max}, max_change={self.max_change}, per={self.per!r})")

    def _data_structures(self, scene) -> list:
        """[(population or None, ds)] the table covers: the populations (or the one named), or the environment
        DataStructure named."""
        found = [(population, population.data_structure) for population in scene.populations
                 if self.population is None or population.name == self.population]
        if not found and self.population is not None:
            entry = scene.data_structures().get(self.population)
            if entry is not None:
                if self.per == "plant":
                    raise ValueError(f"Summary(per='plant'): '{self.population}' is no population")
                found = [(None, entry[0])]
        return found

    def previous_variables(self, scene) -> list:
        """[(ds, names)] whose values before each step the Scene keeps (ds.previous) for max_change."""
        if not self.max_change:
            return []
        return [(ds, list(self.max_change)) for _, ds in self._data_structures(scene)]

    def _selected(self, ds, location: str) -> np.ndarray:
        """Whether each entity at *location* of *ds* is selected by *where*."""
        n = int(np.prod(ds._location_shape(location)))
        selected = np.ones(n, dtype=bool)
        for name, condition in self.where.items():
            if name == "label" or name.endswith(".label"):
                scale = None if name == "label" else name[:-len(".label")]
                if scale is None:
                    values = ds.labels() if location == "node" else ds.labels(location, per_node=False)
                elif location == "node":
                    values = ds.labels(scale)
                else:
                    index = np.asarray(ds.owner(scale, of=location))
                    values = np.where(index >= 0, ds.labels(scale, per_node=False)[np.maximum(index, 0)], None)
                wanted = condition if isinstance(condition, (list, tuple, set)) else [condition]
                selected &= np.isin(np.asarray(values, dtype=object), list(wanted))
                continue
            if not ds.has(name):
                raise KeyError(f"Summary: where's variable '{name}' is not on the DataStructure")
            values = np.asarray(ds.get(name), dtype=float)
            source = ds.location(name)
            if source != location:
                values = ds._mask_values_at("Summary", name, values, location) if location == "node" else \
                    np.asarray(ds._map(values, source, location, "broadcast"))
            comparison = _COMPARISON.match(condition) if isinstance(condition, str) else None
            if comparison is not None:
                selected &= _OPERATORS[comparison.group(1)](values, float(comparison.group(2)))
            elif isinstance(condition, (list, tuple, set, np.ndarray)):
                selected &= np.isin(values, np.asarray(list(condition), dtype=float))
            else:
                selected &= values == condition
        return selected

    def _values(self, ds, name: str, location: str) -> np.ndarray:
        if not ds.has(name):
            raise KeyError(f"Summary: variable '{name}' is not on the DataStructure")
        if ds.location(name) != location:
            raise ValueError(f"Summary at {self.location}: '{name}' is at {ds.location(name)}")
        return np.asarray(ds.get(name), dtype=float).reshape(-1)

    def __call__(self, scene) -> pd.DataFrame:
        recorder = scene.recorder if isinstance(getattr(scene, "recorder", None), SceneRecorder) else SceneRecorder()
        parts = []                                       # (population, ds, location, group of each entity, names)
        for population, ds in self._data_structures(scene):
            location = _scene_location(ds, self.location)
            selected = self._selected(ds, location)
            if self.per == "plant":
                owner = np.asarray(ds.owner("Plant") if location == "node" else ds.owner("Plant", of=location))
            else:
                owner = np.zeros(selected.size, dtype=np.int64)
            parts.append((population, ds, location, np.where(selected, owner, -1)))
        if not parts:
            raise ValueError(f"Summary: no population{' ' + repr(self.population) if self.population else ''} in "
                             "the scene (nor environment DataStructure of that id)")
        rows = []
        if self.per == "plant":
            for population, ds, location, owner in parts:
                names = population.plant_names()
                plants = np.asarray(ds.entity_ids("Plant"))
                table = self._aggregated(recorder, scene, [(ds, location, owner)], plants.size)
                rows.append(pd.DataFrame({"time": float(scene.time),
                                          "plant": [names.get(int(v), str(v)) for v in plants], **table}))
            return pd.concat(rows, ignore_index=True)
        return pd.DataFrame({"time": [float(scene.time)], **self._aggregated(
            recorder, scene, [(ds, location, owner) for _, ds, location, owner in parts], 1)})

    def _aggregated(self, recorder, scene, parts, groups: int) -> dict:
        """{column: values per group} over the *parts* [(ds, location, group of each entity)] pooled."""
        def pooled(name, factor=None):
            values = np.concatenate([self._values(ds, name, location) for ds, location, _ in parts])
            if factor is not None:
                values = values * np.concatenate([self._values(ds, factor, location) for ds, location, _ in parts])
            return values
        grouping = Grouping(np.concatenate([np.asarray(owner, dtype=np.int64) for _, _, owner in parts]), groups)
        out = {}
        for name, factor in self.sum.items():
            out[name] = grouping.aggregate(pooled(name, factor), "sum")
        for name, weight in self.mean.items():
            if weight == "declared":
                plans = [recorder.summary_plan(scene, ds).get(name) for ds, _, _ in parts]
                declared = {plan[2] for plan in plans if plan is not None}
                weight = declared.pop() if len(declared) == 1 else None
            out[name] = grouping.aggregate(pooled(name), "mean", None if weight is None else pooled(weight))
        for name in self.min:
            out[f"{name}_min"] = grouping.aggregate(pooled(name), "min")
        for name in self.max:
            out[f"{name}_max"] = grouping.aggregate(pooled(name), "max")
        for name in self.max_change:
            change = np.concatenate([np.abs(self._values(ds, name, location) - _previous(ds, name, location))
                                     for ds, location, _ in parts])
            out[f"{name}_max_change"] = grouping.aggregate(change, "max")
        return out


def _previous(ds, name: str, location: str) -> np.ndarray:
    """*name*'s values before the step at *location* (NaN: new entities, or none kept yet: the first row)."""
    try:
        return np.asarray(ds.previous(name), dtype=float).reshape(-1)
    except KeyError:
        return np.full(int(np.prod(ds._location_shape(location))), np.nan)


FRAME_WRITINGS = ("background", "synchronous")


def _snapshot(value):
    """*value* (a frame, or a part of it) with every array copied and every container new: nothing in it is shared
    with what built it (the scene's arrays, a component's caches), so that the next steps may change those while it
    is written. The shared parts (encoding3d Template, Operator: immutable, keyed by their content) are kept."""
    import copy
    from openalea.encoding3d import Operator, Template
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if value is None or isinstance(value, (str, bytes, int, float, bool, np.generic, Template, Operator, type)):
        return value
    if hasattr(value, "__dict__"):
        out = copy.copy(value)
        out.__dict__.update({name: _snapshot(item) for name, item in value.__dict__.items()})
        return out
    return copy.deepcopy(value)


class SceneRecorder:
    """
    The outputs of a Scene (see the module; every period a duration, "12 h", or "step"): *folder* (None: in memory,
    no files); *summaries* and their *summary_period*; *segment_plants* (planting names) and *segment_period*;
    *frame_period* and *frame_options*
    (io.encoding3d.scene_to_frame's: geometry, standard_names, data_structures, scales, entities, variables, fold:
    the plants drawn folded into the scene's periodic pattern, for display);
    *display_period*: the frames that also carry the components' display shapes (display_triangles(), e.g. GRANAP's
    sections: drawings only, their data rows are in every frame), every display_period (a multiple of
    frame_period) and in the last frame; None (the default): none, the frames hold the data and the geometry
    light models get. *display_plants*: the plants (planting names, e.g. ("p0", "p3")) whose shapes those frames
    draw, None for all: a stand's frames bounded by the few plants drawn (the others' reduced sections' interiors
    not recovered). A light log leaves out data explicitly, by frame_options (e.g. scales={"Wheat": ["Plant",
    "Axis", "Organ", "SubOrgan"]}: no anatomy rows in population Wheat's; variables={"Wheat": {"Compartment": ["psi",
    "tissue"], "SubOrgan": ["length", "radius"], "*": []}}: a few variables, the structure kept, see
    io.encoding3d.filter_frame; "recorded": the variables the declarations record, their record=: state variables
    and inputs, not parameters, settings or a grid's geometric descriptors); a variables filter keeps, besides what
    it names, what the frames' views select, colour and size tubes by (frame_variables(): the views given, or the
    default ones, those of the frame unfiltered by variables, read at the first frame), so that it names the data
    only.
    *display*: how those frames draw the groups of a population's Reduction (io.encoding3d.to_frame's display):
    "instances" (default; a population without a reduction is drawn as with "full") each group an instance of its
    template, its unknown on the template's nodes a derived field the reader rebuilds for the instances it selects
    (ParaView: Select at scale), the templates and the kinds' recovery operators shared parts written once in
    frames/parts; "full" every interior recovered and written in every display frame; None the reduced groups not
    drawn. *pack_at_finish*: the frames folder packed into frames.fspm.zip when the run finishes (*pack_compression*
    "deflate", the default, "lzma": about 3 times smaller, 5 times slower to pack, or "stored"), the archive checked,
    then the folder removed, unless *pack_keep_folder* (recorder.pack()).
    *checkpoint_period* and *keep_checkpoints* (None: all); *tables* {name: function(scene)} and *table_period*.
    *frame_views*: the frames' views.json (encoding3d's views: layers for ParaView), written at the first frame: None
    (none), "default" (io.encoding3d.default_views: the shoots, the roots as tubes, the soil grids), a views dict, or the
    path of a views file; its names are checked against that frame, with a warning for each one it lacks. A restored
    run keeps the views.json it finds.
    *frame_writing*: "background" (default) each frame built in the step's thread (io.encoding3d.scene_to_frame),
    then a snapshot of it (its arrays copied: nothing the next steps change) written by one writer thread
    (encoding3d.write, then the frames' index) while the next steps run; a frame waits for the one before (one
    snapshot in memory at most); a checkpoint, pack(), the run's end and a restore wait for it, and an error of the
    writer is raised in the run's thread there or at the next frame. "synchronous": written in the step's thread.
    The files are the same, byte for byte.
    """

    def __init__(self, folder: str = None, summaries: bool = True, summary_period="step", segment_plants=(),
                 segment_period="1 d", frame_period=None, frame_options: dict = None,
                 checkpoint_period=None, keep_checkpoints: int = 2, tables: dict = None,
                 table_period="step", frame_views=None, display_period=None, display_plants=None,
                 display: str = "instances", pack_at_finish: bool = False, pack_compression: str = "deflate",
                 pack_keep_folder: bool = False, conservation: bool = True, frame_writing: str = "background"):
        self.folder = os.fspath(folder) if folder is not None else None
        self.summaries, self.summary_period = bool(summaries), _period(summary_period, "summary_period")
        self.conservation = bool(conservation)
        self.segment_plants = {str(name) for name in segment_plants}
        self.segment_period = _period(segment_period, "segment_period")
        self.frame_period = _period(frame_period, "frame_period")
        self.frame_options = dict(frame_options or {})
        if "display_meshes" in self.frame_options:
            raise ValueError("SceneRecorder: the frames' display shapes are given by display_period (a duration; "
                             "None: none), not frame_options['display_meshes']")
        self.display_period = _period(display_period, "display_period")
        self.display_plants = None if display_plants is None else (
            [str(display_plants)] if isinstance(display_plants, str) else [str(name) for name in display_plants])
        if self.display_plants is not None and not self.display_period:
            raise ValueError("SceneRecorder(display_plants=...) are the plants whose display shapes the frames draw: "
                             "give it a display_period")
        if self.display_period and not self.frame_period:
            raise ValueError(f"SceneRecorder(display_period={display_period!r}): the frames carrying display shapes, "
                             "a multiple of frame_period: give a frame_period")
        from openalea.metafspm.io.encoding3d import DISPLAYS
        if display not in DISPLAYS:
            raise ValueError(f"SceneRecorder(display=...): one of {DISPLAYS}, got {display!r}")
        if "display" in self.frame_options:
            raise ValueError("SceneRecorder: the reduced groups' drawing is given by display=, not "
                             "frame_options['display']")
        self.display = display
        if frame_writing not in FRAME_WRITINGS:
            raise ValueError(f"SceneRecorder(frame_writing=...): one of {FRAME_WRITINGS}, got {frame_writing!r}")
        self.frame_writing = frame_writing
        self._writer = None            # the frames' writer thread (frame_writing "background"), made at the first frame
        self._writing = None           # the frame being written there (a Future)
        self.pack_at_finish, self.pack_compression = bool(pack_at_finish), str(pack_compression)
        self.pack_keep_folder = bool(pack_keep_folder)
        if self.pack_at_finish and not self.frame_period:
            raise ValueError("SceneRecorder(pack_at_finish=True) packs the frames: give it a frame_period")
        if self.pack_compression not in ("deflate", "lzma", "stored"):
            raise ValueError(f"SceneRecorder(pack_compression=...): 'deflate', 'lzma' or 'stored', got "
                             f"{pack_compression!r}")
        self.checkpoint_period = _period(checkpoint_period, "checkpoint_period")
        self.keep_checkpoints = keep_checkpoints
        self.tables, self.table_period = dict(tables or {}), _period(table_period, "table_period")
        if self.folder is None:
            for name, value in (("segment_plants", self.segment_plants), ("frame_period", self.frame_period),
                                ("checkpoint_period", self.checkpoint_period)):
                if value:
                    raise ValueError(f"SceneRecorder({name}=...) writes files: give it a folder")
        self.resume = False            # after a restore: files are appended to, under their existing header
        self._columns = {}
        self._memory = {}              # {table: [DataFrame]}, without an output folder
        self._frames = None            # the frames index's entries, read at the first frame written
        self._parts = set()            # the keys of the parts those frames use (frames/parts)
        self._called = {}              # {table: the last iteration its function was called at}
        self._run = None
        self._started = None
        self._warned = set()           # the unweighted means warned of
        if self.folder is not None:
            os.makedirs(self.folder, exist_ok=True)
        if self.frame_period:
            from openalea.metafspm.io.encoding3d import _encoding3d
            _encoding3d()                 # fails now, not after the first steps, without openalea.encoding3d
        self.frame_views = self._views_given(frame_views)
        self._views_written = False
        self._views = None
        self._frame_variables = None   # (the frames' variables filter,), set at the first frame (frame_variables)

    # ── the run ───────────────────────────────────────────────────────────────

    def begin(self, scene) -> None:
        """At the scene's first step (a restored one's too): run.json and scenario.json, and the initial state."""
        self._started = datetime.now()
        self._check_periods(scene.time_step)
        if self.folder is not None:
            previous = self._read_run() if self.resume else {}
            self._run = {"status": "running", "started": previous.get("started", self._started.isoformat()),
                         "time_step": float(scene.time_step), "versions": _versions(), "host": platform.node(),
                         "command": " ".join(sys.argv),
                         "compute": {"threads": scene.compute.thread_info()}}
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
        if self.checkpoint_period and _on(self.checkpoint_period, scene.time):
            self.wait_for_frames()             # the checkpoint sees the frames written, their index complete
            self._checkpoint(scene)
        elif self.folder is not None and self._run is not None:
            self._write_run(scene)

    def finish(self, scene, status: str = "finished", error: BaseException = None) -> None:
        """The run's end: the last state (when not recorded at this step) and its status."""
        self._record(scene, final=True)
        try:
            self.wait_for_frames()
        except Exception:
            if error is None:              # (a run that failed keeps its own error)
                raise
        finally:
            self._close_writer()
        if self.pack_at_finish and self._frames:
            self.pack()
        if self.folder is not None and self._run is not None:
            self._run["status"] = status
            if error is not None:
                self._run["error"] = f"{type(error).__name__}: {error}"
            self._write_run(scene, ended=True)

    def _check_periods(self, time_step: float) -> None:
        """Every period a whole number of the scene's steps (*time_step*, s), the display's of frame_period."""
        for name in ("summary_period", "segment_period", "frame_period", "display_period", "checkpoint_period",
                     "table_period"):
            period = getattr(self, name)
            if period not in (None, "step") and not _whole(period, time_step):
                raise ValueError(f"SceneRecorder({name}={period:g} s): a whole number of the scene's steps "
                                 f"({time_step:g} s)")
        if self.display_period:
            display = time_step if self.display_period == "step" else self.display_period
            frames = time_step if self.frame_period == "step" else self.frame_period
            if not _whole(display, frames):
                raise ValueError(f"SceneRecorder(display_period={display:g} s): the frames carrying display shapes, a "
                                 f"multiple of frame_period ({frames:g} s)")

    def _record(self, scene, initial: bool = False, final: bool = False) -> None:
        iteration, time = scene.iteration, scene.time
        if self.frame_period and (_on(self.frame_period, time) or final) and self._due("frames", iteration):
            self._write_frame(scene, display=bool(self.display_period)
                              and (_on(self.display_period, time) or final))
        if self.summaries and (_on(self.summary_period, time) or final) and self._due("summaries", iteration):
            self.write_summaries(scene)
        if (self.segment_plants and (_on(self.segment_period, time) or final)
                and self._due("segments", iteration)):
            self.write_segments(scene)
        if (self.conservation and not initial and (_on(self.summary_period, time) or final)
                and self._due("conservation", iteration)):
            self.write_conservation(scene)
        for name, function in self.tables.items():
            if (_on(self.table_period, time) or final) and self._due(f"table {name}", iteration):
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
        """The scene's components working on *ds* or on a view of it (an environment component on a population's MPG
        included)."""
        return [c for c in scene.components if base_of(getattr(c, "data_structure", None)) is ds]

    def summary_plan(self, scene, ds) -> dict:
        """summary_plan's plan (_summary_plan), kept for the scene's iteration while *ds*'s variables and topology
        are the same (its version, topology_version): the summaries and the tables of one record read it once."""
        key = (id(ds), getattr(ds, "version", None), getattr(ds, "topology_version", None))
        iteration = (id(scene), getattr(scene, "iteration", None), getattr(scene, "time", None))
        kept = self.__dict__.get("_plans")
        if kept is None or kept[0] != iteration:
            kept = self.__dict__["_plans"] = (iteration, {})
        hit = kept[1].get(key)
        if hit is None or hit[0] is not ds:
            hit = kept[1][key] = (ds, self._summary_plan(scene, ds))
        return hit[1]

    def _summary_plan(self, scene, ds) -> dict:
        """
        {name: (location, kind, weight, holder, part, entities)} of what summaries.csv holds for *ds*: the float state
        variables of every component on *ds* or on one of its views (each at its location on *ds*: a view's
        SubOrgans', the nodes', a Plant's), and of a light model's union of the plants (at the union's entities of
        *ds*); descriptors, the relations' variables (edges, faces) and variables of no kind (below the Plant) left out.
        *weight*: the weight of an intensive variable's mean, its declared weight (state_variable(weight=...)) when
        it is at the same location, else a grid's cell volumes (CELL_VOLUME), else None (a plain mean; a warning
        once, listing them). *holder*: the
        DataStructure the values are read from (*ds*, or the union); *part*: the union's part of *ds* (else None);
        *entities*: the entities of the location the declaring components own (a boolean mask: the entity views' of
        the components declaring it, e.g. a root model's radius over the roots only), None for all.
        """
        plan, unweighted = {}, []
        for component in scene.components:
            on = getattr(component, "data_structure", None)
            parts = [(on, None)] if base_of(on) is ds else [
                (on, part) for part in getattr(on, "parts", ()) if base_of(part) is ds]
            for holder, part in parts:
                for name, spec in getattr(component, "_variable_specs", {}).items():
                    if spec.variable_type != "state_variable" or not holder.has(name):
                        continue
                    if part is None:
                        location = ds.location(name)
                    elif holder.location(name) != "node" or name in plan:
                        continue
                    else:                                 # a union's node: the part's entity
                        location = part.__dict__.get("_at", "node")
                    kind = spec.kind
                    if (location in _NOT_SUMMARISED or kind == "descriptor"
                            or (kind is None and location not in ("Plant", "scalar"))
                            or np.asarray(holder.get(name)).dtype.kind != "f"):
                        continue
                    entities = None
                    if part is None and "_entities" in on.__dict__ and on.__dict__["_at"] == location:
                        # a variable at an entity view's entities: theirs only
                        entities = np.zeros(int(np.prod(ds._location_shape(location))), dtype=bool)
                        entities[np.asarray(ds.index_of(on.entity_ids("node"), location), dtype=np.int64)] = True
                    if name in plan:                      # declared by several components: their entities together
                        known = plan[name]
                        merged = None if known[5] is None or entities is None else known[5] | entities
                        plan[name] = known[:5] + (merged,)
                        continue
                    weight = spec.weight if kind in INTENSIVE_KINDS + MASSIC_KINDS else None
                    if weight is not None and not ((ds if part is None else holder).has(weight) and (
                            ds.location(weight) if part is None else part.__dict__.get("_at", "node")) == location):
                        weight = None
                    if (kind in INTENSIVE_KINDS + MASSIC_KINDS and weight is None and location == "cell"
                            and callable(getattr(ds, "cell_volume", None))):
                        weight = CELL_VOLUME                  # a grid's cells: their volumes
                    if (kind in INTENSIVE_KINDS + MASSIC_KINDS and weight is None
                            and location not in ("Plant", "scalar")):
                        unweighted.append(f"{type(component).__name__}.{name}")
                    plan[name] = (location, kind, weight, holder, part, entities)
        key = (id(ds), tuple(sorted(unweighted)))
        if unweighted and key not in self._warned:
            self._warned.add(key)
            _warn("SceneRecorder: intensive variables averaged without a declared weight (plain means; "
                  "state_variable(weight=...) gives the weight of their mean): " + ", ".join(unweighted))
        return plan

    @staticmethod
    def _columns_of(ds, name: str, values=None, location: str = None) -> dict:
        """{column: values per entity} of variable *name*: one column, or one per component of a vector variable."""
        stored = np.asarray(ds.get(name) if values is None else values, dtype=float)
        n_entities = int(np.prod(ds._location_shape(location or ds.location(name))))
        flat = stored.reshape(n_entities, -1)
        if flat.shape[1] == 1:
            return {name: flat[:, 0]}
        return {f"{name}_{j}": flat[:, j] for j in range(flat.shape[1])}

    @staticmethod
    def _planned_values(ds, name, holder, part, layouts: dict = None):
        """The values of a summarised variable at its entities of *ds*: a union's at the entities of *ds*'s part it
        holds (by their ids: the union's layout may be a step behind the part's growth), NaN elsewhere. *layouts*: a
        dict keeping each union part's layout (its offset in the union, which of its ids *ds* holds and where) for
        the next variables of the same summaries."""
        if name == CELL_VOLUME:
            cells = int(np.prod(ds._location_shape("cell")))
            return np.broadcast_to(np.asarray(ds.cell_volume(), dtype=float), (cells,))
        if part is None:                                   # a view's variable: the base's, every entity
            return np.asarray(ds.get(name), dtype=float)
        layout = None if layouts is None else layouts.get((id(holder), id(part)))
        if layout is None:
            k = next(j for j, other in enumerate(holder.parts) if other is part)
            offset = int(sum(len(ids) for ids in holder._ids[:k]))
            ids = np.asarray(holder._ids[k], dtype=np.int64)
            location = part.__dict__.get("_at", "node")
            known = np.asarray(ds.entity_ids(location), dtype=np.int64)
            present = np.isin(ids, known)
            layout = (offset, ids.size, known.size, present,
                      np.asarray(ds.index_of(ids[present], location), dtype=np.int64))
            if layouts is not None:
                layouts[(id(holder), id(part))] = layout
        offset, size, count, present, rows = layout
        held = np.asarray(holder.get(name), dtype=float)[offset:offset + size]
        out = np.full(count, np.nan)
        out[rows] = held[present]
        return out

    def plant_summaries(self, scene, population) -> pd.DataFrame:
        """One row per plant: time, plant (its planting name), its float state variables: extensive ones summed over
        its entities, intensive ones their mean weighted by their declared weight (else plain), Plant ones as is;
        the entities without a value (NaN) left out."""
        ds = population.data_structure
        plants = np.asarray(ds.entity_ids("Plant"))
        names = population.plant_names()
        table = {"time": float(scene.time), "plant": [names.get(int(v), str(v)) for v in plants]}
        owners, groupings, layouts = {}, {}, {}
        for name, (location, kind, weight, holder, part, entities) in self.summary_plan(scene, ds).items():
            values = self._planned_values(ds, name, holder, part, layouts)
            weights = None if weight is None else self._planned_values(ds, weight, holder, part, layouts)
            if location == "Plant":
                table.update(self._columns_of(ds, name, values, location))
                continue
            if location == "scalar":
                for column, value in self._columns_of(ds, name, values, location).items():
                    table[column] = np.full(plants.size, float(np.ravel(value)[0]))
                continue
            if location not in owners:
                owners[location] = np.asarray(ds.owner("Plant") if location == "node" else
                                              ds.owner("Plant", of=location), dtype=np.int64)
            grouping = groupings.get((location, id(entities)))     # the owners sorted out once per entities
            if grouping is None or grouping[0] is not entities:
                owner = owners[location] if entities is None else np.where(entities, owners[location], -1)
                grouping = groupings[(location, id(entities))] = (entities, Grouping(owner, plants.size))
            for column, column_values in self._columns_of(ds, name, values, location).items():
                table[column] = grouping[1].aggregate(column_values, "sum" if kind in EXTENSIVE_KINDS else "mean",
                                                      weights)
        return pd.DataFrame(table)

    def environment_summary(self, scene, ds) -> pd.DataFrame:
        """One row: time, the float state variables of *ds* (an environment model's), extensive ones summed over
        their entities, intensive ones their mean weighted by their declared weight (a grid's: its cells' volume),
        else plain; the entities without a value (NaN) left out."""
        row, layouts = {"time": float(scene.time)}, {}
        for name, (location, kind, weight, holder, part, entities) in self.summary_plan(scene, ds).items():
            values = self._planned_values(ds, name, holder, part, layouts)
            weights = None if weight is None else self._planned_values(ds, weight, holder, part, layouts)
            for column, column_values in self._columns_of(ds, name, values, location).items():
                if location == "scalar":
                    row[column] = float(np.ravel(column_values)[0])
                    continue
                owner = np.zeros(column_values.size, dtype=np.int64) if entities is None else \
                    np.where(entities, 0, -1)
                row[column] = float(grouped(column_values, owner, 1, "sum" if kind in EXTENSIVE_KINDS else "mean",
                                            weights)[0])
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

    def write_conservation(self, scene) -> None:
        """conservation.csv: a row per conserved quantity of every graph system (graph_system(conserved=...)), its
        balance since the last row (Component.conservation, reset)."""
        rows = []
        for key, (_, model, components) in scene.data_structures().items():
            for component in components:
                if not hasattr(component, "conservation"):
                    continue
                for (system, quantity), balance in component.conservation(reset=True).items():
                    rows.append({"time": float(scene.time), "model": key, "component": type(component).__name__,
                                 "system": system, "quantity": quantity, **balance})
        if rows:
            self._append("conservation", pd.DataFrame(rows))

    def selected_segments(self, scene, population) -> pd.DataFrame:
        """The node state variables of the segment_plants' nodes: time, plant, one row per node (indexed by vid)."""
        ds = population.data_structure
        names = population.plant_names()
        selected = [vid for vid, name in names.items() if name in self.segment_plants]
        if not selected:
            return None
        variables = []
        for component in self._components_on(scene, ds):
            for name, spec in getattr(component, "_variable_specs", {}).items():
                if (spec.variable_type == "state_variable" and name not in variables and ds.has(name)
                        and ds.location(name) == "node" and np.asarray(ds.get(name)).dtype.kind == "f"):
                    variables.append(name)
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
        return frame_views

    def _views_content(self, scene, folder: str, display: bool = False, unfiltered=None):
        """The frames' views (None without): a restored run's views.json, the views given, or the default ones,
        those of the frame unfiltered by variables (*unfiltered*, built when not given; io.encoding3d.default_views)."""
        from openalea.metafspm.io.encoding3d import default_views, scene_to_frame
        from openalea.encoding3d import views
        path = os.path.join(folder, views.VIEWS_FILE)
        if self.frame_views is None:
            return None
        if self.resume and os.path.exists(path):
            return views.read_views(path)
        if self.frame_views != "default":
            return self.frame_views
        if unfiltered is None:
            options = {key: value for key, value in self.frame_options.items() if key != "variables"}
            unfiltered = scene_to_frame(scene, display_meshes=display, display_plants=self.display_plants,
                                        display=self.display, **options)
        return default_views(scene, unfiltered)

    def frame_variables(self, scene, views: dict = None):
        """The frames' variables filter: frame_options' (None: every variable; "recorded": those the declarations
        record, io.encoding3d.unrecorded), with the columns *views* select, colour or size tubes by kept
        (io.encoding3d.with_view_columns): what a view shows is in every frame, whatever the filter names."""
        from openalea.metafspm.io.encoding3d import _for, resolved_variables, with_view_columns
        variables = self.frame_options.get("variables")
        if variables is None:
            return None
        found = scene.data_structures()
        variables = {identifier: resolved_variables(_for(variables, identifier), ds, components)
                     for identifier, (ds, _, components) in found.items()}
        return with_view_columns(variables, views, list(found)) if views else variables

    def _write_views(self, frame, folder: str, content: dict) -> None:
        """views.json next to the first frame, its names checked against *frame* (a warning for each it lacks)."""
        from openalea.metafspm.io.encoding3d import write_views
        from openalea.encoding3d import views
        path = os.path.join(folder, views.VIEWS_FILE)
        if not (self.resume and os.path.exists(path)):
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
        first = self._frames is None
        if not first or self.resume:            # the frames packed, the folder removed: the run goes on in it
            self._unpack_frames()
        if first:
            self._frames, self._parts = [], set()
            if self.resume and os.path.exists(index_path):       # a restored run: the frames up to its step
                index = encoding3d.read_index(index_path)
                self._frames = [entry for entry in index["frames"] if entry["step"] <= scene.iteration]
                for entry in self._frames:
                    entry.pop("path", None)
                self._parts = set(index.get("parts", {}))
            elif os.path.exists(index_path):    # an earlier run's frames in this folder: removed (its parts below)
                for entry in encoding3d.read_index(index_path)["frames"]:
                    _remove_file(os.path.join(folder, entry["file"]))
        name = f"frame_{scene.iteration:06d}.glb"
        if self._frame_variables is None:              # a filter keeps what the views show: the views first
            filtered = self.frame_options.get("variables") is not None
            self._views = self._views_content(scene, folder, display) if filtered else None
            self._frame_variables = (self.frame_variables(scene, self._views),)
        options = {**self.frame_options, "variables": self._frame_variables[0]}
        frame = scene_to_frame(scene, display_meshes=display, display_plants=self.display_plants,
                               display=self.display, **options)
        if self.frame_views is not None and not self._views_written:
            if self._views is None:
                self._views = self._views_content(scene, folder, display, unfiltered=frame)
            self._write_views(frame, folder, self._views)
        entry = {"time": float(scene.time), "step": int(scene.iteration), "file": name}
        if self.frame_writing == "synchronous":
            self._store_frame(frame, folder, entry, first)
        else:                          # a snapshot written by the writer thread, once the frame before is
            self.wait_for_frames()     # (one snapshot in memory at most)
            snapshot = _snapshot(frame)
            del frame
            if self._writer is None:
                from concurrent.futures import ThreadPoolExecutor
                self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="metafspm-frames")
            self._writing = self._writer.submit(self._store_frame, snapshot, folder, entry, first)
        return os.path.join(folder, name)

    def _store_frame(self, frame, folder: str, entry: dict, first: bool) -> None:
        """*frame* written to folder/entry["file"] (its parts each once), and the frames' index updated."""
        from openalea.metafspm.io.encoding3d import _encoding3d
        encoding3d = _encoding3d()
        index_path = os.path.join(folder, encoding3d.codec.INDEX_FILE)
        self._parts.update(encoding3d.write(os.path.join(folder, entry["file"]), frame))   # its parts: once each
        self._frames = [known for known in self._frames if known["step"] != entry["step"]]
        self._frames.append(entry)
        encoding3d.write_index(index_path, self._frames, time_unit="s", parts=sorted(self._parts))
        if first:                     # the parts of frames not listed (an earlier run in this folder) removed
            encoding3d.prune_parts(index_path)
            self._parts = set(encoding3d.read_index(index_path).get("parts", {}))

    def wait_for_frames(self) -> None:
        """The frame being written by the writer thread (frame_writing "background") complete: its file and the
        frames' index written; the writer's error, if any, raised here."""
        writing, self._writing = self._writing, None
        if writing is not None:
            writing.result()

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.shutdown(wait=True)
            self._writer = None

    def __getstate__(self):
        self.wait_for_frames()
        state = dict(self.__dict__)
        state.update(_writer=None, _writing=None)
        return state

    def pack(self, keep_folder: bool = None) -> str:
        """The frames folder packed into frames.fspm.zip next to it (encoding3d.pack, pack_compression: each file
        an unchanged member, read as the folder), the archive checked (every file a member, its size and CRC-32
        read back, the archive valid: else encoding3d.ArchiveCheckError, the folder kept), then the folder removed
        unless *keep_folder* (None: pack_keep_folder); its path. SceneRecorder(pack_at_finish=True) calls it when
        the run finishes. Frames written after it (the run going on, or resumed) unpack the archive first."""
        from openalea.metafspm.io.encoding3d import _encoding3d
        encoding3d = _encoding3d()
        keep_folder = self.pack_keep_folder if keep_folder is None else bool(keep_folder)
        self.wait_for_frames()                 # the frame being written, complete before the folder is packed
        path = encoding3d.pack(os.path.join(self.folder, "frames"), os.path.join(self.folder, "frames.fspm.zip"),
                               compression=self.pack_compression, keep_folder=keep_folder)
        if self._run is not None:
            self._run["frames_archive"] = os.path.relpath(path, self.folder)
            if self._run.get("ended"):                 # packed after the run finished: run.json says so too
                _write_json(os.path.join(self.folder, "run.json"), self._run)
        return path

    def _unpack_frames(self) -> None:
        """frames/ back from frames.fspm.zip when the folder was removed by pack (a run going on, or resumed)."""
        folder = os.path.join(self.folder, "frames")
        archive = os.path.join(self.folder, "frames.fspm.zip")
        if not os.path.exists(os.path.join(folder, "scene.fspm.json")) and os.path.isfile(archive):
            from openalea.metafspm.io.encoding3d import _encoding3d
            _encoding3d().unpack(archive, folder)

    def _checkpoint(self, scene) -> str:
        """Scene.checkpoint into checkpoints/step_<iteration>, their parts saved once in checkpoints/shared; the older
        ones beyond keep_checkpoints removed, and the shared parts none of those kept refers to."""
        from openalea.metafspm.scene.shared import prune
        folder = os.path.join(self.folder, "checkpoints")
        path = os.path.join(folder, f"step_{scene.iteration:06d}")
        shared = os.path.join(folder, "shared")
        scene.checkpoint(path, shared=shared)
        if self.keep_checkpoints is not None:
            kept = sorted(name for name in os.listdir(folder) if name.startswith("step_"))
            for name in kept[:max(0, len(kept) - int(self.keep_checkpoints))]:
                shutil.rmtree(os.path.join(folder, name))
            prune(shared, [os.path.join(folder, name) for name in sorted(os.listdir(folder))
                           if name.startswith("step_")])
        if self._run is not None:
            self._run["last_checkpoint"] = os.path.relpath(path, self.folder)
            self._write_run(scene)
        return path

    def resume_at(self, scene) -> None:
        """After a restore at *scene*'s time: every table cut back to its rows up to that time (the interrupted
        run's later rows removed), the frames' index to its steps up to that iteration; appended to from here."""
        self.wait_for_frames()
        self.resume = True
        self._columns, self._frames = {}, None
        self._called = {output: scene.iteration for output in ("frames", "summaries", "segments", "conservation",
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
        self._unpack_frames()
        index_path = os.path.join(self.folder, "frames", "scene.fspm.json")
        if os.path.exists(index_path):
            from openalea.metafspm.io.encoding3d import _encoding3d
            encoding3d = _encoding3d()
            index = encoding3d.read_index(index_path)
            frames = [entry for entry in index["frames"] if entry["step"] <= scene.iteration]
            for entry in index["frames"]:          # the interrupted run's later frames removed
                if entry["step"] > scene.iteration:
                    _remove_file(entry["path"])
            for entry in frames:
                entry.pop("path", None)
            encoding3d.write_index(index_path, frames, time_unit="s", parts=sorted(index.get("parts", {})))
            encoding3d.prune_parts(index_path)          # the parts only the removed frames used

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
                         timing={key: round(value, 6) for key, value in scene.timing.items()},
                         calls=dict(getattr(scene, "calls", {})))
        if getattr(scene, "coupling", "explicit") != "explicit":         # a fixed point's iterations (Scene)
            self._run["coupling_iterations"] = dict(scene.coupling_iterations)
        self._run.setdefault("compute", {})["timing"] = scene.compute.timing_info()     # the kernels' and solvers'
        device = scene.compute.device_info()     # the GPU used and its transfers, None on the cpu (not recorded)
        if device is not None:
            self._run.setdefault("compute", {})["device"] = device
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
                plant = {key: _described(getattr(row, key)) for key in ("plant", "x", "y", "depth", "rotation")}
                if population.planted_later:
                    plant["planting_time"] = float(row.planting_time)
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
                         "display_period": self.display_period, "display_plants": self.display_plants,
                         "display": self.display, "frame_writing": self.frame_writing,
                         "pack_at_finish": self.pack_at_finish,
                         "pack_compression": self.pack_compression, "pack_keep_folder": self.pack_keep_folder,
                         "frame_views": self.frame_views if self.frame_views in (None, "default") else "given",
                         "checkpoint_period": self.checkpoint_period, "tables": sorted(self.tables),
                         "table_period": self.table_period, "conservation": self.conservation}}
        _write_json(os.path.join(self.folder, "scenario.json"), content)
