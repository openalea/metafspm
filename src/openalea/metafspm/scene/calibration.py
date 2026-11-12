"""
Observations compared with a model's outputs, for calibrations (the optimiser is the user's: scipy, an MCMC, ABC)::

  observations = read_observations("observations.csv")       one row per measured value
  compare(scene, observations)                                each observation with its model value and residual
  cost = objective(run, ["kw", "kaqp"], observations, declared=[MechaHydraulics])
  scipy.optimize.least_squares(cost, x0, bounds=cost.bounds)

An observation names the *variable* it measures (a column of a recorded table: a summary's state variable, an
upscaled one such as a root system's conductance, or a column of one's own tables=), its *value*, and optionally:
the *table* it is read from (default: the recorded table holding that column), the *plant* (a planting name), the
*time* (s; default: the last recorded row), its *uncertainty* (the residual is divided by it), its *unit* (the
model's value is converted to it, from the declared unit) and the *experiment* (a virtual experiment's name: the
observations of one experiment are compared with one run, e.g. a root with and without its aquaporins inhibited).
"""
import os

import numpy as np
import pandas as pd

OBSERVATION_COLUMNS = ("variable", "value", "table", "plant", "time", "uncertainty", "unit", "experiment")


def read_observations(source) -> pd.DataFrame:
    """The observations of a CSV or Excel file, or a DataFrame: the columns of OBSERVATION_COLUMNS, *variable* and
    *value* required, the others optional (empty: not given)."""
    if isinstance(source, pd.DataFrame):
        table = source.copy()
    else:
        path = os.fspath(source)
        table = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls", ".ods")) else pd.read_csv(path)
    table.columns = [str(column).strip() for column in table.columns]
    missing = [column for column in ("variable", "value") if column not in table.columns]
    if missing:
        raise ValueError(f"observations: the columns {missing} are required (columns: {OBSERVATION_COLUMNS})")
    unknown = [column for column in table.columns if column not in OBSERVATION_COLUMNS]
    if unknown:
        raise ValueError(f"observations: unknown columns {unknown} (columns: {OBSERVATION_COLUMNS})")
    for column in OBSERVATION_COLUMNS:
        if column not in table.columns:
            table[column] = None
    table = table[list(OBSERVATION_COLUMNS)].astype(object).where(table.notna(), None)
    table["value"] = table["value"].astype(float)
    return table.reset_index(drop=True)


def _recorder(outputs):
    """(recorder, declared units) of a Scene (its recorder, its components' units), a SceneRecorder or a folder."""
    from openalea.metafspm.scene.recorder import SceneRecorder
    if isinstance(outputs, (str, os.PathLike)):
        return SceneRecorder(outputs), {}
    recorder = getattr(outputs, "recorder", outputs)
    if recorder is None or not hasattr(recorder, "table"):
        raise TypeError("compare(): a Scene with a recorder, a SceneRecorder or an output folder")
    units = {}
    if hasattr(outputs, "data_structures"):
        for _, _, components in outputs.data_structures().values():
            for component in components:
                table = type(component).declarations()
                for name, unit in zip(table.get("name", ()), table.get("unit", ())):
                    units.setdefault(name, unit)
    return recorder, units


def _model_value(recorder, row, cache) -> tuple:
    """(value, table name) of one observation's model value."""
    names = cache.setdefault("_names", recorder.table_names())
    if row["table"] is not None:
        candidates = [row["table"]]
    else:
        candidates = [name for name in names
                      if row["variable"] in cache.setdefault(name, recorder.table(name)).columns]
        if not candidates:
            raise KeyError(f"observation of '{row['variable']}': no recorded table has that column (tables: {names})")
        if len(candidates) > 1:
            raise KeyError(f"observation of '{row['variable']}': several tables have it ({candidates}), give table=")
    name = candidates[0]
    table = cache.setdefault(name, recorder.table(name))
    if row["variable"] not in table.columns:
        raise KeyError(f"observation of '{row['variable']}': table '{name}' has no such column")
    selected = table
    if row["plant"] is not None:
        if "plant" not in table.columns:
            raise KeyError(f"observation of '{row['variable']}' for plant {row['plant']}: '{name}' is not per plant")
        selected = selected[selected["plant"].astype(str) == str(row["plant"])]
    if row["time"] is not None:
        times = selected["time"].to_numpy(dtype=float)
        selected = selected[np.isclose(times, float(row["time"]), rtol=1e-12, atol=1e-9)]
    elif len(selected):
        selected = selected[selected["time"] == selected["time"].max()]
    if len(selected) != 1:
        raise KeyError(f"observation of '{row['variable']}' (plant {row['plant']}, time {row['time']}): "
                       f"{len(selected)} rows of '{name}' match, one expected")
    return float(selected[row["variable"]].iloc[0]), name


def compare(outputs, observations) -> pd.DataFrame:
    """
    The *observations* (read_observations' table) with their model values: the columns of the observations, then
    model (in the observation's unit when it gives one), source (the table read) and residual ((model - value) /
    uncertainty, or model - value). *outputs*: a Scene with its recorder, a SceneRecorder or an output folder, or
    {experiment: one of those} for observations of several experiments.
    """
    from openalea.metafspm.coupling.units import conversion_factor
    observations = read_observations(observations)
    sources = outputs if isinstance(outputs, dict) else {None: outputs}
    opened, caches = {}, {}
    models, tables = [], []
    for _, row in observations.iterrows():
        experiment = row["experiment"]
        if experiment not in sources:
            raise KeyError(f"observation of '{row['variable']}': no outputs for experiment {experiment!r} "
                           f"(given: {list(sources)})")
        if experiment not in opened:
            opened[experiment] = _recorder(sources[experiment])
        recorder, units = opened[experiment]
        value, name = _model_value(recorder, row, caches.setdefault(experiment, {}))
        if row["unit"] is not None:
            declared = units.get(row["variable"])
            if declared is None:
                raise ValueError(f"observation of '{row['variable']}' in {row['unit']}: the model's unit is unknown "
                                 "(compare a Scene, whose components declare it)")
            factor = conversion_factor(row["unit"], declared)
            if factor is None:
                raise ValueError(f"observation of '{row['variable']}': {row['unit']} is not a unit of the declared "
                                 f"{declared}")
            value *= factor
        models.append(value)
        tables.append(name)
    result = observations.copy()
    result["model"] = models
    result["source"] = tables
    uncertainty = result["uncertainty"].map(lambda u: 1. if u is None else float(u)).to_numpy(dtype=float)
    result["residual"] = (result["model"].to_numpy(dtype=float) - result["value"].to_numpy(dtype=float)) / uncertainty
    return result


def residuals(outputs, observations) -> np.ndarray:
    """compare(outputs, observations)'s residuals, one per observation."""
    return compare(outputs, observations)["residual"].to_numpy(dtype=float)


class Objective:
    """
    The residuals of *observations* as a function of the values of *parameters* (objective()): call it with a
    vector of values. Its limits are the parameters' declared bounds (limits: (lows, highs)), and bounds the same in
    its scale (for scipy.optimize.least_squares); history lists each evaluation (values, residuals, cost: half the sum of
    squares).
    """

    def __init__(self, run, parameters, observations, declared=None, scale=None):
        from openalea.metafspm.scene.scenarios import _declared
        self.run, self.parameters = run, list(parameters)
        self.observations = read_observations(observations)
        self.experiments = list(dict.fromkeys(self.observations["experiment"]))
        table = _declared(declared)
        lows, highs = [], []
        for name in self.parameters:
            low = high = None
            if table is not None and name in table.index:
                low, high = table.at[name, "min_value"], table.at[name, "max_value"]
            lows.append(-np.inf if low is None else float(low))
            highs.append(np.inf if high is None else float(high))
        self.limits = (np.array(lows), np.array(highs))
        self.scale = np.ones(len(self.parameters)) if scale is None else np.asarray(scale, dtype=float)
        self.bounds = (self.limits[0] / self.scale, self.limits[1] / self.scale)
        self.history = []

    def values(self, x) -> dict:
        """{parameter: value} of the vector *x* (in the objective's scale: x * scale)."""
        x = np.asarray(x, dtype=float).reshape(-1) * self.scale
        if x.size != len(self.parameters):
            raise ValueError(f"objective: {len(self.parameters)} values expected ({self.parameters}), got {x.size}")
        low, high = self.limits
        outside = [(name, value) for name, value, a, b in zip(self.parameters, x, low, high) if not a <= value <= b]
        if outside:
            raise ValueError(f"objective: {outside} outside the declared bounds")
        return dict(zip(self.parameters, x.tolist()))

    def __call__(self, x) -> np.ndarray:
        values = self.values(x)
        outputs = {experiment: self.run(values, experiment) for experiment in self.experiments}
        found = residuals(outputs, self.observations)
        self.history.append({"values": values, "residuals": found, "cost": 0.5 * float(found @ found)})
        return found


def objective(run, parameters, observations, declared=None, scale=None) -> Objective:
    """
    The calibration's objective: *run*(values, experiment) builds and runs one virtual experiment with the
    parameters' *values* ({name: value}; *experiment*: the observations' experiment, None when they give none) and
    returns its Scene (with a recorder) or its SceneRecorder; the objective returns the residuals of *observations*
    (compare). *declared*: the components (or a parameter_table) whose min_value / max_value bound the parameters;
    *scale*: one factor per parameter, the objective taking x = value / scale (parameters of very different
    magnitudes).
    """
    return Objective(run, parameters, observations, declared=declared, scale=scale)
