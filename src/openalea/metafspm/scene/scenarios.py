"""
Scenarios: reading them from a table, listing a model's parameters, building variants of one::

  scenarios = read_scenarios("scenarios.csv")          {name: scenario}, one per column of the table
  parameter_table(RootGrowth, Carbon, main=("EL",))    every parameter: value, unit, description, bounds, component
  variants(scenarios["reference"], {"EL": None, "IPD": [0.005, 0.01]}, design="grid", declared=[RootGrowth])

A scenario is flat, {name: value} (scene.population). The table has one row per entry: its name in a "name"
column (else the first column), optional description columns (model, type, unit, description, references,
default, explanation), and one column per named scenario::

  name,   type,      unit, description,               reference, dry
  EL,     parameter, m s-1, elongation rate,          1e-3,      8e-4
  IPD,    parameter, m,    inter-primordia distance,   6e-3,
  rain,   forcing,   mm,   the rainfall, by time (s),  rain.csv,  0.

A cell holds a number, True, False, None, a JSON list or dict, or text; an empty cell leaves the default (the entry
is not in that scenario). The rows of type "forcing" hold a number or a CSV file (relative to the table; its
columns time, in s, and the name): they go to the scenario's "forcings" entry, {name: forcing}, for
Scene(forcings=...) (pop it before planting). With a "model" column, each scenario is {model: {name: value}} (and
its "forcings": a forcing row needs no model, forcings being the scene's).

variants() builds scenarios around a base one, within the bounds the components declare (min_value, max_value):
a grid of every combination, one entry changed at a time, or a Latin hypercube sampling; from_samples() takes the
samples of an analysis tool (e.g. SALib's saltelli or morris) within the same bounds, and design_table() lays any of
them out as a table, to join with the outputs. Sensitivity indices are left to that tool.

run_variants(run, scenarios, workers="auto", compute=Compute(workers=1)) runs a function building and running
one scene per scenario, in a process pool (the total of processes bounded: variant_processes).
"""
import itertools
import json
import os

import numpy as np
import pandas as pd

DESCRIPTION_COLUMNS = ("model", "type", "unit", "description", "references", "default", "explanation")


def _cell(value):
    """A table cell as a scenario value (None when empty: the entry is not given)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    special = {"True": True, "False": False, "None": "None"}
    if text in special:
        return special[text]
    if text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() and "." not in text and "e" not in text.lower() else number


def _forcing(value, folder: str, name: str):
    """A forcing row's cell: a number, or a CSV file (columns time, in s, and *name*) as a Series indexed by time."""
    if not isinstance(value, str):
        return value
    path = value if os.path.isabs(value) else os.path.join(folder, value)
    table = pd.read_csv(path)
    column = name if name in table.columns else [c for c in table.columns if c != "time"][0]
    return pd.Series(table[column].to_numpy(dtype=float), index=table["time"].to_numpy(dtype=float), name=name)


def read_scenarios(source, which=None, sheet=None) -> dict:
    """
    {name: scenario} from a scenario table (a CSV or Excel file, or a DataFrame), one per scenario column, or those
    named by *which*. *sheet*: the Excel sheet (the first by default). See the module for the table's layout.
    """
    folder = "."
    if isinstance(source, pd.DataFrame):
        table = source.copy()
    else:
        path = os.fspath(source)
        folder = os.path.dirname(os.path.abspath(path))
        if path.lower().endswith((".xlsx", ".xls", ".ods")):
            table = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0, dtype=object,
                                  keep_default_na=False, na_values=[""])
        else:                                       # "None", "NA" are values, only empty cells are missing
            table = pd.read_csv(path, dtype=object, skipinitialspace=True, keep_default_na=False, na_values=[""])
    columns = {str(column).strip(): column for column in table.columns}
    table.columns = list(columns)
    name_column = "name" if "name" in table.columns else table.columns[0]
    lower = {column.lower(): column for column in table.columns}
    described = {lower[c] for c in DESCRIPTION_COLUMNS if c in lower} | {name_column}
    names = [column for column in table.columns if column not in described]
    if which is not None:
        missing = [name for name in which if name not in names]
        if missing:
            raise KeyError(f"scenarios {missing} not in the table (its scenarios: {names})")
        names = list(which)
    model_column, type_column = lower.get("model"), lower.get("type")
    if model_column is not None:                       # every entry a model's, but the forcings (the scene's)
        given = table[model_column].map(_cell)
        kinds = (table[type_column].map(lambda v: str(_cell(v) or "").strip().lower()) if type_column is not None
                 else pd.Series("", index=table.index))
        missing = given.isna() & (kinds != "forcing")
        if missing.any() and given.notna().any():
            rows = table.loc[missing, name_column].tolist()
            raise ValueError(f"entries {rows} have no model, in a table with a model column")
    scenarios = {}
    for scenario_name in names:
        scenario, forcings = {}, {}
        for _, row in table.iterrows():
            entry = str(row[name_column]).strip()
            value = _cell(row[scenario_name])
            if value is None:
                continue
            value = None if value == "None" else value
            kind = str(_cell(row[type_column]) or "").strip().lower() if type_column is not None else ""
            if kind == "forcing":
                forcings[entry] = _forcing(value, folder, entry)
                continue
            if model_column is not None:
                scenario.setdefault(str(_cell(row[model_column])), {})[entry] = value
            else:
                scenario[entry] = value
        if forcings:
            scenario["forcings"] = forcings
        scenarios[scenario_name] = scenario
    return scenarios


def parameter_table(*components, values: dict = None, main=(), roles=("parameter", "setting"),
                    exclude=()) -> pd.DataFrame:
    """
    The parameters (and settings) the *components* (classes or instances) declare, one row each: name, value (the
    default, or the one in *values*), unit, description, min_value, max_value, component, and main (whether it is
    in *main*: those first, in its order). An entry declared by several components takes one row, the last one's.
    *exclude*: names left out.
    """
    tables = []
    for component in components:
        component = component if isinstance(component, type) else type(component)
        declarations = component.declarations()
        if declarations.empty:
            continue
        table = declarations[declarations["role"].isin(roles)][["name", "default", "unit", "description",
                                                                 "min_value", "max_value"]]
        tables.append(table.assign(component=component.__name__))
    columns = ["name", "value", "unit", "description", "min_value", "max_value", "component", "main"]
    if not tables:
        return pd.DataFrame(columns=columns)
    table = pd.concat(tables).drop_duplicates("name", keep="last").rename(columns={"default": "value"})
    table = table[~table["name"].isin(exclude)].reset_index(drop=True)
    table["value"] = table["value"].astype(object)
    for name, value in (values or {}).items():
        if not (table["name"] == name).any():
            raise KeyError(f"'{name}' is no parameter of {', '.join(sorted(set(table['component'])))}")
        table.loc[table["name"] == name, "value"] = pd.Series([value] * int((table["name"] == name).sum()),
                                                              index=table.index[table["name"] == name], dtype=object)
    table["main"] = table["name"].isin(main)
    order = {name: rank for rank, name in enumerate(main)}
    table = table.assign(rank=table["name"].map(order).fillna(len(order))).sort_values("rank", kind="stable")
    return table.drop(columns="rank").reset_index(drop=True)[columns]


def _declared(declared) -> pd.DataFrame:
    if declared is None:
        return None
    if isinstance(declared, pd.DataFrame):
        return declared.drop_duplicates("name", keep="last").set_index("name")
    components = declared if isinstance(declared, (list, tuple)) else [declared]
    return parameter_table(*components).set_index("name")


def _label(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def variants(base: dict, ranges: dict, design: str = "grid", levels: int = 3, samples: int = 10, seed: int = None,
             declared=None) -> dict:
    """
    {name: scenario}: *base* (a flat scenario) with the entries of *ranges* changed. *ranges*: {name: [values]},
    {name: (low, high)}, or {name: None} for its declared bounds (min_value, max_value of *declared*: component
    classes, or a parameter_table). *design*::

      "grid"            every combination, a (low, high) range in *levels* even steps: "EL=0.0008,IPD=0.006", ...
      "one_at_a_time"   "base", then each value of each entry alone: "EL=0.0008", ...
      "sampling"        *samples* scenarios of a Latin hypercube over the ranges (a list: its values drawn evenly),
                        "sample_000", ... (*seed*: the same draws again)

    A value outside its declared bounds raises; an entry whose base or default value is an integer stays one.
    """
    table = _declared(declared)
    base = dict(base or {})
    bounds, integer = {}, {}
    for name in ranges:
        low = high = None
        if table is not None and name in table.index:
            low, high = table.at[name, "min_value"], table.at[name, "max_value"]
            default = table.at[name, "value"]
        else:
            default = None
        reference = base.get(name, default)
        integer[name] = isinstance(reference, (int, np.integer)) and not isinstance(reference, bool)
        bounds[name] = (low, high)

    def checked(name, value):
        low, high = bounds[name]
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            if (low is not None and value < low) or (high is not None and value > high):
                raise ValueError(f"variants: {name}={value} outside its declared bounds [{low}, {high}]")
        return value

    def spread(name, low, high, fractions):
        values = [low + (high - low) * f for f in fractions]
        return [int(round(v)) if integer[name] else float(v) for v in values]

    axes = {}
    for name, given in ranges.items():
        if given is None:
            low, high = bounds[name]
            if low is None or high is None:
                raise ValueError(f"variants: '{name}' has no declared bounds (min_value, max_value): give its range")
            given = (low, high)
        if isinstance(given, tuple):
            if len(given) != 2:
                raise ValueError(f"variants: the range of '{name}' is (low, high), got {given}")
            axes[name] = ("range", float(given[0]), float(given[1]))
        else:
            axes[name] = ("values", list(given))
        if axes[name][0] == "range":
            checked(name, axes[name][1]), checked(name, axes[name][2])
        else:
            for value in axes[name][1]:
                checked(name, value)

    def values_of(name):
        axis = axes[name]
        if axis[0] == "values":
            return axis[1]
        return spread(name, axis[1], axis[2], np.linspace(0., 1., levels) if levels > 1 else [0.5])

    found = {}
    if design == "grid":
        names = list(axes)
        for combination in itertools.product(*(values_of(name) for name in names)):
            label = ",".join(f"{name}={_label(value)}" for name, value in zip(names, combination))
            found[label] = dict(base, **dict(zip(names, combination)))
    elif design == "one_at_a_time":
        found["base"] = dict(base)
        for name in axes:
            for value in values_of(name):
                found[f"{name}={_label(value)}"] = dict(base, **{name: value})
    elif design == "sampling":
        generator = np.random.default_rng(seed)
        draws = {}
        for name, axis in axes.items():
            fractions = (generator.permutation(samples) + generator.random(samples)) / samples
            if axis[0] == "values":
                draws[name] = [axis[1][min(int(f * len(axis[1])), len(axis[1]) - 1)] for f in fractions]
            else:
                draws[name] = spread(name, axis[1], axis[2], fractions)
        for k in range(samples):
            found[f"sample_{k:03d}"] = dict(base, **{name: draws[name][k] for name in axes})
    else:
        raise ValueError(f"variants(design=...): 'grid', 'one_at_a_time' or 'sampling', got '{design}'")
    return found


def from_samples(base: dict, names, matrix, declared=None) -> dict:
    """
    {"sample_000": scenario, ...}: *base* (a flat scenario) with *names* set from each row of *matrix* (one row per
    sample, one column per name), e.g. a SALib sample (saltelli, morris) to run and analyse with SALib. A value
    outside the bounds the *declared* components (or a parameter_table) give raises; an integer entry stays one.
    """
    names = list(names)
    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    if matrix.shape[1] != len(names):
        raise ValueError(f"from_samples: {len(names)} names for {matrix.shape[1]} columns")
    table = _declared(declared)
    base = dict(base or {})
    found = {}
    width = max(3, len(str(len(matrix) - 1)))
    for k, row in enumerate(matrix):
        scenario = dict(base)
        for name, value in zip(names, row):
            if table is not None and name in table.index:
                low, high = table.at[name, "min_value"], table.at[name, "max_value"]
                if (low is not None and value < low) or (high is not None and value > high):
                    raise ValueError(f"from_samples: {name}={value} (sample {k}) outside its declared bounds "
                                     f"[{low}, {high}]")
            reference = base.get(name, table.at[name, "value"] if table is not None and name in table.index else None)
            integer = isinstance(reference, (int, np.integer)) and not isinstance(reference, bool)
            scenario[name] = int(round(value)) if integer else float(value)
        found[f"sample_{k:0{width}d}"] = scenario
    return found


def design_table(scenarios: dict, names=None) -> pd.DataFrame:
    """The design of *scenarios* ({name: flat scenario}: variants' or from_samples'), one row per scenario (indexed by
    its name), one column per entry of *names* (default: the entries that differ between them): to join with the
    runs' outputs for an analysis."""
    names = list(names) if names is not None else [
        key for key in dict.fromkeys(k for scenario in scenarios.values() for k in scenario)
        if len({repr(scenario.get(key)) for scenario in scenarios.values()}) > 1]
    rows = [{name: scenario.get(name) for name in names} for scenario in scenarios.values()]
    return pd.DataFrame(rows, index=pd.Index(list(scenarios), name="scenario"), columns=names)


def variant_processes(workers, compute=None, variants: int = None) -> tuple:
    """
    (processes, compute): how run_variants runs *variants* scenes at once. *workers*: the variants' processes, an
    integer or "auto" (processors − 1, at most one per variant); *compute*: each scene's Compute (default
    Compute(workers=1)). With more than one process, a scene's workers="auto" becomes 1 (the variants are the
    parallel work: no pool in each); an integer is kept when processes × that many (a scene's pool, its main
    process waiting on it) and processes × its threads stay within the processors, else ValueError (oversubscribed).
    """
    from dataclasses import replace
    from openalea.metafspm.compute import Compute
    processors = os.cpu_count() or 1
    compute = Compute(workers=1) if compute is None else compute
    if workers == "auto":
        workers = max(1, processors - 1)
        if variants is not None:
            workers = min(workers, max(1, variants))
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"run_variants: workers is 'auto' or a positive integer, got {workers!r}")
    if workers > 1 and compute.workers == "auto":
        compute = replace(compute, workers=1)
    per_scene = 1 if compute.workers == "auto" else int(compute.workers)
    if workers > 1 and (workers * per_scene > processors or workers * compute.threads > processors):
        raise ValueError(f"run_variants: {workers} variant processes × Compute(workers={compute.workers}, "
                         f"threads={compute.threads}) oversubscribe the {processors} processors: lower one of them")
    return workers, compute


def run_variants(run, scenarios: dict, workers="auto", compute=None, start_method: str = None) -> dict:
    """
    {name: run(name, scenario, compute)} for each of *scenarios* ({name: scenario}: variants', from_samples'), in
    their order. *run* is a module-level function (picklable by name) that builds its scene with
    Scene(..., compute=compute), runs it, finishes it and returns what is kept (a table, a path: picklable; a
    scene's recorder folder named after *name*).

    The variants run in *workers* processes at once (variant_processes: "auto" processors − 1; each scene's Compute
    *compute*, default Compute(workers=1), a scene's "auto" pool replaced by 1 when several run at once: the total
    of processes stays within the processors), each in a fresh process (its memory given back when it ends), forked
    on Linux and spawned elsewhere or with device="cuda" (each its own device context; *start_method* to choose).
    workers=1: one after the other in this process. A failing variant raises (its name in the message), the others
    not yet started not run, the running ones stopped.
    """
    import multiprocessing
    from multiprocessing.connection import wait
    from openalea.metafspm.compute import START_METHOD
    names = list(scenarios)
    workers, compute = variant_processes(workers, compute, len(names))
    if workers == 1 or len(names) < 2:
        return {name: run(name, scenarios[name], compute) for name in names}
    if start_method is None:
        start_method = "spawn" if compute.device == "cuda" else START_METHOD
    context = multiprocessing.get_context(start_method)
    pending, running, results = list(names), {}, {}
    try:
        while pending or running:
            while pending and len(running) < workers:     # a fresh process per variant (not a daemon: its scene
                name = pending.pop(0)                     # may start its own pool)
                ours, theirs = context.Pipe(duplex=False)
                process = context.Process(target=_variant_main, args=(theirs, run, name, scenarios[name], compute),
                                          name=f"metafspm-variant-{name}")
                process.start()
                theirs.close()
                running[ours] = (name, process)
            for connection in wait(list(running)):
                name, process = running.pop(connection)
                try:
                    status, value = connection.recv()
                except EOFError:
                    process.join()
                    status, value = "error", f"its process ended (exit code {process.exitcode}: killed, e.g. out " \
                                             "of memory?)"
                connection.close()
                process.join()
                if status == "error":
                    raise RuntimeError(f"run_variants: the variant {name!r} failed:\n{value}")
                results[name] = value
    finally:
        for connection, (_, process) in running.items():
            process.terminate()
            process.join()
            connection.close()
    return {name: results[name] for name in names}


def _variant_main(connection, run, name, scenario, compute) -> None:
    """A variant's process: run's result (or its traceback) sent back."""
    import traceback
    try:
        value = ("ok", run(name, scenario, compute))
    except BaseException:      # noqa: BLE001 — every failure goes back to the main process
        value = ("error", traceback.format_exc())
    connection.send(value)
    connection.close()
