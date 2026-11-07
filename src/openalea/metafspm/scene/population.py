"""
A plant population in one MPG.

  table = planting([Planting(x, y, depth, rotation), ...], Model, scenario, plant_scenarios)  given plants
  table = planting(plants, [Roots] * 4, {"deep": {...}, "shallow": {...}},                    two populations of
                   population=["deep", "deep", "shallow", "shallow"])                         one model (intercrop)
  table = planting(stand(20, rows=4, plant_spacing=0.04, seed=1), Model, scenario)            a sown stand
  population = population_data_structure(table, Model, time_step=3600.)    without a Scene: the plants' MPG, its
  population.data_structure, population.instance                           DataStructure and the model built on it

The planting table has one row per plant (columns plant, model, population, x, y, depth, rotation, scenario,
optionally planting_time), its plants named p0, p1, ... in its order; a plant's z (its MPG property, the
planting_variable "z") is the soil surface's height minus its depth; the plants of one population name (by default their
model's class name) are one population, of one model: several populations may be of one model. Each StructuralComponent of `initiators` builds every
plant's initial structure (StructuralComponent.initiate_plant), in order, before any component is constructed;
components are then built once for the whole population, and their per-plant parameters filled from each plant's
scenario.

A scenario is flat, {name: value}: the parameters and settings of the model's components, set by the Scene after the
model is built as a constructor keyword would, the model's own arguments (the keys its __init__ names) and the keys
its initiators read. Any other entry raises.
"""
import inspect
from dataclasses import dataclass

import numpy as np
import pandas as pd

from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.data_structure.mpg import MPG


@dataclass
class Planting:
    """A plant to sow: its position (x, y, m), its depth (m, below the Scene's soil surface: its seed's, where its
    initiators start it; 0 at the surface) and its rotation (°, about the vertical)."""
    x: float = 0.
    y: float = 0.
    depth: float = 0.
    rotation: float = 0.


class Stand(list):
    """Plantings laid out by stand(): the list, with the stand's pattern (xrange, yrange, m: repeated by periodic
    models) and, when several models were drawn, the model of each position (models)."""

    def __init__(self, plantings, xrange: float, yrange: float, models: list = None):
        super().__init__(plantings)
        self.xrange, self.yrange, self.models = xrange, yrange, models


def planting(plants, model, scenario: dict = None, plant_scenarios: list = None, planting_times: list = None,
             xrange: float = None, yrange: float = None, population=None) -> pd.DataFrame:
    """
    The planting table of *plants* ([Planting], or a Stand), named p0, p1, ... in their order. *model*: the plant
    model of every plant, or one per plant (a list, e.g. a Stand's models). *population*: the name of every plant's
    population, or one per plant (by default its model's class name; without a dot): plants of one model in two
    populations are built as two (intercropping: two varieties, two parameter sets of one model). *scenario*:
    shared by every plant, or {model class or population name: scenario} with several; *plant_scenarios*: one dict
    per plant, added to the shared one.
    *planting_times*: one scene time per plant ("2 d", a date with the Scene's start, or s), when it enters the
    scene (its components run on it from then: the mask "planted"), the planting_time column read by the Scene. The
    stand's size
    (table.attrs xrange, yrange) is the Stand's, else the plants' extent (at least 0.1 m), unless given.
    """
    plants = list(plants) if not isinstance(plants, Stand) else plants
    models = list(model) if isinstance(model, (list, tuple)) else [model] * len(plants)
    names = list(population) if isinstance(population, (list, tuple)) else \
        [population if population is not None else getattr(m, "__name__", str(m)) for m in models]
    own = plant_scenarios if plant_scenarios is not None else [{}] * len(plants)
    for name, values in (("model", models), ("population", names), ("plant_scenarios", own),
                         ("planting_times", planting_times)):
        if values is not None and len(values) != len(plants):
            raise ValueError(f"planting: {len(values)} {name} for {len(plants)} plants")
    for name in set(names):
        if not isinstance(name, str) or not name or "." in name:
            raise ValueError(f"planting: population name {name!r}: a text without a dot")
    by_owner = (scenario is not None and len(scenario) > 0 and all(isinstance(key, type) or key in names
                                                                   for key in scenario)
                and all(isinstance(value, dict) for value in scenario.values()))
    rows = []
    for k, (plant, plant_model, name) in enumerate(zip(plants, models, names)):
        shared = {**(scenario or {}).get(plant_model, {}), **(scenario or {}).get(name, {})} if by_owner \
            else (scenario or {})
        rows.append(dict(plant=f"p{k}", model=plant_model, population=name, x=float(plant.x), y=float(plant.y),
                         depth=float(plant.depth), rotation=float(plant.rotation),
                         scenario=_flat_scenario(dict(shared, **(own[k] or {})))))
    table = pd.DataFrame(rows, columns=["plant", "model", "population", "x", "y", "depth", "rotation", "scenario"])
    if planting_times is not None:
        from openalea.metafspm.coupling.durations import is_date, duration
        times = [time if is_date(time) else duration(time, "planting_times") for time in planting_times]
        table["planting_time"] = pd.Series(times, dtype=object if any(map(is_date, times)) else float)
    xs, ys = table["x"].to_numpy(), table["y"].to_numpy()
    if xrange is None:
        xrange = plants.xrange if isinstance(plants, Stand) else max(float(np.ptp(xs)) if len(xs) else 0., 0.1)
    if yrange is None:
        yrange = plants.yrange if isinstance(plants, Stand) else max(float(np.ptp(ys)) if len(ys) else 0., 0.1)
    table.attrs.update(xrange=xrange, yrange=yrange)
    return table


def stand(plants: int = None, rows: int = 1, row_spacing: float = 0.15, plant_spacing: float = None,
          sowing_density: float = None, depth=0.025, jitter: float = 0., rotation="random", models: list = None,
          frequencies: list = None, seed: int = None) -> Stand:
    """
    A sown stand of *plants* in *rows* (metafspm's one stand builder and its axis convention): the rows along x,
    *row_spacing* apart in y, the plants *plant_spacing* apart along their row (m; or, from *sowing_density*,
    plants m-2: 1 / (sowing_density × row_spacing)), filled row by row: plant k of row r at
    x = (k + 1/2) plant_spacing, y = (r + 1/2) row_spacing, the stand's pattern (xrange, yrange) the rows' extent
    (the plants per row × plant_spacing, rows × row_spacing). Each plant at *depth* (m below the soil surface; one per
    model with several), moved off its spot by *jitter* (m: a normal offset in x and in y, of that standard
    deviation), its *rotation* "random" (uniform over a turn), or a number (°: each plant turned by it from the one
    before, the first at 0). With *models*, the model of each plant is drawn from *frequencies* (equal by default) and
    kept in the Stand's models. *seed*: the draws' (numpy's default_rng), the same stand again.
    """
    if plants is None or plants < 1:
        raise ValueError(f"stand: plants is a number of plants, at least 1, got {plants!r}")
    if (plant_spacing is None) == (sowing_density is None):
        raise ValueError("stand: give plant_spacing (m) or sowing_density (plants m-2), one of them")
    if plant_spacing is None:
        plant_spacing = 1. / (float(sowing_density) * float(row_spacing))
    models = list(models) if models is not None else None
    count = len(models) if models is not None else 1
    frequencies = np.asarray(frequencies if frequencies is not None else [1. / count] * count, dtype=float)
    depths = list(np.atleast_1d(depth))
    depths = depths * count if len(depths) == 1 else depths
    if len(depths) != count:
        raise ValueError(f"stand: {len(depths)} depths for {count} models")
    generator = np.random.default_rng(seed)
    per_row = int(np.ceil(plants / rows))
    plantings, drawn = [], []
    for k in range(plants):
        row, position = divmod(k, per_row)
        index = 0 if count == 1 else int(np.searchsorted(np.cumsum(frequencies) / frequencies.sum(),
                                                         generator.random(), side="right").clip(0, count - 1))
        offset = generator.normal(0., jitter, 2) if jitter > 0 else np.zeros(2)
        turn = generator.uniform(0., 360.) if rotation == "random" else float(rotation) * k
        plantings.append(Planting(x=(position + 0.5) * plant_spacing + float(offset[0]),
                                  y=(row + 0.5) * row_spacing + float(offset[1]), depth=float(depths[index]),
                                  rotation=float(turn)))
        drawn.append(index)
    return Stand(plantings, per_row * plant_spacing, rows * row_spacing,
                 [models[i] for i in drawn] if models is not None else None)


class Population:
    """The plants of one population, of one model: their name (the planting's population, by default the model's
    class name), planting rows, Plant vertices, MPGDataStructure and model instance."""

    def __init__(self, model, table: pd.DataFrame, plants: list, data_structure, instance,
                 read_by_initiators=frozenset(), name: str = None):
        self.model, self.table, self.plants = model, table, list(plants)
        self.name = name if name is not None else model.__name__
        self.data_structure, self.instance = data_structure, instance
        self.planted_later = "planting_time" in table.columns      # the plants enter the scene at their times
        self.read_by_initiators = frozenset(read_by_initiators)   # the scenario keys its initiators read

    @property
    def components(self) -> list:
        return list(self.instance.components)

    def plant_names(self) -> dict:
        """{Plant vid: name in the planting table}."""
        return {int(vid): str(name) for vid, name in zip(self.plants, self.table["plant"])}


def population_data_structure(table: pd.DataFrame, model, time_step: float = 3600., scenario: dict = None,
                              soil_surface: float = 0., name: str = None) -> Population:
    """
    The plants of *table* (a planting table's rows of *model*) in one MPG and its MPGDataStructure, *model* built on
    it, every plant's scenario applied: what the Scene builds for each of its models, for a model run without one.
    The model's initiators build each plant's structure (on the nodes of its from_scale, or its own nodes for
    anatomies), its wiring joins them; *scenario*: the one the model is built with (default: the first plant's);
    *soil_surface*: the soil surface's height (m, the Scene's), the MPG's g.soil_surface while the initiators build
    (a crown placed below it) and the DataStructure's after (ds.soil_surface); *name*: the population's (default:
    the table's population column, else the model's class name).
    """
    table = table.reset_index(drop=True)
    if name is None:
        name = str(table["population"].iloc[0]) if "population" in table.columns and len(table) else model.__name__
    scenario = _flat_scenario(table["scenario"].iloc[0] if scenario is None and len(table) else scenario)
    read_by_initiators = set()
    reduction = _model_reduction(model, table, scenario, read_by_initiators)
    from openalea.metafspm.coupling.composite_wrapper import model_initiators, model_translator
    g, plants = _build_population(table, initiators=model_initiators(model),
                                  read_by_initiators=read_by_initiators, reduction=reduction,
                                  translator=model_translator(model), soil_surface=soil_surface)
    scale = getattr(model, "from_scale", None) or "SubOrgan"
    scale = getattr(g.scales, scale) if isinstance(scale, str) else scale
    nodes = getattr(model, "nodes", None)
    if nodes is None:
        g.populate_graph(scale)                 # anatomy mode: the initiators built the Compartments
    g.convert_properties_to_arraydict()
    wiring = getattr(model, "wiring", None)
    wiring = wiring(g) if callable(wiring) else wiring
    ds = MPGDataStructure(g, from_scale=scale, nodes=nodes, wiring=wiring)
    ds.soil_surface = soil_surface
    ds.simulation_time_step = time_step               # the model's components run in it (the Scene's step)
    instance = model(data_structure=ds, time_step=time_step, **_model_keywords(model, scenario))
    if getattr(g, "reduction", None) is not None:
        g.reduction.bind(model, instance)
    population = Population(model, table, plants, ds, instance, read_by_initiators, name=name)
    _apply_model_scenario(model, instance, scenario, read_by_initiators)
    _apply_plant_scenarios(ds, population.components, table, plants)
    return population


def _model_reduction(model, table: pd.DataFrame, scenario: dict, read: set):
    """
    The model's reduction (solve.reduction): its class attribute, a Reduction, or a function of the model's scenario
    (``reduction = staticmethod(lambda scenario: ...)``: a reduction chosen by scenario entries, or None). The entries
    the function reads are the model's (added to *read*); they must be the same for every plant of the population.
    The function's attribute `scenario_keys` names the entries it owns, the model's whether it reads them or not (a
    reduction's labels given while the reduction is off: inactive).
    """
    from openalea.metafspm.solve.reduction import Reduction
    reduction = getattr(model, "reduction", None)
    if reduction is None or isinstance(reduction, Reduction) or not callable(reduction):
        return reduction
    recorded = _ReadScenario(scenario)
    chosen = reduction(recorded)
    recorded.read |= {key for key in getattr(reduction, "scenario_keys", ()) if key in scenario}
    names = getattr(model, "__name__", str(model))
    for row in table.itertuples(index=False):
        own = _flat_scenario(row.scenario)
        for key in recorded.read:
            if not _equal(own.get(key), scenario.get(key)):
                raise ValueError(f"{names}: the scenario entry '{key}' chooses the model's reduction, the same for "
                                 f"every plant of a population (one plant has {own.get(key)!r}, another "
                                 f"{scenario.get(key)!r})")
    read |= recorded.read
    return chosen


def _equal(a, b) -> bool:
    try:
        return bool(a == b) if not isinstance(a, np.ndarray) else bool(np.array_equal(a, b))
    except (TypeError, ValueError):
        return repr(a) == repr(b)


def _flat_scenario(scenario) -> dict:
    """*scenario* as given ({name: value}), its dotted entries folded into their sectioned parameter
    ({"root_anatomy.cortex.n_layers": 4} into root_anatomy's {"cortex": {"n_layers": 4}}, over a nested value given
    too); the former {"parameters": {...}} nesting raises."""
    scenario = dict(scenario or {})
    if "parameters" in scenario:
        raise ValueError("a scenario is flat, {name: value}: write the entries of its 'parameters' dict at its top "
                         "level")
    for key in [key for key in scenario if isinstance(key, str) and "." in key and key != "forcings"]:
        name, *path = key.split(".")
        target = scenario.get(name)
        target = {} if target is None else _nested_copy(target)
        node = target
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = scenario.pop(key)
        scenario[name] = target
    return scenario


def _nested_copy(value):
    """A copy of nested dicts (a sectioned parameter's value), so that folding leaves the given one as it was."""
    return {k: _nested_copy(v) for k, v in value.items()} if isinstance(value, dict) else value


def _build_population(table: pd.DataFrame, initiators=(), read_by_initiators: set = None, reduction=None,
                      translator=None, soil_surface: float = 0.) -> tuple:
    """
    A new MPG with one Plant-scale vertex per row of *table* (Plant properties x, y, rotation and z, the soil
    surface's height *soil_surface* minus the row's depth), each plant's
    initial structure built by every StructuralComponent class of *initiators*, in order. Returns (g, plant vids in
    table order). The scenario keys the initiators read are added to *read_by_initiators* when given. *reduction*
    (the model's, solve.reduction): the MPG's, before the initiators build (they ask it for the reduced templates).
    *translator* (the model's class attribute): its links between initiators applied before each initiator builds a
    plant (coupling.initiation).
    """
    from openalea.metafspm.coupling.initiation import apply_link, initiation_links
    links = initiation_links(translator, initiators)
    g = MPG()
    g.soil_surface = float(soil_surface)          # the soil surface's height, for initiators placing below it
    if reduction is not None:
        g.reduction = reduction.for_population(g)
    plants = []
    for row in table.itertuples(index=False):
        plant = g.add_component(g.root, **PropsConfig(scale=g.scales.Plant, edge_type='/'))
        for name in ("x", "y", "rotation"):
            g.property(name)[plant] = float(getattr(row, name))
        g.property("z")[plant] = float(soil_surface) - float(row.depth)
        plants.append(plant)
    from openalea.metafspm.data_structure.mpg import AnchorError
    for k, initiator in enumerate(initiators):
        for plant, row in zip(plants, table.itertuples(index=False)):
            scenario = _ReadScenario(_flat_scenario(row.scenario))
            for link, provider in links.get(k, ()):     # its inputs linked from earlier initiators
                apply_link(g, plant, link, initiator, provider)
            try:
                initiator.initiate_plant(g, plant, scenario)
            except AnchorError as error:            # the anchors' contract: an earlier initiator builds the anchor
                order = [getattr(i, "__name__", str(i)) for i in initiators]
                raise AnchorError(f"{order[k]}: {error}. The initiators build in order, {order}: its anchor must be "
                                  f"built by one before it ({order[:k] or 'none'})") from None
            if read_by_initiators is not None:
                read_by_initiators |= scenario.read
    return g, plants


class _ReadScenario(dict):
    """A plant's scenario, recording the keys an initiator reads."""

    def __init__(self, *args):
        super().__init__(*args)
        self.read = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __contains__(self, key):
        self.read.add(key)
        return super().__contains__(key)


def _apply_plant_scenarios(ds, components, table: pd.DataFrame, plants: list) -> None:
    """
    Fill the per-plant parameters of *components* (stored at the "Plant" location) from each plant's scenario; a
    parameter a scenario does not give keeps the component's value.
    """
    by_plant = {int(plant): _flat_scenario(row.scenario) for plant, row in zip(plants, table.itertuples(index=False))}
    entities = ds.entity_ids("Plant")
    missing = set(int(e) for e in entities) - set(by_plant)
    if missing:
        raise ValueError(f"Plant vertices {sorted(missing)[:5]} have no row in the planting table")
    given = set().union(*by_plant.values()) if by_plant else set()
    for component in components:
        settings = component.declared_settings() if hasattr(component, "declared_settings") else ()
        for name in given.intersection(settings):     # a setting: one value for every plant
            if not _same_for_all([by_plant[p].get(name, getattr(component, name)) for p in by_plant]):
                raise ValueError(f"{type(component).__name__}.{name} is a setting, one value for the model: give "
                                 "every plant the same")
        for name, spec in getattr(component, "_variable_specs", {}).items():
            if spec.variable_type != "parameter":
                continue
            if ds.location(name) != "Plant":
                if name in given and not _same_for_all([by_plant[p].get(name) for p in by_plant]):
                    raise ValueError(f"{type(component).__name__}.{name} differs between the plants' scenarios but "
                                     f"is stored at '{ds.location(name)}', not per plant: give every plant the same "
                                     "value")
                continue
            stored = np.asarray(ds.get(name))
            current = stored if stored.dtype == object else stored.astype(float)
            values = np.empty(len(entities), dtype=object) if stored.dtype == object else [None] * len(entities)
            for i, e in enumerate(entities):             # a record per plant (dtype object) taken as it is
                values[i] = by_plant[int(e)].get(name, current[i])
            ds.set(name, values)


def _apply_model_scenario(model, instance, scenario: dict, read_by_initiators=(), per_plant: bool = True) -> None:
    """
    Set *scenario*'s entries on the components of *instance* (built from the class *model*), as constructor keywords
    would: a numeric parameter is written to the DataStructure, another goes to the component's field. The entries are
    the scenario's keys, except those *model*'s __init__ names (its own arguments). Parameters
    stored per plant are left to _apply_plant_scenarios (a plant without the entry keeps the default), or, with
    *per_plant* False (an environment model's scenario: a light model's optics on the plants' DataStructure), set for
    every plant. The entry "cadence" ({step name: duration, Cadence or None}) sets the model's steps' cadences
    (CompositeModel.set_cadences). An entry that is no component parameter, no argument of the model and not read by
    its initiators raises.
    """
    named = _model_arguments(model)
    entries = {key: value for key, value in _flat_scenario(scenario).items() if key not in named}
    if "cadence" in entries:                        # the model's steps' cadences (coupling.schedule), by step name
        if not hasattr(instance, "set_cadences"):
            raise ValueError(f"{model.__name__}: its scenario's cadence needs a model with a schedule (CompositeModel)")
        instance.set_cadences(entries.pop("cadence"))
    components = list(getattr(instance, "components", ()))
    unknown = []
    for name, value in entries.items():
        owners = [c for c in components if name in getattr(c, "declared_parameters", tuple)()
                  or name in (c.declared_settings() if hasattr(c, "declared_settings") else ())]
        if not owners:
            if name not in named and name not in read_by_initiators:
                unknown.append(name)
            continue
        for component in owners:
            ds = getattr(component, "data_structure", None)
            if ds is not None and ds.has(name) and ds.location(name) == "Plant":
                if not per_plant:                           # every plant the environment's value
                    ds.set(name, np.full(np.shape(ds.get(name)), value, dtype=np.asarray(ds.get(name)).dtype))
                continue                                    # per plant: _apply_plant_scenarios
            setattr(component, name, value)
    for component in components:                          # a setting's choices, checked as at construction
        if hasattr(component, "check_settings"):
            component.check_settings()
    if unknown:
        known = sorted({name for c in components
                        for name in (*getattr(c, "declared_parameters", tuple)(),
                                     *(c.declared_settings() if hasattr(c, "declared_settings") else ()))})
        raise ValueError(f"{model.__name__}: scenario entries {sorted(unknown)} are neither parameters of its "
                         f"components ({', '.join(known) or 'none'}), arguments of {model.__name__}, nor read by its "
                         "initiators")


def _model_arguments(model) -> set:
    """The keyword arguments *model*'s __init__ names (its own scenario keys)."""
    signature = inspect.signature(model.__init__)
    return {name for name, p in signature.parameters.items()
            if name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}


def _model_keywords(model, scenario: dict) -> dict:
    """The scenario keys *model* is built with: those its __init__ names (its own arguments). The other entries are
    the components' parameters and settings, set once the model is built (_apply_model_scenario)."""
    scenario = dict(scenario or {})
    named = _model_arguments(model)
    return {key: value for key, value in scenario.items() if key in named}


def _same_for_all(values) -> bool:
    first = values[0]
    return all((v is None) == (first is None) and (v is None or bool(np.all(np.asarray(v) == np.asarray(first))))
               for v in values)

