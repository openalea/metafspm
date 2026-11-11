from contextlib import nullcontext
from dataclasses import dataclass, field, fields, MISSING
from typing import Literal, Optional
from time import perf_counter
import numpy as np

import types
import warnings

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.solve import conservation
from openalea.metafspm.solve.decorator import graph_system, postsegmentation, potential, rate, segmentation, state
from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.data_structure.data_api import (DataStructure, EntityView, GraphDataStructure, GraphView,
                                                      MPGDataStructure)
from openalea.metafspm.data_structure.configs import ScalesConfig as _ScalesConfig
from openalea.metafspm.coupling.declaration import _declared_specs, DeclarationError, held_at_scale, _holder_spec
from openalea.metafspm.coupling.declaration import scale_names as declared_scale_names

# Map integer MPG scale constants to the generic "node"/"edge" vocabulary used
# by GraphDataStructure.  Compartment nodes are graph nodes; Connection edges
# are graph edges.
_SCALE_INT_TO_LOC: dict = {
    _ScalesConfig.Compartment: "node",
    _ScalesConfig.Connection: "edge",
}


DECLARATION_COLUMNS = ["name", "role", "kind", "scale", "unit", "description", "default", "by", "min_value",
                       "max_value", "references", "DOI", "unit_comment", "value_comment", "upscaling"]
UPSCALING_LAWS = ("sum", "mean", "weighted_mean", "series", "network")


@dataclass(frozen=True)
class Upscaling:
    """What a coarse-scale state variable is the equivalent of, and how: see upscaling()."""
    of: tuple
    law: str
    protocol: str = ""
    references: str = ""

    def __str__(self) -> str:
        text = f"{self.law} of {', '.join(self.of)}"
        return f"{text}: {self.protocol}" if self.protocol else text


def upscaling(of, law: str = "network", protocol: str = "", references: str = "") -> Upscaling:
    """
    The declaration of an upscaled state variable (state_variable(upscaling=upscaling(...))): the finer-scale
    variables *of* it summarises (a name or several), the *law* that combines them ("sum", "mean", "weighted_mean",
    "series", or "network": a solved network, not a reduction), the *protocol* that defines it (the conditions it is
    computed under, e.g. "uniform soil water potential at the epidermis, the collar's xylem potential fixed"), and
    its *references*. metafspm does not compute it: the model's step does (e.g. with solve.network's
    equivalent_conductance). It is listed by declarations() and written in the encoding3d frames' declarations, so that
    an observation at the coarse scale can be told which model variable it measures.
    """
    of = (of,) if isinstance(of, str) else tuple(of)
    if not of or not all(isinstance(name, str) and name for name in of):
        raise ValueError(f"upscaling(of=...): the names of the variables it summarises, got {of!r}")
    if law not in UPSCALING_LAWS:
        raise ValueError(f"upscaling(law=...): one of {UPSCALING_LAWS}, got {law!r}")
    return Upscaling(of=of, law=law, protocol=protocol, references=references)


def _declare(role: str, unit: str = "", description: str = "", default=None, default_factory=None, kind=None,
             by: str = None, scale=None, location=None, aggregation=None, weight=None, on_grow="default", dtype=None,
             shape=None, min_value=None, max_value=None, unit_comment: str = "", value_comment: str = "",
             references: str = "", DOI=None, upscaling=None, record=None, **extra):
    """
    The metadata of a declared field, shared by state_variable, input_variable, parameter, setting and
    forcing_variable (see them for the arguments). *role*: "state_variable", "input_variable", "parameter", "setting" or
    "forcing_variable"; *extra*: the role's own keys (a setting's choices, a forcing's required and aggregate).
    """
    if aggregation in ("proximal", "distal"):
        raise ValueError(f"aggregation='{aggregation}' is 'child' or 'parent' (the value of the edge's child or "
                         "parent node)")
    metadata = dict(
        unit=unit, unit_comment=unit_comment, description=description, min_value=min_value, max_value=max_value,
        value_comment=value_comment, references=references, DOI=list(DOI or []), role=role, by=by,
        kind=kind, scale=scale, on_grow=on_grow, location=location, aggregation=aggregation, weight=weight,
        dtype=dtype, shape=tuple(shape) if shape is not None else None, upscaling=upscaling, record=record, **extra)
    if default_factory:
        return field(default_factory=default_factory, metadata=metadata)
    return field(default=default, metadata=metadata)


def _bound(value):
    """A declared bound (min_value, max_value): a number, or None when not declared ("")."""
    return None if value is None or (isinstance(value, str) and not value.strip()) else value


# The documentation keywords of every declaration: unit and description at least, the bounds of its values, the
# literature (references, DOI) and comments on the unit and on the value chosen; all listed by declarations()
_DOCUMENTATION = """
    Documentation, keywords: *unit* (UCUM-like, "mol g-1") and *description* at least; *min_value*, *max_value* (the
    bounds of its values, read by scenarios.variants); *references*, *DOI*; *unit_comment*, *value_comment*."""

_PLACE = """
    Place: *scale* (where its MTG property lives, a ScalesConfig scale or its name) and *location* (where the
    DataStructure stores it: "node", "edge", "scalar", "cell", or a coarse scale name; default: *scale*'s), with
    *aggregation* between them when they differ ("broadcast" or "split" down; "sum", "mean" or "weighted_mean" up, with
    *weight*; "child", "parent" or "mean" to edges; default: implied by its kind). *dtype*: "int" for labels and
    indices, "object" for lists and records (not for graph systems); *shape*: the per-entity shape of a vector-valued
    variable. *on_grow*: the value of entities created by growth, "default" or "inherit" (the parent's)."""

_RECORD = """
    *record*: whether the frames that keep the "recorded" variables (SceneRecorder's frame_options
    variables="recorded", io.encoding3d.unrecorded) keep it: None (default) as its role (a state variable or an input
    yes, a parameter no), True or False (e.g. False for an intermediate state of no use to look at)."""

_KINDS = """
    *kind*: how its value depends on the size of what it describes: "extensive" (an amount, summed up),
    "intensive" (a concentration, a potential, averaged up by *weight*), "massic_concentration", "descriptor",
    "non_inertial_extensive", "non_inertial_intensive"; links between kinds that disagree raise."""


def state_variable(unit: str = "", description: str = "", default=None, kind=None, scale=None, location=None,
                   aggregation=None, weight=None, on_grow="default", dtype=None, shape=None, by: str = None,
                   min_value=None, max_value=None, unit_comment: str = "", value_comment: str = "",
                   references: str = "", DOI=None, upscaling=None, record=None):
    return _declare("state_variable", **locals())


def input_variable(unit: str = "", description: str = "", default=None, kind=None, by: str = None, scale=None,
                   location=None, aggregation=None, weight=None, on_grow="default", dtype=None, shape=None,
                   min_value=None, max_value=None, unit_comment: str = "", value_comment: str = "",
                   references: str = "", DOI=None, optional: bool = False, record=None):
    return _declare("input_variable", **locals())


def parameter(unit: str = "", description: str = "", default=None, kind=None, by: str = None, scale=None,
              location=None, aggregation=None, weight=None, on_grow="default", dtype=None, min_value=None,
              max_value=None, unit_comment: str = "", value_comment: str = "", references: str = "", DOI=None,
              sections=None, record=None):
    return _declare("parameter", **locals())


def geometry_input(description: str = "the plants' elements' geometry (DataStructure.triangles)",
                   relight: str = "topology"):
    """
    Declare that the component reads the plants' geometry (data_structure.triangles(): every structural component's), listed by
    declarations() and couplings() (provided by the plants' structural components). *relight*: when an environment
    model holding it runs fully besides its interval: "topology" when elements appeared or went, "geometry" when the
    triangles changed too, None never.
    """
    if relight not in ("topology", "geometry", None):
        raise ValueError(f"geometry_input(relight=...): 'topology', 'geometry' or None, got '{relight}'")
    return _declare("geometry", description=description, default=None, relight=relight)


_PLANTING = {"x": ("m", "the plant's x in the scene (its planting)"),
             "y": ("m", "the plant's y in the scene (its planting)"),
             "z": ("m", "the plant's z in the scene: the soil surface's height minus its planting depth "
                        "(Planting.depth)"),
             "rotation": ("°", "the plant's rotation about the vertical (its planting)")}


def planting_variable(name: str, **options):
    """
    Declare one of the planting's variables, *name* one of x, y, z, rotation: an input at the Plant scale, provided by
    the planting (each Plant vertex gets its row's values before any initiator builds it). A component placing its
    plants declares those it reads and takes them as step arguments. *options*: input_variable's (e.g. default).
    """
    if name not in _PLANTING:
        raise ValueError(f"planting_variable: '{name}' is not a planting variable ({sorted(_PLANTING)})")
    unit, description = _PLANTING[name]
    arguments = dict(unit=unit, description=description, default=0., kind="descriptor", by="Planting",
                     scale=_ScalesConfig.Plant, references="metafspm")
    arguments.update(options)
    return input_variable(**arguments)


state_variable.__doc__ = """Declare a state variable, computed by this component's steps or solves. *default*: its
    initial value (a number, the same everywhere). *upscaling*: for a variable at a coarse scale that is the
    equivalent of finer ones (a root system's conductance), what it summarises and how (upscaling()).""" + _KINDS \
    + _PLACE + _RECORD + _DOCUMENTATION
input_variable.__doc__ = """Declare an input: a variable another component provides (an input named as another
    component's variable of the same DataStructure reads it; a translator link states a rename, a factor or a change
    of scale). *default*: its value when nothing provides it. *by*: the provider, for the documentation. *kind*:
    optional, checked against the provider's. *optional*: when nothing provides it (no other component, no MTG
    property), it is not registered on the DataStructure (e.g. coordinates read for a display, which a geometry
    provider would otherwise take for its own); a step naming it gets its default.""" + _PLACE + _RECORD + _DOCUMENTATION
parameter.__doc__ = """Declare a parameter: a constant of the equations, set by constructor keywords and
    scenarios. A numeric parameter without a place is stored per plant (each plant's scenario may change it); with
    *scale* or *location*, one value per entity. *by*: its owner, for the documentation. *sections*: for a parameter
    whose value is {section: {field: value}} (dtype object, e.g. an anatomy's parametrization), a function returning
    its fields, {"section.field": {"default", "unit", "description", "min_value", "max_value"}}: each is listed by
    declarations() (and parameter_table, variants) as "name.section.field", and a scenario may give it dotted
    ({"root_anatomy.cortex.n_layers": 4}) or nested ({"root_anatomy": {"cortex": {"n_layers": 4}}}).""" + _KINDS \
    + _PLACE + _RECORD + _DOCUMENTATION


def setting(unit: str = "", description: str = "", default=None, choices=None, unit_comment: str = "",
            references: str = "", DOI=None):
    """Declare a setting: a value of the model, not of the plants (a latitude, a light model, a solver, a grid's
    voxel), one for all. Listed by declarations() (role "setting"), set by constructor keywords and scenarios, read
    as `self.name` anywhere (steps and equations included), and passed as the value itself to a step naming it as an
    argument. Never stored on the DataStructure (numeric parameters are, per plant). *choices*: the values allowed,
    checked at construction. A default must be immutable (a tuple, not a list)."""
    return _declare("setting", unit=unit, description=description, default=default, unit_comment=unit_comment,
                    references=references, DOI=DOI, choices=tuple(choices) if choices is not None else None)


def forcing_variable(unit: str = "", description: str = "", required: bool = True, aggregate: str = "mean",
                     unit_comment: str = "", references: str = "", DOI=None):
    """Declare a forcing: a time series given to the scene (its `forcings` table, shared by every model), read by
    `self.forcing(name)` over the (sub-)step the component computes, the same for every entity (meteo, an imposed
    supply). Not a DataStructure variable: listed by declarations(). A Scene raises at its construction when a
    *required* forcing is not given; an optional one (required=False) raises KeyError when read, for the component to
    fall back. *aggregate*: what a read gives, declared once so that the steps do not say it: "mean" (its mean over
    the step, a rate or a state of the air: PAR, temperature), "total" (its integral over the step: a rain's depth
    from its rate, in the forcing's unit times s) or "instant" (its value at the step's end, metafspm's convention for
    implicit solves: what a step longer than the forcing's changes may miss)."""
    from openalea.metafspm.coupling.forcings import AGGREGATES
    if aggregate not in AGGREGATES:
        raise ValueError(f"forcing_variable(aggregate=...): one of {AGGREGATES}, got '{aggregate}'")
    return _declare("forcing_variable", unit=unit, description=description, default=None, unit_comment=unit_comment,
                    references=references, DOI=DOI, required=required, aggregate=aggregate)


class _PlantParameter:
    """
    A numeric parameter stored on the DataStructure: per plant when declared without a place, else at its scale or
    location (one value per entity).
      * inside a step or a graph-system equation, reading self.<name> raises: the parameter is an argument;
      * outside, reading gives the population value when every plant has the same, else the per-plant values;
      * writing (scenario setup, tests) sets every plant's value; before the DataStructure is bound, the value is
        kept and used at registration.
    """

    def __init__(self, name, default):
        self.name, self.default = name, default

    def __get__(self, obj, owner=None):
        if obj is None:
            return self.default
        if obj.__dict__.get("_in_equation"):
            raise AttributeError(f"{type(obj).__name__}.{self.name} is a parameter stored on the DataStructure: take "
                                 f"it as an argument of the step or equation ({self.name}) instead of reading "
                                 f"self.{self.name}")
        if self.name not in obj.__dict__.get("_registered_parameters", ()):
            return obj.__dict__.get("_pending_parameters", {}).get(self.name, self.default)
        values = np.asarray(obj.data_structure.get(self.name))
        if values.size and np.all(values == values.flat[0]):
            return values.flat[0].item()
        return values.copy()

    def __set__(self, obj, value):
        if self.name in obj.__dict__.get("_registered_parameters", ()):
            obj.data_structure.set(self.name, value)
        else:
            obj.__dict__.setdefault("_pending_parameters", {})[self.name] = value


@dataclass
class Component:
    """
    Base component for structuring base FSPM modules

    Variables are declared with state_variable(), input_variable(), parameter(), setting() and forcing_variable();
    the data lives in a DataStructure (see FunctionalComponent).
    """

    choregrapher = Choregrapher()

    # The named entity sets the component publishes, {name: rule}, a rule a {variable: condition} dict (the filters'
    # syntax: ">0", a value, a list of values; label names resolved), e.g. GRANAP's {"soil_nodes": {"kind": 4.}}:
    # assemblies select by their qualified names ("GranapAnatomy.soil_nodes": define_mask(entities=), a Reduction's
    # where=, ports= and keep=, evaluated on a DataStructure or on a section template), never by the codes inside
    entity_sets = {}

    # The variables whose values at its last call it reads (last_call(name)), e.g. a rule interpolating between its
    # calls: kept after each call, aligned by id at the next (no state variable of its own to copy them into)
    kept_from_last_call = ()

    def __init_subclass__(cls, **options):
        """Each declaration of the class checked when it is defined: its kind none ("" or None) or one of
        KIND_FAMILIES, its unit none or readable by the unit grammar (coupling.units); a DeclarationError names the component, the variable and
        the kind or the unit."""
        super().__init_subclass__(**options)
        from dataclasses import Field
        from openalea.metafspm.coupling.declaration import KIND_FAMILIES
        from openalea.metafspm.coupling.units import check_declared_unit
        for name, declared in list(cls.__dict__.items()):
            if not isinstance(declared, Field) or "role" not in declared.metadata:
                continue
            kind = declared.metadata.get("kind")
            if kind not in (None, "") and kind not in KIND_FAMILIES:
                raise DeclarationError(f"{cls.__name__}.{name}: unknown kind {kind!r} (one of "
                                       f"{sorted(KIND_FAMILIES)}, or none)")
            try:
                check_declared_unit(declared.metadata.get("unit"), f"{cls.__name__}.{name}")
            except ValueError as error:
                raise DeclarationError(str(error)) from None

    def __call__(self):
        started = perf_counter()
        self._call()
        self.__dict__["_wall_time"] = self.__dict__.get("_wall_time", 0.) + perf_counter() - started

    def _call(self, overrides=None, stages=None) -> None:
        """One call: the inputs pulled, the steps of its schedule (*stages*: only those of these stages, a model's
        Step(stages=)), the after-call work (*overrides*: {input: private name} during the call, a cadence's
        means)."""
        context = self._before_call()
        with self.private_names(overrides=overrides):
            self.pull_available_inputs()
            self.choregrapher(instance=self, stages=stages)
        self._after_call(context)
        self.__dict__["_ran"] = True                     # its model's run counts it (calls)

    def calls(self, reset: bool = False) -> int:
        """The scene steps in which it ran since the last reset, counted by its model's run (a cadence's skipped steps
        not counted, a component scheduled for two stages or called by a model's method counted once): the Scene's
        calls per component ("<Model>.<Component>") read and reset it."""
        count = self.__dict__.get("_calls", 0)
        if reset:
            self.__dict__.pop("_calls", None)
        return count

    def wall_time(self, reset: bool = False) -> float:
        """The wall time (s) of its calls (and of its steps run together with others, CompositeModel.run_together)
        since the last reset: the Scene's timing per component ("<Model>.<Component>") reads and resets it."""
        seconds = self.__dict__.get("_wall_time", 0.)
        if reset:
            self.__dict__.pop("_wall_time", None)
        return seconds

    def simulation_step(self):
        """The simulation step (s) the component runs in: its DataStructure's (DataStructure.simulation_time_step, the
        Scene's time step), None outside a Scene (a component alone runs one of its own steps per call); during a call
        of a schedule step's cadence (coupling.cadence), its interval."""
        if "_cadence_step" in self.__dict__:
            return self.__dict__["_cadence_step"]
        data_structure = getattr(self, "data_structure", None)
        return None if data_structure is None else getattr(data_structure, "simulation_time_step", None)

    def private_names(self, components=(), overrides=None, component_overrides=None):
        """The block in which the component's (and *components*') variables a link keeps under a private name
        (Link.names) resolve to it on their DataStructure (DataStructure.private_names): its calls, read().
        *overrides*: {variable: private name} of the component's own during the block (a cadence's means), over its
        links'; *component_overrides*: one such dict per component of *components*."""
        from contextlib import ExitStack
        stack = ExitStack()
        given = [overrides] + list(component_overrides or [None] * len(components))
        for component, extra in zip((self, *components), given):
            names = component.__dict__.get("_private_names")
            if extra:
                names = {**(names or {}), **extra}
            data_structure = getattr(component, "data_structure", None)
            if names and data_structure is not None:
                from openalea.metafspm.data_structure.data_api import _root_data_structure
                stack.enter_context(_root_data_structure(data_structure).private_names(names))
        return stack

    def _before_call(self):
        """What a call needs to keep before its steps (the end-of-call repartition's basis): None here."""
        return None

    def _after_call(self, context) -> None:
        """A call's work after its steps (also run once by CompositeModel.run_together)."""
        # The values its next call reads as last_call(name): the variables it names in kept_from_last_call, kept with
        # their entities' ids (saved by checkpoints)
        data_structure = getattr(self, "data_structure", None)
        for name in type(self).kept_from_last_call:
            self.keep(f"last_call:{name}", data_structure.kept_values(name))
        # State variables reach the MTG after every call with mtg_sync = "after_call"; by default ("lazy") the
        # DataStructure writes them when the MTG is read
        if hasattr(self, "mtg_sync"):
            self._check_mtg_sync()
            if self.mtg_sync == "after_call":
                self.write_back_to_mtg()
        # The component's clock (forcings): the start of its next call
        self.__dict__["_clock"] = self.__dict__.get("_clock", 0.) + float(
            self.simulation_step() or self.time_step or 0.)

    def write_back_to_mtg(self) -> None:
        """Write the component's state variables to the MTG; FunctionalComponent implements it."""
        pass

    @classmethod
    def declared_parameters(cls) -> tuple:
        """The names of the parameters the class declares (parameter)."""
        return tuple(f.name for f in fields(cls) if f.metadata.get("role") == "parameter")

    def pull_available_inputs(self):
        """Refresh the inputs derived by the coupling; FunctionalComponent implements it on its DataStructure."""
        pass

    @classmethod
    def declared_settings(cls) -> tuple:
        """The names of the settings the class declares (setting), read once per class."""
        if "_setting_names" not in cls.__dict__:
            cls._setting_names = tuple(f.name for f in fields(cls) if f.metadata.get("role") == "setting")
        return cls._setting_names

    def check_settings(self) -> None:
        """Each setting with choices holds one of them."""
        for f in fields(self):
            choices = f.metadata.get("choices") if f.metadata.get("role") == "setting" else None
            if choices is not None and getattr(self, f.name) not in choices:
                raise ValueError(f"{type(self).__name__}.{f.name}: one of {list(choices)}, got "
                                 f"{getattr(self, f.name)!r}")

    @classmethod
    def declared_forcings(cls) -> dict:
        """{name: required} of the forcings the class declares (forcing_variable)."""
        return {f.name: f.metadata.get("required", True) for f in fields(cls)
                if f.metadata.get("role") == "forcing_variable"}

    @classmethod
    def declarations(cls):
        """
        A pandas DataFrame of the declared variables, one row each: name; role (state_variable, input, parameter,
        setting, forcing); kind (extensive, intensive, ...: how it depends on size, None when not declared); scale
        (its name) or location; unit, description, default ("required" or "optional" for a forcing); by (an input's
        provider, a parameter's owner); min_value, max_value (the bounds of its values, None when not declared);
        references, DOI, unit_comment, value_comment; upscaling (an Upscaling, or None).
        """
        import pandas as pd
        scale_names = declared_scale_names()
        rows = []
        for f in fields(cls):
            meta = f.metadata
            role = meta.get("role")
            if role is None:
                continue
            scale = meta.get("scale")
            if role == "setting":                           # one value of the model
                place = "model"
            elif role == "forcing_variable":                         # a time series, the same for every entity
                place = "time series"
            else:
                place = scale_names.get(scale, scale) if scale is not None else meta.get("location")
            default = f.default if f.default is not MISSING else None
            if role == "forcing_variable":
                default = "required" if meta.get("required", True) else "optional"
            rows.append(dict(name=f.name, role=role, kind=meta.get("kind"), scale=place,
                             unit=meta.get("unit"), description=meta.get("description"), default=default,
                             by=meta.get("by"), min_value=_bound(meta.get("min_value")),
                             max_value=_bound(meta.get("max_value")), references=meta.get("references") or "",
                             DOI=list(meta.get("DOI") or []), unit_comment=meta.get("unit_comment") or "",
                             value_comment=meta.get("value_comment") or "", upscaling=meta.get("upscaling")))
            for suffix, entry in (meta.get("sections")() if callable(meta.get("sections")) else {}).items():
                rows.append(dict(name=f"{f.name}.{suffix}", role=role, kind=None, scale=place,
                                 unit=entry.get("unit", ""), description=entry.get("description", ""),
                                 default=entry.get("default"), by=meta.get("by"),
                                 min_value=_bound(entry.get("min_value")), max_value=_bound(entry.get("max_value")),
                                 references=meta.get("references") or "", DOI=list(meta.get("DOI") or []),
                                 unit_comment="", value_comment="", upscaling=None))
        table = pd.DataFrame(rows, columns=DECLARATION_COLUMNS)
        table["default"] = pd.Series([row["default"] for row in rows], index=table.index, dtype=object)  # ints kept
        for column in ("kind", "min_value", "max_value", "upscaling"):
            table[column] = table[column].astype(object).where(table[column].notna(), None)
        return table

    @classmethod
    def schedule(cls) -> list:
        """[(stages, [step methods])]: the class' steps in the order they run, grouped by their stage."""
        return cls.choregrapher.describe(cls)

    def run_stage(self, stage: str):
        """Run once only the steps of one schedule row (*stage*: "potential", "segmentation", "postsegmentation",
        "rate", ...), after pulling the inputs."""
        self.pull_available_inputs()
        self.choregrapher.run_stage(self, stage)
        self.__dict__["_ran"] = True


@dataclass
class DataStructureComponent(Component):
    """
    Component bound to a DataStructure (base of FunctionalComponent and StructuralComponent).

    Its declared fields are resolved once (resolve_declaration) and registered on the DataStructure when missing,
    from their MTG property or their default; its state variables are written to the MTG after every call, and its
    MTG-backed parameters are re-read before each graph solve.
    """

    @property
    def soil_surface(self) -> float:
        """The soil surface's height (m, the plants' frame): its DataStructure's (Scene(soil_surface=), 0 by
        default), e.g. a light model's cut, the domains "air" and "soil" of data_structure.geometry."""
        return self.data_structure.soil_surface

    def calendar(self, time: float = None, default=None) -> tuple:
        """(day of year, hour of the day) at *time* (default: forcing_time(), the scene time the step's forcings are
        read at), from the Scene's start (Scene(start="2026-05-10 00:00"): the date of its time 0), else from
        *default* (the date of time 0 without a Scene start, in the same form; None: raise)."""
        from openalea.metafspm.coupling.durations import day_and_hour, start_date
        start = self.__dict__.get("_scene_calendar")
        if start is None:
            if default is None:
                raise ValueError(f"{type(self).__name__}.calendar(): the Scene has no start= (a date, "
                                 "'2026-05-10 00:00')")
            start = start_date(default, f"{type(self).__name__}.calendar(default=)")
        time = self.forcing_time() if time is None else float(time)
        return day_and_hour(start, time)

    def set_forcings(self, forcings, start=None) -> None:
        """
        The forcings of a component run outside a Scene, as a Scene gives them (Scene(forcings=..., start=...)):
        {name: forcing} or a DataFrame indexed by time (s), one column each (coupling.forcings), read with
        forcing(name) at the component's clock (its time from 0, advanced by each call); *start*, the date of time 0
        ("2026-05-10 00:00", coupling.durations.start_date: calendar(), the forcings of origin "calendar"). Replaces
        the forcings given before; a Scene gives its own.
        """
        from openalea.metafspm.coupling.durations import start_date
        from openalea.metafspm.coupling.forcings import forcing_table
        table = forcing_table(forcings)
        start = None if start is None else start_date(start)
        for forcing in table.values():
            if getattr(forcing, "origin", None) == "calendar":
                if start is None:
                    raise ValueError("a forcing of origin 'calendar' needs start= (a date, '2026-05-10 00:00')")
                forcing.at_calendar(start)
        self.__dict__["_scene_forcings"] = table
        self.__dict__["_scene_calendar"] = start

    def has_forcing(self, name: str) -> bool:
        """Whether forcing *name* is given (by the Scene or set_forcings) as a time series, readable with
        forcing(name) (not one varying in space)."""
        from openalea.metafspm.coupling.forcings import is_spatial
        forcings = self.__dict__.get("_scene_forcings") or {}
        return name in forcings and not is_spatial(forcings[name])

    def forcing_time(self, at: str = "end") -> float:
        """
        The time at which forcings are read: the end of the current (sub-)step for implicit solves and steps,
        the evaluation time inside an IVP solve; with *at* "start", the start of the current (sub-)step. Times count
        from the component's clock (the scene time).
        """
        start = self.__dict__.get("_clock", 0.)
        offset = self.__dict__.get("_solve_offset", 0.)
        if at == "start":
            return start + offset
        if "_ivp_time" in self.__dict__:
            return start + offset + self.__dict__["_ivp_time"]
        step = self.__dict__.get("_current_dt", None)
        if step is None:                                   # outside a solve: the end of the component's sub-step
            step = float(getattr(self, "time_step", None) or self.simulation_step() or 0.)
        return start + offset + step

    def forcing_interval(self, over="substep") -> tuple:
        """
        (start, end), the scene times a forcing is read over: *over* "substep", the (sub-)step being computed (the
        component's sub-step, a graph-system solve's sub-step: [t, t + dt]); "step", the whole step of the call (the
        scene's step, all its sub-steps); a (start, end) pair of scene times, itself (a part of the step).
        """
        if isinstance(over, tuple):
            start, end = (float(time) for time in over)
            if end < start:
                raise ValueError(f"forcing(over=(start, end)): end {end} before start {start}")
            return start, end
        if over == "substep":
            return self.forcing_time("start"), self.forcing_time("end")
        if over == "step":
            start = self.__dict__.get("_step_clock", self.__dict__.get("_clock", 0.))
            length = self.__dict__.get("_step_length")
            if length is None:
                length = float(self.simulation_step() or getattr(self, "time_step", None) or 0.)
            return start, start + length
        raise ValueError(f"forcing(over=...): 'substep', 'step' or a (start, end) pair, got '{over}'")

    def forcing_panels(self, over="substep") -> list:
        """The interval *over* (forcing_interval's) in panels of at most an hour (coupling.forcings.QUADRATURE_PANEL,
        the forcings' quadrature panels), [(start, end)]: for a model whose response to a forcing is not its mean's
        (the sun's course over a daily step, lit panel by panel), each read with forcing(name, over=panel)."""
        from openalea.metafspm.coupling.forcings import panels
        edges = panels(*self.forcing_interval(over))
        return list(zip(edges[:-1].tolist(), edges[1:].tolist()))

    def forcing(self, name: str, over: str = "substep"):
        """
        Forcing *name* (coupling.forcings: a number, a callable of the time, a table linearly interpolated, a pulse)
        over the (sub-)step being computed (*over* "substep", [t, t + dt]), the call's whole step ("step") or a
        (start, end) pair of scene times (a panel of the step, forcing_panels()), as its
        declaration aggregates it (forcing_variable(aggregate=...): its mean by default, its total, or its value at the
        end; an undeclared name: its mean). Inside an IVP solve, its value at the evaluation time (the integrator
        averages it; a "total" has no value at an instant and raises).
        """
        declared = getattr(type(self), "__dataclass_fields__", {}).get(name)
        aggregate = declared.metadata.get("aggregate", "mean") if declared is not None else "mean"
        forcings = self.__dict__.get("_scene_forcings") or {}
        if name not in forcings:
            raise KeyError(f"{type(self).__name__} has no forcing '{name}' (the Scene's forcings=, or "
                           "set_forcings())")
        from openalea.metafspm.coupling.forcings import forcing_over, forcing_value, is_spatial
        if is_spatial(forcings[name]):
            raise TypeError(f"forcing '{name}' varies in space ((time, ids) -> values): the Scene writes it into an "
                            "input or parameter, a component does not read it with forcing()")
        if "_ivp_time" in self.__dict__:
            if aggregate == "total":
                raise ValueError(f"forcing '{name}' is a total over the step (aggregate='total'): an IVP solve reads "
                                 "rates at instants, declare it aggregate='mean'")
            return forcing_value(forcings[name], self.forcing_time())
        start, end = self.forcing_interval(over)
        return forcing_over(forcings[name], start, end, aggregate)

    def pool_exchange(self, name: str):
        """
        During a graph-system solve with pool unknowns: the sparse (n_nodes, n_pools) map between the nodes of
        the pool's exchange set and the pool of their entity. P @ pool gives each node its pool's value; P.T @ flux
        sums node fluxes into their pools.
        """
        maps = self.__dict__.get("_pool_exchange") or {}
        if name not in maps:
            raise KeyError(f"'{name}' is not a pool unknown of the graph system being solved")
        return maps[name]

    @property
    def step_duration(self) -> float:
        """Length of the current (sub-)step of a graph-system solve; the time step outside sub-stepping (every component's steps)."""
        step = self.__dict__.get("_current_dt", getattr(self, "time_step", None))
        if step is None:
            raise AttributeError(f"{type(self).__name__} has no time step: give its time_step= (s), or build it in a "
                                 "Scene (its time_step=)")
        return step


    data_structure: Optional[DataStructure] = None
    # Time step of the component (s): the dt of its equations, and its sub-step: it runs (the scene's time step) /
    # time_step times per scene step, once when run alone. Default: its class's time_step when it sets one, else the
    # scene's
    time_step: Optional[float] = None

    # MTG synchronisation policy: "lazy" (default) writes the state variables
    # changed since the last synchronisation when the MTG is read (data_structure.mtg, MPG-style steps, data_structure.flush_mtg());
    # "after_call" writes them after every call; "never" leaves the MTG untouched.
    mtg_sync = "lazy"

    # The entities the component owns, when it shares the plant with other models (the roots' and the shoot's): a
    # filters-style declaration, e.g. {"label": ("RootSegment", "RootApex")} (label names resolved), or a callable
    # data_structure -> boolean array. It then works on a view of the DataStructure restricted to them
    # (data_structure.EntityView): its nodes those entities, its topology and mappings theirs. None: every entity
    entities = None

    def __post_init__(self):
        self.check_settings()
        self._install_plant_parameters()
        if self.data_structure is None:
            raise TypeError(
                f"{type(self).__name__}() requires a DataStructure as its first argument. "
                f"Pass an MPGDataStructure (or other DataStructure subclass) instance."
            )
        if not isinstance(self.data_structure, DataStructure) or not hasattr(self.data_structure, "register"):
            raise TypeError(
                f"{type(self).__name__}() data_structure must be a DataStructure with a variable store "
                f"(MPGDataStructure, ArrayDataStructure), got {type(self.data_structure).__name__}."
            )
        self.data_structure = _working_data_structure(self, self.data_structure)
        data_structure = self.data_structure
        self._auto_declare_on_ds(data_structure)
        if not hasattr(self, "pullable_inputs"):
            self.pullable_inputs = {}
        simulation_step = self.simulation_step()
        if self.time_step is None:
            declared = getattr(type(self), "time_step", None)                 # a class attribute of a subclass
            self.time_step = declared if declared is not None else simulation_step
        # The component runs once per time_step: simulation step / time_step times per simulation step; once per
        # call alone, or without a time step (then its equations may not read self.step_duration)
        if simulation_step is not None and self.time_step is not None:
            from openalea.metafspm.coupling.choregrapher import sub_steps
            sub_steps(simulation_step, self.time_step, type(self).__name__)
        # Live reading: steps and solves read and write the DataStructure arrays
        self.choregrapher.bind(self, self.time_step, data_structure)
        # Its after_regrid methods, called by its grid after each refinement or coarsening
        for name in _after_regrid_methods(type(self)):
            data_structure.add_regrid_hook(getattr(self, name))

    def pull_available_inputs(self):
        """
        Before the step: re-read the MTG-backed parameters, and bring the inputs derived by the coupling up
        to date. Derived variables are recomputed when read, so the latter only recomputes those whose sources
        changed; it is the hook of future sub-steps.
        """
        self._refresh_from_bio_scale()
        data_structure = self.data_structure
        for name in getattr(self, "_derived_inputs", []):
            data_structure.get(name)

    @classmethod
    def _optional(cls, name) -> bool:
        """Whether *name* is an optional input (input_variable(optional=True))."""
        field = cls.__dataclass_fields__.get(name)
        return field is not None and bool(field.metadata.get("optional"))

    def read(self, name: str) -> np.ndarray:
        """A declared variable's values at the component's own place (its declaration's location), outside steps (a
        helper, a display): through the mapping when another component holds it elsewhere (held_at_scale), its
        default when it is an optional input nothing provides; else the DataStructure's."""
        with self.private_names():
            return self._read(name)

    def _read(self, name: str) -> np.ndarray:
        from openalea.metafspm.coupling.declaration import _mapped_read
        data_structure, spec = self.data_structure, self.variable_specs.get(name)
        if spec is None:
            raise KeyError(f"{type(self).__name__} declares no '{name}'")
        if held_at_scale(spec, data_structure):
            return _mapped_read(data_structure, spec)
        if not data_structure.has(name) and self._optional(name):
            return np.full(tuple(data_structure._location_shape(spec.location)) + tuple(spec.shape or ()),
                           np.nan if spec.default is None else spec.default)
        return data_structure.get(name)

    @property
    def variable_specs(self) -> dict:
        """{name: VariableSpec} of the component's declared variables as resolved on its DataStructure (location, MTG
        scale, mapping, default, kind...), read-only."""
        from types import MappingProxyType
        return MappingProxyType(self.__dict__.get("_variable_specs", {}))

    def _auto_declare_on_ds(self, data_structure: DataStructure) -> None:
        """
        Register the declared variables that are not yet on *data_structure*.

        Each field is resolved once by resolve_declaration into a VariableSpec (location, MTG scale, mapping),
        kept in self._variable_specs. A variable is registered at its location with the values of its MTG
        property when the property exists (mapped from its scale), else with its declared default.
        Pre-registered variables are never overwritten; their location must match the declaration.
        """
        self._variable_specs = _declared_specs(self, data_structure)
        for name, spec in self._variable_specs.items():
            pending = self.__dict__.get("_pending_parameters", {})
            if data_structure.has(name):
                registered = data_structure.location(name)
                if registered != spec.location and name not in data_structure.aliases() and not held_at_scale(spec, data_structure):
                    raise DeclarationError(f"{type(self).__name__}.{name} is declared at {spec.location} but is "
                                           f"already registered at {registered}")
                if name in pending and pending[name] != spec.default:
                    data_structure.set(name, pending[name])          # a value given to the constructor
            else:
                values = data_structure._read_mtg(spec) if hasattr(data_structure, "_read_mtg") else None
                if values is None and self._optional(name):
                    continue                     # an optional input nothing provides: its default, not registered
                default = pending.get(name, spec.default) if spec.dtype is not object else spec.default
                data_structure.register(name, values, location=spec.location, default=default, on_grow=spec.on_grow,
                            dtype=spec.dtype, shape=spec.shape)
            # Metadata precedence: the component that owns the variable (not an input) sets its default, growth
            # policy and kind; a component reading it as an input only fills what is still unknown.
            meta = data_structure._variable_meta().setdefault(name, {})
            declared = {key: value for key, value in spec.meta().items() if value is not None}
            if spec.role == "input_variable":
                for key, value in declared.items():
                    meta.setdefault(key, value)
            else:
                meta.update(declared, default=spec.default, on_grow=spec.on_grow)
        self.__dict__["_registered_parameters"] = {name for name in getattr(type(self), "_plant_parameter_names", ())
                                                   if name in self._variable_specs}
        self.__dict__.pop("_pending_parameters", None)
        self._check_mtg_sync()
        if self.mtg_sync == "lazy" and hasattr(data_structure, "_track_mtg"):
            for spec in self._variable_specs.values():
                if spec.role == "state_variable" and spec.mtg_backed:
                    data_structure._track_mtg(spec)

    def _check_mtg_sync(self) -> None:
        if self.mtg_sync not in ("lazy", "after_call", "never"):
            raise ValueError(f"{type(self).__name__}.mtg_sync must be 'lazy', 'after_call' or 'never', got "
                             f"'{self.mtg_sync}'")

    @classmethod
    def _plant_parameter_fields(cls) -> dict:
        """
        {name: default} of the numeric parameters, stored on the DataStructure: per plant when declared without a
        place, else at their scale or location. They have one value, the DataStructure's.
        """
        names = {}
        for f in fields(cls):
            meta = f.metadata
            if (meta.get("role") == "parameter"
                    and meta.get("dtype") in (None, float, int, "float", "int") and not isinstance(f.default, bool)
                    and isinstance(f.default, (int, float, np.integer, np.floating))):
                names[f.name] = f.default
        return names

    def _install_plant_parameters(self) -> None:
        """Replace, once per class, each numeric parameter's attribute by a _PlantParameter descriptor."""
        cls = type(self)
        if "_plant_parameter_names" not in cls.__dict__:
            names = cls._plant_parameter_fields()
            for name, default in names.items():
                setattr(cls, name, _PlantParameter(name, default))
            cls._plant_parameter_names = tuple(names)
        # Values set by the dataclass __init__ before the descriptors existed (first instance of the class)
        pending = self.__dict__.setdefault("_pending_parameters", {})
        for name in cls._plant_parameter_names:
            if name in self.__dict__:
                pending[name] = self.__dict__.pop(name)

    @classmethod
    def plant_parameters(cls, scenario) -> dict:
        """
        {name: value} of the class's parameters and settings for one plant: its *scenario*'s ({name: value}, a
        plant's, e.g. initiate_plant's), else the declared default; each checked against its declaration: within its
        bounds (min_value, max_value) when declared and numeric, a whole number when declared dtype "int". An
        initiator reads its parameters with it (StructuralComponent.initiate_plant).
        """
        scenario = scenario or {}
        values = {}
        for f in fields(cls):
            meta = f.metadata
            if meta.get("role") not in ("parameter", "setting"):
                continue
            value = scenario.get(f.name, f.default if f.default is not MISSING else None)
            values[f.name] = value
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
                continue
            if meta.get("dtype") in (int, "int") and float(value) != round(float(value)):
                raise ValueError(f"{cls.__name__}.{f.name}: a whole number, got {value}")
            low, high = _bound(meta.get("min_value")), _bound(meta.get("max_value"))
            if np.isfinite(value) and ((isinstance(low, (int, float)) and value < low)
                                       or (isinstance(high, (int, float)) and value > high)):
                raise ValueError(f"{cls.__name__}.{f.name}: {value} is outside its declared bounds "
                                 f"[{low if low is not None else '-inf'}, {high if high is not None else 'inf'}]")
        return values

    def parameter_value(self, name: str):
        """The single value of parameter *name* for the population, outside steps (structural code, helpers); raises
        when the plants' values differ (use parameter_values then)."""
        values = np.asarray(self.data_structure.get(name))
        if values.size == 0:
            return getattr(type(self), name, None)
        first = values.flat[0]
        same = np.all(values == first) or (values.dtype.kind == "f" and np.isnan(first) and np.all(np.isnan(values)))
        if not same:
            raise ValueError(f"{type(self).__name__}.{name} differs between the plants ({np.unique(values)[:5]}...): "
                             "use parameter_values(name, location)")
        return first.item()

    def keep(self, name: str, value, saved: bool = True) -> None:
        """Keep *value* on the component under *name*: state that is not a DataStructure variable (the time of the
        last lighting), saved by the Scene's checkpoints (checkpoint_state); with *saved* False, a cache that is
        rebuilt when missing (a kept canopy, incremental meshes): one of the component's caches (caches()), left out
        of checkpoints and dropped when one is loaded. Derived values rebuilt on a key are `cached`'s."""
        if saved:
            self.caches().pop(("kept", name), None)
            self.__dict__.setdefault("_kept", {})[name] = value
        else:
            self.__dict__.get("_kept", {}).pop(name, None)
            self.caches()[("kept", name)] = value

    def kept(self, name: str, default=None):
        """What keep() kept under *name* (*default* when nothing is)."""
        kept = self.__dict__.get("_kept", {})
        if name in kept:
            return kept[name]
        return self.__dict__.get("_caches", {}).get(("kept", name), default)

    def last_call(self, name: str, missing=np.nan) -> np.ndarray:
        """
        The values variable *name* (one of the class' kept_from_last_call) had at the end of the component's last
        call, one per entity of its location now, aligned by id: an entity new since, and every entity before the
        first call, gets *missing*. Its schedule step's cadence makes it the last call's, not the last scene step's
        (data_structure.previous: the step's start).
        """
        if name not in type(self).kept_from_last_call:
            raise KeyError(f"{type(self).__name__}.last_call('{name}'): name it in the class' kept_from_last_call "
                           f"({list(type(self).kept_from_last_call)})")
        kept = self.kept(f"last_call:{name}")
        data_structure = self.data_structure
        if kept is None:
            return np.full(np.shape(data_structure.get(name)), missing, dtype=np.result_type(float, np.asarray(missing).dtype))
        return data_structure.aligned(name, *kept, missing=missing)

    def checkpoint_state(self) -> dict:
        """The state checkpoints save: what keep() kept."""
        return {"kept": dict(self.__dict__.get("_kept", {}))}

    def restore_state(self, state: dict) -> None:
        self.__dict__["_kept"] = dict(state.get("kept", {}))

    def cached(self, name: str, build, key=None):
        """A value kept on the component: build() at the first call and again when *key* changes (by default the
        DataStructure's topology version: values read from the structure). One of the component's caches (caches()):
        never checkpointed, dropped when a checkpoint is loaded."""
        key = self.data_structure.topology_version if key is None else key
        store = self.caches()
        entry = store.get(name)
        if entry is None or entry[0] != key:
            entry = (key, build())
            store[name] = entry
        return entry[1]

    def caches(self) -> dict:
        """The component's caches, {key: value}: cached()'s values, keep(saved=False)'s, its graph systems' views,
        pieces and previous solutions. What is built from the DataStructure and rebuilt when missing: never
        checkpointed, all dropped by clear_caches() (Scene.load_checkpoint: built on the topology before the load)."""
        return self.__dict__.setdefault("_caches", {})

    def conservation(self, reset: bool = False) -> dict:
        """The balances of the quantities its graph systems conserve (graph_system(conserved=...),
        solve.conservation), {(system, quantity): {"inflow", "outflow", "change", "residual", "unit"}} since the last
        reset (*reset*: a new balance from now on; the SceneRecorder resets it after each step)."""
        from openalea.metafspm.solve.conservation import balances
        return balances(self, reset=reset)

    def clear_caches(self) -> None:
        """Drop every cache of the component (caches()): what was built before a checkpoint was loaded."""
        self.__dict__.pop("_caches", None)

    def parameter_values(self, name: str, location: str = "node") -> np.ndarray:
        """
        Per-entity values of a per-plant parameter, for code that cannot take it as an argument (MPG-style structural
        steps): each node (or edge) gets its plant's value.
        """
        return self.data_structure.parameter_view(name, location)

    def parameter_table(self, names, location: str = "node") -> dict:
        """
        {name: per-entity values of *location*} of the per-plant parameters *names* (parameter_values for each): the
        entities' owners found once per topology and shared, each parameter one gather from its per-plant values
        (read-only arrays, a zero-stride one when every plant has the same value).
        """
        data_structure = self.data_structure
        return {name: data_structure.parameter_view(name, location) for name in names}

    # Seed of this component's random streams; a scenario may set it
    random_seed = 0

    def random(self, stream: str, distribution: str = "uniform", ids=None, location: str = "node",
               **parameters) -> np.ndarray:
        """
        Reproducible draws per entity for stream *stream*: each call of a stream is a new step, so a model drawing in
        the same order gets the same draws at every run, whatever the visiting order (data_structure.random).
        """
        name = f"{type(self).__name__}.{stream}"
        holder = self.data_structure
        if isinstance(holder, EntityView):
            holder = holder.__dict__["_base"]                  # its DataStructure's, which checkpoints save
        counters = holder.__dict__.setdefault("_random_steps", {})   # kept by checkpoints
        step = counters.get(name, 0)
        counters[name] = step + 1
        return self.data_structure.random(distribution, stream=name, step=step, ids=ids, location=location,
                                          seed=int(self.random_seed), **parameters)

    def active_ids(self, location: str = "node") -> np.ndarray:
        """
        Ids selected by the DataStructure's "active" mask (every one without it), for MPG-style steps, which loop
        over vertices themselves: e.g. plants before their planting time are skipped. At a coarse *location* (e.g.
        "SubOrgan", whose new entities may have no node yet), the entities of the plants planted.
        """
        data_structure = self.data_structure
        ids = np.asarray(data_structure.entity_ids(location))
        if location == "node":
            return ids[np.asarray(data_structure.mask("active"), dtype=bool)] if data_structure.has_mask("active") else ids
        if not data_structure.has("planted"):
            return ids
        plants = data_structure.map_to(np.asarray(data_structure.get("planted"), dtype=float), "node", "Plant", "max")
        if location == "Plant":
            return ids[np.nan_to_num(plants) > 0]
        planted = data_structure.map_to(np.nan_to_num(plants), "Plant", location, "broadcast")
        return ids[np.asarray(planted) > 0]

    def write_back_to_mtg(self) -> None:
        """
        Write the declared state variables with an MTG scale to the MTG, at the vertices of their scale, through the
        inverse of their mapping (MPGDataStructure._write_mtg). Called after every component call with
        mtg_sync = "after_call"; with "lazy" the DataStructure does it when the MTG is read.

        Only ``state_variable`` fields are written: parameters and inputs are owned by whoever sets them, and
        writing their defaults back would overwrite externally set MTG values.
        """
        self._check_mtg_sync()
        data_structure = self.data_structure
        if self.mtg_sync == "never" or not hasattr(data_structure, "_write_mtg"):
            return
        for spec in getattr(self, "_variable_specs", {}).values():
            if spec.role == "state_variable" and spec.mtg_backed and data_structure.has(spec.name):
                data_structure._write_mtg(spec)

    def _refresh_from_bio_scale(self) -> None:
        """
        Re-read the declared parameters with an MTG scale from the MTG, before each graph-system solve, so that
        parameters updated on the MTG by other components (e.g. K_axial by a growth model) are seen.

        State variables and inputs are not refreshed: their DataStructure values are set by steps and couplings.
        Parameters whose MTG property is absent keep their DataStructure values.
        """
        data_structure = self.data_structure
        if not hasattr(data_structure, "_read_mtg"):
            return
        for spec in getattr(self, "_variable_specs", {}).values():
            if spec.role != "parameter" or not spec.mtg_backed or held_at_scale(spec, data_structure):
                continue                             # (one held at its scale by another component: its own)
            if hasattr(data_structure, "flush_mtg"):
                data_structure.flush_mtg([spec.name])            # a value set on the DataStructure is not lost to a stale MTG one
            values = data_structure._read_mtg(spec)
            if values is not None:
                data_structure.set(spec.name, values)


@dataclass
class FunctionalComponent(DataStructureComponent):
    """
    Base for all functional (transport / balance) model components.

    Every subclass must be initialized with a DataStructure instance that
    provides the topology and initial field values.  The DataStructure is the
    single source of truth: steps and graph-system solves read and write its
    arrays live.

    Auto-registration
    -----------------
    __post_init__ calls _auto_declare_on_ds().
    Every field annotated with a scale ("node", "edge", a biological scale, or "cell" / "scalar" on grids) whose name is not
    yet registered on the DataStructure is initialized with the field default
    value (uniform array).  This means:

    * In production, simply construct the model — all fields with declared
      scale appear in data_structure with their default values automatically.
    * When non-default values are needed (tests, scenario setup), call
      data_structure.register(name, values, location=...) BEFORE constructing the
      model.  Pre-registered values are never overwritten.

    Usage::

        data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
        data_structure.register("concentration", c_init, location="node")   # non-default — must pre-set
        model = MyTransportModel(data_structure=data_structure)
        # data_structure now also contains default arrays for K_axial, volumetric_capacity, etc.
    """

    @property
    def _graph_view(self):
        """
        GraphView of the DataStructure, rebuilt when its topology changed (growth), None for grids. During a solve on
        an active subgraph (where=), the subgraph's view.
        """
        solve_view = self.__dict__.get("_solve_view")
        if solve_view is not None:
            return solve_view
        data_structure = self.data_structure
        if not hasattr(data_structure, "to_graph_view"):
            return None
        version = getattr(data_structure, "topology_version", None)
        return self.cached("graph_view", data_structure.to_graph_view, key=version)

    @_graph_view.setter
    def _graph_view(self, view):
        # Explicit views (e.g. with boundary ports built by hand) are kept until the topology changes
        self.caches()["graph_view"] = (getattr(self.data_structure, "topology_version", None), view)

    @property
    def graph_view(self):
        """The graph the component's graph systems solve on (its incidence, edge_difference...): in a graph system's
        terms, the active subgraph during its solve."""
        return self._graph_view

    def previous(self, name: str, at: str = "substep") -> np.ndarray:
        """
        Value of unknown *name*, managed by the framework::

              at="substep" (default): at the start of the current (sub-)step of the solve;
              at="solve":             at the start of this call's solve (the same with integrate="once");
              at="step":              at the start of the component's call, e.g. for operator splitting between
                                      several graph systems of one component.
        """
        if at == "substep":
            state = getattr(self, "_previous_state", None)
        elif at == "solve":
            state = self.__dict__.get("_solve_start_state")
        elif at == "step":
            state = self.__dict__.get("_step_start_state")
            if state is not None and name in state:
                location, values = state[name]
                restriction = self.__dict__.get("_restriction")
                if restriction is not None:
                    values = values[restriction.node_idx if location == "node" else restriction.edge_idx]
                return values
        else:
            raise ValueError(f"previous(at=) must be 'substep', 'solve' or 'step', got '{at}'")
        if state is None or name not in state:
            raise KeyError(f"No previous state for '{name}': previous() is available inside a graph-system solve "
                           "of one of its unknowns.")
        return state[name]

    def pull_available_inputs(self):
        super().pull_available_inputs()
        # previous(fn, at="step"): the unknowns of the component's graph systems at the start of the call
        data_structure = self.data_structure
        unknowns = {fn: location for spec in getattr(type(self), "_graph_system_specs", {}).values()
                    for location, names in (("node", spec["node_unknowns"]), ("edge", spec["edge_unknowns"]))
                    for fn in names}
        self.__dict__["_step_start_state"] = {fn: (location, np.array(data_structure.get(fn), dtype=np.float64).reshape(-1))
                                              for fn, location in unknowns.items() if data_structure.has(fn)}


def after_regrid(method):
    """
    Mark a component's method as its grid's carry-over hook: an adaptive grid (AdaptiveGridDataStructure) calls it
    after each refinement or coarsening, once the grid has carried the variables over (extensive ones by volume,
    the others volume-averaged) and its geometric descriptors are those of its new cells, so that a variable the
    volume averages do not carry (a soil's pressure head, from the water carried) is set again from those.
    """
    method._after_regrid = True
    return method


def _after_regrid_methods(cls) -> list:
    """The names of *cls*' after_regrid methods."""
    return [name for name in dir(cls) if getattr(getattr(cls, name, None), "_after_regrid", False)]


def display_end(name: str):
    """
    Declare one of an element's display ends, *name* one of x1, y1, z1, x2, y2, z2 (the field display_<name>, m, at
    the SubOrgans): metafspm's convention is that x1 ... z2 are topological (each element's base at its parent's
    end, data_structure.mpg.MPG.add_element), and an element drawn elsewhere (a turtle's hidden parts on top of the
    visible ones, a seed drawn as its grain) gives where in display_x1 ... display_z2, NaN at its topological place.
    The default cylinders (coupling.geometry.segment_cylinders) and the readers of the drawing (an anatomy's 3D
    placement) take them where given (coupling.geometry.drawn_ends)::

        display_x1: float = display_end("x1")         # ... the six of them, in a component drawing its elements
    """
    if name not in ("x1", "y1", "z1", "x2", "y2", "z2"):
        raise ValueError(f"display_end: one of x1, y1, z1, x2, y2, z2, got {name!r}")
    end, axis = ("base", "end")[int(name[-1]) - 1], name[0]
    return state_variable(unit="m", description=f"Where the element's {end} is drawn, {axis} (its display end)",
                          value_comment="NaN: at its topological place (x1 ... z2)", kind="descriptor",
                          default=np.nan, scale=_ScalesConfig.SubOrgan)


class NewComplexes:
    """
    New coarse vertices (Phytomers, Organs, ...) a NewNodes batch creates for its nodes, in order::

        anchor   per complex, the vertex it is a component of: an existing vertex's id (>= 0), or an earlier new
                 complex, -1 - its index in this table
        scale    the complexes' scale(s) (one for all, or per complex); a complex is one scale finer than its anchor
        label    their MPG label(s)
        values   {variable: values per complex, or one for all}, variables of the complexes' location (their
                 scale's name)
    """

    def __init__(self, anchor, scale, label, values=None):
        self.anchor = np.asarray(anchor, dtype=np.int64).reshape(-1)
        n = self.anchor.size
        self.scale = np.broadcast_to(np.asarray(scale, dtype=np.int64), (n,))
        self.label = np.broadcast_to(np.asarray(label, dtype=object), (n,))
        self.values = dict(values or {})
        inner = self.anchor < 0
        if np.any(-1 - self.anchor[inner] >= np.flatnonzero(inner)):
            raise ValueError("NewComplexes: a complex's anchor must come before it in the table")

    def __len__(self):
        return self.anchor.size


class NewNodes:
    """
    Nodes an array-style step of a StructuralComponent creates, returned as one of its (name, values) output pairs
    (the name is free): the framework creates them in the MPG in bulk, extends the DataStructure with their values
    and repartitions the other components' variables, without the MPG synchronisation of an MPG-style step::

        return (length, "struct_mass", mass,
                "new_elements", NewNodes(parent=parents, edge=edges, label=labels, values={"length": lengths},
                                         new_complex=laterals,
                                         complex_levels=((scales.Organ, labels.Organ.RootInternode),),
                                         relabel=(apex_ids, segment_label)))

    Its arguments::

        parent          per new node, its parent: an existing node's id (>= 0), or another new node, -1 - its
                        index in the batch (a parent comes before its children)
        edge            "<" or "+", per node or for all
        label           the nodes' MPG label (an integer of the MPG's labels), per node or for all
        values          {variable: values per node, or one for all}: the other variables take their on_grow value
                        (their default, or their parent's with on_grow="inherit")
        new_complex     per node (or for all), whether it starts a complex of its own (e.g. a lateral root's
                        Organ), created under its parent's complex at the scale above complex_levels' first one;
                        otherwise the node joins its parent's complex
        complex_levels  ((scale, label), ...) of the new complexes, coarsest first, down to the scale above the
                        nodes'
        relabel         (existing node ids, their new label(s)): e.g. an apex becoming a segment
        complexes       NewComplexes: complexes created explicitly (e.g. a Phytomer holding three Organs), instead
                        of new_complex / complex_levels
        complex         per node, its complex when complexes is given: an existing vertex's id (>= 0), or a new
                        complex, -1 - its index in complexes

    A node without a topological parent (a plant's first element) has parent NewNodes.NO_PARENT.
    """
    NO_PARENT = np.iinfo(np.int64).min

    def __init__(self, parent, edge="<", label=None, values=None, new_complex=False, complex_levels=(),
                 relabel=None, complexes=None, complex=None):
        self.parent = np.asarray(parent, dtype=np.int64).reshape(-1)
        n = self.parent.size
        self.complexes = complexes
        self.complex = None if complex is None else np.broadcast_to(np.asarray(complex, dtype=np.int64), (n,))
        if (complexes is None) != (complex is None):
            raise ValueError("NewNodes: complexes and complex go together")
        if self.complex is not None and np.any(-1 - self.complex[self.complex < 0] >= len(complexes)):
            raise ValueError("NewNodes: a node's complex is not in complexes")
        self.edge = np.broadcast_to(np.asarray(edge, dtype=object), (n,))
        self.label = np.broadcast_to(np.asarray(label, dtype=object), (n,))
        self.values = dict(values or {})
        self.new_complex = np.broadcast_to(np.asarray(new_complex, dtype=bool), (n,))
        self.complex_levels = tuple(complex_levels)
        self.relabel = relabel
        inner = (self.parent < 0) & (self.parent != self.NO_PARENT)
        if np.any(-1 - self.parent[inner] >= np.flatnonzero(inner)):
            raise ValueError("NewNodes: a new node's parent must come before it in the batch")
        if self.new_complex.any() and not self.complex_levels:
            raise ValueError("NewNodes: new_complex needs complex_levels")
        if not set(self.edge.tolist()) <= {"<", "+"}:
            raise ValueError(f"NewNodes: edges are '<' or '+', got {sorted(set(self.edge.tolist()))}")

    def __len__(self):
        return self.parent.size


@dataclass
class StructuralComponent(DataStructureComponent):
    """
    Component that edits the plant's structure (growth, segmentation, anatomy). It shares the plant's DataStructure, and edits the MPG through ``self.mtg`` with the MPG's own
    methods: the DataStructure does not re-expose them, the MPG is the source of truth for structure.

    Two step styles coexist:
      * a step without arguments ("MPG-style") reads and edits the MPG. Around it, the framework writes the
        component's declared variables to the MPG before, and after it updates the DataStructure's topology if the
        MPG's changed, then re-reads the declared state variables (the structural outputs) from the MPG;
      * a step with arguments ("array-style") is vectorised on DataStructure arrays, like a functional step; its
        outputs reach the MPG at the next MPG-style step or at the end of the call.
    """

    @property
    def mtg(self):
        """The MPG: within the component's MPG-style step, as it is (its declared variables were written to it just
        before); else with every tracked variable written to it first (DataStructure.mtg)."""
        data_structure = self.data_structure
        if self.__dict__.get("_in_mpg_step") and getattr(data_structure, "_mtg", None) is not None:
            return data_structure._mtg
        return data_structure.mtg

    @classmethod
    def initiate_plant(cls, mtg, plant: int, scenario: dict) -> None:
        """
        Build the initial structure of one plant under its Plant-scale vertex *plant* in the population MPG *mtg*,
        from that plant's *scenario* ({name: value}), before any component is constructed. Several structural components
        may each add their part, in order (e.g. roots, then anatomies). The planting writes the plant's position
        before any initiator builds: mtg.values_at("x"/"y"/"z"/"rotation", [plant]), which the initiator may read to place
        its initial elements; a step placing the elements later declares them (planting_variable) and takes them as
        arguments. An initiator's inputs linked from an earlier initiator are written before it builds (the plant
        model's translator, coupling.initiation).
        """
        raise NotImplementedError(f"{cls.__name__} does not initiate plants")

    def triangles(self, where=None):
        """
        The triangles of the elements (coupling.geometry.Triangles: vertices in the scene's frame, each one's node
        index), for light models and display; *where*: a node mask restricting them. By default, for elements with
        ends (x1 ... z2, topological, and radius): a slim cylinder per element of nonzero length, drawn at its display
        ends where given (display_x1 ... display_z2: coupling.geometry.segment_cylinders, the DataStructure's default).
        A component with its own shapes (blades, ...) overrides it.
        """
        from openalea.metafspm.coupling.geometry import segment_cylinders
        drawn = segment_cylinders(self.data_structure, where)
        if drawn is None:
            raise NotImplementedError(f"{type(self).__name__} has no segments (x1 … z2, radius) to triangulate: "
                                      f"its triangles() is to be defined")
        return drawn

    def geometry_stamp(self) -> tuple:
        """What the component's triangles are made from, compared by the geometric variables
        (DataStructure.derive_geometry) to know when to recompute: its calls (its clock: a call may move or reshape
        its elements) and the writes of its segment ends (x1 … z2, radius, the display ends) when it has them. A
        component whose triangles depend on more than these between its calls overrides it."""
        from openalea.metafspm.data_structure.mpg import DISPLAY_ENDS
        data_structure = self.data_structure
        names = ("x1", "y1", "z1", "x2", "y2", "z2", "radius") + DISPLAY_ENDS
        return (self.__dict__.get("_clock", 0.), tuple(data_structure.write_count(name) for name in names if data_structure.has(name)))

    def _mpg_signature(self) -> tuple:
        """Cheap topology signature of the MPG: vertex count and last allocated vertex id (the MPG read without
        flushing the tracked variables to it: the topology does not need them)."""
        data_structure = self.data_structure
        mtg = getattr(data_structure, "_mtg", None) or self.mtg
        return mtg.nb_vertices(), getattr(mtg, "_id", None)

    # Repartition of the other components' variables when the structure changes. partition_weight: a node variable name, or a callable data_structure -> array;
    # active: the rule of the DataStructure's "active" mask ({variable: condition} or a callable). Without a
    # partition weight, entities created by growth only get their on_grow values.
    partition_weight = None
    active = None
    # When the repartition runs: "after_mpg_steps", after each MPG-style step, between the weights before and after
    # it; or "end_of_call", once per call after all its steps, between partition_basis (a node variable name, or a
    # callable data_structure -> array, read after the call: the weight the entities of the call's start refer to, e.g. the
    # initial masses a growth model keeps; default: the weights at the call's start) and the weights after the call,
    # the entities absent at the call's start being the new ones. The latter suits models whose array steps change
    # the weights too (growth, root hairs), as rhizodep's post_growth_updating does
    repartition = "after_mpg_steps"
    partition_basis = None

    def __post_init__(self):
        super().__post_init__()
        self.data_structure.add_geometry_provider(self)     # data_structure.triangles(), for light models
        _warn_if_not_drawn(self)
        if self.active is not None and not self.data_structure.has_mask("active"):
            self.data_structure.define_mask("active", self.active)

    def _weights(self):
        data_structure = self.data_structure
        if self.partition_weight is None:
            return None
        values = self.partition_weight(data_structure) if callable(self.partition_weight) else data_structure.get(self.partition_weight)
        return np.array(values, dtype=float)

    def _before_call(self):
        if self.repartition not in ("after_mpg_steps", "end_of_call"):
            raise ValueError(f"{type(self).__name__}.repartition must be 'after_mpg_steps' or 'end_of_call', got "
                             f"'{self.repartition}'")
        if self.repartition != "end_of_call" or self.partition_weight is None:
            return None
        start = self.data_structure.entity_ids("node").tolist()
        return start, (self._weights() if self.partition_basis is None else None)

    def _after_call(self, context) -> None:
        super()._after_call(context)
        if context is None:
            return
        data_structure = self.data_structure
        start, start_weights = context
        ids = data_structure.entity_ids("node").tolist()
        if start_weights is not None:
            basis = dict(zip(start, start_weights.tolist()))
        else:
            values = (self.partition_basis(data_structure) if callable(self.partition_basis) else data_structure.get(self.partition_basis))
            values = dict(zip(ids, np.asarray(values, dtype=float).tolist()))
            basis = {v: values[v] for v in start if v in values}
        self._repartition(basis, {v: w > 0 for v, w in basis.items()})

    def _active_now(self, weights):
        data_structure = self.data_structure
        return np.array(data_structure.mask("active"), dtype=bool) if data_structure.has_mask("active") else weights > 0

    def _run_mpg_step(self, run) -> None:
        """Run an MPG-style step with the synchronisation of the DataStructure around it, then the repartition."""
        data_structure = self.data_structure
        declared = [spec for spec in getattr(self, "_variable_specs", {}).values()
                    if spec.mtg_backed and data_structure.has(spec.name)]
        specs = [spec for spec in declared if not held_at_scale(spec, data_structure)]
        # its inputs another component holds at another location (an architecture's SubOrgan lengths read at the
        # Compartments): the MTG holds them at their holder's location, as their holder declares them
        held = [_holder_spec(spec, data_structure) for spec in declared if held_at_scale(spec, data_structure)]
        if hasattr(data_structure, "flush_mtg"):
            # The MPG-style code reads the MTG: the variables this component declares (its states, inputs and
            # parameters) changed since their last synchronisation are written first. Others are not: a step
            # reads what its component declares
            for spec in specs + held:
                data_structure._track_mtg(spec)
            data_structure.flush_mtg([spec.name for spec in specs + held])
        else:
            for spec in specs + held:
                data_structure._write_mtg(spec)
        weights = self._weights() if self.repartition == "after_mpg_steps" else None
        if weights is not None:
            ids = data_structure.entity_ids("node").tolist()
            weight_before = dict(zip(ids, weights.tolist()))
            active_before = dict(zip(ids, self._active_now(weights).tolist()))
        before = self._mpg_signature()
        mtg = getattr(data_structure, "_mtg", None)
        states = [spec for spec in specs if spec.role == "state_variable"]
        properties = mtg.properties() if mtg is not None else {}
        versions = {spec.name: _property_version(properties.get(spec.name)) for spec in states}
        instanced = mtg is not None and getattr(mtg, "_instanced", lambda: False)()
        known = {location: data_structure.entity_ids(location) for location in ("node", "edge")} if instanced else {}
        kept = {}                           # per location: its instanced entities known before the step
        self.__dict__["_in_mpg_step"] = True
        try:
            run()
        finally:
            self.__dict__["_in_mpg_step"] = False
        # the MTG ahead of the DataStructure until the step's states are read back: a data_structure.mtg meanwhile (an
        # inheritance rule's, during the topology's update) flushes nothing over the step's writes
        with data_structure.mtg_ahead() if hasattr(data_structure, "mtg_ahead") else nullcontext():
            changed = self._mpg_signature() != before
            if changed:
                data_structure.update_topology()
                self.topology_updates = getattr(self, "topology_updates", 0) + 1
            for spec in states:
                version = _property_version(properties.get(spec.name))
                if not changed and version is not None and version == versions[spec.name]:
                    continue                         # the step left its MPG property as it was: nothing to read
                values = data_structure._read_mtg(spec)
                if values is None:
                    continue
                if instanced and spec.location in known:
                    if spec.location not in kept:
                        ids = data_structure.entity_ids(spec.location)
                        kept[spec.location] = data_structure._instanced_at(spec.location) & np.isin(ids, known[spec.location])
                    values = self._kept_at_instances(spec, values, kept[spec.location])
                data_structure.set(spec.name, values)
                if hasattr(data_structure, "_mark_mtg_synced"):
                    data_structure._mark_mtg_synced(spec.name)
        if weights is not None:
            self._repartition(weight_before, active_before)

    def _kept_at_instances(self, spec, values, known_instanced):
        """*values* read back from the MPG, except at the instanced vertices known before the step
        (*known_instanced*, over the location's entities) without an entry of their own: there the MPG holds their
        template's value, not their current one (the DataStructure's, not written back), which is kept."""
        data_structure, mtg = self.data_structure, self.data_structure._mtg
        ids = data_structure.entity_ids(spec.location)
        prop = mtg.properties().get(spec.name)
        own = prop.contains(ids) if prop is not None else np.zeros(len(ids), dtype=bool)
        keep = known_instanced & ~own
        if not keep.any():
            return values
        current = data_structure.get(spec.name)
        out = np.array(values, dtype=object) if np.asarray(current).dtype == object else \
            np.array(values, dtype=float, copy=True)
        out[keep] = np.asarray(current)[keep]
        return out.tolist() if isinstance(values, list) else out

    def _before_new_nodes(self):
        """The partition weights and active entities before a step's outputs and new nodes (the repartition's basis
        with repartition = "after_mpg_steps"), or None."""
        weights = self._weights() if self.repartition == "after_mpg_steps" else None
        if weights is None:
            return None
        ids = self.data_structure.entity_ids("node").tolist()
        return dict(zip(ids, weights.tolist())), dict(zip(ids, self._active_now(weights).tolist()))

    def _create_nodes(self, new: NewNodes, before=None) -> np.ndarray:
        """
        Create the nodes an array-style step returned (NewNodes): their complexes and themselves in the MPG, in bulk;
        the DataStructure's topology extended, their values written; then the repartition from *before*
        (_before_new_nodes). Returns their ids.
        """
        data_structure = self.data_structure
        mtg = data_structure._mtg                          # not data_structure.mtg: the MPG's properties need no synchronisation here
        n = len(new)
        if new.relabel is not None:
            ids, labels = new.relabel
            ids = np.asarray(ids, dtype=np.int64).reshape(-1)
            labels = np.broadcast_to(np.asarray(labels, dtype=object), ids.shape)
            mtg.set_values("label", ids, labels.tolist())
        if n == 0:
            if new.relabel is not None:
                data_structure._invalidate_topology()
            return np.zeros(0, dtype=np.int64)
        known = np.asarray(data_structure.entity_ids("node"), dtype=np.int64)
        # an entity view's new nodes may be borne by an entity outside it (a nodal root on a stem element)
        bearers = data_structure._bearer_ids() if isinstance(data_structure, EntityView) else known
        outer = new.parent >= 0
        root = new.parent == NewNodes.NO_PARENT
        if outer.any() and not np.isin(new.parent[outer], bearers).all():
            raise ValueError("NewNodes: a parent is not a node")
        if new.complexes is not None:
            complex_vids = self._create_complexes(mtg, new.complexes)
            complex_of = np.where(new.complex >= 0, new.complex, complex_vids[np.maximum(-1 - new.complex, 0)])
        else:
            complex_vids = np.zeros(0, dtype=np.int64)
            complex_of = self._complexes_by_levels(mtg, new, n)
        first = mtg._id + 1
        vids = np.arange(first, first + n, dtype=np.int64)
        parents = np.where(outer, new.parent, vids[np.clip(-1 - new.parent, 0, n - 1)])
        node_scale = mtg.scale(int(bearers[0])) if bearers.size else mtg.scales.SubOrgan
        topo_parents = [None if is_root else int(p) for p, is_root in zip(parents.tolist(), root.tolist())]
        created = mtg.add_components_bulk(complex_of.tolist(), n, topo_parents=topo_parents, scale=node_scale,
                                        edge_type=new.edge.tolist(), label=new.label.tolist(), isanchor=False)
        assert created[0] == first
        data_structure.update_topology()
        self.topology_updates = getattr(self, "topology_updates", 0) + 1
        # an entity view's new nodes get their values (and their complexes' values) at the location's entities, before
        # they are selected: a declaration may read them (an axis' axis_row a NewComplexes gives)
        holder = data_structure._inner if isinstance(data_structure, EntityView) else data_structure
        if new.complexes is not None and new.complexes.values:
            scale_names = declared_scale_names(mtg)
            for name, values in new.complexes.values.items():
                if not holder.has(name):
                    raise KeyError(f"NewComplexes: '{name}' is not a variable of the DataStructure")
                location = holder.location(name)
                these = np.array([scale_names.get(int(scale)) == location for scale in new.complexes.scale])
                if not these.any():
                    raise ValueError(f"NewComplexes: '{name}' is at {location}, no new complex is")
                current = np.array(holder.get(name))
                current[holder.index_of(complex_vids[these], location=location)] = np.broadcast_to(
                    np.asarray(values), (len(new.complexes),))[these]
                holder.set(name, current)
        index = holder.index_of(vids)
        for name, values in new.values.items():
            if not holder.has(name):
                raise KeyError(f"NewNodes: '{name}' is not a variable of the DataStructure")
            current = np.array(holder.get(name))
            current[index] = values
            holder.set(name, current)
        if isinstance(data_structure, EntityView):
            outside = ~np.isin(vids, data_structure.entity_ids("node"))
            if outside.any():
                raise ValueError(f"NewNodes: {int(outside.sum())} new nodes (labels "
                                 f"{sorted(set(map(str, new.label[outside].tolist())))}) are not among "
                                 f"{type(self).__name__}'s entities {self.entities!r}")
        if before is not None:
            self._repartition(*before)
        return vids

    @staticmethod
    def _create_complexes(mtg, complexes: NewComplexes) -> np.ndarray:
        """The vids of the new complexes, created in bulk, generation by generation (an anchor before its
        components)."""
        n = len(complexes)
        vids = np.full(n, -1, dtype=np.int64)
        while (vids < 0).any():
            anchors = np.where(complexes.anchor >= 0, complexes.anchor, vids[np.maximum(-1 - complexes.anchor, 0)])
            ready = (vids < 0) & (anchors >= 0)
            if not ready.any():
                raise ValueError("NewComplexes: an anchor is never created")
            index = np.flatnonzero(ready)
            vids[index] = mtg.add_components_bulk(anchors[index].tolist(), index.size, edge_type='/',
                                                scale=complexes.scale[index].tolist(),
                                                label=complexes.label[index].tolist(), isanchor=False)
        return vids

    @staticmethod
    def _complexes_by_levels(mtg, new: NewNodes, n: int) -> np.ndarray:
        """Each node's complex by new_complex / complex_levels: its parent's, or a new chain of complex_levels."""
        complex_of = np.full(n, -1, dtype=np.int64)
        # A new complex is created under an anchor (the parent's complex at the scale above complex_levels' first);
        # the k-th is numbered -2 - k until created
        anchor_scale = int(new.complex_levels[0][0]) - 1 if new.new_complex.any() else None
        anchor_of = {}                                  # a new node's anchor, when its complex is a new one
        anchors = []
        for i in range(n):
            parent = int(new.parent[i])
            if not new.new_complex[i]:
                complex_of[i] = mtg.complex(parent) if parent >= 0 else complex_of[-1 - parent]
                if parent < 0 and -1 - parent in anchor_of:
                    anchor_of[i] = anchor_of[-1 - parent]
                continue
            if parent >= 0:
                anchor = mtg.complex_at_scale(parent, anchor_scale)
            elif -1 - parent in anchor_of:
                anchor = anchor_of[-1 - parent]
            else:
                anchor = mtg.complex_at_scale(int(complex_of[-1 - parent]), anchor_scale)
            anchor_of[i] = anchor
            complex_of[i] = -2 - len(anchors)
            anchors.append(anchor)
        if anchors:
            created = anchors
            for scale, label in new.complex_levels:
                created = mtg.add_components_bulk(created, len(created), scale=scale, edge_type='/', label=label,
                                                isanchor=False)
            created = np.asarray(created, dtype=np.int64)
            pending = complex_of <= -2
            complex_of[pending] = created[-2 - complex_of[pending]]
        return complex_of

    def _repartition(self, weight_before: dict, active_before: dict) -> None:
        """
        Share the node variables of the other components between the entities of the new structure, by their
        state_variable_type (the rules of rhizodep's post_growth_updating):

          kind                      new active entity v, or one becoming active,    new inactive     existing, weight
                                    with its parent p and f = w_v / (w_v + w_p)     entity           changed
          massic_concentration      the amount c_p * b_p is split by f, 1 - f       parent's value   c * b / w
          extensive                 x_v = f x_p (+ its own when it existed),         0                unchanged
                                    x_p = (1 - f) x_p
          (NonInertial)Intensive    parent's value                                   parent's value   unchanged
          non_inertial_extensive      x_v = f x_p, the parent unchanged                0                unchanged
          descriptor or none        its on_grow value                                on_grow          unchanged

        w is the partition weight after the step and b the weight the concentration refers to: the weight before
        the step for pre-existing entities (so successive steps compose), the current weight once split. Entities
        are processed parents first, so a chain created by one segmentation splits like pairwise steps.
        """
        data_structure = self.data_structure
        if getattr(data_structure, "_anatomy", False):
            raise NotImplementedError("the repartition after structural steps is not available in anatomy mode yet: "
                                      "its lineage needs the owner vertices, not the Compartment graph")
        weights = self._weights()
        active = self._active_now(weights)
        ids = data_structure.entity_ids("node").tolist()
        known = np.array([v in weight_before for v in ids])
        basis = np.array([weight_before.get(v, 0.) for v in ids])
        was_active = np.array([bool(active_before.get(v, False)) for v in ids])
        parents = data_structure.parents()
        meta, derived, own = data_structure._variable_meta(), data_structure.derived(), set(getattr(self, "_variable_specs", {}))
        groups = {"massic": [], "extensive": [], "intensive": [], "non_inertial_extensive": []}
        for name in data_structure._node_data:
            if name in own or name in derived or name == self.partition_weight:
                continue
            kind = meta.get(name, {}).get("kind")
            group = {"massic_concentration": "massic", "extensive": "extensive", "intensive": "intensive",
                     "non_inertial_intensive": "intensive", "non_inertial_extensive": "non_inertial_extensive"}.get(kind)
            if group is not None:
                groups[group].append(name)
        if not any(groups.values()):
            return                                           # nothing to share (e.g. a growth model on its own)
        values = {name: np.array(data_structure.get(name), dtype=float) for group in groups.values() for name in group}

        for v in data_structure.order("pre"):
            p = parents[v]
            if p < 0 or (known[v] and (was_active[v] or not active[v])):
                continue                                     # a root, or an existing entity not becoming active
            if active[v]:
                f = weights[v] / (weights[v] + weights[p])
                for name in groups["massic"]:
                    amount = values[name][p] * (basis[p] if known[p] else weights[p])
                    values[name][v] = amount * f / weights[v]
                    values[name][p] = amount * (1. - f) / weights[p] if weights[p] > 0 else values[name][p]
                for name in groups["extensive"]:
                    own = values[name][v] if known[v] else 0.        # an entity becoming active keeps its own
                    values[name][v], values[name][p] = own + f * values[name][p], (1. - f) * values[name][p]
                for name in groups["non_inertial_extensive"]:
                    values[name][v] = f * values[name][p]
                basis[p], basis[v] = weights[p], weights[v]
                known[p] = True
            else:
                for name in groups["massic"] + groups["intensive"]:
                    values[name][v] = values[name][p]
                for name in groups["extensive"] + groups["non_inertial_extensive"]:
                    values[name][v] = 0.
                basis[v] = weights[v]
            known[v] = True

        # Dilution of the concentrations of existing active entities whose weight changed
        dilute = was_active & active & (basis > 0) & (weights > 0)
        for name in groups["massic"]:
            values[name][dilute] *= basis[dilute] / weights[dilute]
        for name, array in values.items():
            data_structure.set(name, array)


def _property_version(prop):
    """An MPG property's version (ArrayDict.version), None when it is not versioned (a dict: always read); 0 when
    absent."""
    if prop is None:
        return 0
    return getattr(prop, "version", None)


def _warn_if_not_drawn(component) -> None:
    """
    Warn when *component*'s geometry (triangles(), registered on the DataStructure it works on) is on a view of an
    anatomy-mode DataStructure at a location the frames and the light models never draw: they take the triangles of
    the view at its from_scale (data_structure.at("SubOrgan"), the elements), a component working at a coarser scale (its finest
    declaration at Organ, say: data_structure.at("Organ")) registering on that scale's view instead. Its triangles are owned by
    that scale's entities, which no element's mesh row or absorbed light can be given back to: they are not routed
    to the drawn view. The Compartments (the anatomy-mode DataStructure itself, an anatomy's component) are not
    drawn by design: no warning.
    """
    data_structure = component.data_structure
    located = data_structure.__dict__.get("_inner", data_structure) if isinstance(data_structure, EntityView) else data_structure
    base = located.__dict__.get("_base")
    location = located.__dict__.get("_at")
    if location is None or not isinstance(base, MPGDataStructure) or not base._anatomy:
        return
    drawn = base._scale_name(base.from_scale)
    if location != drawn:
        import warnings
        warnings.warn(f"{type(component).__name__} works at the {location} scale of an anatomy-mode DataStructure: "
                      f"its geometry is registered on data_structure.at({location!r}), which the frames and the light models never "
                      f"draw (they take data_structure.at({drawn!r}).triangles()); declare one of its variables at the {drawn} "
                      f"scale (or its entities) for its elements to be drawn and lit", stacklevel=4)


def _working_data_structure(component, data_structure):
    """
    The DataStructure a component works on: *data_structure*, or on an anatomy-mode DataStructure (its nodes the Compartments)
    the view at the component's own scale (data_structure.LocatedView) when the finest scale of its declarations is
    coarser (a SubOrgan-scale architecture, a light or gas-exchange model): its nodes then those of its scale. A
    component declaring its `entities` works on the view restricted to them (data_structure.EntityView), at the nodes
    in segment mode, in anatomy mode at the DataStructure's from_scale (its SubOrgans: the entities segment mode's
    nodes are, whatever coarser scale its declarations' finest is, a Phytomer-scale rule's say), or at its own scale
    when finer.
    """
    entities = getattr(type(component), "entities", None)
    if entities is not None:
        base = data_structure.__dict__.get("_base", data_structure)
        if not isinstance(base, MPGDataStructure):
            raise TypeError(f"{type(component).__name__}.entities: an entity view is of a plant DataStructure "
                            f"(MPGDataStructure), not of a {type(base).__name__}")
        if not base._anatomy:
            return base.entity_view(entities)
        name = _own_scale_name(component, base)
        if name is None:
            raise NotImplementedError(f"{type(component).__name__}.entities on an anatomy-mode DataStructure: the "
                                      "component works on its Compartments; entities are declared by a component "
                                      "of the SubOrgan scale or coarser")
        mtg = base.structure
        if getattr(mtg.scales, name) < base.from_scale:
            name = base._scale_name(base.from_scale)        # its entities the from_scale's, as segment mode's nodes
        return base.entity_view(entities, at=name)
    if not getattr(data_structure, "_anatomy", False) or not hasattr(data_structure, "at"):
        return data_structure
    name = _own_scale_name(component, data_structure)
    return data_structure if name is None else data_structure.at(name)


def _own_scale_name(component, data_structure):
    """The name of the finest scale of *component*'s declarations on an anatomy-mode *data_structure* when it is coarser than
    its Compartments (and finer than the plants), else None (it works on the nodes)."""
    mtg = data_structure.structure
    finest = None
    for f in fields(component):
        meta = f.metadata
        if meta.get("role") not in ("state_variable", "input_variable", "parameter"):
            continue
        scale = meta.get("scale")
        if meta.get("location") in ("node", "edge") or (scale is not None and scale >= mtg.scales.Compartment):
            return None                                     # it works on the nodes (the cells)
        if scale is not None and scale > mtg.scales.Plant:
            finest = scale if finest is None else max(finest, scale)
    if finest is None:
        return None
    return next(n for n, v in vars(type(mtg.scales)).items() if isinstance(v, int) and not n.startswith("_")
                and v == finest)
