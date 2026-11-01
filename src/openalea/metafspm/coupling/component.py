from dataclasses import dataclass, field, fields, MISSING
from typing import Literal, Optional
import numpy as np

from openalea.metafspm.solve.decorator import *
from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.data_structure.data_api import DataStructure, MPGDataStructure, GraphDataStructure
from openalea.metafspm.data_structure.configs import ScalesConfig as _ScalesConfig
from openalea.metafspm.coupling.declaration import declared_specs, DeclarationError
from openalea.metafspm.coupling.declaration import scale_names as declared_scale_names

# Map integer MPG scale constants to the generic "node"/"edge" vocabulary used
# by GraphDataStructure.  Compartment nodes are graph nodes; Connection edges
# are graph edges.
_SCALE_INT_TO_LOC: dict = {
    _ScalesConfig.Compartment: "node",
    _ScalesConfig.Connection: "edge",
}


_RENAMED = {"initialize": "default", "state_variable_type": "kind", "mapping": "aggregation"}
DECLARATION_COLUMNS = ["name", "role", "kind", "scale", "unit", "description", "default", "by", "min_value",
                       "max_value", "references", "DOI", "unit_comment", "value_comment"]


def _renamed(function: str, given: dict) -> dict:
    """The former keyword names of the declarations (initialize, state_variable_type, mapping), accepted with a
    warning for one release; any other unknown keyword raises."""
    import warnings
    found = {}
    for name, value in given.items():
        if name not in _RENAMED:
            raise TypeError(f"{function}() got an unexpected keyword argument '{name}'")
        warnings.warn(f"{function}({name}=...) is renamed {_RENAMED[name]}=", DeprecationWarning, stacklevel=3)
        found[_RENAMED[name]] = value
    return found


def _declare(role: str, unit: str = "", description: str = "", default=None, default_factory=None, kind=None,
             by: str = None, scale=None, location=None, aggregation=None, weight=None, on_grow="default", dtype=None,
             shape=None, min_value=None, max_value=None, unit_comment: str = "", value_comment: str = "",
             references: str = "", DOI=None, **extra):
    """
    The metadata of a declared field, shared by state_variable, input_variable, parameter, setting and
    forcing_variable (see them for the arguments). *role*: "state_variable", "input", "parameter", "setting" or
    "forcing"; *extra*: the role's own keys (a setting's choices, a forcing's required and at).
    """
    if aggregation in ("proximal", "distal"):
        raise ValueError(f"aggregation='{aggregation}' is 'child' or 'parent' (the value of the edge's child or "
                         "parent node)")
    metadata = dict(
        unit=unit, unit_comment=unit_comment, description=description, min_value=min_value, max_value=max_value,
        value_comment=value_comment, references=references, DOI=list(DOI or []), variable_type=role, by=by,
        kind=kind, scale=scale, on_grow=on_grow, location=location, mapping=aggregation, weight=weight,
        dtype=dtype, shape=tuple(shape) if shape is not None else None, **extra)
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

_KINDS = """
    *kind*: how its value depends on the size of what it describes: "extensive" (an amount, summed up),
    "intensive" (a concentration, a potential, averaged up by *weight*), "massic_concentration", "descriptor",
    "NonInertialExtensive", "NonInertialIntensive"; links between kinds that disagree raise."""


def state_variable(unit: str = "", description: str = "", default=None, kind=None, scale=None, location=None,
                   aggregation=None, weight=None, on_grow="default", dtype=None, shape=None, by: str = None,
                   min_value=None, max_value=None, unit_comment: str = "", value_comment: str = "",
                   references: str = "", DOI=None, **renamed):
    return _declare("state_variable", **_merge(locals(), "state_variable"))


def input_variable(unit: str = "", description: str = "", default=None, kind=None, by: str = None, scale=None,
                   location=None, aggregation=None, weight=None, on_grow="default", dtype=None, shape=None,
                   min_value=None, max_value=None, unit_comment: str = "", value_comment: str = "",
                   references: str = "", DOI=None, **renamed):
    return _declare("input", **_merge(locals(), "input_variable"))


def parameter(unit: str = "", description: str = "", default=None, kind=None, by: str = None, scale=None,
              location=None, aggregation=None, weight=None, on_grow="default", dtype=None, min_value=None,
              max_value=None, unit_comment: str = "", value_comment: str = "", references: str = "", DOI=None,
              **renamed):
    return _declare("parameter", **_merge(locals(), "parameter"))


def _merge(arguments: dict, function: str) -> dict:
    arguments = dict(arguments)
    renamed = _renamed(function, arguments.pop("renamed"))
    for name, value in renamed.items():
        if arguments.get(name) is not None:
            raise TypeError(f"{function}(): give {name}= only (its former name is also given)")
        arguments[name] = value
    return arguments


state_variable.__doc__ = """Declare a state variable, computed by this component's steps or solves. *default*: its
    initial value (a number, the same everywhere).""" + _KINDS + _PLACE + _DOCUMENTATION
input_variable.__doc__ = """Declare an input: a variable another component provides (an input named as another
    component's variable of the same DataStructure reads it; a translator link states a rename, a factor or a change
    of scale). *default*: its value when nothing provides it. *by*: the provider, for the documentation. *kind*:
    optional, checked against the provider's.""" + _PLACE + _DOCUMENTATION
parameter.__doc__ = """Declare a parameter: a constant of the equations, set by constructor keywords and
    scenarios. A numeric parameter without a place is stored per plant (each plant's scenario may change it); with
    *scale* or *location*, one value per entity. *by*: its owner, for the documentation.""" + _KINDS + _PLACE \
    + _DOCUMENTATION


def setting(unit: str = "", description: str = "", default=None, choices=None, unit_comment: str = "",
            references: str = "", DOI=None):
    """Declare a setting: a value of the model, not of the plants (a latitude, a light model, a solver, a grid's
    voxel), one for all. Listed by declarations() (role "setting"), set by constructor keywords and scenarios, read
    as `self.name` anywhere (steps and equations included), and passed as the value itself to a step naming it as an
    argument. Never stored on the DataStructure (numeric parameters are, per plant). *choices*: the values allowed,
    checked at construction. A default must be immutable (a tuple, not a list)."""
    return _declare("setting", unit=unit, description=description, default=default, unit_comment=unit_comment,
                    references=references, DOI=DOI, choices=tuple(choices) if choices is not None else None)


def forcing_variable(unit: str = "", description: str = "", required: bool = True, at: str = "end",
                     unit_comment: str = "", references: str = "", DOI=None):
    """Declare a forcing: a time series given to the scene (its `forcings` table, shared by every model), read by
    `self.forcing(name)` at the step's time, the same for every entity (meteo, an imposed supply). Not a
    DataStructure variable: listed by declarations(). A Scene raises at its construction when a *required* forcing
    is not given; an optional one (required=False) raises KeyError when read, for the component to fall back. *at*:
    when a step reads it, "end" (the end of the (sub-)step, metafspm's convention for implicit solves) or "start"
    (the step's start, e.g. a meteo table read as the hour begins)."""
    if at not in ("start", "end"):
        raise ValueError(f"forcing_variable(at=...): 'start' or 'end', got '{at}'")
    return _declare("forcing", unit=unit, description=description, default=None, unit_comment=unit_comment,
                    references=references, DOI=DOI, required=required, at=at)


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

    def __call__(self):
        self.pull_available_inputs()
        self.choregrapher(instance=self)
        # State variables reach the MTG after every call with mtg_sync = "after_call"; by default ("lazy") the
        # DataStructure writes them when the MTG is read
        if hasattr(self, "mtg_sync"):
            self._check_mtg_sync()
            if self.mtg_sync == "after_call":
                self.write_back_to_mtg()
        # The component's clock (forcings): the start of its next call
        self.__dict__["_clock"] = self.__dict__.get("_clock", 0.) + float(
            self.choregrapher.simulation_step() or self.time_step or 0.)

    def write_back_to_mtg(self) -> None:
        """Write the component's state variables to the MTG; FunctionalComponent implements it."""
        pass

    @classmethod
    def declared_parameters(cls) -> tuple:
        """The names of the parameters the class declares (parameter)."""
        return tuple(f.name for f in fields(cls) if f.metadata.get("variable_type") == "parameter")

    def pull_available_inputs(self):
        """Refresh the inputs derived by the coupling; FunctionalComponent implements it on its DataStructure."""
        pass

    @classmethod
    def declared_settings(cls) -> tuple:
        """The names of the settings the class declares (setting), read once per class."""
        if "_setting_names" not in cls.__dict__:
            cls._setting_names = tuple(f.name for f in fields(cls) if f.metadata.get("variable_type") == "setting")
        return cls._setting_names

    def check_settings(self) -> None:
        """Each setting with choices holds one of them."""
        for f in fields(self):
            choices = f.metadata.get("choices") if f.metadata.get("variable_type") == "setting" else None
            if choices is not None and getattr(self, f.name) not in choices:
                raise ValueError(f"{type(self).__name__}.{f.name}: one of {list(choices)}, got "
                                 f"{getattr(self, f.name)!r}")

    @classmethod
    def declared_forcings(cls) -> dict:
        """{name: required} of the forcings the class declares (forcing_variable)."""
        return {f.name: f.metadata.get("required", True) for f in fields(cls)
                if f.metadata.get("variable_type") == "forcing"}

    @classmethod
    def declarations(cls):
        """
        A pandas DataFrame of the declared variables, one row each: name; role (state_variable, input, parameter,
        setting, forcing); kind (extensive, intensive, ...: how it depends on size, None when not declared); scale
        (its name) or location; unit, description, default ("required" or "optional" for a forcing); by (an input's
        provider, a parameter's owner); min_value, max_value (the bounds of its values, None when not declared);
        references, DOI, unit_comment, value_comment.
        """
        import pandas as pd
        scale_names = declared_scale_names()
        rows = []
        for f in fields(cls):
            meta = f.metadata
            role = meta.get("variable_type")
            if role is None:
                continue
            scale = meta.get("scale")
            if role == "setting":                           # one value of the model
                place = "model"
            elif role == "forcing":                         # a time series, the same for every entity
                place = "time series"
            else:
                place = scale_names.get(scale, scale) if scale is not None else meta.get("location")
            default = f.default if f.default is not MISSING else None
            if role == "forcing":
                default = "required" if meta.get("required", True) else "optional"
            rows.append(dict(name=f.name, role=role, kind=meta.get("kind"), scale=place,
                             unit=meta.get("unit"), description=meta.get("description"), default=default,
                             by=meta.get("by"), min_value=_bound(meta.get("min_value")),
                             max_value=_bound(meta.get("max_value")), references=meta.get("references") or "",
                             DOI=list(meta.get("DOI") or []), unit_comment=meta.get("unit_comment") or "",
                             value_comment=meta.get("value_comment") or ""))
        table = pd.DataFrame(rows, columns=DECLARATION_COLUMNS)
        for column in ("kind", "min_value", "max_value"):
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


@dataclass
class DataStructureComponent(Component):
    """
    Component bound to a DataStructure (base of FunctionalComponent and StructuralComponent).

    Its declared fields are resolved once (resolve_declaration) and registered on the DataStructure when missing,
    from their MTG property or their default; its state variables are written to the MTG after every call, and its
    MTG-backed parameters are re-read before each graph solve.
    """

    data_structure: Optional[DataStructure] = None
    # Time step of the component (s): the dt of its equations, and its sub-step: it runs (the scene's time step) /
    # time_step times per scene step, once when run alone. Default: its class's time_step when it sets one, else the
    # scene's
    time_step: Optional[float] = None

    # MTG synchronisation policy: "lazy" (default) writes the state variables
    # changed since the last synchronisation when the MTG is read (ds.mtg, MPG-style steps, ds.flush_mtg());
    # "after_call" writes them after every call; "never" leaves the MTG untouched.
    mtg_sync = "lazy"

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
        ds = self.data_structure
        self._auto_declare_on_ds(ds)
        if not hasattr(self, "pullable_inputs"):
            self.pullable_inputs = {}
        simulation_step = self.choregrapher.simulation_step()
        if self.time_step is None:
            declared = getattr(type(self), "time_step", None)                 # a class attribute of a subclass
            self.time_step = declared if declared is not None else simulation_step
        # The component runs once per time_step: simulation step / time_step times per simulation step; once per
        # call alone, or without a time step (then its equations may not read self.dt)
        if simulation_step is not None and self.time_step is not None:
            from openalea.metafspm.coupling.choregrapher import sub_steps
            sub_steps(simulation_step, self.time_step, type(self).__name__)
        # Live reading: steps and solves read and write the DataStructure arrays
        self.choregrapher.bind(self, self.time_step, ds)

    def pull_available_inputs(self):
        """
        Before the step: re-read the MTG-backed parameters, and bring the inputs derived by the coupling up
        to date. Derived variables are recomputed when read, so the latter only recomputes those whose sources
        changed; it is the hook of future sub-steps.
        """
        self._refresh_from_bio_scale()
        ds = self.data_structure
        for name in getattr(self, "_derived_inputs", []):
            ds.get(name)

    def _auto_declare_on_ds(self, ds: DataStructure) -> None:
        """
        Register the declared variables that are not yet on *ds*.

        Each field is resolved once by resolve_declaration into a VariableSpec (location, MTG scale, mapping),
        kept in self._variable_specs. A variable is registered at its location with the values of its MTG
        property when the property exists (mapped from its scale), else with its declared default.
        Pre-registered variables are never overwritten; their location must match the declaration.
        """
        self._variable_specs = declared_specs(self, ds)
        for name, spec in self._variable_specs.items():
            pending = self.__dict__.get("_pending_parameters", {})
            if ds.has(name):
                registered = ds.location(name)
                if registered != spec.location and name not in ds.aliases():
                    raise DeclarationError(f"{type(self).__name__}.{name} is declared at {spec.location} but is "
                                           f"already registered at {registered}")
                if name in pending and pending[name] != spec.default:
                    ds.set(name, pending[name])          # a value given to the constructor
            else:
                values = ds._read_mtg(spec) if hasattr(ds, "_read_mtg") else None
                default = pending.get(name, spec.default) if spec.dtype is not object else spec.default
                ds.register(name, values, location=spec.location, default=default, on_grow=spec.on_grow,
                            dtype=spec.dtype, shape=spec.shape)
            # Metadata precedence: the component that owns the variable (not an input) sets its default, growth
            # policy and kind; a component reading it as an input only fills what is still unknown.
            meta = ds._variable_meta().setdefault(name, {})
            declared = {key: value for key, value in spec.meta().items() if value is not None}
            if spec.variable_type == "input":
                for key, value in declared.items():
                    meta.setdefault(key, value)
            else:
                meta.update(declared, default=spec.default, on_grow=spec.on_grow)
        self.__dict__["_registered_parameters"] = {name for name in getattr(type(self), "_plant_parameter_names", ())
                                                   if name in self._variable_specs}
        self.__dict__.pop("_pending_parameters", None)
        self._check_mtg_sync()
        if self.mtg_sync == "lazy" and hasattr(ds, "_track_mtg"):
            for spec in self._variable_specs.values():
                if spec.variable_type == "state_variable" and spec.mtg_backed:
                    ds._track_mtg(spec)

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
            if (meta.get("variable_type") == "parameter"
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
        rebuilt when missing (a kept canopy, incremental meshes), left out of checkpoints. Derived values rebuilt
        on a key are `cached`'s."""
        self.__dict__.get("_kept_unsaved" if saved else "_kept", {}).pop(name, None)
        store = self.__dict__.setdefault("_kept" if saved else "_kept_unsaved", {})
        store[name] = value

    def kept(self, name: str, default=None):
        """What keep() kept under *name* (*default* when nothing is)."""
        for store in ("_kept", "_kept_unsaved"):
            values = self.__dict__.get(store, {})
            if name in values:
                return values[name]
        return default

    def checkpoint_state(self) -> dict:
        """The state checkpoints save: what keep() kept."""
        return {"kept": dict(self.__dict__.get("_kept", {}))}

    def restore_state(self, state: dict) -> None:
        self.__dict__["_kept"] = dict(state.get("kept", {}))

    def cached(self, name: str, build, key=None):
        """A value kept on the component: build() at the first call and again when *key* changes (by default the
        DataStructure's topology version: values read from the structure)."""
        key = self.data_structure.topology_version if key is None else key
        store = self.__dict__.setdefault("_cached_values", {})
        entry = store.get(name)
        if entry is None or entry[0] != key:
            entry = (key, build())
            store[name] = entry
        return entry[1]

    def parameter_values(self, name: str, location: str = "node") -> np.ndarray:
        """
        Per-entity values of a per-plant parameter, for code that cannot take it as an argument (MPG-style structural
        steps): each node (or edge) gets its plant's value.
        """
        return self.data_structure.parameter_view(name, location)

    # Seed of this component's random streams; a scenario may set it
    random_seed = 0

    def random(self, stream: str, distribution: str = "uniform", ids=None, location: str = "node",
               **parameters) -> np.ndarray:
        """
        Reproducible draws per entity for stream *stream*: each call of a stream is a new step, so a model drawing in
        the same order gets the same draws at every run, whatever the visiting order (ds.random).
        """
        name = f"{type(self).__name__}.{stream}"
        counters = self.data_structure.__dict__.setdefault("_random_steps", {})   # kept by checkpoints
        step = counters.get(name, 0)
        counters[name] = step + 1
        return self.data_structure.random(distribution, stream=name, step=step, ids=ids, location=location,
                                          seed=int(self.random_seed), **parameters)

    def active_ids(self) -> np.ndarray:
        """
        Node ids selected by the DataStructure's "active" mask (every node without one), for MPG-style steps, which
        loop over vertices themselves: e.g. plants before emergence are skipped.
        """
        ds = self.data_structure
        ids = np.asarray(ds.entity_ids("node"))
        return ids[np.asarray(ds.mask("active"), dtype=bool)] if ds.has_mask("active") else ids

    def write_back_to_mtg(self) -> None:
        """
        Write the declared state variables with an MTG scale to the MTG, at the vertices of their scale, through the
        inverse of their mapping (MPGDataStructure._write_mtg). Called after every component call with
        mtg_sync = "after_call"; with "lazy" the DataStructure does it when the MTG is read.

        Only ``state_variable`` fields are written: parameters and inputs are owned by whoever sets them, and
        writing their defaults back would overwrite externally set MTG values.
        """
        self._check_mtg_sync()
        ds = self.data_structure
        if self.mtg_sync == "never" or not hasattr(ds, "_write_mtg"):
            return
        for spec in getattr(self, "_variable_specs", {}).values():
            if spec.variable_type == "state_variable" and spec.mtg_backed and ds.has(spec.name):
                ds._write_mtg(spec)

    def _refresh_from_bio_scale(self) -> None:
        """
        Re-read the declared parameters with an MTG scale from the MTG, before each graph-system solve, so that
        parameters updated on the MTG by other components (e.g. K_axial by a growth model) are seen.

        State variables and inputs are not refreshed: their DataStructure values are set by steps and couplings.
        Parameters whose MTG property is absent keep their DataStructure values.
        """
        ds = self.data_structure
        if not hasattr(ds, "_read_mtg"):
            return
        for spec in getattr(self, "_variable_specs", {}).values():
            if spec.variable_type != "parameter" or not spec.mtg_backed:
                continue
            if hasattr(ds, "flush_mtg"):
                ds.flush_mtg([spec.name])            # a value set on the DataStructure is not lost to a stale MTG one
            values = ds._read_mtg(spec)
            if values is not None:
                ds.set(spec.name, values)


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
      scale appear in ds with their default values automatically.
    * When non-default values are needed (tests, scenario setup), call
      ds.register(name, values, location=...) BEFORE constructing the
      model.  Pre-registered values are never overwritten.

    Usage::

        ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
        ds.register("concentration", c_init, location="node")   # non-default — must pre-set
        model = MyTransportModel(data_structure=ds)
        # ds now also contains default arrays for K_axial, volumetric_capacity, etc.
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
        ds = self.data_structure
        if not hasattr(ds, "to_graph_view"):
            return None
        version = getattr(ds, "topology_version", None)
        if "_graph_view_cache" not in self.__dict__ or self.__dict__.get("_graph_view_version") != version:
            self.__dict__["_graph_view_cache"] = ds.to_graph_view()
            self.__dict__["_graph_view_version"] = version
        return self.__dict__["_graph_view_cache"]

    @_graph_view.setter
    def _graph_view(self, view):
        # Explicit views (e.g. with boundary ports built by hand) are kept until the topology changes
        self.__dict__["_graph_view_cache"] = view
        self.__dict__["_graph_view_version"] = getattr(self.data_structure, "topology_version", None)

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
            step = float(getattr(self, "time_step", None) or self.choregrapher.simulation_step() or 0.)
        return start + offset + step

    def forcing(self, name: str, at: str = None):
        """
        Forcing *name* at forcing_time() (coupling.forcings): a number, a callable of the time, or a table linearly
        interpolated, as a Series indexed by time or a (times, values) pair. *at*: "end" or "start" of the step (by
        default its declaration's, forcing_variable(at=...), else "end").
        """
        if at is None:
            declared = getattr(type(self), "__dataclass_fields__", {}).get(name)
            at = declared.metadata.get("at", "end") if declared is not None else "end"
        forcings = self.__dict__.get("_scene_forcings") or {}
        if name not in forcings:
            raise KeyError(f"{type(self).__name__} has no forcing '{name}' (the Scene's forcings=)")
        from openalea.metafspm.coupling.forcings import forcing_value, is_spatial
        if is_spatial(forcings[name]):
            raise TypeError(f"forcing '{name}' varies in space ((time, ids) -> values): the Scene writes it into an "
                            "input or parameter, a component does not read it with forcing()")
        return forcing_value(forcings[name], self.forcing_time(at))

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
    def dt(self) -> float:
        """Length of the current (sub-)step of a graph-system solve; the time step outside sub-stepping."""
        step = self.__dict__.get("_current_dt", getattr(self, "time_step", None))
        if step is None:
            raise AttributeError(f"{type(self).__name__} has no time step: give its time_step= (s), or build it in a "
                                 "Scene (its time_step=)")
        return step

    def previous(self, name: str, at: str = "substep") -> np.ndarray:
        """
        Value of unknown *name*, managed by the framework::

              at="substep" (default): at the start of the current (sub-)step of the solve;
              at="solve":             at the start of this call's solve (the same with integrate="step");
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
        ds = self.data_structure
        unknowns = {fn: location for spec in getattr(type(self), "_graph_system_specs", {}).values()
                    for location, names in (("node", spec["node_unknowns"]), ("edge", spec["edge_unknowns"]))
                    for fn in names}
        self.__dict__["_step_start_state"] = {fn: (location, np.array(ds.get(fn), dtype=np.float64).reshape(-1))
                                              for fn, location in unknowns.items() if ds.has(fn)}


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
        return self.data_structure.mtg

    @classmethod
    def initiate_plant(cls, g, plant: int, scenario: dict) -> None:
        """
        Build the initial structure of one plant under its Plant-scale vertex *plant* in the population MPG *g*,
        from that plant's *scenario* ({name: value}), before any component is constructed. Several structural components
        may each add their part, in order (e.g. roots, then anatomies). The plant's position is in
        g.property("x"/"y"/"z"/"rotation")[plant]; the component computes its elements' coordinates from it.
        """
        raise NotImplementedError(f"{cls.__name__} does not initiate plants")

    def triangles(self, where=None):
        """
        The triangles of the elements (coupling.geometry.Triangles: vertices in the scene's frame, each one's node
        index), for light models and display; *where*: a node mask restricting them. By default, for segment models
        (variables x1, y1, z1, x2, y2, z2 and radius on the elements): a slim cylinder per segment of nonzero length.
        A component with its own shapes (blades, ...) overrides it.
        """
        from openalea.metafspm.coupling.geometry import Triangles, segment_triangles
        ds = self.data_structure
        names = ("x1", "y1", "z1", "x2", "y2", "z2", "radius")
        if not all(ds.has(name) for name in names):
            raise NotImplementedError(f"{type(self).__name__} has no segments (x1 … z2, radius) to triangulate: "
                                      f"its triangles() is to be defined")
        x1, y1, z1, x2, y2, z2, radius = (np.asarray(ds.get(name), dtype=float) for name in names)
        nodes = np.arange(x1.size) if where is None else np.flatnonzero(where)
        start = np.stack([x1[nodes], y1[nodes], z1[nodes]], axis=1)
        end = np.stack([x2[nodes], y2[nodes], z2[nodes]], axis=1)
        vertices, drawn = segment_triangles(start, end, radius[nodes])
        return Triangles(vertices[drawn].reshape(-1, 3, 3), np.repeat(nodes[drawn], vertices.shape[1]))

    def _mpg_signature(self) -> tuple:
        """Cheap topology signature of the MPG: vertex count and last allocated vertex id."""
        g = self.mtg
        return g.nb_vertices(), getattr(g, "_id", None)

    # Repartition of the other components' variables when the structure changes. partition_weight: a node variable name, or a callable ds -> array;
    # active: the rule of the DataStructure's "active" mask ({variable: condition} or a callable). Without a
    # partition weight, entities created by growth only get their on_grow values.
    partition_weight = None
    active = None
    # When the repartition runs: "after_mpg_steps", after each MPG-style step, between the weights before and after
    # it; or "end_of_call", once per call after all its steps, between partition_basis (a node variable name, or a
    # callable ds -> array, read after the call: the weight the entities of the call's start refer to, e.g. the
    # initial masses a growth model keeps; default: the weights at the call's start) and the weights after the call,
    # the entities absent at the call's start being the new ones. The latter suits models whose array steps change
    # the weights too (growth, root hairs), as rhizodep's post_growth_updating does
    repartition = "after_mpg_steps"
    partition_basis = None

    def __post_init__(self):
        super().__post_init__()
        self.data_structure.add_geometry_provider(self)     # ds.triangles(), for light models
        if self.active is not None and not self.data_structure.has_mask("active"):
            self.data_structure.define_mask("active", self.active)

    def _weights(self):
        ds = self.data_structure
        if self.partition_weight is None:
            return None
        values = self.partition_weight(ds) if callable(self.partition_weight) else ds.get(self.partition_weight)
        return np.array(values, dtype=float)

    def __call__(self):
        if self.repartition not in ("after_mpg_steps", "end_of_call"):
            raise ValueError(f"{type(self).__name__}.repartition must be 'after_mpg_steps' or 'end_of_call', got "
                             f"'{self.repartition}'")
        if self.repartition != "end_of_call" or self.partition_weight is None:
            return super().__call__()
        ds = self.data_structure
        start = ds.entity_ids("node").tolist()
        start_weights = self._weights() if self.partition_basis is None else None
        super().__call__()
        ids = ds.entity_ids("node").tolist()
        if start_weights is not None:
            basis = dict(zip(start, start_weights.tolist()))
        else:
            values = (self.partition_basis(ds) if callable(self.partition_basis) else ds.get(self.partition_basis))
            values = dict(zip(ids, np.asarray(values, dtype=float).tolist()))
            basis = {v: values[v] for v in start if v in values}
        self._repartition(basis, {v: w > 0 for v, w in basis.items()})

    def _active_now(self, weights):
        ds = self.data_structure
        return np.array(ds.mask("active"), dtype=bool) if ds.has_mask("active") else weights > 0

    def _run_mpg_step(self, run) -> None:
        """Run an MPG-style step with the synchronisation of the DataStructure around it, then the repartition."""
        ds = self.data_structure
        specs = [spec for spec in getattr(self, "_variable_specs", {}).values() if spec.mtg_backed and ds.has(spec.name)]
        if hasattr(ds, "flush_mtg"):
            # The MPG-style code reads the MTG: the variables this component declares (its states, inputs and
            # parameters) changed since their last synchronisation are written first. Others are not: a step
            # reads what its component declares
            for spec in specs:
                ds._track_mtg(spec)
            ds.flush_mtg([spec.name for spec in specs])
        else:
            for spec in specs:
                ds._write_mtg(spec)
        weights = self._weights() if self.repartition == "after_mpg_steps" else None
        if weights is not None:
            ids = ds.entity_ids("node").tolist()
            weight_before = dict(zip(ids, weights.tolist()))
            active_before = dict(zip(ids, self._active_now(weights).tolist()))
        before = self._mpg_signature()
        run()
        if self._mpg_signature() != before:
            ds.update_topology()
            self.topology_updates = getattr(self, "topology_updates", 0) + 1
        for spec in specs:
            if spec.variable_type == "state_variable":
                values = ds._read_mtg(spec)
                if values is not None:
                    ds.set(spec.name, values)
                    if hasattr(ds, "_mark_mtg_synced"):
                        ds._mark_mtg_synced(spec.name)
        if weights is not None:
            self._repartition(weight_before, active_before)

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
        ds = self.data_structure
        g = ds._mtg                          # not ds.mtg: the MPG's properties need no synchronisation here
        n = len(new)
        if new.relabel is not None:
            ids, labels = new.relabel
            ids = np.asarray(ids, dtype=np.int64).reshape(-1)
            labels = np.broadcast_to(np.asarray(labels, dtype=object), ids.shape)
            label = g.property("label")
            for vid, value in zip(ids.tolist(), labels.tolist()):
                label[vid] = value
        if n == 0:
            if new.relabel is not None:
                ds._invalidate_topology()
            return np.zeros(0, dtype=np.int64)
        known = np.asarray(ds.entity_ids("node"), dtype=np.int64)
        outer = new.parent >= 0
        root = new.parent == NewNodes.NO_PARENT
        if outer.any() and not np.isin(new.parent[outer], known).all():
            raise ValueError("NewNodes: a parent is not a node")
        if new.complexes is not None:
            complex_vids = self._create_complexes(g, new.complexes)
            complex_of = np.where(new.complex >= 0, new.complex, complex_vids[np.maximum(-1 - new.complex, 0)])
        else:
            complex_vids = np.zeros(0, dtype=np.int64)
            complex_of = self._complexes_by_levels(g, new, n)
        first = g._id + 1
        vids = np.arange(first, first + n, dtype=np.int64)
        parents = np.where(outer, new.parent, vids[np.clip(-1 - new.parent, 0, n - 1)])
        node_scale = g.scale(int(known[0])) if known.size else g.scales.SubOrgan
        topo_parents = [None if is_root else int(p) for p, is_root in zip(parents.tolist(), root.tolist())]
        created = g.add_components_bulk(complex_of.tolist(), n, topo_parents=topo_parents, scale=node_scale,
                                        edge_type=new.edge.tolist(), label=new.label.tolist(), isanchor=False)
        assert created[0] == first
        ds.update_topology()
        self.topology_updates = getattr(self, "topology_updates", 0) + 1
        index = ds.index_of(vids)
        for name, values in new.values.items():
            if not ds.has(name):
                raise KeyError(f"NewNodes: '{name}' is not a variable of the DataStructure")
            current = np.array(ds.get(name))
            current[index] = values
            ds.set(name, current)
        if new.complexes is not None and new.complexes.values:
            scale_names = declared_scale_names(g)
            for name, values in new.complexes.values.items():
                if not ds.has(name):
                    raise KeyError(f"NewComplexes: '{name}' is not a variable of the DataStructure")
                location = ds.location(name)
                these = np.array([scale_names.get(int(scale)) == location for scale in new.complexes.scale])
                if not these.any():
                    raise ValueError(f"NewComplexes: '{name}' is at {location}, no new complex is")
                current = np.array(ds.get(name))
                current[ds.index_of(complex_vids[these], location=location)] = np.broadcast_to(
                    np.asarray(values), (len(new.complexes),))[these]
                ds.set(name, current)
        if before is not None:
            self._repartition(*before)
        return vids

    @staticmethod
    def _create_complexes(g, complexes: NewComplexes) -> np.ndarray:
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
            vids[index] = g.add_components_bulk(anchors[index].tolist(), index.size, edge_type='/',
                                                scale=complexes.scale[index].tolist(),
                                                label=complexes.label[index].tolist(), isanchor=False)
        return vids

    @staticmethod
    def _complexes_by_levels(g, new: NewNodes, n: int) -> np.ndarray:
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
                complex_of[i] = g.complex(parent) if parent >= 0 else complex_of[-1 - parent]
                if parent < 0 and -1 - parent in anchor_of:
                    anchor_of[i] = anchor_of[-1 - parent]
                continue
            if parent >= 0:
                anchor = g.complex_at_scale(parent, anchor_scale)
            elif -1 - parent in anchor_of:
                anchor = anchor_of[-1 - parent]
            else:
                anchor = g.complex_at_scale(int(complex_of[-1 - parent]), anchor_scale)
            anchor_of[i] = anchor
            complex_of[i] = -2 - len(anchors)
            anchors.append(anchor)
        if anchors:
            created = anchors
            for scale, label in new.complex_levels:
                created = g.add_components_bulk(created, len(created), scale=scale, edge_type='/', label=label,
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
          NonInertialExtensive      x_v = f x_p, the parent unchanged                0                unchanged
          descriptor or none        its on_grow value                                on_grow          unchanged

        w is the partition weight after the step and b the weight the concentration refers to: the weight before
        the step for pre-existing entities (so successive steps compose), the current weight once split. Entities
        are processed parents first, so a chain created by one segmentation splits like pairwise steps.
        """
        ds = self.data_structure
        if getattr(ds, "_anatomy", False):
            raise NotImplementedError("the repartition after structural steps is not available in anatomy mode yet: "
                                      "its lineage needs the owner vertices, not the Compartment graph")
        weights = self._weights()
        active = self._active_now(weights)
        ids = ds.entity_ids("node").tolist()
        known = np.array([v in weight_before for v in ids])
        basis = np.array([weight_before.get(v, 0.) for v in ids])
        was_active = np.array([bool(active_before.get(v, False)) for v in ids])
        parents = ds.parents()
        meta, derived, own = ds._variable_meta(), ds.derived(), set(getattr(self, "_variable_specs", {}))
        groups = {"massic": [], "extensive": [], "intensive": [], "non_inertial_extensive": []}
        for name in ds._node_data:
            if name in own or name in derived or name == self.partition_weight:
                continue
            kind = meta.get(name, {}).get("kind")
            group = {"massic_concentration": "massic", "extensive": "extensive", "intensive": "intensive",
                     "NonInertialIntensive": "intensive", "NonInertialExtensive": "non_inertial_extensive"}.get(kind)
            if group is not None:
                groups[group].append(name)
        if not any(groups.values()):
            return                                           # nothing to share (e.g. a growth model on its own)
        values = {name: np.array(ds.get(name), dtype=float) for group in groups.values() for name in group}

        for v in ds.order("pre"):
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
            ds.set(name, array)
