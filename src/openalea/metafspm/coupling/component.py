from dataclasses import dataclass, field, fields, MISSING
from typing import Literal, Optional
import numpy as np

from openalea.metafspm.solve.decorator import *
from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.data_structure.data_api import DataStructure, MPGDataStructure, GraphDataStructure, DataStructurePropsView
from openalea.metafspm.data_structure.configs import ScalesConfig as _ScalesConfig
from openalea.metafspm.coupling.declaration import declared_specs, DeclarationError

# Map integer MPG scale constants to the generic "node"/"edge" vocabulary used
# by GraphDataStructure.  Compartment nodes are graph nodes; Connection edges
# are graph edges.
_SCALE_INT_TO_LOC: dict = {
    _ScalesConfig.Compartment: "node",
    _ScalesConfig.Connection: "edge",
}


def declare(unit: str, unit_comment: str, description: str, min_value: float, max_value: float,
            value_comment: str, references: str, DOI: list,
            variable_type: Literal["state_variable", "plant_scale_state", "input", "parameter"],
            by: str, state_variable_type: str, edit_by: Literal["user", "dev"],
            default=None, default_factory=None, scale=None, edge_mapping=None, on_grow="default",
            location=None, mapping=None, weight=None, dtype=None, shape=None):
    """
    Constrain component variable declarations in a commonly agreed-upon way.

    :param default:            Default value (scalar).
    :param unit:               SI unit string.
    :param unit_comment:       Precision about the unit (e.g. 'mol of N per g DW').
    :param description:        Full description of purpose, scale, and hypotheses.
    :param value_comment:      Why the default differs from the literature value.
    :param references:         Literature references (Author et al., Year).
    :param DOI:                List of DOI strings for the cited papers.
    :param variable_type:      "input" | "state_variable" | "plant_scale_state" | "parameter"
    :param by:                 Name of the model component that owns this variable.
    :param state_variable_type: "intensive" | "extensive" | "massic_concentration" | …
    :param edit_by:            "user" or "dev" — who may override the default.
    :param scale:              Where this variable lives in the data structure.
                               For graph models: "node" or "edge".
                               For multi-grid models: integer level index or a
                               grid-level descriptor string (e.g. "fine").
                               Used by FunctionalComponent.__post_init__ to
                               auto-register defaults on the bound DataStructure,
                               and by the solver decorator to classify fields
                               during the Newton snapshot.
    :param location:           Where the DataStructure stores the variable: "node", "edge", "scalar", "cell", or a
                               scale name (resolved against the graph). Default: the location of *scale*.
    :param mapping:            How values go between *scale* and *location* when they differ: "broadcast" (down),
                               "sum" / "mean" / "weighted_mean" (up, with *weight*), "child" / "parent" / "mean"
                               (to edges). Default: implied by state_variable_type (design note D9).
    :param weight:             Weight variable of "weighted_mean".
    :param dtype:              "int" for labels, types and indices (kept as integers), "object" for lists and records
                               (not usable by graph systems, derivations or transport); default float.
    :param shape:              Per-entity shape of a vector-valued variable, e.g. (15,) for 15 pools per cell: an
                               (n, 15) array, used by steps, mappings, checkpoints and outputs; not an MTG property
                               nor a graph-system unknown (PT7).
    :param on_grow:            Value of entities created by topology growth: "default" (the declared
                               default) or "inherit" (the parent's value). The growth model may still
                               overwrite them, e.g. from parent states for concentrations.
    """
    metadata = dict(
        unit=unit, unit_comment=unit_comment, description=description,
        min_value=min_value, max_value=max_value, value_comment=value_comment,
        references=references, DOI=DOI, variable_type=variable_type, by=by,
        state_variable_type=state_variable_type, edit_by=edit_by,
        scale=scale, edge_mapping=edge_mapping, on_grow=on_grow,
        location=location, mapping=mapping, weight=weight, dtype=dtype,
        shape=tuple(shape) if shape is not None else None,
    )
    if default_factory:
        return field(default_factory=default_factory, metadata=metadata)
    return field(default=default, metadata=metadata)


def input_variable(unit: str, unit_comment: str, description: str, min_value: float,
                   max_value: float, value_comment: str, references: str, DOI: list,
                   by: str, initialize=None, scale=None, edge_mapping=None, on_grow="default",
                   location=None, mapping=None, weight=None, state_variable_type=None, dtype=None, shape=None):
    """Declare an input field — a variable driven by another model component.

    When the component is run in isolation (not coupled), the field keeps
    *initialize* as its uniform default everywhere.

    :param scale:  "node" | "edge" (graph) or grid-level descriptor (multigrid).
    :param state_variable_type: optional kind of the input ("extensive", "intensive", ...): when given, it must
                   agree with the kind of the variable that provides it through the translator.
    """
    return declare(
        default=initialize, unit=unit, unit_comment=unit_comment,
        description=description, min_value=min_value, max_value=max_value,
        value_comment=value_comment, references=references, DOI=DOI,
        variable_type="input", by=by, state_variable_type=state_variable_type,
        edit_by="user", scale=scale, edge_mapping=edge_mapping, on_grow=on_grow,
        location=location, mapping=mapping, weight=weight, dtype=dtype, shape=shape,
    )


def state_variable(unit: str, unit_comment: str, description: str, min_value: float,
                   max_value: float, value_comment: str, references: str, DOI: list,
                   state_variable_type: Literal[
                       "massic_concentration", "intensive", "extensive",
                       "NonInertialExtensive", "NonInertialIntensive", "descriptor"
                   ] = None,
                   initialize=None, scale=None, by: str = None,
                   edge_mapping=None, on_grow="default", location=None, mapping=None, weight=None, dtype=None,
                   shape=None):
    """Declare a prognostic state variable solved or integrated by this component.

    :param state_variable_type: Thermodynamic / extensive classification.
    :param scale:               "node" | "edge" (graph), biological scale integer,
                                or grid-level descriptor.
    :param edge_mapping:        "proximal" | "distal" | "mean" — required when
                                *scale* is a biological integer and the variable
                                lives on edges.  "proximal" means the value on
                                edge (parent, child) is owned by the child vertex.
    """
    return declare(
        default=initialize, unit=unit, unit_comment=unit_comment,
        description=description, min_value=min_value, max_value=max_value,
        value_comment=value_comment, references=references, DOI=DOI,
        variable_type="state_variable", by=by,
        state_variable_type=state_variable_type, edit_by="user",
        scale=scale, edge_mapping=edge_mapping, on_grow=on_grow,
        location=location, mapping=mapping, weight=weight, dtype=dtype, shape=shape,
    )


def parameter(unit: str, unit_comment: str, description: str, min_value: float,
              max_value: float, value_comment: str, references: str, DOI: list,
              by: str, default=None, scale=None, state_variable_type=None,
              edge_mapping=None, on_grow="default", location=None, mapping=None, weight=None, dtype=None):
    """Declare a model parameter — a constant whose value is set at construction.

    Parameters are not prognostic; they are read by model equations but never
    written back.  Marking them with *scale* tells FunctionalComponent where
    to register the default array on the DataStructure so the solver decorator
    can snapshot them at the correct entity level.

    :param default:            Nominal parameter value.
    :param scale:              "node" | "edge" (graph), biological scale integer,
                               or grid-level descriptor.
    :param edge_mapping:       "proximal" | "distal" | "mean" — see state_variable.
    :param state_variable_type: Optional size-dependence classification
                               ("intensive" / "extensive"); used for coupling
                               consistency checks.
    """
    return declare(
        default=default, unit=unit, unit_comment=unit_comment,
        description=description, min_value=min_value, max_value=max_value,
        value_comment=value_comment, references=references, DOI=DOI,
        variable_type="parameter", by=by,
        state_variable_type=state_variable_type, edit_by="dev",
        scale=scale, edge_mapping=edge_mapping, on_grow=on_grow,
        location=location, mapping=mapping, weight=weight, dtype=dtype,
    )


class _PlantParameter:
    """
    A numeric parameter stored per plant on the DataStructure (devplan_population_scene §7-8, QH2-QH3).
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
            raise AttributeError(f"{type(obj).__name__}.{self.name} is a parameter stored per plant: take it as an "
                                 f"argument of the step or equation ({self.name}) instead of reading self.{self.name}")
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

    def values_at(self, obj, location: str = "node") -> np.ndarray:
        return obj.data_structure.parameter_view(self.name, location)


@dataclass
class Component:
    """
    Base component for structuring base FSPM modules

    Variables are declared with declare() / state_variable() / input_variable() / parameter(); the data lives in a
    DataStructure (see FunctionalComponent).
    """

    choregrapher = Choregrapher()

    def __call__(self, *args):
        self.pull_available_inputs()
        self.choregrapher(instance=self)
        # State variables reach the MTG after every call with mtg_sync = "after_call"; by default ("lazy") the
        # DataStructure writes them when the MTG is read (QF3)
        if hasattr(self, "mtg_sync"):
            self._check_mtg_sync()
            if self.mtg_sync == "after_call":
                self.write_back_to_mtg()
        # The component's clock (forcings, PT4): the start of its next call
        self.__dict__["_clock"] = self.__dict__.get("_clock", 0.) + float(
            getattr(self.choregrapher, "simulation_time_step", 0.) or 0.)

    def write_back_to_mtg(self) -> None:
        """Write the component's state variables to the MTG; FunctionalComponent implements it."""
        pass

    @property
    def inputs(self):
        return [f.name for f in fields(self) if f.metadata.get("variable_type") == "input"]

    @property
    def state_variables(self):
        return [f.name for f in fields(self) if f.metadata.get("variable_type") == "state_variable"]

    @property
    def extensive_variables(self):
        return [f.name for f in fields(self) if (f.metadata.get("variable_type") == "state_variable" and f.metadata.get("state_variable_type") == "extensive")]

    @property
    def massic_concentration(self):
        return [f.name for f in fields(self) if (f.metadata.get("variable_type") == "state_variable" and f.metadata.get("state_variable_type") == "massic_concentration")]

    @property
    def intensive_variables(self):
        return [f.name for f in fields(self) if (f.metadata.get("variable_type") == "state_variable" and f.metadata.get("state_variable_type") == "intensive")]

    @property
    def non_inertial_extensive(self):
        return [f.name for f in fields(self) if (f.metadata.get("variable_type") == "state_variable" and f.metadata.get("state_variable_type") == "NonInertialExtensive")]

    @property
    def non_inertial_intensive(self):
        return [f.name for f in fields(self) if (f.metadata.get("variable_type") == "state_variable" and f.metadata.get("state_variable_type") == "NonInertialIntensive")]

    @property
    def non_inertial_variables(self):
        return [f.name for f in fields(self) if (f.metadata.get("variable_type") == "state_variable" and f.metadata.get("state_variable_type") in ("NonInertialIntensive", "NonInertialExtensive"))]

    @property
    def descriptor(self):
        return [f.name for f in fields(self) if (f.metadata.get("variable_type") == "state_variable" and f.metadata.get("state_variable_type") == "descriptor")]

    @property
    def plant_scale_state(self):
        return [f.name for f in fields(self) if f.metadata.get("variable_type") == "plant_scale_state"]

    @property
    def parameters(self):
        return [f.name for f in fields(self) if f.metadata.get("variable_type") == "parameter"]

    def apply_scenario(self, **kwargs):
        """
        Method to superimpose default parameters in order to create a scenario.
        Use Model.documentation to discover model parameters and state variables.
        :param kwargs: mapping of existing variable to superimpose.
        """
        for changed_parameter, value in kwargs.items():
            if changed_parameter in dir(self):
                setattr(self, changed_parameter, value)

    def pull_available_inputs(self):
        """Refresh the inputs derived by the coupling; FunctionalComponent implements it on its DataStructure."""
        pass


@dataclass
class DataStructureComponent(Component):
    """
    Component bound to a DataStructure (base of FunctionalComponent and StructuralComponent).

    Its declared fields are resolved once (resolve_declaration) and registered on the DataStructure when missing,
    from their MTG property or their default; its state variables are written to the MTG after every call, and its
    MTG-backed parameters are re-read before each graph solve.
    """

    data_structure: Optional[DataStructure] = None

    # MTG synchronisation policy (design note time_and_data §3, DS4; QF3): "lazy" (default) writes the state variables
    # changed since the last synchronisation when the MTG is read (ds.mtg, MPG-style steps, ds.flush_mtg());
    # "after_call" writes them after every call; "never" leaves the MTG untouched.
    mtg_sync = "lazy"

    def __post_init__(self):
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
        if not hasattr(self.choregrapher, "simulation_time_step"):
            self.choregrapher.add_simulation_time_step(1)
        # One iteration per simulation step unless the component declares its own sub time step
        sub_time_step = getattr(self, "sub_time_step", None) or self.choregrapher.simulation_time_step
        # Live reading (design note §8): steps and solves read and write the DataStructure arrays;
        # props is a read-only compatibility view.
        self.props = DataStructurePropsView(ds)
        self.choregrapher.add_time_and_data(self, sub_time_step, ds, compartment="graph")

    def pull_available_inputs(self):
        """
        Before the step: re-read the MTG-backed parameters (DS4), and bring the inputs derived by the coupling up
        to date. Derived variables are recomputed when read (D10), so the latter only recomputes those whose sources
        changed; it is the hook of future sub-steps.
        """
        self._refresh_from_bio_scale()
        ds = self.data_structure
        for name in getattr(self, "_derived_inputs", []):
            ds.get(name)

    def _auto_declare_on_ds(self, ds: DataStructure) -> None:
        """
        Register the declared variables that are not yet on *ds* (design note datastructure_contract §2).

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
                values = ds.read_mtg(spec) if hasattr(ds, "read_mtg") else None
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
        if self.mtg_sync == "lazy" and hasattr(ds, "track_mtg"):
            for spec in self._variable_specs.values():
                if spec.variable_type == "state_variable" and spec.mtg_backed:
                    ds.track_mtg(spec)

    def _check_mtg_sync(self) -> None:
        if self.mtg_sync not in ("lazy", "after_call", "never"):
            raise ValueError(f"{type(self).__name__}.mtg_sync must be 'lazy', 'after_call' or 'never', got "
                             f"'{self.mtg_sync}'")

    @classmethod
    def _plant_parameter_fields(cls) -> dict:
        """{name: default} of the numeric parameters declared without a place, stored per plant (QH3)."""
        names = {}
        for f in fields(cls):
            meta = f.metadata
            if (meta.get("variable_type") == "parameter" and meta.get("scale") is None and meta.get("location") is None
                    and meta.get("dtype") in (None, float, int, "float", "int") and not isinstance(f.default, bool)
                    and isinstance(f.default, (int, float, np.integer, np.floating))):
                names[f.name] = f.default
        return names

    def _install_plant_parameters(self) -> None:
        """Replace, once per class, each per-plant parameter's attribute by a _PlantParameter descriptor (QH2)."""
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

    def parameter_values(self, name: str, location: str = "node") -> np.ndarray:
        """
        Per-entity values of a per-plant parameter, for code that cannot take it as an argument (MPG-style structural
        steps): each node (or edge) gets its plant's value.
        """
        return self.data_structure.parameter_view(name, location)

    # Seed of this component's random streams (PT2); a scenario may set it
    random_seed = 0

    def random(self, stream: str, distribution: str = "uniform", ids=None, location: str = "node",
               **parameters) -> np.ndarray:
        """
        Reproducible draws per entity for stream *stream*: each call of a stream is a new step, so a model drawing in
        the same order gets the same draws at every run, whatever the visiting order (ds.random, PT2).
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
        loop over vertices themselves: e.g. plants before emergence are skipped (P6, QP6c).
        """
        ds = self.data_structure
        ids = np.asarray(ds.entity_ids("node"))
        return ids[np.asarray(ds.mask("active"), dtype=bool)] if ds.has_mask("active") else ids

    def write_back_to_mtg(self) -> None:
        """
        Write the declared state variables with an MTG scale to the MTG, at the vertices of their scale, through the
        inverse of their mapping (MPGDataStructure.write_mtg). Called after every component call with
        mtg_sync = "after_call"; with "lazy" the DataStructure does it when the MTG is read.

        Only ``state_variable`` fields are written: parameters and inputs are owned by whoever sets them, and
        writing their defaults back would overwrite externally set MTG values.
        """
        self._check_mtg_sync()
        ds = self.data_structure
        if self.mtg_sync == "never" or not hasattr(ds, "write_mtg"):
            return
        for spec in getattr(self, "_variable_specs", {}).values():
            if spec.variable_type == "state_variable" and spec.mtg_backed and ds.has(spec.name):
                ds.write_mtg(spec)

    def _refresh_from_bio_scale(self) -> None:
        """
        Re-read the declared parameters with an MTG scale from the MTG, before each graph-system solve, so that
        parameters updated on the MTG by other components (e.g. K_axial by a growth model) are seen.

        State variables and inputs are not refreshed: their DataStructure values are set by steps and couplings.
        Parameters whose MTG property is absent keep their DataStructure values.
        """
        ds = self.data_structure
        if not hasattr(ds, "read_mtg"):
            return
        for spec in getattr(self, "_variable_specs", {}).values():
            if spec.variable_type != "parameter" or not spec.mtg_backed:
                continue
            if hasattr(ds, "flush_mtg"):
                ds.flush_mtg([spec.name])            # a value set on the DataStructure is not lost to a stale MTG one
            values = ds.read_mtg(spec)
            if values is not None:
                ds.set(spec.name, values)


@dataclass
class FunctionalComponent(DataStructureComponent):
    """
    Base for all functional (transport / balance) model components.

    Every subclass must be initialized with a DataStructure instance that
    provides the topology and initial field values.  The DataStructure is the
    single source of truth: steps and graph-system solves read and write its
    arrays live; self.props is a read-only compatibility view of it.

    Auto-registration
    -----------------
    __post_init__ calls _auto_declare_on_ds().
    Every field annotated with a scale ("node", "edge", a biological scale, or "cell" / "scalar" on grids) whose name is not
    yet registered on the DataStructure is initialized with the field default
    value (uniform array).  This means:

    * In production, simply construct the model — all fields with declared
      scale appear in ds with their default values automatically.
    * When non-default values are needed (tests, scenario setup), call
      ds.set_node_property / ds.set_edge_property BEFORE constructing the
      model.  Pre-registered values are never overwritten.

    Usage::

        ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
        ds.set_node_property("concentration", c_init)   # non-default — must pre-set
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
            ports = getattr(self, "_boundary_ports", ())
            if ports:
                import warnings
                warnings.warn(f"{type(self).__name__}: boundary ports set by hand are deprecated, declare boundary "
                              "sets in the graph system (boundary_set)", DeprecationWarning, stacklevel=2)
            self.__dict__["_graph_view_cache"] = ds.to_graph_view(boundary_ports=ports)
            self.__dict__["_graph_view_version"] = version
        return self.__dict__["_graph_view_cache"]

    @_graph_view.setter
    def _graph_view(self, view):
        # Explicit views (e.g. with boundary ports built by hand) are kept until the topology changes
        self.__dict__["_graph_view_cache"] = view
        self.__dict__["_graph_view_version"] = getattr(self.data_structure, "topology_version", None)

    # Forcings read inside steps and equations (PT4): {name: pandas Series indexed by time (s), (times, values),
    # or a callable t -> value}, interpolated linearly
    forcings = None

    def forcing_time(self) -> float:
        """
        The time at which forcings are read (QPj): the end of the current (sub-)step for implicit solves and steps,
        the evaluation time inside an IVP solve; times count from the component's clock (the scene time).
        """
        start = self.__dict__.get("_clock", 0.)
        offset = self.__dict__.get("_solve_offset", 0.)
        if "_ivp_time" in self.__dict__:
            return start + offset + self.__dict__["_ivp_time"]
        step = self.__dict__.get("_current_dt", None)
        if step is None:
            step = float(getattr(self.choregrapher, "simulation_time_step", 0.) or 0.)
        return start + offset + step

    def forcing(self, name: str):
        """Forcing *name* at forcing_time(), linearly interpolated in its table."""
        table = (self.forcings or {}).get(name)
        shared = self.__dict__.get("_scene_forcings")
        if table is None and shared is not None and name in shared:
            table = shared[name]                          # the scene's shared table (PT5)
        if table is None:
            raise KeyError(f"{type(self).__name__} has no forcing '{name}' (set self.forcings, or the scene's)")
        t = self.forcing_time()
        if callable(table):
            return table(t)
        if hasattr(table, "index"):
            times, values = np.asarray(table.index, dtype=np.float64), np.asarray(table, dtype=np.float64)
        else:
            times, values = (np.asarray(x, dtype=np.float64) for x in table)
        return float(np.interp(t, times, values))

    def pool_exchange(self, name: str):
        """
        During a graph-system solve with pool unknowns (PT4): the sparse (n_nodes, n_pools) map between the nodes of
        the pool's exchange set and the pool of their entity. P @ pool gives each node its pool's value; P.T @ flux
        sums node fluxes into their pools.
        """
        maps = self.__dict__.get("_pool_exchange") or {}
        if name not in maps:
            raise KeyError(f"'{name}' is not a pool unknown of the graph system being solved")
        return maps[name]

    @property
    def dt(self) -> float:
        """Length of the current (sub-)step of a graph-system solve; the time step outside sub-stepping (T1)."""
        return self.__dict__.get("_current_dt", getattr(self, "time_step", None))

    def previous(self, name: str, at: str = "substep") -> np.ndarray:
        """
        Value of unknown *name*, managed by the framework (design note time_and_data §2, T2):
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


@dataclass
class StructuralComponent(DataStructureComponent):
    """
    Component that edits the plant's structure (growth, segmentation, anatomy; design note structure_and_boundaries
    §3, DS19). It shares the plant's DataStructure, and edits the MPG through ``self.mtg`` with the MPG's own
    methods: the DataStructure does not re-expose them, the MPG is the source of truth for structure (D11).

    Two step styles coexist (Q17):
      * a step without arguments ("MPG-style") reads and edits the MPG. Around it, the framework writes the
        component's declared variables to the MPG before, and after it updates the DataStructure's topology if the
        MPG's changed, then re-reads the declared state variables (the structural outputs) from the MPG (P1);
      * a step with arguments ("array-style") is vectorised on DataStructure arrays, like a functional step; its
        outputs reach the MPG at the next MPG-style step or at the end of the call.
    """

    @property
    def mtg(self):
        return self.data_structure.mtg

    @classmethod
    def initiate_plant(cls, g, plant_vid: int, parameters: dict) -> None:
        """
        Build the initial structure of one plant under its Plant-scale vertex *plant_vid* in the population MPG *g*,
        from that plant's scenario *parameters*, before any component is constructed (devplan_population_scene §8,
        QP4a). Several structural components may each add their part, in order (e.g. roots, then anatomies). The
        plant's position is in g.property("x"/"y"/"z"/"rotation")[plant_vid]; the component computes its elements'
        coordinates from it (QP4b).
        """
        raise NotImplementedError(f"{cls.__name__} does not initiate plants")

    def _mpg_signature(self) -> tuple:
        """Cheap topology signature of the MPG: vertex count and last allocated vertex id (P3)."""
        g = self.mtg
        return g.nb_vertices(), getattr(g, "_id", None)

    # Repartition of the other components' variables when the structure changes (design note
    # structure_and_boundaries §4, DS20). partition_weight: a node variable name, or a callable ds -> array;
    # active: the rule of the DataStructure's "active" mask ({variable: condition} or a callable). Without a
    # partition weight, entities created by growth only get their on_grow values.
    partition_weight = None
    active = None

    def __post_init__(self):
        super().__post_init__()
        if self.active is not None and not self.data_structure.has_mask("active"):
            self.data_structure.define_mask("active", self.active)

    def _weights(self):
        ds = self.data_structure
        if self.partition_weight is None:
            return None
        values = self.partition_weight(ds) if callable(self.partition_weight) else ds.get(self.partition_weight)
        return np.array(values, dtype=float)

    def _active_now(self, weights):
        ds = self.data_structure
        return np.array(ds.mask("active"), dtype=bool) if ds.has_mask("active") else weights > 0

    def _run_mpg_step(self, run) -> None:
        """Run an MPG-style step with the synchronisation of the DataStructure around it, then the repartition."""
        ds = self.data_structure
        specs = [spec for spec in getattr(self, "_variable_specs", {}).values() if spec.mtg_backed and ds.has(spec.name)]
        if hasattr(ds, "flush_mtg"):
            # The MPG-style code reads the MTG: the variables this component declares (its states, inputs and
            # parameters) changed since their last synchronisation are written first (QF3). Others are not: a step
            # reads what its component declares
            for spec in specs:
                ds.track_mtg(spec)
            ds.flush_mtg([spec.name for spec in specs])
        else:
            for spec in specs:
                ds.write_mtg(spec)
        weights = self._weights()
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
                values = ds.read_mtg(spec)
                if values is not None:
                    ds.set(spec.name, values)
                    if hasattr(ds, "mark_mtg_synced"):
                        ds.mark_mtg_synced(spec.name)
        if weights is not None:
            self._repartition(weight_before, active_before)

    def _repartition(self, weight_before: dict, active_before: dict) -> None:
        """
        Share the node variables of the other components between the entities of the new structure, by their
        state_variable_type (the rules of rhizodep's post_growth_updating, D14):

          kind                      new active entity v, or one becoming active,    new inactive     existing, weight
                                    with its parent p and f = w_v / (w_v + w_p)     entity           changed
          massic_concentration      the amount c_p * b_p is split by f, 1 - f       parent's value   c * b / w
          extensive                 x_v = f x_p, x_p = (1 - f) x_p                   0                unchanged
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
                    values[name][v], values[name][p] = f * values[name][p], (1. - f) * values[name][p]
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
