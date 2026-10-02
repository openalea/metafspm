from dataclasses import dataclass, field, fields, MISSING
from typing import Literal, Optional
import numpy as np

from openalea.metafspm.solve.decorator import *
from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.data_structure.data_api import DataStructure, MPGDataStructure, GraphDataStructure, DataStructurePropsView
from openalea.metafspm.data_structure.configs import ScalesConfig as _ScalesConfig
from openalea.metafspm.coupling.declaration import declared_specs, legacy_edge_convention, DeclarationError

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
            location=None, mapping=None, weight=None):
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
        location=location, mapping=mapping, weight=weight,
    )
    if default_factory:
        return field(default_factory=default_factory, metadata=metadata)
    return field(default=default, metadata=metadata)


def input_variable(unit: str, unit_comment: str, description: str, min_value: float,
                   max_value: float, value_comment: str, references: str, DOI: list,
                   by: str, initialize=None, scale=None, edge_mapping=None, on_grow="default",
                   location=None, mapping=None, weight=None):
    """Declare an input field — a variable driven by another model component.

    When the component is run in isolation (not coupled), the field keeps
    *initialize* as its uniform default everywhere.

    :param scale:  "node" | "edge" (graph) or grid-level descriptor (multigrid).
    """
    return declare(
        default=initialize, unit=unit, unit_comment=unit_comment,
        description=description, min_value=min_value, max_value=max_value,
        value_comment=value_comment, references=references, DOI=DOI,
        variable_type="input", by=by, state_variable_type=None,
        edit_by="user", scale=scale, edge_mapping=edge_mapping, on_grow=on_grow,
        location=location, mapping=mapping, weight=weight,
    )


def state_variable(unit: str, unit_comment: str, description: str, min_value: float,
                   max_value: float, value_comment: str, references: str, DOI: list,
                   state_variable_type: Literal[
                       "massic_concentration", "intensive", "extensive",
                       "NonInertialExtensive", "NonInertialIntensive", "descriptor"
                   ] = None,
                   initialize=None, scale=None, by: str = None,
                   edge_mapping=None, on_grow="default", location=None, mapping=None, weight=None):
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
        location=location, mapping=mapping, weight=weight,
    )


def parameter(unit: str, unit_comment: str, description: str, min_value: float,
              max_value: float, value_comment: str, references: str, DOI: list,
              by: str, default=None, scale=None, state_variable_type=None,
              edge_mapping=None, on_grow="default", location=None, mapping=None, weight=None):
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
        location=location, mapping=mapping, weight=weight,
    )


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
        self.choregrapher(module_family=self.__class__.__name__, *args)

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
class StructuralComponent(Component):
    def non_empty(self):
        pass


@dataclass
class FunctionalComponent(Component):
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

    data_structure: Optional[DataStructure] = None

    def __post_init__(self):
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

    @property
    def _graph_view(self):
        """GraphView of the DataStructure, rebuilt when its topology changed (growth), None for grids."""
        ds = self.data_structure
        if not hasattr(ds, "to_graph_view"):
            return None
        version = getattr(ds, "topology_version", None)
        if "_graph_view_cache" not in self.__dict__ or self.__dict__.get("_graph_view_version") != version:
            self.__dict__["_graph_view_cache"] = ds.to_graph_view(boundary_ports=getattr(self, "_boundary_ports", ()))
            self.__dict__["_graph_view_version"] = version
        return self.__dict__["_graph_view_cache"]

    @_graph_view.setter
    def _graph_view(self, view):
        # Explicit views (e.g. with boundary ports built by hand) are kept until the topology changes
        self.__dict__["_graph_view_cache"] = view
        self.__dict__["_graph_view_version"] = getattr(self.data_structure, "topology_version", None)

    def pull_available_inputs(self):
        """Refresh the derived inputs registered on the DataStructure by the coupling, before the step."""
        ds = self.data_structure
        for name in getattr(self, "_derived_inputs", []):
            ds.refresh(name)

    def previous(self, name: str) -> np.ndarray:
        """
        Value of unknown *name* at the start of the current graph-system solve, managed by the framework
        (replaces the user-managed ``_previous_fields``, deprecated; design note Q21).
        """
        state = getattr(self, "_previous_state", None)
        if state is None or name not in state:
            raise KeyError(f"No previous state for '{name}': previous() is available inside a graph-system solve "
                           "of one of its unknowns.")
        return state[name]

    def _auto_declare_on_ds(self, ds: DataStructure) -> None:
        """
        Register the declared variables that are not yet on *ds* (design note datastructure_contract §2).

        Each field is resolved once by resolve_declaration into a VariableSpec (location, MTG scale, mapping),
        kept in self._variable_specs. A variable is registered at its location with the values of its MTG
        property when the property exists (mapped from its scale), else with its declared default.
        Pre-registered variables are never overwritten; their location must match the declaration.
        """
        self._variable_specs = declared_specs(self, ds)
        # MTG refresh and write-back bookkeeping (node and edge variables with an MTG scale)
        self._bio_scale_node_fields: dict[str, int] = {}
        self._bio_scale_edge_fields: dict[str, str] = {}
        for name, spec in self._variable_specs.items():
            if spec.mtg_backed and spec.location == "node":
                self._bio_scale_node_fields[name] = spec.scale
            elif spec.mtg_backed and spec.location == "edge":
                self._bio_scale_edge_fields[name] = legacy_edge_convention(spec.mapping)

            if ds.has(name):
                registered = ds.location(name)
                if registered != spec.location and name not in ds.aliases():
                    raise DeclarationError(f"{type(self).__name__}.{name} is declared at {spec.location} but is "
                                           f"already registered at {registered}")
            else:
                ds.register(name, self._initial_values(ds, spec), location=spec.location, default=spec.default,
                            on_grow=spec.on_grow)
            meta = ds._variable_meta().setdefault(name, {})
            meta.update({key: value for key, value in spec.meta().items() if value is not None})

    @staticmethod
    def _initial_values(ds, spec):
        """Values of the MTG property of *spec* at its location, or None (no MTG, or no such property)."""
        if not spec.mtg_backed or not hasattr(ds, "_mtg_to_node_array"):
            return None
        if spec.location == "node":
            return ds._mtg_to_node_array(spec.name, scale=spec.scale)
        if spec.location == "edge":
            return ds._mtg_to_edge_array(spec.name, convention=legacy_edge_convention(spec.mapping),
                                         scale=spec.scale)
        if spec.mapping is None:
            # Stored at its own coarse scale: the values of the vertices of that scale
            return ds._mtg_values(spec.name, ds.entity_ids(spec.location))
        values = ds._mtg_to_node_array(spec.name)
        if values is None:
            return None
        weights = None
        if spec.weight is not None:
            if not ds.has(spec.weight):
                raise DeclarationError(f"'{spec.name}' is aggregated with weight '{spec.weight}', which is not "
                                       "registered yet: declare the weight first")
            weights = ds.get(spec.weight)
        return ds._map(values, "node", spec.location, spec.mapping, weights)

    def write_back_to_mtg(self) -> None:
        """Write solver results for biological-scale state_variable fields back to the MTG.

        Only ``state_variable`` fields are written back: these are the fields whose
        values change during a solve step (either updated by the Newton solver or by
        Choregrapher steps such as @rate).  ``parameter`` fields are intentionally
        skipped — parameters are never modified by the solver, so their props arrays
        stay at the construction-time default and writing them back would overwrite
        any externally-set MTG values.

        Node fields (scale=bio-int, no edge_mapping): written via write_node_to_mtg.
        Edge fields (scale=bio-int, edge_mapping set): written via write_edge_to_mtg
          using the declared convention ("proximal" or "distal"; "mean" is skipped
          since no unique endpoint exists for a symmetric property).
        """
        ds = self.data_structure
        sv_names: set[str] = {
            f.name for f in fields(type(self))
            if f.metadata.get("variable_type") == "state_variable"
        }

        for name in getattr(self, "_bio_scale_node_fields", {}):
            if name in sv_names and ds.has(name):
                ds.write_node_to_mtg(name, ds.get(name))
        for name, convention in getattr(self, "_bio_scale_edge_fields", {}).items():
            if name in sv_names and ds.has(name):
                ds.write_edge_to_mtg(name, ds.get(name), convention=convention)

    def _refresh_from_bio_scale(self) -> None:
        """Re-map biological-scale MTG properties to solver node/edge arrays.

        Called at the start of each graph-system solve so that parameters
        updated by other model components at the biological scale (e.g.
        K_axial updated at SubOrgan scale by a growth model) are reflected
        in the solver arrays used during snapshotting.

        Only ``parameter`` fields are refreshed.  State variables and inputs
        that are computed by rate laws or other Choregrapher steps must NOT
        be overwritten here — their props values are set by those steps and
        must remain intact until the solver reads them.

        Only fields whose MTG property is present and fully populated are
        refreshed; fields pre-set via set_node_property/set_edge_property
        without a corresponding MTG property are left unchanged (the MTG
        lookup returns None and the refresh is a no-op for that field).
        """
        ds = self.data_structure
        if not hasattr(ds, "_mtg_to_node_array"):
            return

        # Build a set of parameter field names for fast lookup.
        param_names: set[str] = {
            f.name for f in fields(type(self))
            if f.metadata.get("variable_type") == "parameter"
        }

        for name in getattr(self, "_bio_scale_node_fields", {}):
            if name not in param_names:
                continue
            arr = ds._mtg_to_node_array(name, scale=self._bio_scale_node_fields[name])
            if arr is None:
                continue
            ds.set_node_property(name, arr)

        if not hasattr(ds, "_mtg_to_edge_array"):
            return
        for name, convention in getattr(self, "_bio_scale_edge_fields", {}).items():
            if name not in param_names:
                continue
            arr = ds._mtg_to_edge_array(name, convention=convention)
            if arr is None:
                continue
            ds.set_edge_property(name, arr)
