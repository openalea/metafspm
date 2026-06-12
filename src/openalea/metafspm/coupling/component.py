from dataclasses import dataclass, field, fields, MISSING
from typing import Literal, Optional
import numpy as np

from openalea.metafspm.solve.decorator import *
from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.data_structure.data_api import DataStructure, MPGDataStructure, GraphDataStructure
from openalea.metafspm.data_structure.configs import ScalesConfig as _ScalesConfig

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
            default=None, default_factory=None, scale=None, edge_mapping=None):
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
    """
    metadata = dict(
        unit=unit, unit_comment=unit_comment, description=description,
        min_value=min_value, max_value=max_value, value_comment=value_comment,
        references=references, DOI=DOI, variable_type=variable_type, by=by,
        state_variable_type=state_variable_type, edit_by=edit_by,
        scale=scale, edge_mapping=edge_mapping,
    )
    if default_factory:
        return field(default_factory=default_factory, metadata=metadata)
    return field(default=default, metadata=metadata)


def input_variable(unit: str, unit_comment: str, description: str, min_value: float,
                   max_value: float, value_comment: str, references: str, DOI: list,
                   by: str, initialize=None, scale=None, edge_mapping=None):
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
        edit_by="user", scale=scale, edge_mapping=edge_mapping,
    )


def state_variable(unit: str, unit_comment: str, description: str, min_value: float,
                   max_value: float, value_comment: str, references: str, DOI: list,
                   state_variable_type: Literal[
                       "massic_concentration", "intensive", "extensive",
                       "NonInertialExtensive", "NonInertialIntensive", "descriptor"
                   ] = None,
                   initialize=None, scale=None, by: str = None,
                   edge_mapping=None):
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
        scale=scale, edge_mapping=edge_mapping,
    )


def parameter(unit: str, unit_comment: str, description: str, min_value: float,
              max_value: float, value_comment: str, references: str, DOI: list,
              by: str, default=None, scale=None, state_variable_type=None,
              edge_mapping=None):
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
        scale=scale, edge_mapping=edge_mapping,
    )


@dataclass
class Component:
    """
    Base component for structuring base FSPM modules

    HYPOTHESES:
        self.g.properties() must have been stored self.props during child class __init__
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

    def link_self_to_mtg(self, ignore=[]):
        # for input variables, initialize homogeneous values on each vertices.
        # This behavior will be overwritten in case of module providing the input variable
        for name in self.inputs:
            if not (name in self.props.keys() and len(self.props[name]) == len(self.vertices)):
                # if it is not provided by mtg file, Use by default value everywhere
                self.props.setdefault(name, {})
                self.props[name].update({key: getattr(self, name) for key in self.vertices})

        # for segment scale state variables
        for name in self.state_variables:
            if name not in ignore:
                self.props.setdefault(name, {})
                # set default in mtg, state_variable prevail on inputs
                self.props[name].update({key: getattr(self, name) for key in self.vertices})

        # for plant scale state variables
        for name in self.plant_scale_state:
            self.props.setdefault(name, {})
            # set default in mtg, state_variable prevail on inputs
            self.props[name].update({1: getattr(self, name)})


    def pull_available_inputs(self):
        props = self.props
        for input, source_variables in self.pullable_inputs.items():
            vertices = props[list(source_variables.keys())[0]].keys()
            props[input].update({vid: sum([props[variable][vid]*unit_conversion
                                           for variable, unit_conversion in source_variables.items()])
                                 for vid in vertices})




@dataclass
class StructuralComponent(Component):
    def non_empty(self):
        pass


@dataclass
class FunctionalComponent(Component):
    """
    Base for all functional (transport / balance) model components.

    Every subclass must be initialized with a DataStructure instance that
    provides the graph topology and initial field values.  The DataStructure
    is the single source of truth for node/edge data; __post_init__ derives
    self._graph_view and self.props from it so the solver decorator machinery
    has the legacy dict-of-dicts format it expects.

    Auto-registration
    -----------------
    __post_init__ calls _auto_declare_on_ds() before snapshotting props.
    Every field annotated with scale="node" or scale="edge" whose name is not
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
        if not isinstance(self.data_structure, DataStructure):
            raise TypeError(
                f"{type(self).__name__}() data_structure must be a DataStructure instance, "
                f"got {type(self.data_structure).__name__}."
            )
        ds = self.data_structure
        self._auto_declare_on_ds(ds)
        self._graph_view = ds.to_graph_view()
        self.props       = ds.to_props_dict()
        self.props["focus_elements"] = [int(v) for v in self._graph_view.node_ids]
        if not hasattr(self, "pullable_inputs"):
            self.pullable_inputs = {}
        self.choregrapher.add_time_and_data(self, 1, self.props)
        if not hasattr(self.choregrapher, "simulation_time_step"):
            self.choregrapher.add_simulation_time_step(1)

    def _auto_declare_on_ds(self, ds: DataStructure) -> None:
        """Register default arrays for scale-annotated fields not yet in ds.

        Scale resolution
        ----------------
        * scale=Compartment (9)      → solver node  (no write-back)
        * scale=Connection  (10)     → solver edge  (no write-back)
        * scale=bio-int, no edge_mapping   → biological node; write-back via write_node_to_mtg
        * scale=bio-int, edge_mapping set  → biological edge; write-back via write_edge_to_mtg
                                             edge_mapping controls the bio-VID→edge mapping:
                                               "proximal" = take from child n_id_b (default)
                                               "distal"   = take from parent n_id_a
                                               "mean"     = arithmetic mean (symmetric params)

        Pre-registered fields are never overwritten.
        """
        if not isinstance(ds, GraphDataStructure):
            return
        if not hasattr(self, "_bio_scale_node_fields"):
            self._bio_scale_node_fields: dict[str, int] = {}
        if not hasattr(self, "_bio_scale_edge_fields"):
            self._bio_scale_edge_fields: dict[str, str] = {}

        registered = set(ds.available_vars())
        for f in fields(type(self)):
            if f.metadata.get("variable_type") is None:
                continue
            if f.name in registered:
                continue
            scale_raw    = f.metadata.get("scale")
            edge_mapping = f.metadata.get("edge_mapping")

            is_bio_node = False
            is_bio_edge = False
            if isinstance(scale_raw, int):
                if scale_raw == _ScalesConfig.Connection:
                    solver_scale = "edge"
                elif scale_raw == _ScalesConfig.Compartment:
                    solver_scale = "node"
                elif edge_mapping is not None:
                    solver_scale = "edge"
                    is_bio_edge  = True
                else:
                    solver_scale = "node"
                    is_bio_node  = True
            elif scale_raw in ("node", "edge"):
                solver_scale = scale_raw
            else:
                continue

            default = f.default if f.default is not MISSING else 0.0

            if solver_scale == "node":
                if hasattr(ds, "_mtg_to_node_array"):
                    arr = ds._mtg_to_node_array(f.name)
                    if arr is not None:
                        ds.set_node_property(f.name, arr)
                        if is_bio_node:
                            self._bio_scale_node_fields[f.name] = scale_raw
                        continue
                ds.set_node_property(f.name, np.full(ds.n_nodes(), float(default)))
                if is_bio_node:
                    self._bio_scale_node_fields[f.name] = scale_raw
            else:
                conv = edge_mapping if is_bio_edge else "mean"
                if hasattr(ds, "_mtg_to_edge_array"):
                    arr = ds._mtg_to_edge_array(f.name, convention=conv)
                    if arr is not None:
                        ds.set_edge_property(f.name, arr)
                        if is_bio_edge:
                            self._bio_scale_edge_fields[f.name] = edge_mapping
                        continue
                ds.set_edge_property(f.name, np.full(ds.n_edges(), float(default)))
                if is_bio_edge:
                    self._bio_scale_edge_fields[f.name] = edge_mapping

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

        # ── Bio-scale node fields ─────────────────────────────────────────────
        bio_node = getattr(self, "_bio_scale_node_fields", {})
        if bio_node and hasattr(ds, "write_node_to_mtg") and hasattr(ds, "_idx_to_vid"):
            for name in bio_node:
                if name not in sv_names or name not in self.props:
                    continue
                prop_dict = self.props[name]
                arr = np.array(
                    [float(prop_dict.get(vid, 0.0)) for vid in ds._idx_to_vid],
                    dtype=np.float64,
                )
                ds.write_node_to_mtg(name, arr)

        # ── Bio-scale edge fields ─────────────────────────────────────────────
        bio_edge = getattr(self, "_bio_scale_edge_fields", {})
        if bio_edge and hasattr(ds, "write_edge_to_mtg"):
            gv        = self._graph_view
            edge_vids = [int(v) for v in gv.edge_ids]
            for name, convention in bio_edge.items():
                if name not in sv_names or name not in self.props:
                    continue
                prop_dict = self.props[name]
                arr = np.array(
                    [float(prop_dict.get(j, 0.0)) for j in edge_vids],
                    dtype=np.float64,
                )
                ds.write_edge_to_mtg(name, arr, convention=convention)

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
            arr = ds._mtg_to_node_array(name)
            if arr is None:
                continue
            ds.set_node_property(name, arr)
            prop_dict = self.props.get(name, {})
            for i, vid in enumerate(ds._idx_to_vid):
                if vid in prop_dict:
                    prop_dict[vid] = float(arr[i])

        if not hasattr(ds, "_mtg_to_edge_array"):
            return
        for name, convention in getattr(self, "_bio_scale_edge_fields", {}).items():
            if name not in param_names:
                continue
            arr = ds._mtg_to_edge_array(name, convention=convention)
            if arr is None:
                continue
            ds.set_edge_property(name, arr)
            prop_dict = self.props.get(name, {})
            for j in range(len(arr)):
                if j in prop_dict:
                    prop_dict[j] = float(arr[j])
