"""
Resolution of component field declarations into DataStructure variables (design note
docs/design/datastructure_contract.md §2).

A declaration carries three keys:
  scale     where the MTG property lives (an MTG scale int, or None for solver-only variables);
  location  where the DataStructure stores it ("node", "edge", "scalar", "cell", or a coarse scale name);
  mapping   how values go between *scale* and *location* when they differ (with *weight* if needed).

`resolve_declaration` is the single interpreter of these keys: registration, the solver snapshot and the
MTG write-back all use the VariableSpec it returns. "node" and "edge" are the entities of the graph built by
the MPG traversal; declarations anchored on biological scales are resolved against that graph.
"""
import warnings
from dataclasses import dataclass, fields as dc_fields
from typing import Optional

from openalea.metafspm.data_structure.configs import ScalesConfig as _ScalesConfig

SOLVER_LOCATIONS = ("node", "edge", "scalar", "cell")
UP_MAPPINGS = ("sum", "mean", "weighted_mean")
DOWN_MAPPINGS = ("broadcast",)
EDGE_MAPPINGS = ("child", "parent", "mean")
# Former edge mapping names (N2): "proximal" took the child's value, "distal" the parent's
_EDGE_MAPPING_RENAMES = {"proximal": "child", "distal": "parent"}

EXTENSIVE_KINDS = ("extensive", "NonInertialExtensive")
INTENSIVE_KINDS = ("intensive", "NonInertialIntensive")
MASSIC_KINDS = ("massic_concentration",)


@dataclass(frozen=True)
class VariableSpec:
    """Where a declared variable is stored, and how it maps to its MTG scale."""
    name: str
    location: str
    scale: Optional[int] = None          # MTG scale of the property; None: solver-only variable
    mapping: Optional[str] = None        # scale -> location mapping when they differ
    weight: Optional[str] = None         # weight variable of "weighted_mean"
    kind: Optional[str] = None           # state_variable_type
    variable_type: Optional[str] = None  # "state_variable" | "input" | "parameter" | "plant_scale_state"
    default: float = 0.
    on_grow: str = "default"

    @property
    def mtg_backed(self) -> bool:
        return self.scale is not None

    def meta(self) -> dict:
        """Metadata recorded on the DataStructure next to the values."""
        return {"scale": self.scale, "mapping": self.mapping, "weight": self.weight, "kind": self.kind,
                "variable_type": self.variable_type}


class DeclarationError(ValueError):
    """A declaration that cannot be resolved on the DataStructure it is bound to."""


def edge_mapping_name(name: str) -> str:
    """Current name of an edge mapping, warning for the former names (proximal -> child, distal -> parent)."""
    if name in _EDGE_MAPPING_RENAMES:
        new = _EDGE_MAPPING_RENAMES[name]
        warnings.warn(f"edge mapping '{name}' is renamed '{new}'", DeprecationWarning, stacklevel=3)
        return new
    return name


def legacy_edge_convention(mapping: str) -> str:
    """Name used by the DataStructure's MTG edge readers and writers."""
    return {"child": "proximal", "parent": "distal"}.get(mapping, mapping)


def default_mapping(kind: Optional[str], direction: str, name: str, weight: Optional[str] = None) -> str:
    """
    Mapping implied by the variable's state_variable_type when the declaration gives none (D9, option A).
    direction: "up" (fine -> coarse) or "down" (coarse -> fine).
    """
    if direction == "up":
        if kind in EXTENSIVE_KINDS:
            return "sum"
        if kind in INTENSIVE_KINDS:
            return "mean"
        if kind in MASSIC_KINDS:
            if weight is None:
                raise DeclarationError(f"'{name}' is a massic concentration aggregated to a coarser scale: "
                                       "give the weight of its mass-weighted mean (weight=...)")
            return "weighted_mean"
    else:
        if kind in INTENSIVE_KINDS or kind in MASSIC_KINDS:
            return "broadcast"
        if kind in EXTENSIVE_KINDS:
            raise DeclarationError(f"'{name}' is extensive and goes from a coarse scale down to finer entities: "
                                   "give an explicit mapping (copying it would multiply the amount)")
    raise DeclarationError(f"'{name}' changes scale without a mapping, and its state_variable_type ({kind!r}) "
                           "does not imply one: give mapping=...")


KIND_FAMILIES = {"extensive": "extensive", "NonInertialExtensive": "extensive",
                 "intensive": "intensive", "NonInertialIntensive": "intensive", "massic_concentration": "intensive",
                 "descriptor": "descriptor"}


def kinds_agree(a: Optional[str], b: Optional[str]) -> bool:
    """Whether two state_variable_types can be linked (same family); unknown kinds agree with anything."""
    if a is None or b is None or a not in KIND_FAMILIES or b not in KIND_FAMILIES:
        return True
    return KIND_FAMILIES[a] == KIND_FAMILIES[b]


def link_direction(ds, from_location: str, to_location: str) -> Optional[str]:
    """
    "up" from finer to coarser locations (nodes, then coarse scales, then the scalar plant scale), "down" the
    other way, None when edges are involved (node <-> edge mappings have no default).
    """
    def rank(location):
        if location == "scalar":
            return 0
        if location == "edge":
            return None
        if location in ("node", "cell"):
            return node_scale(ds) or 99
        names = {name: value for value, name in scale_names(ds).items()}
        return names.get(location)

    a, b = rank(from_location), rank(to_location)
    if a is None or b is None or a == b:
        return None
    return "up" if b < a else "down"


# ── Scales of a DataStructure ──────────────────────────────────────────────────

def _scales_of(ds):
    mtg = getattr(ds, "_mtg", None)
    scales = getattr(mtg, "scales", None)
    return type(scales) if scales is not None else _ScalesConfig


def scale_names(ds) -> dict:
    """{scale int: scale name} for the DataStructure's MTG (the class defaults without an MTG)."""
    return {value: name for name, value in vars(_scales_of(ds)).items()
            if isinstance(value, int) and not name.startswith("_")}


def node_scale(ds) -> Optional[int]:
    """MTG scale of the graph's nodes (the vertices the Compartments stand for), or None."""
    scale = ds._node_scale() if hasattr(ds, "_node_scale") else None
    return scale if scale is not None else getattr(ds, "_from_scale", None)


def _is_graph(ds) -> bool:
    """Plant graphs (MTG-backed); grids have a graph view too, but their own locations (cell, edge, scalar)."""
    return hasattr(ds, "_mtg")


def location_of_scale(ds, scale: int) -> str:
    """DataStructure location holding the values of MTG scale *scale*."""
    scales = _scales_of(ds)
    if scale == scales.Compartment:
        return "node"
    if scale == scales.Connection:
        return "edge"
    nodes = node_scale(ds)
    if nodes is None or scale == nodes:
        return "node"
    if scale > nodes:
        raise DeclarationError(f"scale {scale_names(ds).get(scale, scale)} is finer than the graph's nodes "
                               f"({scale_names(ds).get(nodes, nodes)}): a finer scale needs an anatomy (DS8)")
    return scale_names(ds)[scale]


def resolve_location(ds, location: str) -> str:
    """Location name as stored: scale names are accepted and resolved against the graph (N5)."""
    if location in SOLVER_LOCATIONS:
        return location
    names = {name: value for value, name in scale_names(ds).items()}
    if location in names:
        return location_of_scale(ds, names[location])
    raise DeclarationError(f"unknown location '{location}' (expected one of {SOLVER_LOCATIONS} or a scale name "
                           f"among {list(names)})")


def _scale_value(ds, scale):
    """MTG scale int from an int or a scale name; None for none."""
    if scale is None or isinstance(scale, int):
        return scale
    names = {name: value for value, name in scale_names(ds).items()}
    if scale in names:
        return names[scale]
    raise DeclarationError(f"unknown scale '{scale}'")


# ── The interpreter ────────────────────────────────────────────────────────────

def resolve_declaration(f, ds) -> Optional[VariableSpec]:
    """
    VariableSpec of dataclass field *f* on DataStructure *ds*, or None when the field is not a DataStructure
    variable (no declare() metadata, or neither scale nor location given).

    Legacy forms (accepted for one release, D2):
      scale="node" | "edge" | "scalar" | "cell"  -> that location, no MTG property;
      scale=<int>, edge_mapping=m                -> location "edge", mapping m;
      scale=Compartment / Connection             -> "node" / "edge", no MTG property.
    A scale coarser than the nodes without a location is stored at that scale (N1).
    """
    meta = f.metadata
    if meta.get("variable_type") is None:
        return None
    raw_scale, location = meta.get("scale"), meta.get("location")
    mapping, weight = meta.get("mapping"), meta.get("weight")
    legacy_edge = meta.get("edge_mapping")
    kind = meta.get("state_variable_type")
    default = f.default if f.default is not None and not _is_missing(f.default) else 0.
    try:
        default = float(default)
    except (TypeError, ValueError):
        raise DeclarationError(f"'{f.name}': a DataStructure variable needs a numeric default, got {default!r}") from None
    common = dict(name=f.name, kind=kind, variable_type=meta.get("variable_type"),
                  default=float(default), on_grow=meta.get("on_grow") or "default")

    if isinstance(raw_scale, str) and raw_scale in SOLVER_LOCATIONS:
        if location is not None:
            raise DeclarationError(f"'{f.name}': scale='{raw_scale}' is a location, do not also give location=")
        location, raw_scale = raw_scale, None
    if raw_scale is None and location is None:
        return None

    if not _is_graph(ds):
        # Grids: only their own locations; biological scales have no meaning there
        if location in ("cell", "scalar", "edge") and raw_scale is None and mapping is None:
            return VariableSpec(location=location, **common)
        return None

    scale = _scale_value(ds, raw_scale)
    scales = _scales_of(ds)
    if scale == scales.Compartment and node_scale(ds) == scales.Compartment:
        pass    # anatomy mode: the Compartments are the MTG vertices behind the nodes, their properties are read
    elif scale in (scales.Compartment, scales.Connection):
        # Solver entities themselves: no MTG property behind them
        if location is None:
            location = "node" if scale == scales.Compartment else "edge"
        scale = None

    if legacy_edge is not None:
        if mapping is not None:
            raise DeclarationError(f"'{f.name}': give mapping= or the former edge_mapping=, not both")
        mapping = legacy_edge
        location = location or "edge"
    if mapping is not None and mapping in _EDGE_MAPPING_RENAMES:
        mapping = edge_mapping_name(mapping)

    if scale is None:
        location = resolve_location(ds, location)
        if mapping is not None:
            raise DeclarationError(f"'{f.name}' has no MTG scale, its mapping '{mapping}' maps nothing")
        if location == "cell":
            raise DeclarationError(f"'{f.name}': location 'cell' is a grid location, not a graph one")
        return VariableSpec(location=location, **common)

    scale_location = location_of_scale(ds, scale)
    location = resolve_location(ds, location) if location is not None else scale_location

    if location == "scalar":
        raise DeclarationError(f"'{f.name}': a scalar variable has no MTG scale (drop scale=)")
    if location == "edge":
        if scale_location == "edge":
            return VariableSpec(location="edge", scale=None, **common)
        if mapping is None:
            raise DeclarationError(f"'{f.name}' maps scale {scale_names(ds)[scale]} to edges: give "
                                   f"mapping= one of {EDGE_MAPPINGS}")
        if mapping not in EDGE_MAPPINGS:
            raise DeclarationError(f"'{f.name}': edge mapping '{mapping}' is not one of {EDGE_MAPPINGS}")
        if mapping == "mean" and common["variable_type"] == "state_variable":
            raise DeclarationError(f"'{f.name}' is a state variable with edge mapping 'mean': it could not be "
                                   "written back (no single owner), use 'child' or 'parent'")
        if scale_location != "node" and common["variable_type"] == "state_variable":
            raise DeclarationError(f"'{f.name}' is an edge state variable at scale {scale_names(ds)[scale]}, coarser "
                                   "than the nodes: several edges would write the same vertex")
        return VariableSpec(location="edge", scale=scale, mapping=mapping, weight=weight, **common)

    if location == scale_location:
        if mapping is not None:
            raise DeclarationError(f"'{f.name}' is stored at its own scale ({location}): mapping '{mapping}' "
                                   "maps nothing")
        return VariableSpec(location=location, scale=scale, **common)

    # Between the nodes and a coarse scale
    if "node" not in (location, scale_location):
        raise DeclarationError(f"'{f.name}' maps scale {scale_names(ds)[scale]} to {location}: mappings go "
                               "between the graph's nodes and one coarser scale")
    location_scale = node_scale(ds) if location == "node" else _scale_value(ds, location)
    direction = "down" if location_scale > scale else "up"
    allowed = DOWN_MAPPINGS if direction == "down" else UP_MAPPINGS
    if mapping is None:
        mapping = default_mapping(kind, direction, f.name, weight)
    if mapping not in allowed:
        raise DeclarationError(f"'{f.name}' goes {direction} from scale {scale_names(ds)[scale]} to {location}: "
                               f"mapping must be one of {allowed}, got '{mapping}'")
    if mapping == "weighted_mean" and weight is None:
        raise DeclarationError(f"'{f.name}': mapping 'weighted_mean' needs weight=")
    if mapping == "sum" and common["variable_type"] == "state_variable":
        raise DeclarationError(f"'{f.name}' is a state variable summed from scale {scale_names(ds)[scale]} to "
                               f"{location}: it could not be written back to the finer scale, store it at its own "
                               f"scale (scale={location})")
    return VariableSpec(location=location, scale=scale, mapping=mapping, weight=weight, **common)


def _is_missing(value) -> bool:
    from dataclasses import MISSING
    return value is MISSING


def declared_specs(component_or_class, ds) -> dict:
    """{field name: VariableSpec} of every DataStructure variable declared by a component class."""
    cls = component_or_class if isinstance(component_or_class, type) else type(component_or_class)
    specs = {}
    for f in dc_fields(cls):
        try:
            spec = resolve_declaration(f, ds)
        except DeclarationError as error:
            raise DeclarationError(f"{cls.__name__}.{f.name}: {error}") from None
        if spec is not None:
            specs[f.name] = spec
    return specs
