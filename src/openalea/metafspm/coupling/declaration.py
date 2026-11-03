"""
Resolution of component field declarations into DataStructure variables.

A declaration carries three keys:
  scale     where the MTG property lives (an MTG scale int, or None for solver-only variables);
  location  where the DataStructure stores it ("node", "edge", "scalar", "cell", or a coarse scale name);
  mapping   how values go between *scale* and *location* when they differ (with *weight* if needed).

`resolve_declaration` is the single interpreter of these keys: registration, the solver snapshot and the
MTG write-back all use the VariableSpec it returns. "node" and "edge" are the entities of the graph built by
the MPG traversal; declarations anchored on biological scales are resolved against that graph.
"""
import warnings

import numpy as np
from dataclasses import dataclass, fields as dc_fields
from typing import Optional

from openalea.metafspm.data_structure.configs import ScalesConfig as _ScalesConfig

SOLVER_LOCATIONS = ("node", "edge", "scalar", "cell")
UP_MAPPINGS = ("sum", "mean", "weighted_mean", "max", "min")
POOLED_UP_MAPPINGS = ("sum", "mean", "weighted_mean")    # across DataStructures: pooled over the populations
DOWN_MAPPINGS = ("broadcast", "split")
EDGE_MAPPINGS = ("child", "parent", "mean")

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
    kind: Optional[str] = None           # extensive, intensive, ... (its declaration's kind=)
    variable_type: Optional[str] = None  # "state_variable" | "input" | "parameter"
    default: float = 0.
    on_grow: str = "default"
    dtype: type = float                  # float, int or object
    shape: tuple = ()                    # per-entity shape of a vector-valued variable

    @property
    def mtg_backed(self) -> bool:
        return self.scale is not None and not self.shape

    def meta(self) -> dict:
        """Metadata recorded on the DataStructure next to the values."""
        return {"scale": self.scale, "mapping": self.mapping, "weight": self.weight, "kind": self.kind,
                "variable_type": self.variable_type, "dtype": self.dtype, "shape": self.shape}


class DeclarationError(ValueError):
    """A declaration that cannot be resolved on the DataStructure it is bound to."""


def legacy_edge_convention(mapping: str) -> str:
    """Name used by the DataStructure's MTG edge readers and writers."""
    return {"child": "proximal", "parent": "distal"}.get(mapping, mapping)


def default_mapping(kind: Optional[str], direction: str, name: str, weight: Optional[str] = None) -> str:
    """
    Aggregation implied by the variable's kind when the declaration or the link gives none, within a
    DataStructure and across DataStructures (plant entities and cells) alike. direction: "up" (fine -> coarse,
    plant entities -> cells), "down" (coarse -> fine, cells -> plant entities) or "layer" (a column's layers and a
    grid's: "sum" for extensive variables, "mean" for intensive ones).

      extensive   up: sum (the total is conserved); down: split by *weight* (each entity's share of its owner's
                  amount in proportion to its weight, the total conserved), else raise
      intensive   up: the mean weighted by *weight* (its capacity: a volume, a mass, an area...), else raise
                  (give aggregation="mean" for an unweighted mean); down: broadcast. A massic concentration is an
                  intensive variable whose weight is a mass, a flux density one whose weight is an area.
    """
    if direction == "layer":
        if kind in EXTENSIVE_KINDS:
            return "sum"
        if kind in INTENSIVE_KINDS or kind in MASSIC_KINDS:
            return "mean"
    elif direction == "up":
        if kind in EXTENSIVE_KINDS:
            return "sum"
        if kind in INTENSIVE_KINDS or kind in MASSIC_KINDS:
            if weight is None:
                raise DeclarationError(f"'{name}' is {kind} and averaged to a coarser scale: give the weight of its "
                                       "mean (weight=..., its capacity: a volume, a mass, an area), or "
                                       "aggregation='mean' for an unweighted mean")
            return "weighted_mean"
    else:
        if kind in INTENSIVE_KINDS or kind in MASSIC_KINDS:
            return "broadcast"
        if kind in EXTENSIVE_KINDS:
            if weight is not None:
                return "split"
            raise DeclarationError(f"'{name}' is extensive and goes from a coarse scale down to finer entities: "
                                   "give the weight of its split (weight=...: each entity gets its share of the "
                                   "amount), or aggregation= (copying it would multiply the amount)")
    raise DeclarationError(f"'{name}' changes scale without a mapping, and its kind ({kind!r}) "
                           "does not imply one: give aggregation=...")


KIND_FAMILIES = {"extensive": "extensive", "NonInertialExtensive": "extensive",
                 "intensive": "intensive", "NonInertialIntensive": "intensive", "massic_concentration": "intensive",
                 "descriptor": "descriptor"}


def kinds_agree(a: Optional[str], b: Optional[str]) -> bool:
    """Whether two kinds can be linked (same family); unknown kinds agree with anything."""
    if a is None or b is None or a not in KIND_FAMILIES or b not in KIND_FAMILIES:
        return True
    return KIND_FAMILIES[a] == KIND_FAMILIES[b]


def declared_unit(component, name: str) -> Optional[str]:
    """The unit *component*'s class declares for *name* (None when undeclared)."""
    f = getattr(type(component), "__dataclass_fields__", {}).get(name)
    return None if f is None else f.metadata.get("unit")


def declared_kind(component, name: str) -> Optional[str]:
    """The kind of *name* for *component*: its declaration's, else the one recorded on its DataStructure."""
    spec = getattr(component, "_variable_specs", {}).get(name)
    if spec is not None and spec.kind is not None:
        return spec.kind
    ds = getattr(component, "data_structure", None)
    if ds is not None and hasattr(ds, "_variable_meta") and ds.has(name):
        return ds._variable_meta().get(ds._resolve(name), {}).get("kind")
    return None


def check_link(link, receiver, provider) -> None:
    """A link's kinds agree (extensive with extensive, intensive or massic with intensive or massic) and its factors
    convert the provider's declared units into the receiver's (unless unit_check is off or it is a formula)."""
    from openalea.metafspm.coupling.units import check_link_units
    name = f"{link.receiver}.{link.variable} <- {link.provider}"
    received = declared_kind(receiver, link.variable)
    for source in ([] if link.formula is not None else link.sources):
        provided = declared_kind(provider, source)
        if not kinds_agree(received, provided):
            raise ValueError(f"{link.receiver}.{link.variable} ({received}) <- {link.provider}.{source} ({provided}): "
                             "the kinds do not agree (extensive with extensive, intensive or massic with intensive or "
                             "massic)")
    if link.formula is not None or not link.unit_check:
        return
    for source, factor in link.sources.items():
        if isinstance(factor, (int, float)):            # an expression: its conversion is its own
            check_link_units(declared_unit(receiver, link.variable), declared_unit(provider, source), factor,
                             f"{name}.{source}")


def link_direction(ds, from_location: str, to_location: str) -> Optional[str]:
    """
    "up" from finer to coarser locations (nodes, then coarse scales, then the scalar plant scale), "down" the
    other way, None when edges are involved (node <-> edge mappings have no default).
    """
    def rank(location, source=False):
        if location == "scalar":
            return 0
        if location == "edge":                       # edges map up like their (child) nodes; nothing maps down to them
            return (node_scale(ds) or 99) if source else None
        if location in ("node", "cell"):
            return node_scale(ds) or 99
        names = {name: value for value, name in scale_names(ds).items()}
        return names.get(location)

    a, b = rank(from_location, source=True), rank(to_location)
    if a is None or b is None or a == b:
        return None
    return "up" if b < a else "down"


# ── Scales of a DataStructure ──────────────────────────────────────────────────

def _scales_of(ds):
    mtg = getattr(ds, "_mtg", None)
    scales = getattr(mtg, "scales", None)
    return type(scales) if scales is not None else _ScalesConfig


def scale_names(of=None) -> dict:
    """{scale int: scale name} of *of*: a DataStructure (its MTG's scales), an MTG, a ScalesConfig class, or None
    (ScalesConfig's)."""
    if of is None:
        scales = _ScalesConfig
    elif isinstance(of, type):
        scales = of
    elif hasattr(of, "scales") and not hasattr(of, "_mtg"):          # an MTG
        scales = type(of.scales)
    else:
        scales = _scales_of(of)
    return {value: name for name, value in vars(scales).items() if isinstance(value, int) and not name.startswith("_")}


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
                               f"({scale_names(ds).get(nodes, nodes)}): a finer scale needs an anatomy")
    return scale_names(ds)[scale]


def held_at_scale(spec, ds) -> bool:
    """
    Whether *spec*, an input or parameter, is a variable another component holds at another location of *ds* (one
    DataStructure for every scale): read at the nodes from a coarser scale (mapped down: broadcast or split) while
    held at that scale's own location (a SubOrgan-scale architecture's length read at the Compartments), or read at
    its scale while held at the nodes as a broadcast from it (a segment's coordinates seen by its Compartments). It is
    then that variable, read through the mapping (mapped_read), not one of its own.
    """
    if spec is None or spec.variable_type == "state_variable" or spec.scale is None or not ds.has(spec.name):
        return False
    held, own = ds.location(spec.name), location_of_scale(ds, spec.scale)
    if spec.location == "node":                         # read at the nodes, held at its scale
        return spec.mapping in DOWN_MAPPINGS and held == own != "node"
    # read at its scale, held at the nodes as a broadcast from that scale (e.g. a segment's coordinates seen by its
    # Compartments): its entities' values are the nodes' (their mean)
    meta = ds._variable_meta().get(spec.name, {}) if hasattr(ds, "_variable_meta") else {}
    return (spec.location == own != "node" and held == "node" and meta.get("scale") == spec.scale
            and meta.get("mapping") == "broadcast")


def mapped_read(ds, spec) -> np.ndarray:
    """The values at *spec*'s location of a variable held elsewhere (held_at_scale): at the nodes through *spec*'s
    mapping, or at its scale from the nodes' broadcast values (their mean)."""
    if spec.location != "node":
        out = np.asarray(ds.map_to(ds.get(spec.name), "node", spec.location, "mean"))
        out.flags.writeable = False
        return out
    if spec.mapping == "broadcast":
        return ds.parameter_view(spec.name, "node")
    held = ds.location(spec.name)
    out = ds.map_to(ds.get(spec.name), held, "node", spec.mapping, weights=ds.get(spec.weight))
    out.flags.writeable = False
    return out


def resolve_location(ds, location: str) -> str:
    """Location name as stored: scale names are accepted and resolved against the graph."""
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
    variable (no declaration metadata, or neither scale nor location given). scale=Compartment / Connection are the
    graph's nodes / edges (no MTG property behind them, except in anatomy mode); a scale coarser than the nodes
    without a location is stored at that scale.
    """
    meta = f.metadata
    if meta.get("variable_type") in (None, "forcing", "setting"):   # a time series, a model's value: not stored
        return None
    raw_scale, location = meta.get("scale"), meta.get("location")
    mapping, weight = meta.get("mapping"), meta.get("weight")
    kind = meta.get("kind")
    from openalea.metafspm.data_structure.data_api import _canonical_dtype
    dtype = _canonical_dtype(meta.get("dtype"))
    if isinstance(raw_scale, str) and raw_scale in SOLVER_LOCATIONS:
        raise DeclarationError(f"'{f.name}': scale='{raw_scale}' is a location: give location='{raw_scale}'")
    if raw_scale is None and location is None:
        # A numeric parameter or input without a place is stored per plant: at the "Plant" location of plant
        # DataStructures, at "scalar" elsewhere (an input a forcing writes: a plant's value, or the environment's).
        # Other fields are not DataStructure variables.
        numeric = (dtype is not object and not isinstance(f.default, bool)
                   and isinstance(f.default, (int, float, np.integer, np.floating)))
        if meta.get("variable_type") not in ("parameter", "input") or not numeric or mapping is not None:
            return None
        plant = _scales_of(ds).Plant
        on_plants = _is_graph(ds) and "Plant" in getattr(ds, "_coarse_scale_names", lambda: [])()
        raw_scale, location = (plant, "Plant") if on_plants else (None, "scalar")
    if dtype is object:
        default = None if _is_missing(f.default) else f.default
    else:
        default = f.default if f.default is not None and not _is_missing(f.default) else 0.
        try:
            default = dtype(default)
        except (TypeError, ValueError):
            raise DeclarationError(f"'{f.name}': a DataStructure variable needs a numeric default (or dtype='object'), "
                                   f"got {default!r}") from None
    common = dict(name=f.name, kind=kind, variable_type=meta.get("variable_type"),
                  default=default, on_grow=meta.get("on_grow") or "default", dtype=dtype,
                  shape=tuple(meta.get("shape") or ()))

    if not _is_graph(ds):
        # Grids: only their own locations; biological scales have no meaning there. "node" is a grid's cells (the
        # nodes of its graph), so that one component class runs on plant graphs and on grids.
        if location == "node" and raw_scale is None and "cell" in getattr(ds, "_var_stores", dict)():
            location = "cell"
        if location in ("cell", "scalar", "edge") and raw_scale is None and mapping is None:
            return VariableSpec(location=location, **common)
        element_scale = getattr(ds, "element_scale", None)
        if element_scale is not None and (raw_scale is not None or location == "node"):
            # The elements of several populations (UnionDataStructure): their nodes' scale, and "node"
            scale = _scale_value(ds, raw_scale) if raw_scale is not None else element_scale
            if scale == element_scale and location in (None, "node") and mapping is None:
                return VariableSpec(location="node", **common)
            raise DeclarationError(f"'{f.name}': a union of populations holds their nodes only (scale "
                                   f"{scale_names(ds).get(element_scale, element_scale)}), not scale "
                                   f"{scale_names(ds).get(scale, scale)} at {location}")
        return None

    scale = _scale_value(ds, raw_scale)
    scales = _scales_of(ds)
    if scale == scales.Compartment and node_scale(ds) == scales.Compartment:
        pass    # anatomy mode: the Compartments are the MTG vertices behind the nodes, their properties are read
    elif scale == scales.Connection and node_scale(ds) == scales.Compartment:
        # anatomy mode: the edges are the Connections themselves, their properties are read and written
        if location not in (None, "edge") or mapping is not None:
            raise DeclarationError(f"'{f.name}': in anatomy mode, scale Connection is the edges themselves")
        return VariableSpec(location="edge", scale=scale, **common)
    elif scale in (scales.Compartment, scales.Connection):
        # Solver entities themselves: no MTG property behind them
        if location is None:
            location = "node" if scale == scales.Compartment else "edge"
        scale = None

    if scale is None:
        location = resolve_location(ds, location)
        if mapping is not None:
            raise DeclarationError(f"'{f.name}' has no MTG scale, its mapping '{mapping}' maps nothing")
        if location == "cell":
            raise DeclarationError(f"'{f.name}': location 'cell' is a grid location, not a graph one")
        return VariableSpec(location=location, weight=weight, **common)

    scale_location = location_of_scale(ds, scale)
    location = resolve_location(ds, location) if location is not None else scale_location

    if location == "scalar":
        raise DeclarationError(f"'{f.name}': a scalar variable has no MTG scale (drop scale=)")
    if location == "edge":
        if scale_location == "edge":
            return VariableSpec(location="edge", scale=None, weight=weight, **common)
        if mapping is None:
            raise DeclarationError(f"'{f.name}' maps scale {scale_names(ds)[scale]} to edges: give "
                                   f"aggregation= one of {EDGE_MAPPINGS}")
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
        # its weight (a mean's capacity, a split's share) is kept: the links mapping it use it
        return VariableSpec(location=location, scale=scale, weight=weight, **common)

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
    if mapping in ("weighted_mean", "split") and weight is None:
        raise DeclarationError(f"'{f.name}': mapping '{mapping}' needs weight=")
    if mapping == "split" and common["variable_type"] == "state_variable":
        raise DeclarationError(f"'{f.name}' is a state variable split from scale {scale_names(ds)[scale]} to "
                               f"{location}: store it at its own scale and split it in a link")
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
