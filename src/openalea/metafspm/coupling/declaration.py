"""
Resolution of component field declarations into DataStructure variables.

A declaration carries three keys:
  scale     where the MTG property lives (an MTG scale int, or None for solver-only variables);
  location  where the DataStructure stores it ("node", "edge", "scalar", "cell", or a coarse scale name);
  aggregation   how values go between *scale* and *location* when they differ (with *weight* if needed).

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
UP_AGGREGATIONS = ("sum", "mean", "weighted_mean", "max", "min")
POOLED_UP_MAPPINGS = ("sum", "mean", "weighted_mean")    # across DataStructures: pooled over the populations
DOWN_AGGREGATIONS = ("broadcast", "split")
EDGE_AGGREGATIONS = ("child", "parent", "mean")

EXTENSIVE_KINDS = ("extensive", "non_inertial_extensive")
INTENSIVE_KINDS = ("intensive", "non_inertial_intensive")
MASSIC_KINDS = ("massic_concentration",)


@dataclass(frozen=True)
class VariableSpec:
    """Where a declared variable is stored, and how it maps to its MTG scale."""
    name: str
    location: str
    scale: Optional[int] = None          # MTG scale of the property; None: solver-only variable
    aggregation: Optional[str] = None    # scale -> location aggregation when they differ
    weight: Optional[str] = None         # weight variable of "weighted_mean"
    kind: Optional[str] = None           # extensive, intensive, ... (its declaration's kind=)
    role: Optional[str] = None           # "state_variable" | "input_variable" | "parameter" | ...
    default: float = 0.
    on_grow: str = "default"
    dtype: type = float                  # float, int or object
    shape: tuple = ()                    # per-entity shape of a vector-valued variable

    @property
    def mtg_backed(self) -> bool:
        return self.scale is not None and not self.shape

    def meta(self) -> dict:
        """Metadata recorded on the DataStructure next to the values."""
        return {"scale": self.scale, "aggregation": self.aggregation, "weight": self.weight, "kind": self.kind,
                "role": self.role, "dtype": self.dtype, "shape": self.shape}


class DeclarationError(ValueError):
    """A declaration that cannot be resolved on the DataStructure it is bound to."""


def legacy_edge_convention(aggregation: str) -> str:
    """Name used by the DataStructure's MTG edge readers and writers."""
    return {"child": "proximal", "parent": "distal"}.get(aggregation, aggregation)


def default_aggregation(kind: Optional[str], direction: str, name: str, weight: Optional[str] = None) -> str:
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
    raise DeclarationError(f"'{name}' changes scale without an aggregation, and its kind ({kind!r}) "
                           "does not imply one: give aggregation=...")


KIND_FAMILIES = {"extensive": "extensive", "non_inertial_extensive": "extensive",
                 "intensive": "intensive", "non_inertial_intensive": "intensive", "massic_concentration": "intensive",
                 "descriptor": "descriptor"}


def kinds_agree(a: Optional[str], b: Optional[str]) -> bool:
    """Whether two kinds can be linked (same family); an undeclared kind (None or "") agrees with any, an unknown one
    raises (declarations check theirs: Component.__init_subclass__)."""
    for kind in (a, b):
        if kind not in (None, "") and kind not in KIND_FAMILIES:
            raise DeclarationError(f"unknown kind {kind!r} (one of {sorted(KIND_FAMILIES)}, or none)")
    if a in (None, "") or b in (None, ""):
        return True
    return KIND_FAMILIES[a] == KIND_FAMILIES[b]


def declared_unit(component, name: str) -> Optional[str]:
    """The unit *component*'s class (or *component*, a class) declares for *name* (None when undeclared)."""
    f = getattr(component if isinstance(component, type) else type(component), "__dataclass_fields__", {}).get(name)
    return None if f is None else f.metadata.get("unit")


def declared_kind(component, name: str) -> Optional[str]:
    """The kind of *name* for *component*: its declaration's, else the one recorded on its DataStructure."""
    spec = getattr(component, "_variable_specs", {}).get(name)
    if spec is not None and spec.kind is not None:
        return spec.kind
    data_structure = getattr(component, "data_structure", None)
    if data_structure is not None and hasattr(data_structure, "_variable_meta") and data_structure.has(name):
        return data_structure._variable_meta().get(data_structure._resolve(name), {}).get("kind")
    return None


def check_link(link, receiver, provider):
    """A link's kinds agree (extensive with extensive, intensive or massic with intensive or massic) and its factors
    convert the provider's declared units into the receiver's (unless unit_check is off); a formula's units declared
    and readable (its conversion is its own); the link returned, its named equivalences (coupling.units.Equivalence)
    turned into their numbers from the units."""
    from openalea.metafspm.coupling.units import check_link_units, check_readable
    name = f"{link.receiver}.{link.variable} <- {link.provider}"
    received = declared_kind(receiver, link.variable)
    for source in ([] if link.formula is not None else link.sources):
        provided = source.kind if not isinstance(source, str) else declared_kind(provider, source)   # a geometric one
        if not kinds_agree(received, provided):
            raise ValueError(f"{link.receiver}.{link.variable} ({received}) <- {link.provider}.{source} ({provided}): "
                             "the kinds do not agree (extensive with extensive, intensive or massic with intensive or "
                             "massic)")
    received_unit = declared_unit(receiver, link.variable)
    if link.formula is not None:
        if link.unit_check:
            for source in link.sources:
                unit = source.unit if not isinstance(source, str) else declared_unit(provider, source)
                check_readable(received_unit, unit, f"{name}.{source}")
        return link
    for source, factor in link.sources.items():
        unit = source.unit if not isinstance(source, str) else declared_unit(provider, source)
        if isinstance(factor, (int, float)) and link.unit_check:     # an expression: its conversion is its own
            check_link_units(received_unit, unit, factor, f"{name}.{source}")
    return resolve_equivalences(link, receiver, provider)


def resolve_equivalences(link, receiver, provider):
    """*link* with its named equivalences (coupling.units.Equivalence) turned into their factors from the units its
    *receiver* and *provider* (components or their classes) declare; *link* itself without any."""
    from dataclasses import replace
    from openalea.metafspm.coupling.units import Equivalence
    if link.formula is not None or not any(isinstance(factor, Equivalence) for factor in link.sources.values()):
        return link
    name = f"{link.receiver}.{link.variable} <- {link.provider}"
    received_unit = declared_unit(receiver, link.variable)
    sources = {}
    for source, factor in link.sources.items():
        if isinstance(factor, Equivalence):             # its number from the units: checked by construction
            unit = source.unit if not isinstance(source, str) else declared_unit(provider, source)
            sources[source] = factor.factor(received_unit, unit, f"{name}.{source}")
        else:
            sources[source] = link.raw_factors.get(source, factor)
    return replace(link, sources=sources)


def _link_direction(data_structure, from_location: str, to_location: str) -> Optional[str]:
    """
    "up" from finer to coarser locations (nodes, then coarse scales, then the scalar plant scale), "down" the
    other way, None when edges are involved (node <-> edge aggregations have no default).
    """
    def rank(location, source=False):
        if location == "scalar":
            return 0
        if location == "edge":                       # edges map up like their (child) nodes; nothing maps down to them
            return (node_scale(data_structure) or 99) if source else None
        if location in ("node", "cell"):
            return node_scale(data_structure) or 99
        names = {name: value for value, name in scale_names(data_structure).items()}
        return names.get(location)

    a, b = rank(from_location, source=True), rank(to_location)
    if a is None or b is None or a == b:
        return None
    return "up" if b < a else "down"


# ── Scales of a DataStructure ──────────────────────────────────────────────────

def _scales_of(data_structure):
    mtg = getattr(data_structure, "_mtg", None)
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


def node_scale(data_structure) -> Optional[int]:
    """MTG scale of the graph's nodes (the vertices the Compartments stand for), or None."""
    scale = data_structure._node_scale() if hasattr(data_structure, "_node_scale") else None
    return scale if scale is not None else getattr(data_structure, "_from_scale", None)


def _is_graph(data_structure) -> bool:
    """Plant graphs (MTG-backed); grids have a graph view too, but their own locations (cell, edge, scalar)."""
    return hasattr(data_structure, "_mtg")


def location_of_scale(data_structure, scale: int) -> str:
    """DataStructure location holding the values of MTG scale *scale*."""
    scales = _scales_of(data_structure)
    if scale == scales.Compartment:
        return "node"
    if scale == scales.Connection:
        return "edge"
    nodes = node_scale(data_structure)
    if nodes is None or scale == nodes:
        return "node"
    if scale > nodes:
        raise DeclarationError(f"scale {scale_names(data_structure).get(scale, scale)} is finer than the graph's nodes "
                               f"({scale_names(data_structure).get(nodes, nodes)}): a finer scale needs an anatomy")
    return scale_names(data_structure)[scale]


def held_at_scale(spec, data_structure) -> bool:
    """
    Whether *spec*, an input or parameter, is a variable another component holds at another location of *data_structure* (one
    DataStructure for every scale): read at the nodes from a coarser scale (mapped down: broadcast or split) while
    held at that scale's own location (a SubOrgan-scale architecture's length read at the Compartments), or read at
    its scale while held at the nodes as a broadcast from it (a segment's coordinates seen by its Compartments). It is
    then that variable, read through the aggregation (_mapped_read), not one of its own.
    """
    if spec is None or spec.role == "state_variable" or spec.scale is None or not data_structure.has(spec.name):
        return False
    held, own = data_structure.location(spec.name), location_of_scale(data_structure, spec.scale)
    if spec.location == "node":                         # read at the nodes, held at its scale
        return spec.aggregation in DOWN_AGGREGATIONS and held == own != "node"
    # read at its scale, held at the nodes as a broadcast from that scale (e.g. a segment's coordinates seen by its
    # Compartments): its entities' values are the nodes' (their mean)
    meta = data_structure._variable_meta().get(spec.name, {}) if hasattr(data_structure, "_variable_meta") else {}
    return (spec.location == own != "node" and held == "node" and meta.get("scale") == spec.scale
            and meta.get("aggregation") == "broadcast")


def _holder_spec(spec, data_structure) -> "VariableSpec":
    """The spec, at its holder's location, of a variable *spec* reads held elsewhere (held_at_scale): what the MTG
    holds of it (an MPG-style step reads it at its scale's vertices), as the holder declares it."""
    from dataclasses import replace
    held = data_structure.location(spec.name)
    if held == "node":                                  # held at the nodes as a broadcast from its scale
        return replace(spec, location="node", aggregation="broadcast")
    return replace(spec, location=held, aggregation=None)


def _mapped_read(data_structure, spec) -> np.ndarray:
    """The values at *spec*'s location of a variable held elsewhere (held_at_scale): at the nodes through *spec*'s
    aggregation, or at its scale from the nodes' broadcast values (their mean)."""
    if spec.location != "node":
        out = np.asarray(data_structure.map_to(data_structure.get(spec.name), "node", spec.location, "mean"))
        out.flags.writeable = False
        return out
    if spec.aggregation == "broadcast":
        return data_structure.parameter_view(spec.name, "node")
    held = data_structure.location(spec.name)
    out = data_structure.map_to(data_structure.get(spec.name), held, "node", spec.aggregation, weights=data_structure.get(spec.weight))
    out.flags.writeable = False
    return out


def resolve_location(data_structure, location: str) -> str:
    """Location name as stored: scale names are accepted and resolved against the graph."""
    if location in SOLVER_LOCATIONS:
        return location
    names = {name: value for value, name in scale_names(data_structure).items()}
    if location in names:
        return location_of_scale(data_structure, names[location])
    raise DeclarationError(f"unknown location '{location}' (expected one of {SOLVER_LOCATIONS} or a scale name "
                           f"among {list(names)})")


def _scale_value(data_structure, scale):
    """MTG scale int from an int or a scale name; None for none."""
    if scale is None or isinstance(scale, int):
        return scale
    names = {name: value for value, name in scale_names(data_structure).items()}
    if scale in names:
        return names[scale]
    raise DeclarationError(f"unknown scale '{scale}'")


# ── The interpreter ────────────────────────────────────────────────────────────

def resolve_declaration(f, data_structure) -> Optional[VariableSpec]:
    """
    VariableSpec of dataclass field *f* on DataStructure *data_structure*, or None when the field is not a DataStructure
    variable (no declaration metadata, or neither scale nor location given). scale=Compartment / Connection are the
    graph's nodes / edges (no MTG property behind them, except in anatomy mode); a scale coarser than the nodes
    without a location is stored at that scale.
    """
    meta = f.metadata
    if meta.get("role") in (None, "forcing_variable", "setting"):   # a time series, a model's value: not stored
        return None
    raw_scale, location = meta.get("scale"), meta.get("location")
    aggregation, weight = meta.get("aggregation"), meta.get("weight")
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
        if meta.get("role") not in ("parameter", "input_variable") or not numeric or aggregation is not None:
            return None
        plant = _scales_of(data_structure).Plant
        on_plants = _is_graph(data_structure) and "Plant" in getattr(data_structure, "_coarse_scale_names", lambda: [])()
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
    common = dict(name=f.name, kind=kind, role=meta.get("role"),
                  default=default, on_grow=meta.get("on_grow") or "default", dtype=dtype,
                  shape=tuple(meta.get("shape") or ()))

    if not _is_graph(data_structure):
        # Grids: only their own locations; biological scales have no meaning there. "node" is a grid's cells (the
        # nodes of its graph), so that one component class runs on plant graphs and on grids.
        if location == "node" and raw_scale is None and "cell" in getattr(data_structure, "_var_stores", dict)():
            location = "cell"
        if location in ("cell", "scalar", "edge") and raw_scale is None and aggregation is None:
            return VariableSpec(location=location, weight=weight, **common)
        element_scale = getattr(data_structure, "element_scale", None)
        if element_scale is not None and (raw_scale is not None or location == "node"):
            # The elements of several populations (UnionDataStructure): their nodes' scale, and "node"
            scale = _scale_value(data_structure, raw_scale) if raw_scale is not None else element_scale
            if scale == element_scale and location in (None, "node") and aggregation is None:
                return VariableSpec(location="node", weight=weight, **common)
            raise DeclarationError(f"'{f.name}': a union of populations holds their nodes only (scale "
                                   f"{scale_names(data_structure).get(element_scale, element_scale)}), not scale "
                                   f"{scale_names(data_structure).get(scale, scale)} at {location}")
        return None

    scale = _scale_value(data_structure, raw_scale)
    scales = _scales_of(data_structure)
    if scale == scales.Compartment and node_scale(data_structure) == scales.Compartment:
        pass    # anatomy mode: the Compartments are the MTG vertices behind the nodes, their properties are read
    elif scale == scales.Connection and node_scale(data_structure) == scales.Compartment:
        # anatomy mode: the edges are the Connections themselves, their properties are read and written
        if location not in (None, "edge") or aggregation is not None:
            raise DeclarationError(f"'{f.name}': in anatomy mode, scale Connection is the edges themselves")
        return VariableSpec(location="edge", scale=scale, **common)
    elif scale in (scales.Compartment, scales.Connection):
        # Solver entities themselves: no MTG property behind them
        if location is None:
            location = "node" if scale == scales.Compartment else "edge"
        scale = None

    if scale is None:
        location = resolve_location(data_structure, location)
        if aggregation is not None:
            raise DeclarationError(f"'{f.name}' has no MTG scale, its aggregation '{aggregation}' aggregates nothing")
        if location == "cell":
            raise DeclarationError(f"'{f.name}': location 'cell' is a grid location, not a graph one")
        return VariableSpec(location=location, weight=weight, **common)

    scale_location = location_of_scale(data_structure, scale)
    location = resolve_location(data_structure, location) if location is not None else scale_location

    if location == "scalar":
        raise DeclarationError(f"'{f.name}': a scalar variable has no MTG scale (drop scale=)")
    if location == "edge":
        if scale_location == "edge":
            return VariableSpec(location="edge", scale=None, weight=weight, **common)
        if aggregation is None:
            raise DeclarationError(f"'{f.name}' maps scale {scale_names(data_structure)[scale]} to edges: give "
                                   f"aggregation= one of {EDGE_AGGREGATIONS}")
        if aggregation not in EDGE_AGGREGATIONS:
            raise DeclarationError(f"'{f.name}': edge aggregation '{aggregation}' is not one of {EDGE_AGGREGATIONS}")
        if aggregation == "mean" and common["role"] == "state_variable":
            raise DeclarationError(f"'{f.name}' is a state variable with edge aggregation 'mean': it could not be "
                                   "written back (no single owner), use 'child' or 'parent'")
        if scale_location != "node" and common["role"] == "state_variable":
            raise DeclarationError(f"'{f.name}' is an edge state variable at scale {scale_names(data_structure)[scale]}, coarser "
                                   "than the nodes: several edges would write the same vertex")
        return VariableSpec(location="edge", scale=scale, aggregation=aggregation, weight=weight, **common)

    if location == scale_location:
        if aggregation is not None:
            raise DeclarationError(f"'{f.name}' is stored at its own scale ({location}): aggregation '{aggregation}' "
                                   "maps nothing")
        # its weight (a mean's capacity, a split's share) is kept: the links aggregation it use it
        return VariableSpec(location=location, scale=scale, weight=weight, **common)

    # Between the nodes and a coarse scale
    if "node" not in (location, scale_location):
        raise DeclarationError(f"'{f.name}' maps scale {scale_names(data_structure)[scale]} to {location}: mappings go "
                               "between the graph's nodes and one coarser scale")
    location_scale = node_scale(data_structure) if location == "node" else _scale_value(data_structure, location)
    direction = "down" if location_scale > scale else "up"
    allowed = DOWN_AGGREGATIONS if direction == "down" else UP_AGGREGATIONS
    if aggregation is None:
        aggregation = default_aggregation(kind, direction, f.name, weight)
    if aggregation not in allowed:
        raise DeclarationError(f"'{f.name}' goes {direction} from scale {scale_names(data_structure)[scale]} to {location}: "
                               f"aggregation must be one of {allowed}, got '{aggregation}'")
    if aggregation in ("weighted_mean", "split") and weight is None:
        raise DeclarationError(f"'{f.name}': aggregation '{aggregation}' needs weight=")
    if aggregation == "split" and common["role"] == "state_variable":
        raise DeclarationError(f"'{f.name}' is a state variable split from scale {scale_names(data_structure)[scale]} to "
                               f"{location}: store it at its own scale and split it in a link")
    if aggregation == "sum" and common["role"] == "state_variable":
        raise DeclarationError(f"'{f.name}' is a state variable summed from scale {scale_names(data_structure)[scale]} to "
                               f"{location}: it could not be written back to the finer scale, store it at its own "
                               f"scale (scale={location})")
    return VariableSpec(location=location, scale=scale, aggregation=aggregation, weight=weight, **common)


def _is_missing(value) -> bool:
    from dataclasses import MISSING
    return value is MISSING


def _declared_specs(component_or_class, data_structure) -> dict:
    """{field name: VariableSpec} of every DataStructure variable declared by a component class."""
    cls = component_or_class if isinstance(component_or_class, type) else type(component_or_class)
    specs = {}
    for f in dc_fields(cls):
        try:
            spec = resolve_declaration(f, data_structure)
        except DeclarationError as error:
            raise DeclarationError(f"{cls.__name__}.{f.name}: {error}") from None
        if spec is not None:
            specs[f.name] = spec
    return specs


# ── Named entity sets (Component.entity_sets) ──────────────────────────────────

def _component_classes(cls=None):
    from openalea.metafspm.coupling.component import Component
    cls = Component if cls is None else cls
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _component_classes(subclass)


def entity_set(name: str) -> dict:
    """
    The rule of the entity set *name*, "Class.set": the set a component class publishes (Component.entity_sets,
    its own or inherited), e.g. "GranapAnatomy.soil_nodes" ({"kind": 4.}). The class is found among the imported
    component classes by its name (one class of that name, or several publishing the same rule).
    """
    if not isinstance(name, str) or name.count(".") != 1:
        raise ValueError(f"an entity set is named 'Class.set' (e.g. 'GranapAnatomy.soil_nodes'), got {name!r}")
    owner, member = name.split(".")
    classes = {cls for cls in _component_classes() if cls.__name__ == owner}
    if not classes:
        raise KeyError(f"entity set '{name}': no component class '{owner}' is imported")
    rules = []
    for cls in classes:
        sets = getattr(cls, "entity_sets", {}) or {}
        if member in sets and sets[member] not in rules:
            rules.append(sets[member])
    if not rules:
        published = sorted({key for cls in classes for key in (getattr(cls, "entity_sets", {}) or {})})
        raise KeyError(f"entity set '{name}': {owner} publishes {published or 'none'}")
    if len(rules) > 1:
        raise ValueError(f"entity set '{name}': several classes named {owner} publish different rules")
    return rules[0]


class EntitySets:
    """A mask rule (picklable): the entities of any of the entity sets *names* (their rules' union), on a
    DataStructure (define_mask(entities=)) or a section template (template_selection)."""

    def __init__(self, *names):
        self.names = tuple(names)
        for name in self.names:
            entity_set(name)

    def __call__(self, data_structure) -> np.ndarray:
        selected = None
        for name in self.names:
            values = data_structure._evaluate_mask(name, entity_set(name), data_structure._default_location)
            selected = values if selected is None else selected | values
        return selected

    def __eq__(self, other):
        return isinstance(other, EntitySets) and other.names == self.names

    def __hash__(self):
        return hash(self.names)

    def __repr__(self):
        return f"EntitySets{self.names}"


_COMPARISONS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal, "==": np.equal,
                "!=": np.not_equal}


def rule_holds(rule: dict, values: dict, codes=None) -> np.ndarray:
    """
    Where every {variable: condition} of *rule* holds, over arrays *values* ({variable: values}, one per entity): a
    comparison string (">0", "<=0.03", "!=2"), a value, or a list of values; *codes*(variable, condition) turns
    label names into codes (None: none).
    """
    import re
    pattern = re.compile(r"^\s*(<=|>=|==|!=|<|>)\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")
    result = None
    for variable, condition in rule.items():
        if variable not in values:
            raise KeyError(f"rule {rule}: no '{variable}'")
        array = np.asarray(values[variable])
        comparison = pattern.match(condition) if isinstance(condition, str) else None
        if comparison is not None:
            selected = _COMPARISONS[comparison.group(1)](array.astype(float), float(comparison.group(2)))
        else:
            if codes is not None:
                condition = codes(variable, condition)
            if isinstance(condition, (list, tuple, set, np.ndarray)):
                condition = list(condition)
                selected = np.isin(array, np.asarray(condition, dtype=array.dtype if array.dtype == object
                                                     else float))
            else:
                selected = array == condition
        result = selected if result is None else result & selected
    return result


def template_selection(rule, template) -> np.ndarray:
    """
    Over the vertices of a section template (data_structure.instances.SectionTemplate), where *rule* holds: an
    entity set's name ("GranapAnatomy.conduits"), a tuple of them (their union), a {variable: condition} dict on the
    template's properties (a vertex without a property's value does not hold), or a callable template -> booleans.
    """
    if callable(rule) and not isinstance(rule, (str, dict, tuple)):
        return np.asarray(rule(template), dtype=bool)
    if isinstance(rule, tuple):
        selected = np.zeros(template.size, dtype=bool)
        for part in rule:
            selected |= template_selection(part, template)
        return selected
    if isinstance(rule, str):
        rule = entity_set(rule)
    values, present = {}, np.ones(template.size, dtype=bool)
    for variable in rule:
        rows, stored = template.properties.get(variable, (np.zeros(0, dtype=np.int64), np.zeros(0)))
        full = np.full(template.size, np.nan)
        full[np.asarray(rows, dtype=np.int64)] = np.asarray(stored, dtype=float)
        has = np.zeros(template.size, dtype=bool)
        has[np.asarray(rows, dtype=np.int64)] = True
        values[variable], present = full, present & has
    return rule_holds(rule, values) & present
